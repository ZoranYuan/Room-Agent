import asyncio

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore

from config.config import settings
from infra.model.embedding_model_factory import get_embedding_model

_vector_store: MilvusVectorStore | None = None
_index: VectorStoreIndex | None = None
_init_lock = asyncio.Lock()


async def get_vector_index() -> VectorStoreIndex:
    """首次初始化，后续在当前服务进程内复用。"""
    global _vector_store, _index

    async with _init_lock:
        if _index is None:
            embedding_model = await asyncio.to_thread(get_embedding_model)
            uri = settings.milvus_uri
            if "://" not in uri:
                uri = f"http://{uri}"

            _vector_store = MilvusVectorStore(
                uri=uri,
                collection_name=settings.milvus_collection,
                dim=settings.embedding_dimension,
                embedding_field="embedding",
                similarity_metric="COSINE",
                consistency_level="Strong",
                overwrite=False,
                upsert_mode=True,   # 同一个节点 ID 重复写入时更新
                use_async_client=True,
            )

            _index = VectorStoreIndex.from_vector_store(
                vector_store=_vector_store,
                embed_model=embedding_model,
            )

    return _index


async def close_vector_store() -> None:
    global _vector_store, _index

    async with _init_lock:
        if _vector_store is None:
            return

        try:
            if _vector_store.aclient is not None:
                await _vector_store.aclient.close()
        finally:
            await asyncio.to_thread(_vector_store.client.close)
