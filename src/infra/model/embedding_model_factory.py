"""延迟加载 ModelScope 模型，提供 LlamaIndex Embedding 实例."""

import asyncio

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from modelscope import snapshot_download

from config.config import settings

_embedding_model: BaseEmbedding | None = None


async def get_embedding_model() -> BaseEmbedding:
    """首次初始化，后续在当前服务进程内复用。"""
    global _embedding_model

    if _embedding_model is None:
        model_path = await asyncio.to_thread(
            snapshot_download, settings.embedding_model_id
        )
        _embedding_model = await asyncio.to_thread(
            HuggingFaceEmbedding,
            model_name=model_path,
            device=settings.rag_device,
            normalize=True,
            truncate_dim=settings.embedding_dimension,
            embed_batch_size=8,
            query_instruction=(
                "Instruct: Given a question about a chat room's history, retrieve "
                "relevant messages that contain evidence to answer it.\n"
                "Query: "
            ),
        )
    return _embedding_model
