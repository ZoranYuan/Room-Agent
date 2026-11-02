from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    ExactMatchFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from infra.rag.vector.vector_store import get_vector_index


async def retrieve_message_chunks(
    *,
    query: str,
    room_id: str,
    topk: int = 5,
) -> list[NodeWithScore]:
    index = await get_vector_index()

    retriever = index.as_retriever(
        similarity_top_k=topk,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        filters=MetadataFilters(
            filters=[
                ExactMatchFilter(key="room_id", value=room_id),
            ]
        ),
    )

    candidates = await retriever.aretrieve(query)

    # 对召回的内容进行重排
    reranker = SentenceTransformerRerank(
        model="Qwen/Qwen3-Reranker-0.6B",
        top_n=3,
    )

    ranked = await reranker.apostprocess_nodes(candidates, query_str=query)

    return ranked
