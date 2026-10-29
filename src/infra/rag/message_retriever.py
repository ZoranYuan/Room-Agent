from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    ExactMatchFilter,
    MetadataFilters,
)

from infra.rag.vector.vector_store import get_vector_index


async def retrieve_message_chunks(
    *,
    query: str,
    room_id: str,
    topk: int = 5,
) -> list[NodeWithScore]:
    index = await get_vector_index()

    retriver = index.as_retriever(
        similarity_top_k=topk,
        filters=MetadataFilters(
            filters=[
                ExactMatchFilter(
                    key="room_id",
                    value=room_id,
                )
            ]
        ),
    )
    
    return await retriver.aretrieve(query)
