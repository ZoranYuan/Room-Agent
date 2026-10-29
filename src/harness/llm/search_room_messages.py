"""组织子块召回、父块查询和 IM 原文获取。"""

from dataclasses import asdict

from domain.auth_scope import AuthScope
from harness.grpc.search_room_messages import search_room_messages as fetch_room_messages
from harness.utils.merge_seq_ranges import merge_seq_ranges
from infra.mysql.database import get_session_factory
from infra.mysql.message_chunk_repository import MessageChunkRepository
from infra.rag.message_retriever import retrieve_message_chunks


async def search_room_messages(
    *, query: str, auth_scope: AuthScope, topk: int = 5
) -> dict[str, object]:
    """召回父块范围内的可见原文，查询异常直接向外抛出。"""
    hits = await retrieve_message_chunks(query=query, room_id=auth_scope.room_id, topk=topk)
    parent_ids = list(dict.fromkeys(hit.node.metadata["parent_id"] for hit in hits))
    if not parent_ids:
        return {
            "status": "success",
            "messages": [],
            "extra_message": "当前房间的索引中未检索到相关消息。",
        }

    session_factory = get_session_factory()
    async with session_factory() as session:
        parents = await MessageChunkRepository(session).get_parent_chunks_by_ids(
            room_id=auth_scope.room_id, parent_ids=parent_ids,
        )

    if not parents:
        return {
            "status": "success",
            "messages": [],
            "extra_message": "已召回相关索引，但未找到对应父块。",
        }

    # 数据库 Session 已释放，再通过 IM 获取合并范围内的原文。
    ranges = merge_seq_ranges([(parent.seq_start, parent.seq_end) for parent in parents])
    messages = await fetch_room_messages(auth_scope=auth_scope, ranges=ranges)
    return {
        "status": "success",
        "messages": [asdict(message) for message in messages],
        "extra_message": "" if messages else "IM 未返回当前用户可见的有效消息。",
    }
