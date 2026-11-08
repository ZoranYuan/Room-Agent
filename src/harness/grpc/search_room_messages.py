"""组织子块召回、父块查询和 IM 原文获取。"""

from contextlib import aclosing
from dataclasses import asdict
from time import perf_counter

from config.config import settings
from domain.auth_scope import AuthScope
from domain.chat_message import ChatMessage
from domain.time_hint import TimeHint
from harness.utils.merge_seq_ranges import merge_seq_ranges
from infra.mysql.database import get_session_factory
from infra.mysql.message_chunk_repository import MessageChunkRepository
from infra.monitoring import RAG_RETRIEVAL_DURATION
from infra.rag.message_retriever import retrieve_message_chunks
from transport.grpc.client import get_grpc_client
from transport.grpc.proto.room_agent_pb2 import MessageStatus, MessageType


async def get_room_messages(
    *,
    auth_scope: AuthScope,
    ranges: list[tuple[int, int]],
    limit: int = 100,
    timeout_seconds: float = 60.0,
) -> list[ChatMessage]:
    """按 seq 范围收齐 IM 返回的正常文本消息。"""
    if not ranges:
        return []

    client = get_grpc_client()
    messages: list[ChatMessage] = []
    async with aclosing(
        client.search_room_messages(
            token=auth_scope.token.get_secret_value(),
            room_id=auth_scope.room_id,
            ranges=ranges,
            limit=limit,
            timeout_seconds=timeout_seconds,
        )
    ) as batches:
        async for batch in batches:
            messages.extend(
                ChatMessage(
                    message_id=message.message_id,
                    conversation_id=message.conversation_id,
                    sender_id=message.sender_id,
                    sender_name=message.sender_name,
                    seq=message.seq,
                    content=message.content,
                    status=message.status,
                    send_time=message.send_time,
                )
                for message in batch.messages
                if message.type == MessageType.MESSAGE_TYPE_TEXT
                and message.status == MessageStatus.MESSAGE_STATUS_NORMAL
                and message.content.strip()
            )

    return sorted(messages, key=lambda message: message.seq)


async def search_room_messages(
    *,
    query: str,
    auth_scope: AuthScope,
    time_hint: TimeHint | None = None,
    topk: int = settings.rag_candidate_top_k,
) -> dict[str, object]:
    """召回父块范围内的可见原文，查询异常直接向外抛出。"""
    started_at = perf_counter()
    try:
        hits = await retrieve_message_chunks(
            query=query,
            room_id=auth_scope.room_id,
            time_hint=time_hint,
            topk=topk,
        )
    except Exception:
        RAG_RETRIEVAL_DURATION.labels(outcome="error").observe(
            perf_counter() - started_at
        )
        raise
    else:
        RAG_RETRIEVAL_DURATION.labels(outcome="success").observe(
            perf_counter() - started_at
        )

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
            room_id=auth_scope.room_id,
            parent_ids=parent_ids,
        )

    if not parents:
        return {
            "status": "success",
            "messages": [],
            "extra_message": "已召回相关索引，但未找到对应父块。",
        }

    # 数据库 Session 已释放，再通过 IM 获取合并范围内的原文。
    ranges = merge_seq_ranges(
        [(parent.seq_start, parent.seq_end) for parent in parents]
    )
    messages = await get_room_messages(auth_scope=auth_scope, ranges=ranges)
    return {
        "status": "success",
        "messages": [asdict(message) for message in messages],
        "extra_message": "" if messages else "IM 未返回当前用户可见的有效消息。",
    }
