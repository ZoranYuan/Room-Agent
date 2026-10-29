"""按房间收齐增量消息，并转换为领域消息对象."""

from contextlib import aclosing

from domain.auth_scope import AuthScope
from domain.chat_message import ChatMessage
from infra.rag.message_index import index_room_messages
from transport.grpc.client import get_grpc_client


async def sync_room_messages(
    *,
    auth_scope: AuthScope,
    after_seq: int,
    limit: int = 100,
    timeout_seconds: float = 60.0,
    index_messages: bool = True,
) -> list[ChatMessage]:
    """收齐流式响应后返回完整增量消息列表，保留原消息状态.

    收齐后构建 RAG 索引；索引完成进度由调用方在成功返回后更新。
    流中途失败时向外抛出异常，不返回不完整的消息列表。
    index_messages=False 时只拉取消息，由调用方安排后台索引。
    """
    messages: list[ChatMessage] = []
    client = get_grpc_client()
    async with aclosing(
        client.sync_room_messages(
            token=auth_scope.token.get_secret_value(),
            room_id=auth_scope.room_id,
            after_seq=after_seq,
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
            )

    if index_messages:
        await index_room_messages(room_id=auth_scope.room_id, messages=messages)
    return messages
