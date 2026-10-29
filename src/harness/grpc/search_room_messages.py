"""按 seq 范围收齐 IM 流式返回的房间历史消息。"""

from contextlib import aclosing

from domain.auth_scope import AuthScope
from domain.chat_message import ChatMessage
from transport.grpc.client import get_grpc_client
from transport.grpc.proto.room_agent_pb2 import MessageStatus, MessageType


async def search_room_messages(
    *,
    auth_scope: AuthScope,
    ranges: list[tuple[int, int]],
    limit: int = 100,
    timeout_seconds: float = 60.0,
) -> list[ChatMessage]:
    """完整接收后返回有效文本；流中途失败时抛出，不返回部分结果。"""
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
