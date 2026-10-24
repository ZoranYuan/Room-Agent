"""Room 聊天消息领域对象."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Room Agent 使用的文本聊天消息."""

    message_id: str
    conversation_id: str
    sender_id: str
    seq: int
    content: str
    status: int
    send_time: int
