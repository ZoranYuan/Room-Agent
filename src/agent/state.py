"""Room Agent 的运行状态定义."""

from enum import Enum

from pydantic import BaseModel, Field


class AgentTask(str, Enum):
    """Room Agent 支持的任务类型."""

    SUMMARY = "summary"
    CHAT = "chat"


class UnreadMessage(BaseModel):
    """摘要任务使用的单条未读文本消息."""

    message_id: str
    conversation_id: str
    sender_id: str
    seq: int
    content: str
    status: int
    send_time: int


class RoomAgentState(BaseModel):
    """Room Agent 单次运行需要的状态."""

    task: AgentTask
    room_id: str
    user_id: str
    thread_id: str
    question: str | None = None
    unread_message_count: int = 0
    unread_messages: list[UnreadMessage] = Field(default_factory=list)
    last_result: str | None = None
