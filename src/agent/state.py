"""Room Agent 的运行状态定义."""

from enum import Enum
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agent.schemas.chat import InterruptRequest


class AgentTask(str, Enum):
    """Room Agent 支持的任务类型."""

    SUMMARY = "summary"
    CHAT = "chat"


class RoomAgentState(BaseModel):
    """Room Agent 图的任务输入与运行状态."""

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    # None 表示本次请求未提供摘要，避免覆盖检查点中已有的记忆。
    conversation_summary: str | None = None
    task: AgentTask
    question: str | None = None

    # 仅用于需要安全确认的操作；意图不明时由模型直接向用户追问。
    interrupt_request: InterruptRequest | None = None
    # None 表示尚未确认，True 表示同意，False 表示拒绝。
    interrupt_approved: bool | None = None
