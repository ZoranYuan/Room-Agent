"""Checkpointer 的创建与序列化配置；当前使用进程内存保存运行状态."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

_DEFAULT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("agent.state", "AgentTask"),
        ("domain.chat_message", "ChatMessage"),
        ("workflows.summary.state", "Reflection"),
        ("workflows.summary.state", "ReflectionReasonCode"),
        ("workflows.summary.state", "ReflectionAction"),
        ("workflows.summary.state", "InterruptAction"),
        ("workflows.summary.state", "SummaryStatus"),
        ("domain.message", "RoomMessage"),
        ("domain.summary", "SummaryOutput"),
    ],
)


def create_memory_checkpointer() -> InMemorySaver:
    """创建独立的内存 Checkpointer，复用已有状态类型的序列化配置."""
    return InMemorySaver(serde=_DEFAULT_SERDE)
