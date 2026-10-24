"""执行 Agent graph 并产出与传输层无关的事件."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from agent.context import AgentRuntimeContext
from agent.runtime import agent_graph
from agent.state import RoomAgentState
from config.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentEvent:
    """一次 Agent 运行产生的事件."""

    name: str
    data: dict[str, object]


async def stream_agent_events(
    *,
    state: RoomAgentState,
    thread_id: str,
    context: AgentRuntimeContext,
) -> AsyncIterator[AgentEvent]:
    """将模型文本和工具状态分别从 messages、custom 转换为 SSE 业务事件."""
    try:
        async for _namespace, mode, payload in agent_graph.astream(
            state,
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": settings.agent_recursion_limit,
            },
            context=context,
            stream_mode=["messages", "custom"],
            subgraphs=True,
        ):
            if mode == "messages":
                message, _metadata = payload
                if not isinstance(message, AIMessage) or not message.text:
                    continue
                yield AgentEvent(name="token", data={"content": message.text})

            elif mode == "custom":
                if not isinstance(payload, dict):
                    raise TypeError("自定义事件数据必须是字典")
                event_type = payload.get("type")
                if not isinstance(event_type, str) or event_type not in (
                    "tool_start",
                    "tool_end",
                    "compress",
                ):
                    raise ValueError(f"不支持的自定义事件类型：{event_type!r}")
                yield AgentEvent(name=event_type, data=payload)

        yield AgentEvent(
            # 向前端发送一个 completed 事件，表示 LLM 的输出已经结束
            name="completed",
            data={"task": state.task.value},
        )

    except GraphRecursionError:
        logger.warning("Agent 达到最大执行步数")
        yield AgentEvent(
            name="error",
            data={"message": "本次任务执行步骤过多，请缩小问题范围后重试。"},
        )

    except Exception:
        logger.exception(
            "Agent 运行失败，请联系管理员",
            extra={
                "room_id": context.authorized_scope.room_id,
                "task": state.task.value,
            },
        )
        yield AgentEvent(
            name="error",
            data={"message": "Agent 任务执行失败"},
        )
