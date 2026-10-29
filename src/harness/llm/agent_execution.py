"""收集内部 Agent 的消息，转发工具状态，校验最终回答。"""

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from agent.context import AgentRuntimeContext

from constants.exception import (
    DETAIL_AGENT_NO_FINAL_ANSWER,
    DETAIL_AGENT_NO_MESSAGES,
    DETAIL_INVALID_TOOL_CALLS,
)


async def run_agent(
    agent: Any,
    *,
    messages: Sequence[AnyMessage],
    context: AgentRuntimeContext,
    config: RunnableConfig | None = None,
) -> list[AnyMessage]:
    """执行 Agent；文本继续通过 LangGraph 原生 messages 流传播。"""
    writer = get_stream_writer()
    responses: list[AnyMessage] = []
    agent_input: InputAgentState = {"messages": list(messages)}

    async for event in agent.astream_events(
        agent_input,
        config=config,
        context=context,
        version="v2",
    ):
        kind = event["event"]
        data = event["data"]
        if kind == "on_tool_start":
            writer({"type": "tool_start", "name": event["name"]})
        elif kind == "on_tool_end":
            assert "output" in data
            writer({"type": "tool_end", "name": event["name"]})
            responses.append(data["output"])
        elif kind == "on_chat_model_end":
            assert "output" in data
            responses.append(data["output"])

    if not responses:
        raise RuntimeError(DETAIL_AGENT_NO_MESSAGES)

    final_response = responses[-1]
    if isinstance(final_response, AIMessage) and final_response.invalid_tool_calls:
        raise ValueError(DETAIL_INVALID_TOOL_CALLS)
    if (
        not isinstance(final_response, AIMessage)
        or final_response.tool_calls
        or not final_response.text.strip()
    ):
        raise RuntimeError(DETAIL_AGENT_NO_FINAL_ANSWER)

    return responses
