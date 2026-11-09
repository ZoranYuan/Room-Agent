"""收集内部 Agent 的消息，转发工具状态，校验最终回答。"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from agent.context import AgentRuntimeContext
from config.config import settings
from constants.exception import (
    DETAIL_AGENT_NO_FINAL_ANSWER,
    DETAIL_AGENT_NO_MESSAGES,
    DETAIL_INVALID_TOOL_CALLS,
)
from harness.middleware.tool_result_cache import tool_result_cache_scope
from infra.monitoring import LLM_TOKENS


@dataclass
class AgentExecutionResult:
    messages: list[AnyMessage]
    structured_response: object | None


async def run_agent(
    agent: Any,
    *,
    messages: Sequence[AnyMessage],
    context: AgentRuntimeContext,
    config: RunnableConfig | None = None,
) -> AgentExecutionResult:
    """执行 Agent；文本继续通过 LangGraph 原生 messages 流传播。"""
    writer = get_stream_writer()
    # 收集一次模型调用的输出，保存到 graph 的检查点中
    responses: list[AnyMessage] = []
    structured_response: object | None = None
    agent_run_id: str | None = None
    agent_input: InputAgentState = {"messages": list(messages)}

    async with tool_result_cache_scope():
        async for event in agent.astream_events(
            agent_input,
            config=config,
            context=context,
            version="v2",
        ):
            kind = event["event"]
            data = event["data"]

            match kind:
                case "on_tool_start":
                    writer({"type": "tool_start", "name": event["name"]})

                case "on_tool_end":
                    assert "output" in data
                    writer({"type": "tool_end", "name": event["name"]})
                    responses.append(data["output"])

                case "on_chat_model_end":
                    assert "output" in data
                    output = data["output"]
                    responses.append(output)

                    if isinstance(output, AIMessage):
                        usage = output.usage_metadata or {}
                        if not usage:
                            provider_usage = output.response_metadata.get(
                                "token_usage", {}
                            )
                            usage = {
                                "input_tokens": provider_usage.get("prompt_tokens", 0),
                                "output_tokens": provider_usage.get(
                                    "completion_tokens", 0
                                ),
                            }

                        model_name = output.response_metadata.get(
                            "model_name", settings.agent_model
                        )
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)

                        if input_tokens:
                            LLM_TOKENS.labels(model=model_name, direction="input").inc(
                                input_tokens
                            )
                        if output_tokens:
                            LLM_TOKENS.labels(model=model_name, direction="output").inc(
                                output_tokens
                            )

                case "on_chain_start":
                    if agent_run_id is None:
                        # 通过 run_id 来获取根 Agent 的输出
                        agent_run_id = event["run_id"]

                case "on_chain_end" if event.get("run_id") == agent_run_id:
                    output = data.get("output")
                    if (
                        isinstance(output, dict)
                        and output.get("structured_response") is not None
                    ):
                        structured_response = output["structured_response"]

                case _:
                    pass
    if not responses:
        raise RuntimeError(DETAIL_AGENT_NO_MESSAGES)

    final_response = responses[-1]
    if isinstance(final_response, AIMessage) and final_response.invalid_tool_calls:
        raise ValueError(DETAIL_INVALID_TOOL_CALLS)

    if structured_response is None and (
        not isinstance(final_response, AIMessage)
        or final_response.tool_calls
        or not final_response.text.strip()
    ):
        raise RuntimeError(DETAIL_AGENT_NO_FINAL_ANSWER)

    return AgentExecutionResult(
        messages=responses,
        structured_response=structured_response,
    )
