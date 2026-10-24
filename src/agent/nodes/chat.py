"""定义房间聊天问答节点."""

from langchain.agents import create_agent
from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.context import AgentRuntimeContext
from agent.memory import build_chat_messages
from agent.model import chat_model
from agent.prompts.chat import CHAT_PROMPT
from agent.state import RoomAgentState
from agent.tools.search_messages import search_messages

chat_agent = create_agent(
    model=chat_model,
    tools=[search_messages],
    system_prompt=CHAT_PROMPT.format_messages()[0].text,
    context_schema=AgentRuntimeContext,
    # 会话历史统一由外层 graph 保存。
    checkpointer=False,
)


async def chat_node(
    state: RoomAgentState,
    config: RunnableConfig | None = None,
    *,
    runtime: Runtime[AgentRuntimeContext],
) -> dict[str, list[AnyMessage]]:
    """通过原生消息流输出文本，通过 custom 通知工具状态，并保存本轮消息."""
    question = (state.question or "").strip()
    if not question:
        raise ValueError("聊天任务缺少用户问题")

    writer = get_stream_writer()
    user_message = HumanMessage(content=question)

    responses: list[AnyMessage] = []

    agent_input: InputAgentState = {
        "messages": [*build_chat_messages(state, user_message)]
    }

    async for event in chat_agent.astream_events(
        agent_input,
        config=config,
        context=runtime.context,
        version="v2",
    ):
        kind = event["event"]

        data = event["data"]

        if kind == "on_tool_start":
            writer({"type": "tool_start", "name": event["name"]})

        elif kind == "on_tool_end":
            assert "output" in data
            writer({"type": "tool_end", "name": event["name"]})

            # 收集工具节点的输出
            responses.append(data["output"])

        elif kind == "on_chat_model_end":
            assert "output" in data
            # 收集模型的输出
            responses.append(data["output"])

    if not responses:
        raise RuntimeError("Agent 未返回消息")

    final_response = responses[-1]
    if isinstance(final_response, AIMessage) and final_response.invalid_tool_calls:
        raise ValueError("模型返回的工具调用参数无法解析")

    if (
        not isinstance(final_response, AIMessage)
        or final_response.tool_calls
        or not final_response.text.strip()
    ):
        raise RuntimeError("Agent 未返回有效的最终回答")

    return {
        "messages": [user_message, *responses],
    }

    """
    # AI 的输出可能分散在多个 chunk 中，我们需要收集完整之后判断是否是工具调用
    # collected: AIMessageChunk | None = None
    # writer = get_stream_writer()

    # user_message = HumanMessage(content=question)
    # tool_announced = False
    # messages = [*state.messages, user_message]
    # async for chunk in chat_chain.astream(
    #     {
    #         "messages": messages,
    #     },
    #     config=config,
    # ):
    #     if not isinstance(chunk, AIMessageChunk):
    #         raise TypeError("模型返回的流式数据不是 AIMessageChunk")

    #     collected = chunk if collected is None else collected + chunk

    #     if chunk.tool_calls and not tool_announced:
    #         tool_announced = True
    #         writer(
    #             {
    #                 "type": "tool_status",
    #                 "status": "preparing",
    #             }
    #         )

    #     # LLM 输出的是文本
    #     if chunk.text:
    #         writer(
    #             {
    #                 "type": "chat_token",
    #                 "content": chunk.text,
    #             }
    #         )

    # if collected is None:
    #     raise RuntimeError("模型未返回任何消息")

    # response = message_chunk_to_message(collected)

    # if not isinstance(response, AIMessage):
    #     raise TypeError("模型输出不是 AIMessage")

    # if response.invalid_tool_calls:
    #     raise ValueError("模型返回的工具调用参数无法解析")

    # if not response.text.strip() and not response.tool_calls:
    #     raise RuntimeError("模型既没有返回文本，也没有请求调用工具")
    """
