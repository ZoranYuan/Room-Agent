"""读取聊天状态，调用 Harness，并记录本轮消息。"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agent.context import AgentRuntimeContext
from agent.state import RoomAgentState
from agent.utils.memory import build_chat_messages
from constants.exception import (
    DETAIL_CHAT_QUESTION_MISSING,
)
from harness.llm.chat_agent import run_chat


async def chat_node(
    state: RoomAgentState,
    config: RunnableConfig | None = None,
    *,
    runtime: Runtime[AgentRuntimeContext],
) -> dict[str, Any]:
    """组装本次模型输入，仅将本轮新增消息写入 State。"""
    question = (state.question or "").strip()
    if not question:
        raise ValueError(DETAIL_CHAT_QUESTION_MISSING)

    user_message = HumanMessage(content=question)
    messages, chat_agent_output = await run_chat(
        messages=build_chat_messages(state, user_message),
        context=runtime.context,
        config=config,
    )

    answer = chat_agent_output.answer.strip()
    if answer:
        messages.append(AIMessage(content=answer))


    return {
        "messages": [user_message, *messages],
        "interrupt_request": chat_agent_output.interrupt_request,
    }
