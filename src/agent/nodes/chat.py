"""读取聊天状态，调用 Harness，并记录本轮消息。"""

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agent.context import AgentRuntimeContext
from agent.utils.memory import build_chat_messages
from agent.state import RoomAgentState
from harness.llm.chat import run_chat

from constants.exception import (
    DETAIL_CHAT_QUESTION_MISSING,
)


async def chat_node(
    state: RoomAgentState,
    config: RunnableConfig | None = None,
    *,
    runtime: Runtime[AgentRuntimeContext],
) -> dict[str, list[AnyMessage]]:
    """组装本次模型输入，仅将本轮新增消息写入 State。"""
    question = (state.question or "").strip()
    if not question:
        raise ValueError(DETAIL_CHAT_QUESTION_MISSING)

    user_message = HumanMessage(content=question)
    responses = await run_chat(
        messages=build_chat_messages(state, user_message),
        context=runtime.context,
        config=config,
    )
    return {"messages": [user_message, *responses]}
