"""调用未读摘要 Harness，将结果写入图状态。"""

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agent.context import AgentRuntimeContext
from agent.state import RoomAgentState
from harness.llm.summary import run_summary


async def summarize_node(
    state: RoomAgentState,
    config: RunnableConfig | None = None,
    *,
    runtime: Runtime[AgentRuntimeContext],
) -> dict[str, list[AnyMessage]]:
    """摘要输入由当前授权上下文决定，不复用历史聊天作为未读快照。"""
    responses = await run_summary(context=runtime.context, config=config)
    return {"messages": responses}
