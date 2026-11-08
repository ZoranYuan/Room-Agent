"""读取聊天状态，调用 Harness，并记录本轮消息。"""

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agent.context import AgentRuntimeContext
from agent.state import RoomAgentState


async def interrupt_node(
    state: RoomAgentState,
    config: RunnableConfig | None = None,
    *,
    runtime: Runtime[AgentRuntimeContext],
) -> dict[str, list[AnyMessage]]:

    
        
    return {}
