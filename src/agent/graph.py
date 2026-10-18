"""定义 Room Agent 的任务路由图."""

from langgraph.graph import END, START, StateGraph

from agent.nodes.chat import chat_node
from agent.nodes.summarize import create_summarize_node
from agent.state import AgentTask, RoomAgentState


def route_task(state: RoomAgentState) -> str:
    """根据请求中的任务类型选择对应节点."""
    if state.task is AgentTask.SUMMARY:
        return "summarize"
    if state.task is AgentTask.CHAT:
        return "chat"
    raise ValueError(f"不支持的 Agent 任务类型: {state.task}")


def build_agent_graph():
    """构建支持摘要和聊天任务的 Agent graph."""
    graph = StateGraph(RoomAgentState)
    graph.add_node("summarize", create_summarize_node)
    graph.add_node("chat", chat_node)

    graph.add_conditional_edges(
        START,
        route_task,
        {
            "summarize": "summarize",
            "chat": "chat",
        },
    )
    graph.add_edge("summarize", END)
    graph.add_edge("chat", END)

    return graph.compile()
