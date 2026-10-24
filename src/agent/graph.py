"""定义 Room Agent 的任务路由图."""

from langgraph.graph import END, START, StateGraph

from agent.context import AgentRuntimeContext
from agent.nodes.chat import chat_node
from agent.nodes.memory import prepare_memory_node
from agent.nodes.summarize import summarize_node
from agent.state import AgentTask, RoomAgentState
from infra.checkpoint.factory import create_memory_checkpointer


def route_task(state: RoomAgentState) -> str:
    """根据请求中的任务类型选择对应节点."""
    if state.task is AgentTask.SUMMARY:
        return "summarize"
    if state.task is AgentTask.CHAT:
        return "chat"
    raise ValueError(f"不支持的 Agent 任务类型: {state.task}")


def build_dev_agent_graph():
    graph = StateGraph(RoomAgentState, context_schema=AgentRuntimeContext)
    graph.add_node("summarize", summarize_node)
    graph.add_node("chat", chat_node)
    graph.add_node("prepare_memory", prepare_memory_node)

    graph.add_conditional_edges(
        START,
        route_task,
        {
            "summarize": "summarize",
            "chat": "prepare_memory",
        },
    )
    graph.add_edge("summarize", END)
    graph.add_edge("prepare_memory", "chat")
    graph.add_edge("chat", END)

    return graph.compile()


def build_agent_graph():
    """构建支持摘要和聊天任务的 Agent graph."""
    graph = StateGraph(RoomAgentState, context_schema=AgentRuntimeContext)
    graph.add_node("summarize", summarize_node)
    graph.add_node("chat", chat_node)
    graph.add_node("prepare_memory", prepare_memory_node)

    graph.add_conditional_edges(
        START,
        route_task,
        {
            "summarize": "summarize",
            "chat": "prepare_memory",
        },
    )
    graph.add_edge("summarize", END)
    graph.add_edge("prepare_memory", "chat")
    graph.add_edge("chat", END)

    checkpointer = create_memory_checkpointer()

    return graph.compile(checkpointer=checkpointer)
