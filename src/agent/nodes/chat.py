"""定义房间聊天问答节点."""

from agent.state import RoomAgentState


async def chat_node(state: RoomAgentState) -> dict[str, str]:
    """返回聊天节点的占位结果，待接入问答逻辑."""
    return {"last_result": "聊天问答节点尚未接入。"}
