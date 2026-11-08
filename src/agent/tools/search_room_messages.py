"""根据 grpc 调用后端服务，获取关联的消息"""

import logging
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, Field

from agent.context import AgentRuntimeContext
from domain.time_hint import TimeHint
from harness.grpc.search_room_messages import search_room_messages

logger = logging.getLogger(__name__)


class RoomMessageSearchQuery(BaseModel):
    """一个独立的房间历史检索问题。"""

    query: str = Field(
        min_length=1,
        description="完整、独立且保留人物、主题和时间条件的自然语言问题。",
    )
    time_hint: TimeHint | None = Field(
        default=None,
        description="该子问题自己的时间偏好；无明确时间条件时传 null。",
    )


@tool
async def search_messages(
    queries: Annotated[
        list[RoomMessageSearchQuery],
        Field(min_length=1, max_length=2),
    ],
    runtime: ToolRuntime[AgentRuntimeContext],
) -> dict[str, object]:
    """检索当前已授权房间中与用户问题相关的历史文本消息。

    当用户询问房间历史讨论、成员发言、决定或待办事项时使用。
    日常闲聊不需要调用此工具。

    Args:
        queries: 一至两个独立的检索问题；每项包含完整自然语言 query 和自己的
            time_hint。query 应保留时间、人物和主题，不能只传孤立关键词。
        runtime: 框架自动注入的运行时上下文，包含已授权的用户和房间范围，
            不由模型填写。

    Returns:
        检索结果字典：status 表示执行状态；results 按输入问题逐项返回，
        每项包含原 query、检索到的有效 messages 和补充说明。
    """

    auth_scope = runtime.context.auth_scope
    results: list[dict[str, object]] = []
    for item in queries:
        result = await search_room_messages(
            query=item.query,
            auth_scope=auth_scope,
            time_hint=item.time_hint,
        )
        results.append({"query": item.query, **result})

    return {"status": "success", "results": results}
