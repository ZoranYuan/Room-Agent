"""根据 grpc 调用后端服务，获取关联的消息"""

import logging

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from agent.context import AgentRuntimeContext

logger = logging.getLogger(__name__)


@tool
def search_messages(
    query: str, runtime: ToolRuntime[AgentRuntimeContext]
) -> dict[str, object]:
    """检索当前已授权房间中与用户问题相关的历史文本消息。

    当用户询问房间历史讨论、成员发言、决定或待办事项时使用。
    日常闲聊不需要调用此工具。

    Args:
        query: 完整的自然语言检索问题，应保留时间、人物和主题，
            例如“昨天下午大家讨论的部署方案是什么？”，不能只传孤立关键词。
        runtime: 框架自动注入的运行时上下文，包含已授权的用户和房间范围，
            不由模型填写。

    Returns:
        检索结果字典：
        status 表示执行状态；
        messages 包含检索到的有效消息；
        extra_message 提供补充说明。
        当前占位实现返回 not_implemented，不代表未找到相关消息。
    """

    auth_scope = runtime.context.authorized_scope

    logger.info("接收到的请求为 %s", query)
    logger.info("当前用户的权限 scope 为 %s", auth_scope.model_dump_json())

    return {
        "status": "not_implemented",
        "messages": [],
        "message": "消息检索服务尚未接入。",
    }
