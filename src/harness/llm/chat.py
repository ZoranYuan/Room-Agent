"""执行房间问答 Agent。"""

from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig

from agent.context import AgentRuntimeContext
from agent.model import chat_model
from agent.prompts.chat import CHAT_PROMPT
from agent.tools.search_room_messages import search_messages
from agent.tools.sync_room_messages import sync_room_messages
from harness.llm.agent_execution import run_agent
from harness.middleware.tool_result_cache import ToolResultCacheMiddleware


class RoomToolCallLimitMiddleware(ToolCallLimitMiddleware[Any, AgentRuntimeContext]):
    """兼容 Room Agent 运行时上下文的工具调用限额中间件。"""


middleware = [
    RoomToolCallLimitMiddleware(run_limit=3, exit_behavior="end"),
    RoomToolCallLimitMiddleware(
        tool_name="search_messages",
        run_limit=2,
        exit_behavior="continue",
    ),
    RoomToolCallLimitMiddleware(
        tool_name="sync_room_messages",
        run_limit=1,
        exit_behavior="continue",
    ),
    ToolResultCacheMiddleware(),
]

chat_agent = create_agent(
    model=chat_model,
    tools=[search_messages, sync_room_messages],
    system_prompt=CHAT_PROMPT.format_messages()[0].text,
    context_schema=AgentRuntimeContext,
    middleware=middleware,
    checkpointer=False,
)


async def run_chat(
    *,
    messages: Sequence[AnyMessage],
    context: AgentRuntimeContext,
    config: RunnableConfig | None = None,
) -> list[AnyMessage]:
    """传入已组装的历史，返回本次执行新增的模型和工具消息。"""
    return await run_agent(
        chat_agent,
        messages=messages,
        context=context,
        config=config,
    )
