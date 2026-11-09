"""执行房间问答 Agent。"""

from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from agent.context import AgentRuntimeContext
from agent.model import chat_model
from agent.prompts.chat import CHAT_PROMPT
from agent.schemas.chat import ChatAgentOutput
from agent.tools.search_room_messages import search_messages
from agent.tools.sync_room_messages import sync_room_messages
from config.config import settings
from harness.llm.agent_execution import run_agent
from harness.middleware.model_retry import AgentRetryMiddleware
from harness.middleware.tool_result_cache import ToolResultCacheMiddleware

middleware = [
    ToolCallLimitMiddleware(run_limit=3, exit_behavior="continue"),
    ToolCallLimitMiddleware(
        tool_name="search_messages",
        run_limit=1,
        exit_behavior="continue",
    ),
    ToolCallLimitMiddleware(
        tool_name="sync_room_messages",
        run_limit=1,
        exit_behavior="continue",
    ),
    AgentRetryMiddleware(),
    ToolResultCacheMiddleware(),
]

chat_agent = create_agent(
    model=chat_model,
    response_format=ChatAgentOutput,
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
) -> tuple[list[AnyMessage], ChatAgentOutput]:
    """传入已组装的历史，返回本次执行新增的模型和工具消息。"""
    now = datetime.now(ZoneInfo(settings.rag_timezone))
    time_context = SystemMessage(
        content=(
            f"本次请求的当前时间为 {now.isoformat()}，时区为 {settings.rag_timezone}。"
            f"当前年份为 {now.year}；年份换算以此为准：去年是 {now.year - 1} 年，"
            f"今年是 {now.year} 年，明年是 {now.year + 1} 年。"
            f"例如，去年11月15日对应 {now.year - 1}-11-15。"
            "不要使用模型自身的当前日期或知识截止时间推算。"
        )
    )
    response = await run_agent(
        chat_agent,
        messages=[time_context, *messages],
        context=context,
        config=config,
    )

    ms = response.messages

    try:
        chat_agent_output = ChatAgentOutput.model_validate(response.structured_response)
    except ValidationError as error:
        raise (error)

    return ms, chat_agent_output
