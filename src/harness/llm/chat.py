"""执行房间问答 Agent。"""

from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig

from agent.context import AgentRuntimeContext
from agent.model import chat_model
from agent.prompts.chat import CHAT_PROMPT
from agent.tools.search_room_messages import search_messages
from agent.tools.sync_room_messages import sync_room_messages
from harness.llm.agent_execution import run_agent

chat_agent = create_agent(
    model=chat_model,
    tools=[search_messages, sync_room_messages],
    system_prompt=CHAT_PROMPT.format_messages()[0].text,
    context_schema=AgentRuntimeContext,
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
        chat_agent, messages=messages, context=context, config=config,
    )
