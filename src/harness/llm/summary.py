"""执行未读消息摘要 Agent。"""

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.context import AgentRuntimeContext
from agent.model import summary_model
from agent.prompts.summary import SUMMARY_PROMPT
from agent.tools.get_unread_messages import get_unread_messages
from harness.llm.agent_execution import run_agent

summary_agent = create_agent(
    model=summary_model,
    tools=[get_unread_messages],
    system_prompt=SUMMARY_PROMPT.format_messages()[0].text,
    context_schema=AgentRuntimeContext,
    checkpointer=False,
)


async def run_summary(
    *,
    context: AgentRuntimeContext,
    config: RunnableConfig | None = None,
) -> list[AnyMessage]:
    """查询本次已授权范围内的未读消息并生成摘要。"""
    return await run_agent(
        summary_agent,
        messages=[HumanMessage(content="请先调用 get_unread_messages 获取当前房间未读消息，再生成摘要。")],
        context=context,
        config=config,
    )
