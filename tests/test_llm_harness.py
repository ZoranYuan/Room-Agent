import importlib
import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import SecretStr

from agent.context import AgentRuntimeContext
from agent.graph import build_dev_agent_graph
from agent.state import AgentTask, RoomAgentState
from agent.tools.get_unread_messages import get_unread_messages
from agent.tools.search_room_messages import search_messages
from domain.auth_scope import AuthScope

chat = importlib.import_module("harness.llm.chat")
summary = importlib.import_module("harness.llm.summary")
memory = importlib.import_module("harness.llm.memory")


@pytest.fixture
def anyio_backend():
    return "asyncio"


class ToolAwareFakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.anyio
@pytest.mark.parametrize("task", [AgentTask.CHAT, AgentTask.SUMMARY])
async def test_graph_keeps_native_text_stream_tool_events_and_new_messages(monkeypatch, task):
    scope = AuthScope(room_id="room", token=SecretStr("test"))
    context = AgentRuntimeContext(auth_scope=scope)
    if task is AgentTask.CHAT:
        module, attribute, node_name = chat, "chat_agent", "chat"
        tool = search_messages
        tool_args = {"query": "昨天聊了什么？"}
    else:
        module, attribute, node_name = summary, "summary_agent", "summarize"
        tool = get_unread_messages
        tool_args = {}
        tool_module = importlib.import_module("agent.tools.get_unread_messages")
        get_messages = AsyncMock(return_value=[])
        monkeypatch.setattr(tool_module, "get_room_unread_messages", get_messages)

    model = ToolAwareFakeModel(messages=iter([
        AIMessage(content="", additional_kwargs={"tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": tool.name, "arguments": json.dumps(tool_args)},
        }]}),
        AIMessage(content="测试回答"),
    ]))
    monkeypatch.setattr(module, attribute, create_agent(
        model=model, tools=[tool], context_schema=AgentRuntimeContext, checkpointer=False,
    ))
    graph = build_dev_agent_graph()
    state = RoomAgentState(task=task, question="昨天聊了什么？" if task is AgentTask.CHAT else None)
    events = [event async for event in graph.astream(
        state,
        context=context,
        stream_mode=["messages", "custom", "updates"],
        subgraphs=True,
    )]
    tool_events = [payload for _, mode, payload in events if mode == "custom"]
    assert [event["type"] for event in tool_events] == ["tool_start", "tool_end"]
    token_positions = [
        index for index, (_, mode, payload) in enumerate(events)
        if mode == "messages" and isinstance(payload[0], AIMessage) and payload[0].text
    ]
    final_position = next(
        index for index, (namespace, mode, payload) in enumerate(events)
        if not namespace and mode == "updates" and node_name in payload
    )
    assert token_positions and min(token_positions) < final_position
    update = events[final_position][2][node_name]["messages"]
    assert sum(isinstance(message, HumanMessage) for message in update) == (task is AgentTask.CHAT)
    assert isinstance(update[-1], AIMessage) and update[-1].text == "测试回答"
    assert any(isinstance(message, ToolMessage) and message.tool_call_id == "call_1" for message in update)
    if task is AgentTask.SUMMARY:
        get_messages.assert_awaited_once_with(auth_scope=scope)


@pytest.mark.anyio
async def test_memory_model_keeps_nostream_and_returns_valid_summary(monkeypatch):
    chain = Mock()
    chain.ainvoke = AsyncMock(return_value=AIMessage(content="历史摘要"))
    monkeypatch.setattr(memory, "memory_chain", chain)
    result = await memory.summarize_memory(
        previous_summary="旧摘要", messages=[HumanMessage(content="我叫小明")], config={"tags": ["request"]},
    )
    assert result == "历史摘要"
    args, kwargs = chain.ainvoke.call_args
    assert args[0]["previous_summary"] == "旧摘要"
    assert "我叫小明" in args[0]["messages_json"]
    assert set(kwargs["config"]["tags"]) == {"request", "nostream"}


@pytest.mark.anyio
@pytest.mark.parametrize("response", [
    AIMessage(content=""),
    AIMessage(content="被截断", response_metadata={"finish_reason": "length"}),
])
async def test_memory_rejects_empty_or_truncated_summary(monkeypatch, response):
    chain = Mock()
    chain.ainvoke = AsyncMock(return_value=response)
    monkeypatch.setattr(memory, "memory_chain", chain)
    with pytest.raises(ValueError):
        await memory.summarize_memory(previous_summary=None, messages=[], config={})
