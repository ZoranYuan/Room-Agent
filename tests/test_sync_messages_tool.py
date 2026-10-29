import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from domain.auth_scope import AuthScope
from domain.chat_message import ChatMessage

sync_tool = importlib.import_module("agent.tools.sync_room_messages")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("has_messages", [True, False])
async def test_returns_increment_without_waiting_for_background_index(
    monkeypatch, has_messages,
):
    scope = AuthScope(room_id="room", token=SecretStr("test-token"))
    runtime = SimpleNamespace(context=SimpleNamespace(auth_scope=scope))
    factory = MagicMock()
    repository = SimpleNamespace(get_room_latest_seq=AsyncMock(return_value=50))
    monkeypatch.setattr(sync_tool, "get_session_factory", lambda: factory)
    monkeypatch.setattr(sync_tool, "MessageChunkRepository", lambda session: repository)

    messages = [ChatMessage(
        message_id="m51", conversation_id="conv", sender_id="u1",
        sender_name="小明", seq=51, content="最新决定", status=0, send_time=1000,
    )] if has_messages else []
    fetch = AsyncMock(return_value=messages)
    monkeypatch.setattr(sync_tool, "fetch_room_messages", fetch)

    started, release = asyncio.Event(), asyncio.Event()

    async def index(**kwargs):
        assert kwargs == {"room_id": "room", "messages": messages}
        started.set()
        await release.wait()

    monkeypatch.setattr(sync_tool, "index_room_messages", index)
    # 保存测试启动的任务，以便测试结束时等待清理。
    tasks = []
    create_task = asyncio.create_task

    def tracked_task(coro):
        task = create_task(coro)
        tasks.append(task)
        return task

    monkeypatch.setattr(sync_tool.asyncio, "create_task", tracked_task)
    try:
        result = await asyncio.wait_for(sync_tool.sync_room_messages.coroutine(runtime), 1)
        assert result["status"] == "success"
        assert result["messages"] == ([{
            "message_id": "m51", "conversation_id": "conv", "sender_id": "u1",
            "sender_name": "小明", "seq": 51, "content": "最新决定",
            "status": 0, "send_time": 1000,
        }] if has_messages else [])
        fetch.assert_awaited_once_with(
            auth_scope=scope, after_seq=50, index_messages=False,
        )
        if has_messages:
            await asyncio.wait_for(started.wait(), 1)
            assert not tasks[0].done()
        else:
            assert tasks == []
    finally:
        release.set()
        await asyncio.gather(*tasks)
