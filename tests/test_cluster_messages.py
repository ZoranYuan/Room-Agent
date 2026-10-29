import importlib
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from config.config import settings
from domain.chat_message import ChatMessage

cluster = importlib.import_module("harness.llm.cluster_messages")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def messages():
    return [
        ChatMessage(str(seq), "conv", "u1", "小明", seq, "你好", 0, seq)
        for seq in (1, 2)
    ]


def result(seqs=None):
    return {"groups": [{"topic": "闲聊", "seqs": ["1", "2"] if seqs is None else seqs}]}


@pytest.fixture
def chain(monkeypatch):
    chain = Mock()
    chain.ainvoke = AsyncMock(return_value=result())
    monkeypatch.setattr(cluster, "_get_cluster_chain", lambda: chain)
    monkeypatch.setattr(settings, "cluster_max_retries", 1)
    monkeypatch.setattr(cluster.asyncio, "sleep", AsyncMock())
    return chain


@pytest.mark.anyio
async def test_valid_groups_use_string_seqs_and_hide_internal_stream(chain):
    groups = await cluster.cluster_messages(messages())
    assert groups.groups[0].seqs == ["1", "2"]
    args, kwargs = chain.ainvoke.call_args
    assert '"seq": "1"' in args[0]["messages"]
    assert kwargs["config"] == {"tags": ["nostream"]}
    assert chain.ainvoke.await_count == 1


@pytest.mark.anyio
@pytest.mark.parametrize("seqs", [["1"], ["1", "1", "2"], ["1", "unknown"]])
async def test_invalid_coverage_is_retried_then_rejected(chain, seqs):
    chain.ainvoke.return_value = result(seqs)
    with pytest.raises(cluster.ClusterValidationError, match="遗漏、重复或不存在"):
        await cluster.cluster_messages(messages())
    assert chain.ainvoke.await_count == 2


@pytest.mark.anyio
async def test_invalid_structured_result_can_recover(chain):
    chain.ainvoke.side_effect = [{"groups": []}, result()]
    assert (await cluster.cluster_messages(messages())).groups[0].topic == "闲聊"
    assert chain.ainvoke.await_count == 2


@pytest.mark.anyio
async def test_connection_error_can_recover(chain):
    error = APIConnectionError(request=httpx.Request("POST", "https://example.test"))
    chain.ainvoke.side_effect = [error, result()]
    assert (await cluster.cluster_messages(messages())).groups[0].seqs == ["1", "2"]
    assert chain.ainvoke.await_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize("status", [400, 401, 403, 408, 409, 429, 500, 503])
async def test_only_transient_http_errors_are_retried(chain, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
    error = APIStatusError("模拟请求异常", response=response, body=None)
    chain.ainvoke.side_effect = [error, result()]
    if status in (400, 401, 403):
        with pytest.raises(APIStatusError):
            await cluster.cluster_messages(messages())
        assert chain.ainvoke.await_count == 1
    else:
        await cluster.cluster_messages(messages())
        assert chain.ainvoke.await_count == 2


@pytest.mark.anyio
async def test_zero_retries_and_programming_errors_propagate(chain, monkeypatch):
    monkeypatch.setattr(settings, "cluster_max_retries", 0)
    chain.ainvoke.return_value = result(["1"])
    with pytest.raises(cluster.ClusterValidationError):
        await cluster.cluster_messages(messages())
    assert chain.ainvoke.await_count == 1
    chain.ainvoke.reset_mock()
    chain.ainvoke.side_effect = TypeError("代码错误")
    with pytest.raises(TypeError, match="代码错误"):
        await cluster.cluster_messages(messages())
    assert chain.ainvoke.await_count == 1
