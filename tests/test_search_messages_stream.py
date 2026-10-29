import asyncio
import importlib
from contextlib import aclosing, asynccontextmanager

import grpc
import pytest
from pydantic import SecretStr

from domain.auth_scope import AuthScope
from harness.exceptions import InvalidTokenError, RoomAccessDeniedError
from transport.grpc.client import GrpcClient
from transport.grpc.proto import room_agent_pb2 as pb
from transport.grpc.proto import room_agent_pb2_grpc as rpc

search = importlib.import_module("harness.grpc.search_room_messages")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def local_client(servicer):
    server = grpc.aio.server()
    rpc.add_RoomAgentContextServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    client = GrpcClient(server_url=f"127.0.0.1:{port}")
    try:
        yield client
    finally:
        try:
            await client.close()
        finally:
            await server.stop(0)


def scope():
    return AuthScope(room_id="room", token=SecretStr("test-token"))


def message(seq, **overrides):
    fields = dict(
        message_id=f"m{seq}", conversation_id="conv", sender_id="u1",
        sender_name="小明", seq=seq, content=f"消息{seq}", send_time=1000 + seq,
    )
    fields.update(overrides)
    return pb.ChatMessage(**fields)


@pytest.mark.anyio
async def test_search_stream_sends_all_ranges_and_collects_valid_messages(monkeypatch):
    calls = []

    class Servicer(rpc.RoomAgentContextServiceServicer):
        async def SearchRoomMessages(self, request, context):
            calls.append(request)
            assert dict(context.invocation_metadata())["authorization"] == "Bearer test-token"
            yield pb.SearchRoomMessagesResponse(messages=[message(7), message(2, status=1)])
            yield pb.SearchRoomMessagesResponse()
            yield pb.SearchRoomMessagesResponse(messages=[
                message(1), message(4, content=" "), message(5, type=9),
            ])

    async with local_client(Servicer()) as client:
        monkeypatch.setattr(search, "get_grpc_client", lambda: client)
        result = await search.search_room_messages(
            auth_scope=scope(), ranges=[(1, 5), (7, 9)], limit=2,
        )
    assert [item.seq for item in result] == [1, 7]
    assert result[0].sender_name == "小明" and result[0].send_time == 1001
    assert len(calls) == 1 and calls[0].room_id == "room" and calls[0].limit == 2
    assert [(item.seq_start, item.seq_end) for item in calls[0].ranges] == [(1, 5), (7, 9)]


@pytest.mark.anyio
@pytest.mark.parametrize("status, exception", [
    (grpc.StatusCode.UNAUTHENTICATED, InvalidTokenError),
    (grpc.StatusCode.PERMISSION_DENIED, RoomAccessDeniedError),
    (grpc.StatusCode.UNAVAILABLE, grpc.aio.AioRpcError),
])
async def test_search_failure_does_not_return_partial_messages(monkeypatch, status, exception):
    class Servicer(rpc.RoomAgentContextServiceServicer):
        async def SearchRoomMessages(self, request, context):
            yield pb.SearchRoomMessagesResponse(messages=[message(1)])
            await context.abort(status, "模拟流中断")

    async with local_client(Servicer()) as client:
        monkeypatch.setattr(search, "get_grpc_client", lambda: client)
        with pytest.raises(exception):
            await search.search_room_messages(auth_scope=scope(), ranges=[(1, 5)])


@pytest.mark.anyio
async def test_closing_search_generator_cancels_server_stream():
    stopped = asyncio.Event()

    class Servicer(rpc.RoomAgentContextServiceServicer):
        async def SearchRoomMessages(self, request, context):
            try:
                yield pb.SearchRoomMessagesResponse(messages=[message(1)])
                await asyncio.Event().wait()
            finally:
                stopped.set()

    async with local_client(Servicer()) as client:
        async with aclosing(client.search_room_messages(
            token="test", room_id="room", ranges=[(1, 5)],
        )) as stream:
            assert (await anext(stream)).messages[0].seq == 1
        await asyncio.wait_for(stopped.wait(), timeout=3)


@pytest.mark.anyio
async def test_search_deadline_is_applied():
    class Servicer(rpc.RoomAgentContextServiceServicer):
        async def SearchRoomMessages(self, request, context):
            await asyncio.Event().wait()
            yield pb.SearchRoomMessagesResponse()

    async with local_client(Servicer()) as client:
        with pytest.raises(grpc.aio.AioRpcError) as error:
            async for _ in client.search_room_messages(
                token="test", room_id="room", ranges=[(1, 5)], timeout_seconds=0.1,
            ):
                pass
        assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED


@pytest.mark.anyio
async def test_empty_ranges_skip_grpc_call(monkeypatch):
    def fail():
        raise AssertionError("空范围不应发起 RPC")

    monkeypatch.setattr(search, "get_grpc_client", fail)
    assert await search.search_room_messages(auth_scope=scope(), ranges=[]) == []
