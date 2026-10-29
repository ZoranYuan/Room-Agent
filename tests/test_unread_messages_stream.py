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

unread = importlib.import_module("harness.grpc.get_room_unread_messages")


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


def message(seq, **overrides):
    fields = dict(
        message_id=f"m{seq}", conversation_id="conv", sender_id="u1",
        sender_name="小明", seq=seq, content=f"消息{seq}", send_time=1000 + seq,
    )
    fields.update(overrides)
    return pb.ChatMessage(**fields)


@pytest.mark.anyio
async def test_stream_is_collected_filtered_sorted_and_authorized(monkeypatch):
    calls = []

    class Servicer(rpc.RoomAgentContextServiceServicer):
        async def GetRoomUnreadMessages(self, request, context):
            calls.append(request)
            assert dict(context.invocation_metadata())["authorization"] == "Bearer test-token"
            yield pb.GetRoomUnreadMessagesResponse(messages=[message(3), message(2, status=1)])
            yield pb.GetRoomUnreadMessagesResponse()
            yield pb.GetRoomUnreadMessagesResponse(messages=[
                message(1), message(4, content=" "), message(5, type=9),
            ])

    async with local_client(Servicer()) as client:
        monkeypatch.setattr(unread, "get_grpc_client", lambda: client)
        result = await unread.get_room_unread_messages(
            auth_scope=AuthScope(room_id="room", token=SecretStr("test-token")),
            limit=2,
        )
    assert [item.seq for item in result] == [1, 3]
    assert result[0].sender_name == "小明" and result[0].send_time == 1001
    assert len(calls) == 1 and calls[0].room_id == "room" and calls[0].limit == 2


@pytest.mark.anyio
@pytest.mark.parametrize("status, exception", [
    (grpc.StatusCode.UNAUTHENTICATED, InvalidTokenError),
    (grpc.StatusCode.PERMISSION_DENIED, RoomAccessDeniedError),
    (grpc.StatusCode.UNAVAILABLE, grpc.aio.AioRpcError),
])
async def test_stream_failure_does_not_return_partial_snapshot(monkeypatch, status, exception):
    class Servicer(rpc.RoomAgentContextServiceServicer):
        async def GetRoomUnreadMessages(self, request, context):
            yield pb.GetRoomUnreadMessagesResponse(messages=[message(1)])
            await context.abort(status, "模拟流中断")

    async with local_client(Servicer()) as client:
        monkeypatch.setattr(unread, "get_grpc_client", lambda: client)
        with pytest.raises(exception):
            await unread.get_room_unread_messages(
                auth_scope=AuthScope(room_id="room", token=SecretStr("test-token")),
            )


@pytest.mark.anyio
async def test_closing_generator_cancels_server_stream():
    stopped = asyncio.Event()

    class Servicer(rpc.RoomAgentContextServiceServicer):
        async def GetRoomUnreadMessages(self, request, context):
            try:
                yield pb.GetRoomUnreadMessagesResponse(messages=[message(1)])
                await asyncio.Event().wait()
            finally:
                stopped.set()

    async with local_client(Servicer()) as client:
        async with aclosing(client.get_room_unread_messages(token="test", room_id="room")) as stream:
            assert (await anext(stream)).messages[0].seq == 1
        await asyncio.wait_for(stopped.wait(), timeout=3)


@pytest.mark.anyio
async def test_empty_stream_returns_empty_snapshot(monkeypatch):
    class Servicer(rpc.RoomAgentContextServiceServicer):
        async def GetRoomUnreadMessages(self, request, context):
            if False:
                yield pb.GetRoomUnreadMessagesResponse()

    async with local_client(Servicer()) as client:
        monkeypatch.setattr(unread, "get_grpc_client", lambda: client)
        assert await unread.get_room_unread_messages(
            auth_scope=AuthScope(room_id="room", token=SecretStr("test")),
        ) == []


@pytest.mark.anyio
async def test_stream_deadline_is_applied():
    class Servicer(rpc.RoomAgentContextServiceServicer):
        async def GetRoomUnreadMessages(self, request, context):
            await asyncio.Event().wait()
            yield pb.GetRoomUnreadMessagesResponse()

    async with local_client(Servicer()) as client:
        with pytest.raises(grpc.aio.AioRpcError) as error:
            async for _ in client.get_room_unread_messages(
                token="test", room_id="room", timeout_seconds=0.1,
            ):
                pass
        assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
