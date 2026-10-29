from typing import AsyncGenerator

import grpc

from config.config import settings
from transport.exception_maps import GRPC_STATUS_EXCEPTION_MAP
from transport.grpc.exceptions import GrpcAuthNotConfiguredError
from transport.grpc.proto import room_agent_pb2, room_agent_pb2_grpc


class GrpcClient:
    def __init__(
        self,
        *,
        server_url: str,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not server_url:
            raise GrpcAuthNotConfiguredError
        self._channel = grpc.aio.insecure_channel(server_url)
        self._stub = room_agent_pb2_grpc.RoomAgentContextServiceStub(self._channel)
        self._timeout_seconds = timeout_seconds

    async def search_room_messages(
        self,
        *,
        token: str,
        room_id: str,
        ranges: list[tuple[int, int]],
        limit: int = 100,
        timeout_seconds: float = 60.0,
    ) -> AsyncGenerator[room_agent_pb2.SearchRoomMessagesResponse, None]:
        """按指定 seq 范围流式读取当前用户可见的房间消息。"""
        call = self._stub.SearchRoomMessages(
            room_agent_pb2.SearchRoomMessagesRequest(
                room_id=room_id,
                ranges=[
                    room_agent_pb2.MessageSeqRange(seq_start=start, seq_end=end)
                    for start, end in ranges
                ],
                limit=limit,
            ),
            metadata=(("authorization", f"Bearer {token}"),),
            timeout=timeout_seconds,
        )

        try:
            async for batch in call:
                yield batch
        except grpc.aio.AioRpcError as error:
            exception_type = GRPC_STATUS_EXCEPTION_MAP.get(error.code())
            if exception_type is not None:
                raise exception_type from error
            raise
        finally:
            if not call.done():
                call.cancel()

    async def sync_room_messages(
        self,
        *,
        token: str,
        room_id: str,
        after_seq: int,
        limit: int = 100,
        timeout_seconds: float = 60.0,
    ) -> AsyncGenerator[room_agent_pb2.SyncRoomMessagesResponse, None]:
        """按房间流式补齐指定 seq 之后的消息."""
        request = room_agent_pb2.SyncRoomMessagesRequest(
            room_id=room_id,
            after_seq=after_seq,
            limit=limit,
        )

        call = self._stub.SyncRoomMessages(
            request,
            metadata=(("authorization", f"Bearer {token}"),),
            timeout=timeout_seconds,
        )

        try:
            async for batch in call:
                yield batch
        except grpc.aio.AioRpcError as error:
            exception_type = GRPC_STATUS_EXCEPTION_MAP.get(error.code())
            if exception_type is not None:
                raise exception_type from error
            raise
        finally:
            # 消费方提前退出时，取消仍在进行的 RPC。
            if not call.done():
                call.cancel()

    async def verify_room_access(
        self,
        *,
        token: str,
        room_id: str,
    ) -> room_agent_pb2.VerifyRoomAccessResponse:
        try:
            return await self._stub.VerifyRoomAccess(
                room_agent_pb2.VerifyRoomAccessRequest(
                    room_id=room_id,
                ),
                metadata=(("authorization", f"Bearer {token}"),),
                timeout=self._timeout_seconds,
            )
        except grpc.aio.AioRpcError as error:
            exception_type = GRPC_STATUS_EXCEPTION_MAP.get(error.code())
            if exception_type is not None:
                raise exception_type from error
            raise

    async def get_room_unread_messages(
        self,
        *,
        token: str,
        room_id: str,
        limit: int = 100,
        timeout_seconds: float = 60.0,
    ) -> AsyncGenerator[room_agent_pb2.GetRoomUnreadMessagesResponse, None]:
        """流式获取当前用户在指定房间的本次未读消息快照."""
        call = self._stub.GetRoomUnreadMessages(
            room_agent_pb2.GetRoomUnreadMessagesRequest(
                room_id=room_id,
                limit=limit,
            ),
            metadata=(("authorization", f"Bearer {token}"),),
            timeout=timeout_seconds,
        )

        try:
            async for batch in call:
                yield batch
        except grpc.aio.AioRpcError as error:
            exception_type = GRPC_STATUS_EXCEPTION_MAP.get(error.code())
            if exception_type is not None:
                raise exception_type from error
            raise
        finally:
            if not call.done():
                call.cancel()

    async def close(self) -> None:
        await self._channel.close()


_client: GrpcClient | None = None


def get_grpc_client() -> GrpcClient:
    """在异步调用中首次创建客户端，后续调用复用同一实例."""
    global _client
    if _client is None:
        _client = GrpcClient(
            server_url=settings.grpc_url,
            timeout_seconds=settings.grpc_time_out,
        )
    return _client


async def close_grpc_client() -> None:
    """关闭并清除当前进程共享的客户端."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
