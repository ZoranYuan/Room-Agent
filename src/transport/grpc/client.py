import grpc

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

    async def get_room_agent_context(
        self,
        *,
        token: str,
        room_id: str,
    ) -> room_agent_pb2.GetRoomAgentContextResponse:
        try:
            return await self._stub.GetRoomAgentContext(
                room_agent_pb2.GetRoomAgentContextRequest(
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

    async def close(self) -> None:
        await self._channel.close()
