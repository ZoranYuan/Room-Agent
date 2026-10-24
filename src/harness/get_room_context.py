"""调用 IM 后端校验 Room 权限并获取 Room 上下文."""

from transport.grpc.client import GrpcClient
from transport.grpc.proto.room_agent_pb2 import GetRoomAgentContextResponse


async def get_room_context(
    *,
    token: str,
    room_id: str,
    grpc_client: GrpcClient,
) -> GetRoomAgentContextResponse:
    """通过 IM 后端校验 token 和 Room 权限，并获取未读消息上下文."""
    return await grpc_client.get_room_agent_context(
        token=token,
        room_id=room_id,
    )
