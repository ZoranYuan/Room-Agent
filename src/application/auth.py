"""调用 IM 后端校验 Room 访问权限的应用服务占位."""


from transport.grpc.client import GrpcClient
from transport.grpc.proto.room_agent_pb2 import GetRoomAgentContextResponse


async def get_room_context(
    *,
    token: str,
    room_id: str,
    grpc_client: GrpcClient,
) -> GetRoomAgentContextResponse:
    """通过 IM 后端校验 token 和 Room 权限，并返回用户 ID."""
    return await grpc_client.get_room_agent_context(
        token=token,
        room_id=room_id,
    )
