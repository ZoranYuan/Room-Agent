"""通过 IM 校验房间访问权限，不读取未读消息."""

from transport.grpc.client import get_grpc_client
from transport.grpc.proto.room_agent_pb2 import VerifyRoomAccessResponse

from constants.exception import (
    DETAIL_IM_THREAD_ID_MISSING,
)


async def verify_room_access(
    *, token: str, room_id: str
) -> VerifyRoomAccessResponse:
    """校验当前用户的房间访问权限，返回 Agent Thread 标识."""
    grpc_client = get_grpc_client()
    response = await grpc_client.verify_room_access(token=token, room_id=room_id)
    if not response.thread_id:
        raise RuntimeError(DETAIL_IM_THREAD_ID_MISSING)
    return response
