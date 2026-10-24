"""集中定义跨传输层的异常映射."""

import grpc

from harness.exceptions import InvalidTokenError, RoomAccessDeniedError
from constants.exception import (
    DETAIL_IM_GRPC_AUTH_NOT_CONFIGURED,
    DETAIL_INVALID_AUTH_TOKEN,
    DETAIL_MISSING_AUTH_TOKEN,
    DETAIL_ROOM_ACCESS_DENIED,
)
from transport.grpc.exceptions import GrpcAuthNotConfiguredError
from transport.http.exceptions import MissingAuthTokenError

HTTP_EXCEPTION_MAP: dict[type[Exception], tuple[int, str]] = {
    MissingAuthTokenError: (401, DETAIL_MISSING_AUTH_TOKEN),
    GrpcAuthNotConfiguredError: (503, DETAIL_IM_GRPC_AUTH_NOT_CONFIGURED),
    InvalidTokenError: (401, DETAIL_INVALID_AUTH_TOKEN),
    RoomAccessDeniedError: (403, DETAIL_ROOM_ACCESS_DENIED),
}

GRPC_STATUS_EXCEPTION_MAP: dict[grpc.StatusCode, type[Exception]] = {
    grpc.StatusCode.UNAUTHENTICATED: InvalidTokenError,
    grpc.StatusCode.PERMISSION_DENIED: RoomAccessDeniedError,
}
