"""注册 HTTP 层统一异常处理器."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from harness.exceptions import (
    RoomAuthorizationError,
)
from transport.exception_maps import HTTP_EXCEPTION_MAP
from transport.grpc.exceptions import GrpcAuthNotConfiguredError
from transport.http.exceptions import MissingAuthTokenError


async def _handle_room_authorization_error(
    _request: Request,
    error: Exception,
) -> JSONResponse:
    """转换 Harness 认证和权限异常."""
    if not isinstance(error, RoomAuthorizationError):
        raise error

    status_code, detail = HTTP_EXCEPTION_MAP[type(error)]
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
    )


async def _handle_missing_auth_token(
    _request: Request,
    _error: Exception,
) -> JSONResponse:
    """转换 HTTP 请求中缺少 token 的异常."""
    status_code, detail = HTTP_EXCEPTION_MAP[MissingAuthTokenError]
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
    )


async def _handle_grpc_auth_not_configured(
    _request: Request,
    _error: Exception,
) -> JSONResponse:
    """将 gRPC 客户端未配置异常转换为 HTTP 503 响应."""
    status_code, detail = HTTP_EXCEPTION_MAP[GrpcAuthNotConfiguredError]
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
    )


def register_http_exception_handlers(app: FastAPI) -> None:
    """向 FastAPI 应用注册统一的认证异常处理器."""
    app.add_exception_handler(
        RoomAuthorizationError,
        _handle_room_authorization_error,
    )
    app.add_exception_handler(
        MissingAuthTokenError,
        _handle_missing_auth_token,
    )
    app.add_exception_handler(
        GrpcAuthNotConfiguredError,
        _handle_grpc_auth_not_configured,
    )
