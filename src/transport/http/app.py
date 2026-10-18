"""服务入口文件."""

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from config.config import settings
from transport.grpc.client import GrpcClient
from transport.http.exception_handlers import register_http_exception_handlers
from transport.http.routers.agent import router as agent_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    grpc_client = None

    if settings.grpc_url:
        grpc_client = GrpcClient(
            server_url=settings.grpc_url,
            timeout_seconds=settings.grpc_time_out,
        )

    app.state.grpc_client = grpc_client

    try:
        yield
    finally:
        if grpc_client is not None:
            await grpc_client.close()

app = FastAPI(lifespan=lifespan)
register_http_exception_handlers(app)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(agent_router)

app.include_router(api_router)
