"""服务入口文件."""

from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import APIRouter, FastAPI
from prometheus_client import make_asgi_app

from infra.model.reranker_model_factory import get_message_reranker
from infra.mysql.database import close_mysql
from infra.rag.vector.vector_store import close_vector_store, get_vector_index
from transport.grpc.client import close_grpc_client
from transport.http.exception_handlers import register_http_exception_handlers
from transport.http.routers.agent import router as agent_router


# FastAPI 启动时登记清理函数，关闭时统一执行
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        stack.push_async_callback(close_mysql)
        stack.push_async_callback(close_grpc_client)
        stack.push_async_callback(close_vector_store)

        # 启动时完成模型和向量客户端初始化，避免首个请求承担冷启动耗时。
        await get_vector_index()
        await get_message_reranker()

        yield


app = FastAPI(lifespan=lifespan)
register_http_exception_handlers(app)
app.mount("/metrics", make_asgi_app())

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(agent_router)

app.include_router(api_router)
