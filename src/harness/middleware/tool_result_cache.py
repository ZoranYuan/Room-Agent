"""缓存工具调用的结果，按照 roomId + tool_name + 规范化后的参数"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command

from agent.context import AgentRuntimeContext

_CACHEABLE_TOOLS = frozenset(
    {
        "search_messages",
        "sync_room_messages",
    }
)


@dataclass
class _RunToolCache:
    """保存一次 Agent 运行内的成功结果缓存和并发锁。"""

    results: dict[str, ToolMessage] = field(default_factory=dict)
    locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    locks_guard: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def get_lock(self, key: str) -> asyncio.Lock:
        """为相同缓存键返回同一把锁。"""
        async with self.locks_guard:
            lock = self.locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self.locks[key] = lock
            return lock


_current_run_cache: ContextVar[_RunToolCache | None] = ContextVar(
    "agent_tool_result_cache", default=None
)


@asynccontextmanager
async def tool_result_cache_scope() -> AsyncGenerator[None, None]:
    """为一次 Agent 运行创建独立缓存，并在运行结束后清理."""
    token = _current_run_cache.set(_RunToolCache())

    try:
        yield
    finally:
        _current_run_cache.reset(token)


def _build_cache_key(request: ToolCallRequest) -> str | None:
    tool_call = request.tool_call
    tool_name = tool_call["name"]

    if tool_name not in _CACHEABLE_TOOLS:
        return None

    context = request.runtime.context

    if not isinstance(context, AgentRuntimeContext):
        return None

    key_data = {
        "room_id": context.auth_scope.room_id,
        "tool_name": tool_name,
        "args": tool_call.get("args", {}),
    }

    return json.dumps(
        key_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


class ToolResultCacheMiddleware(AgentMiddleware[Any, AgentRuntimeContext, Any]):
    """在同一轮 Agent 执行中复用相同工具调用的成功结果."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        cache = _current_run_cache.get()
        key = _build_cache_key(request)

        if cache is None or key is None:
            return await handler(request)

        # 类似于 singleflight 可以合并请求
        lock = await cache.get_lock(key)
        async with lock:
            cached = cache.results.get(key)
            if cached is not None:
                # 每次模型工具调用都有自己的 tool_call_id。
                return cached.model_copy(
                    update={
                        "tool_call_id": request.tool_call["id"],
                        "id": None,
                    }
                )

            # 缓存工具调用
            result = await handler(request)

            if isinstance(result, ToolMessage) and result.status == "success":
                cache.results[key] = result

            return result
