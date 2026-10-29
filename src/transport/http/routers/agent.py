"""定义 Room Agent 的 HTTP 路由."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Cookie
from fastapi.responses import StreamingResponse
from pydantic import SecretStr

from agent.context import AgentRuntimeContext
from agent.state import AgentTask, RoomAgentState
from domain.auth_scope import AuthScope
from harness.runner import stream_agent_events
from harness.grpc.verify_room_access import verify_room_access
from transport.http.exceptions import MissingAuthTokenError
from transport.http.schemas.agent import AgentTaskRequest, ChatTaskRequest

router = APIRouter(
    prefix="/rooms",
    tags=["Agent"],
)


async def _room_event_stream(
    state: RoomAgentState,
    thread_id: str,
    runtime_context: AgentRuntimeContext,
) -> AsyncIterator[str]:
    """向前端返回一个 connected 事件，表示此次的 SSE 流创建成功"""
    yield _encode_sse(
        "connected",
        {"task": state.task.value, "thread_id": thread_id},
    )

    async for event in stream_agent_events(
        state=state,
        thread_id=thread_id,
        context=runtime_context,
    ):
        yield _encode_sse(event.name, event.data)


def _encode_sse(event: str, data: dict[str, object]) -> str:
    """将事件编码为 SSE 协议文本."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _build_agent_state(
    *,
    request: AgentTaskRequest,
) -> RoomAgentState:
    """只构造本次任务输入，未读快照由摘要节点按需获取."""
    question = request.question if isinstance(request, ChatTaskRequest) else None
    return RoomAgentState(
        task=AgentTask(request.task),
        question=question,
    )


@router.post("/{room_id}/agent")
async def run_room_agent(
    room_id: str,
    body: AgentTaskRequest,
    token: Annotated[str | None, Cookie(alias="access_token")] = None,
) -> StreamingResponse:
    """校验用户后执行一个 Agent 任务，并以 SSE 返回结果."""
    if not token:
        raise MissingAuthTokenError

    auth = await verify_room_access(
        token=token,
        room_id=room_id,
    )

    state = _build_agent_state(
        request=body,
    )

    thread_id = auth.thread_id

    runtime_context = AgentRuntimeContext(
        auth_scope=AuthScope(
            room_id=room_id,
            token=SecretStr(token),
        ),
    )

    return StreamingResponse(
        _room_event_stream(state, thread_id, runtime_context),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
