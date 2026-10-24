"""定义 Room Agent 的 HTTP 路由."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Cookie, Request
from fastapi.responses import StreamingResponse

from agent.context import AgentRuntimeContext
from agent.state import AgentTask, RoomAgentState
from domain.chat_message import ChatMessage
from domain.room_scope import AuthorizedRoomScope
from harness.get_room_context import get_room_context
from harness.runner import stream_agent_events
from transport.grpc.client import GrpcClient
from transport.grpc.exceptions import GrpcAuthNotConfiguredError
from transport.grpc.proto.room_agent_pb2 import GetRoomAgentContextResponse
from transport.http.exceptions import MissingAuthTokenError
from transport.http.schemas.agent import AgentTaskRequest, ChatTaskRequest

router = APIRouter(
    prefix="/rooms",
    tags=["Agent"],
)


def get_grpc_client(request: Request) -> GrpcClient:
    grpc_client = request.app.state.grpc_client
    if grpc_client is None:
        raise GrpcAuthNotConfiguredError
    return grpc_client


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
    room_context_response: GetRoomAgentContextResponse,
) -> RoomAgentState:
    """将 HTTP 任务和 IM 返回的上下文转换为 Graph 状态."""
    unread_messages = [
        ChatMessage(
            message_id=message.message_id,
            conversation_id=message.conversation_id,
            sender_id=message.sender_id,
            seq=message.seq,
            content=message.content,
            status=message.status,
            send_time=message.send_time,
        )
        for message in room_context_response.unread_snapshot.messages
    ]

    question = request.question if isinstance(request, ChatTaskRequest) else None
    return RoomAgentState(
        task=AgentTask(request.task),
        question=question,
        unread_message_count=room_context_response.unread_message_count,
        unread_messages=unread_messages,
    )


@router.post("/{room_id}/agent")
async def run_room_agent(
    room_id: str,
    body: AgentTaskRequest,
    request: Request,
    token: Annotated[str | None, Cookie(alias="access_token")] = None,
) -> StreamingResponse:
    """校验用户后执行一个 Agent 任务，并以 SSE 返回结果."""
    if not token:
        raise MissingAuthTokenError

    # 通过 grpc 和后端服务进行通信，获取本次用户的上下文信息
    room_context_response = await get_room_context(
        token=token,
        room_id=room_id,
        grpc_client=get_grpc_client(request),
    )

    state = _build_agent_state(
        request=body,
        room_context_response=room_context_response,
    )
    thread_id = room_context_response.thread_id
    runtime_context = AgentRuntimeContext(
        authorized_scope=AuthorizedRoomScope(
            room_id=room_id,
            user_id=room_context_response.user_id,
        )
    )

    return StreamingResponse(
        _room_event_stream(state, thread_id, runtime_context),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
