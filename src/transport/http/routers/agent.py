"""定义 Room Agent 的 HTTP 路由."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Cookie, Request
from fastapi.responses import StreamingResponse

from agent.runtime import agent_graph
from agent.state import AgentTask, RoomAgentState, UnreadMessage
from application.auth import get_room_context
from transport.grpc.client import GrpcClient
from transport.grpc.exceptions import GrpcAuthNotConfiguredError
from transport.grpc.proto.room_agent_pb2 import GetRoomAgentContextResponse
from transport.http.exceptions import MissingAuthTokenError
from transport.http.schemas.agent import AgentTaskRequest, ChatTaskRequest

logger = logging.getLogger(__name__)

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
) -> AsyncIterator[str]:
    """执行一次 Agent 任务，并将 Graph 事件编码为 SSE."""
    yield _encode_sse(
        "connected",
        {"task": state.task.value, "thread_id": state.thread_id},
    )

    try:
        async for mode, chunk in agent_graph.astream(
            state,
            config={"configurable": {"thread_id": state.thread_id}},
            stream_mode=["custom", "updates"],
        ):
            if not isinstance(chunk, dict):
                    raise TypeError(f"事件数据不是字典: {type(chunk)}")
            if mode == "custom":
                event_name = chunk.get("type", "agent_event")
                yield _encode_sse(event_name, chunk)
            elif mode == "updates":
                for node_update in chunk.values():
                    result = node_update.get("last_result")
                    if result is not None:
                        yield _encode_sse("task_completed", {"result": result})
    except Exception:
        logger.exception(
            "Agent 运行失败，请联系管理员",
            extra={"room_id": state.room_id, "task": state.task.value},
        )
        yield _encode_sse("error", {"message": "Agent 任务执行失败"})


def _encode_sse(event: str, data: dict[str, object]) -> str:
    """将事件编码为 SSE 协议文本."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _build_agent_state(
    *,
    room_id: str,
    request: AgentTaskRequest,
    context: GetRoomAgentContextResponse,
) -> RoomAgentState:
    """将 HTTP 任务和 IM 返回的上下文转换为 Graph 状态."""
    unread_messages = [
        UnreadMessage(
            message_id=message.message_id,
            conversation_id=message.conversation_id,
            sender_id=message.sender_id,
            seq=message.seq,
            content=message.content,
            status=message.status,
            send_time=message.send_time,
        )
        for message in context.unread_snapshot.messages
    ]

    question = request.question if isinstance(request, ChatTaskRequest) else None
    return RoomAgentState(
        task=AgentTask(request.task),
        room_id=room_id,
        user_id=context.user_id,
        thread_id=context.thread_id,
        question=question,
        unread_message_count=context.unread_message_count,
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
    response = await get_room_context(
        token=token,
        room_id=room_id,
        grpc_client=get_grpc_client(request),
    )

    state = _build_agent_state(room_id=room_id, request=body, context=response)
    return StreamingResponse(
        _room_event_stream(state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
