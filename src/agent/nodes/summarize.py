"""定义房间摘要工作的节点"""

import json
from collections.abc import Awaitable, Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from agent.model import summary_model
from agent.prompts.summary import SUMMARY_PROMPT
from agent.state import RoomAgentState
from transport.grpc.proto.room_agent_pb2 import MessageStatus

SummaryNode = Callable[
    [RoomAgentState],
    Awaitable[dict[str, str]],
]

summary_chain = SUMMARY_PROMPT | summary_model | StrOutputParser()


async def create_summarize_node(
    state: RoomAgentState, config: RunnableConfig
) -> dict[str, str]:
    # 清洗数据
    messages = sorted(
        (
            message
            for message in state.unread_messages
            if message.status == MessageStatus.MESSAGE_STATUS_NORMAL
            and message.content.strip()
        ),
        key=lambda message: message.seq,
    )

    if not messages:
        if state.unread_message_count == 0:
            return {"last_result": "当前没有未读消息。"}

        return {"last_result": "当前未读快照中没有可用于摘要的正常文本消息"}

    unread_messages = json.dumps(
        [
            {
                "seq": message.seq,
                "sender_id": message.sender_id,
                "send_time": message.send_time,
                "content": message.content,
            }
            for message in messages
        ],
        ensure_ascii=False,
    )

    writer = get_stream_writer()
    summary_parts: list[str] = []

    async for chunk in summary_chain.astream(
        {"unread_messages": unread_messages},
        config=config,
    ):
        if not chunk:
            continue
        summary_parts.append(chunk)
        writer({"type": "summary_token", "content": chunk})

    summary = "".join(summary_parts).strip()

    # 更新状态值
    return {"last_result": summary or "未能生成房间摘要。"}
