"""定义房间摘要工作的节点"""

import json
from collections.abc import Awaitable, Callable
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    message_chunk_to_message,
)
from langchain_core.runnables import RunnableConfig

from agent.model import summary_model
from agent.prompts.summary import SUMMARY_PROMPT
from agent.state import RoomAgentState
from transport.grpc.proto.room_agent_pb2 import MessageStatus

SummaryNode = Callable[
    [RoomAgentState, RunnableConfig],
    Awaitable[dict[str, list[AnyMessage]]],
]

summary_chain = SUMMARY_PROMPT | summary_model


async def summarize_node(
    state: RoomAgentState, config: RunnableConfig
) -> dict[str, list[AnyMessage]]:
    """通过原生消息流输出摘要，并保留模型消息 ID 以避免重复转发."""
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
            text = "当前没有未读消息。"
        else:
            text = "当前未读快照中没有可用于摘要的正常文本消息。"
        return {"messages": [AIMessage(content=text)]}

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

    collected: AIMessageChunk | None = None

    async for chunk in summary_chain.astream(
        {"unread_messages": unread_messages},
        config=config,
    ):
        if not isinstance(chunk, AIMessageChunk):
            raise TypeError("摘要模型返回的流式数据不是 AIMessageChunk")
        collected = chunk if collected is None else collected + chunk

    if collected is None or not collected.text.strip():
        return {"messages": [AIMessage(content="未能生成房间摘要。")]}

    response = message_chunk_to_message(collected)
    if not isinstance(response, AIMessage):
        raise TypeError("摘要模型输出不是 AIMessage")
    return {"messages": [response]}
