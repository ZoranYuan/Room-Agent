"""在进入聊天 Agent 前增量压缩较早的会话历史."""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langgraph.config import get_stream_writer

from agent.memory import (
    count_memory_tokens,
    split_recent_turns,
    summary_messages,
)
from agent.model import memory_summary_model
from agent.prompts.memory import MEMORY_PROMPT
from agent.state import RoomAgentState
from config.config import settings

logger = logging.getLogger(__name__)

memory_chain = MEMORY_PROMPT | memory_summary_model


async def prepare_memory_node(
    state: RoomAgentState, config: RunnableConfig
) -> dict[str, Any]:
    """成功后同时更新摘要和移除旧消息；失败时保持原状态，由聊天节点临时裁剪输入."""
    if not state.messages or not (state.question or "").strip():
        return {}

    writer = get_stream_writer()
    estimated_tokens = count_memory_tokens(
        [
            *summary_messages(state.conversation_summary),
            *state.messages,
            HumanMessage(content=state.question or ""),
        ]
    )
    if estimated_tokens < settings.memory_trigger_tokens:
        return {}

    older, _recent = split_recent_turns(
        state.messages, keep_tokens=settings.memory_keep_tokens
    )
    if not older:
        # 没有 older 不需要压缩
        return {}

    try:
        removals = []
        for message in older:
            if message.id is None:
                raise ValueError("待压缩的历史消息缺少消息 ID")
            removals.append(RemoveMessage(id=message.id))

        messages_json = json.dumps(
            [
                {
                    "role": message.type,
                    "content": message.text,
                    "tool_calls": getattr(message, "tool_calls", None),
                    "tool_call_id": getattr(message, "tool_call_id", None),
                }
                for message in older
            ],
            ensure_ascii=False,
        )
        response = await memory_chain.ainvoke(
            {
                "previous_summary": state.conversation_summary or "（暂无）",
                "messages_json": messages_json,
                "max_tokens": settings.memory_summary_max_tokens,
            },
            config=merge_configs(
                config,
                {"tags": ["nostream"]},   # nostream 是原生标签，让原生的 messages 流不再转发 LLM 的调用的消息
            ),
        )
        if (
            not isinstance(response, AIMessage)
            or response.tool_calls
            or response.invalid_tool_calls
        ):
            raise ValueError("会话记忆模型未返回有效的摘要消息")
        if response.response_metadata.get("finish_reason") == "length":
            raise ValueError("会话摘要被输出长度限制截断，本次保留原始消息")
        summary = response.text.strip()
        if not summary:
            raise ValueError("会话记忆模型返回空摘要")
    except Exception:
        logger.warning("会话记忆压缩失败，本轮临时采用最近完整对话", exc_info=True)
        return {}

    logger.info(
        "会话记忆已压缩：移除 %s 条旧消息，原估算 Token=%s",
        len(older),
        estimated_tokens,
    )

    writer(
        {
            "type": "compress",
        }
    )

    return {
        "conversation_summary": summary,
        "summarized_through_message_id": older[-1].id,
        "messages": removals,
    }
