"""在进入聊天 Agent 前增量压缩较早的会话历史."""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.types import Overwrite

from agent.state import RoomAgentState
from agent.utils.memory import (
    count_memory_tokens,
    split_recent_turns,
    summary_messages,
)
from config.config import settings
from constants.exception import (
    LOG_MEMORY_COMPRESSION_FAILED,
)
from harness.llm.memory import summarize_memory

logger = logging.getLogger(__name__)

async def memory_node(
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

    older, recent = split_recent_turns(
        state.messages, keep_tokens=settings.memory_keep_tokens
    )
    if not older:
        # 没有 older 不需要压缩
        return {}

    try:
        # 记录之前的 summary
        summary = await summarize_memory(
            previous_summary=state.conversation_summary,
            messages=older,
            config=config,
        )
    except Exception:
        logger.warning(LOG_MEMORY_COMPRESSION_FAILED, exc_info=True)
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
        "messages": Overwrite(recent),
    }
