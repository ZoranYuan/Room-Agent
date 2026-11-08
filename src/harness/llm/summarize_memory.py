"""调用记忆摘要模型，校验并返回压缩后的历史背景。"""

import json
from collections.abc import Sequence

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig

from agent.model import memory_summary_model
from agent.prompts.memory import MEMORY_PROMPT
from config.config import settings
from constants.exception import (
    DETAIL_EMPTY_MEMORY_SUMMARY,
    DETAIL_INVALID_MEMORY_SUMMARY,
    DETAIL_MEMORY_SUMMARY_TRUNCATED,
)

memory_chain = MEMORY_PROMPT | memory_summary_model


async def summarize_memory(
    *,
    previous_summary: str | None,
    messages: Sequence[AnyMessage],
    config: RunnableConfig,
) -> str:
    """仅返回有效摘要；失败交由节点决定保留原状态。"""
    messages_json = json.dumps(
        [
            {
                "role": message.type,
                "content": message.text,
                "tool_calls": getattr(message, "tool_calls", None),
                "tool_call_id": getattr(message, "tool_call_id", None),
            }
            for message in messages
        ],
        ensure_ascii=False,
    )
    response = await memory_chain.ainvoke(
        {
            "previous_summary": previous_summary or "（暂无）",
            "messages_json": messages_json,
            "max_tokens": settings.memory_summary_max_tokens,
        },
    )
    if (
        not isinstance(response, AIMessage)
        or response.tool_calls
        or response.invalid_tool_calls
    ):
        raise ValueError(DETAIL_INVALID_MEMORY_SUMMARY)
    if response.response_metadata.get("finish_reason") == "length":
        raise ValueError(DETAIL_MEMORY_SUMMARY_TRUNCATED)
    summary = response.text.strip()
    if not summary:
        raise ValueError(DETAIL_EMPTY_MEMORY_SUMMARY)
    return summary
