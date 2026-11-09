"""调用聚类模型，统一校验结果并管理重试。"""

import asyncio
import json
import logging
from collections.abc import Sequence

from langchain_core.exceptions import OutputParserException
from openai import APIConnectionError, APIStatusError
from pydantic import ValidationError

from agent.model import topic_cluster_chain
from agent.schemas.cluster import MessageGroups
from config.config import settings
from constants.exception import (
    DETAIL_INVALID_CLUSTER_RETRIES,
    DETAIL_INVALID_MESSAGE_GROUPS,
    LOG_MESSAGE_CLUSTER_RETRY,
)
from domain.chat_message import ChatMessage

logger = logging.getLogger(__name__)


class ClusterValidationError(ValueError):
    """模型分组包含输入中不存在的消息序列号。"""


async def cluster_messages(messages: Sequence[ChatMessage]) -> MessageGroups:
    """返回校验通过的分组，仅重试临时请求错误和无效模型输出。"""
    expected = {str(message.seq) for message in messages}
    messages_json = json.dumps(
        [
            {
                "seq": str(message.seq),
                "send_time": message.send_time,
                "content": message.content,
            }
            for message in messages
        ],
        ensure_ascii=False,
    )

    # 统一使用 LLM_MAX_RETRIES；max_retries 是首次调用后的额外重试次数。
    for attempt in range(settings.llm_max_retries + 1):
        try:
            result = await topic_cluster_chain.ainvoke(
                {"messages": messages_json},
            )
            groups = MessageGroups.model_validate(result)
            assigned = [seq for group in groups.groups for seq in group.seqs]
            if set(assigned) - expected:
                raise ClusterValidationError(DETAIL_INVALID_MESSAGE_GROUPS)
            return groups
        except (
            APIConnectionError,
            APIStatusError,
            OutputParserException,
            ValidationError,
            ClusterValidationError,
        ) as error:
            # 凭证、权限、参数等请求错误无法通过重复调用解决。
            if isinstance(error, APIStatusError) and not (
                error.status_code in (408, 409, 429) or error.status_code >= 500
            ):
                raise
            if attempt == settings.llm_max_retries:
                raise

            logger.warning(LOG_MESSAGE_CLUSTER_RETRY, attempt + 1)
            delay = min(
                settings.llm_retry_initial_delay_seconds * 2**attempt,
                settings.llm_retry_max_delay_seconds,
            )
            await asyncio.sleep(delay)

    raise AssertionError(DETAIL_INVALID_CLUSTER_RETRIES)
