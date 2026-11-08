# 定义可重试的模型错误
import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage
from openai import APIConnectionError, APIStatusError

from agent.context import AgentRuntimeContext
from config.config import settings
from harness.exceptions import ModelExecutionError

logger = logging.getLogger(__name__)


def is_retryable_model_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, TimeoutError)):
        return True

    if isinstance(error, APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500

    return False


class AgentRetryMiddleware(AgentMiddleware[Any, AgentRuntimeContext, Any]):
    """有限重试可恢复的模型调用异常."""

    async def awrap_model_call(
        self,
        request: ModelRequest[AgentRuntimeContext],
        handler: Callable[
            [ModelRequest[AgentRuntimeContext]], Awaitable[ModelResponse[Any]]
        ],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        max_retries = settings.llm_max_retries

        for attempt in range(max_retries + 1):
            try:
                return await handler(request)
            except Exception as error:
                if is_retryable_model_error(error) and attempt < max_retries:
                    # 对于可重试的错误，指数退避并增加抖动，避免请求同时重试。
                    delay = min(
                        settings.llm_retry_initial_delay_seconds * (2**attempt),
                        settings.llm_retry_max_delay_seconds,
                    )
                    delay = min(
                        settings.llm_retry_max_delay_seconds,
                        delay * random.uniform(0.75, 1.25),
                    )

                    logger.warning(
                        "模型调用失败，将进行第 %s/%s 次重试，异常类型：%s",
                        attempt + 1,
                        max_retries,
                        type(error).__name__,
                    )
                    await asyncio.sleep(delay)
                    continue

                raise ModelExecutionError from error

        raise ModelExecutionError("模型重试配置无效，调用未执行")
