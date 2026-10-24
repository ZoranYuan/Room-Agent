"""模型生成器."""

from typing import Any, Mapping

from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def create_llm_model(
    *,
    model_name: str,
    model_base_url: str,
    model_api_key: str,
    timeout_seconds: float,
    temperature = 0.5,
    max_retries: int = 2,
    extra_body: Mapping[str, Any] | None = None
) -> ChatOpenAI:
    """创建模型入口，当业务层需要创建模型时，可以直接调用."""
    return ChatOpenAI(
        model=model_name,
        api_key=SecretStr(model_api_key) if model_api_key else None,
        base_url=model_base_url,
        timeout=timeout_seconds,
        temperature=temperature,
        max_retries=max_retries,
        extra_body=extra_body,
    )
