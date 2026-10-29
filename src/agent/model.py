"""定义 Agent model"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from config.config import settings
from infra.model.llm_model_factory import create_llm_model

summary_model = create_llm_model(
    model_name=settings.agent_model,
    model_base_url=settings.agent_base_url,
    model_api_key=settings.agent_key,
    timeout_seconds=settings.agent_timeout_seconds,
    temperature=settings.agent_temperature,
    max_retries=10,
    extra_body={"thinking": {"type": "disabled"}},
)

chat_model = create_llm_model(
    model_name=settings.agent_model,
    model_base_url=settings.agent_base_url,
    model_api_key=settings.agent_key,
    timeout_seconds=settings.agent_timeout_seconds,
    temperature=0.3,
    max_retries=5,
    extra_body={"thinking": {"type": "disabled"}},
)

memory_summary_model = create_llm_model(
    model_name=settings.agent_model,
    model_base_url=settings.agent_base_url,
    model_api_key=settings.agent_key,
    timeout_seconds=settings.agent_timeout_seconds,
    temperature=0.0,
    max_retries=2,
    extra_body={"thinking": {"type": "disabled"}},
).bind(max_tokens=settings.memory_summary_max_tokens)


@lru_cache(maxsize=1)
def get_cluster_model() -> ChatOpenAI:
    """复用聚类模型，重试由 Harness 统一管理。"""
    topic_cluster_model = create_llm_model(
        model_name=settings.agent_model,
        model_base_url=settings.agent_base_url,
        model_api_key=settings.agent_key,
        timeout_seconds=settings.agent_timeout_seconds,
        temperature=0.0,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )

    return topic_cluster_model
