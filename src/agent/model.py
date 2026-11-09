"""定义 Agent model"""

from agent.prompts.cluster import CLUSTER_PROMPT
from agent.schemas.cluster import MessageGroups
from config.config import settings
from infra.model.llm_model_factory import create_llm_model

summary_model = create_llm_model(
    model_name=settings.agent_model,
    model_base_url=settings.agent_base_url,
    model_api_key=settings.agent_key,
    timeout_seconds=settings.agent_timeout_seconds,
    temperature=settings.agent_temperature,
    max_retries=0,
    extra_body={
        "thinking": {"type": "disabled"},
        "max_tokens": settings.summary_max_output_tokens,
    },
)

chat_model = create_llm_model(
    model_name=settings.agent_model,
    model_base_url=settings.agent_base_url,
    model_api_key=settings.agent_key,
    timeout_seconds=settings.agent_timeout_seconds,
    temperature=0.3,
    max_retries=0,
    extra_body={
        "thinking": {"type": "disabled"},
        "max_tokens": settings.chat_max_output_tokens,
    },
)

memory_summary_model = create_llm_model(
    model_name=settings.agent_model,
    model_base_url=settings.agent_base_url,
    model_api_key=settings.agent_key,
    timeout_seconds=settings.agent_timeout_seconds,
    temperature=0.0,
    extra_body={
        "thinking": {"type": "disabled"},
        "max_tokens": settings.memory_summary_max_tokens,
    },
)


topic_cluster_model = create_llm_model(
    model_name=settings.agent_model,
    model_base_url=settings.agent_base_url,
    model_api_key=settings.agent_key,
    timeout_seconds=settings.agent_timeout_seconds,
    temperature=0.0,
    # 聚类 Harness 负责重试，避免和模型客户端重试叠加。
    max_retries=0,
    extra_body={
        "thinking": {"type": "disabled"},
        "max_tokens": settings.cluster_max_output_tokens,
    },
)

topic_cluster_chain = CLUSTER_PROMPT | topic_cluster_model.with_structured_output(
    MessageGroups,
    method="function_calling",
)
