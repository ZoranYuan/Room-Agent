"""定义 Agent model"""
from config.config import settings
from infra.model.llm_model_factory import create_llm_model

summary_model = create_llm_model(
    model_name = settings.agent_model,
        model_base_url = settings.agent_base_url,
        model_api_key = settings.agent_key,
        timeout_seconds =  settings.agent_timeout_seconds,
        temperature = settings.agent_temperature,
        max_retries = 10,
)

chat_model = create_llm_model(
    model_name = settings.agent_model,
        model_base_url = settings.agent_base_url,
        model_api_key = settings.agent_key,
        timeout_seconds =  settings.agent_timeout_seconds,
        temperature = 0.8,
        max_retries = 5,
)
