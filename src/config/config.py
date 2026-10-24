"""config 配置文件."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=False)


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name)
    return value if value else (default if default is not None else "")


class _Settings(BaseModel):
    """系统运行时配置."""

    agent_key: str = _env("AGENT_KEY")
    agent_base_url: str = _env("AGENT_BASE_URL")
    agent_model: str = _env("AGENT_MODEL", "deepseek-chat")
    agent_temperature: float = float(_env("AGENT_TEMPERATURE", "0.0"))
    agent_recursion_limit: int = int(_env("AGENT_RECURSION_LIMIT", "14"))
    agent_timeout_seconds: float = float(_env("AGENT_TIMEOUT_SECONDS", "60"))
    app_env: str = _env("APP_ENV", "development")
    grpc_url: str = _env("IM_GRPC_URL", "")
    grpc_time_out: int = int(_env("IM_GRPC_TIMEOUT_SECONDS", "5"))

settings = _Settings()
