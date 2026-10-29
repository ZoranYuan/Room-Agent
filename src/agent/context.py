"""Agent 图的运行时上下文定义."""


from pydantic import BaseModel

from domain.auth_scope import AuthScope


class AgentRuntimeContext(BaseModel):
    """一次图运行内已校验的用户身份和 Room 访问范围."""
    auth_scope: AuthScope