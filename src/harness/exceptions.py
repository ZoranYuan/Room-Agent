"""Agent Harness 业务和访问校验异常."""


class BusinessError(Exception):
    """业务异常基类."""


class RoomAuthorizationError(BusinessError):
    """Room Agent 身份认证和访问校验异常基类."""


class ToolError(BusinessError):
    """工具调用异常基类."""


class ToolAccessDeniedError(ToolError):
    """当前用户或运行上下文无权调用该工具."""


class InvalidTokenError(RoomAuthorizationError):
    """IM 后端判定 token 无效或已过期."""


class RoomAccessDeniedError(RoomAuthorizationError):
    """用户身份有效，但无权访问指定 Room."""
