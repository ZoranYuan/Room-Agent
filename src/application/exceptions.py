"""应用层异常定义."""


class RoomAuthorizationError(Exception):
    """Room Agent 身份认证和访问校验异常基类."""


class InvalidTokenError(RoomAuthorizationError):
    """IM 后端判定 token 无效或已过期."""


class RoomAccessDeniedError(RoomAuthorizationError):
    """用户身份有效，但无权访问指定 Room."""
