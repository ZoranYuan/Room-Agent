"""HTTP 传输层异常定义."""


class MissingAuthTokenError(Exception):
    """请求未携带身份令牌 Cookie."""
