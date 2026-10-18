"""gRPC 传输层异常定义."""


class GrpcAuthNotConfiguredError(Exception):
    """IM gRPC 客户端未配置服务地址."""
