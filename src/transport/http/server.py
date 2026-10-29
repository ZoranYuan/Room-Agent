"""读取配置并启动 Uvicorn HTTP 服务。"""

import uvicorn

from config.config import settings


def main() -> None:
    """创建服务器并开始监听，应用生命周期由 Uvicorn 管理。"""
    config = uvicorn.Config(
        app="transport.http.app:app",
        host=settings.server_host,
        port=settings.server_port,
        # 请求之间的空闲等待时间，不限制 SSE 响应持续时间。
        timeout_keep_alive=settings.server_keep_alive_seconds,
        timeout_graceful_shutdown=settings.server_shutdown_timeout_seconds,
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
