"""按需创建异步连接池和 Session 工厂。"""

import logging
from time import perf_counter

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config.config import settings

from constants.exception import (
    DETAIL_MYSQL_NOT_CONFIGURED,
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
logger = logging.getLogger(__name__)


def _register_slow_query_logging(engine: AsyncEngine) -> None:
    """监听 SQL 执行耗时，不记录绑定参数。"""
    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def before_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info["query_started_at"] = perf_counter()

    @event.listens_for(engine.sync_engine, "after_cursor_execute")
    def after_execute(conn, cursor, statement, parameters, context, executemany):
        started_at = conn.info.pop("query_started_at")
        elapsed_ms = (perf_counter() - started_at) * 1000
        if elapsed_ms >= settings.mysql_slow_threshold_milliseconds:
            logger.warning("慢 SQL，执行耗时 %.2f ms：%s", elapsed_ms, statement)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """获取 Session 工厂，每个独立任务应创建自己的 Session。"""
    global _engine, _session_factory

    if _session_factory is None:
        if not settings.mysql_url:
            raise RuntimeError(DETAIL_MYSQL_NOT_CONFIGURED)
        _engine = create_async_engine(
            settings.mysql_url,
            pool_pre_ping=True,
            pool_size=settings.mysql_max_idle_conns,
            max_overflow=settings.mysql_max_open_conns - settings.mysql_max_idle_conns,
            pool_recycle=settings.mysql_conn_max_lifetime_seconds,
            hide_parameters=True,
        )
        _register_slow_query_logging(_engine)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    return _session_factory


async def close_mysql() -> None:
    """释放连接池，允许后续重新初始化。"""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
