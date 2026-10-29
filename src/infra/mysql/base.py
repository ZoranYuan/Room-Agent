"""SQLAlchemy 表模型基类。"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Agent 数据库表共用的声明式基类。"""
