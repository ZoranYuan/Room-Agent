"""房间消息父块的 ORM 模型，不保存消息正文。"""

from sqlalchemy import BigInteger, CheckConstraint, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from infra.mysql.base import Base


class MessageChunk(Base):
    """通过会话和 seq 范围定位 IM 中的父块消息。"""

    __tablename__ = "message_chunks"
    __table_args__ = (
        Index("ix_parent_room_seq", "room_id", "seq_start", "seq_end"),
        CheckConstraint("seq_start <= seq_end", name="ck_parent_seq_range"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    room_id: Mapped[str] = mapped_column(String(64))
    conversation_id: Mapped[str] = mapped_column(String(64))
    seq_start: Mapped[int] = mapped_column(BigInteger)
    seq_end: Mapped[int] = mapped_column(BigInteger)
