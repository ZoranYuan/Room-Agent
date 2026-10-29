"""消息父块的保存和查询，事务由调用方管理。"""

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from constants.exception import (
    DETAIL_CHUNK_ROOM_MISMATCH,
    DETAIL_INVALID_CHUNK_RANGE,
)
from infra.mysql.models.message_chunk import MessageChunk


class MessageChunkRepository:
    """使用当前任务的数据库 Session 操作父块。"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, *, room_id: str, chunks: Sequence[MessageChunk]) -> None:
        """按主键新增或更新父块，调用方负责提交事务。"""
        if not chunks:
            return
        for chunk in chunks:
            if chunk.room_id != room_id:
                raise ValueError(DETAIL_CHUNK_ROOM_MISMATCH)
            if chunk.seq_start > chunk.seq_end:
                raise ValueError(DETAIL_INVALID_CHUNK_RANGE)

        statement = insert(MessageChunk).values(
            [
                {
                    "id": chunk.id,
                    "room_id": chunk.room_id,
                    "conversation_id": chunk.conversation_id,
                    "seq_start": chunk.seq_start,
                    "seq_end": chunk.seq_end,
                }
                for chunk in chunks
            ]
        )
        await self.session.execute(
            statement.on_duplicate_key_update(
                conversation_id=statement.inserted.conversation_id,
                seq_start=statement.inserted.seq_start,
                seq_end=statement.inserted.seq_end,
            )
        )

    async def get_parent_chunk_by_id(
        self, *, room_id: str, parent_id: str
    ) -> MessageChunk | None:
        """仅在当前房间中查询父块。"""
        result = await self.session.execute(
            select(MessageChunk).where(
                MessageChunk.id == parent_id,
                MessageChunk.room_id == room_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_parent_chunks_by_ids(
        self,
        *,
        room_id: str,
        parent_ids: Sequence[str],
    ) -> list[MessageChunk]:
        """批量查询当前房间的父块，缺失记录不返回。"""
        if not parent_ids:
            return []

        result = await self.session.execute(
            select(MessageChunk).where(
                MessageChunk.room_id == room_id,
                MessageChunk.id.in_(parent_ids),
            )
        )
        return list(result.scalars().all())

    async def find_chunk_by_range(
        self, *, room_id: str, seq_start: int, seq_end: int
    ) -> MessageChunk | None:
        """查找当前房间中范围完全相同的父块。"""
        result = await self.session.execute(
            select(MessageChunk).where(
                MessageChunk.room_id == room_id,
                MessageChunk.seq_start == seq_start,
                MessageChunk.seq_end == seq_end,
            )
        )
        return result.scalar_one_or_none()

    async def get_room_latest_seq(
        self,
        *,
        room_id: str,
    ) -> int:
        res = await self.session.execute(
            select(func.max(MessageChunk.seq_end)).where(
                MessageChunk.room_id == room_id
            )
        )

        return res.scalar_one_or_none() or 0
