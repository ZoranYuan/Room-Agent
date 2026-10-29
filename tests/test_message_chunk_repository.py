from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import AsyncSession

from infra.mysql.message_chunk_repository import MessageChunkRepository
from infra.mysql.models.message_chunk import MessageChunk


@pytest.fixture
def anyio_backend():
    return "asyncio"


def chunk(**overrides):
    fields = dict(id="p1", room_id="room", conversation_id="conv", seq_start=1, seq_end=50)
    fields.update(overrides)
    return MessageChunk(**fields)


@pytest.mark.anyio
async def test_save_uses_mysql_upsert_without_committing_caller_transaction():
    session = AsyncMock(spec=AsyncSession)
    await MessageChunkRepository(session).save(room_id="room", chunks=[chunk()])
    statement = session.execute.call_args_list[-1].args[0]
    compiled = statement.compile(dialect=mysql.dialect())
    assert "ON DUPLICATE KEY UPDATE" in str(compiled)
    assert compiled.params["room_id_m0"] == "room"
    assert set(compiled.params) == {
        "id_m0", "room_id_m0", "conversation_id_m0", "seq_start_m0", "seq_end_m0",
    }
    session.commit.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("fields", [{"room_id": "other"}, {"seq_start": 100}])
async def test_invalid_batch_is_rejected_before_database_access(fields):
    session = AsyncMock(spec=AsyncSession)
    with pytest.raises(ValueError):
        await MessageChunkRepository(session).save(room_id="room", chunks=[chunk(), chunk(**fields)])
    session.execute.assert_not_awaited()


@pytest.mark.anyio
async def test_queries_always_include_room_scope():
    session = AsyncMock(spec=AsyncSession)
    result = Mock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result
    repository = MessageChunkRepository(session)
    assert await repository.get_parent_chunk_by_id(room_id="room", parent_id="p1") is None
    assert await repository.find_chunk_by_range(room_id="room", seq_start=1, seq_end=50) is None
    for call in session.execute.call_args_list:
        compiled = call.args[0].compile(dialect=mysql.dialect())
        assert compiled.params["room_id_1"] == "room"
        assert "message_chunks.room_id =" in str(compiled)
