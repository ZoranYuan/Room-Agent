import importlib
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeRelationship, TextNode

from config.config import settings
from domain.chat_message import ChatMessage
from harness.llm.cluster_messages import MessageGroup, MessageGroups
from infra.rag import message_index as rag


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def index_dependencies(monkeypatch):
    # 模型调用和数据库使用替身，保留真实的切分、节点构建与 Embedding 管道。
    factory = MagicMock()
    repository = Mock()
    repository.find_chunk_by_range = AsyncMock(return_value=None)
    repository.save = AsyncMock()
    monkeypatch.setattr(rag, "get_session_factory", lambda: factory)
    monkeypatch.setattr(rag, "MessageChunkRepository", lambda session: repository)

    async def cluster(messages):
        return MessageGroups(groups=[MessageGroup(
            topic="聊天", seqs=[str(item.seq) for item in messages]
        )])

    monkeypatch.setattr(rag, "cluster_messages", cluster)


def message(seq, *, conversation="conv", content=None, status=0):
    return ChatMessage(
        message_id=f"{conversation}-{seq}",
        conversation_id=conversation,
        sender_id=f"user-{seq % 2}",
        sender_name="小明" if seq % 2 else "小红",
        seq=seq,
        content=content if content is not None else f"消息{seq}",
        status=status,
        send_time=1000 + seq,
    )


@pytest.mark.anyio
async def test_semantic_groups_keep_parent_scope_and_original_messages(monkeypatch):
    async def cluster(messages):
        assert [item.seq for item in messages] == [1, 2, 3]
        return MessageGroups(groups=[
            MessageGroup(topic="电影", seqs=["3", "1"]),
            MessageGroup(topic="天气", seqs=["2"]),
        ])

    monkeypatch.setattr(rag, "cluster_messages", cluster)
    parents, children = await rag.build_message_chunks(
        room_id="room", messages=[message(3), message(1), message(2)]
    )
    assert len(parents) == 1 and len(children) == 2
    assert parents[0].seq_end == 3
    movie = children[0]
    assert movie.metadata["seqs"] == ["1", "3"]
    assert movie.metadata["room_id"] == "room"
    assert movie.metadata["participant_ids"] == ["user-1"]
    assert movie.metadata["start_time"] == 1001
    assert "user-1" in movie.text and "消息1" in movie.text
    assert movie.relationships[NodeRelationship.PARENT].node_id == parents[0].id
    assert "parent_seq_start" not in movie.metadata
    assert "parent_seq_end" not in movie.metadata


@pytest.mark.anyio
async def test_parent_count_splits_current_room_in_seq_order(monkeypatch):
    monkeypatch.setattr(settings, "rag_parent_message_count", 2)

    async def cluster(messages):
        return MessageGroups(groups=[MessageGroup(
            topic="聊天", seqs=[str(item.seq) for item in messages]
        )])

    monkeypatch.setattr(rag, "cluster_messages", cluster)
    parents, _ = await rag.build_message_chunks(
        room_id="room", messages=[message(3), message(1), message(2)]
    )
    assert len(parents) == 2
    assert [(p.seq_start, p.seq_end) for p in parents] == [(1, 2), (3, 3)]


@pytest.mark.anyio
async def test_long_single_message_is_split_after_clustering():
    parents, children = await rag.build_message_chunks(
        room_id="room", messages=[message(1, content="你好。" * 3000)]
    )
    assert len(parents) == 1 and len(children) > 1
    assert all(len(node.text) <= settings.rag_chunk_max_chars for node in children)
    assert all(node.metadata["seqs"] == ["1"] for node in children)


@pytest.mark.anyio
async def test_real_llamaindex_embedding_pipeline_without_network(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dimension", 4)
    monkeypatch.setattr(rag, "get_embedding_model", lambda: MockEmbedding(embed_dim=4))
    written = []
    monkeypatch.setattr(rag, "_write_nodes", lambda children: written.extend(children))
    assert await rag.index_room_messages(room_id="room", messages=[message(1)]) == 1
    assert isinstance(written[0], TextNode)
    assert written[0].embedding == [0.5] * 4
    first_id = written[0].node_id
    await rag.index_room_messages(room_id="room", messages=[message(1)])
    assert written[1].node_id != first_id


@pytest.mark.anyio
async def test_room_is_part_of_node_identity():
    _, left = await rag.build_message_chunks(room_id="left", messages=[message(1)])
    _, right = await rag.build_message_chunks(room_id="right", messages=[message(1)])
    assert left[0].node_id != right[0].node_id


@pytest.mark.anyio
async def test_empty_sync_skips_models_and_storage(monkeypatch):
    def fail():
        raise AssertionError("空消息不应加载模型")

    monkeypatch.setattr(rag, "get_embedding_model", fail)
    assert await rag.index_room_messages(room_id="room", messages=[]) == 0


@pytest.mark.anyio
async def test_milvus_rows_contain_vectors_and_locators_not_original_text(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dimension", 4)
    parents, children = await rag.build_message_chunks(room_id="room", messages=[message(1)])
    children[0].embedding = [0.5] * 4
    rows, closed = [], []

    class Client:
        def __init__(self, **kwargs):
            pass

        def has_collection(self, **kwargs):
            return False

        def create_collection(self, **kwargs):
            assert kwargs["primary_field_name"] == "id"
            assert kwargs["vector_field_name"] == "embedding"
            assert kwargs["dimension"] == 4
            assert kwargs["enable_dynamic_field"]

        def upsert(self, **kwargs):
            rows.extend(kwargs["data"])

        def close(self):
            closed.append(True)

    monkeypatch.setattr(rag, "MilvusClient", Client)
    rag._write_nodes(children)
    assert rows[0]["room_id"] == "room" and rows[0]["parent_id"] == parents[0].id
    assert rows[0]["embedding"] == [0.5] * 4
    assert "text" not in rows[0] and "content" not in rows[0]
    assert "parent_seq_start" not in rows[0] and "parent_seq_end" not in rows[0]
    assert rows[0]["seqs"] == ["1"] and closed == [True]



@pytest.mark.anyio
@pytest.mark.parametrize("fail_stream", [False, True])
async def test_harness_indexes_only_after_complete_stream(monkeypatch, fail_stream):
    from transport.grpc.proto import room_agent_pb2 as pb

    harness = importlib.import_module("harness.grpc.sync_room_messages")
    calls = []

    class Client:
        async def sync_room_messages(self, **kwargs):
            yield pb.SyncRoomMessagesResponse(messages=[pb.ChatMessage(
                message_id="conv-1", conversation_id="conv", seq=1, content="你好"
            )])
            assert not calls
            if fail_stream:
                raise RuntimeError("流中断")
            yield pb.SyncRoomMessagesResponse(messages=[pb.ChatMessage(
                message_id="conv-2", conversation_id="conv", seq=2, content="你好呀"
            )])

    async def index(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(harness, "get_grpc_client", Client)
    monkeypatch.setattr(harness, "index_room_messages", index)
    from pydantic import SecretStr

    from domain.auth_scope import AuthScope

    scope = AuthScope(room_id="room", token=SecretStr("test"))
    if fail_stream:
        with pytest.raises(RuntimeError, match="流中断"):
            await harness.sync_room_messages(auth_scope=scope, after_seq=0)
        assert not calls
    else:
        messages = await harness.sync_room_messages(auth_scope=scope, after_seq=0)
        assert [item.seq for item in messages] == [1, 2]
        assert calls == [{"room_id": "room", "messages": messages}]
