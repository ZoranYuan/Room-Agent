import re
from threading import Lock
from typing import Sequence

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from snowflake import SnowflakeGenerator

from config.config import settings
from domain.chat_message import ChatMessage
from harness.llm.cluster_messages import cluster_messages
from infra.mysql.database import get_session_factory
from infra.mysql.message_chunk_repository import MessageChunkRepository
from infra.mysql.models.message_chunk import MessageChunk
from infra.rag.vector.vector_store import get_vector_index

_id_generator = SnowflakeGenerator(int(settings.snowflake_instance_id))
_id_lock = Lock()


def _next_id() -> str:
    """生成字符串形式的雪花 ID，适配现有数据库字段。"""
    with _id_lock:
        return str(next(_id_generator))


def _message_text(message: ChatMessage) -> str:
    """保留人物、时间和正文，作为 Embedding 的输入内容。"""
    return (
        f"[seq={message.seq}, send_time={message.send_time}] "
        f"[{message.sender_id}]: "
        f"{message.content}"
    )


def _parent_batches(
    messages: Sequence[ChatMessage],
):

    batches: list[list[ChatMessage]] = []
    current: list[ChatMessage] = []
    size = 0

    for message in sorted(messages, key=lambda item: item.seq):
        length = len(_message_text(message))

        if current and (
            len(current) >= settings.rag_parent_message_count
            or size + length > settings.rag_chunk_max_chars
        ):
            batches.append(current)
            current = []
            size = 0

        current.append(message)
        size += length

    # 保留最后一条 current
    if current:
        batches.append(current)

    return batches


async def build_message_chunks(
    *, room_id: str, messages: Sequence[ChatMessage]
) -> tuple[list[MessageChunk], list[TextNode]]:
    parents: list[MessageChunk] = []
    children: list[TextNode] = []

    session_factory = get_session_factory()

    splitter = SentenceSplitter(
        chunk_size=settings.rag_chunk_max_chars,
        chunk_overlap=200,
        tokenizer=list,
        chunking_tokenizer_fn=lambda text: re.findall(
            r"[^。！？\n]+[。！？\n]*|[。！？\n]+",
            text,
        ),
    )

    for batch in _parent_batches(messages=messages):
        # 操作完数据库后立即释放连接
        async with session_factory() as session:
            existing = await MessageChunkRepository(session).find_chunk_by_range(
                room_id=room_id,
                seq_start=batch[0].seq,
                seq_end=batch[-1].seq,
            )

        # 如果当前的父块已经存在了，那么就复用，但是一般来说后端传递的消息不会重复，这里是避免网络问题导致的重复发送
        parent = MessageChunk(
            id=existing.id if existing is not None else _next_id(),
            room_id=room_id,
            conversation_id=batch[0].conversation_id,
            seq_start=batch[0].seq,
            seq_end=batch[-1].seq,
        )

        parents.append(parent)

        message_group = await cluster_messages(batch)
        seqs = {str(message.seq): message for message in batch}

        for group in message_group.groups:
            members = sorted(
                [seqs[seq] for seq in group.seqs], key=lambda item: item.seq
            )

            text = f"主题：{group.topic}\n" + "\n".join(
                _message_text(message) for message in members
            )

            metadata = {
                "parent_id": parent.id,
                "room_id": room_id,
                "conversation_id": parent.conversation_id,
                "topic": group.topic,
                # 保存实际成员的 seq，允许不连续。
                "seqs": [str(message.seq) for message in members],
                "participant_ids": sorted({message.sender_id for message in members}),
                "start_time": min(message.send_time for message in members),
                "end_time": max(message.send_time for message in members),
                "embedding_model_id": settings.embedding_model_id,
            }

            for content in splitter.split_text(text):
                children.append(
                    TextNode(
                        id_=_next_id(),
                        text=content,
                        metadata=metadata,
                        excluded_embed_metadata_keys=list(metadata),
                        excluded_llm_metadata_keys=list(metadata),
                        relationships={
                            NodeRelationship.PARENT: RelatedNodeInfo(
                                node_id=parent.id,
                            )
                        },
                    )
                )

    return parents, children

async def index_room_messages(*, room_id: str, messages: Sequence[ChatMessage]) -> int:
    if not messages:
        return 0

    parents, children = await build_message_chunks(room_id=room_id, messages=messages)

    if children:
        # 模型首次可能需要下载到本地，所以这里异步执行
        index = await get_vector_index()
        session_factory = get_session_factory()
        # save 有两个步骤，检查 创建，需要开启事务
        async with session_factory.begin() as session:
            # 将父块保存到 Mysql 数据库中
            await MessageChunkRepository(session).save(
                room_id=room_id,
                chunks=parents,
            )

        await index.ainsert_nodes(children)

    return len(children)
