import re
from threading import Lock
from typing import Sequence

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
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
    """保留人物、时间和正文，作为 Embedding 与 BM25 的输入内容。"""
    sender = message.sender_name or message.sender_id
    return (
        f"[seq={message.seq}, send_time={message.send_time}] "
        f"[{sender} ({message.sender_id})]: "
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
            or size + length > settings.rag_parent_max_chars
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

        metadata = {
            "parent_id": parent.id,
            "room_id": room_id,
        }

        message_group = await cluster_messages(batch)
        seq_ms = {str(message.seq): message for message in batch}
        chunks: list[str] = []

        for group in message_group.groups:
            members = sorted(
                [seq_ms[seq] for seq in group.seqs],
                key=lambda item: item.seq,
            )

            # 对 group 中的每条消息进行切分，得到符合单挑消息标准的 chunk
            group_records: list[str] = []
            topic_prefix = f"主题：{group.topic}\n"

            for member in members:
                header = (
                    f"[seq={member.seq}, send_time={member.send_time}] "
                    f"[{member.sender_name} ({member.sender_id})]"
                )

                message_splitter = SentenceSplitter(
                    chunk_size=settings.rag_message_max_chars - len(header),
                    chunk_overlap=10,
                    tokenizer=list,
                    chunking_tokenizer_fn=lambda text: re.findall(
                        r"[^。！？\n]+[。！？\n]*|[。！？\n]+",
                        text,
                    ),
                )
                body_parts = message_splitter.split_text(member.content)

                group_records.extend(f"{header}\n{part}" for part in body_parts)

            current_records: list[str] = []
            current_size = len(topic_prefix)

            for record in group_records:
                # 计算是否需要换行符
                separator_size = 1 if current_records else 0
                record_size = separator_size + len(record)

                if (
                    current_records
                    and current_size + record_size > settings.rag_chunk_max_chars
                ):
                    chunks.append(topic_prefix + "\n".join(current_records))
                    current_records = []

                    # 重新计算积累的长度
                    current_size = len(topic_prefix)
                    record_size = len(record)

                current_records.append(record)
                current_size += record_size

            if current_records:
                chunks.append(topic_prefix + "\n".join(current_records))
        children.extend(
            [
                TextNode(
                    id_=_next_id(),
                    text=chunk,
                    metadata=metadata,
                    excluded_embed_metadata_keys=list(metadata),
                    excluded_llm_metadata_keys=list(metadata),
                )
                for chunk in chunks
            ]
        )
    return parents, children


async def index_room_messages(*, room_id: str, messages: Sequence[ChatMessage]) -> int:
    if not messages:
        return 0

    parents, children = await build_message_chunks(room_id=room_id, messages=messages)

    session_factory = get_session_factory()
    # 即使整批消息都没有可索引的主题，也要保存父块以推进同步位置。
    async with session_factory.begin() as session:
        await MessageChunkRepository(session).save(
            room_id=room_id,
            chunks=parents,
        )

    if children:
        # 模型首次可能需要下载到本地，所以这里异步执行
        index = await get_vector_index()
        await index.ainsert_nodes(children)

    return len(children)
