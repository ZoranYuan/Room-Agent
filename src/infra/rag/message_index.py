from threading import Lock
from typing import Sequence

from llama_index.core.schema import TextNode
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
    """用于父块字符预算，包含人物、时间和原始正文。"""
    sender = message.sender_name or message.sender_id
    return (
        f"[seq={message.seq}, send_time={message.send_time}] "
        f"[{sender} ({message.sender_id})]: "
        f"{message.content}"
    )


def _child_message_text(message: ChatMessage, body_part: str) -> str:
    """子块保留原始发言；消息时间范围单独放在 metadata。"""
    sender = message.sender_name or message.sender_id
    return f"{message.send_time}: {sender}: {body_part}"


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

        message_group = await cluster_messages(batch)
        seq_ms = {str(message.seq): message for message in batch}

        for group in message_group.groups:
            members = sorted(
                [seq_ms[seq] for seq in group.seqs],
                key=lambda message: message.seq,
            )
            metadata = {
                "parent_id": parent.id,
                "room_id": room_id,
                "topic": group.topic,
            }

            topic_prefix = f"主题：{group.topic}\n"

            def append_child(current: list[tuple[ChatMessage, str]]) -> None:
                chunk_metadata = {
                    **metadata,
                    "start_time": min(
                        message.send_time for message, _ in current
                    ),
                    "end_time": max(
                        message.send_time for message, _ in current
                    ),
                }
                content = "\n".join(
                    _child_message_text(message, body_part)
                    for message, body_part in current
                )
                children.append(
                    TextNode(
                        id_=_next_id(),
                        # 每个 children 需要单独拼接上主题
                        text=f"{topic_prefix}{content}",
                        metadata=chunk_metadata,
                        excluded_embed_metadata_keys=list(chunk_metadata),
                        excluded_llm_metadata_keys=list(chunk_metadata),
                    )
                )

            current: list[tuple[ChatMessage, str]] = []
            size = 0
            for message in members:
                body = message.content or ""
                if not body:
                    continue

                for start in range(0, len(body), settings.rag_message_max_chars):
                    body_part = body[start : start + settings.rag_message_max_chars]
                    if (
                        current
                        and size + len(body_part) > settings.rag_chunk_max_chars
                    ):
                        # 装箱为 TextNode 放入到 children 中
                        append_child(current)
                        current = []
                        size = 0

                    current.append((message, body_part))
                    size += len(body_part)

            if current:
                append_child(current)
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
