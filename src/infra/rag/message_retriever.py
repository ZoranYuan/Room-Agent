import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.core.vector_stores import (
    ExactMatchFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from config.config import settings
from domain.time_hint import TimeHint
from infra.model.reranker_model_factory import get_message_reranker
from infra.rag.vector.vector_store import get_vector_index

def epoch_seconds(value: object) -> float | None:
    if not isinstance(value, int | float):
        return None
    # IM 与已有索引通常使用 Unix 秒；兼容毫秒值，统一后再比较。
    return value / 1000 if abs(value) >= 100_000_000_000 else float(value)


def _format_datetime(value: datetime) -> str:
    return value.astimezone(ZoneInfo(settings.rag_timezone)).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def _format_epoch(value: object) -> str:
    timestamp = epoch_seconds(value)
    if timestamp is None:
        return "未知时间"

    return datetime.fromtimestamp(
        timestamp,
        tz=ZoneInfo(settings.rag_timezone),
    ).strftime("%Y-%m-%d %H:%M:%S %Z")


def _format_time_hint(time_hint: TimeHint) -> str:
    if time_hint.mode == "latest":
        return "查询时间要求：在相关记录中寻找时间最近的一次。"

    assert time_hint.start_at is not None
    assert time_hint.end_at is not None
    return (
        "查询时间范围："
        f"{_format_datetime(time_hint.start_at)} 至 "
        f"{_format_datetime(time_hint.end_at)}。"
    )


async def rerank_chat_chunks(
    *,
    query: str,
    candidates: list[NodeWithScore],
    time_hint: TimeHint | None = None,
) -> list[NodeWithScore]:
    if not candidates:
        return []

    model = await get_message_reranker()
    query_text = query
    if time_hint is not None:
        query_text = f"{query}\n{_format_time_hint(time_hint)}"

    # (用户问题，检索到的子块)
    pairs = []

    for candidate in candidates:
        passage = candidate.node.get_content(metadata_mode=MetadataMode.EMBED)

        if time_hint is not None:
            # 将时间转为毫秒
            start_time = _format_epoch(candidate.node.metadata.get("start_time"))
            end_time = _format_epoch(candidate.node.metadata.get("end_time"))
            passage = f"候选时间范围：{start_time} 至 {end_time}\n" f"{passage}"
        pairs.append((query_text, passage))
    scores = await asyncio.to_thread(model.predict, pairs)

    ranked = [
        NodeWithScore(node=candidate.node, score=float(score))
        for candidate, score in zip(candidates, scores)
    ]

    return sorted(ranked, key=lambda item: item.score or 0, reverse=True)


def temporal_order(
    candidates: list[NodeWithScore], time_hint: TimeHint
) -> list[NodeWithScore]:
    retrieval_rank = {
        candidate.node.node_id: rank
        for rank, candidate in enumerate(candidates, start=1)
    }

    if time_hint.mode == "latest":

        def latest_order(candidate: NodeWithScore) -> tuple[bool, float, int]:
            end_time = epoch_seconds(candidate.node.metadata.get("end_time"))
            return (
                end_time is None,
                # 最新的时间放在后面， sorted 默认是升序排列，大的在前面
                -(end_time if end_time is not None else 0.0),
                retrieval_rank[candidate.node.node_id],
            )

        return sorted(
            candidates,
            key=latest_order,
        )

    assert time_hint.start_at is not None and time_hint.end_at is not None
    query_start = time_hint.start_at.timestamp()
    query_end = time_hint.end_at.timestamp()

    def range_order(candidate: NodeWithScore) -> tuple[float, float, int]:
        # 兼容成毫秒之后再去对比
        chunk_start = epoch_seconds(candidate.node.metadata.get("start_time"))
        chunk_end = epoch_seconds(candidate.node.metadata.get("end_time"))
        rank = retrieval_rank[candidate.node.node_id]
        if chunk_start is None or chunk_end is None:
            return (1.0, float("inf"), rank)

        if query_start == query_end:
            if chunk_start <= query_start <= chunk_end:
                return (0.0, 0.0, rank)

            gap = (
                query_start - chunk_end
                if chunk_end < query_start
                else chunk_start - query_start
            )
            return (1.0, gap, rank)

        # 计算重叠区间 >= 0
        overlap = max(
            0.0,
            min(chunk_end, query_end) - max(chunk_start, query_start),
        )

        query_width = query_end - query_start
        overlap_ratio = overlap / query_width

        # 计算时间间隔 gap
        if chunk_end < query_start:
            gap = query_start - chunk_end
        elif chunk_start > query_end:
            gap = chunk_start - query_end
        else:
            gap = 0.0

        # 有重叠的块优先；没有重叠时选离目标时间段最近的块。
        return (
            0.0 if overlap > 0 else 1.0,
            -overlap_ratio if overlap > 0 else gap,
            rank,
        )

    return sorted(candidates, key=range_order)


def combine_rerank_and_temporal_ranks(
    reranked_candidates: list[NodeWithScore],
    time_hint: TimeHint,
) -> list[NodeWithScore]:
    """在精排后的完整候选池中融合语义排名与时间排名。"""
    temporal_candidates = temporal_order(reranked_candidates, time_hint)
    rerank_rank = {
        candidate.node.node_id: rank
        for rank, candidate in enumerate(reranked_candidates, start=1)
    }
    temporal_rank = {
        candidate.node.node_id: rank
        for rank, candidate in enumerate(temporal_candidates, start=1)
    }

    combined = []
    for candidate in reranked_candidates:
        node_id = candidate.node.node_id
        score = 1 / (settings.rag_rrf_k + rerank_rank[node_id])
        score += settings.rag_temporal_rank_weight / (
            settings.rag_rrf_k + temporal_rank[node_id]
        )
        combined.append(NodeWithScore(node=candidate.node, score=score))

    return sorted(combined, key=lambda item: item.score or 0, reverse=True)


async def retrieve_message_chunks(
    *,
    query: str,
    room_id: str,
    time_hint: TimeHint | None = None,
    topk: int = settings.rag_candidate_top_k,
) -> list[NodeWithScore]:
    index = await get_vector_index()

    retriever = index.as_retriever(
        similarity_top_k=topk,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        filters=MetadataFilters(
            filters=[
                ExactMatchFilter(key="room_id", value=room_id),
            ]
        ),
    )

    candidates = await retriever.aretrieve(query)
    if not candidates:
        return []

    # RRF 候选池全部进入 Cross-Encoder，时间因素不提前淘汰语义候选。
    rerank_candidates = candidates[: settings.rag_rerank_input_top_n]
    reranked_candidates = await rerank_chat_chunks(
        query=query,
        candidates=rerank_candidates,
        time_hint=time_hint,
    )

    if time_hint is not None:
        reranked_candidates = combine_rerank_and_temporal_ranks(
            reranked_candidates,
            time_hint,
        )

    return reranked_candidates[: settings.rag_final_top_n]
