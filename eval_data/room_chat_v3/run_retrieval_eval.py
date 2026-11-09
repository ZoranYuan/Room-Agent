"""在隔离的 Milvus Collection 中评测 v3 子块召回和时间排序。"""

# Milvus URI must be normalized before importing clients that read it at import time.
# ruff: noqa: E402

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

from config.config import settings

MILVUS_URI = (
    settings.milvus_uri
    if "://" in settings.milvus_uri
    else f"http://{settings.milvus_uri}"
)
os.environ["MILVUS_URI"] = MILVUS_URI

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from pymilvus import MilvusClient

from agent.model import chat_model
from agent.prompts.chat import CHAT_PROMPT
from domain.chat_message import ChatMessage
from domain.time_hint import TimeHint
from harness.llm.cluster_messages import cluster_messages
from infra.model.embedding_model_factory import get_embedding_model
from infra.rag.message_retriever import (
    rerank_chat_chunks,
    select_rerank_candidates,
    temporal_order,
)

DATA_DIR = Path(__file__).parent
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
CHILD_CHUNKS_PATH = DATA_DIR / "child_chunks.jsonl"
CHILD_CHUNKS_HASH_PATH = DATA_DIR / "child_chunks.corpus.sha256"
RESULT_PATH = DATA_DIR / "retrieval_results.json"
OUTPUT_COLLECTION = f"eval_v3_{uuid4().hex[:12]}"


@tool
def search_messages(query: str, time_hint: TimeHint | None = None) -> str:
    """只用于取得聊天模型可见的搜索参数；评测中不会执行搜索工具。"""
    return ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def to_messages(rows: list[dict[str, Any]]) -> list[ChatMessage]:
    return [
        ChatMessage(
            message_id=row["message_id"],
            conversation_id=row["conversation_id"],
            sender_id=row["sender_id"],
            sender_name=row["sender_name"],
            seq=row["seq"],
            content=row["content"],
            status=row["status"],
            send_time=row["send_time"],
        )
        for row in rows
    ]


def message_text(message: ChatMessage) -> str:
    sender = message.sender_name or message.sender_id
    return (
        f"[seq={message.seq}, send_time={message.send_time}] "
        f"[{sender} ({message.sender_id})]: {message.content}"
    )


def parent_batches(messages: list[ChatMessage]) -> list[list[ChatMessage]]:
    batches: list[list[ChatMessage]] = []
    current: list[ChatMessage] = []
    current_size = 0
    for message in sorted(messages, key=lambda item: item.seq):
        size = len(message_text(message))
        if current and (
            len(current) >= settings.rag_parent_message_count
            or current_size + size > settings.rag_parent_max_chars
        ):
            batches.append(current)
            current = []
            current_size = 0
        current.append(message)
        current_size += size
    if current:
        batches.append(current)
    return batches


async def build_child_chunks(messages: list[ChatMessage]) -> list[TextNode]:
    children: list[TextNode] = []
    batches = parent_batches(messages)
    for batch_index, batch in enumerate(batches, start=1):
        groups = await cluster_messages(batch)
        message_by_seq = {str(message.seq): message for message in batch}
        parent_id = f"eval-v3-parent-{batch_index:04d}"
        for group_index, group in enumerate(groups.groups, start=1):
            members = [message_by_seq[seq] for seq in group.seqs]
            metadata = {
                "parent_id": parent_id,
                "room_id": "eval-room-v3",
                "topic": group.topic,
                "seqs": group.seqs,
                "start_time": min(message.send_time for message in members),
                "end_time": max(message.send_time for message in members),
            }
            children.append(
                TextNode(
                    id_=f"eval-v3-child-{batch_index:04d}-{group_index:03d}",
                    text=f"主题：{group.topic}\n摘要：{group.summary}",
                    metadata=metadata,
                    excluded_embed_metadata_keys=list(metadata),
                    excluded_llm_metadata_keys=list(metadata),
                )
            )
        if batch_index % 10 == 0 or batch_index == len(batches):
            print(f"clustered parent batches {batch_index}/{len(batches)}", flush=True)
    return children


def load_or_build_children(
    messages: list[ChatMessage], corpus_hash: str
) -> list[TextNode]:
    if (
        CHILD_CHUNKS_PATH.exists()
        and CHILD_CHUNKS_HASH_PATH.exists()
        and CHILD_CHUNKS_HASH_PATH.read_text(encoding="utf-8").strip() == corpus_hash
    ):
        return [
            TextNode(
                id_=row["id"],
                text=row["text"],
                metadata=row["metadata"],
                excluded_embed_metadata_keys=list(row["metadata"]),
                excluded_llm_metadata_keys=list(row["metadata"]),
            )
            for row in read_jsonl(CHILD_CHUNKS_PATH)
        ]
    raise RuntimeError(
        "子块缓存不存在或语料哈希不匹配，请使用 async_build_children 生成"
    )


async def async_build_children(
    messages: list[ChatMessage], corpus_hash: str
) -> list[TextNode]:
    if (
        CHILD_CHUNKS_PATH.exists()
        and CHILD_CHUNKS_HASH_PATH.exists()
        and CHILD_CHUNKS_HASH_PATH.read_text(encoding="utf-8").strip() == corpus_hash
    ):
        return load_or_build_children(messages, corpus_hash)
    children = await build_child_chunks(messages)
    with CHILD_CHUNKS_PATH.open("w", encoding="utf-8") as file:
        for child in children:
            file.write(
                json.dumps(
                    {
                        "id": child.node_id,
                        "text": child.text,
                        "metadata": child.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    CHILD_CHUNKS_HASH_PATH.write_text(corpus_hash + "\n", encoding="utf-8")
    return children


def target_groups(
    query: dict[str, Any],
    message_seq: dict[str, str],
    seq_nodes: dict[str, set[str]],
) -> list[set[str]]:
    groups = []
    for evidence in query["gold_evidence_groups"]:
        nodes: set[str] = set()
        for message_id in evidence["message_ids"]:
            seq = message_seq[message_id]
            nodes.update(seq_nodes.get(seq, set()))
        groups.append(nodes)
    return groups


def matched_rank(target: set[str], ranked_ids: list[str]) -> int | None:
    return next(
        (rank for rank, node_id in enumerate(ranked_ids, start=1) if node_id in target),
        None,
    )


def single_metrics(records: list[dict[str, Any]], result_key: str) -> dict[str, Any]:
    if not records:
        return {"questions": 0, "recall_at_1": None, "mrr_at_3": None}
    hits = 0
    reciprocal_ranks = []
    for record in records:
        targets = record["target_groups"][0]
        ranked = record[result_key]
        rank = matched_rank(targets, ranked[:3])
        hits += int(bool(ranked and ranked[0] in targets))
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    return {
        "questions": len(records),
        "recall_at_1": round(hits / len(records), 4),
        "mrr_at_3": round(statistics.mean(reciprocal_ranks), 4),
        "hits_at_1": hits,
    }


def multi_metrics(
    records: list[dict[str, Any]], result_key: str, *, cutoff: int = 3
) -> dict[str, Any]:
    if not records:
        return {}
    evidence_count = sum(len(record["target_groups"]) for record in records)
    evidence_hits = 0
    complete = 0
    first_hit_rr = []
    evidence_rr = []
    for record in records:
        ranked = record[result_key][:cutoff]
        group_ranks = [matched_rank(group, ranked) for group in record["target_groups"]]
        evidence_hits += sum(rank is not None for rank in group_ranks)
        complete += int(all(rank is not None for rank in group_ranks))
        first_rank = min(
            (rank for rank in group_ranks if rank is not None), default=None
        )
        first_hit_rr.append(0.0 if first_rank is None else 1.0 / first_rank)
        evidence_rr.extend(0.0 if rank is None else 1.0 / rank for rank in group_ranks)
    return {
        "questions": len(records),
        f"evidence_hits_at_{cutoff}": evidence_hits,
        "evidence_total": evidence_count,
        f"evidence_recall_at_{cutoff}": round(evidence_hits / evidence_count, 4),
        f"complete_evidence_at_{cutoff}": round(complete / len(records), 4),
        f"first_hit_mrr_at_{cutoff}": round(statistics.mean(first_hit_rr), 4),
        f"per_evidence_mrr_at_{cutoff}": round(statistics.mean(evidence_rr), 4),
    }


def same_time_hint(expected: TimeHint, actual: TimeHint | None) -> bool:
    if actual is None or expected.mode != actual.mode:
        return False
    if expected.mode == "latest":
        return True
    return expected.start_at == actual.start_at and expected.end_at == actual.end_at


async def model_search_args(
    question: str,
    *,
    model_with_tool: Any,
    system_message: SystemMessage,
) -> dict[str, Any] | None:
    response = await model_with_tool.ainvoke(
        [system_message, HumanMessage(content=question)]
    )
    call = next(
        (item for item in response.tool_calls if item["name"] == "search_messages"),
        None,
    )
    return None if call is None else call["args"]


async def rank_with_reranker(
    *,
    retriever: Any,
    query: str,
    time_hint: TimeHint | None,
) -> tuple[list[NodeWithScore], list[NodeWithScore], list[NodeWithScore]]:
    candidates = await retriever.aretrieve(query)
    if time_hint is None:
        rerank_input = candidates[: settings.rag_rerank_input_top_n]
    else:
        rerank_input = select_rerank_candidates(
            candidates,
            temporal_order(candidates, time_hint),
        )
    reranked = await rerank_chat_chunks(
        query=query,
        candidates=rerank_input,
        top_n=settings.rag_final_top_n,
    )
    return candidates, rerank_input, reranked


async def run_evaluation() -> None:
    corpus_bytes = CORPUS_PATH.read_bytes()
    corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()
    corpus_rows = read_jsonl(CORPUS_PATH)
    queries = read_jsonl(QUERIES_PATH)
    messages = to_messages(corpus_rows)
    print(
        f"dataset messages={len(messages)} queries={len(queries)} "
        f"answerable={sum(query['answerable'] for query in queries)}",
        flush=True,
    )

    children = await async_build_children(messages, corpus_hash)
    message_seq = {str(row["message_id"]): str(row["seq"]) for row in corpus_rows}
    seq_nodes: dict[str, set[str]] = defaultdict(set)
    for child in children:
        for seq in child.metadata["seqs"]:
            seq_nodes[str(seq)].add(child.node_id)
    print(f"child_chunks={len(children)}", flush=True)

    store = None
    try:
        store = MilvusVectorStore(
            uri=MILVUS_URI,
            collection_name=OUTPUT_COLLECTION,
            dim=settings.embedding_dimension,
            embedding_field="embedding",
            similarity_metric="COSINE",
            enable_sparse=True,
            sparse_embedding_function=cast(
                Any,
                BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
            ),
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": settings.rag_rrf_k},
            consistency_level="Strong",
            overwrite=False,
            upsert_mode=True,
            use_async_client=True,
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=store,
            embed_model=await get_embedding_model(),
        )
        await index.ainsert_nodes(children)
        retriever = index.as_retriever(
            similarity_top_k=settings.rag_candidate_top_k,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            filters=MetadataFilters(
                filters=[ExactMatchFilter(key="room_id", value="eval-room-v3")]
            ),
        )
        print(
            f"indexed collection={OUTPUT_COLLECTION} candidate_top_k={settings.rag_candidate_top_k} "
            f"rerank_input={settings.rag_rerank_input_top_n} final_top_n={settings.rag_final_top_n} "
            f"rrf_k={settings.rag_rrf_k}",
            flush=True,
        )

        now = datetime.now(ZoneInfo(settings.rag_timezone)).isoformat()
        prompt_text = CHAT_PROMPT.format_messages()[0].text
        system_message = SystemMessage(
            content=(
                f"{prompt_text}\n本次请求的当前时间为 {now}，"
                f"时区为 {settings.rag_timezone}。"
            )
        )
        model_with_tool = chat_model.bind_tools([search_messages])
        records = []
        answerable_queries = [query for query in queries if query["answerable"]]
        for position, query in enumerate(answerable_queries, start=1):
            groups = target_groups(query, message_seq, seq_nodes)
            record: dict[str, Any] = {
                "qid": query["qid"],
                "category": query["category"],
                "question": query["question"],
                "target_groups": [sorted(group) for group in groups],
            }

            candidates, rerank_input, reranked = await rank_with_reranker(
                retriever=retriever,
                query=query["question"],
                time_hint=None,
            )
            record["rrf_ids"] = [candidate.node.node_id for candidate in candidates]
            record["reranked_ids"] = [candidate.node.node_id for candidate in reranked]
            record["production_ids"] = record["reranked_ids"]

            if query["category"].startswith("temporal:"):
                expected_hint = TimeHint.model_validate(query["time_hint"])
                args = await model_search_args(
                    query["question"],
                    model_with_tool=model_with_tool,
                    system_message=system_message,
                )
                actual_hint = (
                    TimeHint.model_validate(args["time_hint"])
                    if args and args.get("time_hint") is not None
                    else None
                )
                llm_query = (
                    str(args.get("query", query["question"]))
                    if args
                    else query["question"]
                )
                (
                    llm_candidates,
                    llm_rerank_input,
                    llm_reranked,
                ) = await rank_with_reranker(
                    retriever=retriever,
                    query=llm_query,
                    time_hint=actual_hint,
                )
                _, oracle_rerank_input, oracle_reranked = await rank_with_reranker(
                    retriever=retriever,
                    query=query["question"],
                    time_hint=expected_hint,
                )
                record.update(
                    {
                        "expected_time_hint": query["time_hint"],
                        "llm_tool_args": args,
                        "time_hint_exact": same_time_hint(expected_hint, actual_hint),
                        "no_time_ids": record["reranked_ids"],
                        "no_time_rerank_input_ids": [
                            item.node.node_id for item in rerank_input
                        ],
                        "oracle_time_ids": [
                            item.node.node_id for item in oracle_reranked
                        ],
                        "oracle_time_rerank_input_ids": [
                            item.node.node_id for item in oracle_rerank_input
                        ],
                        "llm_time_ids": [item.node.node_id for item in llm_reranked],
                        "llm_time_rerank_input_ids": [
                            item.node.node_id for item in llm_rerank_input
                        ],
                        "llm_time_rrf_ids": [
                            item.node.node_id for item in llm_candidates
                        ],
                    }
                )
                record["production_ids"] = record["llm_time_ids"]

            records.append(record)
            if position % 10 == 0 or position == len(answerable_queries):
                print(f"retrieved {position}/{len(answerable_queries)}", flush=True)

        single_records = [
            record
            for record in records
            if record["category"].startswith("single_block:")
        ]
        temporal_records = [
            record for record in records if record["category"].startswith("temporal:")
        ]
        multi_records = [
            record
            for record in records
            if record["category"].startswith("multi_block:")
        ]
        all_single_records = single_records + temporal_records
        time_accuracy = (
            sum(bool(record["time_hint_exact"]) for record in temporal_records)
            / len(temporal_records)
            if temporal_records
            else 0.0
        )
        temporal_metrics = {}
        for subset in ("range", "latest"):
            selected = [
                record
                for record in temporal_records
                if record["category"].startswith(f"temporal:{subset}:")
            ]
            temporal_metrics[subset] = {
                "before_reranker": {
                    "rrf_top5": single_metrics(selected, "no_time_rerank_input_ids"),
                    "oracle_time_top5": single_metrics(
                        selected, "oracle_time_rerank_input_ids"
                    ),
                    "llm_time_top5": single_metrics(
                        selected, "llm_time_rerank_input_ids"
                    ),
                },
                "after_reranker": {
                    "no_time": single_metrics(selected, "no_time_ids"),
                    "oracle_time": single_metrics(selected, "oracle_time_ids"),
                    "llm_time": single_metrics(selected, "llm_time_ids"),
                },
            }

        result = {
            "dataset": "room_chat_v3",
            "message_count": len(messages),
            "question_count": len(queries),
            "answerable_count": len(answerable_queries),
            "unanswerable_count": sum(not query["answerable"] for query in queries),
            "child_chunk_count": len(children),
            "mapped_evidence_group_count": sum(
                bool(group) for record in records for group in record["target_groups"]
            ),
            "settings": {
                "embedding_model": settings.embedding_model_id,
                "reranker_model": settings.reranker_model_id,
                "candidate_top_k": settings.rag_candidate_top_k,
                "rerank_input_top_n": settings.rag_rerank_input_top_n,
                "final_top_n": settings.rag_final_top_n,
                "rrf_k": settings.rag_rrf_k,
                "temporal_rank_weight": settings.rag_temporal_rank_weight,
                "bm25_analyzer": "chinese",
            },
            "metrics": {
                "single_evidence": {
                    "rrf": single_metrics(single_records, "rrf_ids"),
                    "rrf_qwen_rerank": single_metrics(single_records, "reranked_ids"),
                    "all_single_including_temporal_production": single_metrics(
                        all_single_records, "production_ids"
                    ),
                },
                "multi_evidence": {
                    "rrf_candidate_pool": multi_metrics(
                        multi_records,
                        "rrf_ids",
                        cutoff=settings.rag_candidate_top_k,
                    ),
                    "rrf": multi_metrics(multi_records, "rrf_ids"),
                    "rrf_qwen_rerank": multi_metrics(multi_records, "reranked_ids"),
                },
                "temporal": {
                    "questions": len(temporal_records),
                    "time_hint_exact_accuracy": round(time_accuracy, 4),
                    "time_hint_exact_count": sum(
                        bool(record["time_hint_exact"]) for record in temporal_records
                    ),
                    "subsets": temporal_metrics,
                },
            },
            "queries": records,
        }
        RESULT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2), flush=True)
        print(f"saved {RESULT_PATH}", flush=True)
    finally:
        if store is not None:
            if store.aclient is not None:
                await store.aclient.close()
            await asyncio.to_thread(store.client.close)
        milvus = MilvusClient(uri=MILVUS_URI, timeout=5)
        if milvus.has_collection(OUTPUT_COLLECTION):
            milvus.drop_collection(OUTPUT_COLLECTION)
            print(f"temporary collection dropped={OUTPUT_COLLECTION}", flush=True)
        milvus.close()


if __name__ == "__main__":
    asyncio.run(run_evaluation())
