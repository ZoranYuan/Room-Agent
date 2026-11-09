"""逐阶段检查 v3 聊天检索的证据流失位置。"""

# Normalize the URI before importing PyMilvus/LlamaIndex Milvus clients.
# ruff: noqa: E402

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
from datetime import datetime
from time import perf_counter
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

from langchain_core.messages import SystemMessage
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from pymilvus import MilvusClient

from agent.model import chat_model
from agent.prompts.chat import CHAT_PROMPT
from domain.time_hint import TimeHint
from infra.model.embedding_model_factory import get_embedding_model
from infra.rag.message_retriever import (
    rerank_chat_chunks,
    select_rerank_candidates,
    temporal_order,
)
from run_retrieval_eval import (
    CORPUS_PATH,
    DATA_DIR,
    QUERIES_PATH,
    async_build_children,
    model_search_args,
    read_jsonl,
    same_time_hint,
    search_messages,
    target_groups,
    to_messages,
)

ROOM_ID = "eval-room-v3"
RESULT_PATH = DATA_DIR / "retrieval_stage_results.json"
OUTPUT_COLLECTION = f"eval_stage_v3_{uuid4().hex[:12]}"


def ids(hits: list[NodeWithScore]) -> list[str]:
    return [hit.node.node_id for hit in hits]


def result_hits(result: Any) -> list[NodeWithScore]:
    similarities = result.similarities or []
    return [
        NodeWithScore(
            node=node,
            score=(
                float(similarities[index])
                if index < len(similarities) and similarities[index] is not None
                else None
            ),
        )
        for index, node in enumerate(result.nodes)
    ]


def manual_rrf(
    routes: list[list[NodeWithScore]], *, limit: int, rrf_k: int
) -> list[NodeWithScore]:
    scores: dict[str, float] = {}
    nodes: dict[str, Any] = {}
    first_seen: dict[str, int] = {}
    for route in routes:
        for rank, hit in enumerate(route, start=1):
            node_id = hit.node.node_id
            if node_id not in scores:
                scores[node_id] = 0.0
                nodes[node_id] = hit.node
                first_seen[node_id] = len(first_seen)
            scores[node_id] += 1.0 / (rrf_k + rank)
    ranked_ids = sorted(
        scores,
        key=lambda node_id: (-scores[node_id], first_seen[node_id]),
    )[:limit]
    return [
        NodeWithScore(node=nodes[node_id], score=scores[node_id])
        for node_id in ranked_ids
    ]


async def search_mode(
    *,
    store: MilvusVectorStore,
    query_text: str,
    query_embedding: list[float],
    mode: VectorStoreQueryMode,
    filters: MetadataFilters,
) -> tuple[list[NodeWithScore], float]:
    query = VectorStoreQuery(
        query_embedding=(
            None if mode == VectorStoreQueryMode.SPARSE else query_embedding
        ),
        query_str=query_text,
        similarity_top_k=settings.rag_candidate_top_k,
        filters=filters,
        mode=mode,
    )
    started = perf_counter()
    if mode == VectorStoreQueryMode.HYBRID:
        result = await store.aquery(query)
        hits = result_hits(result)
    else:
        # Mirror the two AnnSearchRequest definitions used by Milvus HYBRID;
        # SPARSE mode's convenience wrapper adds drop_ratio_search=0.2.
        filter_expr, output_fields = store._prepare_before_search(query)
        if mode == VectorStoreQueryMode.DEFAULT:
            data = [query_embedding]
            anns_field = store.embedding_field
            search_params = {
                "metric_type": store.similarity_metric,
                "params": store.search_config,
            }
        elif mode == VectorStoreQueryMode.SPARSE:
            data = [query_text]
            anns_field = store.sparse_embedding_field
            search_params = {"metric_type": "BM25"}
        else:
            raise ValueError(f"Unsupported diagnostic search mode: {mode}")
        raw_results = await store.aclient.search(
            collection_name=store.collection_name,
            data=data,
            filter=filter_expr,
            limit=query.similarity_top_k,
            output_fields=output_fields,
            search_params=search_params,
            anns_field=anns_field,
        )
        nodes, similarities, _ = store._parse_from_milvus_results(raw_results)
        hits = [
            NodeWithScore(node=node, score=float(similarities[index]))
            for index, node in enumerate(nodes)
        ]
    elapsed_ms = (perf_counter() - started) * 1000
    return hits, elapsed_ms


def rank_of(target: list[str], ranked_ids: list[str]) -> int | None:
    return next(
        (rank for rank, node_id in enumerate(ranked_ids, start=1) if node_id in target),
        None,
    )


def single_metrics(
    records: list[dict[str, Any]], result_key: str, *, cutoff: int
) -> dict[str, Any]:
    if not records:
        return {"questions": 0}
    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_ranks = []
    for record in records:
        target = record["target_groups"][0]
        ranked = record[result_key][:cutoff]
        rank = rank_of(target, ranked)
        hit_at_1 += int(bool(ranked and ranked[0] in target))
        hit_at_k += int(rank is not None)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    return {
        "questions": len(records),
        "hit_at_1": hit_at_1,
        "recall_at_1": round(hit_at_1 / len(records), 4),
        f"hit_at_{cutoff}": hit_at_k,
        f"recall_at_{cutoff}": round(hit_at_k / len(records), 4),
        f"mrr_at_{cutoff}": round(statistics.mean(reciprocal_ranks), 4),
    }


def multi_metrics(
    records: list[dict[str, Any]], result_key: str, *, cutoff: int
) -> dict[str, Any]:
    if not records:
        return {"questions": 0}
    evidence_total = sum(len(record["target_groups"]) for record in records)
    evidence_hits = 0
    complete_questions = 0
    first_hit_rr = []
    evidence_rr = []
    for record in records:
        ranked = record[result_key][:cutoff]
        ranks = [rank_of(group, ranked) for group in record["target_groups"]]
        evidence_hits += sum(rank is not None for rank in ranks)
        complete_questions += int(all(rank is not None for rank in ranks))
        first_rank = min((rank for rank in ranks if rank is not None), default=None)
        first_hit_rr.append(0.0 if first_rank is None else 1.0 / first_rank)
        evidence_rr.extend(0.0 if rank is None else 1.0 / rank for rank in ranks)
    return {
        "questions": len(records),
        "evidence_hits": evidence_hits,
        "evidence_total": evidence_total,
        "evidence_recall": round(evidence_hits / evidence_total, 4),
        "complete_evidence_rate": round(complete_questions / len(records), 4),
        "first_hit_mrr": round(statistics.mean(first_hit_rr), 4),
        "per_evidence_mrr": round(statistics.mean(evidence_rr), 4),
    }


def union_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"questions": 0}
    total = sum(len(record["target_groups"]) for record in records)
    hits = 0
    complete = 0
    for record in records:
        union = set(record["route_union_ids"])
        found = [
            any(node_id in union for node_id in group)
            for group in record["target_groups"]
        ]
        hits += sum(found)
        complete += int(all(found))
    return {
        "questions": len(records),
        "evidence_hits": hits,
        "evidence_total": total,
        "evidence_recall": round(hits / total, 4),
        "complete_evidence_rate": round(complete / len(records), 4),
    }


def stage_transition(
    records: list[dict[str, Any]], source_key: str, destination_key: str
) -> dict[str, int]:
    source_hits = destination_hits = lost = recovered = 0
    for record in records:
        source = set(record[source_key])
        destination = set(record[destination_key])
        for group in record["target_groups"]:
            in_source = any(node_id in source for node_id in group)
            in_destination = any(node_id in destination for node_id in group)
            source_hits += int(in_source)
            destination_hits += int(in_destination)
            lost += int(in_source and not in_destination)
            recovered += int(not in_source and in_destination)
    return {
        "source_evidence_hits": source_hits,
        "destination_evidence_hits": destination_hits,
        "lost_evidence_groups": lost,
        "newly_found_evidence_groups": recovered,
    }


def latency_summary(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = sorted(
        float(record["stage_latency_ms"][key])
        for record in records
        if record["stage_latency_ms"].get(key) is not None
    )
    if not values:
        return {"mean_ms": None, "p95_ms": None}
    return {
        "mean_ms": round(statistics.mean(values), 2),
        "p95_ms": round(values[max(0, int(len(values) * 0.95) - 1)], 2),
    }


async def run_stage_evaluation() -> None:
    corpus_hash = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    corpus_rows = read_jsonl(CORPUS_PATH)
    queries = read_jsonl(QUERIES_PATH)
    messages = to_messages(corpus_rows)
    children = await async_build_children(messages, corpus_hash)
    message_seq = {str(row["message_id"]): str(row["seq"]) for row in corpus_rows}
    seq_nodes: dict[str, set[str]] = {}
    for child in children:
        for seq in child.metadata["seqs"]:
            seq_nodes.setdefault(str(seq), set()).add(child.node_id)

    store: MilvusVectorStore | None = None
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
        embed_model = await get_embedding_model()
        index = VectorStoreIndex.from_vector_store(
            vector_store=store,
            embed_model=embed_model,
        )
        index_started = perf_counter()
        await index.ainsert_nodes(children)
        index_write_ms = (perf_counter() - index_started) * 1000

        # Exclude one-time model loading from per-question reranker timings.
        warmup_started = perf_counter()
        await rerank_chat_chunks(
            query="评测预热",
            candidates=[NodeWithScore(node=children[0], score=0.0)],
            top_n=1,
        )
        reranker_warmup_ms = (perf_counter() - warmup_started) * 1000

        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="room_id", value=ROOM_ID)]
        )
        now = datetime.now(ZoneInfo(settings.rag_timezone)).isoformat()
        system_message = SystemMessage(
            content=(
                f"{CHAT_PROMPT.format_messages()[0].text}\n"
                f"本次请求的当前时间为 {now}，时区为 {settings.rag_timezone}。"
            )
        )
        model_with_tool = chat_model.bind_tools([search_messages])
        answerable = [row for row in queries if row["answerable"]]
        records: list[dict[str, Any]] = []

        for position, row in enumerate(answerable, start=1):
            expected_hint = (
                TimeHint.model_validate(row["time_hint"])
                if row["category"].startswith("temporal:")
                else None
            )
            time_intent_ms = None
            tool_args = None
            query_text = row["question"]
            actual_hint = None
            if expected_hint is not None:
                intent_started = perf_counter()
                tool_args = await model_search_args(
                    row["question"],
                    model_with_tool=model_with_tool,
                    system_message=system_message,
                )
                time_intent_ms = (perf_counter() - intent_started) * 1000
                if tool_args is not None:
                    query_text = str(tool_args.get("query", query_text))
                    if tool_args.get("time_hint") is not None:
                        actual_hint = TimeHint.model_validate(tool_args["time_hint"])

            embedding_started = perf_counter()
            query_embedding = await embed_model.aget_query_embedding(query_text)
            embedding_ms = (perf_counter() - embedding_started) * 1000
            dense, dense_ms = await search_mode(
                store=store,
                query_text=query_text,
                query_embedding=query_embedding,
                mode=VectorStoreQueryMode.DEFAULT,
                filters=filters,
            )
            bm25, bm25_ms = await search_mode(
                store=store,
                query_text=query_text,
                query_embedding=query_embedding,
                mode=VectorStoreQueryMode.SPARSE,
                filters=filters,
            )
            hybrid, hybrid_ms = await search_mode(
                store=store,
                query_text=query_text,
                query_embedding=query_embedding,
                mode=VectorStoreQueryMode.HYBRID,
                filters=filters,
            )
            manual = manual_rrf(
                [dense, bm25],
                limit=settings.rag_candidate_top_k,
                rrf_k=settings.rag_rrf_k,
            )
            dense_ids = ids(dense)
            bm25_ids = ids(bm25)
            hybrid_ids = ids(hybrid)
            manual_ids = ids(manual)
            union_ids = list(dict.fromkeys([*dense_ids, *bm25_ids]))
            no_time_input = hybrid[: settings.rag_rerank_input_top_n]

            rerank_started = perf_counter()
            no_time_output = await rerank_chat_chunks(
                query=query_text,
                candidates=no_time_input,
                top_n=settings.rag_final_top_n,
            )
            no_time_rerank_ms = (perf_counter() - rerank_started) * 1000
            production_input = no_time_input
            production_output = no_time_output
            time_input = None
            time_output = None
            oracle_input = None
            oracle_output = None
            oracle_rerank_ms = None

            if expected_hint is not None:
                time_input = (
                    select_rerank_candidates(
                        hybrid, temporal_order(hybrid, actual_hint)
                    )
                    if actual_hint is not None
                    else no_time_input
                )
                if ids(time_input) == ids(no_time_input):
                    time_output = no_time_output
                    production_rerank_ms = no_time_rerank_ms
                else:
                    started = perf_counter()
                    time_output = await rerank_chat_chunks(
                        query=query_text,
                        candidates=time_input,
                        top_n=settings.rag_final_top_n,
                    )
                    production_rerank_ms = (perf_counter() - started) * 1000
                production_input = time_input
                production_output = time_output

                oracle_input = select_rerank_candidates(
                    hybrid,
                    temporal_order(hybrid, expected_hint),
                )
                if ids(oracle_input) == ids(no_time_input):
                    oracle_output = no_time_output
                    oracle_rerank_ms = no_time_rerank_ms
                elif ids(oracle_input) == ids(time_input):
                    oracle_output = time_output
                    oracle_rerank_ms = production_rerank_ms
                else:
                    started = perf_counter()
                    oracle_output = await rerank_chat_chunks(
                        query=query_text,
                        candidates=oracle_input,
                        top_n=settings.rag_final_top_n,
                    )
                    oracle_rerank_ms = (perf_counter() - started) * 1000
            else:
                production_rerank_ms = no_time_rerank_ms

            groups = target_groups(row, message_seq, seq_nodes)
            record: dict[str, Any] = {
                "qid": row["qid"],
                "category": row["category"],
                "question": row["question"],
                "retrieval_query": query_text,
                "tool_args": tool_args,
                "expected_time_hint": row.get("time_hint"),
                "actual_time_hint": None
                if actual_hint is None
                else actual_hint.model_dump(mode="json"),
                "time_hint_exact": (
                    None
                    if expected_hint is None
                    else same_time_hint(expected_hint, actual_hint)
                ),
                "target_groups": [sorted(group) for group in groups],
                "dense_ids": dense_ids,
                "bm25_ids": bm25_ids,
                "route_union_ids": union_ids,
                "milvus_rrf_ids": hybrid_ids,
                "manual_rrf_ids": manual_ids,
                "rrf_top3_ids": hybrid_ids[:3],
                "rrf_top5_ids": hybrid_ids[: settings.rag_rerank_input_top_n],
                "no_time_reranker_input_ids": ids(no_time_input),
                "no_time_output_ids": ids(no_time_output),
                "production_reranker_input_ids": ids(production_input),
                "production_output_ids": ids(production_output),
                "stage_latency_ms": {
                    "time_intent_llm": time_intent_ms,
                    "query_embedding": round(embedding_ms, 2),
                    "dense_milvus": round(dense_ms, 2),
                    "bm25_milvus": round(bm25_ms, 2),
                    "hybrid_rrf_milvus": round(hybrid_ms, 2),
                    "qwen_no_time": round(no_time_rerank_ms, 2),
                    "qwen_production": round(production_rerank_ms, 2),
                    "qwen_oracle_time": (
                        None if oracle_rerank_ms is None else round(oracle_rerank_ms, 2)
                    ),
                },
                "manual_rrf_exact_order_match": manual_ids == hybrid_ids,
                "manual_rrf_top_k_overlap": len(set(manual_ids) & set(hybrid_ids)),
            }
            if time_input is not None and time_output is not None:
                record["llm_time_input_ids"] = ids(time_input)
                record["llm_time_output_ids"] = ids(time_output)
            if oracle_input is not None and oracle_output is not None:
                record["oracle_time_input_ids"] = ids(oracle_input)
                record["oracle_time_output_ids"] = ids(oracle_output)
            record["target_ranks"] = {
                stage: [
                    rank_of(group, record[key]) for group in record["target_groups"]
                ]
                for stage, key in (
                    ("dense", "dense_ids"),
                    ("bm25", "bm25_ids"),
                    ("route_union", "route_union_ids"),
                    ("milvus_rrf", "milvus_rrf_ids"),
                    ("manual_rrf", "manual_rrf_ids"),
                    ("reranker_input", "production_reranker_input_ids"),
                    ("qwen_output", "production_output_ids"),
                )
            }
            records.append(record)
            if position % 10 == 0 or position == len(answerable):
                print(f"traced {position}/{len(answerable)} queries", flush=True)

        single = [row for row in records if row["category"].startswith("single_block:")]
        multi = [row for row in records if row["category"].startswith("multi_block:")]
        temporal = [row for row in records if row["category"].startswith("temporal:")]
        temporal_metrics: dict[str, Any] = {}
        for time_kind in ("range", "latest"):
            subset = [
                row
                for row in temporal
                if row["category"].startswith(f"temporal:{time_kind}:")
            ]
            temporal_metrics[time_kind] = {
                "rrf_top10": single_metrics(subset, "milvus_rrf_ids", cutoff=10),
                "no_time_top5": single_metrics(
                    subset, "no_time_reranker_input_ids", cutoff=5
                ),
                "oracle_time_top5": single_metrics(
                    subset, "oracle_time_input_ids", cutoff=5
                ),
                "llm_time_top5": single_metrics(subset, "llm_time_input_ids", cutoff=5),
                "no_time_qwen_top3": single_metrics(
                    subset, "no_time_output_ids", cutoff=3
                ),
                "oracle_time_qwen_top3": single_metrics(
                    subset, "oracle_time_output_ids", cutoff=3
                ),
                "llm_time_qwen_top3": single_metrics(
                    subset, "llm_time_output_ids", cutoff=3
                ),
            }

        stage_metrics = {
            "single_evidence": {
                "dense_top10": single_metrics(single, "dense_ids", cutoff=10),
                "bm25_top10": single_metrics(single, "bm25_ids", cutoff=10),
                "route_union": union_coverage(single),
                "milvus_rrf_top10": single_metrics(single, "milvus_rrf_ids", cutoff=10),
                "manual_rrf_top10": single_metrics(single, "manual_rrf_ids", cutoff=10),
                "cross_encoder_input_top5": single_metrics(
                    single, "production_reranker_input_ids", cutoff=5
                ),
                "qwen_final_top3": single_metrics(
                    single, "production_output_ids", cutoff=3
                ),
            },
            "multi_evidence": {
                "dense_top10": multi_metrics(multi, "dense_ids", cutoff=10),
                "bm25_top10": multi_metrics(multi, "bm25_ids", cutoff=10),
                "route_union": union_coverage(multi),
                "milvus_rrf_top10": multi_metrics(multi, "milvus_rrf_ids", cutoff=10),
                "manual_rrf_top10": multi_metrics(multi, "manual_rrf_ids", cutoff=10),
                "milvus_rrf_top5": multi_metrics(multi, "milvus_rrf_ids", cutoff=5),
                "cross_encoder_input_top5": multi_metrics(
                    multi, "production_reranker_input_ids", cutoff=5
                ),
                "qwen_final_top3": multi_metrics(
                    multi, "production_output_ids", cutoff=3
                ),
            },
            "stage_transitions_multi": {
                "route_union_to_rrf_top10": stage_transition(
                    multi, "route_union_ids", "milvus_rrf_ids"
                ),
                "rrf_top10_to_reranker_input_top5": stage_transition(
                    multi, "milvus_rrf_ids", "production_reranker_input_ids"
                ),
                "reranker_input_top5_to_qwen_top3": stage_transition(
                    multi,
                    "production_reranker_input_ids",
                    "production_output_ids",
                ),
                "rrf_top3_to_qwen_top3": stage_transition(
                    multi, "rrf_top3_ids", "production_output_ids"
                ),
            },
            "temporal": {
                "questions": len(temporal),
                "time_hint_exact": sum(
                    bool(row["time_hint_exact"]) for row in temporal
                ),
                "time_hint_total": len(temporal),
                "subsets": temporal_metrics,
            },
        }
        result = {
            "dataset": "room_chat_v3",
            "message_count": len(messages),
            "question_count": len(queries),
            "answerable_count": len(answerable),
            "unanswerable_count": sum(not row["answerable"] for row in queries),
            "child_chunk_count": len(children),
            "mapped_evidence_groups": sum(
                bool(group) for row in records for group in row["target_groups"]
            ),
            "settings": {
                "embedding_model": settings.embedding_model_id,
                "reranker_model": settings.reranker_model_id,
                "per_route_top_k": settings.rag_candidate_top_k,
                "rrf_output_top_k": settings.rag_candidate_top_k,
                "reranker_input_top_n": settings.rag_rerank_input_top_n,
                "final_top_n": settings.rag_final_top_n,
                "rrf_k": settings.rag_rrf_k,
                "temporal_rank_weight": settings.rag_temporal_rank_weight,
                "bm25_analyzer": "chinese",
            },
            "rrf_implementation_check": {
                "exact_order_match_queries": sum(
                    bool(row["manual_rrf_exact_order_match"]) for row in records
                ),
                "query_count": len(records),
                "mean_top_k_overlap": round(
                    statistics.mean(row["manual_rrf_top_k_overlap"] for row in records),
                    2,
                ),
                "note": (
                    "Standalone route queries now mirror the dense and BM25 AnnSearchRequest "
                    "parameters used by HYBRID. A remaining rank difference indicates RRF tie/order "
                    "behavior or a route implementation detail that warrants per-query inspection."
                ),
            },
            "latency_ms": {
                stage: latency_summary(records, stage)
                for stage in (
                    "time_intent_llm",
                    "query_embedding",
                    "dense_milvus",
                    "bm25_milvus",
                    "hybrid_rrf_milvus",
                    "qwen_no_time",
                    "qwen_production",
                )
            },
            "setup": {
                "index_write_ms": round(index_write_ms, 2),
                "reranker_warmup_ms": round(reranker_warmup_ms, 2),
            },
            "metrics": stage_metrics,
            "queries": records,
        }
        RESULT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(stage_metrics, ensure_ascii=False, indent=2), flush=True)
        print(
            json.dumps(result["latency_ms"], ensure_ascii=False, indent=2), flush=True
        )
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
    asyncio.run(run_stage_evaluation())
