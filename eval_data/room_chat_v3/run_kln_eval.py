"""Sweep per-route K, RRF output L, and reranker output N on 80 single-evidence queries."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from config.config import settings

MILVUS_URI = (
    settings.milvus_uri
    if "://" in settings.milvus_uri
    else f"http://{settings.milvus_uri}"
)
os.environ["MILVUS_URI"] = MILVUS_URI

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from pymilvus import AnnSearchRequest, RRFRanker

from infra.model.embedding_model_factory import get_embedding_model
from infra.rag.message_retriever import rerank_chat_chunks


DATA_DIR = Path(__file__).parent
CHILD_PATH = DATA_DIR / "child_chunks.jsonl"
QUERY_PATH = DATA_DIR / "parameter_queries_80.jsonl"
RESULT_PATH = DATA_DIR / "parameter_sweep_80_results.json"
ROOM_ID = "eval-room-v3"
K_GRID = [5, 10, 20, 30]
L_GRID = [5, 10, 20, 30, 40, 60]
N_GRID = [1, 3, 5]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_parent_context() -> dict[str, dict[str, int]]:
    """Rebuild v3 parent batches to estimate source-message expansion per result."""
    messages = sorted(
        load_jsonl(DATA_DIR / "corpus.jsonl"), key=lambda row: int(row["seq"])
    )
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for message in messages:
        sender = message.get("sender_name") or message["sender_id"]
        record = (
            f"[seq={message['seq']}, send_time={message['send_time']}] "
            f"[{sender} ({message['sender_id']})]: {message['content']}"
        )
        if current and (
            len(current) >= settings.rag_parent_message_count
            or current_chars + len(record) > settings.rag_parent_max_chars
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(message)
        current_chars += len(record)
    if current:
        batches.append(current)

    result: dict[str, dict[str, int]] = {}
    for index, batch in enumerate(batches, start=1):
        chars = 0
        for message in batch:
            sender = message.get("sender_name") or message["sender_id"]
            chars += len(
                f"[seq={message['seq']}, send_time={message['send_time']}] "
                f"[{sender} ({message['sender_id']})]: {message['content']}"
            )
        result[f"eval-v3-parent-{index:04d}"] = {
            "messages": len(batch),
            "chars": chars,
        }
    return result


def summarize(
    questions: list[dict[str, Any]],
    ranked: dict[str, list[str]],
    limit: int,
    nodes: dict[str, TextNode],
    parent_context: dict[str, dict[str, int]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    groups = {
        "all": questions,
        "existing_single": [q for q in questions if q["kind"] == "existing_single"],
        "split_subquery": [
            q for q in questions if q["kind"] == "single_evidence_subquery"
        ],
    }
    for name, rows in groups.items():
        hits = 0
        reciprocal_ranks: list[float] = []
        counts: list[int] = []
        chars: list[int] = []
        parent_counts: list[int] = []
        parent_messages: list[int] = []
        parent_chars: list[int] = []
        for row in rows:
            ids = ranked[row["qid"]][:limit]
            rank = next(
                (i for i, node_id in enumerate(ids, 1) if node_id == row["target_child_id"]),
                None,
            )
            hits += int(rank is not None)
            reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
            counts.append(len(ids))
            chars.append(sum(len(nodes[node_id].text) for node_id in ids))
            parent_ids = {str(nodes[node_id].metadata["parent_id"]) for node_id in ids}
            parent_counts.append(len(parent_ids))
            parent_messages.append(
                sum(parent_context[parent_id]["messages"] for parent_id in parent_ids)
            )
            parent_chars.append(
                sum(parent_context[parent_id]["chars"] for parent_id in parent_ids)
            )
        output[name] = {
            "questions": len(rows),
            "hits": hits,
            "recall": round(hits / len(rows), 4),
            "mrr": round(statistics.mean(reciprocal_ranks), 4),
            "mean_returned_children": round(statistics.mean(counts), 2),
            "mean_context_chars": round(statistics.mean(chars), 1),
            "mean_unique_parent_chunks": round(statistics.mean(parent_counts), 2),
            "mean_parent_messages": round(statistics.mean(parent_messages), 1),
            "mean_parent_source_chars": round(statistics.mean(parent_chars), 1),
        }
    return output


async def run() -> None:
    child_rows = load_jsonl(CHILD_PATH)
    questions = load_jsonl(QUERY_PATH)
    parent_context = build_parent_context()
    nodes = {
        row["id"]: TextNode(
            id_=row["id"],
            text=row["text"],
            metadata=row["metadata"],
            excluded_embed_metadata_keys=list(row["metadata"]),
            excluded_llm_metadata_keys=list(row["metadata"]),
        )
        for row in child_rows
    }
    if len(questions) != 80 or len({q["target_child_id"] for q in questions}) != 80:
        raise ValueError("评测集必须包含 80 个不同目标子块的单证据问题")

    collection = f"eval_kln_80_{uuid4().hex[:10]}"
    store: MilvusVectorStore | None = None
    try:
        store = MilvusVectorStore(
            uri=MILVUS_URI,
            collection_name=collection,
            dim=settings.embedding_dimension,
            embedding_field="embedding",
            similarity_metric="COSINE",
            enable_sparse=True,
            sparse_embedding_function=cast(
                Any, BM25BuiltInFunction(analyzer_params={"type": "chinese"})
            ),
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": settings.rag_rrf_k},
            consistency_level="Strong",
            overwrite=False,
            upsert_mode=True,
            use_async_client=True,
        )
        embedding_model = await get_embedding_model()
        index = VectorStoreIndex.from_vector_store(
            vector_store=store,
            embed_model=embedding_model,
        )
        started = perf_counter()
        await index.ainsert_nodes(list(nodes.values()))
        indexing_ms = (perf_counter() - started) * 1000

        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="room_id", value=ROOM_ID)]
        )
        # Exclude one-time model loading from the per-query timing.
        await rerank_chat_chunks(
            query="评测预热",
            candidates=[NodeWithScore(node=next(iter(nodes.values())), score=0.0)],
            top_n=1,
        )

        records: list[dict[str, Any]] = []
        embedding_ms: list[float] = []
        retrieval_ms: list[float] = []
        rerank_ms: list[float] = []
        for position, row in enumerate(questions, 1):
            query = row["query"]
            started = perf_counter()
            query_embedding = await embedding_model.aget_query_embedding(query)
            embedding_ms.append((perf_counter() - started) * 1000)

            rankings: dict[int, list[str]] = {}
            candidate_ids: list[str] = []
            for k in K_GRID:
                request_query = VectorStoreQuery(
                    query_embedding=query_embedding,
                    query_str=query,
                    similarity_top_k=k,
                    filters=filters,
                )
                filter_expr, output_fields = store._prepare_before_search(request_query)
                requests = [
                    AnnSearchRequest(
                        data=[query_embedding],
                        anns_field=store.embedding_field,
                        param={
                            "metric_type": store.similarity_metric,
                            "params": store.search_config,
                        },
                        limit=k,
                        expr=filter_expr,
                    ),
                    AnnSearchRequest(
                        data=[query],
                        anns_field=store.sparse_embedding_field,
                        param={"metric_type": "BM25"},
                        limit=k,
                        expr=filter_expr,
                    ),
                ]
                started = perf_counter()
                result = await store.aclient.hybrid_search(
                    collection_name=collection,
                    reqs=requests,
                    ranker=RRFRanker(k=settings.rag_rrf_k),
                    limit=2 * k,
                    output_fields=output_fields,
                )
                retrieval_ms.append((perf_counter() - started) * 1000)
                ids = [str(hit["id"]) for hit in result[0]]
                rankings[k] = ids
                candidate_ids.extend(ids)

            # Cross-encoder score is independent per (query, child) pair. Score each
            # unique candidate once and reuse it for every L/N setting.
            candidate_ids = list(dict.fromkeys(candidate_ids))
            candidates = [NodeWithScore(node=nodes[node_id], score=0.0) for node_id in candidate_ids]
            started = perf_counter()
            scored = await rerank_chat_chunks(
                query=query,
                candidates=candidates,
                top_n=len(candidates),
            )
            rerank_ms.append((perf_counter() - started) * 1000)
            score_by_id = {
                hit.node.node_id: float(hit.score or 0.0) for hit in scored
            }
            records.append(
                {
                    "qid": row["qid"],
                    "kind": row["kind"],
                    "target": row["target_child_id"],
                    "rankings": rankings,
                    "scores": score_by_id,
                }
            )
            if position % 10 == 0 or position == len(questions):
                print(f"evaluated {position}/{len(questions)}", flush=True)

        result_data: dict[str, Any] = {
            "dataset": "room_chat_v3_parameter_queries_80",
            "questions": len(questions),
            "unique_target_children": len({q["target_child_id"] for q in questions}),
            "child_chunks": len(nodes),
            "question_composition": {
                "existing_single": sum(q["kind"] == "existing_single" for q in questions),
                "split_subquery": sum(q["kind"] == "single_evidence_subquery" for q in questions),
            },
            "models": {
                "embedding": settings.embedding_model_id,
                "reranker": settings.reranker_model_id,
                "chat_llm_called": False,
                "cluster_llm_called": False,
            },
            "fixed_rrf_k": settings.rag_rrf_k,
            "k_grid": K_GRID,
            "l_grid": L_GRID,
            "n_grid": N_GRID,
            "indexing_ms": round(indexing_ms, 1),
            "latency_ms": {},
            "configs": {},
        }
        for k in K_GRID:
            for l in L_GRID:
                if l > 2 * k:
                    continue
                candidate_rank = {
                    row["qid"]: row["rankings"][k][:l] for row in records
                }
                config_key = f"K{k}_L{l}"
                result_data["configs"][config_key] = {
                    "candidate_recall_at_L": summarize(
                        questions, candidate_rank, l, nodes, parent_context
                    ),
                    "reranked": {},
                }
                reranked: dict[str, list[str]] = {}
                for row in records:
                    ids = row["rankings"][k][:l]
                    reranked[row["qid"]] = sorted(
                        ids,
                        key=lambda node_id: row["scores"].get(
                            node_id, float("-inf")
                        ),
                        reverse=True,
                    )
                for n in N_GRID:
                    if n <= l:
                        result_data["configs"][config_key]["reranked"][str(n)] = summarize(
                            questions, reranked, n, nodes, parent_context
                        )

        for name, values in [
            ("query_embedding", embedding_ms),
            ("hybrid_search", retrieval_ms),
            ("rerank_max_candidate_union", rerank_ms),
        ]:
            ordered = sorted(values)
            result_data["latency_ms"][name] = {
                "mean": round(statistics.mean(values), 2),
                "p95": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 2),
                "samples": len(values),
            }

        RESULT_PATH.write_text(
            json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"RESULT {RESULT_PATH}", flush=True)
        for k, l in [(5, 5), (5, 10), (10, 10), (10, 20), (20, 20), (20, 30), (20, 40), (30, 30), (30, 60)]:
            key = f"K{k}_L{l}"
            if key not in result_data["configs"]:
                continue
            config = result_data["configs"][key]
            print(key, "candidate", config["candidate_recall_at_L"]["all"]["recall"], end=" ", flush=True)
            for n, metric in config["reranked"].items():
                all_metric = metric["all"]
                print(
                    f"N{n}: {all_metric['hits']}/80 ({all_metric['recall']:.3f}), "
                    f"MRR {all_metric['mrr']:.3f}, chars {all_metric['mean_context_chars']:.0f}",
                    end=" | ",
                    flush=True,
                )
            print(flush=True)
        print("LATENCY", result_data["latency_ms"], flush=True)
    finally:
        if store is not None:
            try:
                if store.aclient is not None:
                    await store.aclient.close()
            except Exception as exc:
                print("async close:", type(exc).__name__, str(exc)[:150], flush=True)
            try:
                await asyncio.to_thread(store.client.drop_collection, collection)
                print("DROPPED_TEMP_COLLECTION", collection, flush=True)
            except Exception as exc:
                print("drop collection:", type(exc).__name__, str(exc)[:200], flush=True)
            try:
                await asyncio.to_thread(store.client.close)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(run())
