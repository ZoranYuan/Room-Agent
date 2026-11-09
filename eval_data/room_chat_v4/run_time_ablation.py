"""Compare the same hybrid-retrieval candidates with temporal ranking off/on."""

# ruff: noqa: E402
from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
from pathlib import Path
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
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from pymilvus import MilvusClient

from domain.time_hint import TimeHint
from infra.model.embedding_model_factory import get_embedding_model
from infra.rag.message_retriever import (
    rerank_chat_chunks,
    select_rerank_candidates,
    temporal_order,
)


DATA_DIR = Path(__file__).parent
ROOM_ID = "eval-room-v4"
COLLECTION = f"eval_v4_time_{uuid4().hex[:10]}"
RESULT_PATH = DATA_DIR / f"time_ablation_{COLLECTION.rsplit('_', 1)[-1]}.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rank_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        return {"questions": 0, "recall_at_1": None, "recall_at_3": None, "mrr_at_3": None}
    hits_1 = hits_3 = 0
    reciprocal_ranks = []
    for row in rows:
        target = row["target_child_id"]
        ranked = row[key]
        try:
            rank = ranked[:3].index(target) + 1
        except ValueError:
            rank = None
        hits_1 += int(bool(ranked) and ranked[0] == target)
        hits_3 += int(rank is not None)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    return {
        "questions": len(rows),
        "recall_at_1": round(hits_1 / len(rows), 4),
        "recall_at_3": round(hits_3 / len(rows), 4),
        "mrr_at_3": round(statistics.mean(reciprocal_ranks), 4),
        "hits_at_1": hits_1,
        "hits_at_3": hits_3,
    }


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    index = max(0, math.ceil(len(values) * percentile_value) - 1)
    return round(sorted(values)[index], 2)


def recall_at(rows: list[dict[str, Any]], key: str, cutoff: int) -> float:
    if not rows:
        return 0.0
    hits = sum(
        row["target_child_id"] in row[key][:cutoff]
        for row in rows
    )
    return round(hits / len(rows), 4)


async def run() -> None:
    corpus = read_jsonl(DATA_DIR / "corpus.jsonl")
    queries = read_jsonl(DATA_DIR / "queries.jsonl")
    chunks = read_jsonl(DATA_DIR / "child_chunks.jsonl")
    nodes = [
        TextNode(
            id_=row["id"],
            text=row["text"],
            metadata=row["metadata"],
            excluded_embed_metadata_keys=list(row["metadata"]),
            excluded_llm_metadata_keys=list(row["metadata"]),
        )
        for row in chunks
    ]

    sync_client = MilvusClient(uri=MILVUS_URI)
    if sync_client.has_collection(COLLECTION):
        raise RuntimeError(f"评测 Collection 已存在：{COLLECTION}")
    store = MilvusVectorStore(
        uri=MILVUS_URI,
        collection_name=COLLECTION,
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

    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=store,
            embed_model=await get_embedding_model(),
        )
        await index.ainsert_nodes(nodes)
        retriever = index.as_retriever(
            similarity_top_k=settings.rag_candidate_top_k,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            filters=MetadataFilters(
                filters=[ExactMatchFilter(key="room_id", value=ROOM_ID)]
            ),
        )

        temporal_rows: list[dict[str, Any]] = []
        plain_rows: list[dict[str, Any]] = []
        retrieval_ms: list[float] = []
        rerank_off_ms: list[float] = []
        rerank_on_ms: list[float] = []

        for position, query in enumerate(queries, start=1):
            started = time.perf_counter()
            candidates = await retriever.aretrieve(query["question"])
            retrieval_ms.append((time.perf_counter() - started) * 1000)
            target_child_id = query["gold_evidence_groups"][0]["child_id"]
            no_time_input = candidates[: settings.rag_rerank_input_top_n]

            started = time.perf_counter()
            off_ranked = await rerank_chat_chunks(
                query=query["question"],
                candidates=no_time_input,
                top_n=settings.rag_final_top_n,
            )
            rerank_off_ms.append((time.perf_counter() - started) * 1000)

            row = {
                "qid": query["qid"],
                "kind": query["kind"],
                "question": query["question"],
                "target_child_id": target_child_id,
                "candidate_ids": [item.node.node_id for item in candidates],
                "rerank_input_off_ids": [item.node.node_id for item in no_time_input],
                "ranked_off_ids": [item.node.node_id for item in off_ranked],
            }

            if query["kind"] == "temporal":
                hint = TimeHint.model_validate(query["time_hint"])
                temporal_candidates = temporal_order(candidates, hint)
                time_input = select_rerank_candidates(candidates, temporal_candidates)
                started = time.perf_counter()
                on_ranked = await rerank_chat_chunks(
                    query=query["question"],
                    candidates=time_input,
                    top_n=settings.rag_final_top_n,
                    time_hint=hint,
                )
                rerank_on_ms.append((time.perf_counter() - started) * 1000)
                row.update(
                    {
                        "time_hint": query["time_hint"],
                        "rerank_input_on_ids": [item.node.node_id for item in time_input],
                        "ranked_on_ids": [item.node.node_id for item in on_ranked],
                    }
                )
                temporal_rows.append(row)
            else:
                plain_rows.append(row)

            if position % 20 == 0 or position == len(queries):
                print(f"retrieved and reranked {position}/{len(queries)}", flush=True)

        temporal_off = rank_metrics(temporal_rows, "ranked_off_ids")
        temporal_on = rank_metrics(temporal_rows, "ranked_on_ids")
        result = {
            "dataset": "room_chat_v4",
            "message_count": len(corpus),
            "child_chunk_count": len(nodes),
            "question_count": len(queries),
            "temporal_question_count": len(temporal_rows),
            "plain_question_count": len(plain_rows),
            "comparison": {
                "controlled_variable": "same query, same hybrid RRF candidate list; toggle time_hint ranking and reranker time context",
                "temporal_off": temporal_off,
                "temporal_on_oracle_hint": temporal_on,
                "temporal_delta_recall_at_1": round(
                    temporal_on["recall_at_1"] - temporal_off["recall_at_1"], 4
                ),
                "temporal_delta_recall_at_3": round(
                    temporal_on["recall_at_3"] - temporal_off["recall_at_3"], 4
                ),
                "temporal_delta_mrr_at_3": round(
                    temporal_on["mrr_at_3"] - temporal_off["mrr_at_3"], 4
                ),
                "candidate_recall_at_10": recall_at(
                    temporal_rows, "candidate_ids", settings.rag_candidate_top_k
                ),
                "rerank_input_recall_at_5_off": recall_at(
                    temporal_rows, "rerank_input_off_ids", settings.rag_rerank_input_top_n
                ),
                "rerank_input_recall_at_5_on": recall_at(
                    temporal_rows, "rerank_input_on_ids", settings.rag_rerank_input_top_n
                ),
            },
            "non_temporal_control": rank_metrics(plain_rows, "ranked_off_ids"),
            "latency_ms": {
                "hybrid_retrieval_mean": round(statistics.mean(retrieval_ms), 2),
                "hybrid_retrieval_p95": percentile(retrieval_ms, 0.95),
                "rerank_off_mean": round(statistics.mean(rerank_off_ms), 2),
                "rerank_on_mean": round(statistics.mean(rerank_on_ms), 2),
            },
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
            "temporal_records": temporal_rows,
            "plain_records": plain_rows,
        }
        RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if not key.endswith("_records")}, ensure_ascii=False, indent=2))
        print(f"saved {RESULT_PATH}")
    finally:
        try:
            if store.aclient is not None:
                await store.aclient.close()
        finally:
            await asyncio.to_thread(store.client.close)
            if sync_client.has_collection(COLLECTION):
                sync_client.drop_collection(COLLECTION)
            sync_client.close()


if __name__ == "__main__":
    asyncio.run(run())
