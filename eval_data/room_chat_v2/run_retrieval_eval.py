"""Compare retrieval and reranking modes on the broad-topic room_chat_v2 set."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusClient

from config.config import settings
from domain.chat_message import ChatMessage
from harness.llm.cluster_messages import cluster_messages
from infra.model.embedding_model_factory import get_embedding_model
from infra.mysql.database import close_mysql
from infra.rag.message_index import build_message_chunks

ROOT = Path(__file__).parent
CORPUS_PATH = ROOT / "corpus.jsonl"
QUERIES_PATH = ROOT / "queries.jsonl"
RESULT_PATH = ROOT / "retrieval_results.json"
CHUNKS_PATH = ROOT / "child_chunks.jsonl"
CHUNKS_CORPUS_HASH_PATH = ROOT / "child_chunks.corpus.sha256"
ROOM_ID = "eval-room-v2"
TOP_K = 5
RERANK_TOP_N = 3
COLLECTION_STEM = "room_agent_eval_v2_20261101"
DENSE_COLLECTION = f"{COLLECTION_STEM}_dense"
HYBRID_COLLECTION = f"{COLLECTION_STEM}_hybrid"
SEQ_PATTERN = re.compile(r"\[seq=(\d+),")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def to_messages(rows: Sequence[dict[str, Any]]) -> list[ChatMessage]:
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


def node_seqs(hit: NodeWithScore) -> set[int]:
    return {int(value) for value in SEQ_PATTERN.findall(hit.node.text)}


def ranked_seq_sets(hits: Sequence[NodeWithScore]) -> list[set[int]]:
    return [node_seqs(hit) for hit in hits]


def summarize_metrics(
    records: list[dict[str, Any]], method: str, *, cutoff: int | None = None
) -> dict[str, float]:
    answerable = [row for row in records if row["answerable"]]
    if not answerable:
        return {}

    recalls: list[float] = []
    hits: list[float] = []
    complete: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []

    for row in answerable:
        gold = set(row["gold_seqs"])
        ranked = row[method]["ranked_seqs"]
        if cutoff is not None:
            ranked = ranked[:cutoff]
        found = set().union(*ranked) if ranked else set()
        recalls.append(len(gold & found) / len(gold))
        hits.append(float(bool(gold & found)))
        complete.append(float(gold <= found))
        first_rank = next(
            (rank for rank, seqs in enumerate(ranked, start=1) if gold & set(seqs)),
            None,
        )
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        latencies.append(row[method]["latency_ms"])

    return {
        "questions": len(answerable),
        "evidence_recall_at_k": statistics.mean(recalls),
        "hit_rate_at_k": statistics.mean(hits),
        "complete_evidence_rate_at_k": statistics.mean(complete),
        "mrr": statistics.mean(reciprocal_ranks),
        "mean_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
    }


async def close_store(store: MilvusVectorStore | None) -> None:
    if store is None:
        return
    if store.aclient is not None:
        await store.aclient.close()
    await asyncio.to_thread(store.client.close)


async def run() -> None:
    corpus_hash = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    query_hash = hashlib.sha256(QUERIES_PATH.read_bytes()).hexdigest()
    corpus = read_jsonl(CORPUS_PATH)
    queries = read_jsonl(QUERIES_PATH)
    messages = to_messages(corpus)
    print(f"loaded messages={len(messages)} queries={len(queries)}", flush=True)

    cached_corpus_hash = (
        CHUNKS_CORPUS_HASH_PATH.read_text(encoding="utf-8").strip()
        if CHUNKS_CORPUS_HASH_PATH.exists()
        else None
    )
    if CHUNKS_PATH.exists() and cached_corpus_hash == corpus_hash:
        children = [
            TextNode(
                id_=row["id"],
                text=row["text"],
                metadata=row["metadata"],
                excluded_embed_metadata_keys=list(row["metadata"]),
                excluded_llm_metadata_keys=list(row["metadata"]),
            )
            for row in read_jsonl(CHUNKS_PATH)
        ]
        print(f"loaded cached child_chunks={len(children)}", flush=True)
    else:
        _, children = await build_message_chunks(room_id=ROOM_ID, messages=messages)
        with CHUNKS_PATH.open("w", encoding="utf-8") as file:
            for child in children:
                file.write(
                    json.dumps(
                        {"id": child.node_id, "text": child.text, "metadata": child.metadata},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        CHUNKS_CORPUS_HASH_PATH.write_text(corpus_hash + "\n", encoding="utf-8")
        print(f"saved child chunk cache: {CHUNKS_PATH}", flush=True)
    print(f"production_child_chunks={len(children)}", flush=True)

    uri = settings.milvus_uri
    if "://" not in uri:
        uri = f"http://{uri}"
    embedding_model = await get_embedding_model()

    hybrid_store: MilvusVectorStore | None = None
    dense_store: MilvusVectorStore | None = None
    try:
        hybrid_store = MilvusVectorStore(
            uri=uri,
            collection_name=HYBRID_COLLECTION,
            dim=settings.embedding_dimension,
            embedding_field="embedding",
            similarity_metric="COSINE",
            enable_sparse=True,
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 60},
            consistency_level="Strong",
            overwrite=False,
            upsert_mode=True,
            use_async_client=True,
        )
        dense_store = MilvusVectorStore(
            uri=uri,
            collection_name=DENSE_COLLECTION,
            dim=settings.embedding_dimension,
            embedding_field="embedding",
            similarity_metric="COSINE",
            enable_sparse=False,
            consistency_level="Strong",
            overwrite=False,
            upsert_mode=True,
            use_async_client=True,
        )
        hybrid_index = VectorStoreIndex.from_vector_store(hybrid_store, embed_model=embedding_model)
        dense_index = VectorStoreIndex.from_vector_store(dense_store, embed_model=embedding_model)

        # 两路索引使用完全相同的生产切分节点和 embedding 模型。
        await hybrid_index.ainsert_nodes(children)
        await dense_index.ainsert_nodes(children)
        print("indexes_ready", flush=True)

        filters = MetadataFilters(filters=[ExactMatchFilter(key="room_id", value=ROOM_ID)])
        dense_retriever = dense_index.as_retriever(similarity_top_k=TOP_K, filters=filters)
        hybrid_retriever = hybrid_index.as_retriever(
            similarity_top_k=TOP_K,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            filters=filters,
        )
        rerankers = {
            "hybrid_reranked_minilm": SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L6-v2",
                top_n=RERANK_TOP_N,
            ),
            "hybrid_reranked_qwen": SentenceTransformerRerank(
                model="Qwen/Qwen3-Reranker-0.6B",
                top_n=RERANK_TOP_N,
            ),
        }

        records: list[dict[str, Any]] = []
        for number, question in enumerate(queries, start=1):
            record: dict[str, Any] = {
                "qid": question["qid"],
                "question": question["question"],
                "answerable": question["answerable"],
                "gold_seqs": question["gold_seqs"],
            }

            started = time.perf_counter()
            dense_hits = await dense_retriever.aretrieve(question["question"])
            dense_ms = (time.perf_counter() - started) * 1000
            record["dense"] = {
                "latency_ms": dense_ms,
                "ranked_seqs": [sorted(seqs) for seqs in ranked_seq_sets(dense_hits)],
                "scores": [float(hit.score) if hit.score is not None else None for hit in dense_hits],
            }

            started = time.perf_counter()
            hybrid_hits = await hybrid_retriever.aretrieve(question["question"])
            hybrid_ms = (time.perf_counter() - started) * 1000
            record["hybrid"] = {
                "latency_ms": hybrid_ms,
                "ranked_seqs": [sorted(seqs) for seqs in ranked_seq_sets(hybrid_hits)],
                "scores": [float(hit.score) if hit.score is not None else None for hit in hybrid_hits],
            }

            for method, reranker in rerankers.items():
                started = time.perf_counter()
                reranked = await reranker.apostprocess_nodes(
                    hybrid_hits,
                    query_str=question["question"],
                )
                rerank_ms = (time.perf_counter() - started) * 1000
                record[method] = {
                    "latency_ms": hybrid_ms + rerank_ms,
                    "retrieval_latency_ms": hybrid_ms,
                    "rerank_latency_ms": rerank_ms,
                    "ranked_seqs": [sorted(seqs) for seqs in ranked_seq_sets(reranked)],
                    "scores": [float(hit.score) if hit.score is not None else None for hit in reranked],
                }
            records.append(record)
            if number % 10 == 0:
                print(f"evaluated {number}/{len(queries)}", flush=True)

        result = {
            "dataset": "room_chat_v2",
            "room_id": ROOM_ID,
            "message_count": len(messages),
            "query_count": len(queries),
            "answerable_count": sum(bool(q["answerable"]) for q in queries),
            "unanswerable_count": sum(not bool(q["answerable"]) for q in queries),
            "settings": {
                "top_k": TOP_K,
                "rerank_top_n": RERANK_TOP_N,
                "embedding_model": settings.embedding_model_id,
                "cluster_model": settings.agent_model,
                "rerankers": list(rerankers),
                "child_chunk_count": len(children),
                "corpus_sha256": corpus_hash,
                "queries_sha256": query_hash,
                "child_chunks_sha256": hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest(),
                "indexing": "production build_message_chunks; synthetic timestamps are not used for temporal claims",
            },
            "metrics": {
                "dense": summarize_metrics(records, "dense"),
                "hybrid_rrf": summarize_metrics(records, "hybrid"),
                "hybrid_rrf_minilm_reranked": summarize_metrics(
                    records, "hybrid_reranked_minilm"
                ),
                "hybrid_rrf_qwen_reranked": summarize_metrics(
                    records, "hybrid_reranked_qwen"
                ),
            },
            "metrics_same_cutoff_top3": {
                "dense_top3": summarize_metrics(records, "dense", cutoff=3),
                "hybrid_rrf_top3": summarize_metrics(records, "hybrid", cutoff=3),
                "hybrid_rrf_minilm_reranked_top3": summarize_metrics(
                    records, "hybrid_reranked_minilm", cutoff=3
                ),
                "hybrid_rrf_qwen_reranked_top3": summarize_metrics(
                    records, "hybrid_reranked_qwen", cutoff=3
                ),
            },
            "queries": records,
        }
        RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2), flush=True)
        print(f"saved {RESULT_PATH}", flush=True)
    finally:
        await close_store(dense_store)
        await close_store(hybrid_store)
        await close_mysql()
        client = MilvusClient(uri=uri)
        for collection in (DENSE_COLLECTION, HYBRID_COLLECTION):
            if client.has_collection(collection):
                client.drop_collection(collection)
        client.close()


if __name__ == "__main__":
    asyncio.run(run())
