"""Compare reranking the first 5 vs all 10 fixed RRF candidates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent
BASE = DATA / "raw_chunk_eval" / "single_k20_l10_r5_n3"
OUT = DATA / "raw_chunk_eval" / "rerank_pool_size_5_vs_10_time_weight_1"

import sys

sys.path.insert(0, str(ROOT / "src"))

from config.config import settings

milvus_uri = settings.milvus_uri
os.environ["MILVUS_URI"] = (
    milvus_uri if "://" in milvus_uri else f"http://{milvus_uri}"
)

from domain.time_hint import TimeHint
from huggingface_hub import snapshot_download
from infra.model.reranker_model_factory import get_message_reranker
from infra.rag.message_retriever import (
    combine_rerank_and_temporal_ranks,
    rerank_chat_chunks,
)
from llama_index.core.schema import NodeWithScore, TextNode
from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values: list[float], value: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(value * len(ordered)) - 1)]


def summarize(rows: list[dict], mode: str) -> dict:
    reciprocal_ranks = []
    hits = 0
    for row in rows:
        targets = set(row["acceptable_child_ids"])
        rank = next(
            (i for i, item in enumerate(row[mode]["top3"], 1) if item["id"] in targets),
            None,
        )
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank is not None else 0.0)

    latencies = [row[mode]["elapsed_ms"] for row in rows]
    return {
        "questions": len(rows),
        "recall_at_3": hits / len(rows),
        "mrr_at_3": statistics.mean(reciprocal_ranks),
        "rerank_mean_ms": statistics.mean(latencies),
        "rerank_p95_ms": percentile(latencies, 0.95),
        "rerank_p99_ms": percentile(latencies, 0.99),
    }


def candidate_recall(rows: list[dict], count: int) -> dict:
    hits = sum(
        bool(set(row["acceptable_child_ids"]) & set(row["rrf_top10"][:count]))
        for row in rows
    )
    return {"questions": len(rows), "hits": hits, "recall": hits / len(rows)}


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_config = json.loads((BASE / "config.json").read_text())
    chunks_path = DATA / "raw_chunk_eval" / "child_chunks.jsonl"
    if hashlib.sha256(chunks_path.read_bytes()).hexdigest() != run_config["source_chunks_sha256"]:
        raise ValueError("评测子块与已归档的 RRF 候选不匹配")

    # Use the same locally cached reranker checkpoint and prompt as production.
    settings.reranker_model_id = snapshot_download(
        run_config["reranker"], local_files_only=True
    )
    settings.rag_device = run_config["device"]
    settings.rag_rrf_k = run_config["rrf_smoothing_k"]
    AutoTokenizer.from_pretrained(settings.reranker_model_id, local_files_only=True)

    nodes = {}
    for row in read_jsonl(chunks_path):
        nodes[row["id"]] = TextNode(
            id_=row["id"],
            text=row["text"],
            metadata=row["metadata"],
            excluded_embed_metadata_keys=list(row["metadata"]),
            excluded_llm_metadata_keys=list(row["metadata"]),
        )
    questions = {row["qid"]: row for row in read_jsonl(DATA / "queries.jsonl")}
    records = read_jsonl(BASE / "results.jsonl")

    print(
        f"Loading reranker on {settings.rag_device}; {len(records)} questions, "
        "same archived RRF Top10, no LLM calls",
        flush=True,
    )
    model = await get_message_reranker()
    # Warm up outside the timed evaluation.
    await asyncio.to_thread(model.predict, [("warmup", "warmup")])

    output_rows = []
    for index, row in enumerate(records, 1):
        candidates = [NodeWithScore(node=nodes[child_id]) for child_id in row["rrf_top10"]]
        raw_hint = questions[row["qid"]].get("time_hint")
        time_hint = TimeHint.model_validate(raw_hint) if raw_hint else None
        result = {
            key: row[key]
            for key in (
                "qid",
                "question",
                "has_time_intent",
                "acceptable_child_ids",
                "rrf_top10",
            )
        }

        for pool_size in (5, 10):
            mode = f"top{pool_size}"
            pool = candidates[:pool_size]
            started = time.perf_counter()
            ranked = await rerank_chat_chunks(
                query=row["question"],
                candidates=pool,
                time_hint=time_hint,
            )
            if time_hint is not None:
                ranked = combine_rerank_and_temporal_ranks(ranked, time_hint)
            elapsed_ms = (time.perf_counter() - started) * 1000
            result[mode] = {
                "rerank_input_ids": [item.node.node_id for item in pool],
                "top3": [
                    {
                        "id": item.node.node_id,
                        "score": item.score,
                        "metadata": item.node.metadata,
                    }
                    for item in ranked[:3]
                ],
                "elapsed_ms": elapsed_ms,
            }

        output_rows.append(result)
        if index % 10 == 0:
            print(f"evaluated {index}/{len(records)}", flush=True)

    (OUT / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows)
    )

    subsets = {
        "overall": output_rows,
        "temporal": [row for row in output_rows if row["has_time_intent"]],
        "non_temporal": [row for row in output_rows if not row["has_time_intent"]],
    }
    summary = {
        "parameters": {
            "dataset": "room_chat_v5 single-evidence subset",
            "questions": len(output_rows),
            "rrf_candidates": 10,
            "reranker_pools_compared": [5, 10],
            "final_top_n": 3,
            "temporal_rank_weight": settings.rag_temporal_rank_weight,
            "rrf_k": settings.rag_rrf_k,
            "reranker": run_config["reranker"],
            "device": settings.rag_device,
            "prompt": "current production reranker instruction",
            "llm_calls": 0,
        },
        "candidate_recall": {
            label: {
                f"top{count}": candidate_recall(rows, count)
                for count in (5, 10)
            }
            for label, rows in subsets.items()
        },
        "final_metrics": {
            label: {
                f"rerank_top{count}": summarize(rows, f"top{count}")
                for count in (5, 10)
            }
            for label, rows in subsets.items()
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
