"""Compare temporal weight 1 with 0.25 on exactly the same archived RRF pool."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
from pathlib import Path

print("loading reranking modules; reuse archived candidates; no LLM calls", flush=True)
from config.config import settings

uri = settings.milvus_uri
os.environ["MILVUS_URI"] = uri if "://" in uri else f"http://{uri}"

from domain.time_hint import TimeHint
from infra.model.reranker_model_factory import get_message_reranker
from infra.rag.message_retriever import rerank_chat_chunks, select_rerank_candidates, temporal_order
from llama_index.core.schema import NodeWithScore, TextNode

DATA = Path(__file__).parent
BASE = DATA / "raw_chunk_eval" / "single_k20_l10_r5_n3"
OUT = BASE / "time_weight_1"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rank(row, key):
    targets = set(row["acceptable_child_ids"])
    return next((i for i, hit in enumerate(row[key]["top3"], 1) if hit["id"] in targets), None)


def metrics(rows, key):
    ranks = [rank(row, key) for row in rows]
    return {"questions": len(rows), "hits": sum(r is not None for r in ranks),
            "recall_at_3": statistics.mean(int(r is not None) for r in ranks),
            "mrr_at_3": statistics.mean(1/r if r else 0 for r in ranks)}


async def run():
    OUT.mkdir(exist_ok=True)
    parameters = json.loads((BASE / "config.json").read_text())
    chunk_path = DATA / "raw_chunk_eval" / "child_chunks.jsonl"
    assert hashlib.sha256(chunk_path.read_bytes()).hexdigest() == parameters["source_chunks_sha256"]
    settings.rag_temporal_rank_weight = 1.0
    settings.rag_rrf_k = parameters["rrf_smoothing_k"]
    settings.rag_rerank_input_top_n = parameters["rerank_input"]
    settings.reranker_model_id = parameters["reranker"]
    settings.rag_device = parameters["device"]
    parameters = {**parameters, "time_weight": 1.0, "reused_rrf_candidates": True}
    (OUT / "config.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2))
    nodes = {}
    for row in read_jsonl(chunk_path):
        nodes[row["id"]] = TextNode(id_=row["id"], text=row["text"], metadata=row["metadata"],
                                    excluded_embed_metadata_keys=list(row["metadata"]),
                                    excluded_llm_metadata_keys=list(row["metadata"]))
    questions = {q["qid"]: q for q in read_jsonl(DATA / "queries.jsonl")}
    records = read_jsonl(BASE / "results.jsonl")
    await get_message_reranker()
    completed = 0
    with (OUT / "results.jsonl").open("w") as output:
        for row in records:
            row["weight_025"] = row.pop("time_on")
            if not row["has_time_intent"]:
                row["weight_1"] = row["weight_025"]
            else:
                candidates = [NodeWithScore(node=nodes[child_id]) for child_id in row["rrf_top10"]]
                hint = TimeHint.model_validate(questions[row["qid"]]["time_hint"])
                selected = select_rerank_candidates(candidates, temporal_order(candidates, hint))
                ranked = await rerank_chat_chunks(query=row["question"], candidates=selected,
                                                  top_n=parameters["final_n"], time_hint=hint)
                assert len(ranked) == 3
                row["weight_1"] = {
                    "rerank_input_ids": [hit.node.node_id for hit in selected],
                    "top3": [{"id": hit.node.node_id, "text": hit.node.text,
                              "metadata": hit.node.metadata, "score": hit.score} for hit in ranked],
                }
                completed += 1
                if completed % 10 == 0:
                    print(f"reranked time questions {completed}/100", flush=True)
            output.write(json.dumps(row, ensure_ascii=False)+"\n")
            output.flush()
    temporal = [row for row in records if row["has_time_intent"]]
    summary = {
        "parameters": parameters,
        "overall": {key: metrics(records, key) for key in ("weight_025", "weight_1")},
        "temporal": {key: metrics(temporal, key) for key in ("weight_025", "weight_1")},
        "changed_shortlists": sum(row["weight_025"]["rerank_input_ids"] != row["weight_1"]["rerank_input_ids"] for row in temporal),
        "recovered_questions": [row["qid"] for row in temporal if rank(row, "weight_025") is None and rank(row, "weight_1") is not None],
        "lost_questions": [row["qid"] for row in temporal if rank(row, "weight_025") is not None and rank(row, "weight_1") is None],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(run())
