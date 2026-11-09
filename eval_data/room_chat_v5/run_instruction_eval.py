"""Compare reranker instructions on the same archived five candidates."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from run_time_weight_eval import read_jsonl, metrics, rank
from config.config import settings
from domain.time_hint import TimeHint
from infra.model.reranker_model_factory import CHAT_HISTORY_INSTRUCTION, get_message_reranker
from infra.rag.message_retriever import rerank_chat_chunks
from llama_index.core.schema import NodeWithScore, TextNode
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

DATA = Path(__file__).parent
BASE = DATA / "raw_chunk_eval" / "single_k20_l10_r5_n3" / "time_weight_1"
OUT = BASE / "strict_instruction"


async def run():
    OUT.mkdir(exist_ok=True)
    config = json.loads((BASE / "config.json").read_text())
    chunk_path = DATA / "raw_chunk_eval" / "child_chunks.jsonl"
    assert hashlib.sha256(chunk_path.read_bytes()).hexdigest() == config["source_chunks_sha256"]
    # Resolve the same cached checkpoint directly, avoiding offline Hub metadata lookups.
    checkpoint = snapshot_download(config["reranker"], local_files_only=True)
    config = {**config, "instruction": CHAT_HISTORY_INSTRUCTION,
              "reuse_exact_rerank_input": True, "local_checkpoint": checkpoint}
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    settings.reranker_model_id = checkpoint
    settings.rag_device = config["device"]
    nodes = {}
    for row in read_jsonl(chunk_path):
        nodes[row["id"]] = TextNode(id_=row["id"], text=row["text"], metadata=row["metadata"],
                                    excluded_embed_metadata_keys=list(row["metadata"]),
                                    excluded_llm_metadata_keys=list(row["metadata"]))
    questions = {q["qid"]: q for q in read_jsonl(DATA / "queries.jsonl")}
    rows = read_jsonl(BASE / "results.jsonl")
    print("loading reranker; same five candidates and same time hints", flush=True)
    AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    await get_message_reranker()
    with (OUT / "results.jsonl").open("w") as output:
        for position, row in enumerate(rows, 1):
            old = row["weight_1"]
            selected = [NodeWithScore(node=nodes[cid]) for cid in old["rerank_input_ids"]]
            raw_hint = questions[row["qid"]].get("time_hint")
            hint = TimeHint.model_validate(raw_hint) if raw_hint else None
            ranked = await rerank_chat_chunks(query=row["question"], candidates=selected, top_n=5, time_hint=hint)
            hits = [{"id": hit.node.node_id, "text": hit.node.text,
                     "metadata": hit.node.metadata, "score": hit.score} for hit in ranked]
            row["old_instruction"] = old
            row["new_instruction"] = {"rerank_input_ids": old["rerank_input_ids"],
                                      "ranked_top5": hits, "top3": hits[:3]}
            output.write(json.dumps(row, ensure_ascii=False)+"\n")
            output.flush()
            if position % 10 == 0:
                print(f"reranked {position}/{len(rows)}", flush=True)
    def pre5(items):
        count = sum(bool(set(r["acceptable_child_ids"]) & set(r["old_instruction"]["rerank_input_ids"])) for r in items)
        return {"questions": len(items), "hits": count, "recall_at_5": count/len(items)}
    summary = {"parameters": config,
               "pre_rerank": {name: pre5(items) for name, items in (
                   ("overall", rows), ("temporal", [r for r in rows if r["has_time_intent"]]),
                   ("non_temporal", [r for r in rows if not r["has_time_intent"]]))},
               "overall": {key: metrics(rows, key) for key in ("old_instruction", "new_instruction")},
               "temporal": {key: metrics([r for r in rows if r["has_time_intent"]], key) for key in ("old_instruction", "new_instruction")},
               "recovered": [r["qid"] for r in rows if rank(r, "old_instruction") is None and rank(r, "new_instruction") is not None],
               "lost": [r["qid"] for r in rows if rank(r, "old_instruction") is not None and rank(r, "new_instruction") is None]}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(run())
