"""Evaluate current raw-message chunks with strict per-question Top3 results."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import signal
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

print("loading application modules", flush=True)

from config.config import settings

uri = settings.milvus_uri
if "://" not in uri:
    uri = f"http://{uri}"
os.environ["MILVUS_URI"] = uri

from domain.chat_message import ChatMessage
from domain.time_hint import TimeHint
from infra.model.embedding_model_factory import get_embedding_model
from infra.model.reranker_model_factory import get_message_reranker
from infra.rag import message_index
from infra.rag.message_retriever import retrieve_message_chunks
from infra.rag.vector import vector_store
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode

DATA = Path(__file__).parent
ROOT = DATA.parents[1]
OUT = DATA / "raw_chunk_eval"
ROOM = "eval-room-v5"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def node_record(node: TextNode) -> dict[str, Any]:
    return {"id": node.node_id, "text": node.text, "metadata": node.metadata}


def load_node(record: dict[str, Any]) -> TextNode:
    metadata = record["metadata"]
    return TextNode(
        id_=record["id"], text=record["text"], metadata=metadata,
        excluded_embed_metadata_keys=list(metadata),
        excluded_llm_metadata_keys=list(metadata),
    )


def hit_record(hit: NodeWithScore) -> dict[str, Any]:
    return {**node_record(hit.node), "score": hit.score}


@asynccontextmanager
async def unused_session():
    yield None


class OfflineParentRepository:
    """Replace only the existing-parent lookup; retain production chunking logic."""

    def __init__(self, session):
        pass

    async def find_chunk_by_range(self, **kwargs):
        return None


def summarize(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    recalls = []
    reciprocal_ranks = []
    for row in records:
        top_ids = [item["id"] for item in row[mode]["top3"]]
        targets = [set(group["acceptable_child_ids"]) for group in row["evidence"]]
        recalls.append(sum(bool(set(top_ids) & group) for group in targets) / len(targets))
        first_rank = next(
            (rank for rank, child_id in enumerate(top_ids, 1)
             if any(child_id in group for group in targets)), None,
        )
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
    durations = [row[mode]["elapsed_ms"] for row in records]
    return {
        "questions": len(records),
        "evidence_recall_at_3": statistics.mean(recalls) if recalls else 0,
        "mrr_at_3": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0,
        "mean_ms": statistics.mean(durations) if durations else None,
        "p95_ms": sorted(durations)[math.ceil(len(durations) * 0.95) - 1] if durations else None,
    }


async def run() -> None:
    started = time.perf_counter()
    task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    OUT.mkdir(exist_ok=True)
    (OUT / "batches").mkdir(exist_ok=True)
    corpus = read_jsonl(DATA / "corpus.jsonl")
    questions = read_jsonl(DATA / "queries.jsonl")
    field_names = {field.name for field in fields(ChatMessage)}
    messages = [ChatMessage(**{key: value for key, value in row.items() if key in field_names}) for row in corpus]
    by_message_id = {row["message_id"]: row for row in corpus}
    batches = message_index._parent_batches(messages)
    config = {
        "corpus_sha256": hashlib.sha256((DATA / "corpus.jsonl").read_bytes()).hexdigest(),
        "queries_sha256": hashlib.sha256((DATA / "queries.jsonl").read_bytes()).hexdigest(),
        "cluster_model": settings.agent_model,
        "embedding_model": settings.embedding_model_id,
        "reranker_model": settings.reranker_model_id,
        "device": settings.rag_device,
        "parent_message_count": settings.rag_parent_message_count,
        "parent_max_chars": settings.rag_parent_max_chars,
        "child_max_chars": settings.rag_chunk_max_chars,
        "overlap_chars": settings.rag_chunk_overlap_chars,
        "rrf_candidates": settings.rag_candidate_top_k,
        "rerank_input": settings.rag_rerank_input_top_n,
        "final_top_n": settings.rag_final_top_n,
        "rrf_k": settings.rag_rrf_k,
        "temporal_weight": settings.rag_temporal_rank_weight,
        "sources": {},
    }
    for name in ("src/infra/rag/message_index.py", "src/agent/prompts/cluster.py",
                 "src/harness/llm/cluster_messages.py", "src/infra/rag/message_retriever.py"):
        config["sources"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert settings.rag_final_top_n == 3
    config_path = OUT / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Existing checkpoint configuration differs; use a fresh output directory.")
    save_json(config_path, config)
    print(f"data: {len(messages)} messages, {len(batches)} parents, {len(questions)} questions; device={settings.rag_device}", flush=True)

    # Evaluation does not store parent rows or call IM. Topic clustering and
    # child text construction still execute the actual application function.
    message_index.get_session_factory = lambda: unused_session
    message_index.MessageChunkRepository = OfflineParentRepository
    semaphore = asyncio.Semaphore(4)
    completed = 0

    async def build_batch(position: int) -> dict[str, Any]:
        nonlocal completed
        path = OUT / "batches" / f"{position:04d}.json"
        async with semaphore:
            if path.exists():
                result = json.loads(path.read_text())
            else:
                batch_started = time.perf_counter()
                parents, children = await message_index.build_message_chunks(
                    room_id=ROOM, messages=batches[position],
                )
                result = {
                    "parent_id": parents[0].id,
                    "seq_start": parents[0].seq_start, "seq_end": parents[0].seq_end,
                    "elapsed_seconds": time.perf_counter() - batch_started,
                    "children": [node_record(node) for node in children],
                }
                save_json(path, result)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(batches):
                print(f"cluster {completed}/{len(batches)}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
            return result

    # Check one real clustering request before scheduling the remaining batches.
    first = await build_batch(0)
    remaining = await asyncio.gather(*(build_batch(i) for i in range(1, len(batches))), return_exceptions=True)
    failures = [(i + 1, str(value)) for i, value in enumerate(remaining) if isinstance(value, BaseException)]
    if failures:
        save_json(OUT / "build_errors.json", failures)
        raise RuntimeError(f"Clustering failed for {len(failures)} batches; successful batches are checkpointed.")
    built = [first, *remaining]
    nodes = [load_node(row) for batch in built for row in batch["children"]]
    assert all(len(node.text) <= settings.rag_chunk_max_chars for node in nodes)
    with (OUT / "child_chunks.jsonl").open("w") as output:
        for node in nodes:
            output.write(json.dumps(node_record(node), ensure_ascii=False) + "\n")
    save_json(OUT / "parents.json", [{key: value for key, value in batch.items() if key != "children"} for batch in built])

    nodes_by_seq: dict[str, list[TextNode]] = {}
    for node in nodes:
        for seq in set(node.metadata["seqs"]):
            nodes_by_seq.setdefault(seq, []).append(node)
    labels = {}
    missing = []
    for question in questions:
        evidence = []
        for group in question["gold_evidence_groups"]:
            gold_messages = [by_message_id[mid] for mid in group["message_ids"]]
            assert len(gold_messages) == 1
            gold = gold_messages[0]
            targets = [node.node_id for node in nodes_by_seq.get(str(gold["seq"]), [])
                       if gold["content"] in node.text.split("\n", 1)[-1]]
            evidence.append({"group_id": group["group_id"], "message_ids": group["message_ids"],
                             "seq": gold["seq"], "answer_text": gold["content"],
                             "acceptable_child_ids": targets})
            if not targets:
                missing.append({"qid": question["qid"], "seq": gold["seq"], "answer_text": gold["content"]})
        labels[question["qid"]] = evidence
    save_json(OUT / "evidence_mapping.json", {"questions": labels, "unindexed_evidence": missing})
    print(f"chunks ready: {len(nodes)} children; missing evidence groups={len(missing)} (remain in scoring denominator)", flush=True)

    collection = "eval_v5_raw_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    settings.milvus_collection = collection
    save_json(OUT / "collection.json", {"name": collection})
    try:
        stage = time.perf_counter()
        print("loading embedding model", flush=True)
        embedding_model = await get_embedding_model()
        print(f"embedding ready: {time.perf_counter()-stage:.2f}s", flush=True)
        index = await vector_store.get_vector_index()
        embedding_ms = insert_ms = 0.0
        for start in range(0, len(nodes), 32):
            batch_nodes = nodes[start:start+32]
            texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in batch_nodes]
            stage = time.perf_counter()
            vectors = await asyncio.to_thread(embedding_model.get_text_embedding_batch, texts)
            embedding_ms += (time.perf_counter() - stage) * 1000
            for node, vector in zip(batch_nodes, vectors):
                node.embedding = vector
            stage = time.perf_counter()
            await index.ainsert_nodes(batch_nodes)
            insert_ms += (time.perf_counter() - stage) * 1000
            for node in batch_nodes:
                node.embedding = None
            print(f"indexed {min(start+32,len(nodes))}/{len(nodes)}; embedding={embedding_ms/1000:.1f}s write={insert_ms/1000:.1f}s", flush=True)
        store = vector_store._vector_store
        count = await asyncio.to_thread(store.client.query, collection_name=collection, filter="", output_fields=["count(*)"])
        assert count[0]["count(*)"] == len(nodes), count
        print(f"Milvus verified {count[0]['count(*)']} rows", flush=True)
        stage = time.perf_counter()
        print("loading reranker", flush=True)
        await get_message_reranker()
        print(f"reranker ready: {time.perf_counter()-stage:.2f}s", flush=True)
        await retrieve_message_chunks(query=questions[0]["question"], room_id=ROOM)
        records = []
        with (OUT / "results.jsonl").open("w") as output:
            for position, question in enumerate(questions, 1):
                row = {"qid": question["qid"], "question": question["question"],
                       "has_time_intent": question["has_time_intent"], "evidence": labels[question["qid"]]}
                hint = TimeHint.model_validate(question["time_hint"]) if question.get("time_hint") else None
                for mode, mode_hint in (("time_off", None), ("time_on", hint)):
                    if mode == "time_on" and hint is None:
                        row[mode] = {**row["time_off"], "reused_identical_query": True}
                        continue
                    stage = time.perf_counter()
                    hits = await retrieve_message_chunks(query=question["question"], room_id=ROOM, time_hint=mode_hint)
                    assert len(hits) <= 3
                    row[mode] = {"top3": [hit_record(hit) for hit in hits],
                                 "elapsed_ms": (time.perf_counter()-stage)*1000}
                records.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                if position % 10 == 0:
                    print(f"scored {position}/{len(questions)}", flush=True)
        summary = {
            "method": "Original question; strict final Top3; no LLM query decomposition or final answer. Exact answer text plus seq maps evidence groups to new child IDs; unindexed evidence remains a miss.",
            "messages": len(messages), "parents": len(built), "children": len(nodes),
            "unindexed_evidence_groups": len(missing),
            "index_embedding_seconds": embedding_ms/1000, "index_write_seconds": insert_ms/1000,
            "overall": {mode: summarize(records, mode) for mode in ("time_off", "time_on")},
            "temporal": {mode: summarize([row for row in records if row["has_time_intent"]], mode) for mode in ("time_off", "time_on")},
            "non_temporal": summarize([row for row in records if not row["has_time_intent"]], "time_off"),
            "total_seconds": time.perf_counter()-started,
        }
        save_json(OUT / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        store = vector_store._vector_store
        if store is not None:
            try:
                await asyncio.to_thread(store.client.drop_collection, collection_name=collection)
                print(f"removed temporary collection {collection}", flush=True)
            finally:
                await vector_store.close_vector_store()


if __name__ == "__main__":
    print("starting raw chunk evaluation imports completed", flush=True)
    asyncio.run(run())
