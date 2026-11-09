"""Reuse archived raw chunks and evaluate independent questions without LLM calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

print("loading retrieval modules; no clustering or query LLM calls", flush=True)
from config.config import settings

uri = settings.milvus_uri
os.environ["MILVUS_URI"] = uri if "://" in uri else f"http://{uri}"

from domain.time_hint import TimeHint
from infra.model.embedding_model_factory import get_embedding_model
from infra.model.reranker_model_factory import get_message_reranker
from infra.rag.message_retriever import rerank_chat_chunks, select_rerank_candidates, temporal_order
from infra.rag.vector import vector_store
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.vector_stores.types import VectorStoreQueryMode

DATA = Path(__file__).parent
ARCHIVE = DATA / "raw_chunk_eval"
OUT = ARCHIVE / "single_k20_l10_r5_n3"
K, L, RERANK_INPUT, N = 20, 10, 5, 3


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(records, mode):
    hits = []
    reciprocal = []
    for row in records:
        targets = set(row["acceptable_child_ids"])
        rank = next((i for i, hit in enumerate(row[mode]["top3"], 1) if hit["id"] in targets), None)
        hits.append(int(rank is not None))
        reciprocal.append(1 / rank if rank else 0)
    return {"questions": len(records), "hits": sum(hits),
            "recall_at_3": statistics.mean(hits) if hits else 0,
            "mrr_at_3": statistics.mean(reciprocal) if reciprocal else 0}


async def run():
    OUT.mkdir(exist_ok=True)
    questions = [q for q in read_jsonl(DATA / "queries.jsonl") if len(q["gold_evidence_groups"]) == 1]
    mapping = json.loads((ARCHIVE / "evidence_mapping.json").read_text())["questions"]
    nodes = []
    for row in read_jsonl(ARCHIVE / "child_chunks.jsonl"):
        nodes.append(TextNode(id_=row["id"], text=row["text"], metadata=row["metadata"],
                              excluded_embed_metadata_keys=list(row["metadata"]),
                              excluded_llm_metadata_keys=list(row["metadata"])))
    parameters = {"dense_k": K, "bm25_k": K, "rrf_keep_l": L,
                  "rerank_input": RERANK_INPUT, "final_n": N,
                  "rrf_smoothing_k": settings.rag_rrf_k,
                  "time_weight": settings.rag_temporal_rank_weight,
                  "embedding": settings.embedding_model_id, "reranker": settings.reranker_model_id,
                  "device": settings.rag_device, "questions": len(questions), "children": len(nodes),
                  "source_chunks_sha256": hashlib.sha256((ARCHIVE / "child_chunks.jsonl").read_bytes()).hexdigest(),
                  "llm_requests": 0}
    (OUT / "config.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2))
    settings.rag_rerank_input_top_n = RERANK_INPUT
    settings.milvus_collection = "eval_v5_single_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"reuse {len(nodes)} archived children; {len(questions)} independent questions; {K}/{K}->{L}->{RERANK_INPUT}->{N}", flush=True)
    try:
        embedding = await get_embedding_model()
        index = await vector_store.get_vector_index()
        for start in range(0, len(nodes), 32):
            batch = nodes[start:start+32]
            vectors = await asyncio.to_thread(embedding.get_text_embedding_batch,
                                            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in batch])
            for node, vector in zip(batch, vectors):
                node.embedding = vector
            await index.ainsert_nodes(batch)
            for node in batch:
                node.embedding = None
            if start % 320 == 0 or start+32 >= len(nodes):
                print(f"indexed {min(start+32,len(nodes))}/{len(nodes)}", flush=True)
        await get_message_reranker()
        # The adapter uses similarity_top_k for both ANN requests and the RRF
        # response. Taking its first L results preserves the requested RRF TopL.
        retriever = index.as_retriever(similarity_top_k=K,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            filters=MetadataFilters(filters=[ExactMatchFilter(key="room_id", value="eval-room-v5")]))
        records = []
        with (OUT / "results.jsonl").open("w") as output:
            for position, question in enumerate(questions, 1):
                started = time.perf_counter()
                pool = await retriever.aretrieve(question["question"])
                candidates = pool[:L]
                hint = TimeHint.model_validate(question["time_hint"]) if question.get("time_hint") else None
                row = {"qid": question["qid"], "question": question["question"],
                       "has_time_intent": question["has_time_intent"],
                       "acceptable_child_ids": mapping[question["qid"]][0]["acceptable_child_ids"],
                       "rrf_top20": [hit.node.node_id for hit in pool],
                       "rrf_top10": [hit.node.node_id for hit in candidates]}
                for mode, mode_hint in (("time_off", None), ("time_on", hint)):
                    if mode == "time_on" and hint is None:
                        row[mode] = row["time_off"]
                        continue
                    selected = (select_rerank_candidates(candidates, temporal_order(candidates, mode_hint))
                                if mode_hint is not None else candidates[:RERANK_INPUT])
                    ranked = await rerank_chat_chunks(query=question["question"], candidates=selected, top_n=N, time_hint=mode_hint)
                    assert len(ranked) == N
                    row[mode] = {"rerank_input_ids": [hit.node.node_id for hit in selected],
                                 "top3": [{"id": hit.node.node_id, "text": hit.node.text,
                                           "metadata": hit.node.metadata, "score": hit.score} for hit in ranked]}
                row["both_conditions_elapsed_ms"] = (time.perf_counter()-started)*1000
                records.append(row)
                output.write(json.dumps(row, ensure_ascii=False)+"\n")
                output.flush()
                if position % 10 == 0:
                    print(f"scored {position}/{len(questions)}", flush=True)
        summary = {"parameters": parameters,
                   "overall": {mode: summarize(records, mode) for mode in ("time_off", "time_on")},
                   "temporal": {mode: summarize([r for r in records if r["has_time_intent"]], mode) for mode in ("time_off", "time_on")},
                   "non_temporal": summarize([r for r in records if not r["has_time_intent"]], "time_off")}
        (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        store = vector_store._vector_store
        if store is not None:
            try:
                await asyncio.to_thread(store.client.drop_collection, collection_name=settings.milvus_collection)
                print("temporary collection removed", flush=True)
            finally:
                await vector_store.close_vector_store()


if __name__ == "__main__":
    asyncio.run(run())
