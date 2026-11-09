"""Use ten chat-model calls to sample time-hint extraction on v4 questions."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.config import settings

MILVUS_URI = (
    settings.milvus_uri
    if "://" in settings.milvus_uri
    else f"http://{settings.milvus_uri}"
)
os.environ["MILVUS_URI"] = MILVUS_URI

from agent.model import chat_model
from agent.prompts.chat import CHAT_PROMPT
from agent.tools.search_room_messages import search_messages
from domain.time_hint import TimeHint
from infra.rag.message_retriever import (
    rerank_chat_chunks,
    select_rerank_candidates,
    temporal_order,
)
from llama_index.core.schema import NodeWithScore, TextNode


DATA_DIR = Path(__file__).parent
QUESTIONS_PATH = DATA_DIR / "queries.jsonl"
CHUNKS_PATH = DATA_DIR / "child_chunks.jsonl"
ABLATION_PATH = max(DATA_DIR.glob("time_ablation_*.json"), key=lambda path: path.stat().st_mtime)
OUTPUT_PATH = DATA_DIR / "llm_time_sample.json"
SAMPLE_QIDS = [
    "time_01_1", "time_11_2", "time_13_3", "time_14_4", "time_19_1",
    "time_02_5", "time_12_5", "time_15_5", "time_17_5", "time_20_5",
]


def exact_hint(expected: dict[str, Any], actual: dict[str, Any] | None) -> bool:
    if actual is None or expected.get("mode") != actual.get("mode"):
        return False
    if expected["mode"] == "latest":
        return True
    return all(
        int(datetime.fromisoformat(expected[field]).timestamp())
        == int(datetime.fromisoformat(actual[field]).timestamp())
        for field in ("start_at", "end_at")
    )


def hit_rank(target: str, ids: list[str]) -> int | None:
    try:
        return ids.index(target) + 1
    except ValueError:
        return None


async def run() -> None:
    query_rows = {row["qid"]: row for row in (
        json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
    )}
    chunk_rows = {
        row["id"]: row for row in (
            json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        )
    }
    ablation = json.loads(ABLATION_PATH.read_text(encoding="utf-8"))
    retrieved = {row["qid"]: row for row in ablation["temporal_records"]}
    system = CHAT_PROMPT.format_messages()[0].content
    system += (
        "\n本轮是离线检索参数评测，只调用 search_messages，不调用消息同步工具。"
        "请保留问题主题，并按问题中的明确时间填写 time_hint。"
    )
    model = chat_model.bind_tools([search_messages])
    results = []

    for qid in SAMPLE_QIDS:
        query = query_rows[qid]
        llm_started = time.perf_counter()
        response = await model.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=query["question"])]
        )
        llm_ms = (time.perf_counter() - llm_started) * 1000
        tool_call = next(
            (call for call in response.tool_calls if call["name"] == "search_messages"),
            None,
        )
        args = tool_call["args"] if tool_call else {}
        parsed_queries = args.get("queries", []) if isinstance(args, dict) else []
        parsed = parsed_queries[0] if parsed_queries and isinstance(parsed_queries[0], dict) else {}
        query_time = parsed.get("time_hint")
        if isinstance(query_time, dict):
            try:
                query_time = TimeHint.model_validate(query_time).model_dump(mode="json")
            except Exception:
                query_time = None
        else:
            query_time = None

        base = retrieved[qid]
        candidates = [
            NodeWithScore(
                node=TextNode(
                    id_=chunk_rows[chunk_id]["id"],
                    text=chunk_rows[chunk_id]["text"],
                    metadata=chunk_rows[chunk_id]["metadata"],
                    excluded_embed_metadata_keys=list(chunk_rows[chunk_id]["metadata"]),
                    excluded_llm_metadata_keys=list(chunk_rows[chunk_id]["metadata"]),
                ),
                score=float(len(base["candidate_ids"]) - rank),
            )
            for rank, chunk_id in enumerate(base["candidate_ids"])
        ]

        ranked_on_ids = base["ranked_off_ids"]
        if query_time is not None:
            hint = TimeHint.model_validate(query_time)
            time_candidates = select_rerank_candidates(candidates, temporal_order(candidates, hint))
            reranked = await rerank_chat_chunks(
                query=query["question"],
                candidates=time_candidates,
                top_n=3,
                time_hint=hint,
            )
            ranked_on_ids = [item.node.node_id for item in reranked]

        gold = query["gold_evidence_groups"][0]["child_id"]
        baseline_rank = hit_rank(gold, base["ranked_off_ids"])
        model_rank = hit_rank(gold, ranked_on_ids)
        results.append(
            {
                "qid": qid,
                "question": query["question"],
                "expected_time_hint": query["time_hint"],
                "actual_time_hint": query_time,
                "time_hint_exact": exact_hint(query["time_hint"], query_time),
                "llm_parse_ms": round(llm_ms, 2),
                "no_time_rank": baseline_rank,
                "llm_time_rank": model_rank,
                "llm_time_top3": ranked_on_ids,
                "tool_args": args,
            }
        )
        print(
            f"{qid}: time_hint_exact={results[-1]['time_hint_exact']} "
            f"no_time_rank={baseline_rank} llm_time_rank={model_rank}",
            flush=True,
        )

    exact_count = sum(row["time_hint_exact"] for row in results)
    baseline_hits = sum(row["no_time_rank"] is not None for row in results)
    llm_time_hits = sum(row["llm_time_rank"] is not None for row in results)
    baseline_rr = [0 if row["no_time_rank"] is None else 1 / row["no_time_rank"] for row in results]
    time_rr = [0 if row["llm_time_rank"] is None else 1 / row["llm_time_rank"] for row in results]
    summary = {
        "questions": len(results),
        "time_hint_exact": f"{exact_count}/{len(results)}",
        "time_hint_exact_accuracy": round(exact_count / len(results), 4),
        "no_time_recall_at_3": round(baseline_hits / len(results), 4),
        "llm_time_recall_at_3": round(llm_time_hits / len(results), 4),
        "no_time_mrr_at_3": round(statistics.mean(baseline_rr), 4),
        "llm_time_mrr_at_3": round(statistics.mean(time_rr), 4),
        "llm_parse_mean_ms": round(statistics.mean(row["llm_parse_ms"] for row in results), 2),
        "records": results,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(run())
