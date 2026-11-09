"""Run query decomposition and retrieval evaluation without generating answers."""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from config.config import settings

MILVUS_URI = settings.milvus_uri
if "://" not in MILVUS_URI:
    MILVUS_URI = f"http://{MILVUS_URI}"
os.environ["MILVUS_URI"] = MILVUS_URI

from agent.model import chat_model
from agent.prompts.chat import CHAT_PROMPT
from domain.time_hint import TimeHint
from infra.model.embedding_model_factory import get_embedding_model
from infra.rag.message_retriever import (
    rerank_chat_chunks,
    select_rerank_candidates,
    temporal_order,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from pymilvus import MilvusClient

DATA_DIR = Path(__file__).parent
RESULT_PATH = DATA_DIR / "subquestion_retrieval_eval.json"
COLLECTION = f"eval_v5_{uuid4().hex[:10]}"
ROOM_ID = "eval-room-v5"


class SearchQuery(BaseModel):
    query: str = Field(min_length=1, description="完整、独立的房间历史检索问题。")
    time_hint: TimeHint | None = Field(default=None)


@tool
def search_messages(
    queries: Annotated[list[SearchQuery], Field(min_length=1, max_length=2)],
) -> dict[str, str]:
    """按子问题检索房间历史；本离线评测只读取工具参数，不执行工具。"""
    return {"status": "offline"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return round(sorted(values)[max(0, math.ceil(len(values) * p) - 1)], 2)


def parse_hint(raw: object) -> TimeHint | None:
    if raw is None:
        return None
    try:
        return TimeHint.model_validate(raw)
    except (ValidationError, TypeError, ValueError):
        return None


def hint_matches(expected: dict[str, Any], actual: TimeHint | None) -> bool:
    if actual is None or expected.get("mode") != actual.mode:
        return False
    if actual.mode == "latest":
        return True
    expected_start = datetime.fromisoformat(expected["start_at"])
    expected_end = datetime.fromisoformat(expected["end_at"])
    assert actual.start_at is not None and actual.end_at is not None
    # Day ranges may end at 23:59:59 on the model side and 23:59:59.999999
    # in the dataset; compare at the precision supported by the model output.
    actual_start = actual.start_at.replace(microsecond=0)
    actual_end = actual.end_at.replace(microsecond=0)
    expected_start = expected_start.replace(microsecond=0)
    expected_end = expected_end.replace(microsecond=0)
    return actual_start <= expected_start and actual_end >= expected_end


def ids(nodes: list[NodeWithScore]) -> list[str]:
    return [node.node.node_id for node in nodes]


def evidence_recall(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    scores = []
    for row in rows:
        gold = set(row["gold_child_ids"])
        found = set(row[key])
        scores.append(len(gold & found) / len(gold))
    return round(statistics.mean(scores), 4)


async def run() -> None:
    queries = read_jsonl(DATA_DIR / "queries.jsonl")
    chunk_rows = read_jsonl(DATA_DIR / "child_chunks.jsonl")
    nodes = [
        TextNode(
            id_=row["id"],
            text=row["text"],
            metadata=row["metadata"],
            excluded_embed_metadata_keys=list(row["metadata"]),
            excluded_llm_metadata_keys=list(row["metadata"]),
        )
        for row in chunk_rows
    ]
    client = MilvusClient(uri=MILVUS_URI)
    if client.has_collection(COLLECTION):
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

        system_prompt = CHAT_PROMPT.format_messages()[0].content
        system_prompt += (
            "\n当前是离线检索评测：只需分析检索子问题，不要生成面向用户的回答。"
        )
        now = datetime.now(ZoneInfo(settings.rag_timezone))
        time_context = SystemMessage(
            content=(
                f"本次请求的当前时间为 {now.isoformat()}，时区为 {settings.rag_timezone}。"
                f"当前年份为 {now.year}；年份换算以此为准：去年是 {now.year - 1} 年，"
                f"今年是 {now.year} 年，明年是 {now.year + 1} 年。"
                f"例如，去年11月15日对应 {now.year - 1}-11-15。"
                "不要使用模型自身的当前日期或知识截止时间推算。"
            )
        )
        parser_model = chat_model.bind_tools(
            [search_messages], tool_choice="search_messages"
        )
        records: list[dict[str, Any]] = []
        model_ms: list[float] = []
        retrieval_ms: list[float] = []
        rerank_ms: list[float] = []
        token_in = token_out = 0

        semaphore = asyncio.Semaphore(5)

        async def parse_question(question: dict[str, Any]) -> tuple[Any, float]:
            async with semaphore:
                started = time.perf_counter()
                response = await parser_model.ainvoke(
                    [
                        SystemMessage(content=system_prompt),
                        time_context,
                        HumanMessage(content=question["question"]),
                    ]
                )
                return response, (time.perf_counter() - started) * 1000

        parsed = await asyncio.gather(*(parse_question(question) for question in queries))
        print(f"decomposed {len(parsed)}/{len(queries)} questions; no answers generated", flush=True)

        for position, (question, (response, elapsed_ms)) in enumerate(
            zip(queries, parsed), start=1
        ):
            model_ms.append(elapsed_ms)
            usage = response.usage_metadata or {}
            token_in += int(usage.get("input_tokens", 0))
            token_out += int(usage.get("output_tokens", 0))
            call = next(
                (item for item in response.tool_calls if item["name"] == "search_messages"),
                None,
            )
            raw_queries = call["args"].get("queries", []) if call else []
            query_outputs: list[dict[str, Any]] = []

            for item in raw_queries[:2]:
                query_text = item.get("query") if isinstance(item, dict) else None
                if not isinstance(query_text, str) or not query_text.strip():
                    continue
                hint = parse_hint(item.get("time_hint"))
                started = time.perf_counter()
                candidates = await retriever.aretrieve(query_text)
                retrieval_ms.append((time.perf_counter() - started) * 1000)
                if not candidates:
                    query_outputs.append({
                        "query": query_text,
                        "time_hint": hint.model_dump(mode="json") if hint else None,
                        "ranked_off": [],
                        "ranked_on": [],
                    })
                    continue

                off_candidates = candidates[: settings.rag_rerank_input_top_n]
                started = time.perf_counter()
                ranked_off = await rerank_chat_chunks(
                    query=query_text,
                    candidates=off_candidates,
                    top_n=3,
                )
                rerank_ms.append((time.perf_counter() - started) * 1000)
                ranked_on = ranked_off
                if hint is not None:
                    temporal_candidates = temporal_order(candidates, hint)
                    on_candidates = select_rerank_candidates(candidates, temporal_candidates)
                    started = time.perf_counter()
                    ranked_on = await rerank_chat_chunks(
                        query=query_text,
                        candidates=on_candidates,
                        top_n=3,
                        time_hint=hint,
                    )
                    rerank_ms.append((time.perf_counter() - started) * 1000)
                query_outputs.append({
                    "query": query_text,
                    "time_hint": hint.model_dump(mode="json") if hint else None,
                    "ranked_off": ids(ranked_off),
                    "ranked_on": ids(ranked_on),
                })

            gold_ids = [
                evidence["child_id"]
                for evidence in question["gold_evidence_groups"]
            ]
            off_union = list(dict.fromkeys(
                child_id for output in query_outputs for child_id in output["ranked_off"]
            ))
            on_union = list(dict.fromkeys(
                child_id for output in query_outputs for child_id in output["ranked_on"]
            ))
            actual_hints = [output["time_hint"] for output in query_outputs if output["time_hint"]]
            expected_hint = question.get("time_hint")
            hint_ok = (
                any(hint_matches(expected_hint, TimeHint.model_validate(item)) for item in actual_hints)
                if expected_hint
                else all(item is None for item in query_outputs)
            )
            records.append({
                "qid": question["qid"],
                "kind": question["kind"],
                "question": question["question"],
                "gold_child_ids": gold_ids,
                "tool_called": call is not None,
                "subquestion_count": len(query_outputs),
                "expected_time_hint": expected_hint,
                "time_hint_correct": hint_ok if question["kind"] == "temporal" else None,
                "ranked_off_union": off_union,
                "ranked_on_union": on_union,
                "subquestions": query_outputs,
            })
            if position % 10 == 0 or position == len(queries):
                print(f"processed {position}/{len(queries)} questions", flush=True)

        temporal = [row for row in records if row["kind"] == "temporal"]
        plain = [row for row in records if row["kind"] == "plain"]
        temporal_hint_rows = [row for row in temporal if row["tool_called"]]
        result = {
            "dataset": "room_chat_v5",
            "messages": 10_000,
            "senders": 50,
            "parent_count": len(read_jsonl(DATA_DIR / "parent_chunks.jsonl")),
            "child_count": len(nodes),
            "questions": len(queries),
            "subquestion_parse": {
                "tool_call_rate": round(sum(row["tool_called"] for row in records) / len(records), 4),
                "mean_subquestions": round(statistics.mean(row["subquestion_count"] for row in records), 2),
                "temporal_time_hint_accuracy": round(
                    sum(bool(row["time_hint_correct"]) for row in temporal_hint_rows)
                    / max(1, len(temporal_hint_rows)),
                    4,
                ),
            },
            "evidence_recall_at_3": {
                "all_questions": evidence_recall(records, "ranked_off_union"),
                "without_time": evidence_recall(plain, "ranked_off_union"),
                "with_time_off": evidence_recall(temporal, "ranked_off_union"),
                "with_time_on": evidence_recall(temporal, "ranked_on_union"),
            },
            "llm_tokens": {"input": token_in, "output": token_out},
            "latency_ms": {
                "subquestion_mean": round(statistics.mean(model_ms), 2),
                "retrieval_mean": round(statistics.mean(retrieval_ms), 2) if retrieval_ms else None,
                "retrieval_p95": percentile(retrieval_ms, 0.95),
                "rerank_mean": round(statistics.mean(rerank_ms), 2) if rerank_ms else None,
            },
            "settings": {
                "parent_message_limit": 50,
                "parent_char_limit": 2_000,
                "candidate_top_k": settings.rag_candidate_top_k,
                "rerank_input_top_n": settings.rag_rerank_input_top_n,
                "final_top_n": 3,
                "rrf_k": settings.rag_rrf_k,
                "collection": COLLECTION,
            },
            "records": records,
        }
        RESULT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))
        print(f"saved {RESULT_PATH}")
    finally:
        try:
            if store.aclient is not None:
                await store.aclient.close()
        finally:
            await asyncio.to_thread(store.client.close)
            if client.has_collection(COLLECTION):
                client.drop_collection(COLLECTION)
            client.close()


if __name__ == "__main__":
    asyncio.run(run())
