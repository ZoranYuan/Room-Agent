"""通过本机 FastAPI/SSE 跑完整 room_chat_v3 问答集。"""

# 必须在导入配置、PyMilvus 前覆盖本次评测的 collection 与 Milvus URI。
# ruff: noqa: E402

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

os.environ["MILVUS_URI"] = os.environ.get("MILVUS_URI", "http://milvus:19530")
EVAL_RUN_ID = uuid4().hex[:10]
EVAL_COLLECTION = f"eval_room_chat_v3_{EVAL_RUN_ID}"
os.environ["MILVUS_HYBRID_COLLECTION"] = EVAL_COLLECTION

import httpx
import uvicorn
from fastapi import FastAPI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langchain_core.messages import HumanMessage
from llama_index.core.schema import NodeWithScore, TextNode
from pymilvus import AsyncMilvusClient, MilvusClient

from agent.context import AgentRuntimeContext
from config.config import settings
from domain.auth_scope import AuthScope
from domain.time_hint import TimeHint
from harness.grpc import search_room_messages as grpc_search_module
from infra.rag.message_retriever import retrieve_message_chunks, rerank_chat_chunks
from infra.rag.vector.vector_store import close_vector_store, get_vector_index
from transport.grpc.proto import room_agent_pb2

DATA_DIR = Path(__file__).parent
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
CHILD_CHUNKS_PATH = DATA_DIR / "child_chunks.jsonl"
CHILD_HASH_PATH = DATA_DIR / "child_chunks.corpus.sha256"
RESULT_PATH = DATA_DIR / f"fastapi_agent_final_{EVAL_RUN_ID}.json"
ROOM_ID = "eval-room-v3"
PORT = int(os.environ.get("RAG_EVAL_PORT", "8765"))

active_trace: dict[str, Any] | None = None
request_number = 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sorted(values)[max(0, int(len(values) * 0.95 + 0.999) - 1)], 3)


def percentile99(values: list[float]) -> float | None:
    """使用 nearest-rank 计算样本 P99。"""
    if not values:
        return None
    return round(sorted(values)[max(0, math.ceil(len(values) * 0.99) - 1)], 3)


def normalize(text: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:'\"《》【】（）()\[\]{}]", "", text).lower()


def expected_nodes(
    query: dict[str, Any], message_to_seq: dict[str, str], seq_to_nodes: dict[str, set[str]]
) -> list[set[str]]:
    result = []
    for group in query["gold_evidence_groups"]:
        targets: set[str] = set()
        for message_id in group["message_ids"]:
            targets.update(seq_to_nodes.get(message_to_seq[message_id], set()))
        result.append(targets)
    return result


def first_rank(target: set[str], ranked: list[str]) -> int | None:
    return next((i for i, node_id in enumerate(ranked, 1) if node_id in target), None)


def answer_clause_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    clauses = [(record, part.strip()) for record in records
               for part in re.split(r"[；;。\n]+", record["reference_answer"])
               if part.strip()]
    hits = sum(normalize(part) in normalize(record["answer_text"]) for record, part in clauses)
    full = sum(
        bool(record["reference_answer"])
        and all(normalize(part.strip()) in normalize(record["answer_text"])
                for part in re.split(r"[；;。\n]+", record["reference_answer"]) if part.strip())
        for record in records
    )
    return {
        "reference_clauses_hit": hits,
        "reference_clauses_total": len(clauses),
        "reference_clause_match_rate": round(hits / len(clauses), 4) if clauses else None,
        "all_reference_clauses_matched": full,
        "all_reference_clause_rate": round(full / len(records), 4) if records else None,
    }


async def local_search_messages(
    *, query: str, auth_scope: AuthScope, time_hint: TimeHint | None = None,
    topk: int = settings.rag_candidate_top_k,
) -> dict[str, object]:
    """保留生产检索及 reranker，只把召回父块映射到本地合成 IM 消息。"""
    started = time.perf_counter()
    hits = await retrieve_message_chunks(
        query=query, room_id=ROOM_ID, time_hint=time_hint, topk=topk
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    child_ids = [hit.node.node_id for hit in hits]
    parent_ids = list(dict.fromkeys(str(hit.node.metadata["parent_id"]) for hit in hits))

    if active_trace is not None:
        active_trace["search_calls"].append({
            "query": query,
            "time_hint": time_hint.model_dump(mode="json") if time_hint else None,
            "child_ids": child_ids,
            "parent_ids": parent_ids,
            "retrieval_ms": round(elapsed_ms, 3),
        })

    rows = [row for parent_id in parent_ids for row in parent_messages.get(parent_id, [])]
    # 同一父块可能因多个子块命中，只向模型返回一份原始消息。
    unique = {str(row["message_id"]): row for row in rows}
    messages = sorted(unique.values(), key=lambda row: int(row["seq"]))
    return {
        "status": "success",
        "messages": messages,
        "extra_message": "" if messages else "当前房间的索引中未检索到相关消息。",
    }


def build_parent_messages(corpus: list[dict[str, Any]], children: list[TextNode]) -> dict[str, list[dict[str, Any]]]:
    """按索引子块携带的 parent_id 重建该父块对应的连续消息范围。"""
    seqs_by_parent: dict[str, set[int]] = defaultdict(set)
    for child in children:
        seqs_by_parent[str(child.metadata["parent_id"])].update(
            int(seq) for seq in child.metadata["seqs"]
        )

    # 父块范围包含被 LLM 主题聚类忽略的消息；用父块 ID 序号和生产设置重建范围。
    batches: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    size = 0
    for row in sorted(corpus, key=lambda item: int(item["seq"])):
        length = len(f"[seq={row['seq']}, send_time={row['send_time']}] [{row['sender_name']} ({row['sender_id']})]: {row['content']}")
        if batch and (len(batch) >= settings.rag_parent_message_count or size + length > settings.rag_parent_max_chars):
            batches.append(batch)
            batch, size = [], 0
        batch.append(row)
        size += length
    if batch:
        batches.append(batch)

    result: dict[str, list[dict[str, Any]]] = {}
    for index, messages in enumerate(batches, start=1):
        parent_id = f"eval-v3-parent-{index:04d}"
        result[parent_id] = messages
        # 检查缓存中的子块确实指向当前重建的父块范围。
        if parent_id in seqs_by_parent and not seqs_by_parent[parent_id].issubset(
            {int(row["seq"]) for row in messages}
        ):
            raise RuntimeError(f"子块父块映射不匹配：{parent_id}")
    return result


async def main() -> None:
    global active_trace, request_number, parent_messages

    corpus_bytes = CORPUS_PATH.read_bytes()
    corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()
    if CHILD_HASH_PATH.read_text(encoding="utf-8").strip() != corpus_hash:
        raise RuntimeError("child_chunks 缓存与 corpus 哈希不一致")
    corpus = read_jsonl(CORPUS_PATH)
    queries = read_jsonl(QUERIES_PATH)
    test_limit = int(os.environ.get("RAG_EVAL_LIMIT", "0"))
    if test_limit:
        queries = queries[:test_limit]
    children = [
        TextNode(id_=row["id"], text=row["text"], metadata=row["metadata"],
                 excluded_embed_metadata_keys=list(row["metadata"]),
                 excluded_llm_metadata_keys=list(row["metadata"]))
        for row in read_jsonl(CHILD_CHUNKS_PATH)
    ]
    parent_messages = build_parent_messages(corpus, children)
    seq_to_nodes: dict[str, set[str]] = defaultdict(set)
    for child in children:
        for seq in child.metadata["seqs"]:
            seq_to_nodes[str(seq)].add(child.node_id)
    message_to_seq = {str(row["message_id"]): str(row["seq"]) for row in corpus}

    # 使用生产 Retriever 的 collection 初始化方式和索引参数，避免污染正式 collection。
    client = MilvusClient(uri=os.environ["MILVUS_URI"])
    collection = settings.milvus_collection
    if client.has_collection(collection):
        raise RuntimeError(f"评测专用 collection 已存在，停止以避免覆盖：{collection}")
    # Milvus was cold when first contacted; warm the async gRPC channel with a
    # longer connection deadline before LlamaIndex uses its default deadline.
    warmup_client = AsyncMilvusClient(uri=os.environ["MILVUS_URI"], timeout=30)
    await warmup_client.list_collections()
    await warmup_client.close()
    index = await get_vector_index()
    await index.ainsert_nodes(children)
    print(f"Indexed {len(children)} child chunks into temporary collection {collection}.", flush=True)

    # 排除本地模型加载、Milvus 首次连接和模型服务首次连接对样本 P99 的污染。
    await retrieve_message_chunks(query="评测预热", room_id=ROOM_ID)
    await rerank_chat_chunks(
        query="评测预热",
        candidates=[NodeWithScore(node=children[0], score=0.0)],
        top_n=1,
    )
    from agent.model import chat_model

    async for _chunk in chat_model.astream(
        [HumanMessage(content="请只回复：OK")]
    ):
        pass

    # 仅替换 IM 的消息来源；搜索工具仍调用生产混合检索与 Cross-Encoder。
    grpc_search_module.search_room_messages = local_search_messages

    import agent.tools.sync_room_messages as sync_tool_module

    @tool("sync_room_messages")
    async def local_sync_room_messages(runtime: ToolRuntime[AgentRuntimeContext]) -> dict[str, object]:
        """评测适配器：合成语料没有在线增量消息。"""
        return {"status": "success", "messages": [], "extra_message": "当前没有新增消息。"}

    sync_tool_module.sync_room_messages = local_sync_room_messages

    from transport.http.routers import agent as agent_router_module
    from transport.http.app import app

    async def fake_verify(*, token: str, room_id: str) -> Any:
        global request_number
        request_number += 1
        return room_agent_pb2.VerifyRoomAccessResponse(thread_id=f"eval-v3-{request_number}")

    agent_router_module.verify_room_access = fake_verify

    # FastAPI 路由与 SSE 协议走真实 Uvicorn HTTP 服务。
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="critical"))
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        if server_task.done():
            await server_task
            raise RuntimeError("评测 FastAPI 服务未能启动")
        await asyncio.sleep(0.1)

    records: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as http:
            for position, query in enumerate(queries, start=1):
                active_trace = {"search_calls": []}
                started = time.perf_counter()
                first_token_ms: float | None = None
                tokens: list[str] = []
                event_names: list[str] = []
                tool_names: list[str] = []
                http_status: int | None = None
                error: str | None = None

                try:
                    async with http.stream(
                        "POST",
                        f"http://127.0.0.1:{PORT}/api/v1/rooms/{ROOM_ID}/agent",
                        cookies={"access_token": "evaluation-only"},
                        json={"task": "chat", "question": query["question"]},
                    ) as response:
                        http_status = response.status_code
                        if response.status_code != 200:
                            error = (await response.aread()).decode("utf-8", "replace")
                        else:
                            event_name = ""
                            async for line in response.aiter_lines():
                                if line.startswith("event: "):
                                    event_name = line[7:]
                                    event_names.append(event_name)
                                elif line.startswith("data: "):
                                    try:
                                        data = json.loads(line[6:])
                                    except json.JSONDecodeError:
                                        continue
                                    if event_name == "token":
                                        if first_token_ms is None:
                                            first_token_ms = (time.perf_counter() - started) * 1000
                                        tokens.append(str(data.get("content", "")))
                                    elif event_name == "error":
                                        error = str(data.get("message", "Agent error"))
                                    elif event_name == "tool_start":
                                        tool_names.append(str(data.get("name", "")))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"

                answer_text = "".join(tokens).strip()
                call_records = list(active_trace["search_calls"])
                ordered_ids: list[str] = []
                for call in call_records:
                    ordered_ids.extend(call["child_ids"])
                unique_ids = list(dict.fromkeys(ordered_ids))
                targets = expected_nodes(query, message_to_seq, seq_to_nodes)
                target_hit_ranks = [first_rank(group, unique_ids) for group in targets]
                record = {
                    "qid": query["qid"],
                    "category": query["category"],
                    "question": query["question"],
                    "answerable": query["answerable"],
                    "reference_answer": query["answer"],
                    "answer_text": answer_text,
                    "http_status": http_status,
                    "error": error,
                    "events": event_names,
                    "tool_names": tool_names,
                    "search_tool_calls": tool_names.count("search_messages"),
                    "search_calls": call_records,
                    "unique_reranked_child_ids": unique_ids,
                    "target_groups": [sorted(group) for group in targets],
                    "target_hit_ranks": target_hit_ranks,
                    "first_token_ms": round(first_token_ms, 3) if first_token_ms is not None else None,
                    "total_ms": round((time.perf_counter() - started) * 1000, 3),
                    "time_hint_exact": None,
                }
                if query.get("time_hint"):
                    expected_hint = TimeHint.model_validate(query["time_hint"])
                    actual_hints = [call["time_hint"] for call in call_records if call["time_hint"]]
                    record["time_hint_exact"] = any(
                        TimeHint.model_validate(hint).model_dump(mode="json")
                        == expected_hint.model_dump(mode="json")
                        for hint in actual_hints
                    )
                records.append(record)
                RESULT_PATH.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
                status = "ERROR" if error else "ok"
                print(f"[{position}/{len(queries)}] {query['qid']} {status}; searches={len(call_records)}; elapsed={(record['total_ms'] / 1000):.1f}s", flush=True)
                if position == 1 and error:
                    raise RuntimeError(f"首题预跑失败，停止全量评测：{error}")
    finally:
        active_trace = None
        server.should_exit = True
        await server_task
        await close_vector_store()
        client.close()
        cleanup_client = MilvusClient(uri=os.environ["MILVUS_URI"])
        if cleanup_client.has_collection(collection):
            cleanup_client.drop_collection(collection)
        cleanup_client.close()

    succeeded = [row for row in records if not row["error"]]
    completed_answerable = [row for row in succeeded if row["answerable"]]
    single = [row for row in records if row["category"].startswith("single_block:")]
    multi = [row for row in records if row["category"].startswith("multi_block:")]
    temporal = [row for row in records if row["category"].startswith("temporal:")]
    no_answer = [row for row in records if not row["answerable"]]

    single_ranks = []
    for row in single:
        targets = row["target_groups"][0] if row["target_groups"] else []
        ranks = [
            first_rank(set(targets), call["child_ids"][:3])
            for call in row["search_calls"]
        ]
        matched_ranks = [rank for rank in ranks if rank is not None]
        single_ranks.append(min(matched_ranks) if matched_ranks else None)

    def multi_counts() -> tuple[int, int]:
        hits = complete = 0
        for row in multi:
            # Each search_calls item is one independent query result from the batch tool.
            ranked = {
                node_id
                for call in row["search_calls"]
                for node_id in call["child_ids"][:3]
            }
            found = [
                any(node_id in target for node_id in ranked)
                for target in row["target_groups"]
            ]
            hits += sum(found)
            complete += int(bool(found) and all(found))
        return hits, complete

    evidence_hits, complete_evidence_questions = multi_counts()
    refusal_pattern = re.compile(r"没(?:有)?(?:查到|找到|提到)|未(?:查到|找到|提及)|没有相关|无法(?:确认|判断)|消息中没有|记录中没有")
    completed_no_answer = [row for row in no_answer if not row["error"]]
    refusal_hits = sum(bool(refusal_pattern.search(row["answer_text"])) for row in completed_no_answer)
    latency = [row["total_ms"] for row in succeeded]
    first_tokens = [row["first_token_ms"] for row in succeeded if row["first_token_ms"] is not None]
    search_latencies = [call["retrieval_ms"] for row in records for call in row["search_calls"]]
    # 模型与检索连接已在样本外预热，此处所有查询都属于 warm latency。
    warm_search_latencies = search_latencies
    error_counts: dict[str, int] = {}
    for row in records:
        if row["error"]:
            error_counts[row["error"]] = error_counts.get(row["error"], 0) + 1
    all_expected_clauses = sum(
        len([part for part in re.split(r"[；;。\n]+", row["reference_answer"]) if part.strip()])
        for row in records if row["answerable"]
    )
    matched_answer_clauses = answer_clause_metrics(completed_answerable)
    summary = {
        "dataset": "room_chat_v3",
        "corpus_messages": len(corpus),
        "child_chunks": len(children),
        "question_count": len(queries),
        "successful_requests": len(succeeded),
        "failed_requests": len(records) - len(succeeded),
        "request_errors": error_counts,
        "single_evidence": {
            "questions": len(single),
            "search_executed": sum(bool(row["search_calls"]) for row in single),
            "recall_at_1": round(sum(rank == 1 for rank in single_ranks) / len(single), 4) if single else None,
            "recall_at_3": round(sum(rank is not None for rank in single_ranks) / len(single), 4) if single else None,
            "mrr_at_3": round(statistics.mean(0 if rank is None else 1 / rank for rank in single_ranks), 4) if single else None,
        },
        "multi_evidence": {
            "questions": len(multi),
            "evidence_total": sum(len(row["target_groups"]) for row in multi),
            "evidence_recall_at_final_3_per_query": round(evidence_hits / sum(len(row["target_groups"]) for row in multi), 4) if multi else None,
            "complete_evidence_rate": round(complete_evidence_questions / len(multi), 4) if multi else None,
            "questions_with_search": sum(bool(row["search_calls"]) for row in multi),
            "queries_per_question": round(statistics.mean(len(row["search_calls"]) for row in multi), 3) if multi else None,
            "questions_using_two_queries": sum(len(row["search_calls"]) == 2 for row in multi),
        },
        "temporal": {
            "questions": len(temporal),
            "search_executed": sum(bool(row["search_calls"]) for row in temporal),
            "time_hint_exact": sum(row["time_hint_exact"] is True for row in temporal),
            "time_hint_questions": sum(row["time_hint_exact"] is not None for row in temporal),
        },
        "no_answer": {
            "questions": len(no_answer),
            "completed_responses": len(completed_no_answer),
            "refusal_pattern_hits": refusal_hits,
            "refusal_pattern_rate_on_completed": round(refusal_hits / len(completed_no_answer), 4) if completed_no_answer else None,
        },
        "answerable_reference_phrase_match": {
            **matched_answer_clauses,
            "completed_answerable_questions": len(completed_answerable),
            "all_dataset_reference_clauses": all_expected_clauses,
            "matched_clause_lower_bound_over_full_set": round(
                matched_answer_clauses["reference_clauses_hit"] / all_expected_clauses,
                4,
            ) if all_expected_clauses else None,
        },
        "tool_calls": {
            "search_total": sum(row["search_tool_calls"] for row in records),
            "search_mean_per_question": round(statistics.mean(row["search_tool_calls"] for row in records), 3) if records else None,
            "search_query_total": sum(len(row["search_calls"]) for row in records),
            "sync_total": sum(name == "sync_room_messages" for row in records for name in row["tool_names"]),
            "tool_limit_reached_outputs": sum("Tool call limit reached" in row["answer_text"] for row in records),
        },
        "latency_ms": {
            "end_to_end_mean": round(statistics.mean(latency), 3) if latency else None,
            "end_to_end_p95": percentile95(latency),
            "first_token_mean": round(statistics.mean(first_tokens), 3) if first_tokens else None,
            "first_token_p95": percentile95(first_tokens),
            "first_token_p99": percentile99(first_tokens),
            "retrieval_calls": len(search_latencies),
            "retrieval_and_rerank_mean": round(statistics.mean(search_latencies), 3) if search_latencies else None,
            "retrieval_and_rerank_p95": percentile95(search_latencies),
            "retrieval_and_rerank_p99": percentile99(search_latencies),
            "warm_retrieval_and_rerank_mean": round(statistics.mean(warm_search_latencies), 3) if warm_search_latencies else None,
            "warm_retrieval_and_rerank_p95": percentile95(warm_search_latencies),
        },
        "settings": {
            "collection": collection,
            "embedding_model": settings.embedding_model_id,
            "reranker_model": settings.reranker_model_id,
            "retrieval_final_top_n": settings.rag_final_top_n,
            "rerank_input_top_n": settings.rag_rerank_input_top_n,
            "candidate_top_k": settings.rag_candidate_top_k,
            "rrf_k": settings.rag_rrf_k,
            "agent_recursion_limit": settings.agent_recursion_limit,
        },
        "records": records,
    }
    RESULT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFINAL METRICS", flush=True)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {RESULT_PATH}", flush=True)


parent_messages: dict[str, list[dict[str, Any]]] = {}

if __name__ == "__main__":
    asyncio.run(main())
