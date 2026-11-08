"""Prometheus 指标。"""

from prometheus_client import Histogram

_LATENCY_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
)

RAG_RETRIEVAL_DURATION = Histogram(
    "room_agent_rag_retrieval_duration_seconds",
    "Query embedding, hybrid retrieval, temporal ordering, and reranking duration.",
    labelnames=("outcome",),
    buckets=_LATENCY_BUCKETS,
)

AGENT_FIRST_TOKEN_DURATION = Histogram(
    "room_agent_first_token_duration_seconds",
    "Time from HTTP request handling start to the first SSE answer token.",
    labelnames=("task",),
    buckets=_LATENCY_BUCKETS,
)
