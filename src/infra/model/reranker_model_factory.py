"""加载并复用聊天记录重排模型。"""

import asyncio

from sentence_transformers import CrossEncoder

from config.config import settings

_reranker: CrossEncoder | None = None

CHAT_HISTORY_INSTRUCTION = (
    "Given a question about a chat room's history, judge whether the passage "
    "directly provides evidence answering the query. All explicit constraints, "
    "including the requested time range, must be satisfied. A passage outside "
    "the requested time range is not relevant, even if it discusses the same "
    "topic. Merely mentioning the topic or stating that a conclusion was "
    "recorded does not provide sufficient evidence."
)


async def get_message_reranker() -> CrossEncoder:
    """首次初始化，后续在当前服务进程内复用。"""
    global _reranker

    if _reranker is None:
        _reranker = await asyncio.to_thread(
            CrossEncoder,
            settings.reranker_model_id,
            device=settings.rag_device,
            max_length=512,
            trust_remote_code=True,
            prompts={"chat_history": CHAT_HISTORY_INSTRUCTION},
            default_prompt_name="chat_history",
        )
    return _reranker
