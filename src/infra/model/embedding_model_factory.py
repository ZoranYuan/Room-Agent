"""延迟加载 ModelScope 模型，提供 LlamaIndex Embedding 实例."""

from functools import lru_cache

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from modelscope import snapshot_download

from config.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> BaseEmbedding:
    """首次使用时下载或读取缓存模型，后续复用实例."""
    model_path = snapshot_download(settings.embedding_model_id)
    return HuggingFaceEmbedding(
        model_name=model_path,
        normalize=True,
        truncate_dim=settings.embedding_dimension,
        embed_batch_size=8,
        query_instruction=(
            "指令：根据用户提出的聊天室相关问题，检索相关的聊天记录。\n" "查询："
        ),
    )
