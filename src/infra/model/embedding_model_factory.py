"""预留本地 Embedding 模型入口，具体模型尚未接入."""

from langchain_core.embeddings import Embeddings


class EmbeddingModel(Embeddings):
    """Embedding 模型占位定义，检索与向量化能力尚未实现."""

    def __init__(self) -> None:
        """预留模型初始化逻辑."""
