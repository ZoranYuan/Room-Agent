from pydantic import BaseModel, Field


class InterruptRequest(BaseModel):
    """描述一个需要用户确认的请求操作。"""

    # 暂无对应的操作路由，暂不输出 action。
    # action: str = Field(description="操作的稳定标识")
    description: str = Field(description="向用户说明将要执行什么操作")
    arguments: dict[str, object] = Field(
        default_factory=dict,
        description="执行该操作所需的参数，使用 JSON 对象",
    )
    question: str = Field(description="向用户提出的确认问题")


class ChatAgentOutput(BaseModel):
    """聊天 Agent 的最终输出。"""

    answer: str = Field(description="给用户展示的正常回答；需要确认时可为空字符串。")
    interrupt_request: InterruptRequest | None = Field(
        default=None,
        description="需要用户确认的操作；不需要确认时为 null。",
    )
