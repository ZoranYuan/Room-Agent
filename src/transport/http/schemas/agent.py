"""Room Agent HTTP 请求 DTO."""

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, StringConstraints

NonEmptyQuestion = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SummaryTaskRequest(BaseModel):
    """生成 Room 未读消息摘要的请求."""

    task: Literal["summary"]


class ChatTaskRequest(BaseModel):
    """向 Room Agent 提问的请求."""

    task: Literal["chat"]
    question: NonEmptyQuestion


AgentTaskRequest: TypeAlias = Annotated[
    SummaryTaskRequest | ChatTaskRequest,
    Field(discriminator="task"),
]
