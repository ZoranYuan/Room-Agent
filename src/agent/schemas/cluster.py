"""聊天消息的话题分组结构。"""

from pydantic import BaseModel, Field


class MessageGroup(BaseModel):
    """一个语义分组，通过会话内消息序号定位成员。"""

    topic: str = Field(min_length=1, max_length=200)
    seqs: list[str] = Field(min_length=1)


class MessageGroups(BaseModel):
    """一个父块内的话题分组，允许忽略无检索价值的消息。"""

    groups: list[MessageGroup]
