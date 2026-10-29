"""Room 消息访问范围领域对象."""


from pydantic import BaseModel, Field, SecretStr


class AuthScope(BaseModel):
    """已经通过访问校验的用户 Room 范围."""

    room_id: str
    token: SecretStr = Field(exclude=True, repr=False)
