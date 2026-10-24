"""Room 消息访问范围领域对象."""


from pydantic import BaseModel


class AuthorizedRoomScope(BaseModel):
    """已经通过访问校验的用户 Room 范围."""

    room_id: str
    user_id: str
