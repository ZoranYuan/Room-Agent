"""用于消息检索排序的时间意图。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TimeHint(BaseModel):
    """由聊天模型提取的检索时间偏好。"""

    mode: Literal["range", "latest"] = Field(
        description="range 表示指定时间范围；latest 表示寻找最近或最后一次相关讨论。"
    )
    start_at: datetime | None = Field(
        default=None,
        description="range 的开始时间，ISO 8601 格式并包含时区偏移。",
    )
    end_at: datetime | None = Field(
        default=None,
        description="range 的结束时间，ISO 8601 格式并包含时区偏移。",
    )

    @model_validator(mode="after")
    def validate_range(self) -> "TimeHint":
        if self.mode == "range":
            if self.start_at is None or self.end_at is None:
                raise ValueError("range 时间意图必须提供 start_at 和 end_at")
            if self.start_at.utcoffset() is None or self.end_at.utcoffset() is None:
                raise ValueError("时间范围必须包含时区偏移")
            if self.start_at > self.end_at:
                raise ValueError("start_at 不能晚于 end_at")
        elif self.start_at is not None or self.end_at is not None:
            raise ValueError("latest 时间意图不应提供起止时间")

        return self

