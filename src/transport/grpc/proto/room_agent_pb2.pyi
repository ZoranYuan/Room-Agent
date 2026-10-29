from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MessageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MESSAGE_TYPE_TEXT: _ClassVar[MessageType]

class MessageStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MESSAGE_STATUS_NORMAL: _ClassVar[MessageStatus]
    MESSAGE_STATUS_RECALLED: _ClassVar[MessageStatus]
MESSAGE_TYPE_TEXT: MessageType
MESSAGE_STATUS_NORMAL: MessageStatus
MESSAGE_STATUS_RECALLED: MessageStatus

class VerifyRoomAccessRequest(_message.Message):
    __slots__ = ("room_id",)
    ROOM_ID_FIELD_NUMBER: _ClassVar[int]
    room_id: str
    def __init__(self, room_id: _Optional[str] = ...) -> None: ...

class VerifyRoomAccessResponse(_message.Message):
    __slots__ = ("thread_id",)
    THREAD_ID_FIELD_NUMBER: _ClassVar[int]
    thread_id: str
    def __init__(self, thread_id: _Optional[str] = ...) -> None: ...

class GetRoomUnreadMessagesRequest(_message.Message):
    __slots__ = ("room_id", "limit")
    ROOM_ID_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    room_id: str
    limit: int
    def __init__(self, room_id: _Optional[str] = ..., limit: _Optional[int] = ...) -> None: ...

class GetRoomUnreadMessagesResponse(_message.Message):
    __slots__ = ("messages",)
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    def __init__(self, messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ...) -> None: ...

class SyncRoomMessagesRequest(_message.Message):
    __slots__ = ("room_id", "after_seq", "limit")
    ROOM_ID_FIELD_NUMBER: _ClassVar[int]
    AFTER_SEQ_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    room_id: str
    after_seq: int
    limit: int
    def __init__(self, room_id: _Optional[str] = ..., after_seq: _Optional[int] = ..., limit: _Optional[int] = ...) -> None: ...

class SyncRoomMessagesResponse(_message.Message):
    __slots__ = ("messages",)
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    def __init__(self, messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ...) -> None: ...

class SearchRoomMessagesRequest(_message.Message):
    __slots__ = ("room_id", "ranges", "limit")
    ROOM_ID_FIELD_NUMBER: _ClassVar[int]
    RANGES_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    room_id: str
    ranges: _containers.RepeatedCompositeFieldContainer[MessageSeqRange]
    limit: int
    def __init__(self, room_id: _Optional[str] = ..., ranges: _Optional[_Iterable[_Union[MessageSeqRange, _Mapping]]] = ..., limit: _Optional[int] = ...) -> None: ...

class MessageSeqRange(_message.Message):
    __slots__ = ("seq_start", "seq_end")
    SEQ_START_FIELD_NUMBER: _ClassVar[int]
    SEQ_END_FIELD_NUMBER: _ClassVar[int]
    seq_start: int
    seq_end: int
    def __init__(self, seq_start: _Optional[int] = ..., seq_end: _Optional[int] = ...) -> None: ...

class SearchRoomMessagesResponse(_message.Message):
    __slots__ = ("messages",)
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    def __init__(self, messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ...) -> None: ...

class ChatMessage(_message.Message):
    __slots__ = ("message_id", "conversation_id", "sender_id", "seq", "type", "content", "status", "send_time", "sender_name")
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SEND_TIME_FIELD_NUMBER: _ClassVar[int]
    SENDER_NAME_FIELD_NUMBER: _ClassVar[int]
    message_id: str
    conversation_id: str
    sender_id: str
    seq: int
    type: MessageType
    content: str
    status: MessageStatus
    send_time: int
    sender_name: str
    def __init__(self, message_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., sender_id: _Optional[str] = ..., seq: _Optional[int] = ..., type: _Optional[_Union[MessageType, str]] = ..., content: _Optional[str] = ..., status: _Optional[_Union[MessageStatus, str]] = ..., send_time: _Optional[int] = ..., sender_name: _Optional[str] = ...) -> None: ...
