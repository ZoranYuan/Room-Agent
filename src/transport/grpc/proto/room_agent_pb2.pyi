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

class GetRoomAgentContextRequest(_message.Message):
    __slots__ = ("room_id",)
    ROOM_ID_FIELD_NUMBER: _ClassVar[int]
    room_id: str
    def __init__(self, room_id: _Optional[str] = ...) -> None: ...

class GetRoomAgentContextResponse(_message.Message):
    __slots__ = ("user_id", "thread_id", "unread_message_count", "unread_snapshot")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    THREAD_ID_FIELD_NUMBER: _ClassVar[int]
    UNREAD_MESSAGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    UNREAD_SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    thread_id: str
    unread_message_count: int
    unread_snapshot: UnreadSnapshot
    def __init__(self, user_id: _Optional[str] = ..., thread_id: _Optional[str] = ..., unread_message_count: _Optional[int] = ..., unread_snapshot: _Optional[_Union[UnreadSnapshot, _Mapping]] = ...) -> None: ...

class UnreadSnapshot(_message.Message):
    __slots__ = ("messages",)
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    def __init__(self, messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ...) -> None: ...

class ChatMessage(_message.Message):
    __slots__ = ("message_id", "conversation_id", "sender_id", "client_msg_id", "seq", "type", "content", "status", "send_time")
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_MSG_ID_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SEND_TIME_FIELD_NUMBER: _ClassVar[int]
    message_id: str
    conversation_id: str
    sender_id: str
    client_msg_id: str
    seq: int
    type: MessageType
    content: str
    status: MessageStatus
    send_time: int
    def __init__(self, message_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., sender_id: _Optional[str] = ..., client_msg_id: _Optional[str] = ..., seq: _Optional[int] = ..., type: _Optional[_Union[MessageType, str]] = ..., content: _Optional[str] = ..., status: _Optional[_Union[MessageStatus, str]] = ..., send_time: _Optional[int] = ...) -> None: ...
