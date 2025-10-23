"""Protocol Buffer schema type definitions."""

from typing import Literal, TypedDict


class ProtoFieldOptions(TypedDict, total=False):
    """Protocol Buffer field options."""

    deprecated: bool
    packed: bool
    json_name: str


class ProtoField(TypedDict, total=False):
    """Protocol Buffer field definition."""

    name: str
    type: str
    number: int
    label: Literal["optional", "repeated", "required"]
    options: ProtoFieldOptions
    comment: str
    oneof_group: str


class ProtoOneOf(TypedDict, total=False):
    """Protocol Buffer oneof definition."""

    name: str
    fields: list[str]
    comment: str


class ProtoEnum(TypedDict, total=False):
    """Protocol Buffer enum definition."""

    name: str
    values: dict[str, int]
    comment: str


class ProtoMessage(TypedDict, total=False):
    """Protocol Buffer message definition."""

    name: str
    fields: list[ProtoField]
    oneofs: list[ProtoOneOf]
    nested_messages: list["ProtoMessage"]
    nested_enums: list[ProtoEnum]
    comment: str
    field_order: list[tuple[str, str]]


class ProtoFile(TypedDict, total=False):
    """Protocol Buffer file definition."""

    syntax: Literal["proto2", "proto3"]
    package: str
    imports: list[str]
    messages: list[ProtoMessage]
    enums: list[ProtoEnum]
    options: dict[str, str]
