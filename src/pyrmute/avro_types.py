"""Type definitions for Avro schemas."""

from __future__ import annotations

from typing import Literal, NotRequired, TypeAlias, TypedDict

AvroDefaultValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["AvroDefaultValue"]  # Arrays can contain any default values
    | dict[str, "AvroDefaultValue"]  # Maps/records have string keys
)


class AvroLogicalType(TypedDict):
    """Avro logical type schema."""

    type: Literal["int", "long", "bytes", "string"]
    logicalType: Literal[
        "date",
        "time-millis",
        "time-micros",
        "timestamp-millis",
        "timestamp-micros",
        "decimal",
        "uuid",
    ]
    precision: NotRequired[int]  # For decimal
    scale: NotRequired[int]  # For decimal


class AvroEnumSchema(TypedDict):
    """Avro enum type schema."""

    type: Literal["enum"]
    name: str
    symbols: list[str]
    doc: NotRequired[str]
    namespace: NotRequired[str]


class AvroArraySchema(TypedDict):
    """Avro array type schema."""

    type: Literal["array"]
    items: AvroType


class AvroMapSchema(TypedDict):
    """Avro map type schema."""

    type: Literal["map"]
    values: AvroType


class AvroField(TypedDict):
    """Avro record field."""

    name: str
    type: AvroType
    doc: NotRequired[str]
    default: NotRequired[AvroDefaultValue]
    order: NotRequired[Literal["ascending", "descending", "ignore"]]
    aliases: NotRequired[list[str]]


class AvroRecordSchema(TypedDict):
    """Avro record type schema."""

    type: Literal["record"]
    name: str
    namespace: NotRequired[str]
    doc: NotRequired[str]
    fields: list[AvroField]
    aliases: NotRequired[list[str]]


AvroSchema: TypeAlias = (
    str
    | AvroLogicalType
    | AvroEnumSchema
    | AvroArraySchema
    | AvroMapSchema
    | AvroRecordSchema
)
AvroPrimitive: TypeAlias = Literal[
    "null",
    "boolean",
    "int",
    "long",
    "float",
    "double",
    "bytes",
    "string",
]
AvroUnion: TypeAlias = list[str | AvroSchema]
AvroType: TypeAlias = AvroPrimitive | AvroSchema | AvroUnion
