"""Schema document dataclasses for all schema generators."""

import json
from dataclasses import dataclass, field
from typing import Any, cast

from ._protobuf_types import ProtoEnum, ProtoMessage
from .avro_types import AvroEnumSchema, AvroRecordSchema


@dataclass
class AvroSchemaDocument:
    """Avro schema document container.

    Attributes:
        main: The primary record schema.
        namespace: Avro namespace for the schema.
        enums: Collected enum schemas (name -> schema).
    """

    main: AvroRecordSchema
    namespace: str
    enums: dict[str, AvroEnumSchema] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (returns the main schema).

        For Avro, we typically just return the main schema since
        nested types are embedded inline.
        """
        return cast("dict[str, Any]", self.main)

    def to_string(self, indent: int = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation of the schema.
        """
        return json.dumps(self.main, indent=indent)


@dataclass
class ProtoSchemaDocument:
    """Protocol Buffer schema document container.

    Attributes:
        syntax: Proto syntax version ("proto2" or "proto3").
        package: Protobuf package name.
        main: The primary message schema.
        auxiliary_messages: Additional message schemas (nested types).
        enums: Enum schemas.
        imports: Required import paths.
    """

    syntax: str
    package: str
    main: ProtoMessage
    auxiliary_messages: list[ProtoMessage] = field(default_factory=list)
    enums: list[ProtoEnum] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def to_proto_file(self) -> dict[str, Any]:
        """Convert to ProtoFile format (for backwards compatibility).

        Returns:
            ProtoFile TypedDict compatible dictionary.
        """
        return {
            "syntax": self.syntax,
            "package": self.package,
            "imports": self.imports,
            "messages": [*self.auxiliary_messages, self.main],
            "enums": self.enums,
        }

    def to_string(self) -> str:
        """Convert to .proto file string.

        Note: This requires the generator's proto_file_to_string method.
        Use generator.to_string(document) instead.

        Returns:
            Proto file content as string.
        """
        raise NotImplementedError(
            "Use ProtoSchemaGenerator.to_string(document) instead"
        )


@dataclass
class TypeScriptModule:
    """TypeScript module document container.

    Attributes:
        main: The primary interface/type/schema.
        auxiliary: Additional nested interface/type definitions.
        enums: Enum declarations.
        imports: Import statements (e.g., "import { z } from 'zod';").
    """

    main: str
    auxiliary: list[str] = field(default_factory=list)
    enums: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def to_string(self) -> str:
        """Convert to TypeScript source string.

        Returns:
            Complete TypeScript module as a string.
        """
        parts = []

        if self.imports:
            parts.extend(self.imports)
            parts.append("")

        if self.enums:
            parts.extend(self.enums)
            parts.append("")

        if self.auxiliary:
            parts.extend(self.auxiliary)
            parts.append("")

        parts.append(self.main)

        return "\n\n".join(parts)
