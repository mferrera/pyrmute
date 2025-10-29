"""Avro schema generation from Pydantic models."""

import json
import re
import types
from collections.abc import Mapping
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Final, Self, Union, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ._registry import Registry
from .avro_types import (
    AvroArraySchema,
    AvroDefaultValue,
    AvroEnumSchema,
    AvroField,
    AvroLogicalType,
    AvroMapSchema,
    AvroRecordSchema,
    AvroSchema,
    AvroType,
    AvroUnion,
)
from .model_version import ModelVersion


class AvroSchemaGenerator:
    """Generates Apache Avro schemas from Pydantic models."""

    AVRO_SYMBOL_REGEX: Final = re.compile("[A-Za-z_][A-Za-z0-9_]*")

    _BASIC_TYPE_MAPPING: Mapping[type, str] = {
        str: "string",
        int: "int",
        float: "double",
        bool: "boolean",
        bytes: "bytes",
    }

    _LOGICAL_TYPE_MAPPING: Mapping[type, AvroLogicalType] = {
        datetime: {"type": "long", "logicalType": "timestamp-micros"},
        date: {"type": "int", "logicalType": "date"},
        time: {"type": "long", "logicalType": "time-micros"},
        UUID: {"type": "string", "logicalType": "uuid"},
        Decimal: {
            "type": "bytes",
            "logicalType": "decimal",
            "precision": 10,
            "scale": 2,
        },
    }

    def __init__(
        self: Self,
        namespace: str = "com.example",
        include_docs: bool = True,
    ) -> None:
        """Initialize the Avro schema generator.

        Args:
            namespace: Avro namespace for generated schemas (e.g.,
                "com.mycompany.events").
            include_docs: Whether to include field descriptions in schemas.
        """
        self.namespace = namespace
        self.include_docs = include_docs
        self._types_seen: set[str] = set()
        self._current_model = ("", "")

    def generate_avro_schema(
        self: Self,
        model: type[BaseModel],
        name: str,
        namespace_version: str | ModelVersion | None = None,
    ) -> AvroRecordSchema:
        """Generate an Avro schema from a Pydantic model.

        Args:
            model: Pydantic model class.
            name: Model name.
            namespace_version: Optional namespace version. This is often the model
                version.

        Returns:
            Avro record schema.

        Example:
            ```python
            from pydantic import BaseModel, Field
            from datetime import datetime

            class Event(BaseModel):
                '''Event record.'''
                id: UUID = Field(description="Event identifier")
                name: str = Field(description="Event name")
                timestamp: datetime = Field(description="Event timestamp")
                metadata: dict[str, str] = Field(default_factory=dict)

            generator = AvroSchemaGenerator(namespace="com.events")
            schema = generator.generate_avro_schema(Event, "Event")

            # Returns proper Avro schema with logical types
            # {
            #   "type": "record",
            #   "name": "Event",
            #   "namespace": "com.events",
            #   "doc": "Event record.",
            #   "fields": [
            #     {"name": "id", "type": {"type": "string", "logicalType": "uuid"}},
            #     {"name": "name", "type": "string"},
            #     {"name": "timestamp", "type": {"type": "long", "logicalType": "timestamp-micros"}},
            #     {"name": "metadata", "type": {"type": "map", "values": "string"}}
            #   ]
            # }

            # When a version is provided
            schema = generator.generate_avro_schema(Event, "Event", "1.0.0")

            # Returns proper Avro schema with logical types
            # {
            #   "type": "record",
            #   "name": "Event",
            #   "namespace": "com.events.v1_0_0",
            #   "doc": "Event record.",
            #   "fields": [
            #     {"name": "id", "type": {"type": "string", "logicalType": "uuid"}},
            #     {"name": "name", "type": "string"},
            #     {"name": "timestamp", "type": {"type": "long", "logicalType": "timestamp-micros"}},
            #     {"name": "metadata", "type": {"type": "map", "values": "string"}}
            #   ]
            # }
            ```
        """  # noqa: E501
        self._types_seen = set()
        # Mark the root type as seen to prevent infinite recursion
        self._current_model = (model.__name__, name)
        self._types_seen.add(model.__name__)

        full_namespace = self.namespace
        if namespace_version:
            version_str = str(namespace_version).replace(".", "_")
            full_namespace = f"{self.namespace}.v{version_str}"

        schema: AvroRecordSchema = {
            "type": "record",
            "name": name,
            "namespace": full_namespace,
            "fields": [],
        }

        if self.include_docs and model.__doc__:
            schema["doc"] = model.__doc__.strip()

        for field_name, field_info in model.model_fields.items():
            field_schema = self._generate_field_schema(field_name, field_info, model)
            schema["fields"].append(field_schema)

        return schema

    def _generate_field_schema(  # noqa: C901
        self: Self,
        field_name: str,
        field_info: FieldInfo,
        model: type[BaseModel],
    ) -> AvroField:
        """Generate Avro schema for a single field.

        Args:
            field_name: Name of the field.
            field_info: Pydantic field info.
            model: Parent model class.

        Returns:
            Avro field schema.
        """
        field_schema: AvroField = {"name": field_name, "type": "string"}

        # Add documentation
        if self.include_docs and field_info.description:
            field_schema["doc"] = field_info.description

        avro_type = self._python_type_to_avro(field_info.annotation, field_info)
        is_nullable = self._is_optional_type(field_info.annotation)
        has_default = (
            field_info.default is not PydanticUndefined
            or field_info.default_factory is not None
        )

        if is_nullable:
            # Type includes None - wrap in union with null first
            if isinstance(avro_type, list):
                # Remove null if present and re-add at the front
                avro_type = [t for t in avro_type if t != "null"]
                avro_type.insert(0, "null")
            else:
                avro_type = ["null", avro_type]

            # Set default if provided
            if field_info.default is not PydanticUndefined:
                field_schema["default"] = self._convert_default_value(
                    field_info.default
                )
            elif field_info.default_factory is not None:
                try:
                    default_value = field_info.default_factory()  # type: ignore[call-arg]
                    field_schema["default"] = self._convert_default_value(default_value)
                except Exception:
                    field_schema["default"] = None
            else:
                field_schema["default"] = None
        elif has_default:
            # Has default but not nullable - just set the default
            if field_info.default is not PydanticUndefined:
                field_schema["default"] = self._convert_default_value(
                    field_info.default
                )
            elif field_info.default_factory is not None:
                try:
                    default_value = field_info.default_factory()  # type: ignore[call-arg]
                    field_schema["default"] = self._convert_default_value(default_value)
                except Exception:
                    # If factory fails, don't set a default
                    pass

        field_schema["type"] = avro_type
        return field_schema

    def _python_type_to_avro(  # noqa: PLR0911, PLR0912, C901
        self: Self,
        annotation: Any,
        field_info: FieldInfo | None = None,
    ) -> AvroType:
        """Convert Python type annotation to Avro type.

        Args:
            annotation: Python type annotation.
            field_info: Optional field info for constraint checking.

        Returns:
            Avro type specification (string, list, or dict).
        """
        if annotation is None:
            return "null"

        if annotation is int:
            # Only optimize if we have field_info with constraints
            if field_info and hasattr(field_info, "metadata") and field_info.metadata:
                return self._optimize_int_type(field_info)

            return "int"

        if annotation in self._LOGICAL_TYPE_MAPPING:
            return self._LOGICAL_TYPE_MAPPING[annotation].copy()

        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return self._enum_to_avro(annotation)

        # Check for bare list or dict before checking origin
        if annotation is list:
            arr_schema: AvroArraySchema = {"type": "array", "items": "string"}
            return arr_schema

        if annotation is dict:
            m_schema: AvroMapSchema = {"type": "map", "values": "string"}
            return m_schema

        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)

            if self._is_union_type(origin):
                return self._union_to_avro(args)

            if origin is list:
                item_type = self._python_type_to_avro(args[0]) if args else "string"
                array_schema: AvroArraySchema = {"type": "array", "items": item_type}
                return array_schema

            if origin is dict:
                value_type = (
                    self._python_type_to_avro(args[1]) if len(args) > 1 else "string"
                )
                map_schema: AvroMapSchema = {"type": "map", "values": value_type}
                return map_schema

            if origin is tuple:
                return self._tuple_to_avro(args)

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return self._generate_nested_record_schema(annotation)

        # Basic type mapping with integer optimization
        if annotation is int and field_info:
            return self._optimize_int_type(field_info)

        if annotation in self._BASIC_TYPE_MAPPING:
            return self._BASIC_TYPE_MAPPING[annotation]

        type_str = str(annotation).lower()
        if "str" in type_str:
            return "string"
        if "int" in type_str:
            return "int"
        if "float" in type_str:
            return "double"
        if "bool" in type_str:
            return "boolean"
        if "bytes" in type_str:
            return "bytes"

        # Default fallback
        return "string"

    def _optimize_int_type(self: Self, field_info: FieldInfo) -> str:
        """Choose between int (32-bit) and long (64-bit) based on constraints.

        Args:
            field_info: Field info with potential constraints.

        Returns:
            "int" or "long"
        """
        # Check if field has constraints
        if not hasattr(field_info, "metadata") or not field_info.metadata:
            return "long"  # Default to long for safety

        # Look for Annotated constraints
        minimum: int | None = None
        maximum: int | None = None
        for constraint in field_info.metadata:
            if hasattr(constraint, "ge"):
                minimum = constraint.ge
            elif hasattr(constraint, "gt"):
                minimum = constraint.gt + 1
            if hasattr(constraint, "le"):
                maximum = constraint.le
            elif hasattr(constraint, "lt"):
                maximum = constraint.lt - 1

        # If we have both min and max and they fit in 32 bits, use int
        if (
            minimum is not None
            and minimum >= -(2**31)
            and maximum is not None
            and maximum <= (2**31 - 1)
        ):
            return "int"

        return "long"

    def _enum_to_avro(self: Self, enum_class: type[Enum]) -> AvroEnumSchema:
        """Convert Python Enum to Avro enum type.

        Args:
            enum_class: Python Enum class.

        Returns:
            Avro enum schema.

        Example:
            ```python
            from enum import Enum

            class Status(str, Enum):
                PENDING = "pending"
                ACTIVE = "active"
                COMPLETED = "completed"

            # Converts to:
            # {
            #   "type": "enum",
            #   "name": "Status",
            #   "symbols": ["pending", "active", "completed"]
            # }
            ```
        """
        symbols = []
        for member in enum_class:
            value = str(member.value)
            if not re.fullmatch(self.AVRO_SYMBOL_REGEX, value):
                raise ValueError(
                    f"Unable to convert enum '{enum_class.__name__}' to Avro. "
                    "Every symbol must match the regular expression "
                    "'[A-Za-z_][A-Za-z0-9_]*'. Got '{value}'"
                )
            symbols.append(value)

        return {
            "type": "enum",
            "name": enum_class.__name__,
            "symbols": symbols,
        }

    def _union_to_avro(self: Self, args: tuple[Any, ...]) -> AvroUnion:
        """Convert Union type to Avro union.

        Args:
            args: Union type arguments.

        Returns:
            List of Avro types (strings for primitives, dicts for complex types).

        Example:
            ```python
            # str | int | None becomes ["null", "string", "int"]
            # Optional[str] becomes ["null", "string"]
            ```
        """
        avro_types: AvroUnion = []

        for arg in args:
            if arg is type(None):
                avro_types.append("null")
            else:
                avro_type = self._python_type_to_avro(arg)
                if isinstance(avro_type, list):
                    # Flatten nested unions
                    avro_types.extend(avro_type)
                else:
                    avro_types.append(avro_type)

        # Remove duplicates while preserving order
        seen: set[str] = set()
        unique_types: AvroUnion = []
        for t in avro_types:
            # Convert to string for comparison
            t_str = str(t) if not isinstance(t, dict) else json.dumps(t, sort_keys=True)
            if t_str not in seen:
                seen.add(t_str)
                unique_types.append(t)

        return unique_types

    def _tuple_to_avro(self: Self, args: tuple[Any, ...]) -> AvroArraySchema:
        """Convert tuple type to Avro array with union of item types.

        Avro doesn't have a true tuple type (fixed-length with heterogeneous types),
        so we convert to an array with a union of all possible item types.

        Args:
            args: Tuple type arguments.

        Returns:
            Avro array schema with union items.

        Example:
            ```python
            # tuple[str, int, bool] becomes:
            # {"type": "array", "items": ["string", "int", "boolean"]}

            # tuple[float, float, float] becomes:
            # {"type": "array", "items": "double"}
            ```
        """
        if not args:
            empty_array: AvroArraySchema = {"type": "array", "items": "string"}
            return empty_array

        # Collect all unique types in the tuple
        item_types: list[str | AvroSchema] = []
        type_strs: set[str] = set()

        for arg in args:
            avro_type = self._python_type_to_avro(arg)
            if isinstance(avro_type, list):
                # Flatten unions
                for t in avro_type:
                    t_str = (
                        str(t)
                        if not isinstance(t, dict)
                        else json.dumps(t, sort_keys=True)
                    )
                    if t_str not in type_strs:
                        type_strs.add(t_str)
                        item_types.append(t)
            else:
                t_str = (
                    str(avro_type)
                    if not isinstance(avro_type, dict)
                    else json.dumps(avro_type, sort_keys=True)
                )
                if t_str not in type_strs:
                    type_strs.add(t_str)
                    item_types.append(avro_type)

        # If all types are the same, use that type directly
        if len(item_types) == 1:
            single_type_array: AvroArraySchema = {
                "type": "array",
                "items": item_types[0],
            }
            return single_type_array

        # Otherwise use union
        union_array: AvroArraySchema = {"type": "array", "items": item_types}
        return union_array

    def _generate_nested_record_schema(
        self: Self, model: type[BaseModel]
    ) -> AvroRecordSchema | str:
        """Generate Avro schema for a nested Pydantic model.

        If the type has been seen before, return a reference to avoid
        infinite recursion and schema duplication.

        Args:
            model: Nested Pydantic model class.

        Returns:
            Avro record schema or type name reference.

        Example:
            ```python
            # First occurrence: full schema
            # {
            #   "type": "record",
            #   "name": "Address",
            #   "fields": [...]
            # }

            # Subsequent occurrences: just the name
            # "Address"
            ```
        """
        type_name = model.__name__

        # If we've seen this type before, just reference it
        if type_name in self._types_seen:
            if type_name == self._current_model[0]:
                # Is recursive self-reference
                return self._current_model[1]
            return type_name

        self._types_seen.add(type_name)

        schema: AvroRecordSchema = {
            "type": "record",
            "name": type_name,
            "fields": [],
        }

        if self.include_docs and model.__doc__:
            schema["doc"] = model.__doc__.strip()

        for field_name, field_info in model.model_fields.items():
            field_schema = self._generate_field_schema(field_name, field_info, model)
            schema["fields"].append(field_schema)

        return schema

    def _is_union_type(self: Self, origin: Any) -> bool:
        """Check if origin represents a Union type.

        Args:
            origin: Type origin from get_origin().

        Returns:
            True if this is a Union type.
        """
        if origin is Union:
            return True

        if hasattr(types, "UnionType"):
            try:
                return origin is types.UnionType
            except (ImportError, AttributeError):
                pass

        return False

    def _is_optional_type(self: Self, annotation: Any) -> bool:
        """Check if annotation represents an Optional type.

        Args:
            annotation: Type annotation.

        Returns:
            True if this is Optional (Union with None).
        """
        origin = get_origin(annotation)
        if self._is_union_type(origin):
            args = get_args(annotation)
            return type(None) in args
        return False

    def _convert_default_value(self: Self, value: Any) -> AvroDefaultValue:  # noqa: PLR0911, C901, PLR0912
        """Convert Python default value to Avro-compatible format.

        Args:
            value: Python default value.

        Returns:
            Avro-compatible default value.
        """
        if value is None:
            return None
        if isinstance(value, bool):  # Check bool before int (bool is subclass of int)
            return value
        if isinstance(value, (str, int, float)):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, list):
            return [self._convert_default_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._convert_default_value(v) for k, v in value.items()}
        if isinstance(value, Enum):
            return str(value.value)
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, datetime):
            # timestamp-micros: microseconds since epoch
            return int(value.timestamp() * 1_000_000)
        if isinstance(value, date):
            # date: days since epoch
            epoch = date(1970, 1, 1)
            return (value - epoch).days
        if isinstance(value, time):
            # time-micros: microseconds since midnight
            return (
                value.hour * 3600 + value.minute * 60 + value.second
            ) * 1_000_000 + value.microsecond
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, Decimal):
            # Decimal as bytes - this is complex, for now convert to float
            return float(value)

        # For other types, convert to string
        return str(value)


class AvroExporter:
    """Export Pydantic models to Avro schema files.

    This class provides methods to export individual schemas or all schemas from a model
    _registry to .avsc (Avro Schema) files.
    """

    def __init__(
        self: Self,
        registry: Registry,
        namespace: str = "com.example",
        include_docs: bool = True,
    ) -> None:
        """Initialize the Avro exporter.

        Args:
            registry: Model registry instance.
            namespace: Avro namespace for schemas.
            include_docs: Whether to include documentation.
        """
        self._registry = registry
        self.generator = AvroSchemaGenerator(
            namespace=namespace,
            include_docs=include_docs,
        )

    def export_schema(
        self: Self,
        name: str,
        version: str | ModelVersion,
        output_path: str | Path | None = None,
        versioned_namespace: bool = False,
    ) -> AvroRecordSchema:
        """Export a single model version as an Avro schema.

        Args:
            name: Model name.
            version: Model version.
            output_path: Optional file path to save schema.
            versioned_namespace: Include model version in namespace. Default False.

        Returns:
            Avro record schema.

        Example:
            ```python
            exporter = AvroExporter(manager._registry, namespace="com.myapp")

            # Export and save
            schema = exporter.export_schema("User", "1.0.0", "schemas/user_v1.avsc")

            # Or just get the schema
            schema = exporter.export_schema("User", "1.0.0", versioned_namespace=True)
            print(json.dumps(schema, indent=2))
            ```
        """
        model = self._registry.get_model(name, version)
        schema = (
            self.generator.generate_avro_schema(model, name, version)
            if versioned_namespace
            else self.generator.generate_avro_schema(model, name)
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(schema, indent=2))

        return schema

    def export_all_schemas(
        self: Self,
        output_dir: str | Path,
        indent: int = 2,
        versioned_namespace: bool = False,
    ) -> dict[str, dict[str, AvroRecordSchema]]:
        """Export all registered models as Avro schemas.

        Args:
            output_dir: Directory to save schema files.
            indent: JSON indentation level.
            versioned_namespace: Include model version in namespace. Default False.

        Returns:
            Dictionary mapping model names to version to schema.

        Example:
            ```python
            exporter = AvroExporter(manager._registry, namespace="com.myapp")
            schemas = exporter.export_all_schemas("schemas/avro/")

            # Creates files like:
            # schemas/avro/User_v1_0_0.avsc
            # schemas/avro/User_v2_0_0.avsc
            # schemas/avro/Order_v1_0_0.avsc
            ```
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_schemas: dict[str, dict[str, AvroRecordSchema]] = {}

        for model_name in self._registry.list_models():
            all_schemas[model_name] = {}
            versions = self._registry.get_versions(model_name)

            for version in versions:
                model = self._registry.get_model(model_name, version)
                schema = (
                    self.generator.generate_avro_schema(model, model_name, version)
                    if versioned_namespace
                    else self.generator.generate_avro_schema(model, model_name)
                )

                version_str = str(version).replace(".", "_")
                filename = f"{model_name}_v{version_str}.avsc"
                filepath = output_dir / filename

                filepath.write_text(json.dumps(schema, indent=indent))

                all_schemas[model_name][str(version)] = schema

        return all_schemas
