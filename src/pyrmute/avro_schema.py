"""Avro schema generation from Pydantic models."""

import json
import re
from collections.abc import Mapping
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Final, Self, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ._registry import Registry
from ._schema_generator import SchemaGeneratorBase
from ._type_inspector import TypeInspector
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
    CachedAvroEnumSchema,
)
from .model_version import ModelVersion


class AvroSchemaGenerator(SchemaGeneratorBase[AvroRecordSchema]):
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
        super().__init__(include_docs=include_docs)
        self.namespace = namespace
        self._current_model = ("", "")
        self._enum_schemas: dict[str, CachedAvroEnumSchema] = {}

    def _reset_state(self) -> None:
        """Reset internal state before generating a new schema."""
        super()._reset_state()
        self._current_model = ("", "")
        self._enum_schemas = {}

    def generate_schema(
        self: Self,
        model: type[BaseModel],
        name: str,
        version: str | ModelVersion | None = None,
        registry_name_map: dict[str, str] | None = None,
    ) -> AvroRecordSchema:
        """Generate an Avro schema from a Pydantic model.

        Args:
            model: Pydantic model class.
            name: Model name.
            version: Optional namespace version. This is often the model
                version.
            registry_name_map: Optional mapping of class names to registry names.

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
        self._reset_state()

        self._register_model_name(model.__name__, name)
        self._current_model = (model.__name__, name)
        self._types_seen.add(model.__name__)

        self._collect_nested_models(model)
        for nested_class_name in self._nested_models:
            if nested_class_name != model.__name__:
                self._register_model_name(nested_class_name, nested_class_name)

        full_namespace = self.namespace
        if version:
            version_str = str(version).replace(".", "_")
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

    def _generate_field_schema(
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

        context = self._analyze_field(field_info)

        avro_type = self._convert_type(field_info.annotation, field_info)

        if context["is_optional"]:
            # Includes None. Wrap in union with null first
            if isinstance(avro_type, list):
                # Remove null if present and re-add at the front
                avro_type = [t for t in avro_type if t != "null"]
                avro_type.insert(0, "null")
            else:
                avro_type = ["null", avro_type]

            if "default_value" in context:
                field_schema["default"] = self._convert_default_value(
                    context["default_value"]
                )
            else:
                field_schema["default"] = None
        elif context["has_default"]:
            # Has default but not nullable. Just set the default
            if "default_value" in context:
                field_schema["default"] = self._convert_default_value(
                    context["default_value"]
                )

        field_schema["type"] = avro_type
        return field_schema

    def _convert_type(  # noqa: PLR0911, PLR0912, C901
        self: Self,
        python_type: Any,
        field_info: FieldInfo | None = None,
    ) -> AvroType:
        """Convert Python type annotation to Avro type.

        Args:
            python_type: Python type annotation.
            field_info: Optional field info for constraint checking.

        Returns:
            Avro type specification (string, list, or dict).
        """
        if python_type is None:
            return "null"

        if python_type is int:
            if field_info:
                return self._optimize_int_type(field_info)
            return "int"

        if python_type in self._LOGICAL_TYPE_MAPPING:
            return self._LOGICAL_TYPE_MAPPING[python_type].copy()

        if TypeInspector.is_enum(python_type):
            return self._convert_enum(python_type)

        # Check for bare list or dict before checking origin
        if python_type is list:
            arr_schema: AvroArraySchema = {"type": "array", "items": "string"}
            return arr_schema

        if python_type is dict:
            m_schema: AvroMapSchema = {"type": "map", "values": "string"}
            return m_schema

        origin = get_origin(python_type)
        if origin is not None:
            args = get_args(python_type)

            if TypeInspector.is_union_type(origin):
                return self._convert_union(args)

            if TypeInspector.is_list_like(origin):
                item_type = self._convert_type(args[0]) if args else "string"
                array_schema: AvroArraySchema = {"type": "array", "items": item_type}
                return array_schema

            if TypeInspector.is_dict_like(origin, python_type):
                value_type = self._convert_type(args[1]) if len(args) > 1 else "string"
                map_schema: AvroMapSchema = {"type": "map", "values": value_type}
                return map_schema

            if origin is tuple:
                return self._convert_tuple(python_type)

        if TypeInspector.is_base_model(python_type):
            return self._generate_nested_record_schema(python_type)

        if python_type in self._BASIC_TYPE_MAPPING:
            return self._BASIC_TYPE_MAPPING[python_type]

        type_str = str(python_type).lower()
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
        return "string"

    def _optimize_int_type(self: Self, field_info: FieldInfo) -> str:
        """Choose between int (32-bit) and long (64-bit) based on constraints.

        Args:
            field_info: Field info with potential constraints.

        Returns:
            "int" or "long"
        """
        constraints = TypeInspector.get_numeric_constraints(field_info)
        if constraints["ge"] is not None and constraints["ge"] < -(2**31):
            return "long"
        if constraints["gt"] is not None and constraints["gt"] + 1 < -(2**31):
            return "long"
        if constraints["le"] is not None and constraints["le"] > (2**31 - 1):
            return "long"
        if constraints["lt"] is not None and constraints["lt"] - 1 > (2**31 - 1):
            return "long"

        return "int"

    def _convert_enum(self: Self, enum_class: type[Enum]) -> AvroEnumSchema | str:
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
        enum_name = enum_class.__name__

        self._register_enum(enum_class)

        if enum_name in self._enum_schemas:
            return self._enum_schemas[enum_name]["namespace_ref"]

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

        enum_namespace = self._get_enum_namespace(enum_name)

        enum_schema: AvroEnumSchema = {
            "type": "enum",
            "name": enum_name,
            "namespace": enum_namespace,
            "symbols": symbols,
        }

        namespace_ref = f"{enum_namespace}.{enum_name}"
        self._enum_schemas[enum_name] = {
            "schema": enum_schema,
            "namespace_ref": namespace_ref,
        }

        return enum_schema

    def _get_enum_namespace(self: Self, module: str) -> str:
        """Convert Python module to Avro namespace.

        Args:
            module: Python module name.

        Returns:
            Avro-compatible namespace.
        """
        if module in ("__main__", "builtins", None):
            return self.namespace

        return f"{self.namespace}.{module}"

    def _convert_union(self: Self, args: tuple[Any, ...]) -> AvroUnion:
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
                avro_type = self._convert_type(arg)
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

    def _convert_tuple(self: Self, python_type: Any) -> AvroArraySchema:
        """Convert tuple type to Avro array with union of item types.

        Avro doesn't have a true tuple type (fixed-length with heterogeneous types),
        so we convert to an array with a union of all possible item types.

        Args:
            python_type: Tuple annotation.

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
        element_types = TypeInspector.get_tuple_element_types(python_type)

        if not element_types:
            empty_array: AvroArraySchema = {"type": "array", "items": "string"}
            return empty_array

        # Collect all unique types in the tuple
        item_types: list[str | AvroSchema] = []
        type_strs: set[str] = set()

        for arg in element_types:
            avro_type = self._convert_type(arg)
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
        self._register_nested_model(model)

        # If we've seen this type before, just reference it
        if type_name in self._types_seen:
            if type_name == self._current_model[0]:
                # Is recursive self-reference - use the versioned name
                return self._get_model_schema_name(type_name)  # CHANGED
            # Use the mapped name
            return self._get_model_schema_name(type_name)

        self._types_seen.add(type_name)

        schema_name = self._get_model_schema_name(type_name)
        schema: AvroRecordSchema = {
            "type": "record",
            "name": schema_name,
            "fields": [],
        }

        if self.include_docs and model.__doc__:
            schema["doc"] = model.__doc__.strip()

        for field_name, field_info in model.model_fields.items():
            field_schema = self._generate_field_schema(field_name, field_info, model)
            schema["fields"].append(field_schema)

        return schema

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
            self.generator.generate_schema(model, name, version)
            if versioned_namespace
            else self.generator.generate_schema(model, name)
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
                    self.generator.generate_schema(model, model_name, version)
                    if versioned_namespace
                    else self.generator.generate_schema(model, model_name)
                )

                version_str = str(version).replace(".", "_")
                filename = f"{model_name}_v{version_str}.avsc"
                filepath = output_dir / filename

                filepath.write_text(json.dumps(schema, indent=indent))

                all_schemas[model_name][str(version)] = schema

        return all_schemas
