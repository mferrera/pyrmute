"""Protocol Buffer schema generation from Pydantic models."""

import types
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Self, Union, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ._protobuf_types import ProtoEnum, ProtoField, ProtoFile, ProtoMessage, ProtoOneOf
from ._registry import Registry
from .model_version import ModelVersion


class ProtoSchemaGenerator:
    """Generates Protocol Buffer schemas from Pydantic models."""

    _BASIC_TYPE_MAPPING: Mapping[type, str] = {
        str: "string",
        int: "int32",
        float: "double",
        bool: "bool",
        bytes: "bytes",
    }

    _WELLKNOWN_TYPE_MAPPING: Mapping[type, tuple[str, str | None]] = {
        datetime: ("google.protobuf.Timestamp", "google/protobuf/timestamp.proto"),
        UUID: ("string", None),
        Decimal: ("double", None),
    }

    def __init__(
        self: Self,
        package: str = "com.example",
        include_comments: bool = True,
        use_proto3: bool = True,
    ) -> None:
        """Initialize the Protocol Buffer schema generator.

        Args:
            package: Protobuf package name (e.g., "com.mycompany.events").
            include_comments: Whether to include field descriptions as comments.
            use_proto3: Use proto3 syntax (True) or proto2 (False).
        """
        self.package = package
        self.include_comments = include_comments
        self.use_proto3 = use_proto3
        self._types_seen: set[str] = set()
        self._required_imports: set[str] = set()
        self._field_counter = 1
        self._collected_enums: list[ProtoEnum] = []
        self._collected_nested_messages: list[ProtoMessage] = []

    def _generate_proto_schema(
        self: Self,
        model: type[BaseModel],
        name: str,
        version: str | ModelVersion,
    ) -> ProtoMessage:
        """Generate a Protocol Buffer schema from a Pydantic model."""
        self._types_seen = set()
        self._required_imports = set()
        self._field_counter = 1
        self._collected_enums = []
        self._collected_nested_messages = []

        self._types_seen.add(model.__name__)

        message: ProtoMessage = {
            "name": name,
            "fields": [],
            "nested_messages": [],
            "nested_enums": [],
            "field_order": [],
        }

        if self.include_comments and model.__doc__:
            message["comment"] = model.__doc__.strip()

        oneofs_to_add: list[ProtoOneOf] = []

        for field_name, field_info in model.model_fields.items():
            if self._is_union_requiring_oneof(field_info.annotation):
                oneof_fields = self._generate_oneof_fields(
                    field_name, field_info, model
                )
                message["fields"].extend(oneof_fields["fields"])
                if oneof_fields["oneof"]:
                    oneofs_to_add.append(oneof_fields["oneof"])
                    message["field_order"].append(
                        (oneof_fields["oneof"]["name"], "oneof")
                    )
            else:
                field_schema = self._generate_field_schema(
                    field_name, field_info, model
                )
                message["fields"].append(field_schema)
                message["field_order"].append((field_schema["name"], "field"))

        if oneofs_to_add:
            message["oneofs"] = oneofs_to_add

        if self._collected_enums:
            message["nested_enums"] = self._collected_enums

        if self._collected_nested_messages:
            message["nested_messages"] = self._collected_nested_messages

        return message

    def _is_union_requiring_oneof(self: Self, annotation: Any) -> bool:
        """Check if annotation is a union type that requires oneof.

        Args:
            annotation: Type annotation.

        Returns:
            True if this union requires oneof representation.
        """
        origin = get_origin(annotation)
        if not self._is_union_type(origin):
            return False

        args = get_args(annotation)
        non_none_args = [arg for arg in args if arg is not type(None)]

        # If only one type remains (Optional[T]), no oneof needed
        return not len(non_none_args) <= 1

    def _generate_oneof_fields(
        self: Self,
        field_name: str,
        field_info: FieldInfo,
        model: type[BaseModel],
    ) -> dict[str, Any]:
        """Generate oneof fields for a union type.

        Args:
            field_name: Original field name.
            field_info: Pydantic field info.
            model: Parent model class.

        Returns:
            Dictionary with 'fields' list and 'oneof' definition.
        """
        args = get_args(field_info.annotation)
        non_none_args = [arg for arg in args if arg is not type(None)]

        oneof_name = f"{field_name}_value"
        oneof_field_names: list[str] = []
        fields: list[ProtoField] = []

        for arg in non_none_args:
            variant_name = self._get_union_variant_name(arg, field_name)
            proto_type, is_repeated = self._python_type_to_proto(arg, None)

            field_schema: ProtoField = {
                "name": variant_name,
                "type": proto_type,
                "number": self._field_counter,
                "oneof_group": oneof_name,
            }
            self._field_counter += 1

            if is_repeated:
                field_schema["label"] = "repeated"

            if self.include_comments:
                type_name = self._get_type_display_name(arg)
                field_schema["comment"] = f"{field_name} when type is {type_name}"

            fields.append(field_schema)
            oneof_field_names.append(variant_name)

        oneof: ProtoOneOf = {
            "name": oneof_name,
            "fields": oneof_field_names,
        }

        if self.include_comments and field_info.description:
            oneof["comment"] = field_info.description

        return {"fields": fields, "oneof": oneof}

    def _get_union_variant_name(self: Self, typ: type, base_name: str) -> str:
        """Generate a variant name for a union type.

        Args:
            typ: The variant type.
            base_name: Base field name.

        Returns:
            Generated variant field name.
        """
        if typ in self._BASIC_TYPE_MAPPING:
            type_name = self._BASIC_TYPE_MAPPING[typ].replace("<", "_").replace(">", "")
            return f"{base_name}_{type_name}"

        if hasattr(typ, "__name__"):
            type_name = typ.__name__.lower()
            return f"{base_name}_{type_name}"

        # Fallback for complex types
        type_str = str(typ).replace("[", "_").replace("]", "").replace(",", "_")
        return f"{base_name}_{type_str}"

    def _get_type_display_name(self: Self, typ: type) -> str:
        """Get a display name for a type.

        Args:
            typ: Type to get name for.

        Returns:
            Human-readable type name.
        """
        if hasattr(typ, "__name__"):
            return typ.__name__
        return str(typ)

    def _generate_field_schema(
        self: Self,
        field_name: str,
        field_info: FieldInfo,
        model: type[BaseModel],
    ) -> ProtoField:
        """Generate Protocol Buffer field for a single field.

        Args:
            field_name: Name of the field.
            field_info: Pydantic field info.
            model: Parent model class.

        Returns:
            Protocol Buffer field definition.
        """
        field_schema: ProtoField = {
            "name": field_name,
            "type": "string",
            "number": self._field_counter,
        }
        self._field_counter += 1

        if self.include_comments and field_info.description:
            field_schema["comment"] = field_info.description

        proto_type, is_repeated = self._python_type_to_proto(
            field_info.annotation, field_info
        )
        is_optional = self._is_optional_type(field_info.annotation)
        has_default = (
            field_info.default is not PydanticUndefined
            or field_info.default_factory is not None
        )

        if is_repeated:
            field_schema["label"] = "repeated"
        elif is_optional or has_default:
            field_schema["label"] = "optional"
        elif not self.use_proto3:
            field_schema["label"] = "required"

        field_schema["type"] = proto_type
        return field_schema

    def _python_type_to_proto(  # noqa: PLR0911, PLR0912, C901
        self: Self,
        annotation: Any,
        field_info: FieldInfo | None = None,
    ) -> tuple[str, bool]:
        """Convert Python type annotation to Protocol Buffer type.

        Args:
            annotation: Python type annotation.
            field_info: Optional field info for constraint checking.

        Returns:
            Tuple of (proto_type, is_repeated).
        """
        if annotation is None:
            return ("string", False)

        if annotation is int and field_info:
            proto_type = self._optimize_int_type(field_info)
            return (proto_type, False)

        if annotation in self._BASIC_TYPE_MAPPING:
            return (self._BASIC_TYPE_MAPPING[annotation], False)

        if annotation in self._WELLKNOWN_TYPE_MAPPING:
            proto_type, import_path = self._WELLKNOWN_TYPE_MAPPING[annotation]
            if import_path:
                self._required_imports.add(import_path)
            return (proto_type, False)

        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return self._enum_to_proto(annotation)

        origin = get_origin(annotation)
        if self._is_union_type(origin):
            args = get_args(annotation)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return self._python_type_to_proto(non_none_args[0], field_info)

            return ("string", False)

        if origin in (list, tuple) or (
            hasattr(origin, "__origin__") and origin.__origin__ in (list, tuple)
        ):
            args = get_args(annotation)
            if args:
                item_type, _ = self._python_type_to_proto(args[0], None)
                return (item_type, True)
            return ("string", True)

        if (
            annotation is dict
            or origin is dict
            or (hasattr(origin, "__origin__") and origin.__origin__ is dict)
        ):
            args = get_args(annotation)
            if args and len(args) == 2:  # noqa: PLR2004
                key_type, _ = self._python_type_to_proto(args[0], None)
                value_type, _ = self._python_type_to_proto(args[1], None)
                return (f"map<{key_type}, {value_type}>", False)
            return ("map<string, string>", False)

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            model_name = annotation.__name__

            if model_name not in self._types_seen:
                self._types_seen.add(model_name)
                nested_msg = self._generate_nested_message(annotation, model_name)
                self._collected_nested_messages.append(nested_msg)

            return model_name, False

        return ("string", False)

    def _optimize_int_type(self: Self, field_info: FieldInfo) -> str:
        """Optimize integer type based on field constraints.

        Args:
            field_info: Pydantic field info with metadata.

        Returns:
            Optimized protobuf integer type.
        """
        if not hasattr(field_info, "metadata") or not field_info.metadata:
            return "int32"

        ge_value = None
        le_value = None

        for constraint in field_info.metadata:
            if hasattr(constraint, "ge") and constraint.ge is not None:
                ge_value = constraint.ge
            if hasattr(constraint, "le") and constraint.le is not None:
                le_value = constraint.le

        if ge_value is not None and ge_value >= 0:
            if le_value is not None:
                if le_value <= 2**32 - 1:
                    return "uint32"
                return "uint64"
            return "uint32"

        return "int32"

    def _enum_to_proto(self: Self, enum_class: type[Enum]) -> tuple[str, bool]:
        """Convert Python Enum to Protocol Buffer enum.

        Args:
            enum_class: Python Enum class.

        Returns:
            Enum name for reference in proto file.
        """
        enum_name = enum_class.__name__

        if enum_name not in self._types_seen:
            self._types_seen.add(enum_name)
            enum_def = self._generate_enum_schema(enum_class)
            self._collected_enums.append(enum_def)

        return enum_name, False

    def _generate_enum_schema(self: Self, enum_class: type[Enum]) -> ProtoEnum:
        """Generate a Protocol Buffer enum from a Python Enum.

        Args:
            enum_class: Python Enum class.

        Returns:
            Protocol Buffer enum definition.
        """
        enum_def: ProtoEnum = {"name": enum_class.__name__, "values": {}}

        if self.include_comments and enum_class.__doc__:
            enum_def["comment"] = enum_class.__doc__.strip()

        for i, member in enumerate(enum_class):
            enum_def["values"][member.name] = i

        return enum_def

    def _generate_nested_message(
        self: Self,
        model: type[BaseModel],
        name: str,
    ) -> ProtoMessage:
        """Generate a nested Protocol Buffer message.

        This is similar to generate_proto_schema but:

        1. Doesn't reset the global state (types_seen, etc.)
        2. Doesn't take a version parameter
        3. Shares the enum/nested message collections with parent

        Args:
            model: Pydantic model class.
            name: Model name.

        Returns:
            Protocol Buffer message definition.
        """
        message: ProtoMessage = {
            "name": name,
            "fields": [],
            "nested_messages": [],
            "nested_enums": [],
            "field_order": [],
        }

        if self.include_comments and model.__doc__:
            message["comment"] = model.__doc__.strip()

        saved_field_counter = self._field_counter
        self._field_counter = 1

        parent_enums = self._collected_enums
        parent_messages = self._collected_nested_messages

        self._collected_enums = []
        self._collected_nested_messages = []

        oneofs_to_add: list[ProtoOneOf] = []

        for field_name, field_info in model.model_fields.items():
            if self._is_union_requiring_oneof(field_info.annotation):
                oneof_fields = self._generate_oneof_fields(
                    field_name, field_info, model
                )
                message["fields"].extend(oneof_fields["fields"])
                if oneof_fields["oneof"]:
                    oneofs_to_add.append(oneof_fields["oneof"])
                    message["field_order"].append(
                        (oneof_fields["oneof"]["name"], "oneof")
                    )
            else:
                field_schema = self._generate_field_schema(
                    field_name, field_info, model
                )
                message["fields"].append(field_schema)
                message["field_order"].append((field_schema["name"], "field"))

        if oneofs_to_add:
            message["oneofs"] = oneofs_to_add

        if self._collected_enums:
            message["nested_enums"] = self._collected_enums

        if self._collected_nested_messages:
            message["nested_messages"] = self._collected_nested_messages

        self._collected_enums = parent_enums
        self._collected_nested_messages = parent_messages

        self._field_counter = saved_field_counter

        return message

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

    def generate_proto_file(
        self: Self,
        model: type[BaseModel],
        name: str,
        version: str | ModelVersion,
    ) -> ProtoFile:
        """Generate a complete .proto file definition.

        Args:
            model: Pydantic model class.
            name: Model name.
            version: Model version.

        Returns:
            Complete proto file definition.
        """
        message = self._generate_proto_schema(model, name, version)
        proto_file: ProtoFile = {
            "syntax": "proto3" if self.use_proto3 else "proto2",
            "package": self.package,
            "imports": sorted(self._required_imports),
            "messages": [message],
            "enums": [],
        }

        return proto_file

    def proto_file_to_string(self: Self, proto_file: ProtoFile) -> str:
        """Convert ProtoFile definition to .proto file string.

        Args:
            proto_file: Proto file definition.

        Returns:
            Proto file content as string.
        """
        lines = []

        lines.append(f'syntax = "{proto_file["syntax"]}";')
        lines.append("")

        if "package" in proto_file:
            lines.append(f"package {proto_file['package']};")
            lines.append("")

        if proto_file.get("imports"):
            lines.extend(
                f'import "{import_path}";' for import_path in proto_file["imports"]
            )
            lines.append("")

        if proto_file.get("options"):
            for key, value in proto_file["options"].items():
                lines.append(f'option {key} = "{value}";')
            lines.append("")

        for enum in proto_file.get("enums", []):
            lines.extend(self._enum_to_string(enum))
            lines.append("")

        for message in proto_file.get("messages", []):
            lines.extend(self._message_to_string(message))

        return "\n".join(lines)

    def _message_to_string(
        self: Self, message: ProtoMessage, indent: int = 0
    ) -> list[str]:
        """Convert ProtoMessage to string lines.

        Args:
            message: Proto message definition.
            indent: Current indentation level.

        Returns:
            List of string lines.
        """
        lines: list[str] = []
        indent_str = "  " * indent

        if message.get("comment"):
            lines.extend(
                f"{indent_str}// {comment_line}"
                for comment_line in message["comment"].split("\n")
            )

        lines.append(f"{indent_str}message {message['name']} {{")
        lines.extend(self._render_nested_types(message, indent + 1))
        lines.extend(self._render_fields_and_oneofs(message, indent + 1))
        lines.append(f"{indent_str}}}")

        return lines

    def _render_nested_types(
        self: Self, message: ProtoMessage, indent: int
    ) -> list[str]:
        """Render nested enums and messages.

        Args:
            message: Proto message definition.
            indent: Current indentation level.

        Returns:
            List of string lines.
        """
        lines: list[str] = []

        for nested_enum in message.get("nested_enums", []):
            nested_lines = self._enum_to_string(nested_enum, indent)
            lines.extend(nested_lines)
            lines.append("")

        for nested_message in message.get("nested_messages", []):
            nested_lines = self._message_to_string(nested_message, indent)
            lines.extend(nested_lines)
            lines.append("")

        return lines

    def _render_fields_and_oneofs(
        self: Self, message: ProtoMessage, indent: int
    ) -> list[str]:
        """Render fields and oneofs in definition order.

        Args:
            message: Proto message definition.
            indent: Current indentation level.

        Returns:
            List of string lines.
        """
        lines: list[str] = []
        field_order = message.get("field_order")

        oneof_field_names = self._get_oneof_field_names(message)

        if field_order is not None:
            lines.extend(
                self._render_with_field_order(
                    message, field_order, oneof_field_names, indent
                )
            )
        else:
            lines.extend(
                self._render_without_field_order(message, oneof_field_names, indent)
            )

        return lines

    def _get_oneof_field_names(self: Self, message: ProtoMessage) -> set[str]:
        """Get all field names that belong to oneofs.

        Args:
            message: Proto message definition.

        Returns:
            Set of field names that are part of oneofs.
        """
        oneof_field_names = set()
        for oneof in message.get("oneofs", []):
            oneof_field_names.update(oneof.get("fields", []))
        return oneof_field_names

    def _render_with_field_order(
        self: Self,
        message: ProtoMessage,
        field_order: list[tuple[str, str]],
        oneof_field_names: set[str],
        indent: int,
    ) -> list[str]:
        """Render fields and oneofs using field_order.

        Args:
            message: Proto message definition.
            field_order: List of (field_name, type) tuples indicating order.
            oneof_field_names: Set of field names that are part of oneofs.
            indent: Current indentation level.

        Returns:
            List of string lines.
        """
        lines: list[str] = []

        # Lookup dictionaries
        oneofs_dict = {oneof["name"]: oneof for oneof in message.get("oneofs", [])}
        fields_dict = {field["name"]: field for field in message["fields"]}

        for field_name, field_type in field_order:
            if field_type == "oneof":
                oneof = oneofs_dict.get(field_name)
                if oneof:
                    lines.extend(self._oneof_to_string(oneof, message, indent))
                    lines.append("")
            elif field_type == "field":
                field = fields_dict.get(field_name)
                if field:
                    lines.extend(self._field_to_string(field, indent))

        return lines

    def _render_without_field_order(
        self: Self,
        message: ProtoMessage,
        oneof_field_names: set[str],
        indent: int,
    ) -> list[str]:
        """Render fields and oneofs without field_order.

        Args:
            message: Proto message definition.
            oneof_field_names: Set of field names that are part of oneofs.
            indent: Current indentation level.

        Returns:
            List of string lines.
        """
        lines: list[str] = []

        for oneof in message.get("oneofs", []):
            lines.extend(self._oneof_to_string(oneof, message, indent))
            lines.append("")

        for field in message["fields"]:
            if field["name"] not in oneof_field_names:
                lines.extend(self._field_to_string(field, indent))

        return lines

    def _oneof_to_string(
        self: Self, oneof: ProtoOneOf, message: ProtoMessage, indent: int = 0
    ) -> list[str]:
        """Convert ProtoOneOf to string lines.

        Args:
            oneof: Proto oneof definition.
            message: Parent message containing the fields.
            indent: Current indentation level.

        Returns:
            List of string lines.
        """
        lines: list[str] = []
        indent_str = "  " * indent

        if oneof.get("comment"):
            lines.extend(
                f"{indent_str}// {comment_line}"
                for comment_line in oneof["comment"].split("\n")
            )

        lines.append(f"{indent_str}oneof {oneof['name']} {{")

        for field_name in oneof.get("fields", []):
            for field in message["fields"]:
                if field["name"] == field_name:
                    field_lines = self._field_to_string(field, indent + 1)
                    lines.extend(field_lines)
                    break

        lines.append(f"{indent_str}}}")

        return lines

    def _field_to_string(self: Self, field: ProtoField, indent: int = 0) -> list[str]:
        """Convert ProtoField to string lines.

        Args:
            field: Proto field definition.
            indent: Current indentation level.

        Returns:
            List of string lines.
        """
        lines = []
        indent_str = "  " * indent

        if field.get("comment"):
            lines.extend(
                [
                    f"{indent_str}// {comment_line}"
                    for comment_line in field["comment"].split("\n")
                ]
            )

        label = field.get("label", "")
        if label:
            field_line = (
                f"{indent_str}{label} {field['type']} {field['name']} = "
                f"{field['number']};"
            )
        else:
            field_line = (
                f"{indent_str}{field['type']} {field['name']} = {field['number']};"
            )

        lines.append(field_line)

        return lines

    def _enum_to_string(self: Self, enum: ProtoEnum, indent: int = 0) -> list[str]:
        """Convert ProtoEnum to string lines.

        Args:
            enum: Proto enum definition.
            indent: Current indentation level.

        Returns:
            List of string lines.
        """
        lines: list[str] = []
        indent_str = "  " * indent

        if enum.get("comment"):
            lines.extend(
                f"{indent_str}// {comment_line}"
                for comment_line in enum["comment"].split("\n")
            )

        lines.append(f"{indent_str}enum {enum['name']} {{")

        for value_name, value_number in enum["values"].items():
            lines.append(f"{indent_str}  {value_name} = {value_number};")

        lines.append(f"{indent_str}}}")

        return lines


class ProtoExporter:
    """Export Pydantic models to Protocol Buffer .proto files.

    This class provides methods to export individual schemas or all schemas
    from a model registry to .proto files.
    """

    def __init__(
        self: Self,
        registry: Registry,
        package: str = "com.example",
        include_comments: bool = True,
        use_proto3: bool = True,
    ) -> None:
        """Initialize the Protocol Buffer exporter.

        Args:
            registry: Model registry instance.
            package: Protobuf package name.
            include_comments: Whether to include documentation as comments.
            use_proto3: Use proto3 syntax (True) or proto2 (False).
        """
        self._registry = registry
        self.generator = ProtoSchemaGenerator(
            package=package,
            include_comments=include_comments,
            use_proto3=use_proto3,
        )

    def export_schema(
        self: Self,
        name: str,
        version: str | ModelVersion,
        output_path: str | Path | None = None,
    ) -> str:
        """Export a single model version as a Protocol Buffer schema.

        Args:
            name: Model name.
            version: Model version.
            output_path: Optional file path to save schema (.proto file).

        Returns:
            Protocol Buffer file as a string.

        Example:
            ```python
            exporter = ProtoExporter(manager._registry, package="com.myapp")

            # Export and save
            proto_file = exporter.export_schema("User", "1.0.0", "protos/user_v1.proto")

            # Or just get the proto definition
            proto_file = exporter.export_schema("User", "1.0.0")
            ```
        """
        model = self._registry.get_model(name, version)
        proto_file = self.generator.generate_proto_file(model, name, version)
        proto_content = self.generator.proto_file_to_string(proto_file)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(proto_content)

        return proto_content

    def export_all_schemas(
        self: Self,
        output_dir: str | Path,
    ) -> dict[str, dict[str, str]]:
        """Export all registered models as Protocol Buffer schemas.

        Args:
            output_dir: Directory to save .proto files.

        Returns:
            Dictionary mapping model names to version to Protocol Buffer file as a
            string.

        Example:
            ```python
            exporter = ProtoExporter(manager._registry, package="com.myapp")
            proto_files = exporter.export_all_schemas("protos/")

            # Creates files like:
            # protos/User_v1_0_0.proto
            # protos/User_v2_0_0.proto
            # protos/Order_v1_0_0.proto
            ```
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_schemas: dict[str, dict[str, str]] = {}

        for model_name in self._registry.list_models():
            all_schemas[model_name] = {}
            versions = self._registry.get_versions(model_name)

            for version in versions:
                model = self._registry.get_model(model_name, version)
                proto_file = self.generator.generate_proto_file(
                    model, model_name, version
                )

                version_str = str(version).replace(".", "_")
                filename = f"{model_name}_v{version_str}.proto"
                filepath = output_dir / filename

                proto_content = self.generator.proto_file_to_string(proto_file)
                filepath.write_text(proto_content)

                all_schemas[model_name][str(version)] = proto_content

        return all_schemas
