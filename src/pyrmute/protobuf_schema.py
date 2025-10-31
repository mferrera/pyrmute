"""Protocol Buffer schema generation from Pydantic models."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Self, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from ._protobuf_types import ProtoEnum, ProtoField, ProtoFile, ProtoMessage, ProtoOneOf
from ._registry import Registry
from ._schema_generator import SchemaGeneratorBase
from ._type_inspector import TypeInspector
from .model_version import ModelVersion


@dataclass
class ProtoTypeInfo:
    """Information about a converted Protocol Buffer type."""

    type_name: str
    is_repeated: bool = False


class ProtoSchemaGenerator(SchemaGeneratorBase[ProtoFile]):
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
        include_docs: bool = True,
        use_proto3: bool = True,
    ) -> None:
        """Initialize the Protocol Buffer schema generator.

        Args:
            package: Protobuf package name (e.g., "com.mycompany.events").
            include_docs: Whether to include field descriptions as comments.
            use_proto3: Use proto3 syntax (True) or proto2 (False).
        """
        super().__init__(include_docs=include_docs)
        self.package = package
        self.use_proto3 = use_proto3
        self._required_imports: set[str] = set()
        self._field_counter = 1
        self._tuple_counter = 0
        self._collected_enums: list[ProtoEnum] = []
        self._collected_nested_messages: list[ProtoMessage] = []
        self._current_model_class = ""
        self._versioned_name_map: dict[str, str] = {}
        self._types_seen_collecting_enums: set[str] = set()

    def _reset_state(self: Self) -> None:
        """Reset internal state before generating a new schema."""
        super()._reset_state()
        self._required_imports = set()
        self._field_counter = 1
        self._tuple_counter = 0
        self._collected_enums = []
        self._collected_nested_messages = []
        self._current_model_class = ""
        self._types_seen_collecting_enums = set()

    def _generate_proto_schema(  # noqa: C901, PLR0912
        self: Self,
        model: type[BaseModel],
        name: str,
        version: str | ModelVersion,
    ) -> ProtoMessage:
        """Generate a Protocol Buffer schema from a Pydantic model."""
        self._reset_state()
        if not hasattr(self, "_versioned_name_map") or not self._versioned_name_map:
            self._versioned_name_map = {}

        self._versioned_name_map[model.__name__] = name
        self._current_model_class = model.__name__

        self._current_model = (model.__name__, name)
        self._types_seen.add(model.__name__)

        self._collect_nested_models(model)

        for nested_class_name in self._nested_models:
            if nested_class_name != model.__name__:
                self._versioned_name_map[nested_class_name] = nested_class_name

            self._collect_enums_from_model(model)

        for nested_model in self._nested_models.values():
            if nested_model.__name__ != model.__name__:
                self._collect_enums_from_model(nested_model)

        for nested_class_name, nested_model in self._nested_models.items():
            if nested_class_name != model.__name__:
                proto_name = self._versioned_name_map[nested_class_name]
                nested_msg = self._generate_nested_message(nested_model, proto_name)
                self._collected_nested_messages.append(nested_msg)

        message: ProtoMessage = {
            "name": name,
            "fields": [],
            "nested_messages": [],
            "nested_enums": [],
            "field_order": [],
        }

        if self.include_docs and model.__doc__:
            message["comment"] = model.__doc__.strip()

        oneofs_to_add: list[ProtoOneOf] = []

        for field_name, field_info in model.model_fields.items():
            if TypeInspector.is_union_requiring_oneof(field_info.annotation):
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

        return message

    def _collect_enums_from_model(self: Self, model: type[BaseModel]) -> None:
        """Recursively collect all enums from a model and its nested models.

        Args:
            model: Pydantic model class to scan for enums.
        """
        if not hasattr(self, "_types_seen_collecting_enums"):
            self._types_seen_collecting_enums = set()

        if model.__name__ in self._types_seen_collecting_enums:
            return

        self._types_seen_collecting_enums.add(model.__name__)

        for field_info in model.model_fields.values():
            self._collect_enums_from_type(field_info.annotation)

    def _collect_enums_from_type(self: Self, python_type: Any) -> None:  # noqa: C901
        """Recursively collect enums from a type annotation.

        Args:
            python_type: Python type to scan for enums.
        """
        if (
            python_type is None
            or python_type is type(None)
            or isinstance(python_type, str)
        ):
            return

        if TypeInspector.is_enum(python_type):
            enum_name = python_type.__name__
            if enum_name not in self._types_seen:
                self._types_seen.add(enum_name)
                enum_def = self._generate_enum_schema(python_type)
                self._collected_enums.append(enum_def)
            return

        if TypeInspector.is_base_model(python_type):
            self._collect_enums_from_model(python_type)
            return

        origin = get_origin(python_type)
        args = get_args(python_type)

        if TypeInspector.is_union_type(origin):
            for arg in args:
                if arg is not type(None):
                    self._collect_enums_from_type(arg)
            return

        if TypeInspector.is_list_like(origin) or origin is tuple:
            for arg in args:
                self._collect_enums_from_type(arg)
            return

        if TypeInspector.is_dict_like(origin, python_type):
            for arg in args:
                self._collect_enums_from_type(arg)

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
        non_none_args = TypeInspector.get_non_none_union_args(field_info.annotation)

        oneof_name = f"{field_name}_value"
        oneof_field_names: list[str] = []
        fields: list[ProtoField] = []
        used_variant_names: set[str] = set()

        for arg in non_none_args:
            variant_name = self._get_union_variant_name(arg, field_name)
            proto_type = self._convert_type(arg, None)

            original_variant_name = variant_name
            counter = 1
            while variant_name in used_variant_names:
                variant_name = f"{original_variant_name}_{counter}"
                counter += 1
            used_variant_names.add(variant_name)

            if proto_type.type_name.startswith("map"):
                raise ValueError(
                    "Cannot encode unions with Python dictionaries to ProtoBuf: "
                    "Map fields fields are not allowed in oneofs. Relevant model:\n"
                    f"{model.__name__}\n"
                    f"    {field_name}: {field_info.annotation}"
                )

            if proto_type.is_repeated:
                raise ValueError(
                    "Cannot encode unions with Python iterables to ProtoBuf: "
                    "Fields in oneofs must not have labels (required / optional / "
                    "repeated). Relevant model:\n"
                    f"{model.__name__}\n"
                    f"    {field_name}: {field_info.annotation}"
                )

            field_schema: ProtoField = {
                "name": variant_name,
                "type": proto_type.type_name,
                "number": self._field_counter,
                "oneof_group": oneof_name,
            }
            self._field_counter += 1

            if self.include_docs:
                type_name = self._get_type_display_name(arg)
                field_schema["comment"] = f"{field_name} when type is {type_name}"

            fields.append(field_schema)
            oneof_field_names.append(variant_name)

        oneof: ProtoOneOf = {
            "name": oneof_name,
            "fields": oneof_field_names,
        }

        if self.include_docs and field_info.description:
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
        origin = get_origin(typ)
        if origin is not None:
            args = get_args(typ)
            if args:
                typ = args[0]

        if typ in self._BASIC_TYPE_MAPPING:
            type_name = self._BASIC_TYPE_MAPPING[typ].replace("<", "_").replace(">", "")
            return f"{base_name}_{type_name}"

        if TypeInspector.is_base_model(typ):
            model_name = typ.__name__
            proto_name = self._versioned_name_map.get(model_name, model_name)
            return f"{base_name}_{proto_name.lower()}"

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
        origin = get_origin(typ)
        if origin is not None:
            args = get_args(typ)
            if args:
                typ = args[0]

        if TypeInspector.is_base_model(typ):
            model_name = typ.__name__
            return self._versioned_name_map.get(model_name, model_name)

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

        if self.include_docs and field_info.description:
            field_schema["comment"] = field_info.description

        context = self._analyze_field(field_info)

        proto_type = self._convert_type(field_info.annotation, field_info)
        if proto_type.is_repeated:
            field_schema["label"] = "repeated"
        elif context["is_optional"] or context["has_default"]:  # CHANGED: use context
            field_schema["label"] = "optional"
        elif not self.use_proto3:
            field_schema["label"] = "required"

        field_schema["type"] = proto_type.type_name
        return field_schema

    def _convert_type(  # noqa: PLR0911, C901
        self: Self,
        python_type: Any,
        field_info: FieldInfo | None = None,
        field_name: str | None = None,
    ) -> ProtoTypeInfo:
        """Convert Python type annotation to Protocol Buffer type."""
        if python_type is None:
            return ProtoTypeInfo(type_name="string", is_repeated=False)

        if python_type is int and field_info:
            return ProtoTypeInfo(
                type_name=self._optimize_int_type(field_info), is_repeated=False
            )

        if python_type in self._BASIC_TYPE_MAPPING:
            return ProtoTypeInfo(
                type_name=self._BASIC_TYPE_MAPPING[python_type], is_repeated=False
            )

        if python_type in self._WELLKNOWN_TYPE_MAPPING:
            return self._convert_wellknown(python_type)

        if TypeInspector.is_enum(python_type):
            return self._convert_enum(python_type)

        origin = get_origin(python_type)

        if TypeInspector.is_union_type(origin):
            return self._convert_union(python_type, field_info)

        if origin is tuple or (
            hasattr(origin, "__origin__") and origin.__origin__ is tuple
        ):
            return self._convert_tuple(python_type, field_name)

        if TypeInspector.is_list_like(origin) or origin in (set, frozenset):
            return self._convert_list(python_type)

        if TypeInspector.is_dict_like(origin, python_type):
            return self._convert_dict(python_type)

        if TypeInspector.is_base_model(python_type):
            return self._convert_basemodel(python_type)

        return ProtoTypeInfo(type_name="string", is_repeated=False)

    def _convert_wellknown(self: Self, python_type: type) -> ProtoTypeInfo:
        """Convert well-known types to protobuf."""
        proto_type, import_path = self._WELLKNOWN_TYPE_MAPPING[python_type]
        if import_path:
            self._required_imports.add(import_path)
        return ProtoTypeInfo(type_name=proto_type, is_repeated=False)

    def _convert_union(
        self: Self, python_type: Any, field_info: FieldInfo | None
    ) -> ProtoTypeInfo:
        """Convert union types to protobuf."""
        non_none_args = TypeInspector.get_non_none_union_args(python_type)
        if len(non_none_args) == 1:
            return self._convert_type(non_none_args[0], field_info)
        return ProtoTypeInfo(type_name="string", is_repeated=False)

    def _convert_tuple(
        self: Self, python_type: Any, field_name: str | None
    ) -> ProtoTypeInfo:
        """Convert tuple types to protobuf."""
        elements = TypeInspector.get_tuple_element_types(python_type)

        if TypeInspector.is_variable_length_tuple(python_type):
            proto_type = self._convert_type(elements[0], None)
            return ProtoTypeInfo(type_name=proto_type.type_name, is_repeated=True)

        if all(el == elements[0] for el in elements):
            proto_type = self._convert_type(elements[0], None)
            return ProtoTypeInfo(proto_type.type_name, True)

        return self._convert_hetereogeneous_tuple(elements, field_name)

    def _convert_hetereogeneous_tuple(
        self: Self, elements: list[Any], field_name: str | None
    ) -> ProtoTypeInfo:
        """Convert heterogeneous tuple to nested protobuf message."""
        if field_name:
            model_name = f"{self._current_model[1]}{field_name.title()}Tuple"
        else:
            self._tuple_counter += 1
            model_name = f"{self._current_model[1]}Tuple{self._tuple_counter}"

        if model_name not in self._types_seen:
            self._types_seen.add(model_name)

            field_definitions: dict[str, Any] = {
                f"field{idx}": (field_type, ...)
                for idx, field_type in enumerate(elements)
            }
            nested_tuple_model = create_model(model_name, **field_definitions)

            nested_msg = self._generate_nested_message(nested_tuple_model, model_name)
            self._collected_nested_messages.append(nested_msg)

        if model_name == self._versioned_name_map.get(self._current_model_class):
            return ProtoTypeInfo(
                type_name=self._versioned_name_map[self._current_model_class],
                is_repeated=False,
            )

        return ProtoTypeInfo(type_name=model_name, is_repeated=False)

    def _convert_list(self: Self, python_type: Any) -> ProtoTypeInfo:
        """Convert list/set types to protobuf."""
        args = get_args(python_type)
        if args:
            proto_type = self._convert_type(args[0], None)
            return ProtoTypeInfo(type_name=proto_type.type_name, is_repeated=True)
        return ProtoTypeInfo(type_name="string", is_repeated=True)

    def _convert_dict(self: Self, python_type: Any) -> ProtoTypeInfo:
        """Convert dict types to protobuf map."""
        args = get_args(python_type)
        if args and len(args) == 2:  # noqa: PLR2004
            key_type = self._convert_type(args[0], None)
            value_type = self._convert_type(args[1], None)
            return ProtoTypeInfo(
                type_name=f"map<{key_type.type_name}, {value_type.type_name}>",
                is_repeated=False,
            )
        return ProtoTypeInfo(type_name="map<string, string>", is_repeated=False)

    def _convert_basemodel(self: Self, python_type: type[BaseModel]) -> ProtoTypeInfo:
        """Convert nested BaseModel to protobuf message."""
        model_name = python_type.__name__
        proto_name = self._versioned_name_map.get(model_name, model_name)

        if model_name not in self._types_seen:
            self._types_seen.add(model_name)

            if model_name not in self._versioned_name_map:
                self._versioned_name_map[model_name] = model_name
                proto_name = model_name

            nested_msg = self._generate_nested_message(python_type, proto_name)
            self._collected_nested_messages.append(nested_msg)

        if model_name == self._current_model_class:
            return ProtoTypeInfo(
                type_name=self._versioned_name_map[model_name], is_repeated=False
            )

        return ProtoTypeInfo(type_name=proto_name, is_repeated=False)

    def _optimize_int_type(self: Self, field_info: FieldInfo) -> str:
        """Optimize integer type based on field constraints.

        Args:
            field_info: Pydantic field info with metadata.

        Returns:
            Optimized protobuf integer type.
        """
        if TypeInspector.is_unsigned_int(field_info):
            if TypeInspector.can_fit_in_32bit_uint(field_info):
                return "uint32"
            return "uint64"

        return "int32"

    def _convert_enum(self: Self, enum_class: type[Enum]) -> ProtoTypeInfo:
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

        return ProtoTypeInfo(type_name=enum_name, is_repeated=False)

    def _generate_enum_schema(self: Self, enum_class: type[Enum]) -> ProtoEnum:
        """Generate a Protocol Buffer enum from a Python Enum.

        Args:
            enum_class: Python Enum class.

        Returns:
            Protocol Buffer enum definition.
        """
        enum_def: ProtoEnum = {"name": enum_class.__name__, "values": {}}

        if self.include_docs and enum_class.__doc__:
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

        if self.include_docs and model.__doc__:
            message["comment"] = model.__doc__.strip()

        saved_field_counter = self._field_counter
        self._field_counter = 1

        oneofs_to_add: list[ProtoOneOf] = []

        for field_name, field_info in model.model_fields.items():
            if TypeInspector.is_union_requiring_oneof(field_info.annotation):
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

        self._field_counter = saved_field_counter

        return message

    def generate_schema(
        self: Self,
        model: type[BaseModel],
        name: str,
        version: str | ModelVersion | None = None,
        registry_name_map: dict[str, str] | None = None,
    ) -> ProtoFile:
        """Generate a complete .proto file definition.

        Args:
            model: Pydantic model class.
            name: Model name.
            version: Model version.
            registry_name_map: Optional mapping of class names to registry names.

        Returns:
            Complete proto file definition.
        """
        self._reset_state()

        if registry_name_map:
            self._versioned_name_map = registry_name_map.copy()

        message = self._generate_proto_schema(model, name, version or "1.0.0")

        all_messages = []
        all_messages.extend(self._collected_nested_messages)
        all_messages.append(message)

        proto_file: ProtoFile = {
            "syntax": "proto3" if self.use_proto3 else "proto2",
            "package": self.package,
            "imports": sorted(self._required_imports),
            "messages": all_messages,
            "enums": self._collected_enums,
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
        include_docs: bool = True,
        use_proto3: bool = True,
    ) -> None:
        """Initialize the Protocol Buffer exporter.

        Args:
            registry: Model registry instance.
            package: Protobuf package name.
            include_docs: Whether to include documentation as comments.
            use_proto3: Use proto3 syntax (True) or proto2 (False).
        """
        self._registry = registry
        self.generator = ProtoSchemaGenerator(
            package=package,
            include_docs=include_docs,
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

        registry_name_map: dict[str, str] = {}
        for model_name in self._registry.list_models():
            for model_version in self._registry.get_versions(model_name):
                registered_model = self._registry.get_model(model_name, model_version)
                registry_name_map[registered_model.__name__] = model_name

        proto_file = self.generator.generate_schema(
            model, name, version, registry_name_map
        )
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
                proto_file = self.generator.generate_schema(model, model_name, version)

                version_str = str(version).replace(".", "_")
                filename = f"{model_name}_v{version_str}.proto"
                filepath = output_dir / filename

                proto_content = self.generator.proto_file_to_string(proto_file)
                filepath.write_text(proto_content)

                all_schemas[model_name][str(version)] = proto_content

        return all_schemas
