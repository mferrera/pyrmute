"""TypeScript schema generation from Pydantic models."""

from collections.abc import Mapping
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Literal, Self, TypedDict, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ._registry import Registry
from ._schema_generator import SchemaGeneratorBase
from ._type_inspector import TypeInspector
from .model_version import ModelVersion


class TypeScriptConfig(TypedDict, total=False):
    """Configuration options for TypeScript schema generation."""

    include_docs: bool
    date_format: Literal["iso", "timestamp"]
    enum_style: Literal["union", "enum"]
    include_computed_fields: bool
    mark_computed_readonly: bool


class TypeScriptSchemaGenerator(SchemaGeneratorBase[str]):
    """Generates TypeScript schemas from Pydantic models."""

    _BASIC_TYPE_MAPPING: Mapping[type, str] = {
        str: "string",
        int: "number",
        float: "number",
        bool: "boolean",
        bytes: "string",
    }

    _DATE_TYPE_MAPPING: Mapping[type, Mapping[str, str]] = {
        datetime: {"iso": "string", "timestamp": "number"},
        date: {"iso": "string", "timestamp": "number"},
        time: {"iso": "string", "timestamp": "number"},
    }

    def __init__(
        self: Self,
        style: Literal["interface", "type", "zod"] = "interface",
        config: TypeScriptConfig | None = None,
    ) -> None:
        """Initialize the TypeScript schema generator.

        Args:
            style: Output style - 'interface', 'type', or 'zod'.
            config: Optional configuration for schema generation.
        """
        include_docs: bool = config.get("include_docs", True) if config else True
        super().__init__(include_docs=include_docs)
        self.style = style
        self.config = config or TypeScriptConfig()
        self._types_seen_collecting_enums: set[str] = set()
        self._versioned_name_map: dict[str, str] = {}

    def _reset_state(self: Self) -> None:
        """Reset internal state before generating a new schema."""
        super()._reset_state()
        self._types_seen_collecting_enums = set()
        self._versioned_name_map = {}

    def generate_schema(
        self: Self,
        model: type[BaseModel],
        name: str,
        version: str | ModelVersion | None = None,
        registry_name_map: dict[str, str] | None = None,
    ) -> str:
        """Generate a TypeScript schema from a Pydantic model.

        Args:
            model: Pydantic model class.
            name: Model name.
            version: Optional model version.
            registry_name_map: Optional mapping of class names to registry names.

        Returns:
            TypeScript schema code as a string.
        """
        self._reset_state()
        self._types_seen_collecting_enums = set()
        self._versioned_name_map = {}

        versioned_name = self._get_versioned_name(name, version)
        self._versioned_name_map[model.__name__] = versioned_name

        if self.config.get("enum_style") == "enum":
            self._collect_enums_from_model(model)

        self._collect_nested_models(model)

        schemas: list[str] = []

        if self.style == "zod":
            schemas.append("import { z } from 'zod';")
            schemas.append("")

        if self.config.get("enum_style") == "enum" and self._enums_encountered:
            # Get the right method.
            enum_declarations = (
                self._enum_to_zod_declaration
                if self.style == "zod"
                else self._enum_to_typescript
            )
            schemas.extend(
                enum_declarations(enum_class)
                for enum_class in self._enums_encountered.values()
            )
            schemas.append("")

        for nested_name, nested_model in self._nested_models.items():
            if nested_name != model.__name__:
                nested_schema = self._generate_schema_for_model(
                    nested_model, nested_name
                )
                schemas.append(nested_schema)

        main_schema = self._generate_schema_for_model(model, versioned_name)
        schemas.append(main_schema)

        return "\n\n".join(schemas)

    def _generate_schema_for_model(
        self: Self, model: type[BaseModel], type_name: str
    ) -> str:
        """Generate schema for a specific model."""
        if self.style == "zod":
            return self._generate_zod_schema(model, type_name)
        if self.style == "type":
            return self._generate_type_alias(model, type_name)
        return self._generate_interface(model, type_name)

    def _collect_enums_from_model(self: Self, model: type[BaseModel]) -> None:
        """Recursively collect all enums from a model and its nested models.

        Args:
            model: Pydantic model class to scan for enums.
        """
        if model.__name__ in self._types_seen_collecting_enums:
            return

        self._types_seen_collecting_enums.add(model.__name__)

        for field_info in model.model_fields.values():
            self._collect_enums_from_type(field_info.annotation)

        if hasattr(model, "model_computed_fields"):
            for computed_field_info in model.model_computed_fields.values():
                if hasattr(computed_field_info, "return_type"):
                    self._collect_enums_from_type(computed_field_info.return_type)

    def _collect_enums_from_type(  # noqa: C901
        self: Self, python_type: Any
    ) -> None:
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
            enum_style = self.config.get("enum_style", "union")
            if enum_style == "enum":
                self._enums_encountered[python_type.__name__] = python_type
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

    def _get_versioned_name(
        self: Self, name: str, version: str | ModelVersion | None
    ) -> str:
        """Generate versioned name for a model."""
        if version is None:
            return name

        version_str = str(version).replace(".", "_")
        return f"{name}V{version_str}"

    def _generate_interface(self: Self, model: type[BaseModel], name: str) -> str:
        """Generate TypeScript interface."""
        lines: list[str] = []

        if model.__doc__:
            doc_lines = model.__doc__.strip().split("\n")
            lines.append("/**")
            lines.extend(f" * {doc_line}" for doc_line in doc_lines)
            lines.append(" */")

        lines.append(f"export interface {name} {{")

        field_lines = self._generate_field_lines(model)
        lines.extend(field_lines)

        if self._model_allows_extra(model):
            lines.append("  [key: string]: any;")

        lines.append("}")
        return "\n".join(lines)

    def _generate_type_alias(self: Self, model: type[BaseModel], name: str) -> str:
        """Generate TypeScript type alias."""
        if not model.model_fields and not hasattr(model, "model_computed_fields"):
            return f"export type {name} = Record<string, never>;"

        lines = [f"export type {name} = {{"]

        field_lines = self._generate_field_lines(model)
        lines.extend(field_lines)

        if self._model_allows_extra(model):
            lines.append("  [key: string]: any;")

        lines.append("};")
        return "\n".join(lines)

    def _generate_field_lines(self: Self, model: type[BaseModel]) -> list[str]:
        """Generate field definition lines for interface or type."""
        lines: list[str] = []

        for field_name, field_info in model.model_fields.items():
            ts_name = self._get_field_ts_name(field_name, field_info)
            ts_type = self._python_type_to_typescript(field_info.annotation, field_info)

            optional = (
                ""
                if field_info.is_required()
                or (
                    get_origin(field_info.annotation) is Literal
                    and field_info.default != PydanticUndefined
                    and field_info.default in get_args(field_info.annotation)
                )
                else "?"
            )

            if field_info.description:
                lines.append(f"  /** {field_info.description} */")

            lines.append(f"  {ts_name}{optional}: {ts_type};")

        include_computed = self.config.get("include_computed_fields", True)
        if hasattr(model, "model_computed_fields") and include_computed:
            mark_readonly = self.config.get("mark_computed_readonly", False)
            readonly_marker = "readonly " if mark_readonly else ""

            for field_name, computed_field_info in model.model_computed_fields.items():
                if hasattr(computed_field_info, "return_type"):
                    ts_name = self._get_computed_field_ts_name(
                        field_name, computed_field_info
                    )
                    ts_type = self._python_type_to_typescript(
                        computed_field_info.return_type, None
                    )
                    lines.append(f"  {readonly_marker}{ts_name}: {ts_type};")

        return lines

    def _generate_zod_schema(self: Self, model: type[BaseModel], name: str) -> str:
        """Generate Zod schema."""
        schema_name = f"{name}Schema"

        fields = self._generate_zod_fields(model)

        schema_def = f"z.object({{{fields}}})"
        if self._model_allows_extra(model):
            schema_def += ".passthrough()"

        lines = [
            f"export const {schema_name} = {schema_def};",
            f"export type {name} = z.infer<typeof {schema_name}>;",
        ]
        return "\n".join(lines)

    def _generate_zod_fields(self: Self, model: type[BaseModel]) -> str:
        """Generate Zod field definitions."""
        field_parts: list[str] = []

        for field_name, field_info in model.model_fields.items():
            ts_name = self._get_field_ts_name(field_name, field_info)
            validator = self._python_type_to_zod(field_info.annotation, field_info)
            field_parts.append(f"\n  {ts_name}: {validator},")

        if hasattr(model, "model_computed_fields"):
            for field_name, computed_field_info in model.model_computed_fields.items():
                if hasattr(computed_field_info, "return_type"):
                    ts_name = self._get_computed_field_ts_name(
                        field_name, computed_field_info
                    )
                    validator = self._python_type_to_zod(
                        computed_field_info.return_type, None
                    )
                    field_parts.append(f"\n  {ts_name}: {validator},")

        if field_parts:
            return "".join(field_parts) + "\n"
        return ""

    def _get_field_ts_name(self: Self, field_name: str, field_info: FieldInfo) -> str:
        """Get TypeScript field name, using alias if present."""
        if field_info.alias:
            return field_info.alias
        if field_info.serialization_alias:
            return field_info.serialization_alias
        return field_name

    def _get_computed_field_ts_name(
        self: Self, field_name: str, field_info: Any
    ) -> str:
        """Get TypeScript name for computed field, using alias if present."""
        if hasattr(field_info, "alias") and field_info.alias:
            return str(field_info.alias)
        return field_name

    def _model_allows_extra(self: Self, model: type[BaseModel]) -> bool:
        """Check if model allows extra fields."""
        return model.model_config.get("extra") == "allow"

    def _python_type_to_typescript(  # noqa: PLR0911, PLR0912, C901
        self: Self, python_type: Any, field_info: FieldInfo | None = None
    ) -> str:
        """Convert Python type to TypeScript type."""
        if python_type is None or python_type is type(None):
            return "null"

        if isinstance(python_type, str):
            return "any"

        if python_type in self._BASIC_TYPE_MAPPING:
            return self._BASIC_TYPE_MAPPING[python_type]

        if python_type in self._DATE_TYPE_MAPPING:
            date_format = self.config.get("date_format", "iso")
            return self._DATE_TYPE_MAPPING[python_type].get(date_format, "string")

        if python_type in (UUID, Path):
            return "string"

        if python_type is Decimal:
            return "number"

        if TypeInspector.is_enum(python_type):
            return self._enum_to_typescript_type(python_type)

        origin = get_origin(python_type)
        args = get_args(python_type)

        if TypeInspector.is_union_type(origin):
            return self._union_to_typescript(args, field_info)

        if origin is Literal:
            return self._literal_to_typescript(args)

        if TypeInspector.is_dict_like(origin, python_type):
            return self._dict_to_typescript(args, field_info)

        if TypeInspector.is_list_like(origin):
            return self._list_to_typescript(args, field_info)

        if origin is tuple:
            return self._tuple_to_typescript(python_type, field_info)

        if TypeInspector.is_base_model(python_type):
            return self._versioned_name_map.get(  # type: ignore[no-any-return]
                python_type.__name__, python_type.__name__
            )

        if hasattr(python_type, "__origin__") and python_type.__origin__ is Generic:
            return "any"

        return "any"

    def _union_to_typescript(
        self: Self, args: tuple[Any, ...], field_info: FieldInfo | None
    ) -> str:
        """Convert Union type to TypeScript."""
        non_none_types = [arg for arg in args if arg is not type(None)]
        has_none = type(None) in args

        if not non_none_types:
            return "null"

        if len(non_none_types) == 1:
            ts_type = self._python_type_to_typescript(non_none_types[0], field_info)
            if has_none:
                if field_info and not field_info.is_required():
                    return ts_type
                return f"{ts_type} | null"
            return ts_type

        union_types = [
            self._python_type_to_typescript(arg, field_info) for arg in non_none_types
        ]

        if has_none:
            if field_info and not field_info.is_required():
                pass
            else:
                union_types.append("null")

        unique_types = list(dict.fromkeys(union_types))
        return " | ".join(unique_types)

    def _literal_to_typescript(self: Self, args: tuple[Any, ...]) -> str:
        """Convert Literal type to TypeScript."""
        if not args:
            return "never"

        literal_values = [self._format_literal_value(arg) for arg in args]
        return " | ".join(literal_values)

    def _dict_to_typescript(
        self: Self, args: tuple[Any, ...], field_info: FieldInfo | None
    ) -> str:
        """Convert dict type to TypeScript."""
        if args and len(args) == 2:  # noqa: PLR2004
            value_type = self._python_type_to_typescript(args[1], None)
            return f"Record<string, {value_type}>"
        return "Record<string, any>"

    def _list_to_typescript(
        self: Self, args: tuple[Any, ...], field_info: FieldInfo | None
    ) -> str:
        """Convert list type to TypeScript."""
        if args:
            item_type = self._python_type_to_typescript(args[0], None)
            return f"{item_type}[]"
        return "any[]"

    def _tuple_to_typescript(
        self: Self, python_type: Any, field_info: FieldInfo | None
    ) -> str:
        """Convert tuple type to TypeScript."""
        element_types = TypeInspector.get_tuple_element_types(python_type)
        if not element_types:
            return "any[]"

        if len(element_types) == 2 and element_types[1] is Ellipsis:  # noqa: PLR2004
            item_type = self._python_type_to_typescript(element_types[0], None)
            return f"{item_type}[]"

        element_types = [
            self._python_type_to_typescript(element, None) for element in element_types
        ]
        return f"[{', '.join(element_types)}]"

    def _enum_to_typescript_type(self: Self, enum_class: type[Enum]) -> str:
        """Convert Python Enum to TypeScript type."""
        enum_style = self.config.get("enum_style", "union")

        if enum_style == "enum":
            return enum_class.__name__

        values = []
        for member in enum_class:
            if isinstance(member.value, str):
                values.append(f"'{member.value}'")
            elif isinstance(member.value, bool):
                values.append("true" if member.value else "false")
            elif member.value is None:
                values.append("null")
            else:
                values.append(str(member.value))

        return " | ".join(values)

    def _enum_to_typescript(self: Self, enum_class: type[Enum]) -> str:
        """Generate TypeScript enum declaration."""
        lines = [f"export enum {enum_class.__name__} {{"]

        for member in enum_class:
            if isinstance(member.value, str):
                lines.append(f"  {member.name} = '{member.value}',")
            else:
                lines.append(f"  {member.name} = {member.value},")

        lines.append("}")
        return "\n".join(lines)

    def _format_literal_value(self: Self, arg: Any) -> str:
        """Format a literal value for TypeScript/Zod."""
        if isinstance(arg, Enum):
            arg = arg.value

        if isinstance(arg, str):
            return f"'{arg}'"
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if arg is None:
            return "null"
        return str(arg)

    def _python_type_to_zod(  # noqa: PLR0911, PLR0912, C901
        self: Self, python_type: Any, field_info: FieldInfo | None
    ) -> str:
        """Convert Python type to Zod validator."""
        if python_type is None or python_type is type(None):
            return "z.null()"

        if isinstance(python_type, str):
            return "z.any()"

        # Handle BaseModel early
        if TypeInspector.is_base_model(python_type):
            schema_name = f"{python_type.__name__}Schema"
            if field_info and not field_info.is_required():
                return f"{schema_name}.optional()"
            return schema_name

        origin = get_origin(python_type)
        args = get_args(python_type)

        if TypeInspector.is_union_type(origin):
            return self._union_to_zod(args, field_info)

        if origin is Literal:
            if args:
                literal_values = [self._format_literal_value(arg) for arg in args]
                return (
                    f"z.literal({literal_values[0]})"
                    if len(args) == 1
                    else (
                        "z.union(["
                        f"{', '.join(f'z.literal({v})' for v in literal_values)}"
                        "])"
                    )
                )
            return "z.never()"

        if TypeInspector.is_dict_like(origin, python_type):
            if args and len(args) == 2:  # noqa: PLR2004
                value_validator = self._python_type_to_zod(args[1], None)
                return f"z.record({value_validator})"
            return "z.record(z.any())"

        if TypeInspector.is_list_like(origin):
            if args:
                item_validator = self._python_type_to_zod(args[0], None)
                return f"z.array({item_validator})"
            return "z.array(z.any())"

        if origin is tuple:
            if args:
                validators = [self._python_type_to_zod(a, field_info) for a in args]
                return f"z.tuple([{', '.join(validators)}])"
            return "z.array(z.any())"

        if python_type is str:
            return self._apply_string_constraints("z.string()", field_info)

        if python_type in (int, float):
            base = "z.number().int()" if python_type is int else "z.number()"
            return self._apply_numeric_constraints(base, field_info)

        if python_type is bool:
            return "z.boolean()"

        if python_type is datetime:
            return "z.string().datetime()"

        if python_type is date:
            return "z.string().date()"

        if python_type is time:
            return "z.string().time()"

        if python_type is UUID:
            return "z.string().uuid()"

        if python_type is Decimal:
            return "z.number()"

        if python_type is bytes:
            return "z.string()"

        if TypeInspector.is_enum(python_type):
            return self._enum_to_zod(python_type)

        return "z.any()"

    def _union_to_zod(
        self: Self, args: tuple[Any, ...], field_info: FieldInfo | None
    ) -> str:
        """Convert union type to Zod validator."""
        non_none_args = [arg for arg in args if arg is not type(None)]
        has_none = any(arg is type(None) for arg in args)

        if not non_none_args:
            return "z.null()"

        if len(non_none_args) == 1:
            base_validator = self._python_type_to_zod(non_none_args[0], None)
            if has_none:
                return (
                    f"{base_validator}.optional()"
                    if field_info and not field_info.is_required()
                    else f"{base_validator}.nullable()"
                )
            return base_validator

        validators = [self._python_type_to_zod(arg, None) for arg in non_none_args]
        union_validator = f"z.union([{', '.join(validators)}])"

        if has_none:
            return (
                f"{union_validator}.optional()"
                if field_info and not field_info.is_required()
                else f"{union_validator}.nullable()"
            )
        return union_validator

    def _apply_string_constraints(
        self: Self, base: str, field_info: FieldInfo | None
    ) -> str:
        """Apply string constraints to Zod validator."""
        if not field_info:
            return base

        constraints = TypeInspector.get_string_constraints(field_info)

        if constraints["pattern"]:
            base += f".regex(/{constraints['pattern']}/)"
        if constraints["min_length"] is not None:
            base += f".min({constraints['min_length']})"
        if constraints["max_length"] is not None:
            base += f".max({constraints['max_length']})"

        return base

    def _apply_numeric_constraints(
        self: Self, base: str, field_info: FieldInfo | None
    ) -> str:
        """Apply numeric constraints to Zod validator."""
        if not field_info:
            return base

        constraints = TypeInspector.get_numeric_constraints(field_info)

        if constraints["ge"] is not None:
            base += f".gte({constraints['ge']})"
        if constraints["gt"] is not None:
            base += f".gt({constraints['gt']})"
        if constraints["le"] is not None:
            base += f".lte({constraints['le']})"
        if constraints["lt"] is not None:
            base += f".lt({constraints['lt']})"

        return base

    def _enum_to_zod(self: Self, enum_class: type[Enum]) -> str:
        """Convert Python Enum to Zod validator."""
        values = [
            f"'{member.value}'" if isinstance(member.value, str) else str(member.value)
            for member in enum_class
        ]
        return f"z.enum([{', '.join(values)}])"

    def _enum_to_zod_declaration(self: Self, enum_class: type[Enum]) -> str:
        """Generate Zod enum declaration."""
        values = [
            f"'{member.value}'" if isinstance(member.value, str) else str(member.value)
            for member in enum_class
        ]
        return (
            f"export const {enum_class.__name__}Schema = z.enum([{', '.join(values)}]);"
        )


class TypeScriptExporter:
    """Export Pydantic models to TypeScript schema files."""

    def __init__(
        self: Self,
        registry: Registry,
        style: Literal["interface", "type", "zod"] = "interface",
        config: TypeScriptConfig | None = None,
    ) -> None:
        """Initialize the TypeScript exporter.

        Args:
            registry: Model registry instance.
            style: Output style - 'interface', 'type', or 'zod'.
            config: Optional configuration for schema generation.
        """
        self._registry = registry
        self.generator = TypeScriptSchemaGenerator(style=style, config=config)

    def export_schema(
        self: Self,
        name: str,
        version: str | ModelVersion,
        output_path: str | Path | None = None,
    ) -> str:
        """Export a single model version as a TypeScript schema.

        Args:
            name: Model name.
            version: Model version.
            output_path: Optional file path to save schema.

        Returns:
            TypeScript schema code.
        """
        model = self._registry.get_model(name, version)
        schema_code = self.generator.generate_schema(model, name, version)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(schema_code)

        return schema_code

    def export_all_schemas(
        self: Self,
        output_dir: str | Path,
    ) -> dict[str, dict[str, str]]:
        """Export all registered models as TypeScript schemas.

        Args:
            output_dir: Directory to save schema files.

        Returns:
            Dictionary mapping model names to version to schema code.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_schemas: dict[str, dict[str, str]] = {}

        for model_name in self._registry.list_models():
            all_schemas[model_name] = {}
            versions = self._registry.get_versions(model_name)

            for version in versions:
                model = self._registry.get_model(model_name, version)
                schema_code = self.generator.generate_schema(model, model_name, version)

                version_str = str(version).replace(".", "_")
                filename = f"{model_name}_v{version_str}.ts"
                filepath = output_dir / filename

                filepath.write_text(schema_code)

                all_schemas[model_name][str(version)] = schema_code

        return all_schemas
