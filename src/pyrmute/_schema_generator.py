"""Base class for schema generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Self, TypeVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ._type_inspector import TypeInspector
from .model_version import ModelVersion

SchemaType = TypeVar("SchemaType")


@dataclass
class TypeInfo:
    """Common type information across schema generators.

    This provides a standardized way to represent type conversion results
    across different schema formats (Avro, Protocol Buffers, TypeScript).

    Attributes:
        type_representation: The actual type in the target format.
            - For Avro: AvroType (str, list, or dict)
            - For Protobuf: str (the proto type name like "string", "int32")
            - For TypeScript: str (the TS type like "string", "number")
        is_optional: Whether the type allows None/null values.
        is_repeated: Whether the type represents a collection/array.
        metadata: Additional format-specific metadata.
    """

    type_representation: Any
    is_optional: bool = False
    is_repeated: bool = False
    optional_marker: bool = False


@dataclass
class FieldContext:
    """Context information for field generation."""

    is_optional: bool
    is_repeated: bool = False
    has_default: bool = False
    default_value: Any = None


@dataclass
class FieldSchema:
    """Intermediate field representation after analysis but before formatting.

    This provides a common structure across all schema generators that contains
    all the information needed to format a field in any target format.

    Attributes:
        name: Field name to use in the schema (may differ from Python name due to
            aliases).
        python_name: Original Python field name.
        type_info: Converted type information.
        context: Field context with optionality, defaults, etc.
        description: Field documentation/description.
        constraints: Field constraints (min/max length, numeric bounds, etc.).
        is_computed: Whether this is a computed field (read-only).
    """

    name: str
    python_name: str
    type_info: TypeInfo
    context: FieldContext
    description: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    is_computed: bool = False
    aliases: list[str] = field(default_factory=list)


class SchemaGeneratorBase(ABC, Generic[SchemaType]):
    """Base class for all schema generators."""

    def __init__(self: Self, include_docs: bool = True):
        self.include_docs = include_docs
        self._types_seen: set[str] = set()
        self._collected_enums: dict[str, type[Enum]] = {}
        self._nested_models: dict[str, type[BaseModel]] = {}
        self._versioned_name_map: dict[str, str] = {}
        self._current_model_class_name: str = ""
        self._current_model_schema_name: str = ""

    @abstractmethod
    def generate_schema(
        self: Self,
        model: type[BaseModel],
        name: str,
        version: str | ModelVersion,
        registry_name_map: dict[str, str] | None = None,
    ) -> SchemaType:
        """Generate schema from Pydantic model."""

    def _reset_state(self: Self) -> None:
        """Reset internal state before generating a new schema."""
        self._types_seen = set()
        self._collected_enums = {}
        self._nested_models = {}
        self._versioned_name_map = {}
        self._current_model_class_name = ""
        self._current_model_schema_name = ""

    @abstractmethod
    def _convert_type(
        self: Self,
        python_type: Any,
        field_info: FieldInfo | None = None,
    ) -> TypeInfo:
        """Convert Python type annotation to target schema type.

        Args:
            python_type: Python type annotation.
            field_info: Optional field info for constraint checking.

        Returns:
            Target schema type representation.
        """

    def _analyze_field(self: Self, field_info: FieldInfo) -> FieldContext:
        """Analyze field to determine its properties.

        This extracts common field analysis logic that all generators need.

        Args:
            field_info: Pydantic field info.

        Returns:
            Field context with analyzed properties.
        """
        is_optional = TypeInspector.is_optional_type(field_info.annotation)
        has_default = self._has_default_value(field_info)

        default_value = None
        if has_default:
            default_value = self._get_default_value(field_info)

        return FieldContext(
            is_optional=is_optional,
            has_default=has_default,
            default_value=default_value,
        )

    def _build_field_schema(
        self: Self,
        field_name: str,
        field_info: FieldInfo,
        model: type[BaseModel],
    ) -> FieldSchema:
        """Build intermediate field schema representation.

        This is the common analysis phase that extracts all field information
        before format-specific rendering.

        Args:
            field_name: Name of the field.
            field_info: Pydantic field info.
            model: Parent model class.

        Returns:
            Intermediate field schema.
        """
        schema_name = self._get_field_name(field_name, field_info)

        context = self._analyze_field(field_info)
        type_info = self._convert_type(field_info.annotation, field_info)

        description = field_info.description if self.include_docs else None
        constraints = self._extract_constraints(field_info)
        aliases = self._collect_field_aliases(field_name, schema_name, field_info)

        return FieldSchema(
            name=schema_name,
            python_name=field_name,
            type_info=type_info,
            context=context,
            description=description,
            constraints=constraints,
            is_computed=False,
            aliases=aliases,
        )

    def _extract_constraints(self: Self, field_info: FieldInfo) -> dict[str, Any]:
        """Extract field constraints for validation.

        Default implementation extracts common constraints. Subclasses can
        override to extract format-specific constraints.

        Args:
            field_info: Pydantic field info.

        Returns:
            Dictionary of constraints.
        """
        constraints: dict[str, Any] = {}

        numeric = TypeInspector.get_numeric_constraints(field_info)
        if any(v is not None for v in numeric.values()):
            constraints["numeric"] = numeric

        string = TypeInspector.get_string_constraints(field_info)
        if any(v is not None for v in string.values()):
            constraints["string"] = string

        return constraints

    def _collect_field_aliases(
        self: Self,
        python_name: str,
        schema_name: str,
        field_info: FieldInfo,
    ) -> list[str]:
        """Collect all aliases for a field.

        This gathers the Python name and any Pydantic aliases that differ from the
        chosen schema name.

        Args:
            python_name: Original Python field name.
            schema_name: Chosen schema field name.
            field_info: Pydantic field info.

        Returns:
            List of aliases (may be empty).
        """
        aliases: list[str] = []

        if python_name != schema_name:
            aliases.append(python_name)

        if (
            field_info.alias
            and field_info.alias != schema_name
            and field_info.alias not in aliases
        ):
            aliases.append(field_info.alias)

        if (
            field_info.serialization_alias
            and field_info.serialization_alias != schema_name
            and field_info.serialization_alias not in aliases
        ):
            aliases.append(field_info.serialization_alias)

        return aliases

    @abstractmethod
    def _generate_field_schema(
        self: Self,
        field_name: str,
        field_info: FieldInfo,
        model: type[BaseModel],
    ) -> Any:
        """Generate schema for a single field.

        Args:
            field_name: Name of the field.
            field_info: Pydantic field info.
            model: Parent model class.

        Returns:
            Field schema in target format.
        """

    def _get_field_name(self: Self, field_name: str, field_info: FieldInfo) -> str:
        """Get the schema field name, considering aliases.

        Default implementation returns the original field name. Subclasses can override
        to handle aliases differently.

        Args:
            field_name: Original Python field name.
            field_info: Pydantic field info.

        Returns:
            Field name to use in schema.
        """
        return field_name

    def _has_default_value(self: Self, field_info: FieldInfo) -> bool:
        """Check if field has a default value.

        Args:
            field_info: Pydantic field info.

        Returns:
            True if field has a default value.
        """
        return (
            field_info.default is not PydanticUndefined
            or field_info.default_factory is not None
        )

    def _get_default_value(self: Self, field_info: FieldInfo) -> Any:
        """Get the default value for a field.

        Args:
            field_info: Pydantic field info.

        Returns:
            Default value, or None if no default or factory fails.
        """
        if field_info.default is not PydanticUndefined:
            return field_info.default

        if field_info.default_factory is not None:
            try:
                return field_info.default_factory()  # type: ignore
            except Exception:
                return None

        return None

    def _should_collect_enum(self: Self, enum_class: type[Enum]) -> bool:
        """Check if enum should be collected as a separate definition.

        Some formats (TypeScript union style) inline enum values instead of creating
        separate enum definitions. This can be overriden in such cases.

        Args:
            enum_class: Enum class to check.

        Returns:
            True if enum should be collected as a definition.
        """
        return True

    def _register_enum(self: Self, enum_class: type[Enum]) -> None:
        """Register an enum that's been encountered.

        Args:
            enum_class: Enum class to register.
        """
        enum_name = enum_class.__name__
        if enum_name not in self._collected_enums:
            self._collected_enums[enum_name] = enum_class

    @abstractmethod
    def _convert_enum(self: Self, enum_class: type[Enum]) -> Any:
        """Convert Python Enum to target format.

        Args:
            enum_class: Python Enum class.

        Returns:
            Enum representation in target format.
        """

    def _collect_nested_models(self: Self, model: type[BaseModel]) -> None:
        """Recursively collect all nested BaseModel types.

        This uses TypeInspector to find all nested models and stores them for later
        schema generation.

        Args:
            model: Pydantic model class to scan for nested models.
        """
        nested = TypeInspector.collect_nested_models(model, self._types_seen)
        self._nested_models.update(nested)

    def _register_nested_model(self: Self, model: type[BaseModel]) -> None:
        """Register a nested model that's been encountered.

        Args:
            model: Nested model to register.
        """
        model_name = model.__name__
        if model_name not in self._nested_models and model_name not in self._types_seen:
            self._nested_models[model_name] = model

    def _register_model_name(
        self: Self, model_class_name: str, schema_name: str
    ) -> None:
        """Register the mapping from model class name to schema name.

        This is used to track how model class names map to their schema names, which may
        include version information or other transformations.

        Args:
            model_class_name: Original Python class name (e.g., "User").
            schema_name: Name to use in generated schema (e.g., "UserV1_0_0").
        """
        self._versioned_name_map[model_class_name] = schema_name

    def _get_model_schema_name(self: Self, model_class_name: str) -> str:
        """Get the schema name for a model class.

        Args:
            model_class_name: Original Python class name.

        Returns:
            Schema name to use, or the class name if no mapping exists.
        """
        return self._versioned_name_map.get(model_class_name, model_class_name)
