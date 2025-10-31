"""Base class for schema generators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Self, TypedDict, TypeVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ._type_inspector import TypeInspector
from .model_version import ModelVersion

if TYPE_CHECKING:
    from enum import Enum

SchemaType = TypeVar("SchemaType")


class FieldContext(TypedDict, total=False):
    """Context information for field generation."""

    is_optional: bool
    is_repeated: bool
    has_default: bool
    default_value: Any


class SchemaGeneratorBase(ABC, Generic[SchemaType]):
    """Base class for all schema generators."""

    def __init__(self: Self, include_docs: bool = True):
        self.include_docs = include_docs
        self._types_seen: set[str] = set()
        self._enums_encountered: dict[str, type[Enum]] = {}
        self._nested_models: dict[str, type[BaseModel]] = {}

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
        self._enums_encountered = {}
        self._nested_models = {}

    @abstractmethod
    def _convert_type(
        self: Self,
        python_type: Any,
        field_info: FieldInfo | None = None,
    ) -> Any:
        """Convert Python type annotation to target schema type.

        Args:
            python_type: Python type annotation.
            field_info: Optional field info for constraint checking.

        Returns:
            Target schema type representation.
        """

    def _collect_nested_models(self: Self, model: type[BaseModel]) -> None:
        """Shared implementation using TypeInspector."""
        nested = TypeInspector.collect_nested_models(model, self._types_seen)
        self._nested_models.update(nested)

    def _analyze_field(self, field_info: FieldInfo) -> FieldContext:
        """Analyze field to determine its properties.

        This extracts common field analysis logic that all generators need.

        Args:
            field_info: Pydantic field info.

        Returns:
            Field context with analyzed properties.
        """
        is_optional = TypeInspector.is_optional_type(field_info.annotation)
        has_default = self._has_default_value(field_info)

        context: FieldContext = {
            "is_optional": is_optional,
            "has_default": has_default,
        }

        if has_default:
            default_value = self._get_default_value(field_info)
            if default_value is not None:
                context["default_value"] = default_value

        return context

    @abstractmethod
    def _generate_field_schema(
        self,
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

    def _get_field_name(self, field_name: str, field_info: FieldInfo) -> str:
        """Get the schema field name, considering aliases.

        Default implementation returns the original field name.  Subclasses can override
        to handle aliases differently.

        Args:
            field_name: Original Python field name.
            field_info: Pydantic field info.

        Returns:
            Field name to use in schema.
        """
        return field_name

    def _has_default_value(self, field_info: FieldInfo) -> bool:
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

    def _get_default_value(self, field_info: FieldInfo) -> Any:
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
