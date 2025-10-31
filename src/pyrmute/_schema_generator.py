"""Base class for schema generators."""

from abc import ABC
from typing import TYPE_CHECKING, Generic, Self, TypeVar

from pydantic import BaseModel

from ._type_inspector import TypeInspector
from .model_version import ModelVersion

if TYPE_CHECKING:
    from enum import Enum

SchemaType = TypeVar("SchemaType")


class SchemaGeneratorBase(ABC, Generic[SchemaType]):
    """Base class for all schema generators."""

    def __init__(self: Self, include_docs: bool = True):
        self.include_docs = include_docs
        self._types_seen: set[str] = set()
        self._enums_encountered: dict[str, type[Enum]] = {}
        self._nested_models: dict[str, type[BaseModel]] = {}

    def generate_schema(
        self: Self,
        model: type[BaseModel],
        name: str,
        version: str | ModelVersion,
        registry_name_map: dict[str, str] | None = None,
    ) -> SchemaType:
        """Generate schema from Pydantic model."""
        raise NotImplementedError

    def _reset_state(self: Self) -> None:
        """Reset internal state before generating a new schema."""
        self._types_seen = set()
        self._enums_encountered = {}
        self._nested_models = {}

    def _collect_nested_models(self: Self, model: type[BaseModel]) -> None:
        """Shared implementation using TypeInspector."""
        nested = TypeInspector.collect_nested_models(model, self._types_seen)
        self._nested_models.update(nested)
