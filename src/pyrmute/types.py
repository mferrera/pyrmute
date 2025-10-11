"""Type aliases needed in the package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypeVar

from pydantic import BaseModel

from .model_version import ModelVersion

DecoratedBaseModel = TypeVar("DecoratedBaseModel", bound=BaseModel)

JsonValue: TypeAlias = (
    int | float | str | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonSchema: TypeAlias = dict[str, JsonValue]
JsonSchemaMode = Literal["validation", "serialization"]
JsonSchemaDefinitions: TypeAlias = dict[str, JsonValue]
JsonSchemaGenerator: TypeAlias = Callable[[type[BaseModel]], JsonSchema]
SchemaTransformer = Callable[[JsonSchema], JsonSchema]

ModelData: TypeAlias = dict[str, Any]
MigrationFunc: TypeAlias = Callable[[ModelData], ModelData]
MigrationKey: TypeAlias = tuple[ModelVersion, ModelVersion]
MigrationMap: TypeAlias = dict[MigrationKey, MigrationFunc]

ModelName: TypeAlias = str
ModelMetadata: TypeAlias = tuple[ModelName, ModelVersion]
VersionedModels: TypeAlias = dict[ModelVersion, type[BaseModel]]


@dataclass
class NestedModelInfo:
    """Contains information about nested models."""

    name: str
    version: ModelVersion
