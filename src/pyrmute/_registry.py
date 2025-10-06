"""Model registry."""

from collections import defaultdict
from collections.abc import Callable
from typing import Self

from pydantic import BaseModel

from .model_version import ModelVersion
from .types import (
    DecoratedBaseModel,
    JsonSchemaGenerator,
    MigrationMap,
    ModelMetadata,
    ModelName,
    SchemaGenerators,
    VersionedModels,
)


class Registry:
    """Registry for versioned Pydantic models.

    Manages the registration and retrieval of versioned models and their
    associated metadata.

    Attributes:
        _models: Dictionary mapping model names to version-model mappings.
        _migrations: Dictionary storing migration functions between versions.
        _schema_generators: Dictionary storing custom schema generators.
        _model_metadata: Dictionary mapping model classes to (name, version).
        _ref_enabled: Dictionary tracking which models have enable_ref=True.
    """

    def __init__(self: Self) -> None:
        """Initialize the model registry."""
        self._models: dict[ModelName, VersionedModels] = defaultdict(dict)
        self._migrations: dict[ModelName, MigrationMap] = defaultdict(dict)
        self._schema_generators: dict[ModelName, SchemaGenerators] = defaultdict(dict)
        self._model_metadata: dict[type[BaseModel], ModelMetadata] = {}
        self._ref_enabled: dict[ModelName, set[ModelVersion]] = defaultdict(set)
        self._auto_migrate_enabled: dict[ModelName, set[ModelVersion]] = defaultdict(
            set
        )

    def register(
        self: Self,
        name: ModelName,
        version: str | ModelVersion,
        schema_generator: JsonSchemaGenerator | None = None,
        enable_ref: bool = False,
        auto_migrate: bool = False,
    ) -> Callable[[type[DecoratedBaseModel]], type[DecoratedBaseModel]]:
        """Register a versioned model.

        Args:
            name: Name of the model.
            version: Semantic version string or ModelVersion instance.
            schema_generator: Optional custom schema generator function.
            enable_ref: If True, this model can be referenced via $ref in
                separate schema files. If False, it will always be inlined.
            auto_migrate: If True, this model does not need a migration function to
                migrate to the next version. If a migration function is defined it will
                use it.

        Returns:
            Decorator function for model class.

        Example:
            >>> registry = Registry()
            >>> @registry.register("User", "1.0.0", enable_ref=True)
            ... class UserV1(BaseModel):
            ...     name: str
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version

        def decorator(cls: type[DecoratedBaseModel]) -> type[DecoratedBaseModel]:
            self._models[name][ver] = cls
            self._model_metadata[cls] = (name, ver)
            if schema_generator:
                self._schema_generators[name][ver] = schema_generator
            if enable_ref:
                self._ref_enabled[name].add(ver)
            if auto_migrate:
                self._auto_migrate_enabled[name].add(ver)
            return cls

        return decorator

    def get_model(
        self: Self, name: ModelName, version: str | ModelVersion
    ) -> type[BaseModel]:
        """Get a specific version of a model.

        Args:
            name: Name of the model.
            version: Semantic version string or ModelVersion instance.

        Returns:
            Model class for the specified version.

        Raises:
            ValueError: If model or version not found.
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version

        if name not in self._models or ver not in self._models[name]:
            raise ValueError(f"Model {name} v{ver} not found")

        return self._models[name][ver]

    def get_latest(self: Self, name: ModelName) -> type[BaseModel]:
        """Get the latest version of a model.

        Args:
            name: Name of the model.

        Returns:
            Latest version of the model class.

        Raises:
            ValueError: If model not found.
        """
        if name not in self._models:
            raise ValueError(f"Model {name} not found")

        latest_version = max(self._models[name].keys())
        return self._models[name][latest_version]

    def get_versions(self: Self, name: ModelName) -> list[ModelVersion]:
        """Get all versions available for a model.

        Args:
            name: Name of the model.

        Returns:
            Sorted list of available versions.

        Raises:
            ValueError: If model not found.
        """
        if name not in self._models:
            raise ValueError(f"Model {name} not found")

        return sorted(self._models[name].keys())

    def list_models(self: Self) -> list[ModelName]:
        """Get list of all registered model names.

        Returns:
            List of model names.
        """
        return list(self._models.keys())

    def get_model_info(
        self: Self, model_class: type[BaseModel]
    ) -> ModelMetadata | None:
        """Get the name and version for a registered model class.

        Args:
            model_class: The model class to look up.

        Returns:
            Tuple of (name, version) if found, None otherwise.
        """
        return self._model_metadata.get(model_class)

    def is_ref_enabled(
        self: Self, name: ModelName, version: str | ModelVersion
    ) -> bool:
        """Check if a model version is enabled for $ref usage.

        Args:
            name: Name of the model.
            version: Semantic version.

        Returns:
            True if this model can be referenced via $ref, False otherwise.
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version
        return ver in self._ref_enabled.get(name, set())
