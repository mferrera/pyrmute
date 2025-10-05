"""Model manager."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel

from ._migration_manager import MigrationManager
from ._registry import Registry
from ._schema_manager import SchemaManager
from .model_version import ModelVersion
from .types import (
    DecoratedBaseModel,
    JsonSchema,
    JsonSchemaGenerator,
    MigrationData,
    MigrationFunc,
    ModelMetadata,
)


class ModelManager:
    """High-level interface for versioned model management.

    Provides a unified API for model registration, migration, and schema
    management.

    Attributes:
        registry: Registry instance.
        migration_manager: MigrationManager instance.
        schema_manager: SchemaManager instance.

    Example:
        >>> manager = ModelManager()
        >>>
        >>> @manager.model("User", "1.0.0")
        ... class UserV1(BaseModel):
        ...     name: str
        >>>
        >>> @manager.model("User", "2.0.0")
        ... class UserV2(BaseModel):
        ...     name: str
        ...     email: str
        >>>
        >>> @manager.migration("User", "1.0.0", "2.0.0")
        ... def migrate(data: MigrationData) -> MigrationData:
        ...     return {**data, "email": "unknown@example.com"}
    """

    def __init__(self: Self) -> None:
        """Initialize the versioned model manager."""
        self.registry = Registry()
        self.migration_manager = MigrationManager(self.registry)
        self.schema_manager = SchemaManager(self.registry)

    def model(
        self: Self,
        name: str,
        version: str | ModelVersion,
        schema_generator: JsonSchemaGenerator | None = None,
        enable_ref: bool = False,
    ) -> Callable[[type[DecoratedBaseModel]], type[DecoratedBaseModel]]:
        """Register a versioned model.

        Args:
            name: Name of the model.
            version: Semantic version.
            schema_generator: Optional custom schema generator.
            enable_ref: If True, this model can be referenced via $ref in
                separate schema files. If False, it will always be inlined.

        Returns:
            Decorator function for model class.

        Example:
            >>> # Model that will be inlined (default)
            >>> @manager.model("Address", "1.0.0")
            ... class AddressV1(BaseModel):
            ...     street: str
            >>>
            >>> # Model that can be a separate schema with $ref
            >>> @manager.model("City", "1.0.0", enable_ref=True)
            ... class CityV1(BaseModel):
            ...     city: City
        """
        return self.registry.register(name, version, schema_generator, enable_ref)

    def migration(
        self: Self,
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
    ) -> Callable[[MigrationFunc], MigrationFunc]:
        """Register a migration function.

        Args:
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.

        Returns:
            Decorator function for migration function.
        """
        return self.migration_manager.register_migration(name, from_version, to_version)

    def get(
        self: Self, name: str, version: str | ModelVersion | None = None
    ) -> type[BaseModel]:
        """Get a model by name and version.

        Args:
            name: Name of the model.
            version: Semantic version (returns latest if None).

        Returns:
            Model class.
        """
        if version is None:
            return self.registry.get_latest(name)
        return self.registry.get_model(name, version)

    def migrate(
        self: Self,
        data: MigrationData,
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
    ) -> BaseModel:
        """Migrate data between versions.

        Args:
            data: Data dictionary or BaseModel to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.

        Returns:
            Migrated BaseModel.
        """
        migrated_data = self.migration_manager.migrate(
            data, name, from_version, to_version
        )
        target_model = self.get(name, to_version)
        return target_model.model_validate(migrated_data)

    def get_schema(
        self: Self,
        name: str,
        version: str | ModelVersion,
        **kwargs: Any,
    ) -> JsonSchema:
        """Get JSON schema for a specific version.

        Args:
            name: Name of the model.
            version: Semantic version.
            **kwargs: Additional schema generation arguments.

        Returns:
            JSON schema dictionary.
        """
        return self.schema_manager.get_schema(name, version, **kwargs)

    def list_models(self: Self) -> list[str]:
        """Get list of all registered models.

        Returns:
            List of model names.
        """
        return self.registry.list_models()

    def list_versions(self: Self, name: str) -> list[ModelVersion]:
        """Get all versions for a model.

        Args:
            name: Name of the model.

        Returns:
            Sorted list of versions.
        """
        return self.registry.get_versions(name)

    def dump_schemas(
        self: Self,
        output_dir: str | Path,
        indent: int = 2,
        separate_definitions: bool = False,
        ref_template: str | None = None,
    ) -> None:
        """Export all schemas to JSON files.

        Args:
            output_dir: Directory path for output.
            indent: JSON indentation level.
            separate_definitions: If True, create separate schema files for
                nested models and use $ref to reference them.
            ref_template: Template for $ref URLs when separate_definitions=True.
                Defaults to relative file references if not provided.

        Example:
            >>> # Inline definitions (default)
            >>> manager.dump_schemas("schemas/")
            >>>
            >>> # Separate sub-schemas with relative refs
            >>> manager.dump_schemas("schemas/", separate_definitions=True)
            >>>
            >>> # Separate sub-schemas with absolute URLs
            >>> manager.dump_schemas(
            ...     "schemas/",
            ...     separate_definitions=True,
            ...     ref_template="https://example.com/schemas/{model}_v{version}.json"
            ... )
        """
        self.schema_manager.dump_schemas(
            output_dir, indent, separate_definitions, ref_template
        )

    def dump_schemas_with_refs(
        self: Self,
        output_dir: str | Path,
        ref_template: str | None = None,
        indent: int = 2,
    ) -> None:
        """Export schemas with separate files for nested models.

        This is a convenience method that calls dump_schemas with
        separate_definitions=True.

        Args:
            output_dir: Directory path for output.
            ref_template: Template for $ref URLs. Supports {model} and
                {version} placeholders. Defaults to relative file refs.
            indent: JSON indentation level.

        Example:
            >>> # Relative file references (default)
            >>> manager.dump_schemas_with_refs("schemas/")
            >>>
            >>> # Absolute URL references
            >>> manager.dump_schemas_with_refs(
            ...     "schemas/",
            ...     ref_template="https://example.com/schemas/{model}_v{version}.json"
            ... )
        """
        self.schema_manager.dump_schemas(
            output_dir, indent, separate_definitions=True, ref_template=ref_template
        )

    def get_nested_models(
        self: Self,
        name: str,
        version: str | ModelVersion,
    ) -> list[ModelMetadata]:
        """Get all nested models used by a model.

        Args:
            name: Name of the model.
            version: Semantic version.

        Returns:
            List of (model_name, version) tuples for nested models.
        """
        return self.schema_manager.get_nested_models(name, version)
