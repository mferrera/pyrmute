"""Migrations manager."""

import contextlib
from collections.abc import Callable
from typing import Any, Self, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ._registry import Registry
from .model_version import ModelVersion
from .types import MigrationData, MigrationFunc, ModelName


class MigrationManager:
    """Manager for data migrations between model versions.

    Handles registration and execution of migration functions, including
    support for nested Pydantic models.

    Attributes:
        registry: Reference to the Registry.
    """

    def __init__(self: Self, registry: Registry) -> None:
        """Initialize the migration manager.

        Args:
            registry: Registry instance to use.
        """
        self.registry = registry

    def register_migration(
        self: Self,
        name: ModelName,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
    ) -> Callable[[MigrationFunc], MigrationFunc]:
        """Register a migration function between two versions.

        Args:
            name: Name of the model.
            from_version: Source version for migration.
            to_version: Target version for migration.

        Returns:
            Decorator function for migration function.

        Example:
            >>> manager = MigrationManager(registry)
            >>> @manager.register_migration("User", "1.0.0", "2.0.0")
            ... def migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
            ...     return {**data, "email": "unknown@example.com"}
        """
        from_ver = (
            ModelVersion.parse(from_version)
            if isinstance(from_version, str)
            else from_version
        )
        to_ver = (
            ModelVersion.parse(to_version)
            if isinstance(to_version, str)
            else to_version
        )

        def decorator(func: MigrationFunc) -> MigrationFunc:
            self.registry._migrations[name][(from_ver, to_ver)] = func
            return func

        return decorator

    def migrate(
        self: Self,
        data: MigrationData,
        name: ModelName,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
    ) -> MigrationData:
        """Migrate data from one version to another.

        Args:
            data: Data dictionary to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.

        Returns:
            Migrated data dictionary.

        Raises:
            ValueError: If migration path cannot be found.
        """
        from_ver = (
            ModelVersion.parse(from_version)
            if isinstance(from_version, str)
            else from_version
        )
        to_ver = (
            ModelVersion.parse(to_version)
            if isinstance(to_version, str)
            else to_version
        )

        if from_ver == to_ver:
            return data

        path = self.find_migration_path(name, from_ver, to_ver)

        current_data = data
        for i in range(len(path) - 1):
            migration_key = (path[i], path[i + 1])

            if migration_key in self.registry._migrations[name]:
                migration_func = self.registry._migrations[name][migration_key]
                current_data = migration_func(current_data)
            elif path[i + 1] in self.registry._auto_migrate_enabled[name]:
                current_data = self._auto_migrate(
                    current_data, name, path[i], path[i + 1]
                )
            else:
                raise ValueError(
                    f"Unable to find migration path for {name}: "
                    f"{path[i]} → {path[i + 1]}"
                )

        return current_data

    def find_migration_path(
        self: Self,
        name: ModelName,
        from_ver: ModelVersion,
        to_ver: ModelVersion,
    ) -> list[ModelVersion]:
        """Find migration path between versions.

        Args:
            name: Name of the model.
            from_ver: Source version.
            to_ver: Target version.

        Returns:
            List of versions forming the migration path.
        """
        versions = sorted(self.registry.get_versions(name))

        if from_ver not in versions or to_ver not in versions:
            raise ValueError(f"Invalid version range for {name}")

        from_idx = versions.index(from_ver)
        to_idx = versions.index(to_ver)

        if from_idx < to_idx:
            return versions[from_idx : to_idx + 1]
        return versions[to_idx : from_idx + 1][::-1]

    def validate_migration_path(
        self: Self,
        name: ModelName,
        from_ver: ModelVersion,
        to_ver: ModelVersion,
    ) -> None:
        """Validate that a migration path exists and all steps are valid.

        Args:
            name: Name of the model.
            from_ver: Source version.
            to_ver: Target version.

        Raises:
            ValueError: If any step in the migration path is invalid.
        """
        path = self.find_migration_path(name, from_ver, to_ver)

        for i in range(len(path) - 1):
            current_ver = path[i]
            next_ver = path[i + 1]
            migration_key = (current_ver, next_ver)

            has_explicit = migration_key in self.registry._migrations.get(name, {})
            has_auto = next_ver in self.registry._auto_migrate_enabled.get(name, set())

            if not has_explicit and not has_auto:
                raise ValueError(
                    f"No migration found for {name}: {current_ver} → {next_ver}"
                )

    def _auto_migrate(
        self: Self,
        data: MigrationData,
        name: ModelName,
        from_ver: ModelVersion,
        to_ver: ModelVersion,
    ) -> MigrationData:
        """Automatically migrate data when no explicit migration exists.

        This method handles nested Pydantic models recursively, migrating
        them to their corresponding versions.

        Args:
            data: Data dictionary to migrate.
            name: Name of the model.
            from_ver: Source version.
            to_ver: Target version.

        Returns:
            Migrated data dictionary.
        """
        from_model = self.registry.get_model(name, from_ver)
        to_model = self.registry.get_model(name, to_ver)

        from_fields = from_model.model_fields
        to_fields = to_model.model_fields

        result: MigrationData = {}

        for field_name, to_field_info in to_fields.items():
            # Field exists in data, migrate it
            if field_name in data:
                value = data[field_name]
                from_field_info = from_fields.get(field_name)
                result[field_name] = self._migrate_field_value(
                    value, from_field_info, to_field_info
                )

            # Field missing from data, use default if available
            elif to_field_info.default is not PydanticUndefined:
                result[field_name] = to_field_info.default
            elif to_field_info.default_factory is not None:
                with contextlib.suppress(Exception):
                    result[field_name] = to_field_info.default_factory()  # type: ignore

        # Migrate all extra data not in the field, too
        for field_name, value in data.items():
            if field_name not in to_fields:
                result[field_name] = value

        return result

    def _migrate_field_value(
        self: Self,
        value: Any,
        from_field: FieldInfo | None,
        to_field: FieldInfo,
    ) -> Any:
        """Migrate a single field value, handling nested models.

        Args:
            value: The field value to migrate.
            from_field: Source field info (None if field is new).
            to_field: Target field info.

        Returns:
            Migrated field value.
        """
        if value is None:
            return None

        if isinstance(value, dict):
            nested_info = self._extract_nested_model_info(value, from_field, to_field)
            if nested_info:
                nested_name, nested_from_ver, nested_to_ver = nested_info
                return self.migrate(value, nested_name, nested_from_ver, nested_to_ver)

            return {
                k: self._migrate_field_value(v, from_field, to_field)
                for k, v in value.items()
            }

        if isinstance(value, list):
            return [
                self._migrate_field_value(item, from_field, to_field) for item in value
            ]

        return value

    def _extract_nested_model_info(
        self: Self,
        value: MigrationData,
        from_field: FieldInfo | None,
        to_field: FieldInfo,
    ) -> tuple[ModelName, ModelVersion, ModelVersion] | None:
        """Extract nested model migration information.

        Args:
            value: The nested model data.
            from_field: Source field info.
            to_field: Target field info.

        Returns:
            Tuple of (model_name, from_version, to_version) if this is a
            versioned nested model, None otherwise.
        """
        to_model_type = self._get_model_type_from_field(to_field)
        if not to_model_type or not issubclass(to_model_type, BaseModel):
            return None

        to_info = self.registry.get_model_info(to_model_type)
        if not to_info:
            return None

        model_name, to_version = to_info

        # Get the source version
        if from_field:
            from_model_type = self._get_model_type_from_field(from_field)
            if from_model_type and issubclass(from_model_type, BaseModel):
                from_info = self.registry.get_model_info(from_model_type)
                if from_info and from_info[0] == model_name:
                    from_version = from_info[1]
                    return (model_name, from_version, to_version)

        # If we can't determine the source version, assume it's the same as target
        return (model_name, to_version, to_version)

    def _get_model_type_from_field(
        self: Self, field: FieldInfo
    ) -> type[BaseModel] | None:
        """Extract the Pydantic model type from a field.

        Handles Optional, List, and other generic types.

        Args:
            field: The field info to extract from.

        Returns:
            The model type if found, None otherwise.
        """
        annotation = field.annotation

        if annotation is None:
            return None

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation

        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            for arg in args:
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    return arg

        return None
