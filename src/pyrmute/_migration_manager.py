"""Migrations manager."""

import contextlib
import types
from collections.abc import Callable
from typing import Annotated, Any, Literal, Self, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ._registry import Registry
from .exceptions import MigrationError, ModelNotFoundError
from .model_version import ModelVersion
from .types import MigrationFunc, ModelData, ModelName


class MigrationManager:
    """Manager for data migrations between model versions.

    Handles registration and execution of migration functions, including support for
    nested Pydantic models.

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
        from_ver = self._parse_version(from_version)
        to_ver = self._parse_version(to_version)

        def decorator(func: MigrationFunc) -> MigrationFunc:
            self.registry._migrations[name][(from_ver, to_ver)] = func
            return func

        return decorator

    def migrate(
        self: Self,
        data: ModelData,
        name: ModelName,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
    ) -> ModelData:
        """Migrate data from one version to another.

        Args:
            data: Data dictionary to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.

        Returns:
            Migrated data dictionary.

        Raises:
            ModelNotFoundError: If model or versions don't exist.
            MigrationError: If migration path cannot be found.
        """
        from_ver = self._parse_version(from_version)
        to_ver = self._parse_version(to_version)

        if from_ver == to_ver:
            return data

        path = self.find_migration_path(name, from_ver, to_ver)
        return self._execute_migration_path(data, name, path)

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

        Raises:
            ModelNotFoundError: If the model or versions don't exist.
        """
        versions = sorted(self.registry.get_versions(name))

        if from_ver not in versions:
            raise ModelNotFoundError(name, str(from_ver))
        if to_ver not in versions:
            raise ModelNotFoundError(name, str(to_ver))

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
            ModelNotFoundError: If the model or versions don't exist.
            MigrationError: If any step in the migration path is invalid.
        """
        path = self.find_migration_path(name, from_ver, to_ver)

        for i in range(len(path) - 1):
            current_ver = path[i]
            next_ver = path[i + 1]

            if not self._has_migration_step(name, current_ver, next_ver):
                raise MigrationError(
                    name,
                    str(current_ver),
                    str(next_ver),
                    (
                        "No migration path found. Define a migration function or mark "
                        "the target version as backward_compatible."
                    ),
                )

    def _parse_version(self: Self, version: str | ModelVersion) -> ModelVersion:
        """Parse version string or return ModelVersion as-is."""
        return ModelVersion.parse(version) if isinstance(version, str) else version

    def _has_migration_step(
        self: Self, name: ModelName, from_ver: ModelVersion, to_ver: ModelVersion
    ) -> bool:
        """Check if a migration step exists (explicit or auto)."""
        migration_key = (from_ver, to_ver)
        has_explicit = migration_key in self.registry._migrations.get(name, {})
        has_auto = to_ver in self.registry._backward_compatible_enabled.get(name, set())
        return has_explicit or has_auto

    def _execute_migration_path(
        self: Self, data: ModelData, name: ModelName, path: list[ModelVersion]
    ) -> ModelData:
        """Execute migration through a path of versions."""
        current_data = data

        for i in range(len(path) - 1):
            try:
                current_data = self._execute_single_step(
                    current_data, name, path[i], path[i + 1]
                )
            except Exception as e:
                if isinstance(e, MigrationError):
                    raise
                raise MigrationError(
                    name,
                    str(path[i]),
                    str(path[i + 1]),
                    f"Migration failed: {type(e).__name__}: {e}",
                ) from e

        return current_data

    def _execute_single_step(
        self: Self,
        data: ModelData,
        name: ModelName,
        from_ver: ModelVersion,
        to_ver: ModelVersion,
    ) -> ModelData:
        """Execute a single migration step."""
        migration_key = (from_ver, to_ver)

        if migration_key in self.registry._migrations[name]:
            migration_func = self.registry._migrations[name][migration_key]
            try:
                return migration_func(data)
            except Exception as e:
                raise MigrationError(
                    name,
                    str(from_ver),
                    str(to_ver),
                    f"Migration function raised: {type(e).__name__}: {e}",
                ) from e

        if to_ver in self.registry._backward_compatible_enabled[name]:
            try:
                return self._auto_migrate(data, name, from_ver, to_ver)
            except Exception as e:
                raise MigrationError(
                    name,
                    str(from_ver),
                    str(to_ver),
                    f"Auto-migration failed: {type(e).__name__}: {e}",
                ) from e

        raise MigrationError(
            name,
            str(from_ver),
            str(to_ver),
            (
                "No migration path found. Define a migration function or mark "
                "the target version as backward_compatible."
            ),
        )

    def _auto_migrate(
        self: Self,
        data: ModelData,
        name: ModelName,
        from_ver: ModelVersion,
        to_ver: ModelVersion,
    ) -> ModelData:
        """Automatically migrate data when no explicit migration exists.

        This method handles nested Pydantic models recursively, migrating them to their
        corresponding versions. Handles field aliases by building a lookup map.

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

        key_to_field_name = self._build_alias_map(from_fields)

        result: ModelData = {}
        processed_keys: set[str] = set()

        for field_name, to_field_info in to_fields.items():
            field_result = self._migrate_single_field(
                data, field_name, to_field_info, from_fields, key_to_field_name
            )

            if field_result is not None:
                output_key, migrated_value, source_keys = field_result
                result[output_key] = migrated_value
                processed_keys.update(source_keys)

        # Preserve unprocessed extra fields
        for data_key, value in data.items():
            if data_key not in processed_keys:
                result[data_key] = value

        return result

    def _build_alias_map(self: Self, fields: dict[str, FieldInfo]) -> dict[str, str]:
        """Build a mapping from all possible input keys to canonical field names.

        Args:
            fields: Model fields to extract aliases from.

        Returns:
            Dictionary mapping data keys (field names and aliases) to field names.
        """
        key_to_field_name: dict[str, str] = {}

        for field_name, field_info in fields.items():
            key_to_field_name[field_name] = field_name

            if field_info.alias:
                key_to_field_name[field_info.alias] = field_name
            if field_info.serialization_alias:
                key_to_field_name[field_info.serialization_alias] = field_name
            if isinstance(field_info.validation_alias, str):
                key_to_field_name[field_info.validation_alias] = field_name

        return key_to_field_name

    def _migrate_single_field(
        self: Self,
        data: ModelData,
        field_name: str,
        to_field_info: FieldInfo,
        from_fields: dict[str, FieldInfo],
        key_to_field_name: dict[str, str],
    ) -> tuple[str, Any, set[str]] | None:
        """Migrate a single field, handling aliases and defaults.

        Args:
            data: Source data dictionary.
            field_name: Target field name.
            to_field_info: Target field info.
            from_fields: Source model fields.
            key_to_field_name: Mapping from data keys to field names.

        Returns:
            Tuple of (output_key, migrated_value, source_keys_to_mark_processed) if
            field should be included, None if field should be skipped.
        """
        value, data_key_used = self._find_field_value(
            data, field_name, key_to_field_name
        )

        if data_key_used is not None:
            from_field_info = from_fields.get(field_name)
            migrated_value = self._migrate_field_value(
                value, from_field_info, to_field_info
            )
            keys_to_process = {
                k for k, v in key_to_field_name.items() if v == field_name
            }
            return (data_key_used, migrated_value, keys_to_process)

        _NO_DEFAULT = object()
        default_value = self._get_field_default(to_field_info, _NO_DEFAULT)

        if default_value is not _NO_DEFAULT:
            output_key = (
                to_field_info.serialization_alias or to_field_info.alias or field_name
            )
            return (output_key, default_value, set())

        return None

    def _find_field_value(
        self: Self, data: ModelData, field_name: str, key_to_field_name: dict[str, str]
    ) -> tuple[Any, str | None]:
        """Find a field's value in data, checking both field name and aliases.

        Args:
            data: Source data dictionary.
            field_name: Target field name to find.
            key_to_field_name: Mapping from data keys to field names.

        Returns:
            Tuple of (value, data_key) if found, (None, None) otherwise.
        """
        if field_name in data:
            return (data[field_name], field_name)

        for data_key in data:
            if key_to_field_name.get(data_key) == field_name:
                return (data[data_key], data_key)

        return (None, None)

    def _get_field_default(
        self: Self, field_info: FieldInfo, sentinel: Any = None
    ) -> Any:
        """Get the default value for a field.

        Args:
            field_info: Field info to extract default from.
            sentinel: Sentinel value to return if no default is available.

        Returns:
            Default value if available, sentinel otherwise.
        """
        if field_info.default is not PydanticUndefined:
            return field_info.default

        if field_info.default_factory is not None:
            with contextlib.suppress(Exception):
                return field_info.default_factory()  # type: ignore

        return sentinel

    def _migrate_field_value(
        self: Self, value: Any, from_field: FieldInfo | None, to_field: FieldInfo
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
            return self._migrate_dict_value(value, from_field, to_field)

        if isinstance(value, list):
            return self._migrate_list_value(value, from_field, to_field)

        return value

    def _migrate_dict_value(
        self, value: dict[str, Any], from_field: FieldInfo | None, to_field: FieldInfo
    ) -> dict[str, Any]:
        """Migrate a dictionary value (might be a nested model)."""
        nested_info = self._extract_nested_model_info(value, from_field, to_field)
        if nested_info:
            nested_name, nested_from_ver, nested_to_ver = nested_info
            return self.migrate(value, nested_name, nested_from_ver, nested_to_ver)

        return {
            k: self._migrate_field_value(v, from_field, to_field)
            for k, v in value.items()
        }

    def _migrate_list_value(
        self, value: list[Any], from_field: FieldInfo | None, to_field: FieldInfo
    ) -> list[Any]:
        """Migrate a list value (might contain nested models)."""
        from_item_field, to_item_field = self._get_list_item_fields(
            from_field, to_field
        )

        return [
            self._migrate_list_item(item, from_item_field, to_item_field)
            for item in value
        ]

    def _migrate_list_item(
        self, item: Any, from_field: FieldInfo | None, to_field: FieldInfo | None
    ) -> Any:
        """Migrate a single item from a list."""
        if not isinstance(item, dict) or to_field is None:
            return item

        nested_info = self._extract_nested_model_info(item, from_field, to_field)
        if nested_info:
            nested_name, nested_from_ver, nested_to_ver = nested_info
            return self.migrate(item, nested_name, nested_from_ver, nested_to_ver)

        return item

    def _get_list_item_fields(
        self, from_field: FieldInfo | None, to_field: FieldInfo
    ) -> tuple[FieldInfo | None, FieldInfo | None]:
        """Extract field info for items in a list field."""
        to_item_field = self._extract_list_item_field(to_field)
        from_item_field = (
            self._extract_list_item_field(from_field) if from_field else None
        )
        return from_item_field, to_item_field

    def _extract_list_item_field(self, field: FieldInfo | None) -> FieldInfo | None:
        """Extract field info for the items of a list field."""
        if field is None or field.annotation is None:
            return None

        origin = get_origin(field.annotation)
        if origin is not list:
            return None

        args = get_args(field.annotation)
        if not args:
            return None

        item_annotation = args[0]

        discriminator = None
        if get_origin(item_annotation) is Annotated:
            annotated_args = get_args(item_annotation)
            item_annotation = annotated_args[0]
            for metadata in annotated_args[1:]:
                if hasattr(metadata, "discriminator"):
                    discriminator = metadata.discriminator

        synthetic_field = FieldInfo(
            annotation=item_annotation, default=PydanticUndefined
        )
        if discriminator:
            synthetic_field.discriminator = discriminator

        return synthetic_field

    def _extract_nested_model_info(
        self, value: ModelData, from_field: FieldInfo | None, to_field: FieldInfo
    ) -> tuple[ModelName, ModelVersion, ModelVersion] | None:
        """Extract nested model migration information.

        Handles discriminated unions by using the discriminator field to determine which
        model type to migrate.


        Args:
            value: The nested model data.
            from_field: Source field info.
            to_field: Target field info.

        Returns:
            Tuple of (model_name, from_version, to_version) if this is a
            versioned nested model, None otherwise.
        """
        discriminated_info = self._try_extract_discriminated_model(
            value, from_field, to_field
        )
        if discriminated_info:
            return discriminated_info

        return self._try_extract_simple_nested_model(from_field, to_field)

    def _try_extract_discriminated_model(
        self, value: ModelData, from_field: FieldInfo | None, to_field: FieldInfo
    ) -> tuple[ModelName, ModelVersion, ModelVersion] | None:
        """Try to extract model info from a discriminated union field."""
        discriminator_key = self._get_discriminator_key(to_field)
        if not discriminator_key:
            return None

        discriminator_value = self._find_discriminator_value(
            value, discriminator_key, to_field
        )
        if discriminator_value is None:
            return None

        to_model_type = self._find_discriminated_type(
            to_field, discriminator_key, discriminator_value
        )
        if not to_model_type:
            return None

        to_info = self.registry.get_model_info(to_model_type)
        if not to_info:
            return None

        model_name, to_version = to_info
        from_version = self._get_discriminated_source_version(
            from_field, discriminator_key, discriminator_value, model_name, to_version
        )

        return (model_name, from_version, to_version)

    def _try_extract_simple_nested_model(
        self, from_field: FieldInfo | None, to_field: FieldInfo
    ) -> tuple[ModelName, ModelVersion, ModelVersion] | None:
        """Try to extract model info from a simple (non-discriminated) nested field."""
        to_model_type = self._get_model_type_from_field(to_field)
        if not to_model_type:
            return None

        to_info = self.registry.get_model_info(to_model_type)
        if not to_info:
            return None

        model_name, to_version = to_info
        from_version = self._get_simple_source_version(from_field, model_name)

        return (model_name, from_version or to_version, to_version)

    def _get_simple_source_version(
        self, from_field: FieldInfo | None, model_name: str
    ) -> ModelVersion | None:
        """Get the source version for a simple nested field."""
        if not from_field:
            return None

        from_model_type = self._get_model_type_from_field(from_field)
        if not from_model_type:
            return None

        from_info = self.registry.get_model_info(from_model_type)
        if not from_info or from_info[0] != model_name:
            return None

        return from_info[1]

    def _get_discriminator_key(self, field: FieldInfo) -> str | None:
        """Extract the discriminator key from a field."""
        discriminator = field.discriminator
        if discriminator is None:
            return None

        if isinstance(discriminator, str):
            return discriminator

        if hasattr(discriminator, "discriminator") and isinstance(
            discriminator.discriminator, str
        ):
            return discriminator.discriminator

        return None

    def _find_discriminator_value(
        self, value: ModelData, discriminator_key: str, field: FieldInfo
    ) -> Any:
        """Find the discriminator value in data, checking field name and aliases."""
        if discriminator_key in value:
            return value[discriminator_key]

        for model_type in self._get_union_members(field):
            if discriminator_key not in model_type.model_fields:
                continue

            disc_field = model_type.model_fields[discriminator_key]

            for alias_attr in ["alias", "serialization_alias"]:
                alias = getattr(disc_field, alias_attr, None)
                if alias and alias in value:
                    return value[alias]

            val_alias = disc_field.validation_alias
            if isinstance(val_alias, str) and val_alias in value:
                return value[val_alias]

        return None

    def _get_discriminated_source_version(
        self,
        from_field: FieldInfo | None,
        discriminator_key: str,
        discriminator_value: Any,
        model_name: str,
        default_version: ModelVersion,
    ) -> ModelVersion:
        """Get the source version for a discriminated union field."""
        if not from_field:
            return default_version

        from_model_type = self._find_discriminated_type(
            from_field, discriminator_key, discriminator_value
        )
        if not from_model_type:
            return default_version

        from_info = self.registry.get_model_info(from_model_type)
        if not from_info or from_info[0] != model_name:
            return default_version

        return from_info[1]

    def _find_discriminated_type(
        self, field: FieldInfo, discriminator_key: str, discriminator_value: Any
    ) -> type[BaseModel] | None:
        """Find the right type in a discriminated union based on discriminator value."""
        for model_type in self._get_union_members(field):
            if self._model_matches_discriminator(
                model_type, discriminator_key, discriminator_value
            ):
                return model_type
        return None

    def _model_matches_discriminator(
        self,
        model_type: type[BaseModel],
        discriminator_key: str,
        discriminator_value: Any,
    ) -> bool:
        """Check if a model type matches a discriminator value."""
        if discriminator_key not in model_type.model_fields:
            return False

        field_info = model_type.model_fields[discriminator_key]

        if self._literal_matches_value(field_info.annotation, discriminator_value):
            return True

        return (
            field_info.default is not PydanticUndefined
            and field_info.default == discriminator_value
        )

    def _literal_matches_value(self, annotation: Any, value: Any) -> bool:
        """Check if a Literal type annotation contains a specific value."""
        if annotation is None:
            return False

        origin = get_origin(annotation)
        if origin is not Literal:
            return False

        literal_values = get_args(annotation)
        return value in literal_values

    def _get_union_members(self, field: FieldInfo) -> list[type[BaseModel]]:
        """Extract all BaseModel types from a union field."""
        annotation = field.annotation
        if annotation is None:
            return []

        origin = get_origin(annotation)

        if origin is None:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                return [annotation]
            return []

        if not self._is_union_type(origin):
            return []

        args = get_args(annotation)
        return [
            arg
            for arg in args
            if arg is not type(None)
            and isinstance(arg, type)
            and issubclass(arg, BaseModel)
        ]

    def _is_union_type(self, origin: Any) -> bool:
        """Check if an origin type represents a Union."""
        if origin is Union:
            return True

        if hasattr(types, "UnionType"):
            try:
                return origin is types.UnionType
            except (ImportError, AttributeError):
                pass

        return False

    def _get_model_type_from_field(self, field: FieldInfo) -> type[BaseModel] | None:
        """Extract the Pydantic model type from a field."""
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
