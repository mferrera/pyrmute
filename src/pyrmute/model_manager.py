"""Model manager."""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel

from ._migration_manager import MigrationManager
from ._registry import Registry
from ._schema_manager import SchemaManager
from .exceptions import MigrationError, ModelNotFoundError
from .migration_testing import (
    MigrationTestCase,
    MigrationTestResult,
    MigrationTestResults,
)
from .model_diff import ModelDiff
from .model_version import ModelVersion
from .types import (
    DecoratedBaseModel,
    JsonSchema,
    JsonSchemaGenerator,
    MigrationFunc,
    ModelData,
    NestedModelInfo,
)


class ModelManager:
    """High-level interface for versioned model management.

    ModelManager provides a unified API for managing schema evolution across different
    versions of Pydantic models. It handles model registration, automatic migration
    between versions, schema generation, and batch processing operations.

    Attributes:
        registry: Registry instance managing all registered model versions.
        migration_manager: MigrationManager instance handling migration logic and paths.
        schema_manager: SchemaManager instance for JSON schema generation and export.

    Basic Usage:
        >>> manager = ModelManager()
        >>>
        >>> # Register model versions
        >>> @manager.model("User", "1.0.0")
        ... class UserV1(BaseModel):
        ...     name: str
        >>>
        >>> @manager.model("User", "2.0.0")
        ... class UserV2(BaseModel):
        ...     name: str
        ...     email: str
        >>>
        >>> # Define migration between versions
        >>> @manager.migration("User", "1.0.0", "2.0.0")
        ... def migrate(data: ModelData) -> ModelData:
        ...     return {**data, "email": "unknown@example.com"}
        >>>
        >>> # Migrate legacy data
        >>> old_data = {"name": "Alice"}
        >>> user = manager.migrate(old_data, "User", "1.0.0", "2.0.0")
        >>> # Result: UserV2(name="Alice", email="unknown@example.com")

    Advanced Features:
        >>> # Batch migration with parallel processing
        >>> users = manager.migrate_batch(
        ...     legacy_users, "User", "1.0.0", "2.0.0",
        ...     parallel=True, max_workers=4
        ... )
        >>>
        >>> # Stream large datasets efficiently
        >>> for user in manager.migrate_batch_streaming(large_dataset, "User", "1.0.0", "2.0.0"):
        ...     save_to_database(user)
        >>>
        >>> # Compare versions and export schemas
        >>> diff = manager.diff("User", "1.0.0", "2.0.0")
        >>> print(diff.to_markdown())
        >>> manager.dump_schemas("schemas/", separate_definitions=True)
        >>>
        >>> # Test migrations with validation
        >>> results = manager.test_migration(
        ...     "User", "1.0.0", "2.0.0",
        ...     test_cases=[
        ...         ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"})
        ...     ]
        ... )
        >>> results.assert_all_passed()
    """  # noqa: E501

    def __init__(self: Self) -> None:
        """Initialize the versioned model manager."""
        self._registry = Registry()
        self._migration_manager = MigrationManager(self._registry)
        self._schema_manager = SchemaManager(self._registry)

    def model(
        self: Self,
        name: str,
        version: str | ModelVersion,
        schema_generator: JsonSchemaGenerator | None = None,
        enable_ref: bool = False,
        backward_compatible: bool = False,
    ) -> Callable[[type[DecoratedBaseModel]], type[DecoratedBaseModel]]:
        """Register a versioned model.

        Args:
            name: Name of the model.
            version: Semantic version.
            schema_generator: Optional custom schema generator.
            enable_ref: If True, this model can be referenced via $ref in separate
                schema files. If False, it will always be inlined.
            backward_compatible: If True, this model does not need a migration function
                to migrate to the next version. If a migration function is defined it
                will use it.

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
        return self._registry.register(
            name, version, schema_generator, enable_ref, backward_compatible
        )

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
        return self._migration_manager.register_migration(
            name, from_version, to_version
        )

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
            return self._registry.get_latest(name)
        return self._registry.get_model(name, version)

    def has_migration_path(
        self: Self,
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
    ) -> bool:
        """Check if a migration path exists between two versions.

        Args:
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.

        Returns:
            True if a migration path exists, False otherwise.

        Example:
            >>> if manager.has_migration_path("User", "1.0.0", "3.0.0"):
            ...     users = manager.migrate_batch(old_users, "User", "1.0.0", "3.0.0")
            ... else:
            ...     logger.error("Cannot migrate users to v3.0.0")
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
        try:
            self._migration_manager.validate_migration_path(name, from_ver, to_ver)
            return True
        except (KeyError, ModelNotFoundError, MigrationError):
            return False

    def validate_data(
        self: Self,
        data: ModelData,
        name: str,
        version: str | ModelVersion,
    ) -> bool:
        """Check if data is valid for a specific model version.

        Validates whether the provided data conforms to the schema of the specified
        model version without raising an exception.

        Args:
            data: Data dictionary to validate.
            name: Name of the model.
            version: Semantic version to validate against.

        Returns:
            True if data is valid for the model version, False otherwise.

        Example:
            >>> data = {"name": "Alice"}
            >>> is_valid = manager.validate_data(data, "User", "1.0.0")
            >>> # Returns: True
            >>>
            >>> is_valid = manager.validate_data(data, "User", "2.0.0")
            >>> # Returns: False, missing required field 'email'
        """
        try:
            model = self.get(name, version)
            model.model_validate(data)
            return True
        except Exception:
            return False

    def migrate(
        self: Self,
        data: ModelData,
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
    ) -> BaseModel:
        """Migrate data between versions.

        Args:
            data: Data dictionary to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.

        Returns:
            Migrated BaseModel.
        """
        migrated_data = self.migrate_data(data, name, from_version, to_version)
        target_model = self.get(name, to_version)
        return target_model.model_validate(migrated_data)

    def migrate_data(
        self: Self,
        data: ModelData,
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
    ) -> ModelData:
        """Migrate data between versions.

        Args:
            data: Data dictionary to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.

        Returns:
            Raw migrated dictionary.
        """
        return self._migration_manager.migrate(data, name, from_version, to_version)

    def migrate_batch(  # noqa: PLR0913
        self: Self,
        data_list: Iterable[ModelData],
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
        parallel: bool = False,
        max_workers: int | None = None,
        use_processes: bool = False,
    ) -> list[BaseModel]:
        """Migrate multiple data items between versions.

        Args:
            data_list: Iterable of data dictionaries to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.
            parallel: If True, use parallel processing.
            max_workers: Maximum number of workers for parallel processing.  Defaults to
                None (uses executor default).
            use_processes: If True, use ProcessPoolExecutor instead of
                ThreadPoolExecutor. Useful for CPU-intensive migrations.

        Returns:
            List of migrated BaseModel instances.

        Example:
            >>> legacy_users = [
            ...     {"name": "Alice"},
            ...     {"name": "Bob"},
            ...     {"name": "Charlie"}
            ... ]
            >>> users = manager.migrate_batch(
            ...     legacy_users,
            ...     "User",
            ...     from_version="1.0.0",
            ...     to_version="3.0.0",
            ...     parallel=True
            ... )
        """
        data_list = list(data_list)

        if not data_list:
            return []

        if not parallel:
            return [
                self.migrate(item, name, from_version, to_version) for item in data_list
            ]

        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        with executor_class(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.migrate, item, name, from_version, to_version)
                for item in data_list
            ]
            return [future.result() for future in futures]

    def migrate_batch_data(  # noqa: PLR0913
        self: Self,
        data_list: Iterable[ModelData],
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
        parallel: bool = False,
        max_workers: int | None = None,
        use_processes: bool = False,
    ) -> list[ModelData]:
        """Migrate multiple data items between versions, returning raw dictionaries.

        Args:
            data_list: Iterable of data dictionaries to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.
            parallel: If True, use parallel processing.
            max_workers: Maximum number of workers for parallel processing.
            use_processes: If True, use ProcessPoolExecutor.

        Returns:
            List of raw migrated dictionaries.

        Example:
            >>> legacy_data = [{"name": "Alice"}, {"name": "Bob"}]
            >>> migrated_data = manager.migrate_batch_data(
            ...     legacy_data,
            ...     "User",
            ...     from_version="1.0.0",
            ...     to_version="2.0.0"
            ... )
        """
        data_list = list(data_list)

        if not data_list:
            return []

        if not parallel:
            return [
                self.migrate_data(item, name, from_version, to_version)
                for item in data_list
            ]

        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        with executor_class(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.migrate_data, item, name, from_version, to_version)
                for item in data_list
            ]
            return [future.result() for future in futures]

    def migrate_batch_streaming(
        self: Self,
        data_list: Iterable[ModelData],
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
        chunk_size: int = 100,
    ) -> Iterable[BaseModel]:
        """Migrate data in chunks, yielding results as they complete.

        Useful for large datasets where you want to start processing results before all
        migrations complete.

        Args:
            data_list: Iterable of data dictionaries to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.
            chunk_size: Number of items to process in each chunk.

        Yields:
            Migrated BaseModel instances.

        Example:
            >>> legacy_users = load_large_dataset()
            >>> for user in manager.migrate_batch_streaming(
            ...     legacy_users,
            ...     "User",
            ...     from_version="1.0.0",
            ...     to_version="3.0.0"
            ... ):
            ...     # Process each user as it's migrated
            ...     save_to_database(user)
        """
        chunk = []

        for item in data_list:
            chunk.append(item)

            if len(chunk) >= chunk_size:
                yield from self.migrate_batch(chunk, name, from_version, to_version)
                chunk = []

        if chunk:
            yield from self.migrate_batch(chunk, name, from_version, to_version)

    def migrate_batch_data_streaming(
        self: Self,
        data_list: Iterable[ModelData],
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
        chunk_size: int = 100,
    ) -> Iterable[ModelData]:
        """Migrate data in chunks, yielding raw dictionaries as they complete.

        Useful for large datasets where you want to start processing results before all
        migrations complete, without the validation overhead.

        Args:
            data_list: Iterable of data dictionaries to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.
            chunk_size: Number of items to process in each chunk.

        Yields:
            Raw migrated dictionaries.

        Example:
            >>> legacy_data = load_large_dataset()
            >>> for data in manager.migrate_batch_data_streaming(
            ...     legacy_data,
            ...     "User",
            ...     from_version="1.0.0",
            ...     to_version="3.0.0"
            ... ):
            ...     # Process raw data as it's migrated
            ...     bulk_insert_to_database(data)
        """
        chunk = []

        for item in data_list:
            chunk.append(item)

            if len(chunk) >= chunk_size:
                yield from self.migrate_batch_data(
                    chunk, name, from_version, to_version
                )
                chunk = []

        if chunk:
            yield from self.migrate_batch_data(chunk, name, from_version, to_version)

    def diff(
        self: Self,
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
    ) -> ModelDiff:
        """Get a detailed diff between two model versions.

        Compares field names, types, requirements, and default values to provide a
        comprehensive view of what changed between versions.

        Args:
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.

        Returns:
            ModelDiff with detailed change information.

        Example:
            >>> diff = manager.diff("User", "1.0.0", "2.0.0")
            >>> print(diff.to_markdown())
            >>> print(f"Added: {diff.added_fields}")
            >>> print(f"Removed: {diff.removed_fields}")
        """
        from_ver_str = str(
            ModelVersion.parse(from_version)
            if isinstance(from_version, str)
            else from_version
        )
        to_ver_str = str(
            ModelVersion.parse(to_version)
            if isinstance(to_version, str)
            else to_version
        )

        from_model = self.get(name, from_version)
        to_model = self.get(name, to_version)

        return ModelDiff.from_models(
            name=name,
            from_model=from_model,
            to_model=to_model,
            from_version=from_ver_str,
            to_version=to_ver_str,
        )

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
        return self._schema_manager.get_schema(name, version, **kwargs)

    def list_models(self: Self) -> list[str]:
        """Get list of all registered models.

        Returns:
            List of model names.
        """
        return self._registry.list_models()

    def list_versions(self: Self, name: str) -> list[ModelVersion]:
        """Get all versions for a model.

        Args:
            name: Name of the model.

        Returns:
            Sorted list of versions.
        """
        return self._registry.get_versions(name)

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
            separate_definitions: If True, create separate schema files for nested
                models and use $ref to reference them.
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
        self._schema_manager.dump_schemas(
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
            ref_template: Template for $ref URLs. Supports {model} and {version}
                placeholders. Defaults to relative file refs.
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
        self._schema_manager.dump_schemas(
            output_dir, indent, separate_definitions=True, ref_template=ref_template
        )

    def get_nested_models(
        self: Self,
        name: str,
        version: str | ModelVersion,
    ) -> list[NestedModelInfo]:
        """Get all nested models used by a model.

        Args:
            name: Name of the model.
            version: Semantic version.

        Returns:
            List of NestedModelInfo.
        """
        return self._schema_manager.get_nested_models(name, version)

    def test_migration(
        self: Self,
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
        test_cases: list[tuple[ModelData, ModelData] | MigrationTestCase],
    ) -> MigrationTestResults:
        """Test a migration with multiple test cases.

        Executes a migration on multiple test inputs and validates the outputs match
        expected values. Useful for regression testing and validating migration logic.

        Args:
            name: Name of the model.
            from_version: Source version to migrate from.
            to_version: Target version to migrate to.
            test_cases: List of test cases, either as (source, target) tuples or
                MigrationTestCase objects. If target is None, only verifies the
                migration completes without errors.

        Returns:
            MigrationTestResults containing individual results for each test case.

        Example:
            >>> # Using tuples (source, target)
            >>> results = manager.test_migration(
            ...     "User", "1.0.0", "2.0.0",
            ...     test_cases=[
            ...         ({"name": "Alice"}, {"name": "Alice", "email": "alice@example.com"}),
            ...         ({"name": "Bob"}, {"name": "Bob", "email": "bob@example.com"})
            ...     ]
            ... )
            >>> assert results.all_passed
            >>>
            >>> # Using MigrationTestCase objects
            >>> results = manager.test_migration(
            ...     "User", "1.0.0", "2.0.0",
            ...     test_cases=[
            ...         MigrationTestCase(
            ...             source={"name": "Alice"},
            ...             target={"name": "Alice", "email": "alice@example.com"},
            ...             description="Standard user migration"
            ...         )
            ...     ]
            ... )
            >>>
            >>> # Use in pytest
            >>> def test_user_migration():
            ...     results = manager.test_migration("User", "1.0.0", "2.0.0", test_cases)
            ...     results.assert_all_passed()  # Raises AssertionError with details if failed
            >>>
            >>> # Inspect failures
            >>> if not results.all_passed:
            ...     for failure in results.failures:
            ...         print(f"Failed: {failure.test_case.description}")
            ...         print(f"  Error: {failure.error}")
        """  # noqa: E501
        results = []

        for test_case_input in test_cases:
            if isinstance(test_case_input, tuple):
                test_case = MigrationTestCase(
                    source=test_case_input[0], target=test_case_input[1]
                )
            else:
                test_case = test_case_input

            try:
                actual = self.migrate_data(
                    test_case.source, name, from_version, to_version
                )

                if test_case.target is not None:
                    passed = actual == test_case.target
                    error = None if passed else "Output mismatch"
                else:
                    # Just verify it doesn't crash
                    passed = True
                    error = None

                results.append(
                    MigrationTestResult(
                        test_case=test_case, actual=actual, passed=passed, error=error
                    )
                )
            except Exception as e:
                results.append(
                    MigrationTestResult(
                        test_case=test_case, actual={}, passed=False, error=str(e)
                    )
                )

        return MigrationTestResults(results)
