"""ModelManager class."""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

from ._migration_manager import MigrationManager
from ._model_builder import (
    auto_detect_nested_pins,
    build_composite_model,
    collect_nested_model_dependencies,
    generate_stub_file_content,
    merge_nested_pins,
    normalize_pin_specs,
)
from ._registry import Registry
from ._schema_manager import SchemaManager
from .avro_schema import AvroExporter
from .avro_types import AvroRecordSchema
from .exceptions import MigrationError, ModelNotFoundError
from .migration_hooks import MigrationHook
from .migration_testing import (
    MigrationTestCase,
    MigrationTestCases,
    MigrationTestResult,
    MigrationTestResults,
)
from .model_diff import ModelDiff
from .model_version import ModelVersion
from .protobuf_schema import ProtoExporter
from .schema_config import SchemaConfig
from .types import (
    DecoratedBaseModel,
    JsonSchema,
    JsonSchemaGenerator,
    MigrationFunc,
    ModelData,
    NestedModelInfo,
    SchemaTransformer,
)
from .typescript_schema import TypeScriptConfig, TypeScriptExporter


class ModelManager:
    """High-level interface for versioned model management and schema generation.

    ModelManager provides a unified API for managing schema evolution across different
    versions of Pydantic models. It handles model registration, automatic migration
    between versions, customizable schema generation, and batch processing operations.

    Example:
        **Basic Usage**:

        ```python
        from pyrmute import ModelManager, ModelData

        manager = ModelManager()

        # Register model versions
        @manager.model("User", "1.0.0")
        class UserV1(BaseModel):
            name: str

        @manager.model("User", "2.0.0")
        class UserV2(BaseModel):
            name: str
            email: str

        # Define migration between versions
        @manager.migration("User", "1.0.0", "2.0.0")
        def migrate(data: ModelData) -> ModelData:
            return {**data, "email": "unknown@example.com"}

        # Migrate legacy data
        old_data = {"name": "Alice"}
        user = manager.migrate(old_data, "User", "1.0.0", "2.0.0")
        # Result: UserV2(name="Alice", email="unknown@example.com")
        ```

        **Custom Schema Generation**:

        ```python
        from pydantic.json_schema import GenerateJsonSchema

        class CustomSchemaGenerator(GenerateJsonSchema):
            '''Add custom metadata to all schemas.'''
            def generate(
                self,
                schema: Mapping[str, Any],
                mode: JsonSchemaMode = "validation"
            ) -> JsonSchema:
                json_schema = super().generate(schema, mode=mode)
                json_schema["x-company"] = "Acme"
                json_schema["$schema"] = self.schema_dialect
                return json_schema

        # Set at manager level (applies to all schemas)
        manager = ModelManager(
            default_schema_config=SchemaConfig(
                schema_generator=CustomSchemaGenerator,
                mode="validation",
                by_alias=True
            )
        )

        @manager.model("User", "1.0.0")
        class User(BaseModel):
            name: str = Field(title="Full Name")
            email: str

        # Get schema with default config
        schema = manager.get_schema("User", "1.0.0")
        # Will include x-company: "Acme"
        ```

        **Advanced Features**:

        ```python
        # Batch migration with parallel processing
        users = manager.migrate_batch(
            legacy_users, "User", "1.0.0", "2.0.0",
            parallel=True, max_workers=4
        )

        # Stream large datasets efficiently
        for user in manager.migrate_batch_streaming(
            large_dataset, "User", "1.0.0", "2.0.0"
         ):
            save_to_database(user)

        # Compare versions and export schemas
        diff = manager.diff("User", "1.0.0", "2.0.0")
        print(diff.to_markdown())
        manager.dump_schemas("schemas/", separate_definitions=True)

        # Test migrations with validation
        results = manager.test_migration(
            "User", "1.0.0", "2.0.0",
            test_cases=[
                (
                     {"name": "Alice"},
                     {"name": "Alice", "email": "unknown@example.com"}
                )
            ]
        )
        results.assert_all_passed()
        ```

        **Schema Transformers**:

        ```python
        manager = ModelManager()

        @manager.model("Product", "1.0.0")
        class Product(BaseModel):
            name: str
            price: float

        # Add transformer for specific model
        @manager.schema_transformer("Product", "1.0.0")
        def add_examples(schema: JsonSchema) -> JsonSchema:
            schema["examples"] = [{"name": "Widget", "price": 9.99}]
            return schema

        schema = manager.get_schema("Product", "1.0.0")
        # Will include examples
        ```
    """

    def __init__(self: Self, default_schema_config: SchemaConfig | None = None) -> None:
        """Initialize the versioned model manager.

        Args:
            default_schema_config: Default configuration for schema generation
                applied to all schema operations unless overridden.
        """
        self._registry = Registry()
        self._migration_manager = MigrationManager(self._registry)
        self._schema_manager = SchemaManager(
            self._registry, default_config=default_schema_config
        )

    def model(
        self: Self,
        name: str,
        version: str | ModelVersion,
        enable_ref: bool = False,
        backward_compatible: bool = False,
    ) -> Callable[[type[DecoratedBaseModel]], type[DecoratedBaseModel]]:
        """Register a versioned model.

        Args:
            name: Name of the model.
            version: Semantic version.
            enable_ref: If True, this model can be referenced via $ref in separate
                schema files. If False, it will always be inlined.
            backward_compatible: If True, this model does not need a migration function
                to migrate to the next version. If a migration function is defined it
                will use it.

        Returns:
            Decorator function for model class.

        Example:
            ```python
            # Model that will be inlined (default)
            @manager.model("Address", "1.0.0")
            class AddressV1(BaseModel):
                street: str

            # Model that can be a separate schema with $ref
            @manager.model("City", "1.0.0", enable_ref=True)
            class CityV1(BaseModel):
                city: City
            ```
        """
        return self._registry.register(name, version, enable_ref, backward_compatible)

    def get(self: Self, name: str, version: str | ModelVersion) -> type[BaseModel]:
        """Get a model by name and version.

        Args:
            name: Name of the model.
            version: Semantic version (returns latest if None).

        Returns:
            Model class.
        """
        return self._registry.get_model(name, version)

    def get_latest(self: Self, name: str) -> type[BaseModel]:
        """Get the latest version of a model by name.

        Args:
            name: Name of the model.

        Returns:
            Model class.
        """
        return self._registry.get_latest(name)

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

    def get_nested_version_pins(
        self: Self, name: str, version: str | ModelVersion
    ) -> dict[str, tuple[str, str]]:
        """Get the nested version pins for a model version.

        Args:
            name: Model name
            version: Model version

        Returns:
            Map of field paths to (model_name, version_string) tuples

        Example:
            ```python
            pins = manager.get_nested_version_pins("User", "2.0.0")
            # Returns: {"address": ("Address", "2.0.0")}
            ```
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version
        pins = self._registry.get_nested_pins(name, ver)
        return {
            path: (model_name, str(model_ver))
            for path, (model_name, model_ver) in pins.items()
        }

    def has_composite_version(
        self: Self, name: str, version: str | ModelVersion
    ) -> bool:
        """Check if a model version uses composite versioning.

        Args:
            name: Model name
            version: Model version

        Returns:
            True if this version has nested pins defined

        Example:
            ```python
            if manager.has_composite_version("User", "2.0.0"):
                pins = manager.get_nested_version_pins("User", "2.0.0")
                print(f"Composite version with pins: {pins}")
            ```
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version
        return self._registry.has_nested_pins(name, ver)

    def composite(
        self: Self,
        name: str,
        version: str | ModelVersion,
        inherit_from: str | ModelVersion | None = None,
        enable_ref: bool = False,
        **pins: str,
    ) -> Callable[[type[BaseModel]], type[BaseModel]]:
        """Register a composite model with automatic nested version detection.

        This decorator automatically detects nested models from type hints and
        recursively resolves their versions. You only specify immediate children, and
        deeper nesting is handled automatically.

        Args:
            name: Model name to register
            version: Version string (e.g., "1.0.0")
            inherit_from: Inherit nested pins from this version, then override
            enable_ref: If True, this model can be referenced via $ref in schemas
            **pins: Optional explicit pins to override auto-detection
                Format: field_name="version" or field_name="ModelName:version"

        Returns:
            Decorator function for the model class

        Example:
            ```python
            # Simple - auto-detects everything
            @manager.composite("User", "1.0.0")
            class UserV1(BaseModel):
                name: str
                address: AddressV1  # Auto-detected as Address:1.0.0

            # With inheritance - only specify what changed
            @manager.composite("User", "2.0.0", inherit_from="1.0.0")
            class UserV2(BaseModel):
                name: str
                address: AddressV2  # Auto-detected as Address:2.0.0

            # Override auto-detection when needed
            @manager.composite("User", "3.0.0", address="2.0.0")
            class UserV3(BaseModel):
                name: str
                address: AddressV2  # Explicitly pinned to Address:2.0.0

            # Deep nesting - auto-detects entire tree
            @manager.composite("Company", "1.0.0")
            class CompanyV1(BaseModel):
                department: DepartmentV1  # Detects many levels deep
            ```
        """

        def decorator(cls: type[BaseModel]) -> type[BaseModel]:
            ver = ModelVersion.parse(version) if isinstance(version, str) else version

            inherited_pins: dict[str, tuple[str, ModelVersion]] = {}
            if inherit_from is not None:
                inherit_ver = (
                    ModelVersion.parse(inherit_from)
                    if isinstance(inherit_from, str)
                    else inherit_from
                )
                inherited_pins = self._registry.get_nested_pins(name, inherit_ver)

            detected_pins = auto_detect_nested_pins(cls, self._registry.get_model_info)
            normalized_explicit = (
                normalize_pin_specs(cls, pins, self._registry.get_model_info)
                if pins
                else {}
            )

            all_pins = merge_nested_pins(
                inherited_pins, detected_pins, normalized_explicit
            )

            nested_models = {}
            for field_path, (nested_name, nested_ver) in all_pins.items():
                nested_models[field_path] = self._registry.get_model(
                    nested_name, nested_ver
                )

            model_class_name = f"{name}_v{ver.major}_{ver.minor}_{ver.patch}"
            composed_model = build_composite_model(
                model_class_name, cls, nested_models, module=cls.__module__
            )

            self._registry.register(name, ver, enable_ref, False)(composed_model)
            self._registry.set_nested_pins(name, ver, all_pins)

            return composed_model

        return decorator

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

    def add_hook(self: Self, hook: MigrationHook) -> None:
        """Register a migration hook for observability/logging.

        Args:
            hook: Migration hook instance to register.

        Example:
            ```python
            from pyrmute import MetricsHook, MigrationHook
            import logging

            # Use built-in metrics hook
            metrics = MetricsHook()
            manager.add_hook(metrics)

            # Add custom logging hook
            class LoggingHook(MigrationHook):
                def before_migrate(
                    self,
                    name: str,
                    from_version: ModelVersion,
                    to_version: ModelVersion,
                    data: Mapping[str, Any],
                ) -> None:
                    logging.info(f"Starting migration: {name}")
                    return data

            manager.add_hook(LoggingHook())
            ```
        """
        self._migration_manager.add_hook(hook)

    def remove_hook(self: Self, hook: MigrationHook) -> None:
        """Remove a previously registered hook.

        Args:
            hook: Migration hook instance to remove.
        """
        self._migration_manager.remove_hook(hook)

    def clear_hooks(self: Self) -> None:
        """Remove all registered hooks."""
        self._migration_manager.clear_hooks()

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
            ```python
            if manager.has_migration_path("User", "1.0.0", "3.0.0"):
                users = manager.migrate_batch(old_users, "User", "1.0.0", "3.0.0")
            else:
                logger.error("Cannot migrate users to v3.0.0")
            ```
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
            ```python
            data = {"name": "Alice"}
            is_valid = manager.validate_data(data, "User", "1.0.0")
            # Returns: True

            is_valid = manager.validate_data(data, "User", "2.0.0")
            # Returns: False, missing required field 'email'
            ```
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

    def migrate_as(
        self: Self,
        data: ModelData,
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
        target_type: type[DecoratedBaseModel],
    ) -> DecoratedBaseModel:
        """Migrate data between versions with type safety.

        This is a type-safe variant of migrate() that returns a specific model type when
        you provide the target type explicitly.

        Args:
            data: Data dictionary to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.
            target_type: The expected model class type.

        Returns:
            Migrated model instance of the specified type.

        Example:
            ```python
            old_data = {"name": "Alice"}
            user: UserV2 = manager.migrate_as(
                old_data, "User", "1.0.0", "2.0.0", UserV2
            )
            # Type checker knows user is UserV2, not just BaseModel
            ```
        """
        migrated_data = self.migrate_data(data, name, from_version, to_version)
        return target_type.model_validate(migrated_data)

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
            max_workers: Maximum number of workers for parallel processing.
            use_processes: If True, use ProcessPoolExecutor instead of
                ThreadPoolExecutor.

        Returns:
            List of migrated BaseModel instances.
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

    def migrate_batch_as(  # noqa: PLR0913
        self: Self,
        data_list: Iterable[ModelData],
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
        target_type: type[DecoratedBaseModel],
        parallel: bool = False,
        max_workers: int | None = None,
        use_processes: bool = False,
    ) -> list[DecoratedBaseModel]:
        """Migrate multiple data items between versions with type safety.

        This is a type-safe variant of migrate_batch() that returns a specific model
        type when you provide the target type explicitly.

        Args:
            data_list: Iterable of data dictionaries to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.
            target_type: The expected model class type.
            parallel: If True, use parallel processing.
            max_workers: Maximum number of workers for parallel processing.
            use_processes: If True, use ProcessPoolExecutor instead of
                ThreadPoolExecutor.

        Returns:
            List of migrated model instances of the specified type.

        Example:
            ```python
            old_users = [{"name": "Alice"}, {"name": "Bob"}]
            users: list[UserV2] = manager.migrate_batch_as(
                old_users, "User", "1.0.0", "2.0.0", UserV2,
                parallel=True, max_workers=4
            )
            ```
        """
        data_list = list(data_list)

        if not data_list:
            return []

        if not parallel:
            return [
                self.migrate_as(item, name, from_version, to_version, target_type)
                for item in data_list
            ]

        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        with executor_class(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.migrate_as, item, name, from_version, to_version, target_type
                )
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

        Args:
            data_list: Iterable of data dictionaries to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.
            chunk_size: Number of items to process in each chunk.

        Yields:
            Migrated BaseModel instances.
        """
        chunk = []

        for item in data_list:
            chunk.append(item)

            if len(chunk) >= chunk_size:
                yield from self.migrate_batch(chunk, name, from_version, to_version)
                chunk = []

        if chunk:
            yield from self.migrate_batch(chunk, name, from_version, to_version)

    def migrate_batch_streaming_as(  # noqa: PLR0913
        self: Self,
        data_list: Iterable[ModelData],
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
        target_type: type[DecoratedBaseModel],
        chunk_size: int = 100,
    ) -> Iterable[DecoratedBaseModel]:
        """Migrate data in chunks with type safety, yielding results as they complete.

        This is a type-safe variant of migrate_batch_streaming() that returns a specific
        model type when you provide the target type explicitly.

        Args:
            data_list: Iterable of data dictionaries to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.
            target_type: The expected model class type.
            chunk_size: Number of items to process in each chunk.

        Yields:
            Migrated model instances of the specified type.

        Example:
            ```python
            for user in manager.migrate_batch_streaming_as(
                large_dataset, "User", "1.0.0", "2.0.0", UserV2
            ):
                save_to_database(user)  # user is typed as UserV2
            ```
        """
        chunk = []

        for item in data_list:
            chunk.append(item)

            if len(chunk) >= chunk_size:
                yield from self.migrate_batch_as(
                    chunk, name, from_version, to_version, target_type
                )
                chunk = []

        if chunk:
            yield from self.migrate_batch_as(
                chunk, name, from_version, to_version, target_type
            )

    def migrate_batch_data_streaming(
        self: Self,
        data_list: Iterable[ModelData],
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
        chunk_size: int = 100,
    ) -> Iterable[ModelData]:
        """Migrate data in chunks, yielding raw dictionaries as they complete.

        Args:
            data_list: Iterable of data dictionaries to migrate.
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.
            chunk_size: Number of items to process in each chunk.

        Yields:
            Raw migrated dictionaries.
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

    def test_migration(
        self: Self,
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
        test_cases: MigrationTestCases,
    ) -> MigrationTestResults:
        """Test a migration with multiple test cases.

        Args:
            name: Name of the model.
            from_version: Source version to migrate from.
            to_version: Target version to migrate to.
            test_cases: List of test cases.

        Returns:
            MigrationTestResults containing individual results for each test case.
        """
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

    def diff(
        self: Self,
        name: str,
        from_version: str | ModelVersion,
        to_version: str | ModelVersion,
    ) -> ModelDiff:
        """Get a detailed diff between two model versions.

        Args:
            name: Name of the model.
            from_version: Source version.
            to_version: Target version.

        Returns:
            ModelDiff with detailed change information.
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

    def set_default_schema_generator(
        self: Self, generator: JsonSchemaGenerator | type[GenerateJsonSchema]
    ) -> None:
        """Set the default schema generator for all schemas.

        This is a convenience method that updates the default schema configuration.

        Args:
            generator: Custom schema generator - either a callable or GenerateJsonSchema
                class.

        Example:
            **Class**:

            ```python
            from pydantic.json_schema import GenerateJsonSchema


            class MyGenerator(GenerateJsonSchema):
                def generate(
                    self,
                    schema: Mapping[str, Any],
                    mode: JsonSchemaMode = "validation"
                ) -> JsonSchema:
                    json_schema = super().generate(schema, mode=mode)
                    json_schema["x-custom"] = True
                    json_schema["$schema"] = self.schema_dialect
                    return json_schema

            manager = ModelManager()
            manager.set_default_schema_generator(MyGenerator)

            # All subsequent schema calls will use MyGenerator
            schema = manager.get_schema("User", "1.0.0")
            ```

            **Callable**:

            ```python
            def my_generator(model: type[BaseModel]) -> JsonSchema:
                schema = model.model_json_schema()
                schema["x-custom"] = True
                return schema

            manager = ModelManager()
            manager.set_default_schema_generator(my_generator)
            ```
        """
        self._schema_manager.set_default_schema_generator(generator)

    def schema_transformer(
        self: Self,
        name: str,
        version: str | ModelVersion,
    ) -> Callable[[SchemaTransformer], SchemaTransformer]:
        """Decorator to register a schema transformer for a model version.

        Transformers are simple functions that modify a schema after generation.
        They're useful for model-specific customizations that don't require deep
        integration with Pydantic's generation process.

        Args:
            name: Name of the model.
            version: Model version.

        Returns:
            Decorator function.

        Example:
            ```python
            @manager.schema_transformer("User", "1.0.0")
            def add_auth_metadata(schema: JsonSchema) -> JsonSchema:
                schema["x-requires-auth"] = True
                schema["x-auth-level"] = 'admin'
                return schema

            @manager.schema_transformer("Product", "2.0.0")
            def add_product_examples(schema: JsonSchema) -> JsonSchema:
                schema["examples"] = [
                    {"name": "Widget", "price": 9.99},
                    {"name": "Gadget", "price": 19.99}
                ]
                return schema
            ```
        """

        def decorator(func: SchemaTransformer) -> SchemaTransformer:
            self._schema_manager.register_transformer(name, version, func)
            return func

        return decorator

    def get_schema_transformers(
        self: Self,
        name: str,
        version: str | ModelVersion,
    ) -> list[SchemaTransformer]:
        """Get all transformers for a model version.

        Args:
            name: Name of the model.
            version: Model version.

        Returns:
            List of transformer functions.

        Example:
            ```python
            transformers = manager.get_schema_transformers("User", "1.0.0")
            print(f"Found {len(transformers)} transformers")
            ```
        """
        return self._schema_manager.get_transformers(name, version)

    def clear_schema_transformers(
        self: Self,
        name: str | None = None,
        version: str | ModelVersion | None = None,
    ) -> None:
        """Clear schema transformers.

        Args:
            name: Optional model name. If None, clears all.
            version: Optional version. If None, clears all versions of model.

        Example:
            ```python
            # Clear all transformers
            manager.clear_schema_transformers()

            # Clear User transformers
            manager.clear_schema_transformers("User")

            # Clear specific version
            manager.clear_schema_transformers("User", "1.0.0")
            ```
        """
        self._schema_manager.clear_transformers(name, version)

    def get_schema(
        self: Self,
        name: str,
        version: str | ModelVersion,
        config: SchemaConfig | None = None,
        **kwargs: Any,
    ) -> JsonSchema:
        """Get JSON schema for a specific version.

        Args:
            name: Name of the model.
            version: Semantic version.
            config: Optional schema configuration (overrides default).
            **kwargs: Additional schema generation arguments (e.g.,
                mode="serialization").

        Returns:
            JSON schema dictionary.

        Example:
            ```python
            # Use default config
            schema = manager.get_schema("User", "1.0.0")

            # Override with custom config
            config = SchemaConfig(mode="serialization")
            schema = manager.get_schema("User", "1.0.0", config=config)

            # Quick override with kwargs
            schema = manager.get_schema("User", "1.0.0", mode="serialization")
            ```
        """
        return self._schema_manager.get_schema(name, version, config=config, **kwargs)

    def dump_schemas(
        self: Self,
        output_dir: str | Path,
        indent: int = 2,
        separate_definitions: bool = False,
        ref_template: str | None = None,
        config: SchemaConfig | None = None,
    ) -> None:
        """Export all schemas to JSON files.

        Args:
            output_dir: Directory path for output.
            indent: JSON indentation level.
            separate_definitions: If True, create separate schema files for nested
                models and use $ref to reference them.
            ref_template: Template for $ref URLs when separate_definitions=True.
            config: Optional schema configuration for all exported schemas.

        Example:
            ```python
            # Export with custom generator
            config = SchemaConfig(
                schema_generator=CustomGenerator,
                mode="validation"
            )
            manager.dump_schemas("schemas/", config=config)

            # Export validation and serialization schemas separately
            manager.dump_schemas(
                "schemas/validation/",
                config=SchemaConfig(mode="validation")
            )
            manager.dump_schemas(
                "schemas/serialization/",
                config=SchemaConfig(mode="serialization")
            )
            ```
        """
        self._schema_manager.dump_schemas(
            output_dir, indent, separate_definitions, ref_template, config=config
        )

    def get_avro_schema(
        self: Self,
        name: str,
        version: str | ModelVersion,
        namespace: str | None = None,
        include_docs: bool = True,
        versioned_namespace: bool = False,
    ) -> AvroRecordSchema:
        """Get Avro schema for a specific model version.

        Args:
            name: Name of the model.
            version: Semantic version.
            namespace: Avro namespace. If None, uses "com.example".
            include_docs: Whether to include field descriptions.
            versioned_namespace: Include model version in namespace. Default False.

        Returns:
            Avro schema typed dictionary.

        Example:
            ```python
            # Get Avro schema for a model
            schema = manager.get_avro_schema("User", "1.0.0", namespace="com.myapp")

            # Use with fastavro
            import fastavro

            with open("users.avro", "wb") as out:
                fastavro.writer(out, schema, records)

            # Use with Kafka
            from confluent_kafka import avro

            producer = avro.AvroProducer({
                "bootstrap.servers": "localhost:9092",
                "schema.registry.url": "http://localhost:8081"
            }, default_value_schema=avro.loads(json.dumps(schema)))
            ```
        """
        namespace = namespace or "com.example"
        exporter = AvroExporter(
            self._registry,
            namespace=namespace,
            include_docs=include_docs,
        )
        return exporter.export_schema(
            name, version, versioned_namespace=versioned_namespace
        )

    def dump_avro_schemas(
        self: Self,
        output_dir: str | Path,
        namespace: str | None = None,
        indent: int = 2,
        include_docs: bool = True,
        versioned_namespace: bool = False,
    ) -> dict[str, dict[str, AvroRecordSchema]]:
        """Export all schemas as Apache Avro schemas.

        Avro is commonly used in data engineering and event streaming, particularly with
        Apache Kafka, Hadoop, and data lakes.

        Args:
            output_dir: Directory path for output.
            namespace: Avro namespace (e.g., "com.mycompany.events").
                If None, uses "com.example".
            indent: JSON indentation level.
            include_docs: Whether to include field descriptions in schemas.
            versioned_namespace: Include model versions in namespaces. Default False.

        Returns:
            Dictionary mapping model names to versions to Avro schemas.

        Example:
            ```python
            # Export all models as Avro schemas
            manager.dump_avro_schemas(
                "schemas/avro/",
                namespace="com.mycompany.events",
                include_docs=True
            )

            # Creates files like:
            # schemas/avro/User_v1_0_0.avsc
            # schemas/avro/User_v2_0_0.avsc
            # schemas/avro/Order_v1_0_0.avsc

            # Use with Kafka Schema Registry
            from confluent_kafka.schema_registry import SchemaRegistryClient

            client = SchemaRegistryClient({"url": "http://localhost:8081"})

            with open("schemas/avro/User_v1_0_0.avsc") as f:
                schema_str = f.read()
                client.register_schema("user-value", schema_str)
            ```
        """
        namespace = namespace or "com.example"
        exporter = AvroExporter(
            self._registry,
            namespace=namespace,
            include_docs=include_docs,
        )
        return exporter.export_all_schemas(
            output_dir, indent, versioned_namespace=versioned_namespace
        )

    def get_proto_schema(
        self: Self,
        name: str,
        version: str | ModelVersion,
        package: str | None = None,
        include_docs: bool = True,
        use_proto3: bool = True,
    ) -> str:
        """Get Protocol Buffer schema for a specific model version.

        Args:
            name: Name of the model.
            version: Semantic version.
            package: Protobuf package name (e.g., "com.mycompany.users").
                This is a namespace that remains consistent across versions.
                Defaults to "com.example".
            include_docs: Whether to include documentation as comments.
            use_proto3: Use proto3 syntax (True) or proto2 (False).
                Defaults to True (proto3 is recommended for new projects).

        Returns:
            Protocol Buffer file as a string.

        Example:
            ```python
            # Get proto3 schema with custom package
            schema = manager.get_proto_schema(
                "User", "1.0.0",
                package="com.mycompany.users"
            )

            # Use with protobuf compiler
            # protoc --go_out=. schemas/protos/User_v1_0_0.proto
            ```
        """
        package = package or "com.example"
        exporter = ProtoExporter(
            self._registry,
            package=package,
            include_docs=include_docs,
            use_proto3=use_proto3,
        )
        return exporter.export_schema(name, version)

    def dump_proto_schemas(
        self: Self,
        output_dir: str | Path,
        package: str | None = None,
        include_docs: bool = True,
        use_proto3: bool = True,
    ) -> dict[str, dict[str, str]]:
        """Export all schemas as Protocol Buffer schemas.

        Protocol Buffers are commonly used for gRPC, microservices communication,
        and efficient binary serialization.

        Args:
            output_dir: Directory path for output.
            package: Protobuf package name (e.g., "com.mycompany.events").
                This namespace remains consistent across all versions.
                Defaults to "com.example".
            include_docs: Whether to include documentation as comments.
            use_proto3: Use proto3 syntax (True) or proto2 (False).
                Defaults to True (proto3 is recommended for new projects).

        Returns:
            Dictionary mapping model names to versions to Protocol Buffer schemas.

        Example:
            ```python
            # Export all models as proto3 schemas
            manager.dump_proto_schemas(
                "schemas/protos/",
                package="com.mycompany.events",
                include_docs=True,
                use_proto3=True,
            )

            # Creates files like:
            # schemas/protos/User_v1_0_0.proto
            # schemas/protos/User_v2_0_0.proto
            # schemas/protos/Order_v1_0_0.proto

            # Compile with protoc:
            # protoc --go_out=. schemas/protos/*.proto
            ```
        """
        package = package or "com.example"
        exporter = ProtoExporter(
            self._registry,
            package=package,
            include_docs=include_docs,
            use_proto3=use_proto3,
        )
        return exporter.export_all_schemas(output_dir)

    def get_typescript_schema(
        self: Self,
        name: str,
        version: str | ModelVersion,
        style: Literal["interface", "type", "zod"] = "interface",
        config: TypeScriptConfig | None = None,
    ) -> str:
        """Get TypeScript schema for a specific model version.

        Args:
            name: Name of the model.
            version: Semantic version.
            style: Output style - 'interface', 'type', or 'zod'.
            config: Optional configuration for schema generation.

        Returns:
            TypeScript schema code as a string.

        Example:
            ```python
            # Get TypeScript interface for a model
            schema = manager.get_typescript_schema("User", "1.0.0", style="interface")
            print(schema)
            # export interface UserV1_0_0 {
            #   name: string;
            #   email: string;
            #   age?: number;
            # }

            # Get Zod schema with validation
            zod_schema = manager.get_typescript_schema("User", "2.0.0", style="zod")
            print(zod_schema)
            # import { z } from 'zod';
            #
            # export const UserV2_0_0Schema = z.object({
            #   name: z.string(),
            #   email: z.string().email(),
            #   age: z.number().int().positive().optional(),
            # });
            #
            # export type UserV2_0_0 = z.infer<typeof UserV2_0_0Schema>;

            # Customize output with config
            from pyrmute.typescript_types import TypeScriptConfig

            config = TypeScriptConfig(
                date_format="timestamp",
                enum_style="enum",
                strict_null_checks=True,
            )
            schema = manager.get_typescript_schema(
                "Event", "1.0.0", style="interface", config=config
            )

            # Use in frontend
            # Save to file and import in TypeScript:
            # import { UserV1_0_0 } from './types/user_v1_0_0';
            #
            # const user: UserV1_0_0 = {
            #   name: "Alice",
            #   email: "alice@example.com"
            # };

            # Use with Zod for runtime validation
            # import { UserV2_0_0Schema } from './schemas/user_v2_0_0';
            #
            # const result = UserV2_0_0Schema.safeParse(data);
            # if (result.success) {
            #   console.log(result.data);
            # } else {
            #   console.error(result.error);
            # }
            ```
        """
        exporter = TypeScriptExporter(self._registry, style=style, config=config)
        return exporter.export_schema(name, version)

    def dump_typescript_schemas(
        self: Self,
        output_dir: str | Path,
        style: Literal["interface", "type", "zod"] = "interface",
        config: TypeScriptConfig | None = None,
        organization: Literal["flat", "major_version", "model"] = "flat",
        include_barrel_exports: bool = True,
    ) -> dict[str, dict[str, str]]:
        r"""Export all schemas as TypeScript schemas.

        TypeScript schemas enable type-safe frontend development and can include runtime
        validation with Zod. Ideal for full-stack applications where backend Pydantic
        models need to be synchronized with frontend TypeScript code.

        Args:
            output_dir: Directory path for output.
            style: Output style - 'interface' (default), 'type', or 'zod'.
            config: Optional configuration for schema generation.
            organization: Directory structure for output files:
                - 'flat': All files in output directory (Model.v1.0.0.ts)
                - 'major_version': Organize by major version (v1/Model.v1.0.0.ts)
                - 'model': Organize by model name (Model/1.0.0.ts)
            include_barrel_exports: Whether to generate index.ts files for easier
                imports.

        Returns:
            Dictionary mapping model names to versions to TypeScript schema code.

        Example:
            ```python
            # Export all models as TypeScript interfaces (flat structure)
            manager.dump_typescript_schemas(
                "frontend/src/types/",
                style="interface"
            )
            # Creates files like:
            # frontend/src/types/User.v1.0.0.ts
            # frontend/src/types/User.v2.0.0.ts
            # frontend/src/types/Order.v1.0.0.ts

            # Export organized by major version (recommended)
            manager.dump_typescript_schemas(
                "frontend/src/types/",
                style="interface",
                organization="major_version"
            )
            # Creates:
            # frontend/src/types/v1/User.v1.0.0.ts
            # frontend/src/types/v1/index.ts
            # frontend/src/types/v2/User.v2.0.0.ts
            # frontend/src/types/v2/index.ts
            # frontend/src/types/index.ts (re-exports latest)

            # Export organized by model
            manager.dump_typescript_schemas(
                "frontend/src/types/",
                organization="model"
            )
            # Creates:
            # frontend/src/types/User/1.0.0.ts
            # frontend/src/types/User/2.0.0.ts
            # frontend/src/types/User/index.ts (re-exports latest)
            # frontend/src/types/Order/1.0.0.ts
            # frontend/src/types/Order/index.ts
            # frontend/src/types/index.ts

            # Export as Zod schemas with validation
            manager.dump_typescript_schemas(
                "frontend/src/schemas/",
                style="zod",
                organization="major_version"
            )

            # Export with custom configuration
            from pyrmute.typescript_types import TypeScriptConfig

            config = TypeScriptConfig(
                strict_null_checks=True,
                date_format="iso",   # or "timestamp"
                enum_style="union",  # or "enum"
            )
            manager.dump_typescript_schemas(
                "frontend/src/types/",
                style="interface",
                config=config,
                organization="major_version"
            )

            # Use in React application with barrel exports
            # import { User } from '@/types/v1';   // Gets latest v1
            # import { Order } from '@/types';     // Gets latest version
            #
            # interface UserCardProps {
            #   user: User;
            # }
            #
            # export function UserCard({ user }: UserCardProps) {
            #   return <div>{user.name}</div>;
            # }

            # Use Zod for API response validation
            # import { UserSchema } from '@/schemas/v1';
            #
            # async function fetchUser(id: string) {
            #   const response = await fetch(`/api/users/${id}`);
            #   const data = await response.json();
            #   return UserSchema.parse(data); // Validates at runtime
            # }

            # Use with tRPC for end-to-end type safety
            # import { UserSchema } from '@/schemas/v1';
            #
            # export const userRouter = router({
            #   getUser: publicProcedure
            #     .input(z.object({ id: z.string() }))
            #     .output(UserSchema)
            #     .query(async ({ input }) => {
            #       return await db.user.findUnique({ where: { id: input.id } });
            #     }),
            # });

            # Integrate with build pipeline
            # Add to package.json scripts:
            # {
            #   "scripts": {
            #     "generate-types": "python -m pyrmute export -f typescript -o src/types",
            #     "prebuild": "npm run generate-types"
            #   }
            # }
            ```
        """  # noqa: E501
        exporter = TypeScriptExporter(self._registry, style=style, config=config)
        return exporter.export_all_schemas(
            output_dir, organization, include_barrel_exports
        )

    def export_type_stubs(
        self: Self,
        output_dir: str | Path,
        package_name: str | None = None,
        models: list[tuple[str, str]] | None = None,
        include_nested: bool = True,
    ) -> None:
        """Generate .pyi stub files for type checkers.

        This allows consuming packages to get full type hints for dynamically
        created composite models. The stub files should be included in your
        package distribution.

        Args:
            output_dir: Directory to write stub files (typically your package's src dir)
            package_name: Name of your package (for header comments). If None,
                infers from output_dir
            models: Specific models to export as (model_name, version) tuples.
                If None, exports all registered models
            include_nested: If True, automatically include nested model dependencies

        Example:
            ```python
            # In your package's build script:
            from your_package import manager

            manager.export_type_stubs(
                "src/your_package/",
                package_name="your_package"
            )

            # Generates: src/your_package/__init__.pyi
            ```

        Integration:
            Add to pyproject.toml:
            ```toml
                [tool.setuptools.package-data]
                your_package = ["py.typed", "*.pyi"]
            ```

            And create src/your_package/py.typed (empty file) to mark package as typed.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if package_name is None:
            package_name = output_path.name

        if models is None:
            models_to_export = [
                (name, str(version))
                for name in self.list_models()
                for version in self.list_versions(name)
            ]
        else:
            models_to_export = models

        model_data: list[tuple[str, ModelVersion, type[BaseModel]]] = []
        all_dependencies: set[tuple[str, ModelVersion]] = set()

        for model_name, version_str in models_to_export:
            version = ModelVersion.parse(version_str)
            model_class = self.get(model_name, version)
            model_data.append((model_name, version, model_class))

            if include_nested:
                deps = collect_nested_model_dependencies(
                    model_class, self._registry.get_model_info
                )
                all_dependencies.update(deps)

        if include_nested:
            for dep_name, dep_version in all_dependencies:
                if not any(
                    name == dep_name and ver == dep_version
                    for name, ver, _ in model_data
                ):
                    dep_class = self.get(dep_name, dep_version)
                    model_data.append((dep_name, dep_version, dep_class))

        model_data.sort(key=lambda x: (x[0], x[1]))

        stub_content = generate_stub_file_content(
            model_data, self._registry.get_model_info, package_name
        )

        stub_file = output_path / "__init__.pyi"
        stub_file.write_text(stub_content)

        py_typed = output_path / "py.typed"
        if not py_typed.exists():
            py_typed.write_text("")

    def export_type_stubs_by_module(
        self: Self,
        output_dir: str | Path,
        package_name: str,
        organization: dict[str, list[tuple[str, str]]],
    ) -> None:
        """Generate separate .pyi stub files organized by module.

        This is useful for large packages where you want to organize stubs
        into multiple files.

        Args:
            output_dir: Base directory for stub files
            package_name: Package name
            organization: Dict mapping module names to lists of (model_name, version)
                tuples

        Example:
            ```python
            manager.export_type_stubs_by_module(
                "src/myapp/models/",
                "myapp.models",
                {
                    "user": [("User", "1.0.0"), ("User", "2.0.0")],
                    "order": [("Order", "1.0.0")],
                }
            )
            # Generates:
            # src/myapp/models/user.pyi
            # src/myapp/models/order.pyi
            # src/myapp/models/__init__.pyi (with re-exports)
            ```
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        module_files = []

        for module_name, models in organization.items():
            model_data: list[tuple[str, ModelVersion, type[BaseModel]]] = []

            for model_name, version_str in models:
                version = ModelVersion.parse(version_str)
                model_class = self.get(model_name, version)
                model_data.append((model_name, version, model_class))

            model_data.sort(key=lambda x: (x[0], x[1]))
            stub_content = generate_stub_file_content(
                model_data,
                self._registry.get_model_info,
                f"{package_name}.{module_name}",
            )

            stub_file = output_path / f"{module_name}.pyi"
            stub_file.write_text(stub_content)
            module_files.append(module_name)

        init_lines = [
            f'"""Type stubs for {package_name}."""',
            "",
            "# This file is auto-generated. Do not edit manually.",
            "",
        ]
        init_lines.extend(
            f"from .{module_name} import *" for module_name in sorted(module_files)
        )

        init_stub = output_path / "__init__.pyi"
        init_stub.write_text("\n".join(init_lines))

        py_typed = output_path / "py.typed"
        if not py_typed.exists():
            py_typed.write_text("")
