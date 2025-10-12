"""ModelManager class."""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

from ._migration_manager import MigrationManager
from ._registry import Registry
from ._schema_manager import SchemaManager
from .exceptions import MigrationError, ModelNotFoundError
from .migration_testing import (
    MigrationTestCase,
    MigrationTestCases,
    MigrationTestResult,
    MigrationTestResults,
)
from .model_diff import ModelDiff
from .model_version import ModelVersion
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


class ModelManager:
    """High-level interface for versioned model management and schema generation.

    ModelManager provides a unified API for managing schema evolution across different
    versions of Pydantic models. It handles model registration, automatic migration
    between versions, customizable schema generation, and batch processing operations.

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

    Custom Schema Generation:
        >>> from pydantic.json_schema import GenerateJsonSchema
        >>>
        >>> class CustomSchemaGenerator(GenerateJsonSchema):
        ...     '''Add custom metadata to all schemas.'''
        ...     def generate(
        ...         self,
        ...         schema: Mapping[str, Any],
        ...         mode: JsonSchemaMode = "validation"
        ...     ) -> JsonSchema:
        ...         json_schema = super().generate(schema, mode=mode)
        ...         json_schema["x-company"] = "Acme"
        ...         json_schema["$schema"] = self.schema_dialect
        ...         return json_schema
        >>>
        >>> # Set at manager level (applies to all schemas)
        >>> manager = ModelManager(
        ...     default_schema_config=SchemaConfig(
        ...         schema_generator=CustomSchemaGenerator,
        ...         mode="validation",
        ...         by_alias=True
        ...     )
        ... )
        >>>
        >>> @manager.model("User", "1.0.0")
        ... class User(BaseModel):
        ...     name: str = Field(title="Full Name")
        ...     email: str
        >>>
        >>> # Get schema with default config
        >>> schema = manager.get_schema("User", "1.0.0")
        >>> # Will include x-company: 'Acme'

    Schema Transformers:
        >>> manager = ModelManager()
        >>>
        >>> @manager.model("Product", "1.0.0")
        ... class Product(BaseModel):
        ...     name: str
        ...     price: float
        >>>
        >>> # Add transformer for specific model
        >>> @manager.schema_transformer("Product", "1.0.0")
        ... def add_examples(schema: JsonSchema) -> JsonSchema:
        ...     schema["examples"] = [{"name": "Widget", "price": 9.99}]
        ...     return schema
        >>>
        >>> schema = manager.get_schema("Product", "1.0.0")
        >>> # Will include examples

    Advanced Features:
        >>> # Batch migration with parallel processing
        >>> users = manager.migrate_batch(
        ...     legacy_users, "User", "1.0.0", "2.0.0",
        ...     parallel=True, max_workers=4
        ... )
        >>>
        >>> # Stream large datasets efficiently
        >>> for user in manager.migrate_batch_streaming(
        ...     large_dataset, "User", "1.0.0", "2.0.0"
        ...  ):
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
        ...         (
        ...              {"name": "Alice"},
        ...              {"name": "Alice", "email": "unknown@example.com"}
        ...         )
        ...     ]
        ... )
        >>> results.assert_all_passed()
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
        return self._registry.register(name, version, enable_ref, backward_compatible)

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
            >>> old_data = {"name": "Alice"}
            >>> user: UserV2 = manager.migrate_as(
            ...     old_data, "User", "1.0.0", "2.0.0", UserV2
            ... )
            >>> # Type checker knows user is UserV2, not just BaseModel
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
            >>> old_users = [{"name": "Alice"}, {"name": "Bob"}]
            >>> users: list[UserV2] = manager.migrate_batch_as(
            ...     old_users, "User", "1.0.0", "2.0.0", UserV2,
            ...     parallel=True, max_workers=4
            ... )
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
            >>> for user in manager.migrate_batch_streaming_as(
            ...     large_dataset, "User", "1.0.0", "2.0.0", UserV2
            ... ):
            ...     save_to_database(user)  # user is typed as UserV2
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

        Example (Callable):
            >>> def my_generator(model: type[BaseModel]) -> JsonSchema:
            ...     schema = model.model_json_schema()
            ...     schema["x-custom"] = True
            ...     return schema
            >>>
            >>> manager = ModelManager()
            >>> manager.set_default_schema_generator(my_generator)

        Example (Class - Recommended):
            >>> from pydantic.json_schema import GenerateJsonSchema
            >>>
            >>> class MyGenerator(GenerateJsonSchema):
            ...     def generate(
            ...         self,
            ...         schema: Mapping[str, Any],
            ...         mode: JsonSchemaMode = "validation"
            ...     ) -> JsonSchema:
            ...         json_schema = super().generate(schema, mode=mode)
            ...         json_schema["x-custom"] = True
            ...         json_schema["$schema"] = self.schema_dialect
            ...         return json_schema
            >>>
            >>> manager = ModelManager()
            >>> manager.set_default_schema_generator(MyGenerator)
            >>>
            >>> # All subsequent schema calls will use MyGenerator
            >>> schema = manager.get_schema("User", "1.0.0")
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
            >>> @manager.schema_transformer("User", "1.0.0")
            ... def add_auth_metadata(schema: JsonSchema) -> JsonSchema:
            ...     schema["x-requires-auth"] = True
            ...     schema["x-auth-level"] = 'admin'
            ...     return schema
            >>>
            >>> @manager.schema_transformer("Product", "2.0.0")
            ... def add_product_examples(schema: JsonSchema) -> JsonSchema:
            ...     schema["examples"] = [
            ...         {"name": "Widget", "price": 9.99},
            ...         {"name": "Gadget", "price": 19.99}
            ...     ]
            ...     return schema
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
            >>> transformers = manager.get_schema_transformers("User", "1.0.0")
            >>> print(f"Found {len(transformers)} transformers")
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
            >>> # Clear all transformers
            >>> manager.clear_schema_transformers()
            >>>
            >>> # Clear User transformers
            >>> manager.clear_schema_transformers("User")
            >>>
            >>> # Clear specific version
            >>> manager.clear_schema_transformers("User", "1.0.0")
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
            >>> # Use default config
            >>> schema = manager.get_schema("User", "1.0.0")
            >>>
            >>> # Override with custom config
            >>> config = SchemaConfig(mode="serialization")
            >>> schema = manager.get_schema("User", "1.0.0", config=config)
            >>>
            >>> # Quick override with kwargs
            >>> schema = manager.get_schema("User", "1.0.0", mode="serialization")
        """
        return self._schema_manager.get_schema(name, version, config=config, **kwargs)

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
            >>> # Export with custom generator
            >>> config = SchemaConfig(
            ...     schema_generator=CustomGenerator,
            ...     mode="validation"
            ... )
            >>> manager.dump_schemas("schemas/", config=config)
            >>>
            >>> # Export validation and serialization schemas separately
            >>> manager.dump_schemas(
            ...     "schemas/validation/",
            ...     config=SchemaConfig(mode="validation")
            ... )
            >>> manager.dump_schemas(
            ...     "schemas/serialization/",
            ...     config=SchemaConfig(mode="serialization")
            ... )
        """
        self._schema_manager.dump_schemas(
            output_dir, indent, separate_definitions, ref_template, config=config
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
