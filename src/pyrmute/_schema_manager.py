"""Schema manager with customizable generation and transformers."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Self, cast, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema

from ._registry import Registry
from ._type_inspector import TypeInspector
from .exceptions import ModelNotFoundError
from .model_version import ModelVersion
from .schema_config import SchemaConfig
from .types import (
    JsonSchema,
    JsonSchemaDefinitions,
    JsonSchemaGenerator,
    JsonValue,
    ModelMetadata,
    ModelName,
    NestedModelInfo,
    SchemaTransformer,
)


class SchemaManager:
    """Manager for JSON schema generation and export.

    Handles schema generation from Pydantic models with support for custom schema
    generators, global configuration, per-call overrides, and schema transformers.

    Attributes:
        registry: Reference to the Registry.
        default_config: Default schema generation configuration.
    """

    def __init__(
        self: Self, registry: Registry, default_config: SchemaConfig | None = None
    ) -> None:
        """Initialize the schema manager.

        Args:
            registry: Registry instance to use.
            default_config: Default configuration for schema generation.
        """
        self.registry = registry
        self.default_config = default_config or SchemaConfig()
        self._transformers: dict[
            tuple[ModelName, ModelVersion], list[SchemaTransformer]
        ] = defaultdict(list)

    def set_default_schema_generator(
        self: Self, generator: JsonSchemaGenerator | type[GenerateJsonSchema]
    ) -> None:
        """Set the default schema generator for all schemas.

        Args:
            generator: Custom schema generator - either a callable or GenerateJsonSchema
                class.

        Example (Callable):
            ```python
            def custom_gen(model: type[BaseModel]) -> JsonSchema:
                schema = model.model_json_schema()
                schema["x-custom"] = True
                return schema

            manager.set_default_schema_generator(custom_gen)
            ```

        Example (Class):
            ```python
            from pydantic.json_schema import GenerateJsonSchema

            class CustomGenerator(GenerateJsonSchema):
                def generate(
                    self,
                    schema: Mapping[str, Any],
                    mode: JsonSchemaMode = "validation"
                ) -> JsonSchema:
                    json_schema = super().generate(schema, mode=mode)
                    json_schema["x-custom"] = True
                    return json_schema

            manager.set_default_schema_generator(CustomGenerator)
            ```
        """
        self.default_config.schema_generator = generator

    def register_transformer(
        self: Self,
        name: ModelName,
        version: str | ModelVersion,
        transformer: SchemaTransformer,
    ) -> None:
        """Register a schema transformer for a specific model version.

        Transformers are applied after schema generation, allowing simple
        post-processing of schemas without needing to customize the generation process
        itself.

        Args:
            name: Name of the model.
            version: Model version.
            transformer: Function that takes and returns a JsonSchema.

        Example:
            ```python
            def add_examples(schema: JsonSchema) -> JsonSchema:
                schema["examples"] = [{"name": "John", "age": 30}]
                return schema

            manager.register_transformer("User", "1.0.0", add_examples)
            ```
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version
        key = (name, ver)
        self._transformers[key].append(transformer)

    def get_schema(
        self: Self,
        name: ModelName,
        version: str | ModelVersion,
        config: SchemaConfig | None = None,
        apply_transformers: bool = True,
        **schema_kwargs: Any,
    ) -> JsonSchema:
        """Get JSON schema for a specific model version.

        Execution order:
        1. Generate base schema using Pydantic
        2. Apply custom generator (if configured)
        3. Apply registered transformers (if any)

        Args:
            name: Name of the model.
            version: Semantic version.
            config: Optional schema configuration (overrides defaults).
            apply_transformers: If False, skip transformer application.
            **schema_kwargs: Additional arguments for schema generation (overrides
                config).

        Returns:
            JSON schema dictionary.

        Example:
            ```python
            # Use default config
            schema = manager.get_schema("User", "1.0.0")

            # Override with custom config
            config = SchemaConfig(mode="serialization", by_alias=False)
            schema = manager.get_schema("User", "1.0.0", config=config)

            # Quick override with kwargs
            schema = manager.get_schema("User", "1.0.0", mode="serialization")

            # Get base schema without transformers
            base_schema = manager.get_schema(
                "User", "1.0.0",
                apply_transformers=False
            )
            ```
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version
        model = self.registry.get_model(name, ver)

        # Always use the config-based approach
        final_config = self.default_config
        if config is not None:
            final_config = final_config.merge_with(config)

        if schema_kwargs:
            kwargs_config = SchemaConfig(extra_kwargs=schema_kwargs)
            final_config = final_config.merge_with(kwargs_config)

        schema: JsonSchema
        if final_config.is_callable_generator():
            schema = final_config.schema_generator(model)  # type: ignore
        else:
            schema = model.model_json_schema(**final_config.to_kwargs())

        if apply_transformers:
            key = (name, ver)
            if key in self._transformers:
                for transformer in self._transformers[key]:
                    schema = transformer(schema)

        return schema

    def get_transformers(
        self: Self,
        name: ModelName,
        version: str | ModelVersion,
    ) -> list[SchemaTransformer]:
        """Get all transformers registered for a model version.

        Args:
            name: Name of the model.
            version: Model version.

        Returns:
            List of transformer functions.
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version
        key = (name, ver)
        return self._transformers.get(key, [])

    def clear_transformers(
        self: Self,
        name: ModelName | None = None,
        version: str | ModelVersion | None = None,
    ) -> None:
        """Clear registered transformers.

        Args:
            name: Optional model name. If None, clears all transformers.
            version: Optional version. If None (but name provided), clears all versions
                of that model.

        Example:
            ```python
            # Clear all transformers
            manager.clear_transformers()

            # Clear all User transformers
            manager.clear_transformers("User")

            # Clear specific version
            manager.clear_transformers("User", "1.0.0")
            ```
        """
        if name is None:
            self._transformers.clear()
        elif version is None:
            keys_to_remove = [key for key in self._transformers if key[0] == name]
            for key in keys_to_remove:
                del self._transformers[key]
        else:
            ver = ModelVersion.parse(version) if isinstance(version, str) else version
            key = (name, ver)
            if key in self._transformers:
                del self._transformers[key]

    def get_schema_with_separate_defs(
        self: Self,
        name: ModelName,
        version: str | ModelVersion,
        ref_template: str = "{model}_v{version}.json",
        config: SchemaConfig | None = None,
        **schema_kwargs: Any,
    ) -> JsonSchema:
        """Get JSON schema with separate definition files for nested models.

        This creates a schema where nested Pydantic models are referenced as external
        JSON schema files rather than inline definitions.

        Args:
            name: Name of the model.
            version: Semantic version.
            ref_template: Template for generating $ref URLs. Supports {model} and
                {version} placeholders.
            config: Optional schema configuration.
            **schema_kwargs: Additional arguments for schema generation.

        Returns:
            JSON schema dictionary with external $ref for nested models.

        Example:
            ```python
            schema = manager.get_schema_with_separate_defs(
                "User", "2.0.0",
                ref_template="https://example.com/schemas/{model}_v{version}.json",
                mode="serialization"
            )
            ```
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version
        schema = self.get_schema(name, ver, config=config, **schema_kwargs)

        if "$defs" in schema or "definitions" in schema:
            defs_key = "$defs" if "$defs" in schema else "definitions"
            definitions = cast("JsonSchemaDefinitions", schema.pop(defs_key, {}))

            # Update all $ref in the schema to point to external files
            schema = self._replace_refs_with_external(schema, definitions, ref_template)

            # Re-add definitions that weren't converted to external refs
            remaining_defs = self._get_remaining_defs(schema, definitions)
            if remaining_defs:
                schema[defs_key] = remaining_defs

        return schema

    def _replace_refs_with_external(
        self: Self,
        schema: JsonSchema,
        definitions: JsonSchemaDefinitions,
        ref_template: str,
    ) -> JsonSchema:
        """Replace internal $ref with external references.

        Only replaces refs for models that have enable_ref=True.

        Args:
            schema: The schema to process.
            definitions: Dictionary of definitions to replace.
            ref_template: Template for external references.

        Returns:
            Updated schema with external references.
        """

        def process_value(value: JsonValue) -> JsonValue:
            if isinstance(value, dict):
                if "$ref" in value:
                    # Extract the definition name from the ref
                    ref = value["$ref"]
                    if isinstance(ref, str) and ref.startswith(
                        ("#/$defs/", "#/definitions/")
                    ):
                        def_name = ref.split("/")[-1]

                        model_info = self._find_model_for_definition(def_name)
                        if model_info:
                            model_name, model_version = model_info

                            if self.registry.is_ref_enabled(model_name, model_version):
                                # Replace with external reference
                                return {
                                    "$ref": ref_template.format(
                                        model=model_name,
                                        version=str(model_version).replace(".", "_"),
                                    )
                                }
                            # Keep as internal reference (will be inlined)
                            return value

                return {k: process_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [process_value(item) for item in value]
            return value

        return process_value(schema)  # type: ignore[return-value]

    def _get_remaining_defs(
        self: Self,
        schema: JsonSchema,
        original_defs: JsonSchemaDefinitions,
    ) -> JsonSchemaDefinitions:
        """Get definitions that should remain inline.

        Args:
            schema: The processed schema.
            original_defs: Original definitions.

        Returns:
            Dictionary of definitions that weren't converted to external refs.
        """
        internal_refs: set[str] = set()

        def find_internal_refs(value: dict[str, Any] | list[Any]) -> None:
            if isinstance(value, dict):
                if "$ref" in value:
                    ref = value["$ref"]
                    if ref.startswith(("#/$defs/", "#/definitions/")):
                        def_name = ref.split("/")[-1]
                        internal_refs.add(def_name)
                for v in value.values():
                    find_internal_refs(v)
            elif isinstance(value, list):
                for item in value:
                    find_internal_refs(item)

        find_internal_refs(schema)
        return {k: v for k, v in original_defs.items() if k in internal_refs}

    def _find_model_for_definition(self: Self, def_name: str) -> ModelMetadata | None:
        """Find the registered model corresponding to a definition name.

        Args:
            def_name: The definition name from the schema.

        Returns:
            Tuple of (model_name, version) if found, None otherwise.
        """
        for name, versions in self.registry._models.items():
            for version, model_class in versions.items():
                if model_class.__name__ == def_name:
                    return (name, version)
        return None

    def get_all_schemas(
        self: Self, name: ModelName, config: SchemaConfig | None = None
    ) -> dict[ModelVersion, JsonSchema]:
        """Get all schemas for a model across all versions.

        Args:
            name: Name of the model.
            config: Optional schema configuration.

        Returns:
            Dictionary mapping versions to their schemas.

        Raises:
            ModelNotFoundError: If model not found.
        """
        if name not in self.registry._models:
            raise ModelNotFoundError(name)

        return {
            version: self.get_schema(name, version, config=config)
            for version in self.registry._models[name]
        }

    def dump_schemas(
        self: Self,
        output_dir: str | Path,
        indent: int = 2,
        separate_definitions: bool = False,
        ref_template: str | None = None,
        config: SchemaConfig | None = None,
    ) -> None:
        """Dump all schemas to JSON files.

        Args:
            output_dir: Directory path for output files.
            indent: JSON indentation level.
            separate_definitions: If True, create separate schema files for nested
                models that have enable_ref=True.
            ref_template: Template for $ref URLs when separate_definitions=True.
                Defaults to relative file references if not provided.
            config: Optional schema configuration for all exported schemas.

        Example:
            ```python
            # Export with custom schema generator
            config = SchemaConfig(
                schema_generator=CustomGenerator,
                mode="serialization"
            )
            manager.dump_schemas("schemas/", config=config)
            ```
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not separate_definitions:
            for name in self.registry._models:
                for version, schema in self.get_all_schemas(
                    name, config=config
                ).items():
                    version_str = str(version).replace(".", "_")
                    file_path = output_path / f"{name}_v{version_str}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(schema, f, indent=indent)
        else:
            if ref_template is None:
                ref_template = "{model}_v{version}.json"

            for name in self.registry._models:
                for version in self.registry._models[name]:
                    schema = self.get_schema_with_separate_defs(
                        name, version, ref_template, config=config
                    )
                    version_str = str(version).replace(".", "_")
                    file_path = output_path / f"{name}_v{version_str}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(schema, f, indent=indent)

    def get_nested_models(
        self: Self,
        name: ModelName,
        version: str | ModelVersion,
    ) -> list[NestedModelInfo]:
        """Get all nested models referenced by a model.

        Args:
            name: Name of the model.
            version: Semantic version.

        Returns:
            List of NestedModelInfo.
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version
        model = self.registry.get_model(name, ver)

        nested: list[NestedModelInfo] = []

        if TypeInspector.is_root_model(model):
            root_annotation = TypeInspector.get_root_annotation(model)
            nested_from_root = self._extract_nested_from_annotation(root_annotation)
            nested.extend(nested_from_root)
            return nested

        for field_info in model.model_fields.values():
            model_type = self._get_model_type_from_field(field_info)
            if not model_type:
                continue

            model_info = self.registry.get_model_info(model_type)

            if not model_info:
                continue

            name_, version_ = model_info
            nested_model_info = NestedModelInfo(name=name_, version=version_)

            if nested_model_info not in nested:
                nested.append(nested_model_info)

        return nested

    def _extract_nested_from_annotation(
        self: Self, annotation: Any
    ) -> list[NestedModelInfo]:
        """Extract nested models from a type annotation (for RootModel support).

        Args:
            annotation: The type annotation to inspect.

        Returns:
            List of NestedModelInfo found in the annotation.
        """
        nested: list[NestedModelInfo] = []

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            model_info = self.registry.get_model_info(annotation)
            if model_info:
                name, version = model_info
                nested_model_info = NestedModelInfo(name=name, version=version)
                if nested_model_info not in nested:
                    nested.append(nested_model_info)
            return nested

        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            for arg in args:
                if arg is type(None):
                    continue
                nested.extend(self._extract_nested_from_annotation(arg))

        return nested

    def _get_model_type_from_field(
        self: Self, field: FieldInfo
    ) -> type[BaseModel] | None:
        """Extract the Pydantic model type from a field.

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
