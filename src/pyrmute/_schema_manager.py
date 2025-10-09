"""Schema manager."""

import json
from pathlib import Path
from typing import Any, Self, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ._registry import Registry
from .exceptions import ModelNotFoundError
from .model_version import ModelVersion
from .types import (
    JsonSchema,
    JsonSchemaDefinitions,
    JsonValue,
    ModelMetadata,
    ModelName,
)


class SchemaManager:
    """Manager for JSON schema generation and export.

    Handles schema generation from Pydantic models with support for custom schema
    generators and sub-schema references.

    Attributes:
        registry: Reference to the Registry.
    """

    def __init__(self: Self, registry: Registry) -> None:
        """Initialize the schema manager.

        Args:
            registry: Registry instance to use.
        """
        self.registry = registry

    def get_schema(
        self: Self,
        name: ModelName,
        version: str | ModelVersion,
        **schema_kwargs: Any,
    ) -> JsonSchema:
        """Get JSON schema for a specific model version.

        Args:
            name: Name of the model.
            version: Semantic version.
            **schema_kwargs: Additional arguments for schema generation.

        Returns:
            JSON schema dictionary.
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version
        model = self.registry.get_model(name, ver)

        if (
            name in self.registry._schema_generators
            and ver in self.registry._schema_generators[name]
        ):
            generator = self.registry._schema_generators[name][ver]
            return generator(model)

        return model.model_json_schema(**schema_kwargs)

    def get_schema_with_separate_defs(
        self: Self,
        name: ModelName,
        version: str | ModelVersion,
        ref_template: str = "{model}_v{version}.json",
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
            **schema_kwargs: Additional arguments for schema generation.

        Returns:
            JSON schema dictionary with external $ref for nested models.

        Example:
            >>> schema = manager.get_schema_with_separate_defs(
            ...     "User", "2.0.0",
            ...     ref_template="https://example.com/schemas/{model}_v{version}.json"
            ... )
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version
        schema = self.get_schema(name, ver, **schema_kwargs)

        # Extract and replace definitions with external references
        if "$defs" in schema or "definitions" in schema:
            defs_key = "$defs" if "$defs" in schema else "definitions"
            definitions: JsonSchemaDefinitions = schema.pop(defs_key, {})  # type: ignore[assignment]

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
                                        model=model_name, version=str(model_version)
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

    def get_all_schemas(self: Self, name: ModelName) -> dict[ModelVersion, JsonSchema]:
        """Get all schemas for a model across all versions.

        Args:
            name: Name of the model.

        Returns:
            Dictionary mapping versions to their schemas.

        Raises:
            ModelNotFoundError: If model not found.
        """
        if name not in self.registry._models:
            raise ModelNotFoundError(name)

        return {
            version: self.get_schema(name, version)
            for version in self.registry._models[name]
        }

    def dump_schemas(
        self: Self,
        output_dir: str | Path,
        indent: int = 2,
        separate_definitions: bool = False,
        ref_template: str | None = None,
    ) -> None:
        """Dump all schemas to JSON files.

        Args:
            output_dir: Directory path for output files.
            indent: JSON indentation level.
            separate_definitions: If True, create separate schema files for nested
                models that have enable_ref=True.
            ref_template: Template for $ref URLs when separate_definitions=True.
                Defaults to relative file references if not provided.

        Example:
            >>> # Inline definitions (default)
            >>> manager.dump_schemas("schemas/")
            >>>
            >>> # Separate sub-schemas with relative refs (when enable_ref=True models)
            >>> manager.dump_schemas("schemas/", separate_definitions=True)
            >>>
            >>> # Separate sub-schemas with absolute URLs
            >>> manager.dump_schemas(
            ...     "schemas/",
            ...     separate_definitions=True,
            ...     ref_template="https://example.com/schemas/{model}_v{version}.json"
            ... )
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not separate_definitions:
            for name in self.registry._models:
                for version, schema in self.get_all_schemas(name).items():
                    file_path = output_path / f"{name}_v{version}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(schema, f, indent=indent)
        else:
            if ref_template is None:
                ref_template = "{model}_v{version}.json"

            for name in self.registry._models:
                for version in self.registry._models[name]:
                    schema = self.get_schema_with_separate_defs(
                        name, version, ref_template
                    )
                    file_path = output_path / f"{name}_v{version}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(schema, f, indent=indent)

    def get_nested_models(
        self: Self,
        name: ModelName,
        version: str | ModelVersion,
    ) -> list[ModelMetadata]:
        """Get all nested models referenced by a model.

        Args:
            name: Name of the model.
            version: Semantic version.

        Returns:
            List of (model_name, model_version) tuples for nested models.
        """
        ver = ModelVersion.parse(version) if isinstance(version, str) else version
        model = self.registry.get_model(name, ver)

        nested: list[ModelMetadata] = []

        for field_info in model.model_fields.values():
            model_type = self._get_model_type_from_field(field_info)
            if model_type:
                model_info = self.registry.get_model_info(model_type)
                if model_info and model_info not in nested:
                    nested.append(model_info)

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
