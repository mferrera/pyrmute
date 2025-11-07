"""Tests SchemaManager."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema

from pyrmute import (
    JsonSchema,
    JsonSchemaMode,
    ModelNotFoundError,
    ModelVersion,
    NestedModelInfo,
    Registry,
    SchemaManager,
)
from pyrmute.schema_config import SchemaConfig

if TYPE_CHECKING:
    from pyrmute.types import JsonSchemaDefinitions


# Initialization tests
def test_manager_initialization(registry: Registry) -> None:
    """Test SchemaManager initializes with registry."""
    manager = SchemaManager(registry)
    assert manager.registry is registry


# Get schema tests
def test_get_schema_with_string_version(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test getting schema with string version."""
    schema = populated_schema_manager.get_schema("User", "1.0.0")
    assert isinstance(schema, dict)
    assert "properties" in schema or "type" in schema


def test_get_schema_with_model_version(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test getting schema with ModelVersion object."""
    schema = populated_schema_manager.get_schema("User", ModelVersion(1, 0, 0))
    assert isinstance(schema, dict)


def test_get_schema_contains_fields(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test schema contains model fields."""
    schema = populated_schema_manager.get_schema("User", "1.0.0")
    assert "properties" in schema
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert "name" in properties


def test_get_schema_with_kwargs(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test getting schema with additional kwargs."""
    schema = populated_schema_manager.get_schema("User", "1.0.0", by_alias=True)
    assert isinstance(schema, dict)
    assert "properties" in schema


def test_get_schema_multiple_versions(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test getting schemas for different versions."""
    schema_v1 = populated_schema_manager.get_schema("User", "1.0.0")
    schema_v2 = populated_schema_manager.get_schema("User", "2.0.0")

    properties_v1 = schema_v1["properties"]
    properties_v2 = schema_v2["properties"]
    assert isinstance(properties_v1, dict)
    assert isinstance(properties_v2, dict)
    assert "name" in properties_v1
    assert "name" in properties_v2
    assert "email" in properties_v2
    assert "email" not in properties_v1


# Get schema with separate definitions tests
def test_get_schema_with_separate_defs_basic(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test getting schema with separate definitions."""
    schema = populated_schema_manager.get_schema_with_separate_defs("User", "1.0.0")
    assert isinstance(schema, dict)


def test_get_schema_with_separate_defs_custom_template(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test getting schema with custom ref template."""
    schema = populated_schema_manager.get_schema_with_separate_defs(
        "User",
        "1.0.0",
        ref_template="https://example.com/schemas/{model}_v{version}.json",
    )
    assert isinstance(schema, dict)


def test_get_schema_with_separate_defs_nested_models(
    registry: Registry,
) -> None:
    """Test separate defs with nested models."""

    class AddressV1(BaseModel):
        street: str

    class PersonV1(BaseModel):
        name: str
        address: AddressV1

    registry.register("Address", "1.0.0", enable_ref=True)(AddressV1)
    registry.register("Person", "1.0.0")(PersonV1)

    manager = SchemaManager(registry)
    schema = manager.get_schema_with_separate_defs(
        "Person", "1.0.0", ref_template="{model}_v{version}.json"
    )

    assert isinstance(schema, dict)


def test_get_schema_with_separate_defs_model_version(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test separate defs with ModelVersion object."""
    schema = populated_schema_manager.get_schema_with_separate_defs(
        "User",
        ModelVersion(1, 0, 0),
    )
    assert isinstance(schema, dict)


def test_get_schema_with_separate_defs_kwargs(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test separate defs with additional kwargs."""
    schema = populated_schema_manager.get_schema_with_separate_defs(
        "User", "1.0.0", by_alias=True
    )
    assert isinstance(schema, dict)
    assert "properties" in schema


def test_get_schema_with_separate_defs_mixed_ref_settings(
    registry: Registry,
) -> None:
    """Test separate defs preserves inline definitions for models without enable_ref."""

    class AddressV1(BaseModel):
        street: str

    class ContactV1(BaseModel):
        email: str

    class PersonV1(BaseModel):
        name: str
        address: AddressV1  # This will stay inline (enable_ref=False)
        contact: ContactV1  # This will be external (enable_ref=True)

    registry.register("Address", "1.0.0", enable_ref=False)(AddressV1)
    registry.register("Contact", "1.0.0", enable_ref=True)(ContactV1)
    registry.register("Person", "1.0.0")(PersonV1)

    manager = SchemaManager(registry)
    schema = manager.get_schema_with_separate_defs(
        "Person", "1.0.0", ref_template="{model}_v{version}.json"
    )

    # Should have remaining inline definitions for Address
    assert "$defs" in schema or "definitions" in schema
    defs_key = "$defs" if "$defs" in schema else "definitions"

    defs = schema[defs_key]
    assert isinstance(defs, dict)

    # Address should remain in definitions (not external ref)
    assert "AddressV1" in defs

    # Contact should be external ref, not in definitions
    assert "ContactV1" not in defs


# Replace refs with external tests
def test_replace_refs_with_external_no_refs(
    schema_manager: SchemaManager,
) -> None:
    """Test replacing refs when schema has no refs."""
    schema: JsonSchema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    result = schema_manager._replace_refs_with_external(schema, {}, "{model}.json")
    assert result == schema


def test_replace_refs_with_external_internal_ref(
    registry: Registry,
) -> None:
    """Test replacing internal refs with external."""

    class AddressV1(BaseModel):
        street: str

    registry.register("Address", "1.0.0", enable_ref=True)(AddressV1)
    manager = SchemaManager(registry)

    schema: JsonSchema = {"properties": {"address": {"$ref": "#/$defs/AddressV1"}}}
    definitions: JsonSchemaDefinitions = {"AddressV1": {"type": "object"}}

    result = manager._replace_refs_with_external(
        schema, definitions, "{model}_v{version}.json"
    )

    properties = result["properties"]
    assert isinstance(properties, dict)
    address = properties["address"]
    assert isinstance(address, dict)
    assert "$ref" in address
    assert address["$ref"] == "Address_v1_0_0.json"


def test_replace_refs_with_external_disabled_ref(
    registry: Registry,
) -> None:
    """Test that refs not enabled stay internal."""

    class AddressV1(BaseModel):
        street: str

    registry.register("Address", "1.0.0", enable_ref=False)(AddressV1)
    manager = SchemaManager(registry)

    schema: JsonSchema = {"properties": {"address": {"$ref": "#/$defs/AddressV1"}}}
    definitions: JsonSchemaDefinitions = {"AddressV1": {"type": "object"}}

    result = manager._replace_refs_with_external(
        schema, definitions, "{model}_v{version}.json"
    )

    properties = result["properties"]
    assert isinstance(properties, dict)
    address = properties["address"]
    assert isinstance(address, dict)
    assert "$ref" in address
    assert address["$ref"] == "#/$defs/AddressV1"


def test_replace_refs_with_external_nested_dict(
    registry: Registry,
) -> None:
    """Test replacing refs in nested dictionaries."""

    class AddressV1(BaseModel):
        street: str

    registry.register("Address", "1.0.0", enable_ref=True)(AddressV1)
    manager = SchemaManager(registry)

    schema: JsonSchema = {
        "properties": {
            "data": {"properties": {"address": {"$ref": "#/$defs/AddressV1"}}}
        }
    }
    definitions: JsonSchemaDefinitions = {"AddressV1": {"type": "object"}}

    result = manager._replace_refs_with_external(
        schema, definitions, "{model}_v{version}.json"
    )

    properties = result["properties"]
    assert isinstance(properties, dict)
    assert isinstance(properties["data"], dict)
    assert isinstance(properties["data"]["properties"], dict)
    address = properties["data"]["properties"]["address"]
    assert isinstance(address, dict)
    assert "$ref" in address
    assert address["$ref"] == "Address_v1_0_0.json"


def test_replace_refs_with_external_in_list(
    registry: Registry,
) -> None:
    """Test replacing refs in lists."""

    class AddressV1(BaseModel):
        street: str

    registry.register("Address", "1.0.0", enable_ref=True)(AddressV1)
    manager = SchemaManager(registry)

    schema: JsonSchema = {
        "properties": {"addresses": {"items": {"$ref": "#/$defs/AddressV1"}}}
    }
    definitions: JsonSchemaDefinitions = {"AddressV1": {"type": "object"}}

    result = manager._replace_refs_with_external(
        schema, definitions, "{model}_v{version}.json"
    )

    properties = result["properties"]
    assert isinstance(properties, dict)
    addresses = properties["addresses"]
    assert isinstance(addresses, dict)
    items = addresses["items"]
    assert isinstance(items, dict)
    assert "$ref" in items
    assert items["$ref"] == "Address_v1_0_0.json"


# Get remaining defs tests
def test_get_remaining_defs_none_used(
    schema_manager: SchemaManager,
) -> None:
    """Test getting remaining defs when none are used."""
    schema: JsonSchema = {"type": "object"}
    original_defs: JsonSchemaDefinitions = {"Unused": {"type": "string"}}

    remaining = schema_manager._get_remaining_defs(schema, original_defs)
    assert remaining == {}


def test_get_remaining_defs_all_used(
    schema_manager: SchemaManager,
) -> None:
    """Test getting remaining defs when all are used."""
    schema: JsonSchema = {
        "properties": {
            "field1": {"$ref": "#/$defs/Type1"},
            "field2": {"$ref": "#/$defs/Type2"},
        }
    }
    original_defs: JsonSchemaDefinitions = {
        "Type1": {"type": "string"},
        "Type2": {"type": "number"},
    }

    remaining = schema_manager._get_remaining_defs(schema, original_defs)
    assert remaining == original_defs


def test_get_remaining_defs_partial(
    schema_manager: SchemaManager,
) -> None:
    """Test getting remaining defs when some are used."""
    schema: JsonSchema = {"properties": {"field1": {"$ref": "#/$defs/Type1"}}}
    original_defs: JsonSchemaDefinitions = {
        "Type1": {"type": "string"},
        "Type2": {"type": "number"},
    }

    remaining = schema_manager._get_remaining_defs(schema, original_defs)
    assert remaining == {"Type1": {"type": "string"}}


def test_get_remaining_defs_nested_refs(
    schema_manager: SchemaManager,
) -> None:
    """Test getting remaining defs with nested refs."""
    schema: JsonSchema = {
        "properties": {"data": {"properties": {"field": {"$ref": "#/$defs/Type1"}}}}
    }
    original_defs: JsonSchemaDefinitions = {"Type1": {"type": "string"}}

    remaining = schema_manager._get_remaining_defs(schema, original_defs)
    assert remaining == original_defs


def test_get_remaining_defs_definitions_key(
    schema_manager: SchemaManager,
) -> None:
    """Test remaining defs with 'definitions' key instead of '$defs'."""
    schema: JsonSchema = {"properties": {"field": {"$ref": "#/definitions/Type1"}}}
    original_defs: JsonSchemaDefinitions = {"Type1": {"type": "string"}}

    remaining = schema_manager._get_remaining_defs(schema, original_defs)
    assert remaining == original_defs


# Find model for definition tests
def test_find_model_for_definition_exists(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test finding model for existing definition."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    result = manager._find_model_for_definition("UserV1")
    assert result == ("User", ModelVersion(1, 0, 0))


def test_find_model_for_definition_not_exists(
    schema_manager: SchemaManager,
) -> None:
    """Test finding model for non-existent definition."""
    result = schema_manager._find_model_for_definition("NonExistent")
    assert result is None


def test_find_model_for_definition_multiple_versions(
    registry: Registry,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
) -> None:
    """Test finding model with multiple versions registered."""
    registry.register("User", "1.0.0")(user_v1)
    registry.register("User", "2.0.0")(user_v2)
    manager = SchemaManager(registry)

    result_v1 = manager._find_model_for_definition("UserV1")
    result_v2 = manager._find_model_for_definition("UserV2")

    assert result_v1 == ("User", ModelVersion(1, 0, 0))
    assert result_v2 == ("User", ModelVersion(2, 0, 0))


# Get all schemas tests
def test_get_all_schemas_single_version(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test getting all schemas for model with single version."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    schemas = manager.get_all_schemas("User")
    assert len(schemas) == 1
    assert ModelVersion(1, 0, 0) in schemas


def test_get_all_schemas_multiple_versions(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test getting all schemas for model with multiple versions."""
    schemas = populated_schema_manager.get_all_schemas("User")
    assert len(schemas) == 2  # noqa: PLR2004
    assert ModelVersion(1, 0, 0) in schemas
    assert ModelVersion(2, 0, 0) in schemas


def test_get_all_schemas_not_found(
    schema_manager: SchemaManager,
) -> None:
    """Test getting all schemas for non-existent model."""
    with pytest.raises(ModelNotFoundError, match="Model 'NonExistent' not found"):
        schema_manager.get_all_schemas("NonExistent")


def test_get_all_schemas_returns_valid_schemas(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test that all schemas returned are valid."""
    schemas = populated_schema_manager.get_all_schemas("User")
    for schema in schemas.values():
        assert isinstance(schema, dict)
        assert "properties" in schema or "type" in schema


# Dump schemas tests
def test_dump_schemas_creates_directory(
    populated_schema_manager: SchemaManager,
    tmp_path: Path,
) -> None:
    """Test dump_schemas creates output directory."""
    output_dir = tmp_path / "schemas"
    populated_schema_manager.dump_schemas(output_dir)

    assert output_dir.exists()
    assert output_dir.is_dir()


def test_dump_schemas_creates_files(
    populated_schema_manager: SchemaManager,
    tmp_path: Path,
) -> None:
    """Test dump_schemas creates JSON files."""
    populated_schema_manager.dump_schemas(tmp_path)

    assert (tmp_path / "User_v1_0_0.json").exists()
    assert (tmp_path / "User_v2_0_0.json").exists()


def test_dump_schemas_valid_json(
    populated_schema_manager: SchemaManager,
    tmp_path: Path,
) -> None:
    """Test dump_schemas creates valid JSON."""
    populated_schema_manager.dump_schemas(tmp_path)

    with open(tmp_path / "User_v1_0_0.json") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_dump_schemas_with_indent(
    populated_schema_manager: SchemaManager,
    tmp_path: Path,
) -> None:
    """Test dump_schemas respects indent parameter."""
    populated_schema_manager.dump_schemas(tmp_path, indent=4)

    content = (tmp_path / "User_v1_0_0.json").read_text()
    assert "    " in content  # 4 spaces indentation


def test_dump_schemas_with_string_path(
    populated_schema_manager: SchemaManager,
    tmp_path: Path,
) -> None:
    """Test dump_schemas accepts string path."""
    populated_schema_manager.dump_schemas(str(tmp_path))

    assert (tmp_path / "User_v1_0_0.json").exists()


def test_dump_schemas_separate_definitions_false(
    populated_schema_manager: SchemaManager,
    tmp_path: Path,
) -> None:
    """Test dump_schemas with separate_definitions=False (default)."""
    populated_schema_manager.dump_schemas(tmp_path, separate_definitions=False)

    with open(tmp_path / "User_v1_0_0.json") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_dump_schemas_separate_definitions_true(
    registry: Registry,
    tmp_path: Path,
) -> None:
    """Test dump_schemas with separate_definitions=True."""

    class AddressV1(BaseModel):
        street: str

    class PersonV1(BaseModel):
        name: str
        address: AddressV1

    registry.register("Address", "1.0.0", enable_ref=True)(AddressV1)
    registry.register("Person", "1.0.0")(PersonV1)

    manager = SchemaManager(registry)
    manager.dump_schemas(tmp_path, separate_definitions=True)

    assert (tmp_path / "Address_v1_0_0.json").exists()
    assert (tmp_path / "Person_v1_0_0.json").exists()


def test_dump_schemas_with_ref_template(
    registry: Registry,
    tmp_path: Path,
) -> None:
    """Test dump_schemas with custom ref_template."""

    class AddressV1(BaseModel):
        street: str

    registry.register("Address", "1.0.0", enable_ref=True)(AddressV1)

    manager = SchemaManager(registry)
    manager.dump_schemas(
        tmp_path,
        separate_definitions=True,
        ref_template="https://example.com/schemas/{model}_v{version}.json",
    )

    assert (tmp_path / "Address_v1_0_0.json").exists()


def test_dump_schemas_default_ref_template(
    registry: Registry,
    tmp_path: Path,
) -> None:
    """Test dump_schemas uses default ref_template when not provided."""

    class AddressV1(BaseModel):
        street: str

    registry.register("Address", "1.0.0", enable_ref=True)(AddressV1)

    manager = SchemaManager(registry)
    manager.dump_schemas(tmp_path, separate_definitions=True)

    assert (tmp_path / "Address_v1_0_0.json").exists()


def test_dump_schemas_multiple_models(
    registry: Registry,
    tmp_path: Path,
) -> None:
    """Test dump_schemas with multiple different models."""

    class UserV1(BaseModel):
        name: str

    class ProductV1(BaseModel):
        title: str

    registry.register("User", "1.0.0")(UserV1)
    registry.register("Product", "1.0.0")(ProductV1)

    manager = SchemaManager(registry)
    manager.dump_schemas(tmp_path)

    assert (tmp_path / "User_v1_0_0.json").exists()
    assert (tmp_path / "Product_v1_0_0.json").exists()


# Get nested models tests
def test_get_nested_models_no_nesting(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test getting nested models when there are none."""
    nested = populated_schema_manager.get_nested_models("User", "1.0.0")
    assert nested == []


def test_get_nested_root_models_nesting(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test getting nested models in root models."""
    nested = populated_schema_manager.get_nested_models("UserList", "1.0.0")
    assert len(nested) == 1
    assert nested[0].name == "User"
    assert str(nested[0].version) == "1.0.0"


def test_get_nested_models_with_nesting(
    registry: Registry,
) -> None:
    """Test getting nested models with nested BaseModel."""

    class AddressV1(BaseModel):
        street: str

    class PersonV1(BaseModel):
        name: str
        address: AddressV1

    registry.register("Address", "1.0.0")(AddressV1)
    registry.register("Person", "1.0.0")(PersonV1)

    manager = SchemaManager(registry)
    nested = manager.get_nested_models("Person", "1.0.0")

    assert len(nested) == 1
    assert nested[0] == NestedModelInfo(name="Address", version=ModelVersion(1, 0, 0))


def test_get_nested_models_multiple(
    registry: Registry,
) -> None:
    """Test getting multiple nested models."""

    class AddressV1(BaseModel):
        street: str

    class ContactV1(BaseModel):
        email: str

    class PersonV1(BaseModel):
        name: str
        address: AddressV1
        contact: ContactV1

    registry.register("Address", "1.0.0")(AddressV1)
    registry.register("Contact", "1.0.0")(ContactV1)
    registry.register("Person", "1.0.0")(PersonV1)

    manager = SchemaManager(registry)
    nested = manager.get_nested_models("Person", "1.0.0")

    assert len(nested) == 2  # noqa: PLR2004
    assert NestedModelInfo(name="Address", version=ModelVersion(1, 0, 0)) in nested
    assert NestedModelInfo(name="Contact", version=ModelVersion(1, 0, 0)) in nested


def test_get_nested_models_with_model_version(
    registry: Registry,
) -> None:
    """Test getting nested models with ModelVersion object."""

    class AddressV1(BaseModel):
        street: str

    class PersonV1(BaseModel):
        name: str
        address: AddressV1

    registry.register("Address", "1.0.0")(AddressV1)
    registry.register("Person", "1.0.0")(PersonV1)

    manager = SchemaManager(registry)
    nested = manager.get_nested_models("Person", ModelVersion(1, 0, 0))

    assert len(nested) == 1


def test_get_nested_models_no_duplicates(
    registry: Registry,
) -> None:
    """Test that nested models are not duplicated."""

    class AddressV1(BaseModel):
        street: str

    class PersonV1(BaseModel):
        name: str
        home_address: AddressV1
        work_address: AddressV1

    registry.register("Address", "1.0.0")(AddressV1)
    registry.register("Person", "1.0.0")(PersonV1)

    manager = SchemaManager(registry)
    nested = manager.get_nested_models("Person", "1.0.0")

    assert len(nested) == 1
    assert nested[0] == NestedModelInfo(name="Address", version=ModelVersion(1, 0, 0))


def test_get_nested_models_unregistered_ignored(
    registry: Registry,
) -> None:
    """Test that unregistered nested models are ignored."""

    class UnregisteredAddress(BaseModel):
        street: str

    class PersonV1(BaseModel):
        name: str
        address: UnregisteredAddress

    registry.register("Person", "1.0.0")(PersonV1)

    manager = SchemaManager(registry)
    nested = manager.get_nested_models("Person", "1.0.0")

    assert nested == []


# Get model type from field tests
def test_get_model_type_from_field_direct(
    schema_manager: SchemaManager,
) -> None:
    """Test extracting direct model type from field."""

    class TestModel(BaseModel):
        field: str

    field_info = FieldInfo(annotation=TestModel)
    model_type = schema_manager._get_model_type_from_field(field_info)
    assert model_type is TestModel


def test_get_model_type_from_field_optional(
    schema_manager: SchemaManager,
) -> None:
    """Test extracting model type from Optional field."""

    class TestModel(BaseModel):
        field: str

    field_info = FieldInfo(annotation=TestModel | None)  # type: ignore
    model_type = schema_manager._get_model_type_from_field(field_info)
    assert model_type is TestModel


def test_get_model_type_from_field_list(
    schema_manager: SchemaManager,
) -> None:
    """Test extracting model type from List field."""

    class TestModel(BaseModel):
        field: str

    field_info = FieldInfo(annotation=list[TestModel])
    model_type = schema_manager._get_model_type_from_field(field_info)
    assert model_type is TestModel


def test_get_model_type_from_field_none_annotation(
    schema_manager: SchemaManager,
) -> None:
    """Test extracting model type from field with None annotation."""
    field_info = FieldInfo(annotation=None)
    model_type = schema_manager._get_model_type_from_field(field_info)
    assert model_type is None


def test_get_model_type_from_field_primitive(
    schema_manager: SchemaManager,
) -> None:
    """Test extracting model type from primitive field returns None."""
    field_info = FieldInfo(annotation=str)
    model_type = schema_manager._get_model_type_from_field(field_info)
    assert model_type is None


def test_get_model_type_from_field_dict(
    schema_manager: SchemaManager,
) -> None:
    """Test extracting model type from dict field returns None."""
    field_info = FieldInfo(annotation=dict[str, Any])
    model_type = schema_manager._get_model_type_from_field(field_info)
    assert model_type is None


# Custom generators for testing
class CustomGenerator(GenerateJsonSchema):
    """Custom generator that adds metadata."""

    def generate(
        self, schema: Mapping[str, Any], mode: JsonSchemaMode = "validation"
    ) -> JsonSchema:
        """Generate."""
        json_schema = super().generate(schema, mode=mode)
        json_schema["x-custom-generator"] = True
        json_schema["x-mode"] = mode
        return json_schema


class AnotherGenerator(GenerateJsonSchema):
    """Another generator for testing overrides."""

    def generate(
        self, schema: Mapping[str, Any], mode: JsonSchemaMode = "validation"
    ) -> JsonSchema:
        """Generate."""
        json_schema = super().generate(schema, mode=mode)
        json_schema["x-another-generator"] = True
        return json_schema


def callable_generator(model: type[BaseModel]) -> JsonSchema:
    """Callable generator for testing."""
    schema = model.model_json_schema()
    schema["x-callable"] = True
    return schema


# Default config initialization tests
def test_schema_manager_default_config_initialization(
    registry: Registry,
) -> None:
    """Test SchemaManager initializes with default config."""
    config = SchemaConfig(mode="serialization")
    manager = SchemaManager(registry, default_config=config)

    assert manager.default_config == config


def test_schema_manager_no_default_config(
    registry: Registry,
) -> None:
    """Test SchemaManager creates default config when none provided."""
    manager = SchemaManager(registry)

    assert manager.default_config is not None
    assert manager.default_config.mode == "validation"


# set_default_schema_generator tests
def test_set_default_schema_generator_with_class(
    schema_manager: SchemaManager,
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test setting default generator with GenerateJsonSchema class."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)
    manager.set_default_schema_generator(CustomGenerator)

    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-custom-generator"] is True


def test_set_default_schema_generator_with_callable(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test setting default generator with callable."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)
    manager.set_default_schema_generator(callable_generator)

    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-callable"] is True


def test_set_default_schema_generator_applies_to_all_schemas(
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
    registry: Registry,
) -> None:
    """Test default generator applies to all schemas."""
    registry.register("User", "1.0.0")(user_v1)
    registry.register("User", "2.0.0")(user_v2)
    manager = SchemaManager(registry)
    manager.set_default_schema_generator(CustomGenerator)

    schema_v1 = manager.get_schema("User", "1.0.0")
    schema_v2 = manager.get_schema("User", "2.0.0")

    assert schema_v1["x-custom-generator"] is True
    assert schema_v2["x-custom-generator"] is True


# get_schema with config tests
def test_get_schema_with_config_overrides_default(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test config parameter overrides default config."""
    registry.register("User", "1.0.0")(user_v1)
    default_config = SchemaConfig(schema_generator=CustomGenerator)
    manager = SchemaManager(registry, default_config=default_config)

    override_config = SchemaConfig(schema_generator=AnotherGenerator)
    schema = manager.get_schema("User", "1.0.0", config=override_config)

    assert "x-another-generator" in schema
    assert "x-custom-generator" not in schema


def test_get_schema_with_mode_override(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test mode can be overridden via config."""
    registry.register("User", "1.0.0")(user_v1)
    default_config = SchemaConfig(schema_generator=CustomGenerator, mode="validation")
    manager = SchemaManager(registry, default_config=default_config)

    override_config = SchemaConfig(mode="serialization")
    schema = manager.get_schema("User", "1.0.0", config=override_config)

    assert schema["x-mode"] == "serialization"


def test_get_schema_kwargs_override_config(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test kwargs override config parameters."""
    registry.register("User", "1.0.0")(user_v1)
    config = SchemaConfig(by_alias=True)
    manager = SchemaManager(registry, default_config=config)

    # by_alias kwarg should override config
    schema = manager.get_schema("User", "1.0.0", by_alias=False)

    assert isinstance(schema, dict)


def test_get_schema_with_callable_generator_in_config(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test using callable generator via config."""
    registry.register("User", "1.0.0")(user_v1)
    config = SchemaConfig(schema_generator=callable_generator)
    manager = SchemaManager(registry)

    schema = manager.get_schema("User", "1.0.0", config=config)

    assert schema["x-callable"] is True


# Schema transformer tests
def test_register_transformer(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test registering a schema transformer."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    def add_metadata(schema: JsonSchema) -> JsonSchema:
        schema["x-transformed"] = True
        return schema

    manager.register_transformer("User", "1.0.0", add_metadata)
    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-transformed"] is True


def test_register_multiple_transformers(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test registering multiple transformers for same model."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    def transformer1(schema: JsonSchema) -> JsonSchema:
        schema["x-transform-1"] = True
        return schema

    def transformer2(schema: JsonSchema) -> JsonSchema:
        schema["x-transform-2"] = True
        return schema

    manager.register_transformer("User", "1.0.0", transformer1)
    manager.register_transformer("User", "1.0.0", transformer2)
    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-transform-1"] is True
    assert schema["x-transform-2"] is True


def test_transformers_applied_in_order(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test transformers are applied in registration order."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    def transformer1(schema: JsonSchema) -> JsonSchema:
        schema["x-counter"] = 1
        return schema

    def transformer2(schema: JsonSchema) -> JsonSchema:
        schema["x-counter"] = schema["x-counter"] + 1  # type: ignore[operator]
        return schema

    manager.register_transformer("User", "1.0.0", transformer1)
    manager.register_transformer("User", "1.0.0", transformer2)
    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-counter"] == 2  # noqa: PLR2004


def test_transformer_with_model_version(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test transformer with ModelVersion object."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    def add_metadata(schema: JsonSchema) -> JsonSchema:
        schema["x-transformed"] = True
        return schema

    manager.register_transformer("User", ModelVersion(1, 0, 0), add_metadata)
    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-transformed"] is True


def test_transformer_only_affects_specific_version(
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
    registry: Registry,
) -> None:
    """Test transformer only affects the specific version."""
    registry.register("User", "1.0.0")(user_v1)
    registry.register("User", "2.0.0")(user_v2)
    manager = SchemaManager(registry)

    def add_metadata(schema: JsonSchema) -> JsonSchema:
        schema["x-v1-only"] = True
        return schema

    manager.register_transformer("User", "1.0.0", add_metadata)
    schema_v1 = manager.get_schema("User", "1.0.0")
    schema_v2 = manager.get_schema("User", "2.0.0")

    assert schema_v1["x-v1-only"] is True
    assert "x-v1-only" not in schema_v2


def test_transformer_with_generator(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test transformer works with custom generator."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)
    manager.set_default_schema_generator(CustomGenerator)

    def add_metadata(schema: JsonSchema) -> JsonSchema:
        schema["x-transformed"] = True
        return schema

    manager.register_transformer("User", "1.0.0", add_metadata)
    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-custom-generator"] is True
    assert schema["x-transformed"] is True


def test_get_schema_skip_transformers(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test skipping transformers with apply_transformers=False."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    def add_metadata(schema: JsonSchema) -> JsonSchema:
        schema["x-transformed"] = True
        return schema

    manager.register_transformer("User", "1.0.0", add_metadata)
    schema = manager.get_schema("User", "1.0.0", apply_transformers=False)

    assert "x-transformed" not in schema


# get_transformers tests
def test_get_transformers_empty(
    schema_manager: SchemaManager,
) -> None:
    """Test getting transformers when none registered."""
    transformers = schema_manager.get_transformers("User", "1.0.0")

    assert transformers == []


def test_get_transformers_returns_all(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test getting all registered transformers."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    def transformer1(schema: JsonSchema) -> JsonSchema:
        return schema

    def transformer2(schema: JsonSchema) -> JsonSchema:
        return schema

    manager.register_transformer("User", "1.0.0", transformer1)
    manager.register_transformer("User", "1.0.0", transformer2)

    transformers = manager.get_transformers("User", "1.0.0")

    assert len(transformers) == 2  # noqa: PLR2004
    assert transformer1 in transformers
    assert transformer2 in transformers


def test_get_transformers_with_model_version(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test getting transformers with ModelVersion object."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    def transformer(schema: JsonSchema) -> JsonSchema:
        return schema

    manager.register_transformer("User", "1.0.0", transformer)

    transformers = manager.get_transformers("User", ModelVersion(1, 0, 0))

    assert len(transformers) == 1
    assert transformer in transformers


# clear_transformers tests
def test_clear_transformers_all(
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
    registry: Registry,
) -> None:
    """Test clearing all transformers."""
    registry.register("User", "1.0.0")(user_v1)
    registry.register("User", "2.0.0")(user_v2)
    manager = SchemaManager(registry)

    def transformer(schema: JsonSchema) -> JsonSchema:
        schema["x-transformed"] = True
        return schema

    manager.register_transformer("User", "1.0.0", transformer)
    manager.register_transformer("User", "2.0.0", transformer)

    manager.clear_transformers()

    assert manager.get_transformers("User", "1.0.0") == []
    assert manager.get_transformers("User", "2.0.0") == []


def test_clear_transformers_by_model(
    user_v1: type[BaseModel],
    address_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test clearing transformers for specific model."""
    registry.register("User", "1.0.0")(user_v1)
    registry.register("Address", "1.0.0")(address_v1)
    manager = SchemaManager(registry)

    def transformer(schema: JsonSchema) -> JsonSchema:
        return schema

    manager.register_transformer("User", "1.0.0", transformer)
    manager.register_transformer("Address", "1.0.0", transformer)

    manager.clear_transformers("User")

    assert manager.get_transformers("User", "1.0.0") == []
    assert len(manager.get_transformers("Address", "1.0.0")) == 1


def test_clear_transformers_by_version(
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
    registry: Registry,
) -> None:
    """Test clearing transformers for specific version."""
    registry.register("User", "1.0.0")(user_v1)
    registry.register("User", "2.0.0")(user_v2)
    manager = SchemaManager(registry)

    def transformer(schema: JsonSchema) -> JsonSchema:
        return schema

    manager.register_transformer("User", "1.0.0", transformer)
    manager.register_transformer("User", "2.0.0", transformer)

    manager.clear_transformers("User", "1.0.0")

    assert manager.get_transformers("User", "1.0.0") == []
    assert len(manager.get_transformers("User", "2.0.0")) == 1


def test_clear_transformers_with_model_version(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test clearing transformers with ModelVersion object."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    def transformer(schema: JsonSchema) -> JsonSchema:
        return schema

    manager.register_transformer("User", "1.0.0", transformer)

    manager.clear_transformers("User", ModelVersion(1, 0, 0))

    assert manager.get_transformers("User", "1.0.0") == []


# dump_schemas with config tests
def test_dump_schemas_with_config(
    user_v1: type[BaseModel],
    registry: Registry,
    tmp_path: Path,
) -> None:
    """Test dump_schemas uses provided config."""
    registry.register("User", "1.0.0")(user_v1)
    config = SchemaConfig(schema_generator=CustomGenerator)
    manager = SchemaManager(registry)

    manager.dump_schemas(tmp_path, config=config)

    with open(tmp_path / "User_v1_0_0.json") as f:
        data = json.load(f)

    assert data["x-custom-generator"] is True


def test_dump_schemas_with_default_config(
    user_v1: type[BaseModel],
    registry: Registry,
    tmp_path: Path,
) -> None:
    """Test dump_schemas uses default config when none provided."""
    registry.register("User", "1.0.0")(user_v1)
    default_config = SchemaConfig(schema_generator=CustomGenerator)
    manager = SchemaManager(registry, default_config=default_config)

    manager.dump_schemas(tmp_path)

    with open(tmp_path / "User_v1_0_0.json") as f:
        data = json.load(f)

    assert data["x-custom-generator"] is True


def test_dump_schemas_config_overrides_default(
    user_v1: type[BaseModel],
    registry: Registry,
    tmp_path: Path,
) -> None:
    """Test dump_schemas config parameter overrides default."""
    registry.register("User", "1.0.0")(user_v1)
    default_config = SchemaConfig(schema_generator=CustomGenerator)
    manager = SchemaManager(registry, default_config=default_config)

    override_config = SchemaConfig(schema_generator=AnotherGenerator)
    manager.dump_schemas(tmp_path, config=override_config)

    with open(tmp_path / "User_v1_0_0.json") as f:
        data = json.load(f)

    assert "x-another-generator" in data
    assert "x-custom-generator" not in data


def test_dump_schemas_includes_transformers(
    user_v1: type[BaseModel],
    registry: Registry,
    tmp_path: Path,
) -> None:
    """Test dump_schemas includes transformer modifications."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    def add_metadata(schema: JsonSchema) -> JsonSchema:
        schema["x-export-version"] = "1.0"
        return schema

    manager.register_transformer("User", "1.0.0", add_metadata)
    manager.dump_schemas(tmp_path)

    with open(tmp_path / "User_v1_0_0.json") as f:
        data = json.load(f)

    assert data["x-export-version"] == "1.0"


# get_schema_with_separate_defs with config tests
def test_get_schema_with_separate_defs_with_config(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test get_schema_with_separate_defs uses config."""
    registry.register("User", "1.0.0")(user_v1)
    config = SchemaConfig(schema_generator=CustomGenerator)
    manager = SchemaManager(registry)

    schema = manager.get_schema_with_separate_defs("User", "1.0.0", config=config)

    assert schema["x-custom-generator"] is True


# Integration tests
def test_generator_and_transformer_integration(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test generator and transformer work together correctly."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)
    manager.set_default_schema_generator(CustomGenerator)

    def add_examples(schema: JsonSchema) -> JsonSchema:
        schema["examples"] = [{"name": "Alice"}]
        return schema

    def add_version(schema: JsonSchema) -> JsonSchema:
        schema["x-schema-version"] = "1.0"
        return schema

    manager.register_transformer("User", "1.0.0", add_examples)
    manager.register_transformer("User", "1.0.0", add_version)

    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-custom-generator"] is True
    assert schema["examples"] == [{"name": "Alice"}]
    assert schema["x-schema-version"] == "1.0"


def test_complex_configuration_hierarchy(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test complex configuration with default, config, and kwargs."""
    registry.register("User", "1.0.0")(user_v1)

    default_config = SchemaConfig(
        schema_generator=CustomGenerator,
        mode="validation",
        by_alias=True,
    )
    manager = SchemaManager(registry, default_config=default_config)

    # Override mode via config
    call_config = SchemaConfig(mode="serialization")
    # Override by_alias via kwargs
    schema = manager.get_schema("User", "1.0.0", config=call_config, by_alias=False)
    assert schema["x-custom-generator"] is True
    assert schema["x-mode"] == "serialization"


def test_transformer_modifies_nested_properties(
    user_v1: type[BaseModel],
    registry: Registry,
) -> None:
    """Test transformer can modify nested schema properties."""
    registry.register("User", "1.0.0")(user_v1)
    manager = SchemaManager(registry)

    def add_field_metadata(schema: JsonSchema) -> JsonSchema:
        for field_name, field_schema in schema.get("properties", {}).items():  # type: ignore[union-attr]
            field_schema["x-field-name"] = field_name  # type: ignore[index,call-overload]
        return schema

    manager.register_transformer("User", "1.0.0", add_field_metadata)

    schema = manager.get_schema("User", "1.0.0")

    assert schema["properties"]["name"]["x-field-name"] == "name"  # type: ignore


def test_get_all_schemas_with_config(
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
    registry: Registry,
) -> None:
    """Test get_all_schemas applies config to all versions."""
    registry.register("User", "1.0.0")(user_v1)
    registry.register("User", "2.0.0")(user_v2)
    config = SchemaConfig(schema_generator=CustomGenerator)
    manager = SchemaManager(registry)

    schemas = manager.get_all_schemas("User", config=config)

    for schema in schemas.values():
        assert schema["x-custom-generator"] is True
