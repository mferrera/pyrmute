"""Tests SchemaManager."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from pyrmute import ModelVersion
from pyrmute._registry import Registry
from pyrmute._schema_manager import SchemaManager

if TYPE_CHECKING:
    from pyrmute.types import JsonSchema, JsonSchemaDefinitions


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


def test_get_schema_with_custom_generator(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test getting schema with custom schema generator."""

    def custom_generator(model: type[BaseModel]) -> dict[str, Any]:
        return {"custom": True, "model": model.__name__}

    registry.register("User", "1.0.0", schema_generator=custom_generator)(user_v1)
    manager = SchemaManager(registry)

    schema = manager.get_schema("User", "1.0.0")
    assert schema["custom"] is True
    assert schema["model"] == "UserV1"


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
    assert address["$ref"] == "Address_v1.0.0.json"


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
    assert address["$ref"] == "Address_v1.0.0.json"


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
    assert items["$ref"] == "Address_v1.0.0.json"


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
    with pytest.raises(ValueError, match="Model NonExistent not found"):
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

    assert (tmp_path / "User_v1.0.0.json").exists()
    assert (tmp_path / "User_v2.0.0.json").exists()


def test_dump_schemas_valid_json(
    populated_schema_manager: SchemaManager,
    tmp_path: Path,
) -> None:
    """Test dump_schemas creates valid JSON."""
    populated_schema_manager.dump_schemas(tmp_path)

    with open(tmp_path / "User_v1.0.0.json") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_dump_schemas_with_indent(
    populated_schema_manager: SchemaManager,
    tmp_path: Path,
) -> None:
    """Test dump_schemas respects indent parameter."""
    populated_schema_manager.dump_schemas(tmp_path, indent=4)

    content = (tmp_path / "User_v1.0.0.json").read_text()
    assert "    " in content  # 4 spaces indentation


def test_dump_schemas_with_string_path(
    populated_schema_manager: SchemaManager,
    tmp_path: Path,
) -> None:
    """Test dump_schemas accepts string path."""
    populated_schema_manager.dump_schemas(str(tmp_path))

    assert (tmp_path / "User_v1.0.0.json").exists()


def test_dump_schemas_separate_definitions_false(
    populated_schema_manager: SchemaManager,
    tmp_path: Path,
) -> None:
    """Test dump_schemas with separate_definitions=False (default)."""
    populated_schema_manager.dump_schemas(tmp_path, separate_definitions=False)

    with open(tmp_path / "User_v1.0.0.json") as f:
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

    assert (tmp_path / "Address_v1.0.0.json").exists()
    assert (tmp_path / "Person_v1.0.0.json").exists()


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

    assert (tmp_path / "Address_v1.0.0.json").exists()


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

    assert (tmp_path / "Address_v1.0.0.json").exists()


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

    assert (tmp_path / "User_v1.0.0.json").exists()
    assert (tmp_path / "Product_v1.0.0.json").exists()


# Get nested models tests
def test_get_nested_models_no_nesting(
    populated_schema_manager: SchemaManager,
) -> None:
    """Test getting nested models when there are none."""
    nested = populated_schema_manager.get_nested_models("User", "1.0.0")
    assert nested == []


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
    assert nested[0] == ("Address", ModelVersion(1, 0, 0))


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
    assert ("Address", ModelVersion(1, 0, 0)) in nested
    assert ("Contact", ModelVersion(1, 0, 0)) in nested


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
    assert nested[0] == ("Address", ModelVersion(1, 0, 0))


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
