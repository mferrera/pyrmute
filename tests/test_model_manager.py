"""Tests the ModelManager."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from pyrmute import MigrationData, ModelManager, ModelVersion


# Initialization tests
def test_manager_initialization(manager: ModelManager) -> None:
    """Test ModelManager initializes with required components."""
    assert manager.registry is not None
    assert manager.migration_manager is not None
    assert manager.schema_manager is not None


# Model registration tests
def test_model_registration_with_string_version(manager: ModelManager) -> None:
    """Test registering a model with string version."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    retrieved = manager.get("User", "1.0.0")
    assert retrieved == UserV1


def test_model_registration_with_model_version(manager: ModelManager) -> None:
    """Test registering a model with ModelVersion object."""
    version = ModelVersion(1, 0, 0)

    @manager.model("User", version)
    class UserV1(BaseModel):
        name: str

    retrieved = manager.get("User", version)
    assert retrieved == UserV1


def test_model_registration_with_custom_schema_generator(manager: ModelManager) -> None:
    """Test registering a model with custom schema generator."""

    def custom_generator(model: type[BaseModel]) -> dict[str, Any]:
        return {"custom": True, "type": "object"}

    @manager.model("User", "1.0.0", schema_generator=custom_generator)
    class UserV1(BaseModel):
        name: str

    assert manager.get("User", "1.0.0") == UserV1


def test_model_registration_with_enable_ref(manager: ModelManager) -> None:
    """Test registering a model with enable_ref flag."""

    @manager.model("MasterData", "1.0.0", enable_ref=True)
    class MasterDataV1(BaseModel):
        smda: dict[str, Any]

    assert manager.get("MasterData", "1.0.0") == MasterDataV1


def test_multiple_versions_same_model(
    manager: ModelManager,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
) -> None:
    """Test registering multiple versions of the same model."""
    manager.model("User", "1.0.0")(user_v1)
    manager.model("User", "2.0.0")(user_v2)

    v1 = manager.get("User", "1.0.0")
    v2 = manager.get("User", "2.0.0")

    assert v1 == user_v1
    assert v2 == user_v2


def test_different_models(manager: ModelManager) -> None:
    """Test registering different models."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("Product", "1.0.0")
    class ProductV1(BaseModel):
        title: str

    assert manager.get("User", "1.0.0") == UserV1
    assert manager.get("Product", "1.0.0") == ProductV1


# Migration registration tests
def test_migration_registration_with_string_versions(
    manager: ModelManager,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
) -> None:
    """Test registering a migration with string versions."""
    manager.model("User", "1.0.0")(user_v1)
    manager.model("User", "2.0.0")(user_v2)

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "test@example.com"}

    result = manager.migrate({"name": "John"}, "User", "1.0.0", "2.0.0")
    assert result == user_v2(name="John", email="test@example.com")


def test_migration_registration_with_model_versions(
    manager: ModelManager,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
) -> None:
    """Test registering a migration with ModelVersion objects."""
    v1 = ModelVersion(1, 0, 0)
    v2 = ModelVersion(2, 0, 0)

    manager.model("User", v1)(user_v1)
    manager.model("User", v2)(user_v2)

    @manager.migration("User", v1, v2)
    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "test@example.com"}

    result = manager.migrate({"name": "John"}, "User", v1, v2)
    assert result == user_v2(name="John", email="test@example.com")


def test_migrate_validation_catches_invalid(
    registered_manager: ModelManager,
) -> None:
    """Test that validation catches invalid migrated data."""

    # Register a migration that produces invalid data
    @registered_manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: MigrationData) -> MigrationData:
        return {"name": data["name"]}  # Missing required 'email'

    with pytest.raises(ValidationError):
        registered_manager.migrate({"name": "Invalid"}, "User", "1.0.0", "2.0.0")


# Get model tests
def test_get_model_with_string_version(
    registered_manager: ModelManager, user_v1: type[BaseModel]
) -> None:
    """Test getting a model with string version."""
    model = registered_manager.get("User", "1.0.0")
    assert model == user_v1


def test_get_model_with_model_version(
    registered_manager: ModelManager, user_v1: type[BaseModel]
) -> None:
    """Test getting a model with ModelVersion object."""
    model = registered_manager.get("User", ModelVersion(1, 0, 0))
    assert model == user_v1


def test_get_latest_model(
    registered_manager: ModelManager, user_v2: type[BaseModel]
) -> None:
    """Test getting the latest version when version is None."""
    model = registered_manager.get("User", None)
    assert model == user_v2


def test_get_latest_model_without_version_arg(
    registered_manager: ModelManager, user_v2: type[BaseModel]
) -> None:
    """Test getting latest version by omitting version argument."""
    model = registered_manager.get("User")
    assert model == user_v2


# Migration data tests
def test_migrate_data_returns_dict(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_data returns raw dictionary."""
    result = registered_manager.migrate_data(
        {"name": "Alice"}, "User", "1.0.0", "2.0.0"
    )
    assert isinstance(result, dict)
    assert result == {"name": "Alice", "email": "unknown@example.com"}


def test_migrate_data_with_model_versions(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_data with ModelVersion objects."""
    result = registered_manager.migrate_data(
        {"name": "Bob"},
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
    )
    assert isinstance(result, dict)
    assert result == {"name": "Bob", "email": "unknown@example.com"}


def test_migrate_data_preserves_existing_data(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_data preserves all existing fields."""
    result = registered_manager.migrate_data(
        {"name": "Charlie", "extra": "data"},
        "User",
        "1.0.0",
        "2.0.0",
    )
    assert result["name"] == "Charlie"
    assert result["email"] == "unknown@example.com"


def test_migrate_data_does_not_validate(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_data returns dict even if data would fail validation."""

    @registered_manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: MigrationData) -> MigrationData:
        return {"name": data["name"]}  # Missing required 'email'

    result = registered_manager.migrate_data(
        {"name": "Invalid"}, "User", "1.0.0", "2.0.0"
    )
    assert isinstance(result, dict)
    assert result == {"name": "Invalid"}


def test_migrate_returns_model_instance(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate returns validated BaseModel instance."""
    result = registered_manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")
    assert isinstance(result, BaseModel)
    assert isinstance(result, user_v2)
    assert result == user_v2(name="Alice", email="unknown@example.com")


def test_migrate_and_migrate_data_consistency(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test that migrate and migrate_data produce consistent results."""
    data = {"name": "Consistency"}

    dict_result = registered_manager.migrate_data(data, "User", "1.0.0", "2.0.0")
    model_result = registered_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert model_result.model_dump() == dict_result


# Migration tests
def test_migrate_adds_field(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migration adds new field with default value."""
    result = registered_manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")
    assert result == user_v2(name="Alice", email="unknown@example.com")


def test_migrate_with_model_versions(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migration with ModelVersion objects."""
    result = registered_manager.migrate(
        {"name": "Bob"},
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
    )
    assert result == user_v2(name="Bob", email="unknown@example.com")


def test_migrate_preserves_existing_data(registered_manager: ModelManager) -> None:
    """Test migration preserves all existing fields."""
    result = registered_manager.migrate(
        {"name": "Charlie", "extra": "data"},
        "User",
        "1.0.0",
        "2.0.0",
    )
    assert result.name == "Charlie"  # type: ignore
    assert result.email == "unknown@example.com"  # type: ignore


# Schema tests
def test_get_schema(registered_manager: ModelManager) -> None:
    """Test getting JSON schema for a model."""
    schema = registered_manager.get_schema("User", "1.0.0")
    assert isinstance(schema, dict)
    assert "properties" in schema or "type" in schema


def test_get_schema_with_model_version(registered_manager: ModelManager) -> None:
    """Test getting schema with ModelVersion object."""
    schema = registered_manager.get_schema("User", ModelVersion(1, 0, 0))
    assert isinstance(schema, dict)


def test_get_schema_with_kwargs(registered_manager: ModelManager) -> None:
    """Test getting schema with additional kwargs."""
    schema = registered_manager.get_schema("User", "1.0.0", by_alias=True)
    assert isinstance(schema, dict)
    assert "properties" in schema


# List models tests
def test_list_models_empty(manager: ModelManager) -> None:
    """Test listing models when none are registered."""
    models = manager.list_models()
    assert models == []


def test_list_models_single(manager: ModelManager) -> None:
    """Test listing models with one registered."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    models = manager.list_models()
    assert models == ["User"]


def test_list_models_multiple(manager: ModelManager) -> None:
    """Test listing multiple different models."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("Product", "1.0.0")
    class ProductV1(BaseModel):
        title: str

    models = manager.list_models()
    assert set(models) == {"User", "Product"}


def test_list_models_same_model_multiple_versions(
    manager: ModelManager,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
) -> None:
    """Test listing models counts each model name once."""
    manager.model("User", "1.0.0")(user_v1)
    manager.model("User", "2.0.0")(user_v2)

    models = manager.list_models()
    assert models.count("User") == 1


# List versions tests
def test_list_versions_single(manager: ModelManager, user_v1: type[BaseModel]) -> None:
    """Test listing versions for a model with one version."""
    manager.model("User", "1.0.0")(user_v1)

    versions = manager.list_versions("User")
    assert versions == [ModelVersion(1, 0, 0)]


def test_list_versions_multiple(
    manager: ModelManager,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
) -> None:
    """Test listing versions for a model with multiple versions."""
    manager.model("User", "1.0.0")(user_v1)
    manager.model("User", "2.0.0")(user_v2)

    versions = manager.list_versions("User")
    assert len(versions) == 2  # noqa: PLR2004
    assert ModelVersion(1, 0, 0) in versions
    assert ModelVersion(2, 0, 0) in versions


def test_list_versions_sorted(manager: ModelManager) -> None:
    """Test that versions are returned in sorted order."""

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "1.5.0")
    class UserV15(BaseModel):
        name: str

    versions = manager.list_versions("User")
    assert versions == [
        ModelVersion(1, 0, 0),
        ModelVersion(1, 5, 0),
        ModelVersion(2, 0, 0),
    ]


# Schema dumping tests
def test_dump_schemas_creates_directory(manager: ModelManager, tmp_path: Path) -> None:
    """Test dump_schemas creates output directory."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    output_dir = tmp_path / "schemas"
    manager.dump_schemas(output_dir)

    assert output_dir.exists()
    assert output_dir.is_dir()


def test_dump_schemas_creates_files(manager: ModelManager, tmp_path: Path) -> None:
    """Test dump_schemas creates JSON files."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    manager.dump_schemas(tmp_path)

    schema_file = tmp_path / "User_v1.0.0.json"
    assert schema_file.exists()


def test_dump_schemas_valid_json(manager: ModelManager, tmp_path: Path) -> None:
    """Test dump_schemas creates valid JSON files."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    manager.dump_schemas(tmp_path)

    schema_file = tmp_path / "User_v1.0.0.json"
    with open(schema_file) as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_dump_schemas_with_indent(manager: ModelManager, tmp_path: Path) -> None:
    """Test dump_schemas respects indent parameter."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    manager.dump_schemas(tmp_path, indent=4)

    schema_file = tmp_path / "User_v1.0.0.json"
    content = schema_file.read_text()
    # Check that indentation is used (spaces in JSON)
    assert "    " in content


def test_dump_schemas_with_string_path(manager: ModelManager, tmp_path: Path) -> None:
    """Test dump_schemas accepts string path."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    manager.dump_schemas(str(tmp_path))

    schema_file = tmp_path / "User_v1.0.0.json"
    assert schema_file.exists()


def test_dump_schemas_separate_definitions(
    manager: ModelManager, tmp_path: Path
) -> None:
    """Test dump_schemas with separate_definitions flag."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    manager.dump_schemas(tmp_path, separate_definitions=True)

    schema_file = tmp_path / "User_v1.0.0.json"
    assert schema_file.exists()


def test_dump_schemas_with_ref_template(manager: ModelManager, tmp_path: Path) -> None:
    """Test dump_schemas with custom ref_template."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    manager.dump_schemas(
        tmp_path,
        separate_definitions=True,
        ref_template="https://example.com/schemas/{model}_v{version}.json",
    )

    schema_file = tmp_path / "User_v1.0.0.json"
    assert schema_file.exists()


def test_dump_schemas_with_refs_convenience_method(
    manager: ModelManager, tmp_path: Path
) -> None:
    """Test dump_schemas_with_refs convenience method."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    manager.dump_schemas_with_refs(tmp_path)

    schema_file = tmp_path / "User_v1.0.0.json"
    assert schema_file.exists()


def test_dump_schemas_with_refs_and_template(
    manager: ModelManager, tmp_path: Path
) -> None:
    """Test dump_schemas_with_refs with ref_template."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    manager.dump_schemas_with_refs(
        tmp_path, ref_template="https://example.com/schemas/{model}_v{version}.json"
    )

    schema_file = tmp_path / "User_v1.0.0.json"
    assert schema_file.exists()


def test_dump_schemas_with_refs_custom_indent(
    manager: ModelManager, tmp_path: Path
) -> None:
    """Test dump_schemas_with_refs with custom indent."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    manager.dump_schemas_with_refs(tmp_path, indent=4)

    schema_file = tmp_path / "User_v1.0.0.json"
    content = schema_file.read_text()
    assert "    " in content


def test_dump_schemas_multiple_models(manager: ModelManager, tmp_path: Path) -> None:
    """Test dump_schemas with multiple models."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("Product", "1.0.0")
    class ProductV1(BaseModel):
        title: str

    manager.dump_schemas(tmp_path)

    assert (tmp_path / "User_v1.0.0.json").exists()
    assert (tmp_path / "Product_v1.0.0.json").exists()


def test_dump_schemas_multiple_versions(
    manager: ModelManager,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
    tmp_path: Path,
) -> None:
    """Test dump_schemas with multiple versions of same model."""
    manager.model("User", "1.0.0")(user_v1)
    manager.model("User", "2.0.0")(user_v2)

    manager.dump_schemas(tmp_path)

    assert (tmp_path / "User_v1.0.0.json").exists()
    assert (tmp_path / "User_v2.0.0.json").exists()


# Nested models tests
def test_get_nested_models(manager: ModelManager) -> None:
    """Test getting nested models for a model."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    nested = manager.get_nested_models("User", "1.0.0")
    assert isinstance(nested, list)


def test_get_nested_models_with_model_version(manager: ModelManager) -> None:
    """Test getting nested models with ModelVersion object."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    nested = manager.get_nested_models("User", ModelVersion(1, 0, 0))
    assert isinstance(nested, list)


def test_get_nested_models_returns_tuples(manager: ModelManager) -> None:
    """Test that get_nested_models returns list of tuples."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    nested = manager.get_nested_models("User", "1.0.0")
    for item in nested:
        assert isinstance(item, tuple)
        assert len(item) == 2  # noqa: PLR2004
        assert isinstance(item[0], str)
        assert isinstance(item[1], ModelVersion)


# Diff tests
def test_diff_added_fields(manager: ModelManager) -> None:
    """Test detection of newly added fields."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str
        age: int

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert diff.model_name == "User"
    assert diff.from_version == "1.0.0"
    assert diff.to_version == "2.0.0"
    assert set(diff.added_fields) == {"email", "age"}
    assert diff.removed_fields == []
    assert diff.unchanged_fields == ["name"]
    assert diff.modified_fields == {}
    assert diff.added_field_info == {
        "age": {"default": None, "required": True, "type": int},
        "email": {"default": None, "required": True, "type": str},
    }


def test_diff_removed_fields(manager: ModelManager) -> None:
    """Test detection of removed fields."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        username: str
        age: int

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert diff.added_fields == []
    assert set(diff.removed_fields) == {"username", "age"}
    assert diff.unchanged_fields == ["name"]
    assert diff.modified_fields == {}


def test_diff_type_changed(manager: ModelManager) -> None:
    """Test detection of field type changes."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        age: int

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        age: str

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert diff.added_fields == []
    assert diff.removed_fields == []
    assert diff.unchanged_fields == []
    assert "age" in diff.modified_fields
    assert "type_changed" in diff.modified_fields["age"]
    assert "from" in diff.modified_fields["age"]["type_changed"]
    assert "to" in diff.modified_fields["age"]["type_changed"]


def test_diff_required_to_optional(manager: ModelManager) -> None:
    """Test detection of required field becoming optional."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        email: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        email: str | None = None

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert "email" in diff.modified_fields
    assert "required_changed" in diff.modified_fields["email"]
    assert diff.modified_fields["email"]["required_changed"]["from"] is True
    assert diff.modified_fields["email"]["required_changed"]["to"] is False


def test_diff_optional_to_required(manager: ModelManager) -> None:
    """Test detection of optional field becoming required."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        email: str | None = None

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        email: str

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert "email" in diff.modified_fields
    assert "required_changed" in diff.modified_fields["email"]
    assert diff.modified_fields["email"]["required_changed"]["from"] is False
    assert diff.modified_fields["email"]["required_changed"]["to"] is True


def test_diff_default_value_changed(manager: ModelManager) -> None:
    """Test detection of default value changes."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        status: str = "active"

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        status: str = "pending"

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert "status" in diff.modified_fields
    assert "default_changed" in diff.modified_fields["status"]
    assert diff.modified_fields["status"]["default_changed"]["from"] == "active"
    assert diff.modified_fields["status"]["default_changed"]["to"] == "pending"


def test_diff_default_value_added(manager: ModelManager) -> None:
    """Test detection of default value being added."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        status: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        status: str = "pending"

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert "status" in diff.modified_fields
    assert "default_added" in diff.modified_fields["status"]
    assert diff.modified_fields["status"]["default_added"] == "pending"


def test_diff_default_value_removed(manager: ModelManager) -> None:
    """Test detection of default value being removed."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        status: str = "active"

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        status: str

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert "status" in diff.modified_fields
    assert "default_removed" in diff.modified_fields["status"]
    assert diff.modified_fields["status"]["default_removed"] == "active"


def test_diff_multiple_changes_same_field(manager: ModelManager) -> None:
    """Test detection of multiple changes to the same field."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        age: int

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        age: str | None = "0"

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert "age" in diff.modified_fields
    assert "type_changed" in diff.modified_fields["age"]
    assert "required_changed" in diff.modified_fields["age"]
    assert "default_added" in diff.modified_fields["age"]


def test_diff_no_changes(manager: ModelManager) -> None:
    """Test diff when models are identical."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        email: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert diff.added_fields == []
    assert diff.removed_fields == []
    assert set(diff.unchanged_fields) == {"name", "email"}
    assert diff.modified_fields == {}


def test_diff_with_model_version_objects(manager: ModelManager) -> None:
    """Test diff with ModelVersion objects instead of strings."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    diff = manager.diff("User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0))

    assert "email" in diff.added_fields
    assert diff.unchanged_fields == ["name"]


def test_diff_complex_scenario(manager: ModelManager) -> None:
    """Test diff with a complex mix of changes."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        username: str
        age: int
        status: str = "active"

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str
        age: str | None = None
        role: str = "user"

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert set(diff.added_fields) == {"email", "role"}
    assert set(diff.removed_fields) == {"username", "status"}
    assert diff.unchanged_fields == ["name"]
    assert "age" in diff.modified_fields
    assert "type_changed" in diff.modified_fields["age"]
    assert "required_changed" in diff.modified_fields["age"]
    assert "default_added" in diff.modified_fields["age"]


def test_diff_with_field_validator(manager: ModelManager) -> None:
    """Test diff works correctly with Pydantic Field validators."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        age: int = Field(ge=0)

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        age: int = Field(ge=0, le=120)

    diff = manager.diff("User", "1.0.0", "2.0.0")

    assert diff.unchanged_fields == ["age"]
