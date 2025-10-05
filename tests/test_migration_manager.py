"""Tests MigrationManager."""

from typing import Any

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from pyrmute import MigrationData, ModelVersion
from pyrmute._migration_manager import MigrationManager
from pyrmute._registry import Registry


# Initialization tests
def test_manager_initialization(registry: Registry) -> None:
    """Test MigrationManager initializes with registry."""
    manager = MigrationManager(registry)
    assert manager.registry is registry


# Migration registration tests
def test_register_migration_with_string_versions(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test registering migration with string versions."""

    @populated_migration_manager.register_migration("User", "1.0.0", "2.0.0")
    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "test@example.com"}

    migrations = populated_migration_manager.registry._migrations["User"]
    key = (ModelVersion(1, 0, 0), ModelVersion(2, 0, 0))
    assert key in migrations
    assert migrations[key] == migrate


def test_register_migration_with_model_versions(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test registering migration with ModelVersion objects."""
    from_ver = ModelVersion(1, 0, 0)
    to_ver = ModelVersion(2, 0, 0)

    @populated_migration_manager.register_migration("User", from_ver, to_ver)
    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "test@example.com"}

    migrations = populated_migration_manager.registry._migrations["User"]
    assert (from_ver, to_ver) in migrations


def test_register_migration_returns_function(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test that register_migration returns the decorated function."""

    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return data

    result = populated_migration_manager.register_migration("User", "1.0.0", "2.0.0")(
        migrate
    )
    assert result is migrate


def test_register_multiple_migrations(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test registering multiple migrations for same model."""

    @populated_migration_manager.register_migration("User", "1.0.0", "2.0.0")
    def migrate_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "default@example.com"}

    @populated_migration_manager.register_migration("User", "2.0.0", "3.0.0")
    def migrate_2_to_3(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "age": 0}

    migrations = populated_migration_manager.registry._migrations["User"]
    assert len(migrations) == 2  # noqa: PLR2004


def test_register_migration_different_models(
    registry: Registry,
) -> None:
    """Test registering migrations for different models."""

    class ProductV1(BaseModel):
        name: str

    class ProductV2(BaseModel):
        name: str
        price: float

    registry.register("Product", "1.0.0")(ProductV1)
    registry.register("Product", "2.0.0")(ProductV2)

    manager = MigrationManager(registry)

    @manager.register_migration("Product", "1.0.0", "2.0.0")
    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "price": 0.0}

    assert (
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
    ) in manager.registry._migrations["Product"]


# Migration execution tests
def test_migrate_same_version_returns_unchanged(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migrating to same version returns data unchanged."""
    data: MigrationData = {"name": "Alice"}
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "1.0.0")
    assert result == data
    assert result is data


def test_migrate_with_explicit_migration(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migration uses registered migration function."""

    @populated_migration_manager.register_migration("User", "1.0.0", "2.0.0")
    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "migrated@example.com"}

    data: MigrationData = {"name": "Bob"}
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "2.0.0")
    assert result == {"name": "Bob", "email": "migrated@example.com"}


def test_migrate_with_model_versions(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migration with ModelVersion objects."""

    @populated_migration_manager.register_migration("User", "1.0.0", "2.0.0")
    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "test@example.com"}

    from_ver = ModelVersion(1, 0, 0)
    to_ver = ModelVersion(2, 0, 0)

    data: MigrationData = {"name": "Charlie"}
    result = populated_migration_manager.migrate(data, "User", from_ver, to_ver)
    assert result == {"name": "Charlie", "email": "test@example.com"}


def test_migrate_chain_multiple_versions(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migration chains through multiple versions."""

    @populated_migration_manager.register_migration("User", "1.0.0", "2.0.0")
    def migrate_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "default@example.com"}

    @populated_migration_manager.register_migration("User", "2.0.0", "3.0.0")
    def migrate_2_to_3(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "age": 25}

    data: MigrationData = {"name": "David"}
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "3.0.0")
    assert result == {"name": "David", "email": "default@example.com", "age": 25}


def test_migrate_backward_compatibility(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migration can go backwards through versions."""

    @populated_migration_manager.register_migration("User", "3.0.0", "2.0.0")
    def migrate_3_to_2(data: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in data.items() if k != "age"}

    data: MigrationData = {"name": "Eve", "email": "eve@example.com", "age": 30}
    result = populated_migration_manager.migrate(data, "User", "3.0.0", "2.0.0")
    assert result == {"name": "Eve", "email": "eve@example.com"}


def test_migrate_preserves_extra_fields(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migration preserves fields not in migration function."""

    @populated_migration_manager.register_migration("User", "1.0.0", "2.0.0")
    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "new@example.com"}

    data: MigrationData = {"name": "Frank", "custom_field": "value"}
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "2.0.0")
    assert result["custom_field"] == "value"


# Auto-migration tests
def test_auto_migrate_copies_common_fields(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test auto-migration copies fields present in both versions."""
    data: MigrationData = {"name": "Grace"}
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "2.0.0")
    assert result == {"name": "Grace"}


def test_auto_migrate_skips_missing_fields(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test auto-migration skips fields not in source data."""
    data: MigrationData = {"name": "Henry"}
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "3.0.0")
    # Should only have name, as email and age are not in source data
    assert "name" in result
    assert "email" not in result
    assert "age" not in result


def test_auto_migrate_handles_none_values(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test auto-migration handles None values correctly."""
    data: MigrationData = {"name": None}
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "2.0.0")
    assert result["name"] is None


# Nested model migration tests
def test_migrate_nested_model(registry: Registry) -> None:
    """Test migration with nested Pydantic models."""

    class AddressV1(BaseModel):
        street: str

    class AddressV2(BaseModel):
        street: str
        city: str

    class PersonV1(BaseModel):
        name: str
        address: AddressV1

    class PersonV2(BaseModel):
        name: str
        address: AddressV2

    registry.register("Address", "1.0.0")(AddressV1)
    registry.register("Address", "2.0.0")(AddressV2)
    registry.register("Person", "1.0.0")(PersonV1)
    registry.register("Person", "2.0.0")(PersonV2)

    manager = MigrationManager(registry)

    @manager.register_migration("Address", "1.0.0", "2.0.0")
    def migrate_address(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "city": "Unknown"}

    data: MigrationData = {"name": "Iris", "address": {"street": "123 Main St"}}

    result = manager.migrate(data, "Person", "1.0.0", "2.0.0")
    assert result["address"]["street"] == "123 Main St"
    assert result["address"]["city"] == "Unknown"


def test_migrate_list_of_nested_models(registry: Registry) -> None:
    """Test migration with list of nested models."""

    class ItemV1(BaseModel):
        name: str

    class ItemV2(BaseModel):
        name: str
        quantity: int

    class OrderV1(BaseModel):
        items: list[ItemV1]

    class OrderV2(BaseModel):
        items: list[ItemV2]

    registry.register("Item", "1.0.0")(ItemV1)
    registry.register("Item", "2.0.0")(ItemV2)
    registry.register("Order", "1.0.0")(OrderV1)
    registry.register("Order", "2.0.0")(OrderV2)

    manager = MigrationManager(registry)

    @manager.register_migration("Item", "1.0.0", "2.0.0")
    def migrate_item(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "quantity": 1}

    data: MigrationData = {"items": [{"name": "Apple"}, {"name": "Banana"}]}

    result = manager.migrate(data, "Order", "1.0.0", "2.0.0")
    assert len(result["items"]) == 2  # noqa: PLR2004
    assert result["items"][0]["quantity"] == 1
    assert result["items"][1]["quantity"] == 1


def test_migrate_dict_values(populated_migration_manager: MigrationManager) -> None:
    """Test migration handles dictionary values."""

    @populated_migration_manager.register_migration("User", "1.0.0", "2.0.0")
    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "default@example.com"}

    data: MigrationData = {
        "name": "Jack",
        "metadata": {"key1": "value1", "key2": "value2"},
    }
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "2.0.0")
    assert result["metadata"] == {"key1": "value1", "key2": "value2"}


def test_auto_migrate_only_copies_model_fields(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test auto-migration only copies fields defined in models."""
    data: MigrationData = {
        "name": "Jack",
        "metadata": {"key1": "value1", "key2": "value2"},  # Extra field
    }
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "2.0.0")
    assert result == {"name": "Jack"}  # metadata is dropped
    assert "metadata" not in result


# Migration path tests
def test_find_migration_path_forward(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test finding migration path from lower to higher version."""
    path = populated_migration_manager._find_migration_path(
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(3, 0, 0),
    )
    assert path == [
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
        ModelVersion(3, 0, 0),
    ]


def test_find_migration_path_backward(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test finding migration path from higher to lower version."""
    path = populated_migration_manager._find_migration_path(
        "User",
        ModelVersion(3, 0, 0),
        ModelVersion(1, 0, 0),
    )
    assert path == [
        ModelVersion(3, 0, 0),
        ModelVersion(2, 0, 0),
        ModelVersion(1, 0, 0),
    ]


def test_find_migration_path_adjacent(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test finding migration path for adjacent versions."""
    path = populated_migration_manager._find_migration_path(
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
    )
    assert path == [ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)]


def test_find_migration_path_same_version(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test finding migration path for same version."""
    path = populated_migration_manager._find_migration_path(
        "User",
        ModelVersion(2, 0, 0),
        ModelVersion(2, 0, 0),
    )
    assert path == [ModelVersion(2, 0, 0)]


def test_find_migration_path_invalid_from_version(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test finding migration path with invalid from version."""
    with pytest.raises(ValueError, match="Invalid version range"):
        populated_migration_manager._find_migration_path(
            "User",
            ModelVersion(0, 0, 1),
            ModelVersion(2, 0, 0),
        )


def test_find_migration_path_invalid_to_version(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test finding migration path with invalid to version."""
    with pytest.raises(ValueError, match="Invalid version range"):
        populated_migration_manager._find_migration_path(
            "User",
            ModelVersion(1, 0, 0),
            ModelVersion(9, 0, 0),
        )


# Field value migration tests
def test_migrate_field_value_none(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migrating None field value."""
    field_info = FieldInfo(annotation=str)
    result = populated_migration_manager._migrate_field_value(
        None, field_info, field_info
    )
    assert result is None


def test_migrate_field_value_list(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migrating list field value."""
    field_info = FieldInfo(annotation=list[str])
    value = ["a", "b", "c"]
    result = populated_migration_manager._migrate_field_value(
        value, field_info, field_info
    )
    assert result == ["a", "b", "c"]


def test_migrate_field_value_dict(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migrating dict field value."""
    field_info = FieldInfo(annotation=dict[str, Any])
    value = {"key": "value"}
    result = populated_migration_manager._migrate_field_value(
        value, field_info, field_info
    )
    assert result == {"key": "value"}


def test_migrate_field_value_primitive(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migrating primitive field value."""
    field_info = FieldInfo(annotation=str)
    result = populated_migration_manager._migrate_field_value(
        "test", field_info, field_info
    )
    assert result == "test"


# Model type extraction tests
def test_get_model_type_from_field_direct(
    populated_migration_manager: MigrationManager,
    user_v1: type[BaseModel],
) -> None:
    """Test extracting direct model type from field."""
    field_info = FieldInfo(annotation=user_v1)
    model_type = populated_migration_manager._get_model_type_from_field(field_info)
    assert model_type is user_v1


def test_get_model_type_from_field_optional(
    populated_migration_manager: MigrationManager,
    user_v1: type[BaseModel],
) -> None:
    """Test extracting model type from optional field."""
    field_info = FieldInfo(annotation=user_v1 | None)  # type: ignore
    model_type = populated_migration_manager._get_model_type_from_field(field_info)
    assert model_type is user_v1


def test_get_model_type_from_field_list(
    populated_migration_manager: MigrationManager,
    user_v1: type[BaseModel],
) -> None:
    """Test extracting model type from list field."""
    field_info = FieldInfo(annotation=list[user_v1])  # type: ignore
    model_type = populated_migration_manager._get_model_type_from_field(field_info)
    assert model_type is user_v1


def test_get_model_type_from_field_none_annotation(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test extracting model type from field with None annotation."""
    field_info = FieldInfo(annotation=None)
    model_type = populated_migration_manager._get_model_type_from_field(field_info)
    assert model_type is None


def test_get_model_type_from_field_primitive(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test extracting model type from primitive field returns None."""
    field_info = FieldInfo(annotation=str)
    model_type = populated_migration_manager._get_model_type_from_field(field_info)
    assert model_type is None


# Nested model info extraction tests
def test_extract_nested_model_info_registered(registry: Registry) -> None:
    """Test extracting info for registered nested model."""

    class AddressV1(BaseModel):
        street: str

    class AddressV2(BaseModel):
        street: str
        city: str

    registry.register("Address", "1.0.0")(AddressV1)
    registry.register("Address", "2.0.0")(AddressV2)

    manager = MigrationManager(registry)

    from_field = FieldInfo(annotation=AddressV1)
    to_field = FieldInfo(annotation=AddressV2)

    info = manager._extract_nested_model_info(
        {"street": "123 Main"},
        from_field,
        to_field,
    )

    assert info is not None
    assert info[0] == "Address"
    assert info[1] == ModelVersion(1, 0, 0)
    assert info[2] == ModelVersion(2, 0, 0)


def test_extract_nested_model_info_not_basemodel(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test extracting info when field is not BaseModel returns None."""
    field = FieldInfo(annotation=str)
    info = populated_migration_manager._extract_nested_model_info(
        {"value": "test"},
        field,
        field,
    )
    assert info is None


def test_extract_nested_model_info_unregistered(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test extracting info for unregistered model returns None."""

    class UnregisteredModel(BaseModel):
        field: str

    field = FieldInfo(annotation=UnregisteredModel)
    info = populated_migration_manager._extract_nested_model_info(
        {"field": "value"},
        field,
        field,
    )
    assert info is None


def test_extract_nested_model_info_no_from_field(registry: Registry) -> None:
    """Test extracting info when from_field is None."""

    class AddressV1(BaseModel):
        street: str

    registry.register("Address", "1.0.0")(AddressV1)
    manager = MigrationManager(registry)

    to_field = FieldInfo(annotation=AddressV1)

    info = manager._extract_nested_model_info(
        {"street": "123 Main"},
        None,
        to_field,
    )

    assert info is not None
    assert info[0] == "Address"
    # Should default to same version when from_field is None
    assert info[1] == ModelVersion(1, 0, 0)
    assert info[2] == ModelVersion(1, 0, 0)
