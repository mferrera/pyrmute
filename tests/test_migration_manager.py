"""Tests MigrationManager."""

from typing import Any

import pytest
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from pyrmute import (
    MigrationError,
    ModelData,
    ModelManager,
    ModelNotFoundError,
    ModelVersion,
)
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
    data: ModelData = {"name": "Alice"}
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

    data: ModelData = {"name": "Bob"}
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

    data: ModelData = {"name": "Charlie"}
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

    data: ModelData = {"name": "David"}
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "3.0.0")
    assert result == {"name": "David", "email": "default@example.com", "age": 25}


def test_migrate_backward_compatibility(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migration can go backwards through versions."""

    @populated_migration_manager.register_migration("User", "3.0.0", "2.0.0")
    def migrate_3_to_2(data: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in data.items() if k != "age"}

    data: ModelData = {"name": "Eve", "email": "eve@example.com", "age": 30}
    result = populated_migration_manager.migrate(data, "User", "3.0.0", "2.0.0")
    assert result == {"name": "Eve", "email": "eve@example.com"}


def test_migrate_preserves_extra_fields(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migration preserves fields not in migration function."""

    @populated_migration_manager.register_migration("User", "1.0.0", "2.0.0")
    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "new@example.com"}

    data: ModelData = {"name": "Frank", "custom_field": "value"}
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "2.0.0")
    assert result["custom_field"] == "value"


def test_migration_fails_if_no_direct_path(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migration fails if no direct migration path is found."""
    data: ModelData = {"name": "Grace"}
    with pytest.raises(
        MigrationError,
        match=r"Migration failed for 'User': 1.0.0 → 2.0.0",
    ) as e:
        populated_migration_manager.migrate(data, "User", "1.0.0", "2.0.0")
        assert "no path" in str(e)


def test_migration_fails_if_no_transient_path(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test migration fails if no transient migration path is found."""
    data: ModelData = {"name": "Grace"}
    with pytest.raises(
        MigrationError,
        match=r"Migration failed for 'User': 1.0.0 → 2.0.0",
    ) as e:
        populated_migration_manager.migrate(data, "User", "1.0.0", "3.0.0")
        assert "no path" in str(e)


# Auto-migration tests
def test_backward_compatible_adds_default_fields(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test auto-migration adds new, required fields with defaults."""
    data: ModelData = {"name": "Grace", "email": "foo@bar.com"}
    result = populated_migration_manager.migrate(data, "User", "2.0.0", "3.0.0")
    assert result == {**data, "age": 0}


def test_backward_compatible_adds_default_fields_and_uses_migration_func(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test auto-migration with default field uses migration func first."""
    data: ModelData = {"name": "Grace", "email": "foo@bar.com"}

    @populated_migration_manager.register_migration("User", "2.0.0", "3.0.0")
    def migrate_user_age(data: ModelData) -> ModelData:
        return {**data, "age": 5}

    result = populated_migration_manager.migrate(data, "User", "2.0.0", "3.0.0")
    assert result == {**data, "age": 5}


def test_backward_compatible_adds_default_factory_fields(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test auto-migration adds new, required fields with a default factory."""
    data: ModelData = {"name": "Grace", "email": "foo@bar.com"}
    result = populated_migration_manager.migrate(data, "User", "2.0.0", "4.0.0")
    assert result == {**data, "age": 0, "aliases": []}


def test_backward_compatible_adds_default_factory_fields_uses_migration_func(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test auto-migration with default factory uses migration func first."""
    data: ModelData = {"name": "Grace", "email": "foo@bar.com"}

    @populated_migration_manager.register_migration("User", "2.0.0", "3.0.0")
    def migrate_user_age(data: ModelData) -> ModelData:
        return {**data, "age": 5}

    @populated_migration_manager.register_migration("User", "3.0.0", "4.0.0")
    def migrate_user_aliases(data: ModelData) -> ModelData:
        return {**data, "aliases": ["Bob"]}

    result = populated_migration_manager.migrate(data, "User", "2.0.0", "4.0.0")
    assert result == {**data, "age": 5, "aliases": ["Bob"]}


def test_migration_with_default_factory(manager: ModelManager) -> None:
    """Test that default_factory is called for missing fields."""

    @manager.model("Optional", "1.0.0", backward_compatible=True)
    class OptionalV1(BaseModel):
        field1: str = "default1"

    @manager.model("Optional", "2.0.0", backward_compatible=True)
    class OptionalV2(BaseModel):
        field1: str = "default1"
        field3: list[str] = Field(default_factory=list)

    # When field is missing, default_factory should be called
    result = manager.migration_manager.migrate(
        {"field1": "test"}, "Optional", "1.0.0", "2.0.0"
    )
    assert result["field3"] == []


def test_migration_preserves_explicit_none(manager: ModelManager) -> None:
    """Test that explicit None values are preserved."""

    @manager.model("Optional", "1.0.0", backward_compatible=True)
    class OptionalV1(BaseModel):
        field1: str = "default1"
        field3: list[str] | None = None

    @manager.model("Optional", "2.0.0", backward_compatible=True)
    class OptionalV2(BaseModel):
        field1: str = "default1"
        field3: list[str] | None = Field(default_factory=list)

    # When field is explicitly None, it should be preserved
    result = manager.migration_manager.migrate(
        {"field1": "test", "field3": None}, "Optional", "1.0.0", "2.0.0"
    )
    assert result["field3"] is None  # Preserved, not replaced with []


def test_backward_compatible_handles_none_values(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test auto-migration handles None values correctly."""
    data: ModelData = {"name": None, "email": "foo@bar.com"}
    result = populated_migration_manager.migrate(data, "User", "2.0.0", "3.0.0")
    assert result["name"] is None


def test_backward_compatible_preserves_extra_fields(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test auto-migration handles None values correctly."""
    data: ModelData = {"name": "Grace", "email": "foo@bar.com", "foo": "bar"}
    result = populated_migration_manager.migrate(data, "User", "2.0.0", "3.0.0")
    assert result == {"name": "Grace", "email": "foo@bar.com", "foo": "bar", "age": 0}


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
    registry.register("Person", "2.0.0", backward_compatible=True)(PersonV2)

    manager = MigrationManager(registry)

    @manager.register_migration("Address", "1.0.0", "2.0.0")
    def migrate_address(data: ModelData) -> ModelData:
        return {**data, "city": "Unknown"}

    data: ModelData = {"name": "Iris", "address": {"street": "123 Main St"}}

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
    registry.register("Order", "2.0.0", backward_compatible=True)(OrderV2)

    manager = MigrationManager(registry)

    @manager.register_migration("Item", "1.0.0", "2.0.0")
    def migrate_item(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "quantity": 1}

    data: ModelData = {"items": [{"name": "Apple"}, {"name": "Banana"}]}

    result = manager.migrate(data, "Order", "1.0.0", "2.0.0")
    assert len(result["items"]) == 2  # noqa: PLR2004
    assert result["items"][0]["quantity"] == 1
    assert result["items"][1]["quantity"] == 1


def test_migrate_dict_values(populated_migration_manager: MigrationManager) -> None:
    """Test migration handles dictionary values."""

    @populated_migration_manager.register_migration("User", "1.0.0", "2.0.0")
    def migrate(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "email": "default@example.com"}

    data: ModelData = {
        "name": "Jack",
        "metadata": {"key1": "value1", "key2": "value2"},
    }
    result = populated_migration_manager.migrate(data, "User", "1.0.0", "2.0.0")
    assert result["metadata"] == {"key1": "value1", "key2": "value2"}


# Migration path tests
def test_find_migration_path_forward(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test finding migration path from lower to higher version."""
    path = populated_migration_manager.find_migration_path(
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
    path = populated_migration_manager.find_migration_path(
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
    path = populated_migration_manager.find_migration_path(
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
    )
    assert path == [ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)]


def test_find_migration_path_same_version(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test finding migration path for same version."""
    path = populated_migration_manager.find_migration_path(
        "User",
        ModelVersion(2, 0, 0),
        ModelVersion(2, 0, 0),
    )
    assert path == [ModelVersion(2, 0, 0)]


def test_find_migration_path_invalid_from_version(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test finding migration path with invalid from version."""
    with pytest.raises(ModelNotFoundError, match=r"Model 'User' version '0.0.1'"):
        populated_migration_manager.find_migration_path(
            "User",
            ModelVersion(0, 0, 1),
            ModelVersion(2, 0, 0),
        )


def test_find_migration_path_invalid_to_version(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test finding migration path with invalid to version."""
    with pytest.raises(ModelNotFoundError, match=r"Model 'User' version '9.0.0'"):
        populated_migration_manager.find_migration_path(
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


def test_validate_migration_path_direct_migration(
    registered_manager: ModelManager,
) -> None:
    """Test validate_migration_path with direct migration."""
    # Should not raise
    registered_manager.migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)
    )


def test_validate_migration_path_no_migration_raises(
    manager: ModelManager,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
) -> None:
    """Test validate_migration_path raises when no migration exists."""
    manager.model("User", "1.0.0")(user_v1)
    manager.model("User", "2.0.0")(user_v2)

    with pytest.raises(
        MigrationError, match=r"Migration failed for 'User': 1.0.0 → 2.0.0"
    ):
        manager.migration_manager.validate_migration_path(
            "User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)
        )


def test_validate_migration_path_backward_compatible_enabled(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path succeeds with backward_compatible."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        email: str = "default@example.com"

    # Should not raise
    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)
    )


def test_validate_migration_path_multi_hop_complete(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path with complete multi-hop chain."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.model("User", "3.0.0")
    class UserV3(BaseModel):
        name: str
        email: str
        age: int

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate_1_to_2(data: ModelData) -> ModelData:
        return {**data, "email": "test@example.com"}

    @manager.migration("User", "2.0.0", "3.0.0")
    def migrate_2_to_3(data: ModelData) -> ModelData:
        return {**data, "age": 0}

    # Should not raise
    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(3, 0, 0)
    )


def test_validate_migration_path_multi_hop_broken_chain(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path raises with broken chain."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.model("User", "3.0.0")
    class UserV3(BaseModel):
        name: str
        email: str
        age: int

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate_1_to_2(data: ModelData) -> ModelData:
        return {**data, "email": "test@example.com"}

    with pytest.raises(
        MigrationError, match=r"Migration failed for 'User': 2.0.0 → 3.0.0"
    ):
        manager.migration_manager.validate_migration_path(
            "User", ModelVersion(1, 0, 0), ModelVersion(3, 0, 0)
        )


def test_validate_migration_path_multi_hop_first_step_missing(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path raises when first step is missing."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.model("User", "3.0.0")
    class UserV3(BaseModel):
        name: str
        email: str
        age: int

    @manager.migration("User", "2.0.0", "3.0.0")
    def migrate_2_to_3(data: ModelData) -> ModelData:
        return {**data, "age": 0}

    with pytest.raises(
        MigrationError, match=r"Migration failed for 'User': 1.0.0 → 2.0.0"
    ):
        manager.migration_manager.validate_migration_path(
            "User", ModelVersion(1, 0, 0), ModelVersion(3, 0, 0)
        )


def test_validate_migration_path_complex_chain(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path with complex multi-hop chain."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "1.5.0")
    class UserV15(BaseModel):
        name: str
        middle_name: str = ""

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        middle_name: str
        email: str

    @manager.model("User", "3.0.0")
    class UserV3(BaseModel):
        name: str
        middle_name: str
        email: str
        age: int

    @manager.migration("User", "1.0.0", "1.5.0")
    def migrate_1_to_15(data: ModelData) -> ModelData:
        return {**data, "middle_name": ""}

    @manager.migration("User", "1.5.0", "2.0.0")
    def migrate_15_to_2(data: ModelData) -> ModelData:
        return {**data, "email": "test@example.com"}

    @manager.migration("User", "2.0.0", "3.0.0")
    def migrate_2_to_3(data: ModelData) -> ModelData:
        return {**data, "age": 0}

    # Should not raise for any valid path
    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(1, 5, 0)
    )
    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(1, 5, 0), ModelVersion(2, 0, 0)
    )
    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(2, 0, 0), ModelVersion(3, 0, 0)
    )
    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(3, 0, 0)
    )


def test_validate_migration_path_nonexistent_model(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path with nonexistent model."""
    with pytest.raises(ModelNotFoundError, match="Model 'NonExistent' not found"):
        manager.migration_manager.validate_migration_path(
            "NonExistent", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)
        )


def test_validate_migration_path_nonexistent_from_version(
    manager: ModelManager,
    user_v1: type[BaseModel],
) -> None:
    """Test validate_migration_path with nonexistent source version."""
    manager.model("User", "1.0.0")(user_v1)

    with pytest.raises(ModelNotFoundError, match=r"Model 'User' version '2.0.0'"):
        manager.migration_manager.validate_migration_path(
            "User", ModelVersion(2, 0, 0), ModelVersion(1, 0, 0)
        )


def test_validate_migration_path_nonexistent_to_version(
    manager: ModelManager,
    user_v1: type[BaseModel],
) -> None:
    """Test validate_migration_path with nonexistent target version."""
    manager.model("User", "1.0.0")(user_v1)

    with pytest.raises(ModelNotFoundError, match=r"Model 'User' version '2.0.0'"):
        manager.migration_manager.validate_migration_path(
            "User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)
        )


def test_validate_migration_path_same_version(
    manager: ModelManager,
    user_v1: type[BaseModel],
) -> None:
    """Test validate_migration_path with same source and target."""
    manager.model("User", "1.0.0")(user_v1)

    # Should not raise
    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(1, 0, 0)
    )


# Backward migration tests
def test_validate_migration_path_backward_no_migration(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path for backward migration without migration."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    # Forward migration only
    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate_forward(data: ModelData) -> ModelData:
        return {**data, "email": "test@example.com"}

    with pytest.raises(
        MigrationError, match=r"Migration failed for 'User': 2.0.0 → 1.0.0"
    ):
        manager.migration_manager.validate_migration_path(
            "User", ModelVersion(2, 0, 0), ModelVersion(1, 0, 0)
        )


def test_validate_migration_path_bidirectional(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path with bidirectional migrations."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate_forward(data: ModelData) -> ModelData:
        return {**data, "email": "test@example.com"}

    @manager.migration("User", "2.0.0", "1.0.0")
    def migrate_backward(data: ModelData) -> ModelData:
        result = dict(data)
        result.pop("email", None)
        return result

    # Both directions should not raise
    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)
    )
    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(2, 0, 0), ModelVersion(1, 0, 0)
    )


def test_validate_migration_path_mixed_auto_explicit(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path with mix of auto and explicit migrations."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        email: str = "default@example.com"

    @manager.model("User", "3.0.0")
    class UserV3(BaseModel):
        name: str
        email: str
        age: int

    @manager.migration("User", "2.0.0", "3.0.0")
    def migrate_2_to_3(data: ModelData) -> ModelData:
        return {**data, "age": 0}

    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(3, 0, 0)
    )


def test_validate_migration_path_all_backward_compatible(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path with all auto-migrations."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        email: str = "default@example.com"

    @manager.model("User", "3.0.0", backward_compatible=True)
    class UserV3(BaseModel):
        name: str
        email: str
        age: int = 0

    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(3, 0, 0)
    )


def test_validate_migration_path_explicit_overrides_auto(
    manager: ModelManager,
) -> None:
    """Test that explicit migrations take precedence over auto-migrate."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        email: str = "auto@example.com"

    @manager.migration("User", "1.0.0", "2.0.0")
    def explicit_migration(data: ModelData) -> ModelData:
        return {**data, "email": "explicit@example.com"}

    # Should not raise
    manager.migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)
    )


def test_validate_migration_path_middle_version_backward_compatible_disabled(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path fails when middle version has no migration."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0", backward_compatible=False)
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.model("User", "3.0.0", backward_compatible=True)
    class UserV3(BaseModel):
        name: str
        email: str
        age: int = 0

    with pytest.raises(
        MigrationError, match=r"Migration failed for 'User': 1.0.0 → 2.0.0"
    ):
        manager.migration_manager.validate_migration_path(
            "User", ModelVersion(1, 0, 0), ModelVersion(3, 0, 0)
        )


def test_auto_migration_raises_on_field_processing_error(
    manager: ModelManager,
) -> None:
    """Test that auto-migration wraps exceptions during field value migration."""

    class BrokenDict(dict[str, Any]):
        """Dict that raises on iteration."""

        def items(self) -> Any:
            raise RuntimeError("Intentionally broken dict")

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        metadata: dict[str, str]

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        metadata: dict[str, str]
        email: str = "default@example.com"

    data: ModelData = {"name": "Alice", "metadata": BrokenDict()}
    with pytest.raises(
        MigrationError,
        match=r"Migration failed for 'User': 1.0.0 → 2.0.0",
    ) as exc_info:
        manager.migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert "Auto-migration failed" in str(exc_info.value)
    assert "RuntimeError" in str(exc_info.value)


def test_auto_migration_raises_on_default_factory_error(
    manager: ModelManager,
) -> None:
    """Test that auto-migration handles default_factory exceptions."""

    def bad_factory() -> list[str]:
        raise RuntimeError("Factory intentionally broken")

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        tags: list[str]

    UserV2.model_fields["tags"].default_factory = bad_factory
    data: ModelData = {"name": "Bob"}
    result = manager.migration_manager.migrate(data, "User", "1.0.0", "2.0.0")
    assert "tags" not in result


def test_auto_migration_nested_model_migration_error(
    manager: ModelManager,
) -> None:
    """Test that auto-migration propagates nested migration errors."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str

    @manager.model("Address", "2.0.0")
    class AddressV2(BaseModel):
        street: str
        city: str  # Required field with no default

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        address: AddressV2

    data: ModelData = {"name": "Charlie", "address": {"street": "123 Main St"}}
    with pytest.raises(
        MigrationError,
        match=r"Migration failed for 'Address': 1.0.0 → 2.0.0",
    ):
        manager.migration_manager.migrate(data, "User", "1.0.0", "2.0.0")


def test_auto_migration_preserves_exception_chain(manager: ModelManager) -> None:
    """Test that auto-migration preserves the exception chain."""

    class BrokenList(list[Any]):
        """List that raises on iteration."""

        def __iter__(self) -> Any:
            raise ValueError("Intentional error in list iteration")

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        items: list[str]

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        items: list[str]
        email: str = "default@example.com"

    data: ModelData = {"name": "Diana", "items": BrokenList(["a", "b"])}

    with pytest.raises(MigrationError) as exc_info:
        manager.migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, Exception)
    assert "Intentional error in list iteration" in str(exc_info.value.__cause__)
