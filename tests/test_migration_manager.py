"""Tests MigrationManager."""

from typing import Annotated, Any, Literal, Union, get_origin

import pytest
from pydantic import AliasChoices, BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

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
    result = manager._migration_manager.migrate(
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
    result = manager._migration_manager.migrate(
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


def test_find_migration_path_direct_when_available(
    manager: ModelManager,
) -> None:
    """Test that find_migration_path returns direct path when migration exists."""

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

    @manager.migration("User", "1.0.0", "3.0.0")
    def migrate_direct(data: ModelData) -> ModelData:
        return data

    path = manager._migration_manager.find_migration_path(
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(3, 0, 0),
    )

    # Should return direct path, not sequential
    assert path == [ModelVersion(1, 0, 0), ModelVersion(3, 0, 0)]


def test_find_migration_path_sequential_when_no_direct(
    manager: ModelManager,
) -> None:
    """Test find_migration_path returns sequential when no direct migration exists."""

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

    path = manager._migration_manager.find_migration_path(
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(3, 0, 0),
    )

    assert path == [
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
        ModelVersion(3, 0, 0),
    ]


def test_find_migration_path_direct_backward(
    manager: ModelManager,
) -> None:
    """Test that find_migration_path finds direct backward migration."""

    @manager.model("Config", "1.0.0")
    class ConfigV1(BaseModel):
        value: str

    @manager.model("Config", "2.0.0")
    class ConfigV2(BaseModel):
        value: str
        extra: str

    # Register direct backward migration
    @manager.migration("Config", "2.0.0", "1.0.0")
    def downgrade(data: ModelData) -> ModelData:
        return {"value": data["value"]}

    path = manager._migration_manager.find_migration_path(
        "Config",
        ModelVersion(2, 0, 0),
        ModelVersion(1, 0, 0),
    )

    assert path == [ModelVersion(2, 0, 0), ModelVersion(1, 0, 0)]


def test_find_migration_path_prefers_direct_over_sequential_backward(
    manager: ModelManager,
) -> None:
    """Test that direct path is preferred even for backward migration."""

    @manager.model("Schema", "1.0.0")
    class SchemaV1(BaseModel):
        a: str

    @manager.model("Schema", "2.0.0")
    class SchemaV2(BaseModel):
        a: str
        b: str

    @manager.model("Schema", "3.0.0")
    class SchemaV3(BaseModel):
        a: str
        b: str
        c: str

    # Register direct 3->1 migration
    @manager.migration("Schema", "3.0.0", "1.0.0")
    def downgrade_direct(data: ModelData) -> ModelData:
        return {"a": data["a"]}

    path = manager._migration_manager.find_migration_path(
        "Schema",
        ModelVersion(3, 0, 0),
        ModelVersion(1, 0, 0),
    )

    # Should return direct path, skipping v2
    assert path == [ModelVersion(3, 0, 0), ModelVersion(1, 0, 0)]


def test_find_migration_path_multiple_direct_options(
    manager: ModelManager,
) -> None:
    """Test path finding when multiple direct jumps are available."""

    @manager.model("Data", "1.0.0")
    class DataV1(BaseModel):
        x: int

    @manager.model("Data", "2.0.0")
    class DataV2(BaseModel):
        x: int
        y: int

    @manager.model("Data", "3.0.0")
    class DataV3(BaseModel):
        x: int
        y: int
        z: int

    @manager.model("Data", "4.0.0")
    class DataV4(BaseModel):
        x: int
        y: int
        z: int
        w: int

    @manager.migration("Data", "1.0.0", "3.0.0")
    def jump_1_to_3(data: ModelData) -> ModelData:
        return {**data, "y": 0, "z": 0}

    @manager.migration("Data", "3.0.0", "4.0.0")
    def jump_3_to_4(data: ModelData) -> ModelData:
        return {**data, "w": 0}

    path = manager._migration_manager.find_migration_path(
        "Data",
        ModelVersion(1, 0, 0),
        ModelVersion(4, 0, 0),
    )

    assert len(path) >= 2  # noqa: PLR2004
    assert path[0] == ModelVersion(1, 0, 0)
    assert path[-1] == ModelVersion(4, 0, 0)


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
    registered_manager._migration_manager.validate_migration_path(
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
        manager._migration_manager.validate_migration_path(
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
    manager._migration_manager.validate_migration_path(
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
    manager._migration_manager.validate_migration_path(
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
        manager._migration_manager.validate_migration_path(
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
        manager._migration_manager.validate_migration_path(
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
    manager._migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(1, 5, 0)
    )
    manager._migration_manager.validate_migration_path(
        "User", ModelVersion(1, 5, 0), ModelVersion(2, 0, 0)
    )
    manager._migration_manager.validate_migration_path(
        "User", ModelVersion(2, 0, 0), ModelVersion(3, 0, 0)
    )
    manager._migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(3, 0, 0)
    )


def test_validate_migration_path_nonexistent_model(
    manager: ModelManager,
) -> None:
    """Test validate_migration_path with nonexistent model."""
    with pytest.raises(ModelNotFoundError, match="Model 'NonExistent' not found"):
        manager._migration_manager.validate_migration_path(
            "NonExistent", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)
        )


def test_validate_migration_path_nonexistent_from_version(
    manager: ModelManager,
    user_v1: type[BaseModel],
) -> None:
    """Test validate_migration_path with nonexistent source version."""
    manager.model("User", "1.0.0")(user_v1)

    with pytest.raises(ModelNotFoundError, match=r"Model 'User' version '2.0.0'"):
        manager._migration_manager.validate_migration_path(
            "User", ModelVersion(2, 0, 0), ModelVersion(1, 0, 0)
        )


def test_validate_migration_path_nonexistent_to_version(
    manager: ModelManager,
    user_v1: type[BaseModel],
) -> None:
    """Test validate_migration_path with nonexistent target version."""
    manager.model("User", "1.0.0")(user_v1)

    with pytest.raises(ModelNotFoundError, match=r"Model 'User' version '2.0.0'"):
        manager._migration_manager.validate_migration_path(
            "User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)
        )


def test_validate_migration_path_same_version(
    manager: ModelManager,
    user_v1: type[BaseModel],
) -> None:
    """Test validate_migration_path with same source and target."""
    manager.model("User", "1.0.0")(user_v1)

    # Should not raise
    manager._migration_manager.validate_migration_path(
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
        manager._migration_manager.validate_migration_path(
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
    manager._migration_manager.validate_migration_path(
        "User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0)
    )
    manager._migration_manager.validate_migration_path(
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

    manager._migration_manager.validate_migration_path(
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

    manager._migration_manager.validate_migration_path(
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
    manager._migration_manager.validate_migration_path(
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
        manager._migration_manager.validate_migration_path(
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
        manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

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
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")
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
        manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")


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
        manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, Exception)
    assert "Intentional error in list iteration" in str(exc_info.value.__cause__)


def test_auto_migration_with_field_aliases(manager: ModelManager) -> None:
    """Test that auto-migration handles field aliases correctly."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str = Field(alias="userId")
        user_name: str = Field(alias="userName")

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        user_id: str = Field(alias="userId")
        user_name: str = Field(alias="userName")
        email: str = "default@example.com"

    data: ModelData = {"userId": "123", "userName": "Alice"}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert result["userId"] == "123"
    assert result["userName"] == "Alice"
    assert result["email"] == "default@example.com"


def test_auto_migration_with_nested_model_aliases(manager: ModelManager) -> None:
    """Test that auto-migration handles aliases in nested models."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street_name: str = Field(alias="streetName")

    @manager.model("Address", "2.0.0", backward_compatible=True)
    class AddressV2(BaseModel):
        street_name: str = Field(alias="streetName")
        city_name: str = Field(alias="cityName", default="Unknown")

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str = Field(alias="userId")
        address: AddressV1

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        user_id: str = Field(alias="userId")
        address: AddressV2

    data: ModelData = {
        "userId": "456",
        "address": {"streetName": "123 Main St"},
    }
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert result["userId"] == "456"
    assert result["address"]["streetName"] == "123 Main St"
    assert result["address"]["cityName"] == "Unknown"


def test_auto_migration_with_both_alias_and_field_name_present(
    manager: ModelManager,
) -> None:
    """Test behavior when data contains both aliased and non-aliased versions."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str = Field(alias="userId")

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        user_id: str = Field(alias="userId")
        email: str = "default@example.com"

    data: ModelData = {"userId": "123", "user_id": "456"}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert result["user_id"] == "456"
    assert "userId" in result or "user_id" in result


def test_auto_migration_with_multiple_alias_types(
    manager: ModelManager,
) -> None:
    """Test field with alias, validation_alias, and serialization_alias all set."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str = Field(
            alias="userId",
            validation_alias="user_identifier",
            serialization_alias="userID",
        )

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        user_id: str = Field(
            alias="userId",
            validation_alias="user_identifier",
            serialization_alias="userID",
        )
        email: str = "default@example.com"

    data: ModelData = {"user_identifier": "789"}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert "user_identifier" in result or "userId" in result or "userID" in result
    assert "789" in str(result.values())


def test_auto_migration_with_validation_alias_choices(
    manager: ModelManager,
) -> None:
    """Test field with AliasChoices (non-string validation_alias)."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str = Field(validation_alias=AliasChoices("userId", "user_id"))

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        user_id: str = Field(validation_alias=AliasChoices("userId", "user_id"))
        email: str = "default@example.com"

    data: ModelData = {"userId": "999"}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert "email" in result


def test_auto_migration_none_value_vs_missing_field(
    manager: ModelManager,
) -> None:
    """Test distinction between explicit None and missing field."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str | None = None
        age: int | None = None

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str | None = "default_name"
        age: int | None = 999

    data: ModelData = {"name": None}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert result["name"] is None
    assert result["age"] == 999  # noqa: PLR2004


def test_auto_migration_alias_changes_between_versions(
    manager: ModelManager,
) -> None:
    """Test when field has different aliases in source vs target."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str = Field(alias="userId")

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        user_id: str = Field(alias="userIdentifier")  # Different alias

    data: ModelData = {"userId": "123"}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert "userIdentifier" in result or "userId" in result
    assert "123" in str(result.values())


def test_auto_migration_extra_field_conflicts_with_alias(
    manager: ModelManager,
) -> None:
    """Test extra field that has same name as a model field's alias."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str = Field(alias="userId")

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        user_id: str = Field(alias="userId")
        email: str = "default@example.com"

    data: ModelData = {"userId": "123", "user_id": "extra_value"}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert len(result) >= 2  # noqa: PLR2004


def test_auto_migration_empty_data(
    manager: ModelManager,
) -> None:
    """Test migration with empty data dictionary."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str = "default"

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str = "default"
        email: str = "default@example.com"

    data: ModelData = {}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert result["name"] == "default"
    assert result["email"] == "default@example.com"


def test_auto_migration_default_factory_returns_none(
    manager: ModelManager,
) -> None:
    """Test field where default_factory intentionally returns None."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        metadata: dict[str, Any] | None = Field(default_factory=lambda: None)

    data: ModelData = {"name": "Alice"}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert result["metadata"] is None


def test_auto_migration_nested_model_with_different_aliases(
    manager: ModelManager,
) -> None:
    """Test nested model where parent and child have different alias conventions."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street_name: str = Field(alias="streetName")

    @manager.model("Address", "2.0.0", backward_compatible=True)
    class AddressV2(BaseModel):
        street_name: str = Field(alias="street")  # Different alias
        city: str = "Unknown"

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_name: str = Field(alias="userName")
        address: AddressV1

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        user_name: str = Field(alias="name")  # Different alias
        address: AddressV2

    data: ModelData = {
        "userName": "Bob",
        "address": {"streetName": "123 Main St"},
    }
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert "userName" in result
    assert result["userName"] == "Bob"
    assert "address" in result
    assert "streetName" in result["address"]
    assert result["address"]["streetName"] == "123 Main St"
    assert result["address"]["city"] == "Unknown"


def test_auto_migration_field_name_takes_precedence_over_alias(
    manager: ModelManager,
) -> None:
    """Test documented behavior that field_name takes precedence over alias."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str = Field(alias="userId")

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        user_id: str = Field(alias="userId")
        email: str = "default@example.com"

    data: ModelData = {"userId": "from_alias", "user_id": "from_field_name"}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert (
        result.get("user_id") == "from_field_name"
        or result.get("userId") == "from_field_name"
    )
    if "userId" in result and "user_id" in result:
        assert result["userId"] == result["user_id"], (
            "Should not have conflicting values"
        )
    user_id_keys = [k for k in result if k in ("user_id", "userId")]
    assert len(user_id_keys) == 1, "Should only have one key for user_id field"


def test_auto_migration_preserves_extra_fields_with_aliases(
    manager: ModelManager,
) -> None:
    """Test that extra fields are preserved even when model uses aliases."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str = Field(alias="userId")

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        user_id: str = Field(alias="userId")
        email: str = "default@example.com"

    data: ModelData = {
        "userId": "123",
        "custom_field": "custom_value",
        "another_extra": 42,
    }
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert "userId" in result
    assert result["userId"] == "123"
    assert result["email"] == "default@example.com"
    assert result["custom_field"] == "custom_value"
    assert result["another_extra"] == 42  # noqa: PLR2004


def test_discriminated_union_basic_migration(
    manager: ModelManager,
) -> None:
    """Test basic discriminated union migration with two types."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    @manager.model("PetOwner", "1.0.0")
    class PetOwnerV1(BaseModel):
        name: str
        pet: CatV1 | DogV1 = Field(discriminator="type")

    @manager.model("Cat", "2.0.0", backward_compatible=True)
    class CatV2(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str
        favorite_toy: str = "ball"

    @manager.model("Dog", "2.0.0", backward_compatible=True)
    class DogV2(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str
        breed: str = "mixed"

    @manager.model("PetOwner", "2.0.0", backward_compatible=True)
    class PetOwnerV2(BaseModel):
        name: str
        pet: CatV2 | DogV2 = Field(discriminator="type")

    dog_data: ModelData = {"name": "Bob", "pet": {"type": "dog", "bark": "woof"}}
    result = manager._migration_manager.migrate(dog_data, "PetOwner", "1.0.0", "2.0.0")

    assert result["name"] == "Bob"
    assert result["pet"]["type"] == "dog"
    assert result["pet"]["bark"] == "woof"
    assert result["pet"]["breed"] == "mixed"
    assert "favorite_toy" not in result["pet"]


def test_discriminated_union_cat_migration(
    manager: ModelManager,
) -> None:
    """Test discriminated union correctly migrates cat type."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    @manager.model("PetOwner", "1.0.0")
    class PetOwnerV1(BaseModel):
        name: str
        pet: CatV1 | DogV1 = Field(discriminator="type")

    @manager.model("Cat", "2.0.0", backward_compatible=True)
    class CatV2(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str
        favorite_toy: str = "ball"

    @manager.model("Dog", "2.0.0", backward_compatible=True)
    class DogV2(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str
        breed: str = "mixed"

    @manager.model("PetOwner", "2.0.0", backward_compatible=True)
    class PetOwnerV2(BaseModel):
        name: str
        pet: CatV2 | DogV2 = Field(discriminator="type")

    cat_data: ModelData = {"name": "Alice", "pet": {"type": "cat", "meow": "loud"}}
    result = manager._migration_manager.migrate(cat_data, "PetOwner", "1.0.0", "2.0.0")

    assert result["name"] == "Alice"
    assert result["pet"]["type"] == "cat"
    assert result["pet"]["meow"] == "loud"
    assert result["pet"]["favorite_toy"] == "ball"
    assert "breed" not in result["pet"]


def test_discriminated_union_with_explicit_migrations(
    manager: ModelManager,
) -> None:
    """Test discriminated union where each type has explicit migration logic."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        sound: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        type: Literal["dog"] = "dog"
        sound: str

    @manager.model("PetOwner", "1.0.0")
    class PetOwnerV1(BaseModel):
        name: str
        pet: CatV1 | DogV1 = Field(discriminator="type")

    @manager.model("Cat", "2.0.0")
    class CatV2(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "2.0.0")
    class DogV2(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    @manager.model("PetOwner", "2.0.0", backward_compatible=True)
    class PetOwnerV2(BaseModel):
        name: str
        pet: CatV2 | DogV2 = Field(discriminator="type")

    @manager.migration("Cat", "1.0.0", "2.0.0")
    def migrate_cat(data: ModelData) -> ModelData:
        return {"type": "cat", "meow": data["sound"].upper()}

    @manager.migration("Dog", "1.0.0", "2.0.0")
    def migrate_dog(data: ModelData) -> ModelData:
        return {"type": "dog", "bark": data["sound"].lower()}

    dog_data: ModelData = {"name": "Bob", "pet": {"type": "dog", "sound": "WOOF"}}
    result = manager._migration_manager.migrate(dog_data, "PetOwner", "1.0.0", "2.0.0")

    assert result["pet"]["type"] == "dog"
    assert result["pet"]["bark"] == "woof"
    assert "meow" not in result["pet"]


def test_discriminated_union_three_types(
    manager: ModelManager,
) -> None:
    """Test discriminated union with more than two types."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    @manager.model("Bird", "1.0.0")
    class BirdV1(BaseModel):
        type: Literal["bird"] = "bird"
        chirp: str

    @manager.model("PetOwner", "1.0.0")
    class PetOwnerV1(BaseModel):
        name: str
        pet: CatV1 | DogV1 | BirdV1 = Field(discriminator="type")

    @manager.model("Cat", "2.0.0", backward_compatible=True)
    class CatV2(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str
        favorite_toy: str = "ball"

    @manager.model("Dog", "2.0.0", backward_compatible=True)
    class DogV2(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str
        breed: str = "mixed"

    @manager.model("Bird", "2.0.0", backward_compatible=True)
    class BirdV2(BaseModel):
        type: Literal["bird"] = "bird"
        chirp: str
        can_fly: bool = True

    @manager.model("PetOwner", "2.0.0", backward_compatible=True)
    class PetOwnerV2(BaseModel):
        name: str
        pet: CatV2 | DogV2 | BirdV2 = Field(discriminator="type")

    bird_data: ModelData = {
        "name": "Charlie",
        "pet": {"type": "bird", "chirp": "tweet"},
    }
    result = manager._migration_manager.migrate(bird_data, "PetOwner", "1.0.0", "2.0.0")

    assert result["pet"]["type"] == "bird"
    assert result["pet"]["chirp"] == "tweet"
    assert result["pet"]["can_fly"] is True
    assert "breed" not in result["pet"]
    assert "favorite_toy" not in result["pet"]


def test_discriminated_union_missing_discriminator_field(
    manager: ModelManager,
) -> None:
    """Test discriminated union when data lacks discriminator field.

    Falls back to treating as first type in union when discriminator is missing.
    """

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    @manager.model("PetOwner", "1.0.0")
    class PetOwnerV1(BaseModel):
        name: str
        pet: CatV1 | DogV1 = Field(discriminator="type")

    @manager.model("Cat", "2.0.0", backward_compatible=True)
    class CatV2(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str
        favorite_toy: str = "ball"

    @manager.model("Dog", "2.0.0", backward_compatible=True)
    class DogV2(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str
        breed: str = "mixed"

    @manager.model("PetOwner", "2.0.0", backward_compatible=True)
    class PetOwnerV2(BaseModel):
        name: str
        pet: CatV2 | DogV2 = Field(discriminator="type")

    invalid_data: ModelData = {
        "name": "Alice",
        "pet": {"meow": "loud"},
    }
    result = manager._migration_manager.migrate(
        invalid_data, "PetOwner", "1.0.0", "2.0.0"
    )

    # Falls back to non-discriminated behavior, migrates as Cat (first in union)
    assert result["pet"]["meow"] == "loud"
    assert result["pet"]["type"] == "cat"  # Default added
    assert result["pet"]["favorite_toy"] == "ball"  # Cat's default


def test_discriminated_union_nested_in_list(
    manager: ModelManager,
) -> None:
    """Test discriminated union types nested within a list."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    @manager.model("PetOwner", "1.0.0")
    class PetOwnerV1(BaseModel):
        name: str
        pets: list[Annotated[CatV1 | DogV1, Field(discriminator="type")]]

    @manager.model("Cat", "2.0.0", backward_compatible=True)
    class CatV2(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str
        favorite_toy: str = "ball"

    @manager.model("Dog", "2.0.0", backward_compatible=True)
    class DogV2(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str
        breed: str = "mixed"

    @manager.model("PetOwner", "2.0.0", backward_compatible=True)
    class PetOwnerV2(BaseModel):
        name: str
        pets: list[Annotated[CatV2 | DogV2, Field(discriminator="type")]]

    data: ModelData = {
        "name": "Alice",
        "pets": [
            {"type": "cat", "meow": "loud"},
            {"type": "dog", "bark": "woof"},
            {"type": "cat", "meow": "quiet"},
        ],
    }
    result = manager._migration_manager.migrate(data, "PetOwner", "1.0.0", "2.0.0")

    assert len(result["pets"]) == 3  # noqa: PLR2004
    assert result["pets"][0]["type"] == "cat"
    assert result["pets"][0]["favorite_toy"] == "ball"
    assert "breed" not in result["pets"][0]

    assert result["pets"][1]["type"] == "dog"
    assert result["pets"][1]["breed"] == "mixed"
    assert "favorite_toy" not in result["pets"][1]

    assert result["pets"][2]["type"] == "cat"
    assert result["pets"][2]["favorite_toy"] == "ball"


def test_discriminated_union_with_aliases(
    manager: ModelManager,
) -> None:
    """Test discriminated union where discriminator field uses aliases."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        pet_type: Literal["cat"] = Field("cat", alias="petType")
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        pet_type: Literal["dog"] = Field("dog", alias="petType")
        bark: str

    @manager.model("PetOwner", "1.0.0")
    class PetOwnerV1(BaseModel):
        name: str
        # Use field name as discriminator, data uses alias
        pet: CatV1 | DogV1 = Field(discriminator="pet_type")

    @manager.model("Cat", "2.0.0", backward_compatible=True)
    class CatV2(BaseModel):
        pet_type: Literal["cat"] = Field("cat", alias="petType")
        meow: str
        favorite_toy: str = "ball"

    @manager.model("Dog", "2.0.0", backward_compatible=True)
    class DogV2(BaseModel):
        pet_type: Literal["dog"] = Field("dog", alias="petType")
        bark: str
        breed: str = "mixed"

    @manager.model("PetOwner", "2.0.0", backward_compatible=True)
    class PetOwnerV2(BaseModel):
        name: str
        pet: CatV2 | DogV2 = Field(discriminator="pet_type")

    data: ModelData = {
        "name": "Bob",
        "pet": {
            "petType": "dog",  # Using alias in data
            "bark": "woof",
        },
    }
    result = manager._migration_manager.migrate(data, "PetOwner", "1.0.0", "2.0.0")

    assert result["pet"]["petType"] == "dog"
    assert result["pet"]["bark"] == "woof"
    assert result["pet"]["breed"] == "mixed"


def test_discriminated_union_chained_migrations(
    manager: ModelManager,
) -> None:
    """Test discriminated union through multiple version migrations."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    @manager.model("PetOwner", "1.0.0")
    class PetOwnerV1(BaseModel):
        pet: CatV1 | DogV1 = Field(discriminator="type")

    @manager.model("Cat", "2.0.0", backward_compatible=True)
    class CatV2(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str
        favorite_toy: str = "ball"

    @manager.model("Dog", "2.0.0", backward_compatible=True)
    class DogV2(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str
        breed: str = "mixed"

    @manager.model("PetOwner", "2.0.0", backward_compatible=True)
    class PetOwnerV2(BaseModel):
        pet: CatV2 | DogV2 = Field(discriminator="type")

    @manager.model("Cat", "3.0.0", backward_compatible=True)
    class CatV3(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str
        favorite_toy: str = "ball"
        indoor: bool = True

    @manager.model("Dog", "3.0.0", backward_compatible=True)
    class DogV3(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str
        breed: str = "mixed"
        trained: bool = False

    @manager.model("PetOwner", "3.0.0", backward_compatible=True)
    class PetOwnerV3(BaseModel):
        pet: CatV3 | DogV3 = Field(discriminator="type")

    data: ModelData = {"pet": {"type": "cat", "meow": "loud"}}
    result = manager._migration_manager.migrate(data, "PetOwner", "1.0.0", "3.0.0")

    assert result["pet"]["type"] == "cat"
    assert result["pet"]["meow"] == "loud"
    assert result["pet"]["favorite_toy"] == "ball"
    assert result["pet"]["indoor"] is True
    assert "breed" not in result["pet"]
    assert "trained" not in result["pet"]


def test_discriminated_union_preserves_extra_fields(
    manager: ModelManager,
) -> None:
    """Test that extra fields in discriminated union data are preserved."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    @manager.model("PetOwner", "1.0.0")
    class PetOwnerV1(BaseModel):
        name: str
        pet: CatV1 | DogV1 = Field(discriminator="type")

    @manager.model("Cat", "2.0.0", backward_compatible=True)
    class CatV2(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "2.0.0", backward_compatible=True)
    class DogV2(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    @manager.model("PetOwner", "2.0.0", backward_compatible=True)
    class PetOwnerV2(BaseModel):
        name: str
        pet: CatV2 | DogV2 = Field(discriminator="type")

    data: ModelData = {
        "name": "Alice",
        "pet": {
            "type": "cat",
            "meow": "loud",
            "custom_field": "custom_value",
            "extra_number": 42,
        },
    }
    result = manager._migration_manager.migrate(data, "PetOwner", "1.0.0", "2.0.0")

    assert result["pet"]["type"] == "cat"
    assert result["pet"]["meow"] == "loud"
    assert result["pet"]["custom_field"] == "custom_value"
    assert result["pet"]["extra_number"] == 42  # noqa: PLR2004


def test_migration_function_raises_exception(manager: ModelManager) -> None:
    """Test that exceptions in migration functions are properly wrapped."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def broken_migration(data: ModelData) -> ModelData:
        raise ValueError("Intentional migration error")

    data: ModelData = {"name": "Alice"}
    with pytest.raises(
        MigrationError,
        match=r"Migration failed for 'User': 1.0.0 → 2.0.0",
    ) as exc_info:
        manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert "Migration function raised: ValueError" in str(exc_info.value)
    assert "Intentional migration error" in str(exc_info.value)


def test_get_field_default_with_pydantic_undefined(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_field_default when default is PydanticUndefined."""
    field_info = FieldInfo(annotation=str, default=PydanticUndefined)
    sentinel = object()
    result = populated_migration_manager._get_field_default(field_info, sentinel)
    assert result is sentinel


def test_get_field_default_with_default_value(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_field_default when field has a default value."""
    field_info = FieldInfo(annotation=str, default="test_default")
    result = populated_migration_manager._get_field_default(field_info)
    assert result == "test_default"


def test_get_field_default_with_working_factory(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_field_default when default_factory works."""
    field_info = FieldInfo(annotation=list[str], default_factory=list)
    result = populated_migration_manager._get_field_default(field_info)
    assert result == []


def test_get_field_default_with_broken_factory(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_field_default when default_factory raises exception."""

    def broken_factory() -> list[str]:
        raise RuntimeError("Factory broken")

    field_info = FieldInfo(annotation=list[str])
    field_info.default_factory = broken_factory
    sentinel = object()
    result = populated_migration_manager._get_field_default(field_info, sentinel)
    assert result is sentinel


def test_find_field_value_with_field_name(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _find_field_value finds value using field name."""
    data: ModelData = {"user_id": "123", "name": "Alice"}
    key_to_field_name = {"user_id": "user_id", "name": "name"}

    value, key = populated_migration_manager._find_field_value(
        data, "user_id", key_to_field_name
    )

    assert value == "123"
    assert key == "user_id"


def test_find_field_value_with_alias(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _find_field_value finds value using alias."""
    data: ModelData = {"userId": "456", "name": "Bob"}
    key_to_field_name = {"userId": "user_id", "user_id": "user_id", "name": "name"}

    value, key = populated_migration_manager._find_field_value(
        data, "user_id", key_to_field_name
    )

    assert value == "456"
    assert key == "userId"


def test_find_field_value_not_found(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _find_field_value returns None when field not found."""
    data: ModelData = {"name": "Charlie"}
    key_to_field_name = {"name": "name"}

    value, key = populated_migration_manager._find_field_value(
        data, "email", key_to_field_name
    )

    assert value is None
    assert key is None


def test_migrate_single_field_no_value_no_default(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _migrate_single_field returns None when no value and no default."""
    data: ModelData = {"name": "Alice"}
    to_field_info = FieldInfo(annotation=str, default=PydanticUndefined)
    from_fields = {"name": FieldInfo(annotation=str)}
    key_to_field_name = {"name": "name"}

    result = populated_migration_manager._migrate_single_field(
        data, "email", to_field_info, from_fields, key_to_field_name
    )

    assert result is None


def test_migrate_field_value_nested_dict_without_model(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _migrate_field_value with nested dict that's not a model."""
    field_info = FieldInfo(annotation=dict[str, Any])
    value = {"nested": {"key": "value"}}

    result = populated_migration_manager._migrate_field_value(
        value, field_info, field_info
    )

    assert result == {"nested": {"key": "value"}}


def test_migrate_list_item_with_primitive(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _migrate_list_item with primitive value."""
    result = populated_migration_manager._migrate_list_item("string", None, None)
    assert result == "string"


def test_migrate_list_item_with_none_to_field(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _migrate_list_item with None to_field."""
    result = populated_migration_manager._migrate_list_item(
        {"key": "value"}, None, None
    )
    assert result == {"key": "value"}


def test_extract_list_item_field_with_none(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _extract_list_item_field with None field."""
    result = populated_migration_manager._extract_list_item_field(None)
    assert result is None


def test_extract_list_item_field_with_none_annotation(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _extract_list_item_field with field that has None annotation."""
    field_info = FieldInfo(annotation=None)
    result = populated_migration_manager._extract_list_item_field(field_info)
    assert result is None


def test_extract_list_item_field_non_list_origin(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _extract_list_item_field with non-list type."""
    field_info = FieldInfo(annotation=dict[str, str])
    result = populated_migration_manager._extract_list_item_field(field_info)
    assert result is None


def test_extract_list_item_field_no_args(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _extract_list_item_field with list that has no type args."""
    field_info = FieldInfo(annotation=list)
    result = populated_migration_manager._extract_list_item_field(field_info)
    assert result is None


def test_extract_list_item_field_with_discriminator(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _extract_list_item_field extracts discriminator from Annotated."""

    class DummyModel(BaseModel):
        type: Literal["test"] = "test"

    annotated_type = Annotated[DummyModel, Field(discriminator="type")]
    field_info = FieldInfo(annotation=list[annotated_type])

    result = populated_migration_manager._extract_list_item_field(field_info)

    assert result is not None
    assert result.annotation == DummyModel
    assert hasattr(result, "discriminator")
    assert result.discriminator == "type"


def test_try_extract_discriminated_model_no_discriminator(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _try_extract_discriminated_model with no discriminator."""
    field_info = FieldInfo(annotation=dict[str, Any])
    value: ModelData = {"key": "value"}

    result = populated_migration_manager._try_extract_discriminated_model(
        value, None, field_info
    )

    assert result is None


def test_get_discriminator_key_with_none(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_discriminator_key with None discriminator."""
    field_info = FieldInfo(annotation=str)
    result = populated_migration_manager._get_discriminator_key(field_info)
    assert result is None


def test_get_discriminator_key_with_string(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_discriminator_key with string discriminator."""
    field_info = FieldInfo(annotation=str)
    field_info.discriminator = "type"
    result = populated_migration_manager._get_discriminator_key(field_info)
    assert result == "type"


def test_get_discriminator_key_with_object_having_discriminator_attr(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_discriminator_key with object that has discriminator attribute."""

    class DiscriminatorObj:
        discriminator = "pet_type"

    field_info = FieldInfo(annotation=str)
    field_info.discriminator = DiscriminatorObj()  # type: ignore[assignment]
    result = populated_migration_manager._get_discriminator_key(field_info)
    assert result == "pet_type"


def test_get_discriminator_key_with_object_without_string_attr(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_discriminator_key with object that has non-string discriminator."""

    class DiscriminatorObj:
        discriminator = 123  # Not a string

    field_info = FieldInfo(annotation=str)
    field_info.discriminator = DiscriminatorObj()  # type: ignore[assignment]
    result = populated_migration_manager._get_discriminator_key(field_info)
    assert result is None


def test_find_discriminator_value_with_field_name(
    manager: ModelManager,
) -> None:
    """Test _find_discriminator_value finds value using field name."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        pet_type: Literal["cat"] = "cat"
        meow: str

    field_info = FieldInfo(annotation=CatV1)
    field_info.discriminator = "pet_type"
    value: ModelData = {"pet_type": "cat", "meow": "loud"}

    result = manager._migration_manager._find_discriminator_value(
        value, "pet_type", field_info
    )

    assert result == "cat"


def test_find_discriminator_value_with_alias(
    manager: ModelManager,
) -> None:
    """Test _find_discriminator_value finds value using alias."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        pet_type: Literal["cat"] = Field("cat", alias="petType")
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        pet_type: Literal["dog"] = Field("dog", alias="petType")
        bark: str

    field_info = FieldInfo(annotation=CatV1 | DogV1)  # type: ignore[arg-type]
    field_info.discriminator = "pet_type"
    value: ModelData = {"petType": "cat", "meow": "loud"}

    result = manager._migration_manager._find_discriminator_value(
        value, "pet_type", field_info
    )

    assert result == "cat"


def test_find_discriminator_value_with_serialization_alias(
    manager: ModelManager,
) -> None:
    """Test _find_discriminator_value finds value using serialization_alias."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        pet_type: Literal["cat"] = Field("cat", serialization_alias="petType")
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        pet_type: Literal["dog"] = Field("dog", serialization_alias="petType")
        bark: str

    field_info = FieldInfo(annotation=CatV1 | DogV1)  # type: ignore[arg-type]
    field_info.discriminator = "pet_type"
    value: ModelData = {"petType": "dog", "bark": "woof"}

    result = manager._migration_manager._find_discriminator_value(
        value, "pet_type", field_info
    )

    assert result == "dog"


def test_find_discriminator_value_with_validation_alias_string(
    manager: ModelManager,
) -> None:
    """Test _find_discriminator_value finds value using string validation_alias."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        pet_type: Literal["cat"] = Field("cat", validation_alias="petType")
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        pet_type: Literal["dog"] = Field("dog", validation_alias="petType")
        bark: str

    field_info = FieldInfo(annotation=CatV1 | DogV1)  # type: ignore[arg-type]
    field_info.discriminator = "pet_type"
    value: ModelData = {"petType": "cat", "meow": "loud"}

    result = manager._migration_manager._find_discriminator_value(
        value, "pet_type", field_info
    )

    assert result == "cat"


def test_find_discriminator_value_not_found(
    manager: ModelManager,
) -> None:
    """Test _find_discriminator_value returns None when not found."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        pet_type: Literal["cat"] = "cat"
        meow: str

    field_info = FieldInfo(annotation=CatV1)
    field_info.discriminator = "pet_type"
    value: ModelData = {"meow": "loud"}  # Missing discriminator

    result = manager._migration_manager._find_discriminator_value(
        value, "pet_type", field_info
    )

    assert result is None


def test_find_discriminated_type_no_union_members(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _find_discriminated_type with no union members."""
    field_info = FieldInfo(annotation=str)
    result = populated_migration_manager._find_discriminated_type(
        field_info, "type", "cat"
    )
    assert result is None


def test_find_discriminated_type_no_match(
    manager: ModelManager,
) -> None:
    """Test _find_discriminated_type when no type matches discriminator."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    field_info = FieldInfo(annotation=CatV1 | DogV1)  # type: ignore[arg-type]

    result = manager._migration_manager._find_discriminated_type(
        field_info,
        "type",
        "bird",  # No bird type
    )

    assert result is None


def test_get_union_members_with_none_annotation(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_union_members with None annotation."""
    field_info = FieldInfo(annotation=None)
    result = populated_migration_manager._get_union_members(field_info)
    assert result == []


def test_get_union_members_with_single_model(
    manager: ModelManager,
) -> None:
    """Test _get_union_members with single BaseModel (not a union)."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    field_info = FieldInfo(annotation=CatV1)
    result = manager._migration_manager._get_union_members(field_info)
    assert result == [CatV1]


def test_get_union_members_filters_none_type(
    manager: ModelManager,
) -> None:
    """Test _get_union_members filters out None type."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    field_info = FieldInfo(annotation=CatV1 | None)  # type: ignore
    result = manager._migration_manager._get_union_members(field_info)
    assert result == [CatV1]
    assert type(None) not in result  # type: ignore[comparison-overlap]


def test_is_union_type_with_union(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _is_union_type with Union type."""
    result = populated_migration_manager._is_union_type(Union)
    assert result is True


def test_is_union_type_with_pipe_syntax(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _is_union_type with | syntax (UnionType)."""
    union_type = str | int
    origin = get_origin(union_type)

    result = populated_migration_manager._is_union_type(origin)
    assert result is True


def test_is_union_type_with_non_union(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _is_union_type with non-union type."""
    result = populated_migration_manager._is_union_type(list)
    assert result is False


def test_is_union_type_handles_missing_uniontype(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _is_union_type gracefully handles missing UnionType attribute."""
    result = populated_migration_manager._is_union_type(dict)
    assert result is False


def test_model_matches_discriminator_field_not_in_model(
    manager: ModelManager,
) -> None:
    """Test _model_matches_discriminator when field not in model."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        meow: str  # No 'type' field

    result = manager._migration_manager._model_matches_discriminator(
        CatV1, "type", "cat"
    )

    assert result is False


def test_model_matches_discriminator_with_literal(
    manager: ModelManager,
) -> None:
    """Test _model_matches_discriminator with Literal annotation."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    result = manager._migration_manager._model_matches_discriminator(
        CatV1, "type", "cat"
    )

    assert result is True


def test_model_matches_discriminator_with_default_value(
    manager: ModelManager,
) -> None:
    """Test _model_matches_discriminator with default value match."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: str = "cat"  # Not a Literal, but has default
        meow: str

    result = manager._migration_manager._model_matches_discriminator(
        CatV1, "type", "cat"
    )

    assert result is True


def test_model_matches_discriminator_default_mismatch(
    manager: ModelManager,
) -> None:
    """Test _model_matches_discriminator with non-matching default."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: str = "dog"  # Mismatched default
        meow: str

    result = manager._migration_manager._model_matches_discriminator(
        CatV1, "type", "cat"
    )

    assert result is False


def test_literal_matches_value_with_none_annotation(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _literal_matches_value with None annotation."""
    result = populated_migration_manager._literal_matches_value(None, "cat")
    assert result is False


def test_literal_matches_value_with_non_literal(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _literal_matches_value with non-Literal annotation."""
    result = populated_migration_manager._literal_matches_value(str, "cat")
    assert result is False


def test_literal_matches_value_value_in_literal(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _literal_matches_value when value is in Literal."""
    result = populated_migration_manager._literal_matches_value(
        Literal["cat", "dog"], "cat"
    )
    assert result is True


def test_literal_matches_value_value_not_in_literal(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _literal_matches_value when value is not in Literal."""
    result = populated_migration_manager._literal_matches_value(
        Literal["cat", "dog"], "bird"
    )
    assert result is False


def test_get_model_type_from_field_with_none_origin(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_model_type_from_field with type that has no origin."""
    field_info = FieldInfo(annotation=int)
    result = populated_migration_manager._get_model_type_from_field(field_info)
    assert result is None


def test_get_model_type_from_field_with_generic_no_model_args(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_model_type_from_field with generic type but no BaseModel args."""
    field_info = FieldInfo(annotation=list[str])
    result = populated_migration_manager._get_model_type_from_field(field_info)
    assert result is None


def test_discriminated_union_in_nested_list_with_auto_migration(
    manager: ModelManager,
) -> None:
    """Test discriminated union in nested list with auto-migration."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    @manager.model("PetGroup", "1.0.0")
    class PetGroupV1(BaseModel):
        name: str
        pets: list[Annotated[CatV1 | DogV1, Field(discriminator="type")]]

    @manager.model("Owner", "1.0.0")
    class OwnerV1(BaseModel):
        groups: list[PetGroupV1]

    @manager.model("Cat", "2.0.0", backward_compatible=True)
    class CatV2(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str
        favorite_toy: str = "ball"

    @manager.model("Dog", "2.0.0", backward_compatible=True)
    class DogV2(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str
        breed: str = "mixed"

    @manager.model("PetGroup", "2.0.0", backward_compatible=True)
    class PetGroupV2(BaseModel):
        name: str
        pets: list[Annotated[CatV2 | DogV2, Field(discriminator="type")]]

    @manager.model("Owner", "2.0.0", backward_compatible=True)
    class OwnerV2(BaseModel):
        groups: list[PetGroupV2]

    data: ModelData = {
        "groups": [
            {
                "name": "Group A",
                "pets": [
                    {"type": "cat", "meow": "loud"},
                    {"type": "dog", "bark": "woof"},
                ],
            },
            {
                "name": "Group B",
                "pets": [
                    {"type": "dog", "bark": "arf"},
                ],
            },
        ]
    }
    result = manager._migration_manager.migrate(data, "Owner", "1.0.0", "2.0.0")

    assert len(result["groups"]) == 2  # noqa: PLR2004
    assert result["groups"][0]["pets"][0]["favorite_toy"] == "ball"
    assert result["groups"][0]["pets"][1]["breed"] == "mixed"
    assert result["groups"][1]["pets"][0]["breed"] == "mixed"


def test_auto_migrate_branch_used(manager: ModelManager) -> None:
    """Test that auto-migration path is used when no explicit migration exists."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        email: str = "default@example.com"

    data: ModelData = {"name": "Alice"}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert result["name"] == "Alice"
    assert result["email"] == "default@example.com"


def test_get_field_default_none_factory(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_field_default when default_factory is explicitly None."""
    field_info = FieldInfo(annotation=str, default=PydanticUndefined)
    field_info.default_factory = None
    sentinel = object()
    result = populated_migration_manager._get_field_default(field_info, sentinel)
    assert result is sentinel


def test_build_alias_map_with_all_alias_types(manager: ModelManager) -> None:
    """Test _build_alias_map handles all alias types."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str = Field(
            alias="userId",
            serialization_alias="userID",
            validation_alias="user_identifier",
        )
        name: str = Field(alias="userName")
        email: str  # No alias

    fields = UserV1.model_fields
    result = manager._migration_manager._build_alias_map(fields)

    assert result["user_id"] == "user_id"
    assert result["userId"] == "user_id"
    assert result["userID"] == "user_id"
    assert result["user_identifier"] == "user_id"
    assert result["name"] == "name"
    assert result["userName"] == "name"
    assert result["email"] == "email"


def test_build_alias_map_with_non_string_validation_alias(
    manager: ModelManager,
) -> None:
    """Test _build_alias_map skips non-string validation_alias."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str = Field(validation_alias=AliasChoices("userId", "user_id"))

    fields = UserV1.model_fields
    result = manager._migration_manager._build_alias_map(fields)

    assert result["user_id"] == "user_id"
    assert "userId" not in result


def test_migrate_single_field_uses_serialization_alias_for_output(
    manager: ModelManager,
) -> None:
    """Test _migrate_single_field uses serialization_alias for output key."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        email: str = Field("default@example.com", serialization_alias="emailAddress")

    data: ModelData = {"name": "Alice"}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert "emailAddress" in result
    assert result["emailAddress"] == "default@example.com"


def test_migrate_single_field_uses_alias_for_output_when_no_serialization_alias(
    manager: ModelManager,
) -> None:
    """Test _migrate_single_field uses alias when no serialization_alias."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0", backward_compatible=True)
    class UserV2(BaseModel):
        name: str
        email: str = Field("default@example.com", alias="emailAddr")

    data: ModelData = {"name": "Alice"}
    result = manager._migration_manager.migrate(data, "User", "1.0.0", "2.0.0")

    assert "emailAddr" in result
    assert result["emailAddr"] == "default@example.com"


def test_get_list_item_fields_from_field_is_none(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_list_item_fields when from_field is None."""
    to_field = FieldInfo(annotation=list[str])

    from_item, to_item = populated_migration_manager._get_list_item_fields(
        None, to_field
    )

    assert from_item is None
    assert to_item is not None


def test_get_list_item_fields_extract_returns_none(
    populated_migration_manager: MigrationManager,
) -> None:
    """Test _get_list_item_fields when extraction returns None."""
    from_field = FieldInfo(annotation=str)
    to_field = FieldInfo(annotation=str)

    from_item, to_item = populated_migration_manager._get_list_item_fields(
        from_field, to_field
    )

    assert from_item is None
    assert to_item is None


def test_extract_nested_model_info_discriminated_found(manager: ModelManager) -> None:
    """Test _extract_nested_model_info returns discriminated result first."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str

    @manager.model("Cat", "2.0.0", backward_compatible=True)
    class CatV2(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str
        favorite_toy: str = "ball"

    @manager.model("Dog", "2.0.0", backward_compatible=True)
    class DogV2(BaseModel):
        type: Literal["dog"] = "dog"
        bark: str
        breed: str = "mixed"

    from_field = FieldInfo(annotation=CatV1 | DogV1)  # type: ignore[arg-type]
    from_field.discriminator = "type"
    to_field = FieldInfo(annotation=CatV2 | DogV2)  # type: ignore[arg-type]
    to_field.discriminator = "type"

    value: ModelData = {"type": "cat", "meow": "loud"}

    info = manager._migration_manager._extract_nested_model_info(
        value, from_field, to_field
    )

    assert info is not None
    assert info[0] == "Cat"
    assert info[1] == ModelVersion(1, 0, 0)
    assert info[2] == ModelVersion(2, 0, 0)


def test_try_extract_discriminated_model_discriminator_value_none(
    manager: ModelManager,
) -> None:
    """Test _try_extract_discriminated_model when discriminator value not found."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    field_info = FieldInfo(annotation=CatV1)
    field_info.discriminator = "type"
    value: ModelData = {"meow": "loud"}

    result = manager._migration_manager._try_extract_discriminated_model(
        value, None, field_info
    )

    assert result is None


def test_try_extract_discriminated_model_type_not_basemodel(
    manager: ModelManager,
) -> None:
    """Test _try_extract_discriminated_model when type is not BaseModel."""

    class NotAModel:
        type: str = "not_model"

    field_info = FieldInfo(annotation=NotAModel | dict)  # type: ignore
    field_info.discriminator = "type"
    value: ModelData = {"type": "not_model"}

    result = manager._migration_manager._try_extract_discriminated_model(
        value, None, field_info
    )

    assert result is None


def test_try_extract_discriminated_model_not_registered(
    manager: ModelManager,
) -> None:
    """Test _try_extract_discriminated_model when model not registered."""

    class UnregisteredModel(BaseModel):
        type: Literal["unregistered"] = "unregistered"

    field_info = FieldInfo(annotation=UnregisteredModel)
    field_info.discriminator = "type"
    value: ModelData = {"type": "unregistered"}

    result = manager._migration_manager._try_extract_discriminated_model(
        value, None, field_info
    )

    assert result is None


def test_get_discriminated_source_version_from_model_not_basemodel(
    manager: ModelManager,
) -> None:
    """Test _get_discriminated_source_version when from model is not BaseModel."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    class NotAModel:
        type: str = "not_model"

    from_field = FieldInfo(annotation=NotAModel | dict)  # type: ignore
    from_field.discriminator = "type"

    result = manager._migration_manager._get_discriminated_source_version(
        from_field,
        "type",
        "cat",
        "Cat",
        ModelVersion(1, 0, 0),
    )

    assert result == ModelVersion(1, 0, 0)


def test_get_discriminated_source_version_model_not_registered(
    manager: ModelManager,
) -> None:
    """Test _get_discriminated_source_version when model not in registry."""

    class UnregisteredCat(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    from_field = FieldInfo(annotation=UnregisteredCat)
    from_field.discriminator = "type"

    result = manager._migration_manager._get_discriminated_source_version(
        from_field,
        "type",
        "cat",
        "Cat",
        ModelVersion(1, 0, 0),
    )

    assert result == ModelVersion(1, 0, 0)


def test_get_discriminated_source_version_different_model_name(
    manager: ModelManager,
) -> None:
    """Test _get_discriminated_source_version when model name doesn't match."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"] = "cat"
        meow: str

    from_field = FieldInfo(annotation=CatV1)
    from_field.discriminator = "type"

    result = manager._migration_manager._get_discriminated_source_version(
        from_field,
        "type",
        "cat",
        "Dog",  # Different model name
        ModelVersion(1, 0, 0),
    )

    assert result == ModelVersion(1, 0, 0)


def test_model_matches_discriminator_literal_no_match_but_default_undefined(
    manager: ModelManager,
) -> None:
    """Test _model_matches_discriminator when literal doesn't match and no default."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"]
        meow: str

    result = manager._migration_manager._model_matches_discriminator(
        CatV1,
        "type",
        "dog",
    )

    assert result is False


def test_model_matches_discriminator_literal_no_match_default_pydantic_undefined(
    manager: ModelManager,
) -> None:
    """Test when literal doesn't match and default is PydanticUndefined."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        type: Literal["cat"]
        meow: str

    CatV1.model_fields["type"].default = PydanticUndefined
    result = manager._migration_manager._model_matches_discriminator(
        CatV1, "type", "dog"
    )

    assert result is False


def test_complex_nested_discriminated_union_migration(
    manager: ModelManager,
) -> None:
    """Integration test covering multiple edge cases together."""

    @manager.model("Cat", "1.0.0")
    class CatV1(BaseModel):
        pet_type: Literal["cat"] = Field("cat", alias="petType")
        name: str

    @manager.model("Dog", "1.0.0")
    class DogV1(BaseModel):
        pet_type: Literal["dog"] = Field("dog", alias="petType")
        name: str

    @manager.model("PetStore", "1.0.0")
    class PetStoreV1(BaseModel):
        inventory: list[Annotated[CatV1 | DogV1, Field(discriminator="pet_type")]]

    @manager.model("Cat", "2.0.0", backward_compatible=True)
    class CatV2(BaseModel):
        pet_type: Literal["cat"] = Field("cat", alias="petType")
        name: str
        toys: list[str] = Field(default_factory=list, serialization_alias="toyList")

    @manager.model("Dog", "2.0.0", backward_compatible=True)
    class DogV2(BaseModel):
        pet_type: Literal["dog"] = Field("dog", alias="petType")
        name: str
        collar_color: str = Field("blue", alias="collarColor")

    @manager.model("PetStore", "2.0.0", backward_compatible=True)
    class PetStoreV2(BaseModel):
        inventory: list[Annotated[CatV2 | DogV2, Field(discriminator="pet_type")]]
        store_name: str = Field("Pet Paradise", serialization_alias="storeName")

    data: ModelData = {
        "inventory": [
            {"petType": "cat", "name": "Whiskers"},
            {"petType": "dog", "name": "Buddy"},
        ]
    }
    result = manager._migration_manager.migrate(data, "PetStore", "1.0.0", "2.0.0")

    assert result["storeName"] == "Pet Paradise"
    assert len(result["inventory"]) == 2  # noqa: PLR2004

    cat = result["inventory"][0]
    assert cat["petType"] == "cat"
    assert cat["toyList"] == []

    dog = result["inventory"][1]
    assert dog["petType"] == "dog"
    assert dog["collarColor"] == "blue"
