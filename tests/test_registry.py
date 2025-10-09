"""Test Registry."""

from typing import Any

import pytest
from pydantic import BaseModel

from pyrmute import ModelNotFoundError, ModelVersion
from pyrmute._registry import Registry


# Initialization tests
def test_registry_initialization() -> None:
    """Test Registry initializes with empty collections."""
    registry = Registry()
    assert len(registry._models) == 0
    assert len(registry._migrations) == 0
    assert len(registry._schema_generators) == 0
    assert len(registry._model_metadata) == 0
    assert len(registry._ref_enabled) == 0


# Registration tests
def test_register_model_with_string_version(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test registering a model with string version."""
    registry.register("User", "1.0.0")(user_v1)

    assert "User" in registry._models
    assert ModelVersion(1, 0, 0) in registry._models["User"]
    assert registry._models["User"][ModelVersion(1, 0, 0)] == user_v1


def test_register_model_with_model_version(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test registering a model with ModelVersion object."""
    version = ModelVersion(1, 0, 0)
    registry.register("User", version)(user_v1)

    assert registry._models["User"][version] == user_v1


def test_register_returns_decorator(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test that register returns a decorator function."""
    decorator = registry.register("User", "1.0.0")
    result = decorator(user_v1)
    assert result is user_v1


def test_register_decorator_pattern(registry: Registry) -> None:
    """Test using register as decorator with @ syntax."""

    @registry.register("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    assert registry._models["User"][ModelVersion(1, 0, 0)] == UserV1


def test_register_stores_metadata(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test that registration stores model metadata."""
    registry.register("User", "1.0.0")(user_v1)

    assert user_v1 in registry._model_metadata
    assert registry._model_metadata[user_v1] == ("User", ModelVersion(1, 0, 0))


def test_register_multiple_versions(
    registry: Registry,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
) -> None:
    """Test registering multiple versions of same model."""
    registry.register("User", "1.0.0")(user_v1)
    registry.register("User", "2.0.0")(user_v2)

    assert len(registry._models["User"]) == 2  # noqa: PLR2004
    assert registry._models["User"][ModelVersion(1, 0, 0)] == user_v1
    assert registry._models["User"][ModelVersion(2, 0, 0)] == user_v2


def test_register_different_models(registry: Registry) -> None:
    """Test registering different model types."""

    @registry.register("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @registry.register("Product", "1.0.0")
    class ProductV1(BaseModel):
        title: str

    assert "User" in registry._models
    assert "Product" in registry._models
    assert len(registry._models) == 2  # noqa: PLR2004


def test_register_with_schema_generator(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test registering model with custom schema generator."""

    def custom_generator(model: type[BaseModel]) -> dict[str, Any]:
        return {"custom": True}

    registry.register("User", "1.0.0", schema_generator=custom_generator)(user_v1)

    assert "User" in registry._schema_generators
    assert ModelVersion(1, 0, 0) in registry._schema_generators["User"]
    assert (
        registry._schema_generators["User"][ModelVersion(1, 0, 0)] == custom_generator
    )


def test_register_without_schema_generator(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test registering model without schema generator."""
    registry.register("User", "1.0.0")(user_v1)

    # Should not add entry to _schema_generators
    assert (
        "User" not in registry._schema_generators
        or ModelVersion(1, 0, 0) not in registry._schema_generators["User"]
    )


def test_register_with_enable_ref_true(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test registering model with enable_ref=True."""
    registry.register("User", "1.0.0", enable_ref=True)(user_v1)

    assert "User" in registry._ref_enabled
    assert ModelVersion(1, 0, 0) in registry._ref_enabled["User"]


def test_register_with_enable_ref_false(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test registering model with enable_ref=False (default)."""
    registry.register("User", "1.0.0", enable_ref=False)(user_v1)

    assert ModelVersion(1, 0, 0) not in registry._ref_enabled.get("User", set())


def test_register_default_enable_ref_false(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test that enable_ref defaults to False."""
    registry.register("User", "1.0.0")(user_v1)

    assert ModelVersion(1, 0, 0) not in registry._ref_enabled.get("User", set())


# Get model tests
def test_get_model_with_string_version(
    populated_registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test getting model with string version."""
    model = populated_registry.get_model("User", "1.0.0")
    assert model == user_v1


def test_get_model_with_model_version(
    populated_registry: Registry,
    user_v2: type[BaseModel],
) -> None:
    """Test getting model with ModelVersion object."""
    model = populated_registry.get_model("User", ModelVersion(2, 0, 0))
    assert model == user_v2


def test_get_model_all_versions(
    populated_registry: Registry,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
    user_v3: type[BaseModel],
) -> None:
    """Test getting all registered versions."""
    assert populated_registry.get_model("User", "1.0.0") == user_v1
    assert populated_registry.get_model("User", "2.0.0") == user_v2
    assert populated_registry.get_model("User", "3.0.0") == user_v3


def test_get_model_not_found(registry: Registry) -> None:
    """Test getting non-existent model raises ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError, match=r"Model 'NonExistent' not found"):
        registry.get_model("NonExistent", "1.0.0")


def test_get_model_version_not_found(
    populated_registry: Registry,
) -> None:
    """Test getting non-existent version raises ModelNotFoundError."""
    with pytest.raises(
        ModelNotFoundError, match=r"Model 'User' version '9.9.9' not found"
    ):
        populated_registry.get_model("User", "9.9.9")


# Get latest tests
def test_get_latest_single_version(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test getting latest with single version."""
    registry.register("User", "1.0.0")(user_v1)

    latest = registry.get_latest("User")
    assert latest == user_v1


def test_get_latest_multiple_versions(
    populated_registry: Registry,
    user_v4: type[BaseModel],
) -> None:
    """Test getting latest with multiple versions."""
    latest = populated_registry.get_latest("User")
    assert latest == user_v4


def test_get_latest_unordered_registration(registry: Registry) -> None:
    """Test getting latest when versions registered out of order."""

    @registry.register("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @registry.register("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @registry.register("User", "1.5.0")
    class UserV15(BaseModel):
        name: str
        age: int

    latest = registry.get_latest("User")
    assert latest == UserV2


def test_get_latest_not_found(registry: Registry) -> None:
    """Test getting latest for non-existent model raises ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError, match="Model 'NonExistent' not found"):
        registry.get_latest("NonExistent")


# Get versions tests
def test_get_versions_single(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test getting versions list with single version."""
    registry.register("User", "1.0.0")(user_v1)

    versions = registry.get_versions("User")
    assert versions == [ModelVersion(1, 0, 0)]


def test_get_versions_multiple(
    populated_registry: Registry,
) -> None:
    """Test getting versions list with multiple versions."""
    versions = populated_registry.get_versions("User")
    assert len(versions) == 4  # noqa: PLR2004
    assert ModelVersion(1, 0, 0) in versions
    assert ModelVersion(2, 0, 0) in versions
    assert ModelVersion(3, 0, 0) in versions
    assert ModelVersion(4, 0, 0) in versions


def test_get_versions_sorted(registry: Registry) -> None:
    """Test that versions are returned in sorted order."""

    @registry.register("User", "2.0.0")
    class UserV2(BaseModel):
        name: str

    @registry.register("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @registry.register("User", "1.5.0")
    class UserV15(BaseModel):
        name: str

    versions = registry.get_versions("User")
    assert versions == [
        ModelVersion(1, 0, 0),
        ModelVersion(1, 5, 0),
        ModelVersion(2, 0, 0),
    ]


def test_get_versions_not_found(registry: Registry) -> None:
    """Test getting versions for non-existent model raises ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError, match="Model 'NonExistent' not found"):
        registry.get_versions("NonExistent")


# List models tests
def test_list_models_empty(registry: Registry) -> None:
    """Test listing models when registry is empty."""
    models = registry.list_models()
    assert models == []


def test_list_models_single(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test listing models with single model."""
    registry.register("User", "1.0.0")(user_v1)

    models = registry.list_models()
    assert models == ["User"]


def test_list_models_multiple(registry: Registry) -> None:
    """Test listing multiple different models."""

    @registry.register("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @registry.register("Product", "1.0.0")
    class ProductV1(BaseModel):
        title: str

    @registry.register("Order", "1.0.0")
    class OrderV1(BaseModel):
        order_id: str

    models = registry.list_models()
    assert len(models) == 3  # noqa: PLR2004
    assert "User" in models
    assert "Product" in models
    assert "Order" in models


def test_list_models_same_name_multiple_versions(
    populated_registry: Registry,
) -> None:
    """Test listing models counts each name once."""
    models = populated_registry.list_models()
    assert models.count("User") == 1
    assert len(models) == 1


# Get model info tests
def test_get_model_info_registered(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test getting info for registered model."""
    registry.register("User", "1.0.0")(user_v1)

    info = registry.get_model_info(user_v1)
    assert info == ("User", ModelVersion(1, 0, 0))


def test_get_model_info_multiple_versions(
    populated_registry: Registry,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
) -> None:
    """Test getting info for different versions."""
    info_v1 = populated_registry.get_model_info(user_v1)
    info_v2 = populated_registry.get_model_info(user_v2)

    assert info_v1 == ("User", ModelVersion(1, 0, 0))
    assert info_v2 == ("User", ModelVersion(2, 0, 0))


def test_get_model_info_not_registered(registry: Registry) -> None:
    """Test getting info for unregistered model returns None."""

    class UnregisteredModel(BaseModel):
        field: str

    info = registry.get_model_info(UnregisteredModel)
    assert info is None


def test_get_model_info_different_models(registry: Registry) -> None:
    """Test getting info for different model types."""

    @registry.register("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @registry.register("Product", "1.0.0")
    class ProductV1(BaseModel):
        title: str

    user_info = registry.get_model_info(UserV1)
    product_info = registry.get_model_info(ProductV1)

    assert user_info == ("User", ModelVersion(1, 0, 0))
    assert product_info == ("Product", ModelVersion(1, 0, 0))


# Is ref enabled tests
def test_is_ref_enabled_true_with_string(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test is_ref_enabled with string version when enabled."""
    registry.register("User", "1.0.0", enable_ref=True)(user_v1)

    assert registry.is_ref_enabled("User", "1.0.0") is True


def test_is_ref_enabled_true_with_model_version(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test is_ref_enabled with ModelVersion when enabled."""
    registry.register("User", "1.0.0", enable_ref=True)(user_v1)

    assert registry.is_ref_enabled("User", ModelVersion(1, 0, 0)) is True


def test_is_ref_enabled_false_explicit(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test is_ref_enabled when explicitly set to False."""
    registry.register("User", "1.0.0", enable_ref=False)(user_v1)

    assert registry.is_ref_enabled("User", "1.0.0") is False


def test_is_ref_enabled_false_default(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test is_ref_enabled defaults to False."""
    registry.register("User", "1.0.0")(user_v1)

    assert registry.is_ref_enabled("User", "1.0.0") is False


def test_is_ref_enabled_unregistered_model(registry: Registry) -> None:
    """Test is_ref_enabled for unregistered model."""
    assert registry.is_ref_enabled("NonExistent", "1.0.0") is False


def test_is_ref_enabled_unregistered_version(
    registry: Registry,
    user_v1: type[BaseModel],
) -> None:
    """Test is_ref_enabled for unregistered version."""
    registry.register("User", "1.0.0", enable_ref=True)(user_v1)

    assert registry.is_ref_enabled("User", "2.0.0") is False


def test_is_ref_enabled_mixed_versions(registry: Registry) -> None:
    """Test is_ref_enabled with mixed settings across versions."""

    @registry.register("User", "1.0.0", enable_ref=True)
    class UserV1(BaseModel):
        name: str

    @registry.register("User", "2.0.0", enable_ref=False)
    class UserV2(BaseModel):
        name: str
        email: str

    @registry.register("User", "3.0.0")
    class UserV3(BaseModel):
        name: str
        email: str
        age: int

    assert registry.is_ref_enabled("User", "1.0.0") is True
    assert registry.is_ref_enabled("User", "2.0.0") is False
    assert registry.is_ref_enabled("User", "3.0.0") is False


# Integration tests
def test_full_registration_workflow(registry: Registry) -> None:
    """Test complete registration and retrieval workflow."""

    @registry.register("User", "1.0.0", enable_ref=True)
    class UserV1(BaseModel):
        name: str

    # Verify registration
    assert "User" in registry.list_models()
    assert ModelVersion(1, 0, 0) in registry.get_versions("User")

    # Verify retrieval
    retrieved = registry.get_model("User", "1.0.0")
    assert retrieved == UserV1

    # Verify metadata
    info = registry.get_model_info(UserV1)
    assert info == ("User", ModelVersion(1, 0, 0))

    # Verify ref enabled
    assert registry.is_ref_enabled("User", "1.0.0") is True


def test_multiple_models_workflow(registry: Registry) -> None:
    """Test workflow with multiple models and versions."""

    @registry.register("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @registry.register("User", "2.0.0", enable_ref=True)
    class UserV2(BaseModel):
        name: str
        email: str

    @registry.register("Product", "1.0.0")
    class ProductV1(BaseModel):
        title: str

    # List all models
    models = registry.list_models()
    assert len(models) == 2  # noqa: PLR2004
    assert "User" in models
    assert "Product" in models

    # Get User versions
    user_versions = registry.get_versions("User")
    assert len(user_versions) == 2  # noqa: PLR2004

    # Get latest User
    latest_user = registry.get_latest("User")
    assert latest_user == UserV2

    # Check ref enabled only for v2
    assert registry.is_ref_enabled("User", "1.0.0") is False
    assert registry.is_ref_enabled("User", "2.0.0") is True


def test_migrations_storage(registry: Registry) -> None:
    """Test that migrations dictionary is properly initialized."""
    # Migrations dict should exist but be empty initially
    assert isinstance(registry._migrations, dict)
    assert len(registry._migrations) == 0

    registry._migrations["User"][(ModelVersion(1, 0, 0), ModelVersion(2, 0, 0))] = (
        lambda x: x
    )

    assert len(registry._migrations["User"]) == 1
