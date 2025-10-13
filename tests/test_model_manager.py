"""Tests the ModelManager."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError
from pydantic.json_schema import GenerateJsonSchema

from pyrmute import (
    JsonSchema,
    JsonSchemaMode,
    MigrationHook,
    MigrationTestCase,
    ModelData,
    ModelManager,
    ModelVersion,
)
from pyrmute.schema_config import SchemaConfig


# Initialization tests
def test_manager_initialization(manager: ModelManager) -> None:
    """Test ModelManager initializes with required components."""
    assert manager._registry is not None
    assert manager._migration_manager is not None
    assert manager._schema_manager is not None


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
    def bad_migration(data: ModelData) -> ModelData:
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
    model = registered_manager.get_latest("User")
    assert model == user_v2


# Type-safe migration tests
def test_migrate_as_returns_correct_type(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_as returns instance of specified type."""
    result = registered_manager.migrate_as(
        {"name": "Alice"}, "User", "1.0.0", "2.0.0", user_v2
    )
    assert isinstance(result, user_v2)
    assert result.name == "Alice"  # type: ignore
    assert result.email == "unknown@example.com"  # type: ignore


def test_migrate_as_with_model_versions(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_as with ModelVersion objects."""
    result = registered_manager.migrate_as(
        {"name": "Bob"},
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
        user_v2,
    )
    assert isinstance(result, user_v2)
    assert result.name == "Bob"  # type: ignore


def test_migrate_as_validates_data(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_as validates data against target type."""

    @registered_manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: ModelData) -> ModelData:
        return {"name": data["name"]}  # Missing required 'email'

    with pytest.raises(ValidationError):
        registered_manager.migrate_as(
            {"name": "Invalid"}, "User", "1.0.0", "2.0.0", user_v2
        )


def test_migrate_as_consistency_with_migrate(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_as produces same result as migrate."""
    data = {"name": "Consistency"}

    result_migrate = registered_manager.migrate(data, "User", "1.0.0", "2.0.0")
    result_migrate_as = registered_manager.migrate_as(
        data, "User", "1.0.0", "2.0.0", user_v2
    )

    assert result_migrate.model_dump() == result_migrate_as.model_dump()


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
    def bad_migration(data: ModelData) -> ModelData:
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


# Basic batch migration tests
def test_migrate_batch_empty_list(registered_manager: ModelManager) -> None:
    """Test migrate_batch with empty list returns empty list."""
    result = registered_manager.migrate_batch([], "User", "1.0.0", "2.0.0")
    assert result == []


def test_migrate_batch_single_item(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch with single item."""
    result = registered_manager.migrate_batch(
        [{"name": "Alice"}],
        "User",
        "1.0.0",
        "2.0.0",
    )
    assert len(result) == 1
    assert isinstance(result[0], user_v2)
    assert result[0] == user_v2(name="Alice", email="unknown@example.com")


def test_migrate_batch_multiple_items(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch with multiple items."""
    data = [
        {"name": "Alice"},
        {"name": "Bob"},
        {"name": "Charlie"},
    ]
    result = registered_manager.migrate_batch(data, "User", "1.0.0", "2.0.0")
    assert len(result) == 3  # noqa: PLR2004
    assert all(isinstance(item, user_v2) for item in result)
    assert result[0].name == "Alice"  # type: ignore
    assert result[1].name == "Bob"  # type: ignore
    assert result[2].name == "Charlie"  # type: ignore


def test_migrate_batch_preserves_order(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch preserves input order."""
    data = [{"name": f"User{i}"} for i in range(10)]
    result = registered_manager.migrate_batch(data, "User", "1.0.0", "2.0.0")
    for i, item in enumerate(result):
        assert item.name == f"User{i}"  # type: ignore


def test_migrate_batch_with_string_versions(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch with string version parameters."""
    result = registered_manager.migrate_batch(
        [{"name": "Alice"}],
        "User",
        "1.0.0",
        "2.0.0",
    )
    assert isinstance(result[0], user_v2)


def test_migrate_batch_with_model_versions(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch with ModelVersion objects."""
    result = registered_manager.migrate_batch(
        [{"name": "Bob"}],
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
    )
    assert isinstance(result[0], user_v2)


def test_migrate_batch_validation_catches_invalid(
    registered_manager: ModelManager,
) -> None:
    """Test that validation catches invalid data in batch."""

    @registered_manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: ModelData) -> ModelData:
        return {"name": data["name"]}  # Missing required 'email'

    with pytest.raises(ValidationError):
        registered_manager.migrate_batch(
            [{"name": "Invalid"}],
            "User",
            "1.0.0",
            "2.0.0",
        )


def test_migrate_batch_stops_on_first_validation_error(
    registered_manager: ModelManager,
) -> None:
    """Test that batch migration stops on first validation error."""

    @registered_manager.migration("User", "1.0.0", "2.0.0")
    def selective_bad_migration(data: ModelData) -> ModelData:
        if data["name"] == "Bad":
            return {"name": data["name"]}  # Missing email
        return {**data, "email": "test@example.com"}

    data = [
        {"name": "Good"},
        {"name": "Bad"},
        {"name": "Good"},
    ]

    with pytest.raises(ValidationError):
        registered_manager.migrate_batch(data, "User", "1.0.0", "2.0.0")


# Type-safe batch migration tests
def test_migrate_batch_as_empty_list(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch_as with empty list returns empty list."""
    result = registered_manager.migrate_batch_as([], "User", "1.0.0", "2.0.0", user_v2)
    assert result == []


def test_migrate_batch_as_single_item(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch_as with single item."""
    result = registered_manager.migrate_batch_as(
        [{"name": "Alice"}],
        "User",
        "1.0.0",
        "2.0.0",
        user_v2,
    )
    assert len(result) == 1
    assert isinstance(result[0], user_v2)
    assert result[0].name == "Alice"  # type: ignore


def test_migrate_batch_as_multiple_items(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch_as with multiple items."""
    data = [
        {"name": "Alice"},
        {"name": "Bob"},
        {"name": "Charlie"},
    ]
    result = registered_manager.migrate_batch_as(
        data, "User", "1.0.0", "2.0.0", user_v2
    )
    assert len(result) == 3  # noqa: PLR2004
    assert all(isinstance(item, user_v2) for item in result)
    assert result[0].name == "Alice"  # type: ignore
    assert result[1].name == "Bob"  # type: ignore
    assert result[2].name == "Charlie"  # type: ignore


def test_migrate_batch_as_parallel(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch_as with parallel processing."""
    data = [{"name": f"User{i}"} for i in range(5)]
    result = registered_manager.migrate_batch_as(
        data,
        "User",
        "1.0.0",
        "2.0.0",
        user_v2,
        parallel=True,
    )
    assert len(result) == 5  # noqa: PLR2004
    assert all(isinstance(item, user_v2) for item in result)


def test_migrate_batch_as_parallel_with_max_workers(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch_as with parallel and max_workers."""
    data = [{"name": f"User{i}"} for i in range(10)]
    result = registered_manager.migrate_batch_as(
        data,
        "User",
        "1.0.0",
        "2.0.0",
        user_v2,
        parallel=True,
        max_workers=2,
    )
    assert len(result) == 10  # noqa: PLR2004
    assert all(isinstance(item, user_v2) for item in result)


def test_migrate_batch_as_parallel_with_processes(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch_as with process-based parallelism."""
    data = [{"name": f"User{i}"} for i in range(5)]
    result = registered_manager.migrate_batch_as(
        data,
        "User",
        "1.0.0",
        "2.0.0",
        user_v2,
        parallel=True,
        use_processes=True,
    )
    assert len(result) == 5  # noqa: PLR2004
    assert all(isinstance(item, user_v2) for item in result)


def test_migrate_batch_as_consistency_with_migrate_batch(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch_as produces same results as migrate_batch."""
    data = [{"name": "Alice"}, {"name": "Bob"}]

    result_migrate_batch = registered_manager.migrate_batch(
        data, "User", "1.0.0", "2.0.0"
    )
    result_migrate_batch_as = registered_manager.migrate_batch_as(
        data, "User", "1.0.0", "2.0.0", user_v2
    )

    assert len(result_migrate_batch) == len(result_migrate_batch_as)
    for item1, item2 in zip(result_migrate_batch, result_migrate_batch_as, strict=True):
        assert item1.model_dump() == item2.model_dump()


# Parallel migration tests
def test_migrate_batch_parallel_basic(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch with parallel=True."""
    data = [{"name": f"User{i}"} for i in range(5)]
    result = registered_manager.migrate_batch(
        data,
        "User",
        "1.0.0",
        "2.0.0",
        parallel=True,
    )
    assert len(result) == 5  # noqa: PLR2004
    assert all(isinstance(item, user_v2) for item in result)


def test_migrate_batch_parallel_with_max_workers(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch with parallel and max_workers."""
    data = [{"name": f"User{i}"} for i in range(10)]
    result = registered_manager.migrate_batch(
        data,
        "User",
        "1.0.0",
        "2.0.0",
        parallel=True,
        max_workers=2,
    )
    assert len(result) == 10  # noqa: PLR2004


def test_migrate_batch_parallel_with_processes(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch with process-based parallelism."""
    data = [{"name": f"User{i}"} for i in range(5)]
    result = registered_manager.migrate_batch(
        data,
        "User",
        "1.0.0",
        "2.0.0",
        parallel=True,
        use_processes=True,
    )
    assert len(result) == 5  # noqa: PLR2004


def test_migrate_batch_parallel_preserves_order(
    registered_manager: ModelManager,
) -> None:
    """Test that parallel migration preserves input order."""
    data = [{"name": f"User{i}"} for i in range(20)]
    result = registered_manager.migrate_batch(
        data,
        "User",
        "1.0.0",
        "2.0.0",
        parallel=True,
    )
    for i, item in enumerate(result):
        assert item.name == f"User{i}"  # type: ignore


def test_migrate_batch_parallel_empty_list(
    registered_manager: ModelManager,
) -> None:
    """Test parallel migration with empty list."""
    result = registered_manager.migrate_batch(
        [],
        "User",
        "1.0.0",
        "2.0.0",
        parallel=True,
    )
    assert result == []


# migrate_batch_data tests
def test_migrate_batch_data_returns_dicts(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch_data returns list of dictionaries."""
    data = [{"name": "Alice"}, {"name": "Bob"}]
    result = registered_manager.migrate_batch_data(data, "User", "1.0.0", "2.0.0")
    assert len(result) == 2  # noqa: PLR2004
    assert all(isinstance(item, dict) for item in result)
    assert result[0] == {"name": "Alice", "email": "unknown@example.com"}
    assert result[1] == {"name": "Bob", "email": "unknown@example.com"}


def test_migrate_batch_data_empty_list(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch_data with empty list."""
    result = registered_manager.migrate_batch_data([], "User", "1.0.0", "2.0.0")
    assert result == []


def test_migrate_batch_data_with_model_versions(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch_data with ModelVersion objects."""
    result = registered_manager.migrate_batch_data(
        [{"name": "Alice"}],
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
    )
    assert isinstance(result[0], dict)


def test_migrate_batch_data_does_not_validate(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch_data returns dicts even if invalid."""

    @registered_manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: ModelData) -> ModelData:
        return {"name": data["name"]}  # Missing required 'email'

    result = registered_manager.migrate_batch_data(
        [{"name": "Invalid"}],
        "User",
        "1.0.0",
        "2.0.0",
    )
    assert isinstance(result[0], dict)
    assert result[0] == {"name": "Invalid"}


def test_migrate_batch_data_parallel(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch_data with parallel processing."""
    data = [{"name": f"User{i}"} for i in range(5)]
    result = registered_manager.migrate_batch_data(
        data,
        "User",
        "1.0.0",
        "2.0.0",
        parallel=True,
    )
    assert len(result) == 5  # noqa: PLR2004
    assert all(isinstance(item, dict) for item in result)


def test_migrate_batch_data_parallel_with_processes(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch_data with process-based parallelism."""
    data = [{"name": f"User{i}"} for i in range(5)]
    result = registered_manager.migrate_batch_data(
        data,
        "User",
        "1.0.0",
        "2.0.0",
        parallel=True,
        use_processes=True,
    )
    assert len(result) == 5  # noqa: PLR2004


def test_migrate_batch_and_migrate_batch_data_consistency(
    registered_manager: ModelManager,
) -> None:
    """Test that migrate_batch and migrate_batch_data are consistent."""
    data = [{"name": "Alice"}, {"name": "Bob"}]

    dict_results = registered_manager.migrate_batch_data(data, "User", "1.0.0", "2.0.0")
    model_results = registered_manager.migrate_batch(data, "User", "1.0.0", "2.0.0")

    assert len(dict_results) == len(model_results)
    for dict_result, model_result in zip(dict_results, model_results, strict=False):
        assert model_result.model_dump() == dict_result


# Streaming migration tests
def test_migrate_batch_streaming_empty_list(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch_streaming with empty list."""
    result = list(
        registered_manager.migrate_batch_streaming([], "User", "1.0.0", "2.0.0")
    )
    assert result == []


def test_migrate_batch_streaming_single_chunk(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch_streaming with data smaller than chunk size."""
    data = [{"name": f"User{i}"} for i in range(5)]
    result = list(
        registered_manager.migrate_batch_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=10,
        )
    )
    assert len(result) == 5  # noqa: PLR2004
    assert all(isinstance(item, user_v2) for item in result)


def test_migrate_batch_streaming_multiple_chunks(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch_streaming with multiple chunks."""
    data = [{"name": f"User{i}"} for i in range(25)]
    result = list(
        registered_manager.migrate_batch_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=10,
        )
    )
    assert len(result) == 25  # noqa: PLR2004


def test_migrate_batch_streaming_preserves_order(
    registered_manager: ModelManager,
) -> None:
    """Test that streaming migration preserves input order."""
    data = [{"name": f"User{i}"} for i in range(30)]
    result = list(
        registered_manager.migrate_batch_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=10,
        )
    )
    for i, item in enumerate(result):
        assert item.name == f"User{i}"  # type: ignore


def test_migrate_batch_streaming_exact_chunk_size(
    registered_manager: ModelManager,
) -> None:
    """Test streaming with data size exactly matching chunk size."""
    data = [{"name": f"User{i}"} for i in range(10)]
    result = list(
        registered_manager.migrate_batch_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=10,
        )
    )
    assert len(result) == 10  # noqa: PLR2004


def test_migrate_batch_streaming_with_generator(
    registered_manager: ModelManager,
) -> None:
    """Test streaming with generator input."""

    def data_generator() -> Iterable[dict[str, str]]:
        for i in range(15):
            yield {"name": f"User{i}"}

    result = list(
        registered_manager.migrate_batch_streaming(
            data_generator(),
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=5,
        )
    )
    assert len(result) == 15  # noqa: PLR2004


def test_migrate_batch_streaming_custom_chunk_size(
    registered_manager: ModelManager,
) -> None:
    """Test streaming with custom chunk size."""
    data = [{"name": f"User{i}"} for i in range(7)]
    result = list(
        registered_manager.migrate_batch_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=3,
        )
    )
    assert len(result) == 7  # noqa: PLR2004


def test_migrate_batch_streaming_with_model_versions(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test streaming with ModelVersion objects."""
    data = [{"name": "Alice"}]
    result = list(
        registered_manager.migrate_batch_streaming(
            data,
            "User",
            ModelVersion(1, 0, 0),
            ModelVersion(2, 0, 0),
            chunk_size=10,
        )
    )
    assert isinstance(result[0], user_v2)


def test_migrate_batch_large_dataset(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch with larger dataset."""
    data = [{"name": f"User{i}"} for i in range(1000)]
    result = registered_manager.migrate_batch(data, "User", "1.0.0", "2.0.0")
    assert len(result) == 1000  # noqa: PLR2004


def test_migrate_batch_parallel_large_dataset(
    registered_manager: ModelManager,
) -> None:
    """Test parallel migration with larger dataset."""
    data = [{"name": f"User{i}"} for i in range(1000)]
    result = registered_manager.migrate_batch(
        data,
        "User",
        "1.0.0",
        "2.0.0",
        parallel=True,
        max_workers=4,
    )
    assert len(result) == 1000  # noqa: PLR2004


def test_migrate_batch_streaming_large_dataset(
    registered_manager: ModelManager,
) -> None:
    """Test streaming migration with larger dataset."""
    data = [{"name": f"User{i}"} for i in range(1000)]
    result = list(
        registered_manager.migrate_batch_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=100,
        )
    )
    assert len(result) == 1000  # noqa: PLR2004


def test_migrate_batch_data_streaming_empty_list(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch_data_streaming with empty list."""
    result = list(
        registered_manager.migrate_batch_data_streaming([], "User", "1.0.0", "2.0.0")
    )
    assert result == []


def test_migrate_batch_data_streaming_returns_dicts(
    registered_manager: ModelManager,
) -> None:
    """Test migrate_batch_data_streaming returns dictionaries."""
    data = [{"name": f"User{i}"} for i in range(5)]
    result = list(
        registered_manager.migrate_batch_data_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=10,
        )
    )
    assert len(result) == 5  # noqa: PLR2004
    assert all(isinstance(item, dict) for item in result)
    assert all(item["email"] == "unknown@example.com" for item in result)


def test_migrate_batch_data_streaming_multiple_chunks(
    registered_manager: ModelManager,
) -> None:
    """Test data streaming with multiple chunks."""
    data = [{"name": f"User{i}"} for i in range(25)]
    result = list(
        registered_manager.migrate_batch_data_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=10,
        )
    )
    assert len(result) == 25  # noqa: PLR2004
    assert all(isinstance(item, dict) for item in result)


def test_migrate_batch_data_streaming_preserves_order(
    registered_manager: ModelManager,
) -> None:
    """Test that data streaming preserves input order."""
    data = [{"name": f"User{i}"} for i in range(30)]
    result = list(
        registered_manager.migrate_batch_data_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=10,
        )
    )
    for i, item in enumerate(result):
        assert item["name"] == f"User{i}"


def test_migrate_batch_data_streaming_does_not_validate(
    registered_manager: ModelManager,
) -> None:
    """Test data streaming returns dicts even if invalid."""

    @registered_manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: ModelData) -> ModelData:
        return {"name": data["name"]}  # Missing required 'email'

    data = [{"name": "Invalid"}]
    result = list(
        registered_manager.migrate_batch_data_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=10,
        )
    )
    assert isinstance(result[0], dict)
    assert result[0] == {"name": "Invalid"}


def test_migrate_batch_data_streaming_with_generator(
    registered_manager: ModelManager,
) -> None:
    """Test data streaming with generator input."""

    def data_generator() -> Iterable[ModelData]:
        for i in range(15):
            yield {"name": f"User{i}"}

    result = list(
        registered_manager.migrate_batch_data_streaming(
            data_generator(),
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=5,
        )
    )
    assert len(result) == 15  # noqa: PLR2004
    assert all(isinstance(item, dict) for item in result)


def test_migrate_batch_data_streaming_custom_chunk_size(
    registered_manager: ModelManager,
) -> None:
    """Test data streaming with custom chunk size."""
    data = [{"name": f"User{i}"} for i in range(7)]
    result = list(
        registered_manager.migrate_batch_data_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=3,
        )
    )
    assert len(result) == 7  # noqa: PLR2004


def test_migrate_batch_data_streaming_with_model_versions(
    registered_manager: ModelManager,
) -> None:
    """Test data streaming with ModelVersion objects."""
    data = [{"name": "Alice"}]
    result = list(
        registered_manager.migrate_batch_data_streaming(
            data,
            "User",
            ModelVersion(1, 0, 0),
            ModelVersion(2, 0, 0),
            chunk_size=10,
        )
    )
    assert isinstance(result[0], dict)


def test_migrate_batch_streaming_and_data_streaming_consistency(
    registered_manager: ModelManager,
) -> None:
    """Test consistency between streaming variants."""
    data = [{"name": f"User{i}"} for i in range(10)]

    dict_results = list(
        registered_manager.migrate_batch_data_streaming(
            data, "User", "1.0.0", "2.0.0", chunk_size=3
        )
    )
    model_results = list(
        registered_manager.migrate_batch_streaming(
            data, "User", "1.0.0", "2.0.0", chunk_size=3
        )
    )

    assert len(dict_results) == len(model_results)
    for dict_result, model_result in zip(dict_results, model_results, strict=False):
        assert model_result.model_dump() == dict_result


# Type-safe streaming migration tests
def test_migrate_batch_streaming_as_empty_list(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch_streaming_as with empty list."""
    result = list(
        registered_manager.migrate_batch_streaming_as(
            [], "User", "1.0.0", "2.0.0", user_v2
        )
    )
    assert result == []


def test_migrate_batch_streaming_as_single_chunk(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch_streaming_as with data smaller than chunk size."""
    data = [{"name": f"User{i}"} for i in range(5)]
    result = list(
        registered_manager.migrate_batch_streaming_as(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            user_v2,
            chunk_size=10,
        )
    )
    assert len(result) == 5  # noqa: PLR2004
    assert all(isinstance(item, user_v2) for item in result)


def test_migrate_batch_streaming_as_multiple_chunks(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test migrate_batch_streaming_as with multiple chunks."""
    data = [{"name": f"User{i}"} for i in range(25)]
    result = list(
        registered_manager.migrate_batch_streaming_as(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            user_v2,
            chunk_size=10,
        )
    )
    assert len(result) == 25  # noqa: PLR2004
    assert all(isinstance(item, user_v2) for item in result)


def test_migrate_batch_streaming_as_preserves_order(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test that streaming_as migration preserves input order."""
    data = [{"name": f"User{i}"} for i in range(30)]
    result = list(
        registered_manager.migrate_batch_streaming_as(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            user_v2,
            chunk_size=10,
        )
    )
    for i, item in enumerate(result):
        assert item.name == f"User{i}"  # type: ignore


def test_migrate_batch_streaming_as_with_generator(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test streaming_as with generator input."""

    def data_generator() -> Iterable[dict[str, str]]:
        for i in range(15):
            yield {"name": f"User{i}"}

    result = list(
        registered_manager.migrate_batch_streaming_as(
            data_generator(),
            "User",
            "1.0.0",
            "2.0.0",
            user_v2,
            chunk_size=5,
        )
    )
    assert len(result) == 15  # noqa: PLR2004
    assert all(isinstance(item, user_v2) for item in result)


def test_migrate_batch_streaming_as_consistency_with_streaming(
    registered_manager: ModelManager,
    user_v2: type[BaseModel],
) -> None:
    """Test consistency between streaming_as and streaming variants."""
    data = [{"name": f"User{i}"} for i in range(10)]

    result_streaming = list(
        registered_manager.migrate_batch_streaming(
            data, "User", "1.0.0", "2.0.0", chunk_size=3
        )
    )
    result_streaming_as = list(
        registered_manager.migrate_batch_streaming_as(
            data, "User", "1.0.0", "2.0.0", user_v2, chunk_size=3
        )
    )

    assert len(result_streaming) == len(result_streaming_as)
    for item1, item2 in zip(result_streaming, result_streaming_as, strict=True):
        assert item1.model_dump() == item2.model_dump()


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


def test_has_migration_path_returns_true_when_valid(
    registered_manager: ModelManager,
) -> None:
    """Test has_migration_path returns True for valid paths."""
    assert registered_manager.has_migration_path("User", "1.0.0", "2.0.0") is True


def test_has_migration_path_returns_false_when_invalid(
    manager: ModelManager,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
) -> None:
    """Test has_migration_path returns False for invalid paths."""
    manager.model("User", "1.0.0")(user_v1)
    manager.model("User", "2.0.0")(user_v2)

    assert manager.has_migration_path("User", "1.0.0", "2.0.0") is False


def test_has_migration_path_returns_false_for_nonexistent_model(
    manager: ModelManager,
) -> None:
    """Test has_migration_path returns False for nonexistent models."""
    assert manager.has_migration_path("NonExistent", "1.0.0", "2.0.0") is False


def test_has_migration_path_accepts_string_versions(
    registered_manager: ModelManager,
) -> None:
    """Test has_migration_path accepts string versions."""
    assert registered_manager.has_migration_path("User", "1.0.0", "2.0.0") is True


def test_has_migration_path_accepts_model_versions(
    registered_manager: ModelManager,
) -> None:
    """Test has_migration_path accepts ModelVersion objects."""
    result = registered_manager.has_migration_path(
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
    )
    assert result is True


# Tests for validate_data
def test_validate_data_returns_true_for_valid_data(
    registered_manager: ModelManager,
) -> None:
    """Test validate_data returns True for valid data."""
    data = {"name": "Alice"}
    assert registered_manager.validate_data(data, "User", "1.0.0") is True


def test_validate_data_returns_false_for_missing_required_field(
    registered_manager: ModelManager,
) -> None:
    """Test validate_data returns False when required field is missing."""
    data = {"name": "Alice"}
    assert registered_manager.validate_data(data, "User", "2.0.0") is False


def test_validate_data_returns_false_for_wrong_type(
    registered_manager: ModelManager,
) -> None:
    """Test validate_data returns False when field has wrong type."""
    data = {"name": 123}  # Should be string
    assert registered_manager.validate_data(data, "User", "1.0.0") is False


def test_validate_data_returns_true_with_optional_field(
    manager: ModelManager,
) -> None:
    """Test validate_data returns True when optional field is omitted."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        email: str | None = None

    data = {"name": "Alice"}
    assert manager.validate_data(data, "User", "1.0.0") is True


def test_validate_data_returns_true_with_optional_field_present(
    manager: ModelManager,
) -> None:
    """Test validate_data returns True when optional field is provided."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        email: str | None = None

    data = {"name": "Alice", "email": "alice@example.com"}
    assert manager.validate_data(data, "User", "1.0.0") is True


def test_validate_data_returns_true_with_default_value(
    manager: ModelManager,
) -> None:
    """Test validate_data returns True when field with default is omitted."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        status: str = "active"

    data = {"name": "Alice"}
    assert manager.validate_data(data, "User", "1.0.0") is True


def test_validate_data_returns_false_for_empty_dict(
    registered_manager: ModelManager,
) -> None:
    """Test validate_data returns False for empty dict when fields required."""
    assert registered_manager.validate_data({}, "User", "1.0.0") is False


def test_validate_data_returns_true_for_all_fields_optional(
    manager: ModelManager,
) -> None:
    """Test validate_data returns True for empty dict when all fields optional."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str | None = None
        email: str | None = None

    assert manager.validate_data({}, "User", "1.0.0") is True


def test_validate_data_with_string_version(
    registered_manager: ModelManager,
) -> None:
    """Test validate_data with string version parameter."""
    data = {"name": "Alice"}
    assert registered_manager.validate_data(data, "User", "1.0.0") is True


def test_validate_data_with_model_version(
    registered_manager: ModelManager,
) -> None:
    """Test validate_data with ModelVersion object."""
    data = {"name": "Alice"}
    assert registered_manager.validate_data(data, "User", ModelVersion(1, 0, 0)) is True


def test_validate_data_different_versions(
    registered_manager: ModelManager,
) -> None:
    """Test validate_data with different versions of same model."""
    data_v1 = {"name": "Alice"}
    data_v2 = {"name": "Alice", "email": "alice@example.com"}

    assert registered_manager.validate_data(data_v1, "User", "1.0.0") is True
    assert registered_manager.validate_data(data_v1, "User", "2.0.0") is False
    assert registered_manager.validate_data(data_v2, "User", "2.0.0") is True


def test_validate_data_with_field_constraints(
    manager: ModelManager,
) -> None:
    """Test validate_data respects Field constraints."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        age: int = Field(ge=0, le=120)

    assert manager.validate_data({"age": 25}, "User", "1.0.0") is True
    assert manager.validate_data({"age": -5}, "User", "1.0.0") is False
    assert manager.validate_data({"age": 150}, "User", "1.0.0") is False


def test_validate_data_with_string_constraints(
    manager: ModelManager,
) -> None:
    """Test validate_data respects string Field constraints."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str = Field(min_length=1, max_length=50)

    assert manager.validate_data({"name": "Alice"}, "User", "1.0.0") is True
    assert manager.validate_data({"name": ""}, "User", "1.0.0") is False
    assert manager.validate_data({"name": "x" * 100}, "User", "1.0.0") is False


def test_validate_data_with_nested_dict(
    manager: ModelManager,
) -> None:
    """Test validate_data with nested dictionary."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: dict[str, str]

    data = {"name": "Alice", "address": {"city": "NYC", "state": "NY"}}
    assert manager.validate_data(data, "User", "1.0.0") is True


def test_validate_data_with_list(
    manager: ModelManager,
) -> None:
    """Test validate_data with list field."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        tags: list[str]

    data = {"name": "Alice", "tags": ["admin", "user"]}
    assert manager.validate_data(data, "User", "1.0.0") is True


def test_validate_data_with_wrong_list_type(
    manager: ModelManager,
) -> None:
    """Test validate_data returns False for wrong list element type."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        tags: list[str]

    data = {"name": "Alice", "tags": [1, 2, 3]}  # Should be strings
    assert manager.validate_data(data, "User", "1.0.0") is False


def test_validate_data_with_extra_fields(
    manager: ModelManager,
) -> None:
    """Test validate_data with extra fields (default: ignore)."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    data = {"name": "Alice", "extra": "data"}
    assert manager.validate_data(data, "User", "1.0.0") is True


def test_validate_data_returns_false_for_nonexistent_model(
    manager: ModelManager,
) -> None:
    """Test validate_data returns False for nonexistent model."""
    data = {"name": "Alice"}
    assert manager.validate_data(data, "NonExistent", "1.0.0") is False


def test_validate_data_returns_false_for_nonexistent_version(
    manager: ModelManager,
    user_v1: type[BaseModel],
) -> None:
    """Test validate_data returns False for nonexistent version."""
    manager.model("User", "1.0.0")(user_v1)
    data = {"name": "Alice"}
    assert manager.validate_data(data, "User", "9.9.9") is False


def test_validate_data_returns_false_for_none_data(
    registered_manager: ModelManager,
) -> None:
    """Test validate_data returns False for None data."""
    assert registered_manager.validate_data(None, "User", "1.0.0") is False  # type: ignore


def test_validate_data_returns_false_for_non_dict_data(
    registered_manager: ModelManager,
) -> None:
    """Test validate_data returns False for non-dict data."""
    assert registered_manager.validate_data("invalid", "User", "1.0.0") is False  # type: ignore
    assert registered_manager.validate_data(123, "User", "1.0.0") is False  # type: ignore
    assert registered_manager.validate_data(["list"], "User", "1.0.0") is False  # type: ignore


def test_validate_data_with_different_models(
    manager: ModelManager,
) -> None:
    """Test validate_data works correctly with different models."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("Product", "1.0.0")
    class ProductV1(BaseModel):
        title: str
        price: float

    assert manager.validate_data({"name": "Alice"}, "User", "1.0.0") is True
    assert manager.validate_data({"name": "Alice"}, "Product", "1.0.0") is False
    assert (
        manager.validate_data({"title": "Widget", "price": 9.99}, "Product", "1.0.0")
        is True
    )


def test_validate_data_after_migration(
    registered_manager: ModelManager,
) -> None:
    """Test validate_data works with migrated data."""
    old_data = {"name": "Alice"}
    migrated_data = registered_manager.migrate_data(old_data, "User", "1.0.0", "2.0.0")

    assert registered_manager.validate_data(old_data, "User", "1.0.0") is True
    assert registered_manager.validate_data(old_data, "User", "2.0.0") is False
    assert registered_manager.validate_data(migrated_data, "User", "2.0.0") is True


def test_validate_data_before_migration(
    registered_manager: ModelManager,
) -> None:
    """Test validate_data to check if migration is needed."""
    data = {"name": "Alice"}

    if not registered_manager.validate_data(data, "User", "2.0.0"):
        migrated = registered_manager.migrate(data, "User", "1.0.0", "2.0.0")
        assert (
            registered_manager.validate_data(migrated.model_dump(), "User", "2.0.0")
            is True
        )


# Basic test_migration tests
def test_test_migration_with_tuples(registered_manager: ModelManager) -> None:
    """Test test_migration with tuple format."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
            ({"name": "Bob"}, {"name": "Bob", "email": "unknown@example.com"}),
        ],
    )

    assert len(results) == 2  # noqa: PLR2004
    assert results.all_passed is True


def test_test_migration_with_test_case_objects(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration with MigrationTestCase objects."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            MigrationTestCase(
                source={"name": "Alice"},
                target={"name": "Alice", "email": "unknown@example.com"},
                description="Alice migration",
            ),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is True


def test_test_migration_mixed_formats(registered_manager: ModelManager) -> None:
    """Test test_migration with mixed tuple and object formats."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
            MigrationTestCase(
                source={"name": "Bob"},
                target={"name": "Bob", "email": "unknown@example.com"},
            ),
        ],
    )

    assert len(results) == 2  # noqa: PLR2004
    assert results.all_passed is True


def test_test_migration_single_test_case(registered_manager: ModelManager) -> None:
    """Test test_migration with single test case."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is True


def test_test_migration_empty_test_cases(registered_manager: ModelManager) -> None:
    """Test test_migration with empty test cases list."""
    results = registered_manager.test_migration("User", "1.0.0", "2.0.0", test_cases=[])

    assert len(results) == 0
    assert results.all_passed is True


# Version parameter tests
def test_test_migration_with_string_versions(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration with string version parameters."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
        ],
    )

    assert results.all_passed is True


def test_test_migration_with_model_versions(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration with ModelVersion objects."""
    results = registered_manager.test_migration(
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
        ],
    )

    assert results.all_passed is True


def test_test_migration_with_mixed_version_types(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration with mixed version types."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        ModelVersion(2, 0, 0),
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
        ],
    )

    assert results.all_passed is True


# Validation tests
def test_test_migration_detects_mismatch(registered_manager: ModelManager) -> None:
    """Test test_migration detects output mismatch."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "wrong@example.com"}),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is False
    assert len(results.failures) == 1
    assert results.failures[0].error == "Output mismatch"


def test_test_migration_detects_missing_field(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration detects missing field in output."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice"}),
        ],
    )

    assert results.all_passed is False


def test_test_migration_detects_extra_field(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration detects extra field in output."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            (
                {"name": "Alice"},
                {"name": "Alice", "email": "unknown@example.com", "extra": "field"},
            ),
        ],
    )

    assert results.all_passed is False


def test_test_migration_detects_wrong_value(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration detects wrong field value."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Bob", "email": "unknown@example.com"}),
        ],
    )

    assert results.all_passed is False


# Smoke testing (no expected output)
def test_test_migration_without_expected(registered_manager: ModelManager) -> None:
    """Test test_migration without expected output (smoke test)."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            MigrationTestCase(source={"name": "Alice"}, target=None),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is True


def test_test_migration_multiple_smoke_tests(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration with multiple smoke tests."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            MigrationTestCase(source={"name": "Alice"}, target=None),
            MigrationTestCase(source={"name": "Bob"}, target=None),
            MigrationTestCase(source={"name": "Charlie"}, target=None),
        ],
    )

    assert len(results) == 3  # noqa: PLR2004
    assert results.all_passed is True


def test_test_migration_mixed_smoke_and_validation(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration with mix of smoke and validation tests."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            MigrationTestCase(source={"name": "Alice"}, target=None),
            ({"name": "Bob"}, {"name": "Bob", "email": "unknown@example.com"}),
        ],
    )

    assert len(results) == 2  # noqa: PLR2004
    assert results.all_passed is True


# Exception handling tests
def test_test_migration_catches_migration_exception(manager: ModelManager) -> None:
    """Test test_migration catches and reports migration exceptions."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: ModelData) -> ModelData:
        raise ValueError("Migration failed")

    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "alice@example.com"}),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is False
    assert results.failures[0].error
    assert "Migration failed" in results.failures[0].error


def test_test_migration_catches_key_error(manager: ModelManager) -> None:
    """Test test_migration catches KeyError in migration."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: ModelData) -> ModelData:
        return {**data, "email": data["nonexistent"]}

    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "alice@example.com"}),
        ],
    )

    assert results.all_passed is False
    assert results.failures[0].error
    assert "nonexistent" in results.failures[0].error


def test_test_migration_continues_after_exception(manager: ModelManager) -> None:
    """Test test_migration continues testing after exception."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def conditional_migration(data: ModelData) -> ModelData:
        if data["name"] == "Alice":
            raise ValueError("Alice not allowed")
        return {**data, "email": "unknown@example.com"}

    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
            ({"name": "Bob"}, {"name": "Bob", "email": "unknown@example.com"}),
        ],
    )

    assert len(results) == 2  # noqa: PLR2004
    assert results.all_passed is False
    assert len(results.failures) == 1


# Description preservation tests
def test_test_migration_preserves_descriptions(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration preserves test case descriptions."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            MigrationTestCase(
                source={"name": "Alice"},
                target={"name": "Alice", "email": "unknown@example.com"},
                description="Test Alice",
            ),
        ],
    )

    assert results.results[0].test_case.description == "Test Alice"


def test_test_migration_tuple_has_empty_description(
    registered_manager: ModelManager,
) -> None:
    """Test that tuple format results in empty description."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
        ],
    )

    assert results.results[0].test_case.description == ""


# Multiple test cases tests
def test_test_migration_multiple_passing(registered_manager: ModelManager) -> None:
    """Test test_migration with multiple passing tests."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
            ({"name": "Bob"}, {"name": "Bob", "email": "unknown@example.com"}),
            ({"name": "Charlie"}, {"name": "Charlie", "email": "unknown@example.com"}),
        ],
    )

    assert len(results) == 3  # noqa: PLR2004
    assert results.all_passed is True
    assert len(results.failures) == 0


def test_test_migration_multiple_failures(manager: ModelManager) -> None:
    """Test test_migration with multiple failures."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "default@example.com"}

    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "alice@example.com"}),
            ({"name": "Bob"}, {"name": "Bob", "email": "bob@example.com"}),
            ({"name": "Charlie"}, {"name": "Charlie", "email": "default@example.com"}),
        ],
    )

    assert len(results) == 3  # noqa: PLR2004
    assert results.all_passed is False
    assert len(results.failures) == 2  # noqa: PLR2004


def test_test_migration_mixed_pass_fail(manager: ModelManager) -> None:
    """Test test_migration with mix of passing and failing tests."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": f"{data['name'].lower()}@example.com"}

    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "alice@example.com"}),
            ({"name": "Bob"}, {"name": "Bob", "email": "wrong@example.com"}),
            ({"name": "Charlie"}, {"name": "Charlie", "email": "charlie@example.com"}),
        ],
    )

    assert len(results) == 3  # noqa: PLR2004
    assert results.all_passed is False
    assert len(results.failures) == 1


# Chained migration tests
def test_test_migration_chain_through_versions(manager: ModelManager) -> None:
    """Test test_migration works with chained migrations."""

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
        return {**data, "email": "default@example.com"}

    @manager.migration("User", "2.0.0", "3.0.0")
    def migrate_2_to_3(data: ModelData) -> ModelData:
        return {**data, "age": 25}

    results = manager.test_migration(
        "User",
        "1.0.0",
        "3.0.0",
        test_cases=[
            (
                {"name": "Alice"},
                {"name": "Alice", "email": "default@example.com", "age": 25},
            ),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is True


def test_test_migration_multi_hop_chain(manager: ModelManager) -> None:
    """Test test_migration with longer migration chain."""

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

    @manager.model("User", "4.0.0")
    class UserV4(BaseModel):
        name: str
        email: str
        age: int
        active: bool

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate_1_to_2(data: ModelData) -> ModelData:
        return {**data, "email": "default@example.com"}

    @manager.migration("User", "2.0.0", "3.0.0")
    def migrate_2_to_3(data: ModelData) -> ModelData:
        return {**data, "age": 0}

    @manager.migration("User", "3.0.0", "4.0.0")
    def migrate_3_to_4(data: ModelData) -> ModelData:
        return {**data, "active": True}

    results = manager.test_migration(
        "User",
        "1.0.0",
        "4.0.0",
        test_cases=[
            (
                {"name": "Alice"},
                {
                    "name": "Alice",
                    "email": "default@example.com",
                    "age": 0,
                    "active": True,
                },
            ),
        ],
    )

    assert results.all_passed is True


# Actual output preservation tests
def test_test_migration_preserves_actual_on_failure(
    registered_manager: ModelManager,
) -> None:
    """Test that actual output is preserved even on failure."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "wrong@example.com"}),
        ],
    )

    assert results.failures[0].actual == {
        "name": "Alice",
        "email": "unknown@example.com",
    }


def test_test_migration_preserves_actual_on_success(
    registered_manager: ModelManager,
) -> None:
    """Test that actual output is preserved on success."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
        ],
    )

    assert results.results[0].actual == {
        "name": "Alice",
        "email": "unknown@example.com",
    }


def test_test_migration_preserves_actual_on_exception(
    manager: ModelManager,
) -> None:
    """Test that actual is empty dict on exception."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: ModelData) -> ModelData:
        raise ValueError("Failed")

    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "alice@example.com"}),
        ],
    )

    assert results.failures[0].actual == {}


# Edge case tests
def test_test_migration_with_complex_data(registered_manager: ModelManager) -> None:
    """Test test_migration with complex nested data."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            (
                {"name": "Alice", "metadata": {"key": "value"}},
                {
                    "name": "Alice",
                    "email": "unknown@example.com",
                    "metadata": {"key": "value"},
                },
            ),
        ],
    )

    assert results.all_passed is True


def test_test_migration_with_none_values(manager: ModelManager) -> None:
    """Test test_migration handles None values correctly."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str | None

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str | None
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "default@example.com"}

    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": None}, {"name": None, "email": "default@example.com"}),
        ],
    )

    assert results.all_passed is True


def test_test_migration_same_version(manager: ModelManager) -> None:
    """Test test_migration with same source and target version."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    results = manager.test_migration(
        "User",
        "1.0.0",
        "1.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice"}),
        ],
    )

    assert results.all_passed is True


# Custom generators for testing
class CustomTestGenerator(GenerateJsonSchema):
    """Test generator that adds custom metadata."""

    def generate(
        self, schema: Mapping[str, Any], mode: JsonSchemaMode = "validation"
    ) -> JsonSchema:
        """Generate."""
        json_schema = super().generate(schema, mode=mode)
        json_schema["x-test-generator"] = True
        json_schema["$schema"] = self.schema_dialect
        return json_schema


class AnotherTestGenerator(GenerateJsonSchema):
    """Another test generator."""

    def generate(
        self, schema: Mapping[str, Any], mode: JsonSchemaMode = "validation"
    ) -> JsonSchema:
        """Generate."""
        json_schema = super().generate(schema, mode=mode)
        json_schema["x-another"] = True
        return json_schema


def simple_callable_generator(model: type[BaseModel]) -> JsonSchema:
    """Simple callable generator for testing."""
    schema = model.model_json_schema()
    schema["x-simple"] = True
    return schema


# Initialization with default_schema_config tests
def test_model_manager_init_with_default_schema_config() -> None:
    """Test ModelManager initialization with default schema config."""
    config = SchemaConfig(
        schema_generator=CustomTestGenerator,
        mode="validation",
    )
    manager = ModelManager(default_schema_config=config)

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-test-generator"] is True


def test_model_manager_init_without_schema_config() -> None:
    """Test ModelManager initialization without schema config."""
    manager = ModelManager()

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    schema = manager.get_schema("User", "1.0.0")

    # Should generate normal schema
    assert "properties" in schema
    assert "name" in schema["properties"]  # type: ignore


# set_default_schema_generator tests
def test_set_default_schema_generator_with_class(manager: ModelManager) -> None:
    """Test setting default generator with GenerateJsonSchema class."""
    manager.set_default_schema_generator(CustomTestGenerator)

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-test-generator"] is True


def test_set_default_schema_generator_with_callable(manager: ModelManager) -> None:
    """Test setting default generator with callable function."""
    manager.set_default_schema_generator(simple_callable_generator)

    @manager.model("Product", "1.0.0")
    class Product(BaseModel):
        title: str

    schema = manager.get_schema("Product", "1.0.0")

    assert schema["x-simple"] is True


def test_set_default_schema_generator_applies_to_all_models(
    manager: ModelManager,
) -> None:
    """Test default generator applies to all registered models."""
    manager.set_default_schema_generator(CustomTestGenerator)

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.model("Product", "1.0.0")
    class Product(BaseModel):
        title: str

    user_schema = manager.get_schema("User", "1.0.0")
    product_schema = manager.get_schema("Product", "1.0.0")

    assert user_schema["x-test-generator"] is True
    assert product_schema["x-test-generator"] is True


# schema_transformer decorator tests
def test_schema_transformer_decorator(manager: ModelManager) -> None:
    """Test schema_transformer decorator registers transformer."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.schema_transformer("User", "1.0.0")
    def add_metadata(schema: JsonSchema) -> JsonSchema:
        schema["x-metadata"] = "test"
        return schema

    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-metadata"] == "test"


def test_schema_transformer_multiple_transformers(manager: ModelManager) -> None:
    """Test multiple transformers for same model."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.schema_transformer("User", "1.0.0")
    def add_examples(schema: JsonSchema) -> JsonSchema:
        schema["examples"] = [{"name": "Alice"}]
        return schema

    @manager.schema_transformer("User", "1.0.0")
    def add_version(schema: JsonSchema) -> JsonSchema:
        schema["x-version"] = "1.0"
        return schema

    schema = manager.get_schema("User", "1.0.0")

    assert schema["examples"] == [{"name": "Alice"}]
    assert schema["x-version"] == "1.0"


def test_schema_transformer_version_specific(manager: ModelManager) -> None:
    """Test transformer only affects specific version."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.schema_transformer("User", "1.0.0")
    def mark_v1(schema: JsonSchema) -> JsonSchema:
        schema["x-v1-only"] = True
        return schema

    schema_v1 = manager.get_schema("User", "1.0.0")
    schema_v2 = manager.get_schema("User", "2.0.0")

    assert schema_v1["x-v1-only"] is True
    assert "x-v1-only" not in schema_v2


def test_schema_transformer_with_model_version(manager: ModelManager) -> None:
    """Test transformer with ModelVersion object."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.schema_transformer("User", ModelVersion(1, 0, 0))
    def add_metadata(schema: JsonSchema) -> JsonSchema:
        schema["x-metadata"] = True
        return schema

    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-metadata"] is True


def test_schema_transformer_returns_function(manager: ModelManager) -> None:
    """Test transformer decorator returns the function."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.schema_transformer("User", "1.0.0")
    def my_transformer(schema: JsonSchema) -> JsonSchema:
        return schema

    # The decorator should return the original function
    assert callable(my_transformer)


# get_schema_transformers tests
def test_get_schema_transformers_empty(manager: ModelManager) -> None:
    """Test get_schema_transformers with no transformers."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    transformers = manager.get_schema_transformers("User", "1.0.0")

    assert transformers == []


def test_get_schema_transformers_returns_all(manager: ModelManager) -> None:
    """Test get_schema_transformers returns all registered transformers."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.schema_transformer("User", "1.0.0")
    def transformer1(schema: JsonSchema) -> JsonSchema:
        return schema

    @manager.schema_transformer("User", "1.0.0")
    def transformer2(schema: JsonSchema) -> JsonSchema:
        return schema

    transformers = manager.get_schema_transformers("User", "1.0.0")

    assert len(transformers) == 2  # noqa: PLR2004
    assert transformer1 in transformers
    assert transformer2 in transformers


def test_get_schema_transformers_with_model_version(manager: ModelManager) -> None:
    """Test get_schema_transformers with ModelVersion object."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.schema_transformer("User", "1.0.0")
    def transformer(schema: JsonSchema) -> JsonSchema:
        return schema

    transformers = manager.get_schema_transformers("User", ModelVersion(1, 0, 0))

    assert len(transformers) == 1


# clear_schema_transformers tests
def test_clear_schema_transformers_all(manager: ModelManager) -> None:
    """Test clearing all transformers."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.schema_transformer("User", "1.0.0")
    def transformer1(schema: JsonSchema) -> JsonSchema:
        return schema

    @manager.schema_transformer("User", "2.0.0")
    def transformer2(schema: JsonSchema) -> JsonSchema:
        return schema

    manager.clear_schema_transformers()

    assert manager.get_schema_transformers("User", "1.0.0") == []
    assert manager.get_schema_transformers("User", "2.0.0") == []


def test_clear_schema_transformers_by_model(manager: ModelManager) -> None:
    """Test clearing transformers for specific model."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.model("Product", "1.0.0")
    class Product(BaseModel):
        title: str

    @manager.schema_transformer("User", "1.0.0")
    def user_transformer(schema: JsonSchema) -> JsonSchema:
        return schema

    @manager.schema_transformer("Product", "1.0.0")
    def product_transformer(schema: JsonSchema) -> JsonSchema:
        return schema

    manager.clear_schema_transformers("User")

    assert manager.get_schema_transformers("User", "1.0.0") == []
    assert len(manager.get_schema_transformers("Product", "1.0.0")) == 1


def test_clear_schema_transformers_by_version(manager: ModelManager) -> None:
    """Test clearing transformers for specific version."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.schema_transformer("User", "1.0.0")
    def transformer1(schema: JsonSchema) -> JsonSchema:
        return schema

    @manager.schema_transformer("User", "2.0.0")
    def transformer2(schema: JsonSchema) -> JsonSchema:
        return schema

    manager.clear_schema_transformers("User", "1.0.0")

    assert manager.get_schema_transformers("User", "1.0.0") == []
    assert len(manager.get_schema_transformers("User", "2.0.0")) == 1


# get_schema with config tests
def test_get_schema_with_config(manager: ModelManager) -> None:
    """Test get_schema with custom config."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    config = SchemaConfig(schema_generator=CustomTestGenerator)
    schema = manager.get_schema("User", "1.0.0", config=config)

    assert schema["x-test-generator"] is True


def test_get_schema_config_overrides_default(manager: ModelManager) -> None:
    """Test config parameter overrides default config."""
    manager.set_default_schema_generator(CustomTestGenerator)

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    override_config = SchemaConfig(schema_generator=AnotherTestGenerator)
    schema = manager.get_schema("User", "1.0.0", config=override_config)

    assert "x-another" in schema
    assert "x-test-generator" not in schema


def test_get_schema_kwargs_override_config(manager: ModelManager) -> None:
    """Test kwargs override config."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    config = SchemaConfig(by_alias=True)
    schema = manager.get_schema("User", "1.0.0", config=config, by_alias=False)

    assert isinstance(schema, dict)


# dump_schemas with config tests
def test_dump_schemas_with_config(manager: ModelManager, tmp_path: Path) -> None:
    """Test dump_schemas with custom config."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    config = SchemaConfig(schema_generator=CustomTestGenerator)
    manager.dump_schemas(tmp_path, config=config)

    with open(tmp_path / "User_v1.0.0.json") as f:
        data = json.load(f)

    assert data["x-test-generator"] is True


def test_dump_schemas_uses_default_config(tmp_path: Path) -> None:
    """Test dump_schemas uses default config."""
    config = SchemaConfig(schema_generator=CustomTestGenerator)
    manager = ModelManager(default_schema_config=config)

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    manager.dump_schemas(tmp_path)

    with open(tmp_path / "User_v1.0.0.json") as f:
        data = json.load(f)

    assert data["x-test-generator"] is True


def test_dump_schemas_includes_transformers(
    manager: ModelManager, tmp_path: Path
) -> None:
    """Test dump_schemas includes transformer modifications."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.schema_transformer("User", "1.0.0")
    def add_export_info(schema: JsonSchema) -> JsonSchema:
        schema["x-exported"] = True
        return schema

    manager.dump_schemas(tmp_path)

    with open(tmp_path / "User_v1.0.0.json") as f:
        data = json.load(f)

    assert data["x-exported"] is True


# Integration tests
def test_generator_and_transformer_integration(manager: ModelManager) -> None:
    """Test generator and transformer work together."""
    manager.set_default_schema_generator(CustomTestGenerator)

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.schema_transformer("User", "1.0.0")
    def add_examples(schema: JsonSchema) -> JsonSchema:
        schema["examples"] = [{"name": "Test User"}]
        return schema

    schema = manager.get_schema("User", "1.0.0")

    # Generator applied
    assert schema["x-test-generator"] is True
    # Transformer applied
    assert schema["examples"] == [{"name": "Test User"}]


def test_complete_workflow_with_custom_schema(manager: ModelManager) -> None:
    """Test complete workflow: models, migrations, and custom schemas."""
    manager.set_default_schema_generator(CustomTestGenerator)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "unknown@example.com"}

    @manager.schema_transformer("User", "1.0.0")
    def mark_deprecated(schema: JsonSchema) -> JsonSchema:
        schema["deprecated"] = True
        return schema

    @manager.schema_transformer("User", "2.0.0")
    def mark_current(schema: JsonSchema) -> JsonSchema:
        schema["x-current"] = True
        return schema

    # Test migration still works
    old_data = {"name": "Alice"}
    migrated = manager.migrate(old_data, "User", "1.0.0", "2.0.0")
    assert migrated.email == "unknown@example.com"  # type: ignore

    # Test schemas have custom generation + transformers
    schema_v1 = manager.get_schema("User", "1.0.0")
    schema_v2 = manager.get_schema("User", "2.0.0")

    assert schema_v1["x-test-generator"] is True
    assert schema_v1["deprecated"] is True

    assert schema_v2["x-test-generator"] is True
    assert schema_v2["x-current"] is True


def test_transformer_with_nested_modifications(manager: ModelManager) -> None:
    """Test transformer can modify nested schema properties."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str
        age: int

    @manager.schema_transformer("User", "1.0.0")
    def add_field_metadata(schema: JsonSchema) -> JsonSchema:
        for field_name, field_schema in schema.get("properties", {}).items():  # type: ignore
            field_schema["x-field-id"] = f"user_{field_name}"  # type: ignore
        return schema

    schema = manager.get_schema("User", "1.0.0")

    assert schema["properties"]["name"]["x-field-id"] == "user_name"  # type: ignore
    assert schema["properties"]["age"]["x-field-id"] == "user_age"  # type: ignore


def test_multiple_models_with_different_transformers(manager: ModelManager) -> None:
    """Test different models can have different transformers."""

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.model("Product", "1.0.0")
    class Product(BaseModel):
        title: str

    @manager.schema_transformer("User", "1.0.0")
    def user_transformer(schema: JsonSchema) -> JsonSchema:
        schema["x-model-type"] = "user"
        return schema

    @manager.schema_transformer("Product", "1.0.0")
    def product_transformer(schema: JsonSchema) -> JsonSchema:
        schema["x-model-type"] = "product"
        return schema

    user_schema = manager.get_schema("User", "1.0.0")
    product_schema = manager.get_schema("Product", "1.0.0")

    assert user_schema["x-model-type"] == "user"
    assert product_schema["x-model-type"] == "product"


def test_callable_generator_and_transformer(manager: ModelManager) -> None:
    """Test callable generator works with transformers."""
    manager.set_default_schema_generator(simple_callable_generator)

    @manager.model("User", "1.0.0")
    class User(BaseModel):
        name: str

    @manager.schema_transformer("User", "1.0.0")
    def add_metadata(schema: JsonSchema) -> JsonSchema:
        schema["x-transformed"] = True
        return schema

    schema = manager.get_schema("User", "1.0.0")

    assert schema["x-simple"] is True
    assert schema["x-transformed"] is True


def test_hook_can_read_data_with_get(manager: ModelManager) -> None:
    """Test hook can use .get() method on data."""

    class GetTestHook(MigrationHook):
        def __init__(self) -> None:
            self.name_value: str | None = None
            self.missing_value: str | None = None

        def before_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            data: Mapping[str, Any],
        ) -> None:
            self.name_value = data.get("name")
            self.missing_value = data.get("missing", "default")

    hook = GetTestHook()
    manager.add_hook(hook)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "unknown@example.com"}

    manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")

    assert hook.name_value == "Alice"
    assert hook.missing_value == "default"


def test_hook_can_check_key_membership(manager: ModelManager) -> None:
    """Test hook can use 'in' operator on data."""

    class MembershipTestHook(MigrationHook):
        def __init__(self) -> None:
            self.has_name: bool = False
            self.has_email: bool = False

        def before_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            data: Mapping[str, Any],
        ) -> None:
            self.has_name = "name" in data
            self.has_email = "email" in data

    hook = MembershipTestHook()
    manager.add_hook(hook)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "unknown@example.com"}

    manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")

    assert hook.has_name is True
    assert hook.has_email is False


def test_hook_can_iterate_over_keys(manager: ModelManager) -> None:
    """Test hook can iterate over data keys."""

    class IterationTestHook(MigrationHook):
        def __init__(self) -> None:
            self.keys: list[str] = []

        def before_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            data: Mapping[str, Any],
        ) -> None:
            self.keys = list(data.keys())

    hook = IterationTestHook()
    manager.add_hook(hook)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "unknown@example.com"}

    manager.migrate({"name": "Alice", "age": 30}, "User", "1.0.0", "2.0.0")

    assert set(hook.keys) == {"name", "age"}


def test_hook_can_access_values(manager: ModelManager) -> None:
    """Test hook can access data values."""

    class ValueAccessTestHook(MigrationHook):
        def __init__(self) -> None:
            self.all_values: list[Any] = []

        def before_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            data: Mapping[str, Any],
        ) -> None:
            self.all_values = list(data.values())

    hook = ValueAccessTestHook()
    manager.add_hook(hook)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "unknown@example.com"}

    manager.migrate({"name": "Alice", "age": 30}, "User", "1.0.0", "2.0.0")

    assert "Alice" in hook.all_values
    assert 30 in hook.all_values  # noqa: PLR2004


def test_hook_can_access_items(manager: ModelManager) -> None:
    """Test hook can access data items (key-value pairs)."""

    class ItemsTestHook(MigrationHook):
        def __init__(self) -> None:
            self.items: dict[str, Any] = {}

        def before_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            data: Mapping[str, Any],
        ) -> None:
            self.items = dict(data.items())

    hook = ItemsTestHook()
    manager.add_hook(hook)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "unknown@example.com"}

    manager.migrate({"name": "Alice", "age": 30}, "User", "1.0.0", "2.0.0")

    assert hook.items == {"name": "Alice", "age": 30}


def test_hook_can_get_length(manager: ModelManager) -> None:
    """Test hook can get length of data."""

    class LengthTestHook(MigrationHook):
        def __init__(self) -> None:
            self.length: int = 0

        def before_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            data: Mapping[str, Any],
        ) -> None:
            self.length = len(data)

    hook = LengthTestHook()
    manager.add_hook(hook)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "unknown@example.com"}

    manager.migrate(
        {"name": "Alice", "age": 30, "city": "NYC"}, "User", "1.0.0", "2.0.0"
    )

    assert hook.length == 3  # noqa: PLR2004


def test_hook_can_use_bracket_notation(manager: ModelManager) -> None:
    """Test hook can access data with bracket notation."""

    class BracketAccessTestHook(MigrationHook):
        def __init__(self) -> None:
            self.name_value: str | None = None

        def before_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            data: Mapping[str, Any],
        ) -> None:
            self.name_value = data["name"]

    hook = BracketAccessTestHook()
    manager.add_hook(hook)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "unknown@example.com"}

    manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")

    assert hook.name_value == "Alice"


def test_hook_bracket_access_raises_keyerror_for_missing_key(
    manager: ModelManager,
) -> None:
    """Test hook gets KeyError when accessing missing key with brackets."""

    class BracketErrorTestHook(MigrationHook):
        def __init__(self) -> None:
            self.error_raised: bool = False

        def before_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            data: Mapping[str, Any],
        ) -> None:
            try:
                _ = data["missing_key"]
            except KeyError:
                self.error_raised = True

    hook = BracketErrorTestHook()
    manager.add_hook(hook)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "unknown@example.com"}

    manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")

    assert hook.error_raised is True
