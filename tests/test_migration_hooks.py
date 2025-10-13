"""Tests for migration hooks."""

from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel

from pyrmute import (
    MetricsHook,
    MigrationHook,
    ModelData,
    ModelManager,
    ModelVersion,
)


class CustomHook(MigrationHook):
    """Hook that tracks all calls for testing."""

    def __init__(self) -> None:
        """Initialize tracking lists."""
        self.before_calls: list[
            tuple[str, ModelVersion, ModelVersion, Mapping[str, Any]]
        ] = []
        self.after_calls: list[
            tuple[str, ModelVersion, ModelVersion, Mapping[str, Any], Mapping[str, Any]]
        ] = []
        self.error_calls: list[
            tuple[str, ModelVersion, ModelVersion, Mapping[str, Any], Exception]
        ] = []

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        """Track before_migrate call."""
        self.before_calls.append((name, from_version, to_version, dict(data)))

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        """Track after_migrate call."""
        self.after_calls.append(
            (
                name,
                from_version,
                to_version,
                dict(original_data),
                dict(migrated_data),
            )
        )

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        """Track on_error call."""
        self.error_calls.append((name, from_version, to_version, dict(data), error))


# MigrationHook base class tests
def test_migration_hook_default_before_migrate() -> None:
    """Test MigrationHook default before_migrate does nothing."""
    hook = MigrationHook()
    data = {"name": "Alice"}

    hook.before_migrate("User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), data)
    assert data == {"name": "Alice"}


def test_migration_hook_default_after_migrate() -> None:
    """Test MigrationHook default after_migrate does nothing."""
    hook = MigrationHook()
    original = {"name": "Alice"}
    migrated = {"name": "Alice", "email": "alice@example.com"}

    hook.after_migrate(
        "User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), original, migrated
    )

    assert original == {"name": "Alice"}
    assert migrated == {"name": "Alice", "email": "alice@example.com"}


def test_migration_hook_default_on_error() -> None:
    """Test MigrationHook default on_error does nothing."""
    hook = MigrationHook()
    hook.on_error(
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
        {"name": "Alice"},
        ValueError("test error"),
    )


# MetricsHook tests
def test_metrics_hook_initialization() -> None:
    """Test MetricsHook initializes with zero counts."""
    hook = MetricsHook()

    assert hook.total_count == 0
    assert hook.error_count == 0
    assert hook.migrations_by_model == {}
    assert hook.errors_by_model == {}


def test_metrics_hook_tracks_migration_attempt() -> None:
    """Test MetricsHook tracks migration attempts."""
    hook = MetricsHook()
    data = {"name": "Alice"}

    hook.before_migrate("User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), data)

    assert hook.total_count == 1
    assert hook.migrations_by_model["User"] == 1


def test_metrics_hook_tracks_multiple_migrations() -> None:
    """Test MetricsHook tracks multiple migrations."""
    hook = MetricsHook()
    data = {"name": "Alice"}

    hook.before_migrate("User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), data)
    hook.before_migrate("User", ModelVersion(2, 0, 0), ModelVersion(3, 0, 0), data)
    hook.before_migrate("Product", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), data)

    assert hook.total_count == 3  # noqa: PLR2004
    assert hook.migrations_by_model["User"] == 2  # noqa: PLR2004
    assert hook.migrations_by_model["Product"] == 1


def test_metrics_hook_tracks_errors() -> None:
    """Test MetricsHook tracks errors."""
    hook = MetricsHook()
    error = ValueError("test error")

    hook.on_error(
        "User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), {"name": "Alice"}, error
    )

    assert hook.error_count == 1
    assert hook.errors_by_model["User"] == 1


def test_metrics_hook_tracks_multiple_errors() -> None:
    """Test MetricsHook tracks multiple errors."""
    hook = MetricsHook()
    error = ValueError("test error")

    hook.on_error(
        "User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), {"name": "Alice"}, error
    )
    hook.on_error(
        "User", ModelVersion(2, 0, 0), ModelVersion(3, 0, 0), {"name": "Bob"}, error
    )
    hook.on_error(
        "Product",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
        {"name": "Widget"},
        error,
    )

    assert hook.error_count == 3  # noqa: PLR2004
    assert hook.errors_by_model["User"] == 2  # noqa: PLR2004
    assert hook.errors_by_model["Product"] == 1


def test_metrics_hook_success_rate_all_passed() -> None:
    """Test MetricsHook success_rate when all migrations succeed."""
    hook = MetricsHook()
    data = {"name": "Alice"}

    hook.before_migrate("User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), data)
    hook.before_migrate("User", ModelVersion(2, 0, 0), ModelVersion(3, 0, 0), data)

    assert hook.success_rate == 1.0


def test_metrics_hook_success_rate_all_failed() -> None:
    """Test MetricsHook success_rate when all migrations fail."""
    hook = MetricsHook()
    data = {"name": "Alice"}
    error = ValueError("test error")

    hook.before_migrate("User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), data)
    hook.on_error("User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), data, error)

    hook.before_migrate("User", ModelVersion(2, 0, 0), ModelVersion(3, 0, 0), data)
    hook.on_error("User", ModelVersion(2, 0, 0), ModelVersion(3, 0, 0), data, error)

    assert hook.success_rate == 0.0


def test_metrics_hook_success_rate_partial() -> None:
    """Test MetricsHook success_rate with partial failures."""
    hook = MetricsHook()
    data = {"name": "Alice"}
    error = ValueError("test error")

    hook.before_migrate("User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), data)
    hook.before_migrate("User", ModelVersion(2, 0, 0), ModelVersion(3, 0, 0), data)
    hook.on_error("User", ModelVersion(2, 0, 0), ModelVersion(3, 0, 0), data, error)
    hook.before_migrate("Product", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), data)
    hook.before_migrate("Product", ModelVersion(2, 0, 0), ModelVersion(3, 0, 0), data)

    assert hook.success_rate == 0.75  # noqa: PLR2004


def test_metrics_hook_success_rate_zero_migrations() -> None:
    """Test MetricsHook success_rate with no migrations."""
    hook = MetricsHook()

    assert hook.success_rate == 0.0


def test_metrics_hook_does_not_modify_data() -> None:
    """Test MetricsHook does not modify data it observes."""
    hook = MetricsHook()
    data = {"name": "Alice", "count": 5}
    original_data = dict(data)

    hook.before_migrate("User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0), data)

    assert data == original_data


# ModelManager hook integration tests
def test_manager_add_hook(manager: ModelManager) -> None:
    """Test adding a hook to ModelManager."""
    hook = CustomHook()
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

    assert len(hook.before_calls) == 1
    assert len(hook.after_calls) == 1
    assert len(hook.error_calls) == 0


def test_manager_remove_hook(manager: ModelManager) -> None:
    """Test removing a hook from ModelManager."""
    hook = CustomHook()
    manager.add_hook(hook)
    manager.remove_hook(hook)

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

    assert len(hook.before_calls) == 0
    assert len(hook.after_calls) == 0


def test_manager_clear_hooks(manager: ModelManager) -> None:
    """Test clearing all hooks from ModelManager."""
    hook1 = CustomHook()
    hook2 = CustomHook()
    manager.add_hook(hook1)
    manager.add_hook(hook2)
    manager.clear_hooks()

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

    assert len(hook1.before_calls) == 0
    assert len(hook2.before_calls) == 0


def test_hooks_called_in_order(manager: ModelManager) -> None:
    """Test hooks are called in registration order."""

    class OrderTrackingHook(MigrationHook):
        def __init__(self, order_list: list[int], hook_id: int) -> None:
            self.order_list = order_list
            self.hook_id = hook_id

        def before_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            data: Mapping[str, Any],
        ) -> None:
            self.order_list.append(self.hook_id)

    order: list[int] = []
    hook1 = OrderTrackingHook(order, 1)
    hook2 = OrderTrackingHook(order, 2)
    hook3 = OrderTrackingHook(order, 3)

    manager.add_hook(hook1)
    manager.add_hook(hook2)
    manager.add_hook(hook3)

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

    assert order == [1, 2, 3]


def test_before_hook_receives_correct_data(manager: ModelManager) -> None:
    """Test before_migrate hook receives correct parameters."""
    hook = CustomHook()
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

    input_data = {"name": "Alice"}
    manager.migrate(input_data, "User", "1.0.0", "2.0.0")

    assert len(hook.before_calls) == 1
    name, from_v, to_v, data = hook.before_calls[0]
    assert name == "User"
    assert from_v == ModelVersion(1, 0, 0)
    assert to_v == ModelVersion(2, 0, 0)
    assert data == input_data


def test_after_hook_receives_correct_data(manager: ModelManager) -> None:
    """Test after_migrate hook receives correct parameters."""
    hook = CustomHook()
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

    input_data = {"name": "Alice"}
    manager.migrate(input_data, "User", "1.0.0", "2.0.0")

    assert len(hook.after_calls) == 1
    name, from_v, to_v, original, migrated = hook.after_calls[0]
    assert name == "User"
    assert from_v == ModelVersion(1, 0, 0)
    assert to_v == ModelVersion(2, 0, 0)
    assert original == input_data
    assert migrated == {"name": "Alice", "email": "unknown@example.com"}


def test_error_hook_called_on_migration_failure(manager: ModelManager) -> None:
    """Test on_error hook is called when migration fails."""
    hook = CustomHook()
    manager.add_hook(hook)

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

    input_data = {"name": "Alice"}
    with pytest.raises(Exception):  # noqa: B017
        manager.migrate(input_data, "User", "1.0.0", "2.0.0")

    assert len(hook.error_calls) == 1
    name, from_v, to_v, data, error = hook.error_calls[0]
    assert name == "User"
    assert from_v == ModelVersion(1, 0, 0)
    assert to_v == ModelVersion(2, 0, 0)
    assert data == input_data
    assert isinstance(error, Exception)


def test_error_hook_called_before_exception_raised(manager: ModelManager) -> None:
    """Test error hooks are called before exception propagates."""
    hook = CustomHook()
    manager.add_hook(hook)

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

    with pytest.raises(Exception):  # noqa: B017
        manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")

    # Error hook should have been called
    assert len(hook.error_calls) == 1
    # But after hooks should not have been called
    assert len(hook.after_calls) == 0


def test_hook_does_not_modify_input_data(manager: ModelManager) -> None:
    """Test hooks cannot modify input data (read-only observation)."""

    class ReadOnlyCustomHook(MigrationHook):
        def __init__(self) -> None:
            self.observed_data: Mapping[str, Any] | None = None

        def before_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            data: Mapping[str, Any],
        ) -> None:
            self.observed_data = dict(data)

    hook = ReadOnlyCustomHook()
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

    input_data = {"name": "Alice"}
    result = manager.migrate_data(input_data, "User", "1.0.0", "2.0.0")

    # Hook observed the original data
    assert hook.observed_data == {"name": "Alice"}
    # Result has email added by migration, not by hook
    assert result == {"name": "Alice", "email": "unknown@example.com"}
    # Original input unchanged
    assert input_data == {"name": "Alice"}


def test_hooks_work_with_batch_migration(manager: ModelManager) -> None:
    """Test hooks are called for each item in batch migration."""
    hook = CustomHook()
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

    data_list = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}]
    manager.migrate_batch(data_list, "User", "1.0.0", "2.0.0")

    assert len(hook.before_calls) == 3  # noqa: PLR2004
    assert len(hook.after_calls) == 3  # noqa: PLR2004


def test_hooks_work_with_streaming_migration(manager: ModelManager) -> None:
    """Test hooks are called for streaming migrations."""
    hook = CustomHook()
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

    data_list = [{"name": "Alice"}, {"name": "Bob"}]
    list(manager.migrate_batch_streaming(data_list, "User", "1.0.0", "2.0.0"))

    assert len(hook.before_calls) == 2  # noqa: PLR2004
    assert len(hook.after_calls) == 2  # noqa: PLR2004


def test_metrics_hook_integration(manager: ModelManager) -> None:
    """Test MetricsHook works with real migrations."""
    metrics = MetricsHook()
    manager.add_hook(metrics)

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
    manager.migrate({"name": "Bob"}, "User", "1.0.0", "2.0.0")

    assert metrics.total_count == 2  # noqa: PLR2004
    assert metrics.error_count == 0
    assert metrics.success_rate == 1.0
    assert metrics.migrations_by_model["User"] == 2  # noqa: PLR2004


def test_metrics_hook_tracks_errors_in_integration(manager: ModelManager) -> None:
    """Test MetricsHook tracks errors in real scenario."""
    metrics = MetricsHook()
    manager.add_hook(metrics)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: ModelData) -> ModelData:
        if data["name"] == "Bob":
            raise ValueError("Bob not allowed")
        return {**data, "email": "unknown@example.com"}

    manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")

    with pytest.raises(Exception):  # noqa: B017
        manager.migrate({"name": "Bob"}, "User", "1.0.0", "2.0.0")

    assert metrics.total_count == 2  # noqa: PLR2004
    assert metrics.error_count == 1
    assert metrics.success_rate == 0.5  # noqa: PLR2004
    assert metrics.errors_by_model["User"] == 1


def test_hook_does_not_affect_same_version_migration(manager: ModelManager) -> None:
    """Test hooks are not called for same-version migration."""
    hook = CustomHook()
    manager.add_hook(hook)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    result = manager.migrate_data({"name": "Alice"}, "User", "1.0.0", "1.0.0")

    # No hooks should be called for same version
    assert len(hook.before_calls) == 0
    assert len(hook.after_calls) == 0
    assert result == {"name": "Alice"}


def test_validation_hook_can_raise_error(manager: ModelManager) -> None:
    """Test hook can raise an error for validation purposes."""

    class ValidationHook(MigrationHook):
        def after_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            original_data: Mapping[str, Any],
            migrated_data: Mapping[str, Any],
        ) -> None:
            if "email" not in migrated_data:
                raise ValueError("Email is required after migration")

    hook = ValidationHook()
    manager.add_hook(hook)

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: ModelData) -> ModelData:
        # Forget to add email
        return data

    with pytest.raises(ValueError, match="Email is required"):
        manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")


def test_multiple_hooks_all_observe_same_data(manager: ModelManager) -> None:
    """Test multiple hooks all observe the same data."""

    class ObserverHook(MigrationHook):
        def __init__(self, hook_id: int) -> None:
            self.hook_id = hook_id
            self.observed_data: Mapping[str, Any] | None = None

        def before_migrate(
            self,
            name: str,
            from_version: ModelVersion,
            to_version: ModelVersion,
            data: Mapping[str, Any],
        ) -> None:
            self.observed_data = dict(data)

    hook1 = ObserverHook(1)
    hook2 = ObserverHook(2)
    hook3 = ObserverHook(3)

    manager.add_hook(hook1)
    manager.add_hook(hook2)
    manager.add_hook(hook3)

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

    # All hooks should observe the same original data
    assert hook1.observed_data == {"name": "Alice"}
    assert hook2.observed_data == {"name": "Alice"}
    assert hook3.observed_data == {"name": "Alice"}
