"""pytest fixtures."""

import pytest
from pydantic import BaseModel, Field

from pyrmute import MigrationData, ModelManager
from pyrmute._migration_manager import MigrationManager
from pyrmute._registry import Registry
from pyrmute._schema_manager import SchemaManager


@pytest.fixture
def manager() -> ModelManager:
    """Create a fresh ModelManager instance."""
    return ModelManager()


class UserV1(BaseModel):
    """UserV1 model."""

    name: str


class UserV2(BaseModel):
    """UserV2 model."""

    name: str
    email: str


class UserV3(BaseModel):
    """UserV3 model."""

    name: str
    email: str
    age: int = 0


class UserV4(BaseModel):
    """UserV4 model."""

    name: str
    email: str
    age: int = 0
    aliases: list[str] = Field(default_factory=list)


@pytest.fixture
def user_v1() -> type[UserV1]:
    """Create a sample model version 1."""
    return UserV1


@pytest.fixture
def user_v2() -> type[UserV2]:
    """Create a sample model version 2."""
    return UserV2


@pytest.fixture
def user_v3() -> type[BaseModel]:
    """Create User model version 3."""
    return UserV3


@pytest.fixture
def user_v4() -> type[BaseModel]:
    """Create User model version 4."""
    return UserV4


class AddressV1(BaseModel):
    """AddressV1 model."""

    street: str


@pytest.fixture
def address_v1() -> type[BaseModel]:
    """Create Address model version 1."""
    return AddressV1


def _migrate_user_v1_to_v2(data: MigrationData) -> MigrationData:
    """Migration function for User 1.0.0 -> 2.0.0.

    This is separated out so it can be pickled.
    """
    return {**data, "email": "unknown@example.com"}


@pytest.fixture
def registered_manager(
    manager: ModelManager,
    user_v1: type[UserV1],
    user_v2: type[UserV2],
) -> ModelManager:
    """Create a manager with pre-registered models and migrations."""
    manager.model("User", "1.0.0")(user_v1)
    manager.model("User", "2.0.0")(user_v2)
    manager.migration("User", "1.0.0", "2.0.0")(_migrate_user_v1_to_v2)

    return manager


@pytest.fixture
def registry() -> Registry:
    """Create a fresh Registry instance."""
    return Registry()


@pytest.fixture
def populated_registry(
    registry: Registry,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
    user_v3: type[BaseModel],
    user_v4: type[BaseModel],
) -> Registry:
    """Create a registry with multiple registered models."""
    registry.register("User", "1.0.0")(user_v1)
    registry.register("User", "2.0.0")(user_v2)
    registry.register("User", "3.0.0", auto_migrate=True)(user_v3)
    registry.register("User", "4.0.0", auto_migrate=True)(user_v4)
    return registry


@pytest.fixture
def migration_manager(registry: Registry) -> MigrationManager:
    """Create a MigrationManager with a registry."""
    return MigrationManager(registry)


@pytest.fixture
def populated_migration_manager(
    registry: Registry,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
    user_v3: type[BaseModel],
    user_v4: type[BaseModel],
) -> MigrationManager:
    """Create a manager with registered models."""
    registry.register("User", "1.0.0")(user_v1)
    registry.register("User", "2.0.0")(user_v2)
    registry.register("User", "3.0.0", auto_migrate=True)(user_v3)
    registry.register("User", "4.0.0", auto_migrate=True)(user_v4)
    return MigrationManager(registry)


@pytest.fixture
def schema_manager(registry: Registry) -> SchemaManager:
    """Create a SchemaManager with a registry."""
    return SchemaManager(registry)


@pytest.fixture
def populated_schema_manager(
    registry: Registry,
    user_v1: type[BaseModel],
    user_v2: type[BaseModel],
) -> SchemaManager:
    """Create a schema manager with registered models."""
    registry.register("User", "1.0.0")(user_v1)
    registry.register("User", "2.0.0")(user_v2)
    return SchemaManager(registry)
