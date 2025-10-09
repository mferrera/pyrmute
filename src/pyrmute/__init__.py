"""pyrmute - versioned Pydantic models and schemas with migrations."""

from ._migration_manager import MigrationManager
from ._registry import Registry
from ._schema_manager import SchemaManager
from ._version import __version__
from .exceptions import (
    InvalidVersionError,
    MigrationError,
    ModelNotFoundError,
    VersionedModelError,
)
from .migration_testing import (
    MigrationTestCase,
    MigrationTestResult,
    MigrationTestResults,
)
from .model_diff import ModelDiff
from .model_manager import ModelManager
from .model_version import ModelVersion
from .types import (
    JsonSchema,
    MigrationFunc,
    ModelData,
    NestedModelInfo,
)

__all__ = [
    "InvalidVersionError",
    "JsonSchema",
    "MigrationError",
    "MigrationFunc",
    "MigrationManager",
    "MigrationTestCase",
    "MigrationTestResult",
    "MigrationTestResults",
    "ModelData",
    "ModelDiff",
    "ModelManager",
    "ModelNotFoundError",
    "ModelVersion",
    "NestedModelInfo",
    "Registry",
    "SchemaManager",
    "VersionedModelError",
    "__version__",
]
