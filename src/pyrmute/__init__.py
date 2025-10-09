"""pyrmute - versioned Pydantic models and schemas with migrations.

A package for managing versioned Pydantic models with automatic migrations
and schema management.
"""

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
    JsonValue,
    MigrationFunc,
    ModelData,
    ModelMetadata,
)

__all__ = [
    "InvalidVersionError",
    "JsonSchema",
    "JsonValue",
    "MigrationError",
    "MigrationFunc",
    "MigrationTestCase",
    "MigrationTestResult",
    "MigrationTestResults",
    "ModelData",
    "ModelDiff",
    "ModelManager",
    "ModelMetadata",
    "ModelNotFoundError",
    "ModelVersion",
    "VersionedModelError",
    "__version__",
]
