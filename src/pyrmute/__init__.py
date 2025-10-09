"""pyrmute - versioned Pydantic models and schemas with migrations.

A package for managing versioned Pydantic models with automatic migrations
and schema management.
"""

from ._version import __version__
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
    MigrationData,
    MigrationFunc,
    ModelMetadata,
)

__all__ = [
    "JsonSchema",
    "MigrationData",
    "MigrationFunc",
    "MigrationTestCase",
    "MigrationTestResult",
    "MigrationTestResults",
    "ModelDiff",
    "ModelManager",
    "ModelMetadata",
    "ModelVersion",
    "__version__",
]
