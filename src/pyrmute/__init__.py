"""pyrmute - versioned Pydantic models and schemas with migrations.

A package for managing versioned Pydantic models with automatic migrations
and schema management.
"""

from ._version import __version__
from .model_manager import ModelManager
from .model_version import ModelVersion
from .types import JsonSchema, MigrationData, MigrationFunc, ModelMetadata

__all__ = [
    "JsonSchema",
    "MigrationData",
    "MigrationFunc",
    "ModelManager",
    "ModelMetadata",
    "ModelVersion",
    "__version__",
]
