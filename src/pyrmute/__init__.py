"""pyrmute - versioned Pydantic models and schemas with migrations."""

from ._migration_manager import MigrationManager
from ._registry import Registry
from ._schema_manager import SchemaManager
from ._version import __version__
from .avro_schema import AvroExporter, AvroSchemaGenerator
from .avro_types import AvroRecordSchema
from .exceptions import (
    InvalidVersionError,
    MigrationError,
    ModelNotFoundError,
    VersionedModelError,
)
from .migration_hooks import MetricsHook, MigrationHook
from .migration_testing import (
    MigrationTestCase,
    MigrationTestCases,
    MigrationTestResult,
    MigrationTestResults,
)
from .model_diff import ModelDiff
from .model_manager import ModelManager
from .model_version import ModelVersion
from .protobuf_schema import ProtoExporter
from .schema_config import SchemaConfig
from .types import (
    JsonSchema,
    JsonSchemaMode,
    MigrationFunc,
    ModelData,
    NestedModelInfo,
)

__all__ = [
    "AvroExporter",
    "AvroRecordSchema",
    "AvroSchemaGenerator",
    "InvalidVersionError",
    "JsonSchema",
    "JsonSchemaMode",
    "MetricsHook",
    "MigrationError",
    "MigrationFunc",
    "MigrationHook",
    "MigrationManager",
    "MigrationTestCase",
    "MigrationTestCases",
    "MigrationTestResult",
    "MigrationTestResults",
    "ModelData",
    "ModelDiff",
    "ModelManager",
    "ModelNotFoundError",
    "ModelVersion",
    "NestedModelInfo",
    "ProtoExporter",
    "Registry",
    "SchemaConfig",
    "SchemaManager",
    "VersionedModelError",
    "__version__",
]
