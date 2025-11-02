<div align="center" markdown=1>

<img src="https://raw.githubusercontent.com/mferrera/pyrmute/main/docs/logo.svg" width="120" height="120" alt="pyrmute logo">

# pyrmute

**Version, migrate, and export Pydantic models to any schema format**

[![ci](https://img.shields.io/github/actions/workflow/status/mferrera/pyrmute/ci.yml?branch=main&logo=github&label=ci)](https://github.com/mferrera/pyrmute/actions?query=event%3Apush+branch%3Amain+workflow%3Aci)
[![codecov](https://codecov.io/gh/mferrera/pyrmute/graph/badge.svg?token=4J9G3CEZQF)](https://codecov.io/gh/mferrera/pyrmute)
[![pypi](https://img.shields.io/pypi/v/pyrmute.svg)](https://pypi.python.org/pypi/pyrmute)
[![versions](https://img.shields.io/pypi/pyversions/pyrmute.svg)](https://github.com/mferrera/pyrmute)
[![license](https://img.shields.io/github/license/mferrera/pyrmute.svg)](https://github.com/mferrera/pyrmute/blob/main/LICENSE)

[Documentation](https://pyrmute.readthedocs.io) | [Examples](https://github.com/mferrera/pyrmute/tree/main/examples)

</div>

---

Pydantic model versioning, migrations, and multi-format schema generation.

Pyrmute handles the complexity of data model evolution so you can confidently
make changes without breaking your production systems. Version your models,
define transformations, export to TypeScript/JSON Schema/Protobuf, and let
pyrmute automatically migrate legacy data through multiple versions.

Pyrmute is to Pydantic models what Alembic is to SQLAlchemy. It offers a
structured, composable way to evolve and migrate schemas across versions.

**Key Features**

- **Version your models** - Track schema evolution with semantic versioning.
- **Automatic migration chains** - Transform data across multiple versions
    (1.0.0 → 2.0.0 → 3.0.0) in a single call.
- **Type-safe transformations** - Migrations return validated Pydantic models,
    catching errors before they reach production.
- **Migration hooks** - Observe migrations with built-in metrics tracking or
    custom hooks for logging, monitoring, and validation.
- **Flexible schema export** - Generates, with support for nested models:
    - JSON Schema with support for `$ref`, custom generators, and schema transformers
    - Apache Avro schemas
    - Protocol Buffer schemas
    - TypeScript interfaces, types, and Zod schemas
- **Production-ready** - Batch processing, parallel execution, and streaming
    support for large datasets.
- **Only one dependency** - Pydantic.

## When to Use Pyrmute

Pyrmute is useful for handling schema evolution in production systems:

- **Configuration files** - Upgrade user config files as your CLI/desktop app
    evolves (`.apprc`, `config.json`, `settings.yaml`).
- **Message queues & event streams** - Handle messages from multiple service
    versions publishing different schemas (Kafka, RabbitMQ, SQS).
- **ETL & data imports** - Import CSV/JSON/Excel files exported over years
    with evolving structures.
- **ML model serving** - Manage feature schema evolution across model versions
    and A/B tests.
- **API versioning** - Support multiple API versions with automatic
    request/response migration.
- **Database migrations** - Transparently migrate legacy data on read without
    downtime.
- **Data archival** - Process historical data dumps with various schema
    versions.

See the [examples/](https://github.com/mferrera/pyrmute/tree/main/examples)
directory for complete, runnable code demonstrating these patterns.

## When Not to Use

Pyrmute may not be the right choice if you have:

- **High-throughput systems** - Runtime migration adds latency to hot paths.
    Use upfront batch migrations instead.
- **Existing schema registries** - Already using Confluent/AWS Glue? Stick
    with them for compatibility enforcement and governance.
- **Stable schemas** - Models rarely change? Traditional migration tools are
    simpler and more maintainable.
- **Database DDL changes** - pyrmute transforms data, not database schemas.
    Alembic/Flyway or other ORMs may still be needed to alter tables.

## Help

See [documentation](https://pyrmute.readthedocs.io/en/latest/) for complete
guides and API reference.

## Installation

```bash
pip install pyrmute
```

## Quick Start

```python
from pydantic import BaseModel
from pyrmute import ModelManager, ModelData

manager = ModelManager()


# Version 1: Simple user model
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    age: int


# Version 2: Split name into components
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    first_name: str
    last_name: str
    age: int


# Version 3: Add email and make age optional
@manager.model("User", "3.0.0")
class UserV3(BaseModel):
    first_name: str
    last_name: str
    email: str
    age: int | None = None


# Define how to migrate between versions
@manager.migration("User", "1.0.0", "2.0.0")
def split_name(data: ModelData) -> ModelData:
    parts = data["name"].split(" ", 1)
    return {
        "first_name": parts[0],
        "last_name": parts[1] if len(parts) > 1 else "",
        "age": data["age"],
    }


@manager.migration("User", "2.0.0", "3.0.0")
def add_email(data: ModelData) -> ModelData:
    return {
        **data,
        "email": f"{data['first_name'].lower()}@example.com"
    }


# Migrate legacy data to the latest version
legacy_data = {"name": "John Doe", "age": 30}  # or, legacy.model_dump()
current_user = manager.migrate(legacy_data, "User", "1.0.0", "3.0.0")

print(current_user)
# UserV3(first_name='John', last_name='Doe', email='john@example.com', age=30)
```

## Advanced Usage

### Compare Model Versions

```python
# See exactly what changed between versions
diff = manager.diff("User", "1.0.0", "3.0.0")
print(f"Added: {diff.added_fields}")
print(f"Removed: {diff.removed_fields}")
# Render a changelog to Markdown
print(diff.to_markdown(header_depth=4))
```

With `header_depth=4` the output can be embedded nicely into this document.

#### User: 1.0.0 → 3.0.0

##### Added Fields

- `email: str` (required)
- `first_name: str` (required)
- `last_name: str` (required)

##### Removed Fields

- `name`

##### Modified Fields

- `age` - type: `int` → `int | None` - now optional - default added: `None`

##### Breaking Changes

- ⚠️ New required field 'last_name' will fail for existing data without defaults
- ⚠️ New required field 'first_name' will fail for existing data without defaults
- ⚠️ New required field 'email' will fail for existing data without defaults
- ⚠️ Removed fields 'name' will be lost during migration
- ⚠️ Field 'age' type changed - may cause validation errors


### Batch Processing

```python
# Migrate thousands of records efficiently
legacy_users = [
    {"name": "Alice Smith", "age": 28},
    {"name": "Bob Johnson", "age": 35},
    # ... thousands more
]

# Parallel processing for CPU-intensive migrations
users = manager.migrate_batch(
    legacy_users,
    "User",
    from_version="1.0.0",
    to_version="3.0.0",
    parallel=True,
    max_workers=4,
)
```

### Streaming Large Datasets

```python
# Process huge datasets without loading everything into memory
def load_users_from_database() -> Iterator[dict[str, Any]]:
    yield from database.stream_users()


# Migrate and save incrementally
for user in manager.migrate_batch_streaming(
    load_users_from_database(),
    "User",
    from_version="1.0.0",
    to_version="3.0.0",
    chunk_size=1000
):
    database.save(user)
```

### Test Your Migrations

```python
# Validate migration logic with test cases
results = manager.test_migration(
    "User",
    from_version="1.0.0",
    to_version="2.0.0",
    test_cases=[
        # (input, expected_output)
        (
            {"name": "Alice Smith", "age": 28},
            {"first_name": "Alice", "last_name": "Smith", "age": 28}
        ),
        (
            {"name": "Bob", "age": 35},
            {"first_name": "Bob", "last_name": "", "age": 35}
        ),
    ]
)

# Use in your test suite
assert results.all_passed, f"Migration failed: {results.failures}"
```

### Bidirectional Migrations

```python
# Support both upgrades and downgrades
@manager.migration("Config", "2.0.0", "1.0.0")
def downgrade_config(data: ModelData) -> ModelData:
    """Rollback to v1 format."""
    return {k: v for k, v in data.items() if k in ["setting1", "setting2"]}

# Useful for:
# - Rolling back deployments
# - Normalizing outputs from multiple model versions
# - Supporting legacy systems during transitions
```

### Nested Model Migrations

```python
# Automatically migrates nested Pydantic models
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str
    city: str

@manager.model("Address", "2.0.0")
class AddressV2(BaseModel):
    street: str
    city: str
    postal_code: str

@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    address: AddressV2  # Nested model

# When migrating User, Address is automatically migrated too
@manager.migration("Address", "1.0.0", "2.0.0")
def add_postal_code(data: ModelData) -> ModelData:
    return {**data, "postal_code": "00000"}
```

### Discriminated Unions

```python
from typing import Literal, Union
from pydantic import Field

# Handle complex type hierarchies
@manager.model("CreditCard", "1.0.0")
class CreditCardV1(BaseModel):
    type: Literal["credit_card"] = "credit_card"
    card_number: str

@manager.model("PayPal", "1.0.0")
class PayPalV1(BaseModel):
    type: Literal["paypal"] = "paypal"
    email: str

@manager.model("Payment", "1.0.0")
class PaymentV1(BaseModel):
    method: Union[CreditCardV1, PayPalV1] = Field(discriminator="type")

# Migrations respect discriminated unions
```

### Export Schemas

```python
# Generate schemas for all versions
manager.dump_schemas("schemas/")
# Creates: User_v1_0_0.json, User_v2_0_0.json, User_v3_0_0.json

# Use separate files with $ref for nested models with 'enable_ref=True'.
manager.dump_schemas(
    "schemas/",
    separate_definitions=True,
    ref_template="https://api.example.com/schemas/{model}_v{version}.json"
)

# Generate Avro or Protocol Buffer schemas
manager.dump_avro_schemas("schemas/avro/")
# Creates: User_v1_0_0.avsc, User_v2_0_0.avsc, User_v3_0_0.avsc
manager.dump_proto_schemas("schemas/protos/")
# Creates: User_v1_0_0.proto, User_v2_0_0.proto, User_v3_0_0.proto
```

### Auto-Migration

```python
# Skip writing migration functions for simple changes
@manager.model("Config", "1.0.0")
class ConfigV1(BaseModel):
    timeout: int = 30


@manager.model("Config", "2.0.0", backward_compatible=True)
class ConfigV2(BaseModel):
    timeout: int = 30
    retries: int = 3  # New field with default


# No migration function needed - defaults are applied automatically
config = manager.migrate({"timeout": 60}, "Config", "1.0.0", "2.0.0")
# ConfigV2(timeout=60, retries=3)
```

### Migration Hooks

```python
from pyrmute import MetricsHook

# Track migration performance and success rates
metrics = MetricsHook()
manager.add_hook(metrics)

# Hooks observe migrations without modifying data
users = manager.migrate_batch(legacy_users, "User", "1.0.0", "3.0.0")

print(f"Migrations: {metrics.total_count}")
print(f"Success rate: {metrics.success_rate:.1%}")
print(f"Per model: {metrics.migrations_by_model}")


# Create custom hooks for logging, monitoring, auditing
class LoggingHook(MigrationHook):

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        logger.info(f"Migrating {name} {from_version} → {to_version}")

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        logger.info(f"Migration completed successfully")

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        logger.error(f"Migration failed: {error}")


manager.add_hook(LoggingHook())
```

## Command-Line Interface

pyrmute includes a CLI for working with models outside of Python:

```bash
# Initialize a new project
pyrmute init

# View registered models
pyrmute info

# Validate data against a schema
pyrmute validate -d data.json -s User -v 1.0.0

# Migrate data between versions
pyrmute migrate -d user_v1.json -s User -f 1.0.0 -t 2.0.0 -o user_v2.json

# Compare schema versions
pyrmute diff -s User -f 1.0.0 -t 2.0.0

# Export schemas in various formats
pyrmute export -f json-schema -o schemas/
```

Install with CLI support:

```bash
pip install pyrmute[cli]
```

See the [CLI User
Guide](https://pyrmute.readthedocs.io/en/latest/user-guide/cli/) for complete
documentation.

## Real-World Examples

See [examples/](https://github.com/mferrera/pyrmute/tree/main/examples) for
complete, runnable code demonstrating:

- **Configuration File Evolution** (`config_file_migration.py`) -
    Automatically upgrade CLI/desktop app config files as schemas evolve
- **Message Queue Consumer** (`message_queue_consumer.py`) - Handle messages
    from multiple service versions with different schemas (Kafka, RabbitMQ, SQS)
- **ETL Data Import** (`etl_data_import.py`) - Import historical
    CSV/JSON/Excel files with evolving structures
- **ML Model Serving** (`ml_inference_pipeline.py`) - Manage feature schema
    evolution across model versions and A/B tests
- **Advanced Features** (`advanced_features.py`) - Complex Pydantic features
    including unions, nested models, and validators

## Contributing

For guidance on setting up a development environment and how to make a
contribution to pyrmute, see [Contributing to
pyrmute](https://pyrmute.readthedocs.io/en/latest/contributing/).

## Reporting a Security Vulnerability

See our [security
policy](https://github.com/mferrera/pyrmute/security/policy).

## License

MIT License - see
[LICENSE](https://github.com/mferrera/pyrmute/blob/main/LICENSE) for details.
