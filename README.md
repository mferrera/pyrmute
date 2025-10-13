<div align="center">

<img src="https://raw.githubusercontent.com/mferrera/pyrmute/main/docs/logo.svg" width="120" height="120" alt="pyrmute logo">

# pyrmute

**Schema evolution and migrations for Pydantic models**

[![ci](https://img.shields.io/github/actions/workflow/status/mferrera/pyrmute/ci.yml?branch=main&logo=github&label=ci)](https://github.com/mferrera/pyrmute/actions?query=event%3Apush+branch%3Amain+workflow%3Aci)
[![codecov](https://codecov.io/gh/mferrera/pyrmute/graph/badge.svg?token=4J9G3CEZQF)](https://codecov.io/gh/mferrera/pyrmute)
[![pypi](https://img.shields.io/pypi/v/pyrmute.svg)](https://pypi.python.org/pypi/pyrmute)
[![versions](https://img.shields.io/pypi/pyversions/pyrmute.svg)](https://github.com/mferrera/pyrmute)
[![license](https://img.shields.io/github/license/mferrera/pyrmute.svg)](https://github.com/mferrera/pyrmute/blob/main/LICENSE)

[Documentation](https://pyrmute.readthedocs.io) | [Examples](https://github.com/mferrera/pyrmute/tree/main/examples)

</div>

---

Pydantic model migrations and schema management with semantic versioning.

pyrmute handles the complexity of data model evolution so you can confidently
make changes without breaking your production systems. Version your models,
define transformations, and let pyrmute automatically migrate legacy data
through multiple versions.

**Key Features**

- **Version your models** - Track schema evolution with semantic versioning
- **Automatic migration chains** - Transform data across multiple versions
    (1.0.0 → 2.0.0 → 3.0.0) in a single call
- **Type-safe transformations** - Migrations return validated Pydantic models,
    catching errors before they reach production
- **Migration hooks** - Observe migrations with built-in metrics tracking or
    custom hooks for logging, monitoring, and validation
- **Flexible schema export** - Generate JSON schemas for all versions with
    support for `$ref`, custom generators, and nested models
- **Production-ready** - Batch processing, parallel execution, and streaming
    support for large datasets
- **Only one dependency** - Pydantic

## When to Use pyrmute

pyrmute is useful for handling schema evolution in production systems:

- **Configuration files** - Upgrade user config files as your CLI/desktop app
    evolves (`.apprc`, `config.json`, `settings.yaml`)
- **Message queues & event streams** - Handle messages from multiple service
    versions publishing different schemas (Kafka, RabbitMQ, SQS)
- **ETL & data imports** - Import CSV/JSON/Excel files exported over years
    with evolving structures
- **ML model serving** - Manage feature schema evolution across model versions
    and A/B tests
- **API versioning** - Support multiple API versions with automatic
    request/response migration
- **Database migrations** - Transparently migrate legacy data on read without
    downtime
- **Data archival** - Process historical data dumps with various schema
    versions

See the [examples/](examples/) directory for complete, runnable code
demonstrating these patterns.

## When Not to Use

pyrmute may not be the right choice if you have:

- **High-throughput systems** - Runtime migration adds latency to hot paths.
    Use upfront batch migrations instead.
- **Multi-language services** - Python-only. Use Protobuf, Avro, or JSON
    Schema for polyglot architectures.
- **Existing schema registries** - Already using Confluent/AWS Glue? Stick
    with them for compatibility enforcement and governance.
- **Stable schemas** - Models rarely change? Traditional migration tools are
    simpler and more maintainable.
- **Database DDL changes** - pyrmute transforms data, not database schemas.
    Alembic/Flyway or other ORMs may still be needed to alter tables.

## Help

See [documentation](https://mferrera.github.io/pyrmute/) for complete guides
and API reference.

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

### Export JSON Schemas

```python
# Generate schemas for all versions
manager.dump_schemas("schemas/")
# Creates: User_v1.0.0.json, User_v2.0.0.json, User_v3.0.0.json

# Use separate files with $ref for nested models with 'enable_ref=True'.
manager.dump_schemas(
    "schemas/",
    separate_definitions=True,
    ref_template="https://api.example.com/schemas/{model}_v{version}.json"
)
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

## Real-World Examples

### Configuration File Evolution

```python
# Your CLI tool evolves over time
@manager.model("AppConfig", "1.0.0")
class AppConfigV1(BaseModel):
    api_key: str
    debug: bool = False

@manager.model("AppConfig", "2.0.0")
class AppConfigV2(BaseModel):
    api_key: str
    api_endpoint: str = "https://api.example.com"
    log_level: Literal["DEBUG", "INFO", "ERROR"] = "INFO"

@manager.migration("AppConfig", "1.0.0", "2.0.0")
def upgrade_config(data: dict) -> dict:
    return {
        "api_key": data["api_key"],
        "api_endpoint": "https://api.example.com",
        "log_level": "DEBUG" if data.get("debug") else "INFO",
    }

# Load and auto-upgrade user's config file
def load_config(config_path: Path) -> AppConfigV2:
    with open(config_path) as f:
        data = json.load(f)

    version = data.get("_version", "1.0.0")

    # Migrate to current version
    config = manager.migrate(
        data,
        "AppConfig",
        from_version=version,
        to_version="2.0.0"
    )

    # Save upgraded config with version tag
    with open(config_path, "w") as f:
        json.dump({**config.model_dump(), "_version": "2.0.0"}, f, indent=2)

    return config
```

### Message Queue Consumer

```python
# Handle messages from multiple service versions
@manager.model("OrderEvent", "1.0.0")
class OrderEventV1(BaseModel):
    order_id: str
    customer_email: str
    items: list[dict]  # Unstructured

@manager.model("OrderEvent", "2.0.0")
class OrderEventV2(BaseModel):
    order_id: str
    customer_email: str
    items: list[OrderItem]  # Structured
    total: Decimal

def process_message(message: dict, schema_version: str) -> None:
    # Migrate to current schema regardless of source version
    event = manager.migrate(
        message,
        "OrderEvent",
        from_version=schema_version,
        to_version="2.0.0"
    )
    # Process with current schema only
    fulfill_order(event)
```

### ETL Data Import

```python
# Import historical exports with evolving schemas
import csv

def import_customers(file_path: Path, file_version: str) -> None:
    with open(file_path) as f:
        reader = csv.DictReader(f)

        # Stream migration for memory efficiency
        for customer in manager.migrate_batch_streaming(
            reader,
            "Customer",
            from_version=file_version,
            to_version="3.0.0",
            chunk_size=1000
        ):
            database.save(customer)

# Handle files from different years
import_customers("exports/2022_customers.csv", "1.0.0")
import_customers("exports/2023_customers.csv", "2.0.0")
import_customers("exports/2024_customers.csv", "3.0.0")
```

### ML Model Serving

```python
# Route requests to appropriate model versions
class InferenceService:
    def predict(self, features: dict, request_version: str) -> BaseModel:
        # Determine target model version (A/B testing, gradual rollout, etc.)
        model_version = self.get_model_version(features["user_id"])

        # Migrate request to model's expected format
        model_input = manager.migrate(
            features,
            "PredictionRequest",
            from_version=request_version,
            to_version=model_version
        )

        # Run inference
        prediction = self.models[model_version].predict(model_input)

        # Normalize output for logging/analytics
        return manager.migrate(
            prediction,
            "PredictionResponse",
            from_version=model_version,
            to_version="3.0.0"
        )
```

See [examples/](examples/) for complete runnable code:

- `config_file_migration.py` - CLI/desktop app config file evolution
- `message_queue_consumer.py` - Kafka/RabbitMQ/SQS consumer handling multiple
    schemas
- `etl_data_import.py` - CSV/JSON/Excel import pipeline with historical data
- `ml_inference_pipeline.py` - ML model serving with feature evolution
- `advanced_features.py` - Complex Pydantic features (unions, nested models,
    validators)

## Contributing

For guidance on setting up a development environment and how to make a
contribution to pyrmute, see [Contributing to
pyrmute](https://mferrera.github.io/pyrmute/contributing/).

## Reporting a Security Vulnerability

See our [security
policy](https://github.com/mferrera/pyrmute/security/policy).
