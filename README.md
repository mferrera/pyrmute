# pyrmute
[![ci](https://img.shields.io/github/actions/workflow/status/mferrera/pyrmute/ci.yml?branch=main&logo=github&label=ci)](https://github.com/mferrera/pyrmute/actions?query=event%3Apush+branch%3Amain+workflow%3Aci)
[![pypi](https://img.shields.io/pypi/v/pyrmute.svg)](https://pypi.python.org/pypi/pyrmute)
[![versions](https://img.shields.io/pypi/pyversions/pyrmute.svg)](https://github.com/mferrera/pyrmute)
[![license](https://img.shields.io/github/license/mferrera/pyrmute.svg)](https://github.com/mferrera/pyrmute/blob/main/LICENSE)

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
- **Flexible schema export** - Generate JSON schemas for all versions with
  support for `$ref`, custom generators, and nested models
- **Production-ready** - Batch processing, parallel execution, and streaming
  support for large datasets
- **Only one dependency** - Pydantic

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

## Real-World Example

```python
from datetime import datetime
from pydantic import BaseModel, EmailStr
from pyrmute import ModelManager, ModelData

manager = ModelManager()


# API v1: Basic order
@manager.model("Order", "1.0.0")
class OrderV1(BaseModel):
    id: str
    items: list[str]
    total: float


# API v2: Add customer info
@manager.model("Order", "2.0.0")
class OrderV2(BaseModel):
    id: str
    items: list[str]
    total: float
    customer_email: EmailStr


# API v3: Structured items and timestamps
@manager.model("Order", "3.0.0")
class OrderItemV3(BaseModel):
    product_id: str
    quantity: int
    price: float


@manager.model("Order", "3.0.0")
class OrderV3(BaseModel):
    id: str
    items: list[OrderItemV3]
    total: float
    customer_email: EmailStr
    created_at: datetime


# Define migrations
@manager.migration("Order", "1.0.0", "2.0.0")
def add_customer_email(data: ModelData) -> ModelData:
    return {**data, "customer_email": "customer@example.com"}


@manager.migration("Order", "2.0.0", "3.0.0")
def structure_items(data: ModelData) -> ModelData:
    # Convert simple strings to structured items
    structured_items = [
        {
            "product_id": item,
            "quantity": 1,
            "price": data["total"] / len(data["items"])
        }
        for item in data["items"]
    ]
    return {
        **data,
        "items": structured_items,
        "created_at": datetime.now().isoformat()
    }

# Migrate old orders from your database
old_order = {"id": "123", "items": ["widget", "gadget"], "total": 29.99}
new_order = manager.migrate(old_order, "Order", "1.0.0", "3.0.0")
database.save(new_order)
```

## Contributing

For guidance on setting up a development environment and how to make a
contribution to pyrmute, see [Contributing to
pyrmute](https://mferrera.github.io/pyrmute/contributing/).

## Reporting a Security Vulnerability

See our [security
policy](https://github.com/mferrera/pyrmute/security/policy).
