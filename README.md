# pyrmute

[![ci](https://img.shields.io/github/actions/workflow/status/mferrera/pyrmute/ci.yml?branch=main&logo=github&label=ci)](https://github.com/mferrera/pyrmute/actions?query=event%3Apush+branch%3Amain+workflow%3Aci)
[![pypi](https://img.shields.io/pypi/v/pyrmute.svg)](https://pypi.python.org/pypi/pyrmute)
[![versions](https://img.shields.io/pypi/pyversions/pyrmute.svg)](https://github.com/mferrera/pyrmute)
[![license](https://img.shields.io/github/license/pyrmute/pyrmute.svg)](https://github.com/mferrera/pyrmute/blob/main/LICENSE)

Pydantic model migrations and schema management with semantic versioning.

Pyrmute helps you evolve your data models over time without breaking changes.
Version your Pydantic models, define migrations between versions, and
automatically transform legacy data to current schemas. Export JSON schemas
for all versions to maintain API contracts.

**Key features:**
- **Version your models** - Use semantic versioning to track model evolution
- **Automatic migrations** - Chain migrations across multiple versions (1.0.0 → 2.0.0 → 3.0.0)
- **Validated transformations** - Migrations return validated Pydantic models by default
- **Schema export** - Generate JSON schemas for all versions, with support for `$ref` to external schemas and custom schema generators
- **Nested model support** - Automatically migrates nested Pydantic models

## Help

See [documentation](https://mferrera.github.io/pyrmute/) for more details.

## Installation

Install using `pip install -U pyrmute`.

## Simple Example

```python
from pydantic import BaseModel
from pyrmute import ModelManager, MigrationData

manager = ModelManager()


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    """Version 1.0.0: Initial user model."""
    name: str
    age: int


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    """Version 2.0.0: Split name into first/last."""
    first_name: str
    last_name: str
    age: int


@manager.model("User", "3.0.0")
class UserV3(BaseModel):
    """Version 3.0.0: Add email, make age optional."""
    first_name: str
    last_name: str
    email: str
    age: int | None = None


# Define migrations
@manager.migration("User", "1.0.0", "2.0.0")
def split_name(data: MigrationData) -> MigrationData:
    parts = data["name"].split(" ", 1)
    return {
        "first_name": parts[0],
        "last_name": parts[1] if len(parts) > 1 else "",
        "age": data["age"],
    }


@manager.migration("User", "2.0.0", "3.0.0")
def add_email(data: MigrationData) -> MigrationData:
    return {**data, "email": f"{data['first_name'].lower()}@example.com"}


# Migrate old data forward, from raw data or dumped from the Pydantic model
legacy_data = {"name": "John Doe", "age": 30}
# Returns a validated Pydantic model
current_user = manager.migrate(legacy_data, "User", "1.0.0", "3.0.0")

print(current_user)
# first_name='John' last_name='Doe' email='john@example.com' age=30

# Export schemas for all versions
manager.dump_schemas("schemas/")
# Creates: schemas/User_v1.0.0.json, schemas/User_v2.0.0.json, schemas/User_v3.0.0.json
```

## Contributing

For guidance on setting up a development environment and how to make a
contribution to pyrmute, see
[Contributing to pyrmute](https://mferrera.github.io/pyrmute/contributing/).

## Reporting a Security Vulnerability

See our [security policy](https://github.com/mferrera/pyrmute/security/policy).
