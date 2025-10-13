# Registering Models

Model registration is how you tell pyrmute about your versioned Pydantic
models. This guide covers registration patterns, versioning strategies, and
best practices.

## Basic Registration

Register models using the `@manager.model()` decorator:

```python
from pydantic import BaseModel
from pyrmute import ModelManager

manager = ModelManager()


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    """User model version 1.0.0."""
    name: str
    email: str
```

**Key points:**

- First argument is the model name (string)
- Second argument is the semantic version (string)
- The decorator returns the class unchanged - it's still a normal Pydantic
    model
- Multiple versions of the same model have different class names but same
    model name

## Model Names vs. Class Names

The model name and class name serve different purposes:

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):  # Class name
    name: str


@manager.model("User", "2.0.0")
class UserV2(BaseModel):  # Different class name
    first_name: str
    last_name: str


# Both registered under model name "User"
# Access by model name and version:
user_v1 = manager.get("User", "1.0.0")  # Returns UserV1 class
user_v2 = manager.get("User", "2.0.0")  # Returns UserV2 class
```

**Model name** - Used for:

- Grouping versions together
- Referring to models in migrations
- Organizing related schemas

**Class name** - Used for:

- Python code references
- Type hints
- Import statements

## Versioning Models

### Semantic Versioning

pyrmute uses [Semantic Versioning](https://semver.org/):

```python
# MAJOR.MINOR.PATCH
@manager.model("User", "1.0.0")  # Initial version
@manager.model("User", "1.1.0")  # Added optional field
@manager.model("User", "2.0.0")  # Breaking change
@manager.model("User", "2.0.1")  # Bug fix (rarely needed)
```

**Version guidelines:**

- **MAJOR** (1.0.0 → 2.0.0): Breaking changes, removed fields, type changes
- **MINOR** (1.0.0 → 1.1.0): Backward-compatible additions
- **PATCH** (1.0.0 → 1.0.1): Bug fixes, no schema changes (rare for models)

See [Versioning Strategy](../best-practices/versioning-strategy.md) for
detailed guidelines.

### Version Objects

You can use `ModelVersion` objects instead of strings:

```python
from pyrmute import ModelVersion

version = ModelVersion(major=1, minor=0, patch=0)


@manager.model("User", version)
class UserV1(BaseModel):
    name: str


# Or parse from string
@manager.model("User", ModelVersion.parse("2.0.0"))
class UserV2(BaseModel):
    first_name: str
    last_name: str
```

Prefer strings for simplicity, but `ModelVersion` objects are useful for:

- Programmatic version construction
- Version comparison logic
- Version arithmetic

## Naming Conventions

### Model Names

Use clear, descriptive names:

```python
# ✅ GOOD - Clear and descriptive
@manager.model("User", "1.0.0")
@manager.model("Order", "1.0.0")
@manager.model("ShippingAddress", "1.0.0")
@manager.model("PaymentMethod", "1.0.0")

# ❌ BAD - Too generic or unclear
@manager.model("Data", "1.0.0")
@manager.model("Record", "1.0.0")
@manager.model("Entity", "1.0.0")
```

**Best practices:**

- Use PascalCase for model names
- Be specific (not just "Config" but "DatabaseConfig")
- Match domain terminology
- Keep names stable across versions

### Class Names

Include version in class name for clarity:

```python
# ✅ GOOD - Clear version suffix
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    pass


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    pass


# ✅ ALSO GOOD - Explicit version
@manager.model("User", "1.0.0")
class UserV1_0_0(BaseModel):
    pass


# ⚠️ ACCEPTABLE - But can be confusing
@manager.model("User", "1.0.0")
class User(BaseModel):  # No version indicator
    pass


@manager.model("User", "2.0.0")
class UserLatest(BaseModel):  # Unclear
    pass
```

**Recommended pattern:**

- Suffix with major version: `UserV1`, `UserV2`, `UserV3`
- For minor versions: `UserV1_1`, `UserV1_2` (if needed)
- Full version: `UserV1_0_0` (verbose but unambiguous)

## Registration Options

### enable_ref

Control whether a model can be referenced in separate schema files:

```python
# Model will be inlined in schemas (default)
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str
    city: str


# Model can be referenced via $ref
@manager.model("Country", "1.0.0", enable_ref=True)
class CountryV1(BaseModel):
    code: str
    name: str
```

**When to use `enable_ref=True`:**

- Model is reused across many other models
- You want separate schema files for organization
- You're generating OpenAPI specs with shared components

**When to keep default (`enable_ref=False`):**

- Model is only used in one place
- You prefer simpler, self-contained schemas
- You're not using `separate_definitions=True` in schema export

Example with nested models:

```python
@manager.model("Address", "1.0.0", enable_ref=True)
class AddressV1(BaseModel):
    street: str
    city: str


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1  # Will use $ref if separate_definitions=True

# When exporting with separate_definitions=True:
# User schema: {"name": "...", "address": {"$ref": "Address_v1.0.0.json"}}
# Address schema: Separate Address_v1.0.0.json file
```

### backward_compatible

Mark a version as backward compatible for auto-migration:

```python
@manager.model("Config", "1.0.0")
class ConfigV1(BaseModel):
    timeout: int


# Version 2 can accept v1 data
@manager.model("Config", "2.0.0", backward_compatible=True)
class ConfigV2(BaseModel):
    timeout: int
    retries: int = 3  # New optional field with default


# No migration function needed!
config = manager.migrate({"timeout": 30}, "Config", "1.0.0", "2.0.0")
# Result: ConfigV2(timeout=30, retries=3)
```

**When to use `backward_compatible=True`:**

- Only adding fields with defaults
- Making required fields optional
- Changes are purely additive

**When NOT to use it:**

- Removing fields (they'll be silently ignored)
- Renaming fields (won't work)
- Changing field types (will fail validation)
- Any complex transformation needed

See [Auto-Migrations](auto-migrations.md) for guidance.

## Organization Patterns

### Single File (Small Projects)

For small projects, keep everything in one file:

```python
# models.py
from pydantic import BaseModel
from pyrmute import ModelManager, ModelData

manager = ModelManager()


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    first_name: str
    last_name: str


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user(data: ModelData) -> ModelData:
    parts = data["name"].split(" ", 1)
    return {
        "first_name": parts[0],
        "last_name": parts[1] if len(parts) > 1 else ""
    }
```

### Separate Files by Domain (Medium Projects)

```
myapp/
├── models/
│   ├── __init__.py
│   ├── _manager.py     # ModelManager instance
│   ├── user.py         # User models and migrations
│   ├── order.py        # Order models and migrations
│   └── product.py      # Product models and migrations
```

```python
# models/_manager.py
from pyrmute import ModelManager

manager = ModelManager()

# models/user.py
from pydantic import BaseModel
from pyrmute import ModelData

from ._manager import manager


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    first_name: str
    last_name: str


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user(data: ModelData) -> ModelData:
    parts = data["name"].split(" ", 1)
    return {
        "first_name": parts[0],
        "last_name": parts[1] if len(parts) > 1 else ""
    }

# models/__init__.py
from ._manager import manager
from .user import UserV1, UserV2
from .order import OrderV1, OrderV2

__all__ = ["manager", "UserV1", "UserV2", "OrderV1", "OrderV2"]
```

### Separate Files by Version (Large Projects)

```
myapp/
├── models/
│   ├── __init__.py
│   ├── _manager.py
│   ├── user/
│   │   ├── __init__.py
│   │   ├── v1.py       # UserV1
│   │   ├── v2.py       # UserV2
│   │   └── v3.py       # UserV3
│   └── migrations/
│       ├── __init__.py
│       └── user.py     # All user migrations
```

```python
# models/user/v1.py
from pydantic import BaseModel
from myapp.models import manager


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


# models/user/v2.py
from pydantic import BaseModel
from myapp.models import manager


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    first_name: str
    last_name: str


# models/migrations/user.py
from pyrmute import ModelData
from myapp.models import manager


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_v1_to_v2(data: ModelData) -> ModelData:
    parts = data["name"].split(" ", 1)
    return {
        "first_name": parts[0],
        "last_name": parts[1] if len(parts) > 1 else ""
    }

# models/__init__.py
from ._manager import manager
from .user.v1 import UserV1
from .user.v2 import UserV2

# Import migrations to register them
from .migrations import user

__all__ = ["manager", "UserV1", "UserV2"]
```

## Retrieving Models

### Get by Version

```python
# Get specific version
UserV1 = manager.get("User", "1.0.0")
UserV2 = manager.get("User", "2.0.0")

# Create instances
user_v1 = UserV1(name="Alice")
user_v2 = UserV2(first_name="Alice", last_name="Smith")
```

### Get Latest Version

```python
# Get the most recent version
UserLatest = manager.get_latest("User")

# Useful when you always want current version
user = UserLatest(first_name="Alice", last_name="Smith")
```

### List All Versions

```python
# Get all versions for a model
versions = manager.list_versions("User")
# Returns: [ModelVersion(1.0.0), ModelVersion(2.0.0)]

# List all registered models
models = manager.list_models()
# Returns: ["User", "Order", "Product"]
```

### Check Model Existence

```python
# Check if model and version exist
try:
    model = manager.get("User", "1.0.0")
except ModelNotFoundError:
    print("Model not found")

# Or validate migration paths
if manager.has_migration_path("User", "1.0.0", "2.0.0"):
    user = manager.migrate(data, "User", "1.0.0", "2.0.0")
else:
    print("Cannot migrate to v2")
```

## Common Patterns

### Aliasing Current Version

```python
# Register all versions
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    first_name: str
    last_name: str


# Alias for convenience in application code
User = UserV2  # Current version


# Use throughout application
def create_user(first_name: str, last_name: str) -> User:
    return User(first_name=first_name, last_name=last_name)
```

### Multiple Managers

Use separate managers for different domains:

```python
# users/models.py
user_manager = ModelManager()


@user_manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


# orders/models.py
order_manager = ModelManager()


@order_manager.model("Order", "1.0.0")
class OrderV1(BaseModel):
    order_id: str
```

This is useful for:

- Microservices (each service has its own manager)
- Plugin systems (each plugin registers models)
- Testing (isolated managers per test)

### Conditional Registration

Register models conditionally:

```python
manager = ModelManager()


# Always register v1
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


# Only register v2 in production
if not DEBUG:
    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        first_name: str
        last_name: str
```

!!! warning "Be Careful with Conditional Registration"
    Conditional registration can make testing harder and behavior
    unpredictable. Only use when you have a clear reason (feature flags,
    environment-specific models, etc.).

## Registration Order

Models and migrations can be registered in any order:

```python
# ✅ Models then migrations
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    first_name: str
    last_name: str


@manager.migration("User", "1.0.0", "2.0.0")
def migrate(data: ModelData) -> ModelData:
    return {**data}


# ✅ Migrations then models (also works)
@manager.migration("User", "1.0.0", "2.0.0")
def migrate(data: ModelData) -> ModelData:
    return {**data}


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    first_name: str
    last_name: str
```

**Why it works:** Registration happens at module import time. By the time you
call `migrate()`, all decorators have executed.

**Best practice:** Define models before migrations for readability.

## Common Mistakes

### Reusing Class Names

```python
# ❌ BAD - Same class name for different versions
@manager.model("User", "1.0.0")
class User(BaseModel):
    name: str


@manager.model("User", "2.0.0")
class User(BaseModel):  # Overwrites previous User!
    first_name: str
    last_name: str


# ✅ GOOD - Different class names
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    first_name: str
    last_name: str
```

### Forgetting to Register

```python
# ❌ BAD - Forgot decorator
class UserV1(BaseModel):
    name: str


# Later...
manager.migrate(data, "User", "1.0.0", "2.0.0")  # ModelNotFoundError!


# ✅ GOOD - Always use decorator
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
```

### Inconsistent Model Names

```python
# ❌ BAD - Inconsistent naming
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    pass


@manager.model("Users", "2.0.0")  # Different name!
class UserV2(BaseModel):
    pass


# ✅ GOOD - Consistent naming
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    pass


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    pass
```

### Duplicate Versions

```python
# ❌ BAD - Same version registered twice
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "1.0.0")  # Error!
class UserV1Fixed(BaseModel):
    name: str
    email: str


# ✅ GOOD - Use new version
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "1.1.0")  # New version
class UserV1_1(BaseModel):
    name: str
    email: str
```

## Next Steps

Now that you understand model registration:

**Continue learning:**

- [Writing Migrations](writing-migrations.md) - Transform data between your
    registered versions
- [Auto-Migrations](auto-migrations.md) - Skip migrations for
    backward-compatible changes
- [Schema Generation](schema-generation.md) - Export JSON schemas from
    registered models

**Advanced topics:**

- [Nested Models](nested-models.md) - Register models that contain other
    models
- [Discriminated Unions](../advanced/discriminated-unions.md) - Register
    polymorphic model types
- [Migration Hooks](../advanced/migration-hooks.md) - Register
    before, after, and on error migration hooks for observability

**API Reference:**

- [`ModelManager` API](../reference/model-manager.md) - Complete
    `ModelManager` details
- [Exceptions](../reference/exceptions.md) - Exceptions pyrmute raises
- [Types](../reference/types.md) - Type alises exported by pyrmute
