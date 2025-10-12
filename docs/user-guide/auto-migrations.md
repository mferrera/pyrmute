# Auto-Migration

Auto-migration lets you skip writing migration functions for
backward-compatible changes. This guide covers when to use auto-migration, how
it works, and important limitations.

## What is Auto-Migration?

When you mark a model version as `backward_compatible=True`, pyrmute
automatically applies Pydantic's default values instead of requiring an
explicit migration function:

```python
from pydantic import BaseModel
from pyrmute import ModelManager

manager = ModelManager()


@manager.model("Config", "1.0.0")
class ConfigV1(BaseModel):
    timeout: int


@manager.model("Config", "2.0.0", backward_compatible=True)
class ConfigV2(BaseModel):
    timeout: int
    retries: int = 3  # New field with default


# No migration function needed!
config = manager.migrate({"timeout": 30}, "Config", "1.0.0", "2.0.0")
print(config)
# ConfigV2(timeout=30, retries=3)
```

**How it works:**

1. Old data is passed to the new model
2. Missing fields use Pydantic's default values
3. Extra fields are preserved
4. Result is validated against the new model

## When to Use Auto-Migration

### ✅ Safe for Auto-Migration

#### Adding Optional Fields with Defaults

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    email: str


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str
    email: str
    created_at: str = "2024-01-01T00:00:00Z"  # New with default
    is_active: bool = True  # New with default


# Old data automatically gets defaults
user = manager.migrate(
    {"name": "Alice", "email": "alice@example.com"},
    "User",
    "1.0.0",
    "2.0.0"
)
# Result: UserV2(name="Alice", email="alice@example.com",
#                created_at="2024-01-01T00:00:00Z", is_active=True)
```

#### Making Required Fields Optional

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    email: str  # Required


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str
    email: str | None = None  # Now optional


# Old data with email works
user1 = manager.migrate(
    {"name": "Alice", "email": "alice@example.com"},
    "User",
    "1.0.0",
    "2.0.0"
)
# Result: UserV2(name="Alice", email="alice@example.com")

# Old data without email also works (though v1 required it)
user2 = manager.migrate(
    {"name": "Bob"},
    "User",
    "1.0.0",
    "2.0.0"
)
# Result: UserV2(name="Bob", email=None)
```

#### Widening Field Types

```python
@manager.model("Config", "1.0.0")
class ConfigV1(BaseModel):
    port: int  # Only int


@manager.model("Config", "2.0.0", backward_compatible=True)
class ConfigV2(BaseModel):
    port: int | str  # Now accepts both


# Old int data still works
config = manager.migrate({"port": 8080}, "Config", "1.0.0", "2.0.0")
# Result: ConfigV2(port=8080)
```

#### Adding Fields with Factories

```python
from typing import List

@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str
    tags: List[str] = []  # Mutable default (handled by Pydantic)
    metadata: dict = {}


# Old data gets empty collections
user = manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")
# Result: UserV2(name="Alice", tags=[], metadata={})
```

### ❌ Requires Explicit Migration

#### Removing Fields

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    email: str
    deprecated_field: str


# ❌ DON'T mark backward_compatible - write migration!
@manager.model("User", "2.0.0")  # No backward_compatible
class UserV2(BaseModel):
    name: str
    email: str
    # deprecated_field removed


@manager.migration("User", "1.0.0", "2.0.0")
def remove_deprecated(data: ModelData) -> ModelData:
    """Explicitly handle field removal."""
    return {
        "name": data["name"],
        "email": data["email"]
        # deprecated_field intentionally dropped
    }
```

**Why?** Auto-migration would silently ignore the removed field. You should
explicitly decide what to do with it (log it, save it elsewhere, etc.).

#### Renaming Fields

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    user_name: str  # Old name


# ❌ DON'T mark backward_compatible - write migration!
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    username: str  # New name


@manager.migration("User", "1.0.0", "2.0.0")
def rename_field(data: ModelData) -> ModelData:
    """Map old field name to new."""
    return {
        "username": data["user_name"]
    }
```

**Why?** Auto-migration won't know `user_name` maps to `username`. The old
field would be lost and the new field would fail validation (missing required
field).

#### Changing Field Types (Narrowing)

```python
@manager.model("Config", "1.0.0")
class ConfigV1(BaseModel):
    timeout: str  # Was string


# ❌ DON'T mark backward_compatible - write migration!
@manager.model("Config", "2.0.0")
class ConfigV2(BaseModel):
    timeout: int  # Now int


@manager.migration("Config", "1.0.0", "2.0.0")
def convert_timeout(data: ModelData) -> ModelData:
    """Convert string to int."""
    return {
        "timeout": int(data["timeout"])
    }
```

**Why?** Pydantic will try to coerce the string to int, which might work for
"30" but fail for invalid values. Better to handle conversion explicitly.

#### Complex Transformations

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


# ❌ DON'T mark backward_compatible - write migration!
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    first_name: str
    last_name: str


@manager.migration("User", "1.0.0", "2.0.0")
def split_name(data: ModelData) -> ModelData:
    """Split name into components."""
    parts = data["name"].split(" ", 1)
    return {
        "first_name": parts[0],
        "last_name": parts[1] if len(parts) > 1 else ""
    }
```

**Why?** Auto-migration can't split the name field. You need custom logic.

#### Data Validation or Enrichment

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    email: str


# ❌ DON'T mark backward_compatible - write migration!
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    email: str
    email_verified: bool


@manager.migration("User", "1.0.0", "2.0.0")
def add_verification_status(data: ModelData) -> ModelData:
    """Check if email looks valid."""
    email = data.get("email", "")
    is_valid = "@" in email and "." in email
    return {
        **data,
        "email_verified": is_valid
    }
```

**Why?** Auto-migration would just use a default. You want custom logic based
on the data.

## How Auto-Migration Works

### Field Matching

Auto-migration matches fields by name:

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    age: int


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str      # Matched by name
    age: int       # Matched by name
    city: str = "Unknown"  # New, uses default

# Migration:
# 1. name: "Alice" → name: "Alice" (matched)
# 2. age: 30 → age: 30 (matched)
# 3. city: missing → city: "Unknown" (default)
```

### Handling Extra Fields

Extra fields from old data are preserved:

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    age: int
    legacy_field: str


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str
    age: int
    # legacy_field not in v2


user = manager.migrate(
    {"name": "Alice", "age": 30, "legacy_field": "old_data"},
    "User",
    "1.0.0",
    "2.0.0"
)
# Result includes legacy_field if model allows extra fields
```

If you want to explicitly drop fields, write a migration.

### Nested Models

Auto-migration works recursively with nested models:

```python
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str
    city: str


@manager.model("Address", "2.0.0", backward_compatible=True)
class AddressV2(BaseModel):
    street: str
    city: str
    postal_code: str = "00000"  # New field


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str
    address: AddressV2  # Uses AddressV2


# Both models marked backward_compatible
user = manager.migrate(
    {
        "name": "Alice",
        "address": {"street": "123 Main St", "city": "NYC"}
    },
    "User",
    "1.0.0",
    "2.0.0"
)
# Result: UserV2(
#   name="Alice",
#   address=AddressV2(
#     street="123 Main St",
#     city="NYC",
#     postal_code="00000"  # Default applied to nested model
#   )
# )
```

## Combining Auto-Migration with Explicit Migrations

You can mark a version as `backward_compatible=True` AND provide an explicit
migration. The explicit migration takes precedence:

```python
@manager.model("Config", "1.0.0")
class ConfigV1(BaseModel):
    timeout: int


@manager.model("Config", "2.0.0", backward_compatible=True)
class ConfigV2(BaseModel):
    timeout: int
    retries: int = 3


# Explicit migration overrides auto-migration
@manager.migration("Config", "1.0.0", "2.0.0")
def custom_defaults(data: ModelData) -> ModelData:
    """Use custom default instead of model default."""
    return {
        **data,
        "retries": 5  # Custom default instead of 3
    }


config = manager.migrate({"timeout": 30}, "Config", "1.0.0", "2.0.0")
# Result: ConfigV2(timeout=30, retries=5)  # Uses custom default
```

**Use cases:**

- Custom default values based on data
- Conditional logic during migration
- Data validation during migration
- Still mark as backward_compatible to document intent

## Validation During Auto-Migration

Auto-migration validates the result against the target model:

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str
    age: int  # Required, no default!


# This will fail!
try:
    user = manager.migrate(
        {"name": "Alice"},
        "User",
        "1.0.0",
        "2.0.0"
    )
except ValidationError:
    print("Missing required field: age")
```

**Key point:** Auto-migration only works when old data is valid for the new
model (with defaults applied). If validation fails, you need an explicit
migration.

## Testing Auto-Migration

Always test auto-migrations:

```python
def test_auto_migration() -> None:
    """Test that auto-migration applies defaults correctly."""
    results = manager.test_migration(
        "Config",
        "1.0.0",
        "2.0.0",
        test_cases=[
            # Basic case
            (
                {"timeout": 30},
                {"timeout": 30, "retries": 3}
            ),
            # With extra fields
            (
                {"timeout": 30, "extra_field": "value"},
                {"timeout": 30, "retries": 3, "extra_field": "value"}
            ),
            # Different timeout value
            (
                {"timeout": 60},
                {"timeout": 60, "retries": 3}
            ),
        ]
    )
    results.assert_all_passed()
```

## Decision Flowchart

Use this flowchart to decide whether to use auto-migration:

```
Is the change adding optional fields with defaults?
├─ YES → Use backward_compatible=True
└─ NO → Continue

Is the change making required fields optional?
├─ YES → Use backward_compatible=True
└─ NO → Continue

Is the change widening a type (int → int | str)?
├─ YES → Use backward_compatible=True
└─ NO → Continue

Does the change involve:

- Removing fields?
- Renaming fields?
- Narrowing types?
- Complex transformations?
- Custom logic?

├─ YES → Write explicit migration
└─ NO → Use backward_compatible=True (but test carefully!)
```

## Common Patterns

### Progressive Enhancement

Add optional features over time:

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    email: str


@manager.model("User", "1.1.0", backward_compatible=True)
class UserV1_1(BaseModel):
    name: str
    email: str
    phone: str | None = None  # Optional feature


@manager.model("User", "1.2.0", backward_compatible=True)
class UserV1_2(BaseModel):
    name: str
    email: str
    phone: str | None = None
    avatar_url: str | None = None  # Another optional feature

# All versions are backward compatible
# v1.0.0 → v1.1.0 → v1.2.0 all work without migrations
```

### Soft Deprecation

Make fields optional before removing them:

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    legacy_id: str  # Required


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str
    legacy_id: str | None = None  # Now optional (soft deprecation)


# Later, in v3.0.0, remove it entirely with explicit migration
@manager.model("User", "3.0.0")
class UserV3(BaseModel):
    name: str
    # legacy_id removed


@manager.migration("User", "2.0.0", "3.0.0")
def remove_legacy_id(data: ModelData) -> ModelData:
    return {"name": data["name"]}
```

### Feature Flags as Fields

```python
@manager.model("Config", "1.0.0")
class ConfigV1(BaseModel):
    api_key: str


@manager.model("Config", "1.1.0", backward_compatible=True)
class ConfigV1_1(BaseModel):
    api_key: str
    enable_feature_x: bool = False  # Feature flag


@manager.model("Config", "1.2.0", backward_compatible=True)
class ConfigV1_2(BaseModel):
    api_key: str
    enable_feature_x: bool = False
    enable_feature_y: bool = False  # Another feature flag

# New features opt-in by default
```

## Limitations and Gotchas

### Pydantic's Validation Still Applies

```python
from pydantic import field_validator


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    email: str


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email")
        return v


# This will fail if v1 data has invalid emails!
try:
    user = manager.migrate(
        {"email": "not-an-email"},
        "User",
        "1.0.0",
        "2.0.0"
    )
except ValidationError:
    print("Validation failed on auto-migration")
```

### Default Factories Run for Each Instance

```python
from datetime import datetime


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str
    created_at: str = datetime.now().isoformat()  # ⚠️ Evaluated at class definition


# Better: Use default_factory
from pydantic import Field


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
```

### Silent Field Loss

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    important_data: str


@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str
    # important_data removed (forgotten?)


# Data is silently lost!
user = manager.migrate(
    {"name": "Alice", "important_data": "CRITICAL"},
    "User",
    "1.0.0",
    "2.0.0"
)
# important_data is gone, no error raised
```

**Solution:** Write explicit migrations for field removals to handle them
deliberately.

## Best Practices

1. **Document backward compatibility** in model docstrings
2. **Test auto-migrations** thoroughly with realistic data
3. **Use explicit migrations** when in doubt
4. **Don't rely on silent field dropping** - be explicit about removals
5. **Validate assumptions** - just because it's backward compatible doesn't
   mean it's correct
6. **Monitor production** - track migration failures

## Next Steps

Now that you understand auto-migration:

**Continue learning:**

- [Writing Migrations](writing-migrations.md) - When you need explicit control
- [Testing Migrations](testing-migrations.md) - Validate auto-migrations work
    correctly
- [Nested Models](nested-models.md) - Auto-migration with nested Pydantic
    models

**Best practices:**

- [Versioning Strategy](../best-practices/versioning-strategy.md) - When to
    use auto-migration vs explicit

**Related topics:**
- [Registering Models](registering-models.md) - The `backward_compatible` flag
- [Schema Generation](schema-generation.md) - Document backward compatibility
    in schemas

**API Reference:**

- [`ModelManager` API](../reference/model-manager.md) - Complete
    `ModelManager` details
- [Exceptions](../reference/exceptions.md) - Exceptions pyrmute raises
- [Types](../reference/types.md) - Type alises exported by pyrmute
