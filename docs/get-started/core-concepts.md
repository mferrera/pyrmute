# Core Concepts

Understanding these core concepts will help you use pyrmute effectively.

## The Version-Migration-Model Triangle

pyrmute centers on three interconnected concepts:

```
┌────────┐
│ Models │ ←── Versioned Pydantic models
└────┬───┘
     │
     ├──── Version: "1.0.0", "2.0.0", etc.
     │
┌────┴───────┐
│ Migrations │ ←── Functions transforming data
└────┬───────┘
     │
     └──── Migration Path: 1.0.0 -> 2.0.0 -> 3.0.0
```

### Models

Models are Pydantic `BaseModel` classes registered with semantic versions:

```python
from pydantic import BaseModel
from pyrmute import ModelManager, ModelData

manager = ModelManager()


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    age: int
```

**Key points:**

- Each version is a separate class
- Versions follow [Semantic Versioning](https://semver.org/)
- Models are regular Pydantic models - no special requirements

### Migrations

Migrations are functions that transform data from one version to another:

```python
@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user(data: ModelData) -> ModelData:
    """Transform v1 data to v2 format.

    `ModelData` is a type exported by pyrmute. It is a type alias over
    dict[str, Any].
    """
    parts = data["name"].split(" ", 1)
    return {
        "first_name": parts[0],
        "last_name": parts[1] if len(parts) > 1 else "",
        "age": data["age"],
    }
```

**Key points:**

- Take a `ModelData` (dict), return a `ModelData` (dict)
- Pure functions - no side effects
- Can be chained automatically

### Migration Paths

pyrmute automatically chains migrations to move data across multiple versions:

```python
# You define individual steps
@manager.migration("User", "1.0.0", "2.0.0")
def step1(data: ModelData): ...


@manager.migration("User", "2.0.0", "3.0.0")
def step2(data: ModelData): ...


# pyrmute chains them automatically
user = manager.migrate(old_data, "User", "1.0.0", "3.0.0")
# Executes: old_data -> step1 -> step2 -> UserV3
```

## The ModelManager

[`ModelManager`](../reference/model-manager.md) is the central interface to
pyrmute:

```python
from pydantic import BaseModel
from pyrmute import ModelManager

# Create once, typically at module level
manager = ModelManager()


# Register models
@manager.model("User", "1.0.0")
class UserV1(BaseModel): ...


# Register migrations
@manager.migration("User", "1.0.0", "2.0.0")
def migrate(data: ModelData): ...


# Use anywhere
user = manager.migrate(data, "User", "1.0.0", "2.0.0")
```

**Key points:**

- One manager per application (usually)
- Thread-safe for reads (migrations)
- Models and migrations registered at module import time

## When Migrations Run vs. Don't Run

Understanding when migrations execute is crucial:

### Explicit Migrations

When you define a migration function, it **always** runs:

```python
@manager.migration("User", "1.0.0", "2.0.0")
def explicit_migration(data: ModelData) -> ModelData:
    return {"name": data["name"].upper()}  # Always runs


user = manager.migrate(data, "User", "1.0.0", "2.0.0")
# Your function is called
```

### Auto-Migration

When a model is marked `backward_compatible=True`, pyrmute uses Pydantic's
defaults **instead of** requiring a migration function:

```python
@manager.model("Config", "1.0.0")
class ConfigV1(BaseModel):
    timeout: int

@manager.model("Config", "2.0.0", backward_compatible=True)
class ConfigV2(BaseModel):
    timeout: int
    retries: int = 3  # New field with default


# No migration function needed!
config = manager.migrate({"timeout": 30}, "Config", "1.0.0", "2.0.0")
# Result: ConfigV2(timeout=30, retries=3)
```

**Key points:**

- `backward_compatible=True` means "old data is valid for this version"
- Missing fields use their Pydantic defaults
- Explicit migrations override auto-migration

### Priority Order

When migrating from version A to B:

1. **Explicit migration** - If defined, always runs
2. **Auto-migration** - If `backward_compatible=True` on version B
3. **Error** - If neither exists, migration fails

## Data Flow

Here's what happens when you call `migrate()`:

```
Input Data (ModelData/dict)
    ↓
Validation: Is this the right version?
    ↓
Migration Chain: Transform through versions
    ↓
    ├─ Version 1.0.0 → 2.0.0 (explicit or auto)
    ├─ Version 2.0.0 → 3.0.0 (explicit or auto)
    ├─ Version 3.0.0 → 4.0.0 (explicit or auto)
    └─ ...
    ↓
Target Model Validation (Pydantic `target_model.model_validate(data)`)
    ↓
Return: Validated Pydantic model instance
```

**At each step:**

1. Find migration function or check `backward_compatible`
2. Execute transformation
3. Pass result to next step

**After all steps:**

1. Validate result against target model
2. Return typed Pydantic instance

!!! important "Validation Happens Once"
    Validation occurs **only at the end** of the migration chain, not between
    steps. This improves performance but means intermediate data might be
    invalid.

    **Why this matters:** If step 2 of a 3-step migration produces bad data,
    you won't know until the final validation fails.

    **If you need it:** You can validate intermediate steps within your
    migration functions:

    ```python
        @manager.migration("User", "1.0.0", "2.0.0")
        def validate_intermediate(data: ModelData) -> ModelData:
            result = {...}  # Your transformation

            # Validate if critical
            IntermediateModel.model_validate(result)

            return result
    ```

## Type Safety

pyrmute provides type-safe migrations:

```python
from pydantic import BaseModel

# Without type hint - returns BaseModel
user: BaseModel = manager.migrate(data, "User", "1.0.0", "2.0.0")

# With type hint - type checkers know the exact type
user: UserV2 = manager.migrate_as(
    data,
    "User",
    "1.0.0",
    "2.0.0",
    UserV2  # Explicit type
)
```

**Benefits:**

- IDE autocomplete works
- Type checkers (mypy, pyright) verify correctness
- Runtime validation via Pydantic

Unfortunately, `migrate()` falls victim to the fact that Python is
fundamentally a dynamic language. Type checkers cannot guarantee static types
for returned models unless they have an input argument to consult for the
type.

## Semantic Versioning

pyrmute uses [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
  2  . 1  . 3
```

**Guidelines:**

- **PATCH** (1.0.0 → 1.0.1): Bug fixes, no schema changes
- **MINOR** (1.0.0 → 1.1.0): Backward-compatible additions (new optional
    fields)
- **MAJOR** (1.0.0 → 2.0.0): Breaking changes (removed fields, type changes)

See [Versioning Strategy](../best-practices/versioning-strategy.md) for
detailed guidelines.

## Common Patterns

### Pattern 1: Adding a Field

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


# Option A: Explicit migration
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    email: str


@manager.migration("User", "1.0.0", "2.0.0")
def add_email(data: ModelData) -> ModelData:
    return {**data, "email": "unknown@example.com"}


# Option B: Auto-migration with default
@manager.model("User", "2.0.0", backward_compatible=True)
class UserV2(BaseModel):
    name: str
    email: str = "unknown@example.com"
```

### Pattern 2: Renaming a Field

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    user_name: str


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    username: str  # Renamed


@manager.migration("User", "1.0.0", "2.0.0")
def rename_field(data: ModelData) -> ModelData:
    return {"username": data["user_name"]}
```

### Pattern 3: Splitting a Field

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    first_name: str
    last_name: str


@manager.migration("User", "1.0.0", "2.0.0")
def split_name(data: ModelData) -> ModelData:
    parts = data["name"].split(" ", 1)
    return {
        "first_name": parts[0],
        "last_name": parts[1] if len(parts) > 1 else "",
    }
```

## What's Not Included

pyrmute focuses on data transformation. It does **not**:

- ❌ Migrate database schemas (use Alembic, Flyway, etc.)
- ❌ Store version metadata in your data (you handle that)
- ❌ Automatically detect version from data (you must specify)

pyrmute assumes you know:

1. What version your data is
2. What version you want to migrate to

## Next Steps

Now that you understand the core concepts:

- [Install](install.md) - Install pyrmute
- [First Migration](first-migration.md) - Build a complete example
    step-by-step
