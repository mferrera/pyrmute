# Your First Migration

This tutorial walks you through creating a complete migration from scratch.
We'll build a user management system that evolves over time.

## The Scenario

You're building a CLI tool that stores user preferences in a JSON file. As
your tool evolves, you need to update the configuration format without
breaking existing users' files. You have to plan for the fact that you may
iterate on the configuration format multiple time before a user goes through
an upgrade.

## Step 1: Initial Version

Let's start with a simple configuration:

```python
from pydantic import BaseModel
from pyrmute import ModelManager, ModelData

# Create the manager
manager = ModelManager()


# Version 1.0.0: Basic config
@manager.model("AppConfig", "1.0.0")
class AppConfigV1(BaseModel):
    """Initial configuration format."""
    api_key: str
    debug: bool = False
```

At this point, your users have config files like:

```json
{
    "api_key": "sk-123456",
    "debug": true
}
```

## Step 2: Save and Load Functions

Let's create helper functions to work with config files:

```python
import json
from pathlib import Path

def save_config(config: AppConfigV1, path: Path) -> None:
    """Save config to file with version metadata."""
    data = config.model_dump()
    data["_version"] = "1.0.0"

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_config(path: Path) -> AppConfigV1:
    """Load config from file."""
    with open(path) as f:
        data = json.load(f)

    # For now, just validate as v1
    return AppConfigV1.model_validate(data)
```

**Key point:** We're storing the version in the file (`_version` field) so we
can migrate it later.

## Step 3: Evolution - Adding Features

Six months later, you want to add new features:

1. Remove the `debug` boolean - too simplistic
2. Add a proper `log_level` with multiple options
3. Add an `api_endpoint` for self-hosted users

```python
from typing import Literal


# Version 2.0.0: Better logging and endpoint support
@manager.model("AppConfig", "2.0.0")
class AppConfigV2(BaseModel):
    """Improved configuration with better logging and endpoint control."""
    api_key: str
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_endpoint: str = "https://api.example.com"
```

## Step 4: Define the Migration

Now we need to transform v1 configs to v2 format:

```python
@manager.migration("AppConfig", "1.0.0", "2.0.0")
def upgrade_to_v2(data: ModelData) -> ModelData:
    """Migrate v1 config to v2 format.

    Changes:
    - Remove 'debug' field
    - Add 'log_level' based on old debug value
    - Add 'api_endpoint' with default
    """
    # Convert debug boolean to log level
    log_level = "DEBUG" if data.get("debug", False) else "INFO"

    return {
        "api_key": data["api_key"],
        "log_level": log_level,
        "api_endpoint": "https://api.example.com",
    }
```

**Key points:**

- Migration takes a `ModelData` (dict), returns a `ModelData` (dict)
- We handle the `debug` field carefully (might not exist)
- We provide sensible defaults for new fields

## Step 5: Update Load Function

Now update the load function to handle both versions:

```python
def load_config(path: Path) -> AppConfigV2:
    """Load config from file, migrating if necessary."""
    with open(path) as f:
        data = json.load(f)

    # Get version from file (default to 1.0.0 for old files)
    version = data.get("_version", "1.0.0")

    # Remove metadata before migration
    data.pop("_version", None)

    # Migrate to current version
    config = manager.migrate(
        data,
        "AppConfig",
        from_version=version,
        to_version="2.0.0"
    )

    return config
```

## Step 6: Test the Migration

pyrmute includes built-in testing utilities to validate your migrations:

```python
# Test the migration with various scenarios
results = manager.test_migration(
    "AppConfig",
    from_version="1.0.0",
    to_version="2.0.0",
    test_cases=[
        # (source_data, expected_output)
        (
            {"api_key": "sk-old", "debug": True},
            {
                "api_key": "sk-old",
                "log_level": "DEBUG",
                "api_endpoint": "https://api.example.com"
            }
        ),
        (
            {"api_key": "sk-test", "debug": False},
            {
                "api_key": "sk-test",
                "log_level": "INFO",
                "api_endpoint": "https://api.example.com"
            }
        ),
        # Test missing debug field
        (
            {"api_key": "sk-minimal"},
            {
                "api_key": "sk-minimal",
                "log_level": "INFO",
                "api_endpoint": "https://api.example.com"
            }
        ),
    ]
)

# Check results
if results.all_passed:
    print("✓ All migration tests passed!")
else:
    print(f"✗ {len(results.failures)} test(s) failed")
    for failure in results.failures:
        print(failure)

# Or use in test suites with pytest/unittest
results.assert_all_passed()  # Raises AssertionError if any failed
```

**Key benefits of built-in testing:**

- Validates migration logic before production
- Catches edge cases (missing fields, etc.)
- Clear error messages showing expected vs actual
- Works with pytest/unittest via `assert_all_passed()`

You can also add descriptions to make failures clearer:

```python
from pyrmute import MigrationTestCase

results = manager.test_migration(
    "AppConfig",
    "1.0.0",
    "2.0.0",
    test_cases=[
        MigrationTestCase(
            source={"api_key": "sk-test", "debug": True},
            target={
                "api_key": "sk-test",
                "log_level": "DEBUG",
                "api_endpoint": "https://api.example.com"
            },
            description="debug=True should map to DEBUG log level"
        ),
        MigrationTestCase(
            source={"api_key": "sk-test"},
            target={
                "api_key": "sk-test",
                "log_level": "INFO",
                "api_endpoint": "https://api.example.com"
            },
            description="Missing debug field should default to INFO"
        ),
    ]
)
```

## Step 7: Upgrade User Files (Optional)

You might want to automatically upgrade user files:

```python
def upgrade_config_file(path: Path) -> None:
    """Upgrade a config file to the latest version."""
    # Load (this migrates automatically)
    config = load_config(path)

    # Save with new version
    save_config(config, path)

    print(f"✓ Upgraded {path} to v2.0.0")


def save_config(config: AppConfigV2, path: Path, version: str) -> None:
    """Save config to file."""
    data = config.model_dump()
    data["_version"] = version

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
```

## Complete Example

Here's the full working code:

```python
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from pyrmute import ModelManager, ModelData

manager = ModelManager()


# Models
@manager.model("AppConfig", "1.0.0")
class AppConfigV1(BaseModel):
    api_key: str
    debug: bool = False


@manager.model("AppConfig", "2.0.0")
class AppConfigV2(BaseModel):
    api_key: str
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_endpoint: str = "https://api.example.com"


# Migration
@manager.migration("AppConfig", "1.0.0", "2.0.0")
def upgrade_to_v2(data: ModelData) -> ModelData:
    log_level = "DEBUG" if data.get("debug", False) else "INFO"
    return {
        "api_key": data["api_key"],
        "log_level": log_level,
        "api_endpoint": "https://api.example.com",
    }


# Usage
def load_config(path: Path) -> AppConfigV2:
    with open(path) as f:
        data = json.load(f)

    version = data.get("_version", "1.0.0")
    data.pop("_version", None)

    return manager.migrate(data, "AppConfig", version, "2.0.0")


def save_config(config: AppConfigV2, path: Path, version: str) -> None:
    data = config.model_dump()
    data["_version"] = version

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# Testing
def test_migrations() -> None:
    """Validate migration logic."""
    results = manager.test_migration(
        "AppConfig",
        "1.0.0",
        "2.0.0",
        test_cases=[
            (
                {"api_key": "sk-old", "debug": True},
                {
                    "api_key": "sk-old",
                    "log_level": "DEBUG",
                    "api_endpoint": "https://api.example.com"
                }
            ),
            (
                {"api_key": "sk-test", "debug": False},
                {
                    "api_key": "sk-test",
                    "log_level": "INFO",
                    "api_endpoint": "https://api.example.com"
                }
            ),
        ]
    )
    results.assert_all_passed()
    print("✓ All migration tests passed!")


# Example
if __name__ == "__main__":
    # Run tests first
    test_migrations()

    # Simulate old config file
    old_config = {
        "_version": "1.0.0",
        "api_key": "sk-test-123",
        "debug": True
    }
    test_path = Path("config.json")
    with open(test_path, "w") as f:
        json.dump(old_config, f)

    # Load and migrate
    config = load_config(test_path)
    print(f"Loaded config: {config}")
    print(f"Log level: {config.log_level}")  # DEBUG (from debug=True)

    # Save upgraded version
    save_config(config, test_path, "2.0.0")
    print("✓ Config upgraded to v2.0.0")
```

## Key Takeaways

1. **Store version metadata** - Include `_version` in your data files
2. **One model class per version** - `AppConfigV1`, `AppConfigV2`, etc.
3. **Pure migration functions** - Take dict, return dict, no side effects
4. **Test your migrations** - Use `manager.test_migration()` to validate logic
5. **Migrate on load** - Transform data transparently when reading files

## Common Mistakes to Avoid

### ❌ Mutating Input Data

```python
# BAD - mutates the input
def bad_migration(data: ModelData) -> ModelData:
    data["new_field"] = "value"
    return data
```

```python
# GOOD - creates new dict
def good_migration(data: ModelData) -> ModelData:
    return {**data, "new_field": "value"}
```

### ❌ Not Handling Missing Fields

```python
# BAD - crashes if debug doesn't exist
def bad_migration(data: ModelData) -> ModelData:
    return {"log_level": "DEBUG" if data["debug"] else "INFO"}
```

```python
# GOOD - uses .get() with default
def good_migration(data: ModelData) -> ModelData:
    return {"log_level": "DEBUG" if data.get("debug", False) else "INFO"}
```

### ❌ Forgetting Version Metadata

```python
# BAD - no way to know version
{
    "api_key": "sk-123"
}
```

```python
# GOOD - includes version
{
    "_version": "1.0.0",
    "api_key": "sk-123"
}
```

## Next Steps

Now that you've built your first migration:

- [Writing Migrations](../user-guide/writing-migrations.md) - Best practices and
  patterns
- [Testing Migrations](../user-guide/testing-migrations.md) - Testing strategies
- [Schema Generation](../user-guide/schema-generation.md) - Generate schemas for
   your models

## Practice Exercise

Try adding a third version (3.0.0) that:

1. Adds a `retry_config` nested object with `max_attempts` and `backoff_ms`
2. Makes `api_endpoint` optional (with same default)
3. Adds a migration from 2.0.0 to 3.0.0

**Hint:** Use `backward_compatible=True` for the optional field!

<details>
<summary>Solution</summary>

```python
class RetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_ms: int = 1000


@manager.model("AppConfig", "3.0.0", backward_compatible=True)
class AppConfigV3(BaseModel):
    api_key: str
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Now optional via backward_compatible
    api_endpoint: str = "https://api.example.com"
    retry_config: RetryConfig = RetryConfig()


@manager.migration("AppConfig", "2.0.0", "3.0.0")
def upgrade_to_v3(data: ModelData) -> ModelData:
    return {
        **data,
        "retry_config": {
            "max_attempts": 3,
            "backoff_ms": 1000
        }
    }
```

</details>
