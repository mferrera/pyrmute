# Writing Migrations

Migration functions are the core of pyrmute's data transformation system. This
guide covers best practices, common patterns, and pitfalls to avoid when
writing migrations.

## The Basics

A migration function transforms data from one version to another:

```python
from pyrmute import ModelManager, ModelData

manager = ModelManager()


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user(data: ModelData) -> ModelData:
    """Transform v1 user data to v2 format."""
    return {
        "id": data["id"],
        "name": data["name"],
        "email": data.get("email", "unknown@example.com")
    }
```

**Key characteristics:**

- Takes a `ModelData` (dict) as input
- Returns a `ModelData` (dict) as output
- Should be pure functions (no side effects)
- Can transform, add, or remove fields

## Core Principles

### 1. Pure Functions

Migrations should be **pure functions** - same input always produces same
output, with no side effects:

```python
# ✅ GOOD - Pure function
@manager.migration("User", "1.0.0", "2.0.0")
def good_migration(data: ModelData) -> ModelData:
    return {
        "id": data["id"],
        "full_name": f"{data['first_name']} {data['last_name']}"
    }


# ❌ BAD - Side effects
@manager.migration("User", "1.0.0", "2.0.0")
def bad_migration(data: ModelData) -> ModelData:
    # Don't do this!
    database.log_migration(data["id"])  # Side effect
    send_email(data["email"])  # Side effect
    return data
```

**Why?** Pure functions are:

- Easier to test
- Easier to debug
- Safe to retry
- Can be parallelized

### 2. Immutability

Never mutate the input data - always return a new dictionary:

```python
# ❌ BAD - Mutates input
@manager.migration("User", "1.0.0", "2.0.0")
def bad_migration(data: ModelData) -> ModelData:
    data["new_field"] = "value"  # Mutates input!
    return data


# ✅ GOOD - Creates new dict
@manager.migration("User", "1.0.0", "2.0.0")
def good_migration(data: ModelData) -> ModelData:
    return {**data, "new_field": "value"}
```

### 3. Defensive Programming

Always handle edge cases and missing data:

```python
# ❌ BAD - Assumes field exists
@manager.migration("User", "1.0.0", "2.0.0")
def bad_migration(data: ModelData) -> ModelData:
    return {"email": data["email"].lower()}  # KeyError if missing!


# ✅ GOOD - Handles missing fields
@manager.migration("User", "1.0.0", "2.0.0")
def good_migration(data: ModelData) -> ModelData:
    email = data.get("email", "")
    return {"email": email.lower() if email else "unknown@example.com"}
```

## Common Patterns

### Adding a Field

```python
@manager.migration("User", "1.0.0", "2.0.0")
def add_field(data: ModelData) -> ModelData:
    """Add a new field with default value."""
    return {
        **data,  # Keep all existing fields
        "created_at": "2026-01-01T00:00:00Z"  # Add new field
    }
```

### Removing a Field

```python
@manager.migration("User", "1.0.0", "2.0.0")
def remove_field(data: ModelData) -> ModelData:
    """Remove a field that's no longer needed."""
    return {
        k: v for k, v in data.items()
        if k != "deprecated_field"  # Exclude this field
    }
```

Alternative approach:

```python
@manager.migration("User", "1.0.0", "2.0.0")
def remove_field(data: ModelData) -> ModelData:
    """Remove a field using dict unpacking."""
    result = {**data}
    result.pop("deprecated_field", None)  # Remove if exists
    return result
```

### Renaming a Field

```python
@manager.migration("User", "1.0.0", "2.0.0")
def rename_field(data: ModelData) -> ModelData:
    """Rename user_name to username."""
    return {
        "username": data["user_name"],  # New name
        **{k: v for k, v in data.items() if k != "user_name"}  # Other fields
    }
```

Cleaner approach:

```python
@manager.migration("User", "1.0.0", "2.0.0")
def rename_field(data: ModelData) -> ModelData:
    """Rename user_name to username."""
    result = {**data}
    result["username"] = result.pop("user_name")
    return result
```

### Splitting a Field

```python
@manager.migration("User", "1.0.0", "2.0.0")
def split_name(data: ModelData) -> ModelData:
    """Split name into first_name and last_name."""
    name = data.get("name", "")
    parts = name.split(" ", 1)  # Split on first space only

    return {
        **{k: v for k, v in data.items() if k != "name"},
        "first_name": parts[0] if parts else "",
        "last_name": parts[1] if len(parts) > 1 else ""
    }
```

### Combining Fields

```python
@manager.migration("User", "2.0.0", "3.0.0")
def combine_name(data: ModelData) -> ModelData:
    """Combine first_name and last_name into full_name."""
    first = data.get("first_name", "")
    last = data.get("last_name", "")

    return {
        **{k: v for k, v in data.items()
           if k not in ("first_name", "last_name")},
        "full_name": f"{first} {last}".strip()
    }
```

### Type Conversion

```python
@manager.migration("Config", "1.0.0", "2.0.0")
def convert_types(data: ModelData) -> ModelData:
    """Convert string timeout to int."""
    timeout = data.get("timeout", "30")

    # Handle various input types
    if isinstance(timeout, str):
        timeout_int = int(timeout)
    elif isinstance(timeout, int):
        timeout_int = timeout
    else:
        timeout_int = 30  # Fallback default

    return {
        **data,
        "timeout": timeout_int
    }
```

### Data Enrichment

```python
@manager.migration("User", "1.0.0", "2.0.0")
def add_computed_fields(data: ModelData) -> ModelData:
    """Add computed fields based on existing data."""
    email = data.get("email", "")

    return {
        **data,
        "email_domain": email.split("@")[1] if "@" in email else "",
        "has_email": bool(email)
    }
```

### Normalizing Data

```python
@manager.migration("User", "1.0.0", "2.0.0")
def normalize_email(data: ModelData) -> ModelData:
    """Normalize email to lowercase and trim whitespace."""
    email = data.get("email", "")

    return {
        **data,
        "email": email.strip().lower() if email else ""
    }
```

### Restructuring Nested Data

```python
@manager.migration("User", "1.0.0", "2.0.0")
def restructure_address(data: ModelData) -> ModelData:
    """Move address fields into nested object."""
    return {
        "id": data["id"],
        "name": data["name"],
        "address": {
            "street": data.get("street", ""),
            "city": data.get("city", ""),
            "zip": data.get("zip", "")
        }
    }
```

### Flattening Nested Data

```python
@manager.migration("User", "2.0.0", "3.0.0")
def flatten_address(data: ModelData) -> ModelData:
    """Flatten nested address object."""
    address = data.get("address", {})

    return {
        "id": data["id"],
        "name": data["name"],
        "street": address.get("street", ""),
        "city": address.get("city", ""),
        "zip": address.get("zip", "")
    }
```

## Advanced Patterns

### Conditional Transformations

```python
@manager.migration("User", "1.0.0", "2.0.0")
def conditional_migration(data: ModelData) -> ModelData:
    """Apply different logic based on data values."""
    user_type = data.get("type", "regular")

    if user_type == "admin":
        return {
            **data,
            "permissions": ["read", "write", "delete"],
            "role": "administrator"
        }
    elif user_type == "moderator":
        return {
            **data,
            "permissions": ["read", "write"],
            "role": "moderator"
        }
    else:
        return {
            **data,
            "permissions": ["read"],
            "role": "user"
        }
```

### Handling Lists

```python
@manager.migration("User", "1.0.0", "2.0.0")
def transform_list(data: ModelData) -> ModelData:
    """Transform items in a list field."""
    tags = data.get("tags", [])

    # Normalize all tags to lowercase
    normalized_tags = [tag.lower() for tag in tags if isinstance(tag, str)]

    return {
        **data,
        "tags": normalized_tags
    }
```

### Complex Validation and Defaults

```python
@manager.migration("Config", "1.0.0", "2.0.0")
def validate_and_default(data: ModelData) -> ModelData:
    """Validate values and apply defaults."""
    timeout = data.get("timeout", 30)

    # Ensure timeout is within valid range
    if not isinstance(timeout, int) or timeout < 1:
        timeout = 30
    elif timeout > 300:
        timeout = 300

    retries = data.get("retries", 3)
    if not isinstance(retries, int) or retries < 0:
        retries = 3

    return {
        **data,
        "timeout": timeout,
        "retries": retries
    }
```

### Data Migration with Lookups

Sometimes you need reference data during migration:

```python
# Define lookup tables at module level
COUNTRY_CODE_MAP = {
    "USA": "US",
    "United States": "US",
    "UK": "GB",
    "United Kingdom": "GB",
    # ...
}

@manager.migration("User", "1.0.0", "2.0.0")
def normalize_country(data: ModelData) -> ModelData:
    """Normalize country names to ISO codes."""
    country = data.get("country", "")
    country_code = COUNTRY_CODE_MAP.get(country, country)

    return {
        **data,
        "country_code": country_code
    }
```

!!! warning "Keep Lookups Simple"
    Don't fetch data from databases or external APIs in migrations. This
    breaks the pure function principle and makes migrations slow and fragile.

    If you need external data:

    - Pre-load it into a dict/cache
    - Include it in your data files
    - Handle it outside the migration

## Error Handling

### When to Raise Exceptions

Raise exceptions for truly invalid data that cannot be migrated:

```python
from pyrmute import MigrationError, ModelData


@manager.migration("User", "1.0.0", "2.0.0")
def strict_migration(data: ModelData) -> ModelData:
    """Migration that requires certain fields."""
    if "id" not in data:
        raise MigrationError(
            "User",
            "1.0.0",
            "2.0.0",
            "Missing required field 'id'"
        )

    if not isinstance(data["id"], (int, str)):
        raise MigrationError(
            "User",
            "1.0.0",
            "2.0.0",
            f"Invalid id type: {type(data['id'])}"
        )

    return {"user_id": str(data["id"])}
```

### When to Use Defaults

For missing or invalid data that can be reasonably defaulted:

```python
@manager.migration("User", "1.0.0", "2.0.0")
def lenient_migration(data: ModelData) -> ModelData:
    """Migration that handles missing data gracefully."""
    # Provide sensible defaults instead of failing
    return {
        "id": data.get("id", -1),  # Use sentinel value
        "name": data.get("name", "Unknown"),
        "email": data.get("email", "unknown@example.com")
    }
```

### Logging Issues

For production systems, log issues without failing:

```python
import logging

logger = logging.getLogger(__name__)


@manager.migration("User", "1.0.0", "2.0.0")
def logged_migration(data: ModelData) -> ModelData:
    """Migration with logging for monitoring."""
    if "email" not in data:
        logger.warning(
            "User missing email during migration",
            extra={"user_id": data.get("id"), "data": data}
        )

    return {
        **data,
        "email": data.get("email", "unknown@example.com")
    }
```

## Testing Migrations

Always test your migrations with realistic data:

```python
def test_user_migration() -> None:
    """Test user v1 to v2 migration."""
    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            # Happy path
            (
                {"id": 1, "name": "Alice Smith"},
                {"id": 1, "first_name": "Alice", "last_name": "Smith"}
            ),
            # Single name
            (
                {"id": 2, "name": "Bob"},
                {"id": 2, "first_name": "Bob", "last_name": ""}
            ),
            # Empty name
            (
                {"id": 3, "name": ""},
                {"id": 3, "first_name": "", "last_name": ""}
            ),
            # Missing name field
            (
                {"id": 4},
                {"id": 4, "first_name": "", "last_name": ""}
            ),
        ]
    )

    results.assert_all_passed()
```

See [Testing Migrations](testing-migrations.md) for testing strategies.

## Performance Considerations

### Keep Migrations Fast

Migrations run on every piece of data, so performance matters:

```python
# ✅ GOOD - Fast operations
@manager.migration("User", "1.0.0", "2.0.0")
def fast_migration(data: ModelData) -> ModelData:
    return {
        **data,
        "email": data.get("email", "").lower()
    }


# ❌ BAD - Slow operations
@manager.migration("User", "1.0.0", "2.0.0")
def slow_migration(data: ModelData) -> ModelData:
    # Don't do expensive operations here!
    time.sleep(0.1)  # Bad
    response = requests.get(f"https://api.com/users/{data['id']}")  # Very bad

    return {**data, "external_data": response.json()}
```

### Avoid Nested Loops

```python
# ❌ BAD - O(n²) complexity
@manager.migration("User", "1.0.0", "2.0.0")
def slow_list_processing(data: ModelData) -> ModelData:
    tags = data.get("tags", [])
    unique_tags = []

    for tag in tags:
        if tag not in unique_tags:  # Linear search for each tag
            unique_tags.append(tag)

    return {**data, "tags": unique_tags}


# ✅ GOOD - O(n) complexity
@manager.migration("User", "1.0.0", "2.0.0")
def fast_list_processing(data: ModelData) -> ModelData:
    tags = data.get("tags", [])
    unique_tags = list(dict.fromkeys(tags))  # Preserves order, O(n)

    return {**data, "tags": unique_tags}
```

## Common Pitfalls

### 1. Not Handling None Values

```python
# ❌ BAD
@manager.migration("User", "1.0.0", "2.0.0")
def bad_migration(data: ModelData) -> ModelData:
    return {"email": data["email"].lower()}  # Fails if email is None!


# ✅ GOOD
@manager.migration("User", "1.0.0", "2.0.0")
def good_migration(data: ModelData) -> ModelData:
    email = data.get("email")
    return {"email": email.lower() if email else None}
```

### 2. Forgetting to Preserve Fields

```python
# ❌ BAD - Loses other fields!
@manager.migration("User", "1.0.0", "2.0.0")
def bad_migration(data: ModelData) -> ModelData:
    return {"new_field": "value"}  # Where did everything else go?


# ✅ GOOD - Preserves existing fields
@manager.migration("User", "1.0.0", "2.0.0")
def good_migration(data: ModelData) -> ModelData:
    return {**data, "new_field": "value"}
```

### 3. Inconsistent Return Types

```python
# ❌ BAD - Returns different structures
@manager.migration("User", "1.0.0", "2.0.0")
def bad_migration(data: ModelData) -> ModelData:
    if data.get("type") == "admin":
        return {"id": data["id"], "role": "admin"}
    else:
        return {"id": data["id"]}  # Missing role field!


# ✅ GOOD - Consistent structure
@manager.migration("User", "1.0.0", "2.0.0")
def good_migration(data: ModelData) -> ModelData:
    role = "admin" if data.get("type") == "admin" else "user"
    return {"id": data["id"], "role": role}
```

### 4. Depending on Execution Order

```python
# ❌ BAD - Assumes migration order
global_counter = 0


@manager.migration("User", "1.0.0", "2.0.0")
def bad_migration(data: ModelData) -> ModelData:
    global global_counter
    global_counter += 1  # Don't rely on order!
    return {**data, "migration_order": global_counter}


# ✅ GOOD - Self-contained
@manager.migration("User", "1.0.0", "2.0.0")
def good_migration(data: ModelData) -> ModelData:
    return {**data, "migrated_at": "2024-01-01T00:00:00Z"}
```

## Documentation

Document your migrations clearly:

```python
@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user_v1_to_v2(data: ModelData) -> ModelData:
    """Migrate User from v1.0.0 to v2.0.0.

    Changes:
    - Split 'name' field into 'first_name' and 'last_name'
    - Add 'email' field with default value
    - Remove deprecated 'username' field

    Args:
        data: User data in v1.0.0 format

    Returns:
        User data in v2.0.0 format

    Example:
        >>> migrate_user_v1_to_v2({"name": "John Doe", "username": "jdoe"})
        {"first_name": "John", "last_name": "Doe", "email": "unknown@example.com"}
    """
    name = data.get("name", "")
    parts = name.split(" ", 1)

    return {
        "first_name": parts[0] if parts else "",
        "last_name": parts[1] if len(parts) > 1 else "",
        "email": data.get("email", "unknown@example.com")
        # Note: username field intentionally removed
    }
```

## Next Steps

Now that you understand migration best practices:

**Continue learning:**

- [Testing Migrations](testing-migrations.md) - Validate your migrations
    thoroughly
- [Auto-Migrations](auto-migrations.md) - When you can skip writing
    migrations
- [Batch Processing](batch-processing.md) - Migrate large datasets
    efficiently

**Advanced topics:**

- [Nested Models](nested-models.md) - Handle nested Pydantic model
    transformations
- [Discriminated Unions](../advanced/discriminated-unions.md) - Migrate
    polymorphic types

**API Reference:**

- [`ModelManager` API](../reference/model-manager.md) - Complete
    `ModelManager` details
- [`ModelDiff` API](../reference/model-diff.md) - Complete `ModelDiff`
    details
- [Exceptions](../reference/exceptions.md) - Exceptions pyrmute raises
- [Types](../reference/types.md) - Type alises exported by pyrmute
