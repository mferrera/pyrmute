# Testing Migrations

Testing your migrations is crucial for ensuring data integrity and preventing
production issues. This guide covers testing strategies, best practices, and
integration with testing frameworks.

## Why Test Migrations?

Migrations transform your data. Bugs in migrations can:

- Cause data loss
- Create invalid data that fails validation
- Produce incorrect results silently
- Break production systems during deployment

**Testing migrations before production is not optional.**

## Basic Testing

pyrmute includes built-in testing utilities:

```python
from pyrmute import ModelManager, ModelData

manager = ModelManager()


# Define your models and migrations
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
        "last_name": parts[1] if len(parts) > 1 else ""
    }


# Test the migration
results = manager.test_migration(
    "User",
    "1.0.0",
    "2.0.0",
    test_cases=[
        (
            {"name": "Alice Smith"},
            {"first_name": "Alice", "last_name": "Smith"}
        ),
        (
            {"name": "Bob"},
            {"first_name": "Bob", "last_name": ""}
        ),
    ]
)

# Check results
if results.all_passed:
    print("✓ All tests passed!")
else:
    print(f"✗ {len(results.failures)} test(s) failed")
    for failure in results.failures:
        print(failure)
```

## Test Case Formats

### Tuple Format (Simple)

Use tuples for straightforward tests:

```python
results = manager.test_migration(
    "User",
    "1.0.0",
    "2.0.0",
    test_cases=[
        # (source_data, expected_output)
        ({"name": "Alice"}, {"first_name": "Alice", "last_name": ""}),
        ({"name": "Bob Jones"}, {"first_name": "Bob", "last_name": "Jones"}),
    ]
)
```

### MigrationTestCase (Descriptive)

Use `MigrationTestCase` for better documentation:

```python
from pyrmute import MigrationTestCase


results = manager.test_migration(
    "User",
    "1.0.0",
    "2.0.0",
    test_cases=[
        MigrationTestCase(
            source={"name": "Alice"},
            target={"first_name": "Alice", "last_name": ""},
            description="Single name should leave last_name empty"
        ),
        MigrationTestCase(
            source={"name": "Bob Jones"},
            target={"first_name": "Bob", "last_name": "Jones"},
            description="Full name should split correctly"
        ),
        MigrationTestCase(
            source={"name": "Maria de la Cruz"},
            target={"first_name": "Maria", "last_name": "de la Cruz"},
            description="Multiple spaces should split on first space only"
        ),
    ]
)

# Better error messages with descriptions
if not results.all_passed:
    for failure in results.failures:
        print(failure)
        # Prints: ✗ Test failed - Multiple spaces should split on first space only
```

### Testing Without Expected Output

Sometimes you just want to verify migrations don't crash:

```python
results = manager.test_migration(
    "User",
    "1.0.0",
    "2.0.0",
    test_cases=[
        MigrationTestCase(
            source={"name": "Alice"},
            target=None,  # Don't check output, just verify no errors
            description="Should handle basic case without errors"
        ),
    ]
)
```

This is useful for:

- Smoke testing complex migrations
- Testing that edge cases don't crash
- Initial migration development

## Integration with Test Frameworks

### pytest

```python
import pytest
from pyrmute import ModelManager

manager = ModelManager()

# ... define models and migrations ...

def test_user_migration_basic() -> None:
    """Test basic user migration scenarios."""
    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"first_name": "Alice", "last_name": ""}),
            ({"name": "Bob Smith"}, {"first_name": "Bob", "last_name": "Smith"}),
        ]
    )
    results.assert_all_passed()


def test_user_migration_edge_cases() -> None:
    """Test edge cases in user migration."""
    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": ""}, {"first_name": "", "last_name": ""}),
            ({"name": "   "}, {"first_name": "", "last_name": ""}),
            ({"name": "X"}, {"first_name": "X", "last_name": ""}),
        ]
    )
    results.assert_all_passed()

def test_user_migration_preserves_fields() -> None:
    """Test that migration preserves other fields."""
    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            (
                {"name": "Alice", "id": 123, "email": "alice@example.com"},
                {
                    "first_name": "Alice",
                    "last_name": "",
                    "id": 123,
                    "email": "alice@example.com"
                }
            ),
        ]
    )
    results.assert_all_passed()
```

### unittest

```python
import unittest
from pyrmute import ModelManager

manager = ModelManager()

# ... define models and migrations ...

class TestUserMigration(unittest.TestCase):
    def test_basic_migration(self) -> None:
        """Test basic user migration."""
        results = manager.test_migration(
            "User",
            "1.0.0",
            "2.0.0",
            test_cases=[
                ({"name": "Alice"}, {"first_name": "Alice", "last_name": ""}),
            ]
        )
        results.assert_all_passed()

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        results = manager.test_migration(
            "User",
            "1.0.0",
            "2.0.0",
            test_cases=[
                ({"name": ""}, {"first_name": "", "last_name": ""}),
            ]
        )
        results.assert_all_passed()

if __name__ == "__main__":
    unittest.main()
```

## Test Coverage Guidelines

### Essential Test Cases

Every migration should test:

1. **Happy path** - Normal, expected data
2. **Edge cases** - Empty strings, None values, extreme values
3. **Missing fields** - Optional fields not present
4. **Invalid data** - Data that might exist in production
5. **Boundary conditions** - Min/max values, empty collections

### Example: Comprehensive Test Suite

```python
def test_user_migration_comprehensive() -> None:
    """Comprehensive test suite for user migration."""
    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            # Happy path
            MigrationTestCase(
                source={"name": "Alice Smith"},
                target={"first_name": "Alice", "last_name": "Smith"},
                description="Normal full name"
            ),

            # Single name
            MigrationTestCase(
                source={"name": "Alice"},
                target={"first_name": "Alice", "last_name": ""},
                description="Single name only"
            ),

            # Multiple spaces
            MigrationTestCase(
                source={"name": "Alice Mary Smith"},
                target={"first_name": "Alice", "last_name": "Mary Smith"},
                description="Multiple word last name"
            ),

            # Empty name
            MigrationTestCase(
                source={"name": ""},
                target={"first_name": "", "last_name": ""},
                description="Empty name string"
            ),

            # Whitespace only
            MigrationTestCase(
                source={"name": "   "},
                target={"first_name": "", "last_name": ""},
                description="Whitespace only name"
            ),

            # Missing name field
            MigrationTestCase(
                source={},
                target={"first_name": "", "last_name": ""},
                description="Missing name field"
            ),

            # Extra whitespace
            MigrationTestCase(
                source={"name": "  Alice   Smith  "},
                target={"first_name": "Alice", "last_name": "Smith"},
                description="Extra whitespace around name"
            ),

            # Special characters
            MigrationTestCase(
                source={"name": "O'Brien"},
                target={"first_name": "O'Brien", "last_name": ""},
                description="Name with apostrophe"
            ),

            # Unicode characters
            MigrationTestCase(
                source={"name": "José García"},
                target={"first_name": "José", "last_name": "García"},
                description="Unicode characters in name"
            ),

            # Preserves other fields
            MigrationTestCase(
                source={"name": "Alice", "id": 123, "active": True},
                target={
                    "first_name": "Alice",
                    "last_name": "",
                    "id": 123,
                    "active": True
                },
                description="Preserves additional fields"
            ),
        ]
    )

    results.assert_all_passed()
```

## Testing Migration Chains

Test migrations across multiple versions:

```python
def test_migration_chain() -> None:
    """Test migrating across multiple versions."""
    # v1 -> v2 -> v3
    results_v1_to_v2 = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"first_name": "Alice", "last_name": ""}),
        ]
    )
    results_v1_to_v2.assert_all_passed()

    results_v2_to_v3 = manager.test_migration(
        "User",
        "2.0.0",
        "3.0.0",
        test_cases=[
            (
                {"first_name": "Alice", "last_name": "Smith"},
                {"full_name": "Alice Smith", "email": "unknown@example.com"}
            ),
        ]
    )
    results_v2_to_v3.assert_all_passed()

    # Test full chain v1 -> v3
    results_v1_to_v3 = manager.test_migration(
        "User",
        "1.0.0",
        "3.0.0",
        test_cases=[
            (
                {"name": "Alice Smith"},
                {"full_name": "Alice Smith", "email": "unknown@example.com"}
            ),
        ]
    )
    results_v1_to_v3.assert_all_passed()
```

## Testing with Real Data

Use anonymized production data for testing:

```python
import json
from pathlib import Path


def test_with_production_data() -> None:
    """Test migration with real production data samples."""
    # Load anonymized production samples
    samples_path = Path("tests/fixtures/user_samples_v1.json")
    with open(samples_path) as f:
        production_samples = json.load(f)

    test_cases = []
    for sample in production_samples:
        # You might not have expected output for all samples
        test_cases.append(
            MigrationTestCase(
                source=sample,
                target=None,  # Just verify no crashes
                description=f"Production sample: user_id={sample.get('id')}"
            )
        )

    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=test_cases
    )

    results.assert_all_passed()
```

## Property-Based Testing

Use hypothesis for property-based testing:

```python
import pytest
from pyrmute import ModelData
from hypothesis import given, strategies as st


@given(st.text())
def test_name_migration_always_splits_on_first_space(name: str) -> None:
    """Test that name splitting is consistent."""
    result = manager.migrate_data(
        {"name": name},
        "User",
        "1.0.0",
        "2.0.0"
    )

    # Property: reconstructing name should give original (minus extra spaces)
    if name.strip():
        reconstructed = f"{result['first_name']} {result['last_name']}".strip()
        assert reconstructed == " ".join(name.split())


@given(st.dictionaries(
    keys=st.text(min_size=1),
    values=st.one_of(st.text(), st.integers(), st.booleans())
))
def test_migration_preserves_extra_fields(data: ModelData) -> None:
    """Test that migrations preserve fields they don't touch."""
    # Add name field for migration
    data["name"] = "Test User"

    result = manager.migrate_data(data, "User", "1.0.0", "2.0.0")

    # All original fields (except name) should be preserved
    for key, value in data.items():
        if key != "name":
            assert result.get(key) == value
```

## Snapshot Testing

Compare migration outputs to saved snapshots:

```python
import json
from pathlib import Path


def test_migration_snapshot() -> None:
    """Test that migration output matches saved snapshot."""
    # Input data
    input_data = {"name": "Alice Smith", "id": 123}

    # Run migration
    result = manager.migrate_data(input_data, "User", "1.0.0", "2.0.0")

    # Load or create snapshot
    snapshot_path = Path("tests/snapshots/user_v1_to_v2.json")

    if snapshot_path.exists():
        with open(snapshot_path) as f:
            expected = json.load(f)
        assert result == expected, "Migration output changed"
    else:
        # Create snapshot on first run
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "w") as f:
            json.dump(result, f, indent=2)
        pytest.skip("Created new snapshot")
```

## Testing Error Cases

Test that migrations fail appropriately:

```python
import pytest
from pyrmute import MigrationError


def test_migration_fails_on_invalid_data() -> None:
    """Test that migration raises appropriate errors."""
    with pytest.raises(MigrationError) as exc_info:
        manager.migrate(
            {"invalid": "data"},  # Missing required 'name' field
            "User",
            "1.0.0",
            "2.0.0"
        )

    assert "name" in str(exc_info.value).lower()


def test_migration_handles_none_gracefully() -> None:
    """Test that None values are handled correctly."""
    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            (
                {"name": None},
                {"first_name": "", "last_name": ""}  # Or whatever your handling is
            ),
        ]
    )
    results.assert_all_passed()
```

## Performance Testing

Test migration performance with large datasets:

```python
import time


def test_migration_performance() -> None:
    """Test that migration performs adequately."""
    # Generate test data
    test_data = [{"name": f"User {i}"} for i in range(10000)]

    start = time.time()
    results = manager.migrate_batch(
        test_data,
        "User",
        "1.0.0",
        "2.0.0"
    )
    duration = time.time() - start

    # Should process 10k records in reasonable time
    assert duration < 5.0, f"Migration too slow: {duration:.2f}s"
    assert len(results) == 10000

    # Calculate throughput
    throughput = len(test_data) / duration
    print(f"Throughput: {throughput:.0f} records/second")
```

## Testing Best Practices

### 1. Test Before Production

Never deploy untested migrations:

```python
# In your CI/CD pipeline
def test_all_migrations() -> None:
    """Test all registered migrations."""
    for model_name in manager.list_models():
        versions = manager.list_versions(model_name)

        for i in range(len(versions) - 1):
            from_version = versions[i]
            to_version = versions[i + 1]

            # Verify migration path exists
            assert manager.has_migration_path(
                model_name,
                from_version,
                to_version
            ), f"No migration from {from_version} to {to_version}"
```

### 2. Use Realistic Data

Don't just test happy paths:

```python
# ❌ BAD - Only happy path
test_cases = [
    ({"name": "Alice Smith"}, {"first_name": "Alice", "last_name": "Smith"})
]

# ✅ GOOD - Multiple scenarios
test_cases = [
    ({"name": "Alice Smith"}, {"first_name": "Alice", "last_name": "Smith"}),
    ({"name": "Alice"}, {"first_name": "Alice", "last_name": ""}),
    ({"name": ""}, {"first_name": "", "last_name": ""}),
    ({}, {"first_name": "", "last_name": ""}),
    ({"name": None}, {"first_name": "", "last_name": ""}),
]
```

### 3. Test Idempotency

Verify migrations can be run multiple times safely:

```python
def test_migration_idempotency() -> None:
    """Test that running migration twice is safe."""
    data = {"name": "Alice Smith"}

    # First migration
    result1 = manager.migrate_data(data, "User", "1.0.0", "2.0.0")

    # Second migration of same data
    result2 = manager.migrate_data(data, "User", "1.0.0", "2.0.0")

    # Should produce identical results
    assert result1 == result2
```

### 4. Version Your Test Data

Keep test data in version control:

```
tests/
├── data/
│   ├── user_v1_samples.json
│   ├── user_v2_samples.json
│   └── config_v1_samples.json
└── test_migrations.py
```

### 5. Document Test Scenarios

```python
"""
Test scenarios for User v1.0.0 -> v2.0.0 migration:

1. Standard full name (first last)
2. Single name only
3. Multiple word last name
4. Empty/missing name
5. Unicode characters
6. Special characters (apostrophes, hyphens)
7. Excessive whitespace
8. Field preservation (ensure other fields not lost)
"""
```

## Debugging Failed Tests

When tests fail, use the detailed output:

```python
results = manager.test_migration("User", "1.0.0", "2.0.0", test_cases)

if not results.all_passed:
    for failure in results.failures:
        print(f"\n{failure}")
        # Prints:
        # ✗ Test failed - Single name should leave last_name empty
        #   Source:   {'name': 'Alice'}
        #   Expected: {'first_name': 'Alice', 'last_name': ''}
        #   Actual:   {'first_name': 'Alice', 'last_name': None}
        #   Error:    Output mismatch
```

You can also inspect individual results:

```python
for result in results:
    print(f"Test: {result.test_case.description}")
    print(f"Passed: {result.passed}")
    if not result.passed:
        print(f"Source: {result.test_case.source}")
        print(f"Expected: {result.test_case.target}")
        print(f"Actual: {result.actual}")
        print(f"Error: {result.error}")
```

## Next Steps

Now that you understand migration testing:

**Continue learning:**

- [Batch Processing](batch-processing.md) - Test large-scale migration
    scenarios
- [Nested Models](nested-models.md) - Test nested model migrations
- [Discriminated Unions](../advanced/discriminated-unions.md) - Test
    polymorphic type migrations

**Improve your migrations:**

- [Writing Migrations](writing-migrations.md) - Write testable migration
    functions

**API Reference:**

- [Migration Testing API](../reference/migration-testing.md) - Complete
    details of migration testing objects
- [`ModelManager` API](../reference/model-manager.md) - Complete
    `ModelManager` details
- [Exceptions](../reference/exceptions.md) - Exceptions pyrmute raises
