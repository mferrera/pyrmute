"""Tests for migration testing."""

import pytest
from pydantic import BaseModel

from pyrmute import (
    MigrationTestCase,
    MigrationTestResult,
    MigrationTestResults,
    ModelData,
    ModelManager,
    ModelVersion,
)


# MigrationTestCase tests
def test_migration_test_case_initialization() -> None:
    """Test MigrationTestCase can be initialized."""
    test_case = MigrationTestCase(
        source={"name": "Alice"},
        target={"name": "Alice", "email": "alice@example.com"},
        description="Test migration",
    )

    assert test_case.source == {"name": "Alice"}
    assert test_case.target == {"name": "Alice", "email": "alice@example.com"}
    assert test_case.description == "Test migration"


def test_migration_test_case_default_values() -> None:
    """Test MigrationTestCase default values."""
    test_case = MigrationTestCase(source={"name": "Alice"})

    assert test_case.source == {"name": "Alice"}
    assert test_case.target is None
    assert test_case.description == ""


def test_migration_test_case_without_target() -> None:
    """Test MigrationTestCase with no target for smoke testing."""
    test_case = MigrationTestCase(
        source={"name": "Bob"}, target=None, description="Smoke test"
    )

    assert test_case.target is None


# MigrationTestResult tests
def test_migration_test_result_passed() -> None:
    """Test MigrationTestResult for passed test."""
    test_case = MigrationTestCase(
        source={"name": "Alice"}, target={"name": "Alice", "email": "alice@example.com"}
    )
    result = MigrationTestResult(
        test_case=test_case,
        actual={"name": "Alice", "email": "alice@example.com"},
        passed=True,
        error=None,
    )

    assert result.passed is True
    assert result.error is None


def test_migration_test_result_failed() -> None:
    """Test MigrationTestResult for failed test."""
    test_case = MigrationTestCase(
        source={"name": "Alice"}, target={"name": "Alice", "email": "alice@example.com"}
    )
    result = MigrationTestResult(
        test_case=test_case,
        actual={"name": "Alice"},
        passed=False,
        error="Output mismatch",
    )

    assert result.passed is False
    assert result.error == "Output mismatch"


def test_migration_test_result_str_passed() -> None:
    """Test MigrationTestResult string representation for passed test."""
    test_case = MigrationTestCase(source={"name": "Alice"})
    result = MigrationTestResult(
        test_case=test_case,
        actual={"name": "Alice", "email": "alice@example.com"},
        passed=True,
        error=None,
    )

    assert "✓ Test passed" in str(result)


def test_migration_test_result_str_passed_with_description() -> None:
    """Test MigrationTestResult string includes description when passed."""
    test_case = MigrationTestCase(
        source={"name": "Alice"}, description="Adds email field"
    )
    result = MigrationTestResult(
        test_case=test_case,
        actual={"name": "Alice", "email": "alice@example.com"},
        passed=True,
        error=None,
    )

    result_str = str(result)
    assert "✓ Test passed" in result_str
    assert "Adds email field" in result_str


def test_migration_test_result_str_failed() -> None:
    """Test MigrationTestResult string representation for failed test."""
    test_case = MigrationTestCase(
        source={"name": "Alice"}, target={"name": "Alice", "email": "alice@example.com"}
    )
    result = MigrationTestResult(
        test_case=test_case,
        actual={"name": "Alice"},
        passed=False,
        error="Output mismatch",
    )

    result_str = str(result)
    assert "✗ Test failed" in result_str
    assert "Source:" in result_str
    assert "Expected:" in result_str
    assert "Actual:" in result_str
    assert "Error:" in result_str


def test_migration_test_result_str_failed_with_description() -> None:
    """Test MigrationTestResult string includes description when failed."""
    test_case = MigrationTestCase(
        source={"name": "Alice"},
        target={"name": "Alice", "email": "alice@example.com"},
        description="Should add email",
    )
    result = MigrationTestResult(
        test_case=test_case,
        actual={"name": "Alice"},
        passed=False,
        error="Output mismatch",
    )

    result_str = str(result)
    assert "Should add email" in result_str


# MigrationTestResults tests
def test_migration_test_results_initialization() -> None:
    """Test MigrationTestResults can be initialized."""
    test_case = MigrationTestCase(source={"name": "Alice"})
    result = MigrationTestResult(
        test_case=test_case, actual={"name": "Alice"}, passed=True, error=None
    )
    results = MigrationTestResults([result])

    assert len(results.results) == 1
    assert results.results[0] == result


def test_migration_test_results_all_passed_true() -> None:
    """Test all_passed returns True when all tests pass."""
    test_case1 = MigrationTestCase(source={"name": "Alice"})
    test_case2 = MigrationTestCase(source={"name": "Bob"})

    result1 = MigrationTestResult(
        test_case=test_case1, actual={"name": "Alice"}, passed=True, error=None
    )
    result2 = MigrationTestResult(
        test_case=test_case2, actual={"name": "Bob"}, passed=True, error=None
    )

    results = MigrationTestResults([result1, result2])
    assert results.all_passed is True


def test_migration_test_results_all_passed_false() -> None:
    """Test all_passed returns False when any test fails."""
    test_case1 = MigrationTestCase(source={"name": "Alice"})
    test_case2 = MigrationTestCase(source={"name": "Bob"})

    result1 = MigrationTestResult(
        test_case=test_case1, actual={"name": "Alice"}, passed=True, error=None
    )
    result2 = MigrationTestResult(
        test_case=test_case2, actual={"name": "Bob"}, passed=False, error="Failed"
    )

    results = MigrationTestResults([result1, result2])
    assert results.all_passed is False


def test_migration_test_results_failures_empty() -> None:
    """Test failures returns empty list when all pass."""
    test_case = MigrationTestCase(source={"name": "Alice"})
    result = MigrationTestResult(
        test_case=test_case, actual={"name": "Alice"}, passed=True, error=None
    )

    results = MigrationTestResults([result])
    assert results.failures == []


def test_migration_test_results_failures_contains_failed() -> None:
    """Test failures contains only failed tests."""
    test_case1 = MigrationTestCase(source={"name": "Alice"})
    test_case2 = MigrationTestCase(source={"name": "Bob"})
    test_case3 = MigrationTestCase(source={"name": "Charlie"})

    result1 = MigrationTestResult(
        test_case=test_case1, actual={"name": "Alice"}, passed=True, error=None
    )
    result2 = MigrationTestResult(
        test_case=test_case2, actual={"name": "Bob"}, passed=False, error="Failed"
    )
    result3 = MigrationTestResult(
        test_case=test_case3, actual={"name": "Charlie"}, passed=False, error="Failed"
    )

    results = MigrationTestResults([result1, result2, result3])
    assert len(results.failures) == 2  # noqa: PLR2004
    assert result2 in results.failures
    assert result3 in results.failures
    assert result1 not in results.failures


def test_migration_test_results_assert_all_passed_succeeds() -> None:
    """Test assert_all_passed does not raise when all pass."""
    test_case = MigrationTestCase(source={"name": "Alice"})
    result = MigrationTestResult(
        test_case=test_case, actual={"name": "Alice"}, passed=True, error=None
    )

    results = MigrationTestResults([result])
    results.assert_all_passed()  # Should not raise


def test_migration_test_results_assert_all_passed_fails() -> None:
    """Test assert_all_passed raises when any test fails."""
    test_case = MigrationTestCase(source={"name": "Alice"})
    result = MigrationTestResult(
        test_case=test_case, actual={"name": "Alice"}, passed=False, error="Failed"
    )

    results = MigrationTestResults([result])
    with pytest.raises(AssertionError, match="1 migration test\\(s\\) failed"):
        results.assert_all_passed()


def test_migration_test_results_assert_all_passed_includes_details() -> None:
    """Test assert_all_passed includes failure details in error."""
    test_case = MigrationTestCase(
        source={"name": "Alice"},
        target={"name": "Alice", "email": "alice@example.com"},
        description="Should add email",
    )
    result = MigrationTestResult(
        test_case=test_case,
        actual={"name": "Alice"},
        passed=False,
        error="Output mismatch",
    )

    results = MigrationTestResults([result])
    with pytest.raises(AssertionError) as exc_info:
        results.assert_all_passed()

    error_message = str(exc_info.value)
    assert "Should add email" in error_message
    assert "Output mismatch" in error_message


def test_migration_test_results_len() -> None:
    """Test __len__ returns correct count."""
    test_case1 = MigrationTestCase(source={"name": "Alice"})
    test_case2 = MigrationTestCase(source={"name": "Bob"})

    result1 = MigrationTestResult(
        test_case=test_case1, actual={"name": "Alice"}, passed=True, error=None
    )
    result2 = MigrationTestResult(
        test_case=test_case2, actual={"name": "Bob"}, passed=True, error=None
    )

    results = MigrationTestResults([result1, result2])
    assert len(results) == 2  # noqa: PLR2004


def test_migration_test_results_iter() -> None:
    """Test __iter__ allows iteration."""
    test_case1 = MigrationTestCase(source={"name": "Alice"})
    test_case2 = MigrationTestCase(source={"name": "Bob"})

    result1 = MigrationTestResult(
        test_case=test_case1, actual={"name": "Alice"}, passed=True, error=None
    )
    result2 = MigrationTestResult(
        test_case=test_case2, actual={"name": "Bob"}, passed=True, error=None
    )

    results = MigrationTestResults([result1, result2])
    items = list(results)
    assert len(items) == 2  # noqa: PLR2004
    assert items[0] == result1
    assert items[1] == result2


def test_migration_test_results_str_all_passed() -> None:
    """Test __str__ when all tests pass."""
    test_case1 = MigrationTestCase(source={"name": "Alice"})
    test_case2 = MigrationTestCase(source={"name": "Bob"})

    result1 = MigrationTestResult(
        test_case=test_case1, actual={"name": "Alice"}, passed=True, error=None
    )
    result2 = MigrationTestResult(
        test_case=test_case2, actual={"name": "Bob"}, passed=True, error=None
    )

    results = MigrationTestResults([result1, result2])
    assert str(results) == "✓ All 2 test(s) passed"


def test_migration_test_results_str_some_failed() -> None:
    """Test __str__ when some tests fail."""
    test_case1 = MigrationTestCase(source={"name": "Alice"})
    test_case2 = MigrationTestCase(source={"name": "Bob"})
    test_case3 = MigrationTestCase(source={"name": "Charlie"})

    result1 = MigrationTestResult(
        test_case=test_case1, actual={"name": "Alice"}, passed=True, error=None
    )
    result2 = MigrationTestResult(
        test_case=test_case2, actual={"name": "Bob"}, passed=False, error="Failed"
    )
    result3 = MigrationTestResult(
        test_case=test_case3, actual={"name": "Charlie"}, passed=False, error="Failed"
    )

    results = MigrationTestResults([result1, result2, result3])
    assert str(results) == "✗ 2 of 3 test(s) failed (1 passed)"


def test_migration_test_results_empty() -> None:
    """Test MigrationTestResults with empty list."""
    results = MigrationTestResults([])
    assert len(results) == 0
    assert results.all_passed is True
    assert results.failures == []
    assert str(results) == "✓ All 0 test(s) passed"


# test_migration method tests
def test_test_migration_with_tuples(registered_manager: ModelManager) -> None:
    """Test test_migration with tuple format."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
            ({"name": "Bob"}, {"name": "Bob", "email": "unknown@example.com"}),
        ],
    )

    assert len(results) == 2  # noqa: PLR2004
    assert results.all_passed is True


def test_test_migration_with_test_case_objects(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration with MigrationTestCase objects."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            MigrationTestCase(
                source={"name": "Alice"},
                target={"name": "Alice", "email": "unknown@example.com"},
                description="Alice migration",
            ),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is True


def test_test_migration_mixed_formats(registered_manager: ModelManager) -> None:
    """Test test_migration with mixed tuple and object formats."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
            MigrationTestCase(
                source={"name": "Bob"},
                target={"name": "Bob", "email": "unknown@example.com"},
            ),
        ],
    )

    assert len(results) == 2  # noqa: PLR2004
    assert results.all_passed is True


def test_test_migration_detects_mismatch(registered_manager: ModelManager) -> None:
    """Test test_migration detects output mismatch."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "wrong@example.com"}),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is False
    assert len(results.failures) == 1
    assert results.failures[0].error == "Output mismatch"


def test_test_migration_without_expected(registered_manager: ModelManager) -> None:
    """Test test_migration without expected output (smoke test)."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            MigrationTestCase(source={"name": "Alice"}, target=None),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is True


def test_test_migration_catches_exceptions(manager: ModelManager) -> None:
    """Test test_migration catches and reports exceptions."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def bad_migration(data: ModelData) -> ModelData:
        raise ValueError("Migration failed")

    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "alice@example.com"}),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is False
    assert results.failures[0].error is not None
    assert "Migration failed" in results.failures[0].error


def test_test_migration_preserves_descriptions(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration preserves test case descriptions."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            MigrationTestCase(
                source={"name": "Alice"},
                target={"name": "Alice", "email": "unknown@example.com"},
                description="Test Alice",
            ),
        ],
    )

    assert results.results[0].test_case.description == "Test Alice"


def test_test_migration_multiple_failures(manager: ModelManager) -> None:
    """Test test_migration with multiple failures."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate(data: ModelData) -> ModelData:
        return {**data, "email": "default@example.com"}

    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "alice@example.com"}),
            ({"name": "Bob"}, {"name": "Bob", "email": "bob@example.com"}),
            ({"name": "Charlie"}, {"name": "Charlie", "email": "default@example.com"}),
        ],
    )

    assert len(results) == 3  # noqa: PLR2004
    assert results.all_passed is False
    assert len(results.failures) == 2  # noqa: PLR2004


def test_test_migration_empty_test_cases(registered_manager: ModelManager) -> None:
    """Test test_migration with empty test cases list."""
    results = registered_manager.test_migration("User", "1.0.0", "2.0.0", test_cases=[])

    assert len(results) == 0
    assert results.all_passed is True


def test_test_migration_with_model_versions(
    registered_manager: ModelManager,
) -> None:
    """Test test_migration with ModelVersion objects."""
    results = registered_manager.test_migration(
        "User",
        ModelVersion(1, 0, 0),
        ModelVersion(2, 0, 0),
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "unknown@example.com"}),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is True


def test_test_migration_chain_through_versions(manager: ModelManager) -> None:
    """Test test_migration works with chained migrations."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.model("User", "3.0.0")
    class UserV3(BaseModel):
        name: str
        email: str
        age: int

    @manager.migration("User", "1.0.0", "2.0.0")
    def migrate_1_to_2(data: ModelData) -> ModelData:
        return {**data, "email": "default@example.com"}

    @manager.migration("User", "2.0.0", "3.0.0")
    def migrate_2_to_3(data: ModelData) -> ModelData:
        return {**data, "age": 25}

    results = manager.test_migration(
        "User",
        "1.0.0",
        "3.0.0",
        test_cases=[
            (
                {"name": "Alice"},
                {"name": "Alice", "email": "default@example.com", "age": 25},
            ),
        ],
    )

    assert len(results) == 1
    assert results.all_passed is True


def test_test_migration_preserves_actual_on_failure(
    registered_manager: ModelManager,
) -> None:
    """Test that actual output is preserved even on failure."""
    results = registered_manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            ({"name": "Alice"}, {"name": "Alice", "email": "wrong@example.com"}),
        ],
    )

    assert results.failures[0].actual == {
        "name": "Alice",
        "email": "unknown@example.com",
    }
