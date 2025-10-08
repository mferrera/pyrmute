"""Migration testing utilities."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

from .types import MigrationData


@dataclass
class MigrationTestCase:
    """Test case for migration validation.

    Defines input data and expected output for testing a migration function. If target
    is None, the test only verifies the migration doesn't crash.

    Attributes:
        source: Input data to migrate.
        target: Expected output after migration. If None, only validates that migration
            completes without errors.
        description: Optional description of what this test case validates.

    Example:
        >>> test_case = MigrationTestCase(
        ...     source={"name": "Alice"},
        ...     target={"name": "Alice", "email": "alice@example.com"},
        ...     description="Adds default email field"
        ... )
    """

    source: MigrationData
    target: MigrationData | None = None
    description: str = ""


@dataclass
class MigrationTestResult:
    """Result of a single migration test case.

    Contains the test case, actual output, pass/fail status, and any error message.

    Attributes:
        test_case: Original test case that was executed.
        actual: Actual output produced by the migration.
        passed: Whether the test passed (output matched expected or no errors).
        error: Error message if test failed, None if passed.

    Example:
        >>> result = MigrationTestResult(
        ...     test_case=test_case,
        ...     actual={"name": "Alice", "email": "alice@example.com"},
        ...     passed=True,
        ...     error=None
        ... )
    """

    test_case: MigrationTestCase
    actual: MigrationData
    passed: bool
    error: str | None = None

    def __str__(self: Self) -> str:
        """Format test result as human-readable string."""
        if self.passed:
            desc = (
                f" - {self.test_case.description}" if self.test_case.description else ""
            )
            return f"✓ Test passed{desc}"

        desc = f" - {self.test_case.description}" if self.test_case.description else ""
        return f"""✗ Test failed{desc}
  Source:   {self.test_case.source}
  Expected: {self.test_case.target}
  Actual:   {self.actual}
  Error:    {self.error}"""


class MigrationTestResults:
    """Collection of migration test results.

    Provides convenient methods for checking overall test status and accessing failed
    tests.

    Attributes:
        results: List of individual test results.

    Example:
        >>> results = MigrationTestResults([result1, result2, result3])
        >>> if results.all_passed:
        ...     print("All tests passed!")
        >>> else:
        ...     print(f"{len(results.failures)} test(s) failed")
        ...     for failure in results.failures:
        ...         print(failure)
    """

    def __init__(self: Self, results: list[MigrationTestResult]) -> None:
        """Initialize test results collection.

        Args:
            results: List of individual test results.
        """
        self.results = results

    @property
    def all_passed(self: Self) -> bool:
        """Check if all tests passed.

        Returns:
            True if all tests passed, False if any failed.
        """
        return all(r.passed for r in self.results)

    @property
    def failures(self: Self) -> list[MigrationTestResult]:
        """Get list of failed tests.

        Returns:
            List of test results that failed.
        """
        return [r for r in self.results if not r.passed]

    def assert_all_passed(self: Self) -> None:
        """Assert all tests passed, raising detailed error if any failed.

        Raises:
            AssertionError: If any tests failed, with details about failures.

        Example:
            >>> # Use in pytest
            >>> def test_user_migration():
            ...     res = manager.test_migration("User", "1.0.0", "2.0.0", test_cases)
            ...     res.assert_all_passed()
        """
        if not self.all_passed:
            messages = [str(f) for f in self.failures]
            raise AssertionError(
                f"\n{len(self.failures)} migration test(s) failed:\n"
                + "\n\n".join(messages)
            )

    def __len__(self: Self) -> int:
        """Get total number of test results."""
        return len(self.results)

    def __iter__(self: Self) -> Iterator[MigrationTestResult]:
        """Iterate over test results."""
        return iter(self.results)

    def __str__(self) -> str:
        """Format results summary as string."""
        total_count = len(self.results)

        if self.all_passed:
            return f"✓ All {total_count} test(s) passed"

        passed_count = total_count - len(self.failures)
        return (
            f"✗ {len(self.failures)} of {total_count} test(s) failed "
            f"({passed_count} passed)"
        )
