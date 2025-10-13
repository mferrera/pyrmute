"""Migration hooks for observability and custom behavior."""

from collections.abc import Mapping
from typing import Any, Self

from .model_version import ModelVersion


class MigrationHook:
    """Base class for migration hooks.

    Hooks are read-only observers that allow you to inject custom behavior before,
    after, or on error during migrations. Useful for logging, monitoring, metrics,
    validation, or auditing.

    Important:
        Hook methods receive read-only views of data and should NOT modify it.
        Data transformation should only happen in migration functions.

    Example:
        ```python
        class LoggingHook(MigrationHook):
            def before_migrate(
                self,
                name: str,
                from_version: ModelVersion,
                to_version: ModelVersion,
                data: Mapping[str, Any],
            ) -> None:
                logger.info(f"Migrating {name} from {from_version} to {to_version}")

            def after_migrate(
                self,
                name: str,
                from_version: ModelVersion,
                to_version: ModelVersion,
                original_data: Mapping[str, Any],
                migrated_data: Mapping[str, Any],
            ) -> None:
                logger.info(f"Successfully migrated {name}")

            def on_error(
                self,
                name: str,
                from_version: ModelVersion,
                to_version: ModelVersion,
                data: Mapping[str, Any],
                error: Exception,
            ) -> None:
                logger.error(f"Migration failed: {error}")
        ```
    """

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        """Called before migration starts.

        Args:
            name: Model name being migrated.
            from_version: Source version.
            to_version: Target version.
            data: Original data to be migrated (read-only view).

        Warning:
            Do not modify the data parameter. Hooks are for observation only.
        """

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        """Called after successful migration.

        Args:
            name: Model name that was migrated.
            from_version: Source version.
            to_version: Target version.
            original_data: Original input data (read-only view).
            migrated_data: Result of migration (read-only view).

        Warning:
            Do not modify the data parameters. Hooks are for observation only.
        """

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        """Called when migration fails.

        Args:
            name: Model name that failed to migrate.
            from_version: Source version.
            to_version: Target version.
            data: Original data that failed to migrate (read-only view).
            error: Exception that occurred.

        Warning:
            Do not modify the data parameter. Hooks are for observation only.
        """


class MetricsHook(MigrationHook):
    """Hook for collecting migration metrics.

    Tracks migration counts, timing, and errors for monitoring purposes.

    Example:
        ```python
        metrics = MetricsHook()
        manager.add_hook(metrics)

        # After migrations
        print(f"Total migrations: {metrics.total_count}")
        print(f"Failed migrations: {metrics.error_count}")
        print(f"Success rate: {metrics.success_rate:.1%}")
        ```
    """

    def __init__(self: Self) -> None:
        """Initialize metrics collection."""
        self.total_count = 0
        self.error_count = 0
        self.migrations_by_model: dict[str, int] = {}
        self.errors_by_model: dict[str, int] = {}

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        """Track migration attempt."""
        self.total_count += 1
        self.migrations_by_model[name] = self.migrations_by_model.get(name, 0) + 1

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        """Track migration error."""
        self.error_count += 1
        self.errors_by_model[name] = self.errors_by_model.get(name, 0) + 1

    @property
    def success_rate(self: Self) -> float:
        """Calculate overall success rate.

        Returns:
            Success rate as a float between 0 and 1.
        """
        if self.total_count == 0:
            return 0.0
        return (self.total_count - self.error_count) / self.total_count
