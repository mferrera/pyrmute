# Migration Hooks

Migration hooks provide observability into the migration process. This guide
covers creating custom hooks, built-in hooks, and patterns for logging,
monitoring, and validation.

## Understanding Migration Hooks

Hooks are **read-only observers** that let you inject custom behavior before,
after, or on error during migrations. They're perfect for:

- **Logging** - Track migration activity
- **Metrics** - Collect performance data
- **Monitoring** - Alert on failures
- **Auditing** - Record data changes
- **Validation** - Verify migration results

!!! warning "Important"
    Hooks observe data but cannot modify it. Data transformation happens only
    in migration functions.

## Basic Hook Usage

### Creating a Simple Hook

```python
from pyrmute import MigrationHook, ModelManager, ModelVersion
from typing import Any, Mapping


class LoggingHook(MigrationHook):
    """Log all migration activity."""

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        """Called before migration starts."""
        print(f"Migrating {name} from {from_version} to {to_version}")

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        """Called after successful migration."""
        print(f"Successfully migrated {name}")

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        """Called when migration fails."""
        print(f"Migration failed: {error}")


# Register the hook
manager = ModelManager()
manager.add_hook(LoggingHook())

# Migrations will now trigger hook methods
user = manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")
# Output: Migrating User from 1.0.0 to 2.0.0
# Output: Successfully migrated User
```

### Hook Lifecycle

Hooks are called in this order:

```
1. before_migrate()  -> Called first
2. Migration runs    -> Data transforms
3. after_migrate()   -> Called on success
   OR
3. on_error()        -> Called on failure
```

```python
class LifecycleHook(MigrationHook):
    """Demonstrate hook lifecycle."""

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        print("1. Before migration")

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        print("3. After migration (success)")

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        print("3. On error (failure)")


manager.add_hook(LifecycleHook())

# Successful migration
manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")
# Output: 1. Before migration
# Output: 3. After migration (success)

# Failed migration
try:
    manager.migrate({"invalid": "data"}, "User", "1.0.0", "2.0.0")
except Exception:
    pass
# Output: 1. Before migration
# Output: 3. On error (failure)
```

## Built-in Hooks

### MetricsHook

Track migration statistics:

```python
from pyrmute import MetricsHook

metrics = MetricsHook()
manager.add_hook(metrics)

# Perform migrations
manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")
manager.migrate({"name": "Bob"}, "User", "1.0.0", "2.0.0")

# Check metrics
print(f"Total migrations: {metrics.total_count}")
# Output: Total migrations: 2

print(f"Success rate: {metrics.success_rate:.1%}")
# Output: Success rate: 100.0%

print(f"User migrations: {metrics.migrations_by_model['User']}")
# Output: User migrations: 2

# Errors are tracked too
try:
    manager.migrate({"bad": "data"}, "User", "1.0.0", "2.0.0")
except Exception:
    pass

print(f"Error count: {metrics.error_count}")
# Output: Error count: 1

print(f"Success rate: {metrics.success_rate:.1%}")
# Output: Success rate: 66.7%
```

### MetricsHook Attributes

```python
metrics = MetricsHook()

# Track these attributes:
metrics.total_count              # Total migration attempts
metrics.error_count              # Total failures
metrics.migrations_by_model      # dict[str, int] - counts per model
metrics.errors_by_model          # dict[str, int] - errors per model
metrics.success_rate             # float (0.0 to 1.0)
```

## Hook Management

### Adding and Removing Hooks

```python
manager = ModelManager()

# Add hooks
hook1 = LoggingHook()
hook2 = MetricsHook()

manager.add_hook(hook1)
manager.add_hook(hook2)

# Remove specific hook
manager.remove_hook(hook1)

# Clear all hooks
manager.clear_hooks()
```

### Multiple Hooks

Hooks are called in registration order:

```python
class Hook1(MigrationHook):
    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        print("Hook 1")


class Hook2(MigrationHook):
    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        print("Hook 2")


class Hook3(MigrationHook):
    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        print("Hook 3")


manager.add_hook(Hook1())
manager.add_hook(Hook2())
manager.add_hook(Hook3())

manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")
# Output: Hook 1
# Output: Hook 2
# Output: Hook 3
```

## Common Patterns

### Structured Logging

Use Python's logging module:

```python
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class StructuredLoggingHook(MigrationHook):
    """Log with structured data."""

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        logger.info(
            "Migration started",
            extra={
                "model": name,
                "from_version": str(from_version),
                "to_version": str(to_version),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        logger.info(
            "Migration completed",
            extra={
                "model": name,
                "from_version": str(from_version),
                "to_version": str(to_version),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        logger.error(
            "Migration failed",
            extra={
                "model": name,
                "from_version": str(from_version),
                "to_version": str(to_version),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": datetime.utcnow().isoformat(),
            },
            exc_info=True
        )
```

### Performance Monitoring

Track migration timing:

```python
import time

class PerformanceHook(MigrationHook):
    """Monitor migration performance."""

    def __init__(self) -> None:
        self.timings: dict[str, list[float]] = {}
        self._start_time: float | None = None

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        self._start_time = time.time()

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        if self._start_time is not None:
            duration = time.time() - self._start_time
            key = f"{name}:{from_version}→{to_version}"

            if key not in self.timings:
                self.timings[key] = []
            self.timings[key].append(duration)

            self._start_time = None

    def get_average_time(
        self, name: str, from_version: str, to_version: str
    ) -> float:
        """Get average migration time."""
        key = f"{name}:{from_version}→{to_version}"
        timings = self.timings.get(key, [])
        return sum(timings) / len(timings) if timings else 0.0

    def get_slowest_migrations(self, limit: int = 10) -> list[tuple[str, float]]:
        """Get slowest migrations."""
        averages = [
            (key, sum(times) / len(times))
            for key, times in self.timings.items()
        ]
        return sorted(averages, key=lambda x: x[1], reverse=True)[:limit]


# Usage
perf = PerformanceHook()
manager.add_hook(perf)

# Run migrations
for _ in range(100):
    manager.migrate({"name": "Alice"}, "User", "1.0.0", "2.0.0")

# Check performance
avg_time = perf.get_average_time("User", "1.0.0", "2.0.0")
print(f"Average migration time: {avg_time*1000:.2f}ms")

slowest = perf.get_slowest_migrations(limit=5)
for migration, avg_time in slowest:
    print(f"{migration}: {avg_time*1000:.2f}ms")
```

### Audit Trail

Record all migrations for compliance:

```python
from datetime import datetime
from typing import List
import json

class AuditHook(MigrationHook):
    """Create audit trail of all migrations."""

    def __init__(self, audit_file: str = "migrations.log"):
        self.audit_file = audit_file

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        """Record successful migration."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": name,
            "from_version": str(from_version),
            "to_version": str(to_version),
            "status": "success",
            "original_keys": list(original_data.keys()),
            "migrated_keys": list(migrated_data.keys()),
        }

        self._write_audit(audit_entry)

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        """Record failed migration."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": name,
            "from_version": str(from_version),
            "to_version": str(to_version),
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "data_keys": list(data.keys()),
        }

        self._write_audit(audit_entry)

    def _write_audit(self, entry: dict) -> None:
        """Write audit entry to file."""
        with open(self.audit_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

### Validation Hook

Verify migration results:

```python
class ValidationHook(MigrationHook):
    """Validate migration results."""

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        """Validate the migrated data."""
        # Ensure no data loss
        if not migrated_data:
            raise ValueError("Migration resulted in empty data")

        # Ensure required fields present
        if name == "User":
            if "name" not in migrated_data:
                raise ValueError("User migration missing required 'name' field")

        # Custom validation logic
        self._validate_specific_fields(name, migrated_data)

    def _validate_specific_fields(
        self,
        name: str,
        data: Mapping[str, Any]
    ) -> None:
        """Model-specific validation."""
        if name == "User":
            # Validate email format if present
            if "email" in data:
                email = data["email"]
                if "@" not in email:
                    raise ValueError(f"Invalid email format: {email}")

        elif name == "Product":
            # Validate price is positive
            if "price" in data:
                price = data["price"]
                if price < 0:
                    raise ValueError(f"Price cannot be negative: {price}")
```

### Alerting Hook

Send alerts on migration failures:

```python
class AlertingHook(MigrationHook):
    """Send alerts on migration failures."""

    def __init__(self, alert_threshold: int = 3):
        self.alert_threshold = alert_threshold
        self.recent_errors: list[dict] = []

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        """Track errors and alert if threshold exceeded."""
        self.recent_errors.append({
            "model": name,
            "from_version": str(from_version),
            "to_version": str(to_version),
            "error": str(error),
            "timestamp": datetime.utcnow(),
        })

        # Keep only recent errors (last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.recent_errors = [
            e for e in self.recent_errors
            if e["timestamp"] > cutoff
        ]

        # Alert if threshold exceeded
        if len(self.recent_errors) >= self.alert_threshold:
            self._send_alert()

    def _send_alert(self) -> None:
        """Send alert to monitoring system."""
        message = (
            f"ALERT: {len(self.recent_errors)} migration failures "
            f"in the last hour"
        )
        print(f"🚨 {message}")
        # In production: send to Slack, PagerDuty, etc.
```

## Advanced Patterns

### Contextual Hooks

Pass context to hooks:

```python
class ContextualHook(MigrationHook):
    """Hook that uses request context."""

    def __init__(
        self, user_id: str | None = None, request_id: str | None = None
    ) -> None:
        self.user_id = user_id
        self.request_id = request_id

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        logger.info(
            "Migration started",
            extra={
                "model": name,
                "user_id": self.user_id,
                "request_id": self.request_id,
            }
        )


# Usage with different contexts
def migrate_for_request(
    data: ModelData,
    name: str,
    from_v: str,
    to_v: str,
    user_id: str | None = None,
    request_id: str | None = None
) -> type[BaseModel]:
    """Migrate with request context."""
    manager.clear_hooks()
    manager.add_hook(ContextualHook(user_id, request_id))
    return manager.migrate(data, name, from_v, to_v)
```

### Conditional Hooks

Enable hooks based on conditions:

```python
import os


class ConditionalHook(MigrationHook):
    """Hook that only runs in specific environments."""

    def __init__(self) -> None:
        self.environment = os.getenv("ENVIRONMENT", "production")
        self.enabled = self.environment == "development"

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return

        # Only log in development
        print(f"DEV: Migrating {name}")
        print(f"DEV: Input data: {dict(data)}")

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return

        print(f"DEV: Output data: {dict(migrated_data)}")
```

### Sampling Hook

Only observe a percentage of migrations:

```python
import random


class SamplingHook(MigrationHook):
    """Sample a percentage of migrations for detailed logging."""

    def __init__(self, sample_rate: float = 0.1) -> None:
        """
        Args:
            sample_rate: Percentage to sample (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.should_log = False

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        # Randomly decide whether to log this migration
        self.should_log = random.random() < self.sample_rate

        if self.should_log:
            logger.debug(f"[SAMPLE] Migration: {name} {from_version}→{to_version}")
            logger.debug(f"[SAMPLE] Input: {dict(data)}")

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        if self.should_log:
            logger.debug(f"[SAMPLE] Output: {dict(migrated_data)}")


# Log 10% of migrations for debugging
manager.add_hook(SamplingHook(sample_rate=0.1))
```

### Composite Hook

Combine multiple behaviors:

```python
class CompositeHook(MigrationHook):
    """Hook that delegates to multiple sub-hooks."""

    def __init__(self, hooks: list[MigrationHook]) -> None:
        self.hooks = hooks

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        for hook in self.hooks:
            hook.before_migrate(name, from_version, to_version, data)

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        for hook in self.hooks:
            hook.after_migrate(
                name, from_version, to_version, original_data, migrated_data
            )

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        for hook in self.hooks:
            hook.on_error(name, from_version, to_version, data, error)


# Combine logging, metrics, and auditing
combined = CompositeHook([
    LoggingHook(),
    MetricsHook(),
    AuditHook(),
])
manager.add_hook(combined)
```

## Integration with Monitoring Systems

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram


class PrometheusHook(MigrationHook):
    """Export metrics to Prometheus."""

    def __init__(self) -> None:
        self.migrations_total = Counter(
            'migrations_total',
            'Total number of migrations',
            ['model', 'from_version', 'to_version', 'status']
        )
        self.migration_duration = Histogram(
            'migration_duration_seconds',
            'Migration duration in seconds',
            ['model', 'from_version', 'to_version']
        )
        self._start_time: float | None = None

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        self._start_time = time.time()

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        # Record success
        self.migrations_total.labels(
            model=name,
            from_version=str(from_version),
            to_version=str(to_version),
            status='success'
        ).inc()

        # Record duration
        if self._start_time:
            duration = time.time() - self._start_time
            self.migration_duration.labels(
                model=name,
                from_version=str(from_version),
                to_version=str(to_version)
            ).observe(duration)

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        # Record failure
        self.migrations_total.labels(
            model=name,
            from_version=str(from_version),
            to_version=str(to_version),
            status='error'
        ).inc()
```

### Datadog Integration

```python
from datadog import statsd


class DatadogHook(MigrationHook):
    """Send metrics to Datadog."""

    def __init__(self, prefix: str = "migrations") -> None:
        self.prefix = prefix
        self._start_time: float | None = None

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        self._start_time = time.time()
        statsd.increment(
            f"{self.prefix}.started",
            tags=[
                f"model:{name}",
                f"from_version:{from_version}",
                f"to_version:{to_version}",
            ]
        )

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        tags = [
            f"model:{name}",
            f"from_version:{from_version}",
            f"to_version:{to_version}",
        ]

        statsd.increment(f"{self.prefix}.success", tags=tags)

        if self._start_time:
            duration = time.time() - self._start_time
            statsd.histogram(f"{self.prefix}.duration", duration, tags=tags)

    def on_error(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
        error: Exception,
    ) -> None:
        statsd.increment(
            f"{self.prefix}.error",
            tags=[
                f"model:{name}",
                f"from_version:{from_version}",
                f"to_version:{to_version}",
                f"error_type:{type(error).__name__}",
            ]
        )
```

## Reading Hook Data

Hooks receive read-only `Mapping` views of data. You can use standard dict operations:

```python
class InspectionHook(MigrationHook):
    """Demonstrate reading data in hooks."""

    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        if "email" in data:
            print(f"Has email: {data['email']}")

        age = data.get("age", "unknown")
        print(f"Age: {age}")

        for key in data.keys():
            print(f"Key: {key}")

        for value in data.values():
            print(f"Value: {value}")

        for key, value in data.items():
            print(f"{key}: {value}")

        field_count = len(data)
        print(f"Fields: {field_count}")
```

## Testing Hooks

Always test your hooks:

```python
import io
import sys


def test_logging_hook() -> None:
    """Test logging hook behavior."""

    captured_output = io.StringIO()
    sys.stdout = captured_output

    hook = LoggingHook()
    manager = ModelManager()
    manager.add_hook(hook)

    @manager.model("Test", "1.0.0")
    class TestV1(BaseModel):
        value: str


    @manager.model("Test", "2.0.0")
    class TestV2(BaseModel):
        value: str
        extra: str = "default"


    @manager.migration("Test", "1.0.0", "2.0.0")
    def migrate(data):
        return {**data, "extra": "migrated"}

    manager.migrate({"value": "test"}, "Test", "1.0.0", "2.0.0")
    output = captured_output.getvalue()
    assert "Migrating Test" in output
    assert "Successfully migrated" in output

    # Restore stdout
    sys.stdout = sys.__stdout__


def test_metrics_hook() -> None:
    """Test metrics hook tracks correctly."""
    metrics = MetricsHook()
    manager = ModelManager()
    manager.add_hook(metrics)

    # Setup models...

    # Perform migrations
    manager.migrate({"value": "test1"}, "Test", "1.0.0", "2.0.0")
    manager.migrate({"value": "test2"}, "Test", "1.0.0", "2.0.0")

    # Verify metrics
    assert metrics.total_count == 2
    assert metrics.error_count == 0
    assert metrics.success_rate == 1.0
    assert metrics.migrations_by_model["Test"] == 2
```

## Best Practices

1. **Keep hooks lightweight** - They run on every migration
2. **Handle exceptions** - Don't let hook errors break migrations (unless
   validating)
3. **Use async carefully** - Hooks are synchronous, don't block
4. **Don't modify data** - Hooks are observers, not transformers
5. **Test thoroughly** - Verify hooks don't introduce bugs
6. **Monitor hook performance** - Slow hooks slow all migrations
7. **Use appropriate log levels** - Debug for detail, info for key events
8. **Document behavior** - Explain what each hook does

## Common Pitfalls

### Trying to Modify Data

```python
# ❌ BAD - Cannot modify data
class BadHook(MigrationHook):
    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        data["new_field"] = "value"  # Type error - Mapping is read-only


# ✅ GOOD - Only observe data
class GoodHook(MigrationHook):
    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        has_field = "new_field" in data  # Read-only access
```

### Slow Hooks

```python
# ❌ BAD - Slow operation blocks migrations
class SlowHook(MigrationHook):
    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        time.sleep(5)  # Blocks for 5 seconds!


# ✅ GOOD - Fast logging, defer expensive operations
class FastHook(MigrationHook):
    def __init__(self) -> None:
        self.queue = []

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        # Just record, process later
        self.queue.append({"name": name, "data": dict(migrated)})
```

### Not Handling Errors

```python
# ❌ BAD - Unhandled error breaks migration
class BrittleHook(MigrationHook):
    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        external_service.notify(name)  # Could fail!


# ✅ GOOD - Handle errors gracefully
class RobustHook(MigrationHook):
    def before_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        data: Mapping[str, Any],
    ) -> None:
        try:
            external_service.notify(name)
        except Exception as e:
            logger.warning(f"Failed to notify service: {e}")
```

### Stateful Hooks Without Cleanup

```python
# ❌ BAD - Resources not cleaned up
class LeakyHook(MigrationHook):
    def __init__(self) -> None:
        self.file = open("migrations.log", "a")  # Never closed!


# ✅ GOOD - Proper resource management
class CleanHook(MigrationHook):
    def __init__(self, log_file: str) -> None:
        self.log_file = log_file

    def after_migrate(
        self,
        name: str,
        from_version: ModelVersion,
        to_version: ModelVersion,
        original_data: Mapping[str, Any],
        migrated_data: Mapping[str, Any],
    ) -> None:
        with open(self.log_file, "a") as f:
            f.write(f"Migrated {name}\n")
```

## Next Steps

Now that you understand migration hooks:

**Continue learning:**

- [Writing Migrations](../user-guide/writing-migrations.md) - Combine hooks
    with migrations
- [Testing Migrations](../user-guide/testing-migrations.md) - Test hook
    behavior
- [Batch Migrations](../user-guide/batch-processing.md) - Use hooks with batch
    processing

**Related topics:**

- [Custom Schema Generators](custom-generators.md) - Advanced
    customization patterns
- [Auto-Migration](../user-guide/auto-migrations.md) - Hooks work with
    auto-migrations too

**API Reference:**

- [Migration Hooks API](../reference/migration-hooks.md) - Complete hook
    interface
- [ModelManager API](../reference/model-manager.md) - Hook management methods
