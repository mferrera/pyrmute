# Batch Processing

pyrmute provides batch processing capabilities for migrating large datasets
efficiently. This guide covers batch migration methods, streaming, parallel
processing, and performance optimization.

## Basic Batch Migration

Migrate multiple records at once using `migrate_batch()`:

```python
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
def split_name(data: ModelData) -> ModelData:
    parts = data["name"].split(" ", 1)
    return {
        "first_name": parts[0],
        "last_name": parts[1] if len(parts) > 1 else ""
    }

# Migrate batch of users
old_users = [
    {"name": "Alice Smith"},
    {"name": "Bob Jones"},
    {"name": "Carol White"},
]

new_users = manager.migrate_batch(
    old_users,
    "User",
    "1.0.0",
    "2.0.0"
)

print(new_users)
# [
#   UserV2(first_name="Alice", last_name="Smith"),
#   UserV2(first_name="Bob", last_name="Jones"),
#   UserV2(first_name="Carol", last_name="White")
# ]
```

**Key points:**

- Takes an iterable of data dictionaries
- Returns a list of validated Pydantic models
- Processes records sequentially by default
- All records must migrate successfully (fails fast)

## Return Types

pyrmute offers three return types for batch migrations:

### Validated Models (Default)

Returns Pydantic model instances:

```python
users = manager.migrate_batch(
    old_users,
    "User",
    "1.0.0",
    "2.0.0"
)
# Returns: List[UserV2]
# Each item is a validated Pydantic model
```

**Use when:**

- You need type safety
- You want Pydantic validation
- You'll use model methods/properties

### Type-Safe Models

Returns specific model type for better type checking:

```python
users: list[UserV2] = manager.migrate_batch_as(
    old_users,
    "User",
    "1.0.0",
    "2.0.0",
    UserV2  # Explicit type
)
# Returns: List[UserV2]
# Type checkers know the exact type
```

**Use when:**

- You need IDE autocomplete
- You're using type checkers (mypy, pyright)
- You want explicit type annotations

### Raw Dictionaries

Returns plain dictionaries without validation:

```python
users = manager.migrate_batch_data(
    old_users,
    "User",
    "1.0.0",
    "2.0.0"
)
# Returns: List[dict]
# Raw migrated data, no validation
```

**Use when:**

- You'll validate later
- You need maximum performance
- You're serializing immediately to JSON/database

## Streaming Large Datasets

For datasets that don't fit in memory, use streaming:

```python
def load_users_from_database() -> None:
    """Generator that yields users from database."""
    # Fetch in chunks to avoid loading all into memory
    offset = 0
    chunk_size = 1000

    while True:
        users = database.query(
            "SELECT * FROM users LIMIT ? OFFSET ?",
            chunk_size, offset
        )

        if not users:
            break

        for user in users:
            yield user

        offset += chunk_size


# Stream migration - processes in chunks
for migrated_user in manager.migrate_batch_streaming(
    load_users_from_database(),
    "User",
    "1.0.0",
    "2.0.0",
    chunk_size=1000
):
    # Save each migrated user immediately
    database.save(migrated_user)
    print(f"Migrated user: {migrated_user.first_name}")
```

**Key benefits:**

- Constant memory usage regardless of dataset size
- Can process millions of records
- Fails fast on first error
- Progress visible during processing

### Streaming Variants

Three streaming methods match the three return types:

```python
# Streaming with validated models (default)
for user in manager.migrate_batch_streaming(
    data_source,
    "User",
    "1.0.0",
    "2.0.0",
    chunk_size=1000
):
    process(user)  # user is UserV2

# Streaming with type safety
for user in manager.migrate_batch_streaming_as(
    data_source,
    "User",
    "1.0.0",
    "2.0.0",
    UserV2,
    chunk_size=1000
):
    process(user)  # Type checker knows user is UserV2

# Streaming raw dictionaries
for user_data in manager.migrate_batch_data_streaming(
    data_source,
    "User",
    "1.0.0",
    "2.0.0",
    chunk_size=1000
):
    process(user_data)  # user_data is dict
```

### Chunk Size Selection

Choose chunk size based on your constraints:

```python
# Small chunks (100-500) - Lower memory, more overhead
for user in manager.migrate_batch_streaming(
    data_source,
    "User",
    "1.0.0",
    "2.0.0",
    chunk_size=100  # Good for: Memory-constrained environments
):
    process(user)

# Medium chunks (1000-5000) - Balanced (default: 100)
for user in manager.migrate_batch_streaming(
    data_source,
    "User",
    "1.0.0",
    "2.0.0",
    chunk_size=1000  # Good for: Most use cases
):
    process(user)

# Large chunks (10000+) - Higher memory, less overhead
for user in manager.migrate_batch_streaming(
    data_source,
    "User",
    "1.0.0",
    "2.0.0",
    chunk_size=10000  # Good for: Fast migrations, ample memory
):
    process(user)
```

## Parallel Processing

Enable parallel processing for CPU-intensive migrations:

```python
# Sequential processing (default)
users = manager.migrate_batch(
    old_users,
    "User",
    "1.0.0",
    "2.0.0"
)

# Parallel processing with threads
users = manager.migrate_batch(
    old_users,
    "User",
    "1.0.0",
    "2.0.0",
    parallel=True,
    max_workers=4
)

# Parallel processing with processes
users = manager.migrate_batch(
    old_users,
    "User",
    "1.0.0",
    "2.0.0",
    parallel=True,
    use_processes=True,
    max_workers=4
)
```

### Threads vs. Processes

#### ThreadPoolExecutor (Default)

```python
users = manager.migrate_batch(
    old_users,
    "User",
    "1.0.0",
    "2.0.0",
    parallel=True,
    use_processes=False,  # Use threads (default)
    max_workers=4
)
```

**Use threads when:**

- Migrations are I/O-bound (reading files, API calls)
- Working with shared state
- Lower overhead needed
- GIL isn't a bottleneck

**Limitations:**

- Python's GIL limits CPU parallelism
- Not effective for pure computation

#### ProcessPoolExecutor

```python
users = manager.migrate_batch(
    old_users,
    "User",
    "1.0.0",
    "2.0.0",
    parallel=True,
    use_processes=True,  # Use processes
    max_workers=4
)
```

**Use processes when:**

- Migrations are CPU-intensive (complex transformations)
- Need true parallelism
- No shared state required
- Have enough memory

**Limitations:**

- Higher memory overhead (copies data)
- Slower startup time
- Serialization overhead
- Can't share state between processes

### Worker Count Selection

```python
import os

# Automatic: Use CPU count
users = manager.migrate_batch(
    old_users,
    "User",
    "1.0.0",
    "2.0.0",
    parallel=True,
    max_workers=None  # Auto-detects CPU count
)

# Manual: Specific number
users = manager.migrate_batch(
    old_users,
    "User",
    "1.0.0",
    "2.0.0",
    parallel=True,
    max_workers=4  # Explicit count
)

# Conservative: Leave cores for other work
cpu_count = os.cpu_count() or 1
users = manager.migrate_batch(
    old_users,
    "User",
    "1.0.0",
    "2.0.0",
    parallel=True,
    max_workers=max(1, cpu_count - 2)  # Reserve 2 cores
)
```

## Performance Optimization

### When to Use Parallel Processing

**Use parallel processing when:**

```python
import time


# CPU-intensive migration
@manager.migration("Document", "1.0.0", "2.0.0")
def complex_transformation(data: ModelData) -> ModelData:
    # Expensive text processing
    text = data["content"]

    # Complex transformations
    processed = analyze_sentiment(text)
    keywords = extract_keywords(text)
    summary = generate_summary(text)

    return {
        **data,
        "sentiment": processed,
        "keywords": keywords,
        "summary": summary
    }


# Large dataset + CPU-intensive = Good candidate for parallelization
documents = manager.migrate_batch(
    large_document_list,
    "Document",
    "1.0.0",
    "2.0.0",
    parallel=True,
    use_processes=True,
    max_workers=8
)
```

**Don't use parallel processing when:**

```python
# Simple migration - overhead exceeds benefit
@manager.migration("User", "1.0.0", "2.0.0")
def simple_migration(data: ModelData) -> ModelData:
    return {**data, "status": "active"}


# Sequential is faster for simple migrations
users = manager.migrate_batch(
    users_list,
    "User",
    "1.0.0",
    "2.0.0"
    # parallel=False (default)
)
```

### Benchmarking

Always benchmark to determine optimal settings:

```python
import time

def benchmark_migration(
    data: Iterable[ModelData],
    parallel: bool = False,
    use_processes: bool = False,
    workers: int | None = None
) -> list[UserV2]:
    """Benchmark migration performance."""
    start = time.time()

    results = manager.migrate_batch(
        data,
        "User",
        "1.0.0",
        "2.0.0",
        parallel=parallel,
        use_processes=use_processes,
        max_workers=workers
    )

    duration = time.time() - start
    throughput = len(data) / duration

    print(f"Mode: {'Parallel' if parallel else 'Sequential'}")
    if parallel:
        mode = "Processes" if use_processes else "Threads"
        print(f"  Type: {mode}, Workers: {workers}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Throughput: {throughput:.0f} records/sec")

    return results


# Test different configurations
test_data = [{"name": f"User {i}"} for i in range(10000)]

print("Sequential:")
benchmark_migration(test_data)

print("\nThreads (4 workers):")
benchmark_migration(test_data, parallel=True, workers=4)

print("\nProcesses (4 workers):")
benchmark_migration(test_data, parallel=True, use_processes=True, workers=4)
```

### Memory Management

Control memory usage with streaming:

```python
# BAD - Loads entire dataset into memory
with open("large_file.json") as f:
    all_data = json.load(f)  # 10GB in memory!

users = manager.migrate_batch(all_data, "User", "1.0.0", "2.0.0")


# GOOD - Streams data, constant memory
def read_jsonl(filepath: str | Path) -> Generator[str]:
    """Read JSON lines file (one JSON per line)."""
    with open(filepath) as f:
        for line in f:
            yield json.loads(line)


for user in manager.migrate_batch_streaming(
    read_jsonl("large_file.jsonl"),
    "User",
    "1.0.0",
    "2.0.0",
    chunk_size=1000
):
    database.save(user)
```

## Error Handling

### Fail Fast (Default)

By default, batch migrations fail on first error:

```python
data = [
    {"name": "Alice Smith"},
    {"name": ""},  # Invalid - will cause failure
    {"name": "Bob Jones"},
]

try:
    users = manager.migrate_batch(data, "User", "1.0.0", "2.0.0")
except Exception as e:
    print(f"Migration failed: {e}")
    # Stops at first error
```

### Graceful Error Handling

Handle errors manually for better control:

```python
def migrate_with_error_handling(
    data: Iterable[ModelData]
) -> tuple[list[UserV2], list[UserV2]]:
    """Migrate data with error tracking."""
    successful = []
    failed = []

    for i, record in enumerate(data):
        try:
            migrated = manager.migrate(
                record,
                "User",
                "1.0.0",
                "2.0.0"
            )
            successful.append(migrated)
        except Exception as e:
            failed.append({
                "index": i,
                "data": record,
                "error": str(e)
            })
            print(f"Failed to migrate record {i}: {e}")

    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    return successful, failed


successful_users, failed_records = migrate_with_error_handling(user_data)

# Save successful migrations
for user in successful_users:
    database.save(user)

# Log failures for manual review
for failure in failed_records:
    logger.error(f"Migration failed", extra=failure)
```

### Retry Logic

Implement retry logic for transient failures:

```python
import time


def migrate_with_retry(
    data: Iterable[ModelData], max_retries: int = 3
) -> list[UserV2]:
    """Migrate with retry logic."""
    results = []

    for record in data:
        for attempt in range(max_retries):
            try:
                migrated = manager.migrate(
                    record,
                    "User",
                    "1.0.0",
                    "2.0.0"
                )
                results.append(migrated)
                break  # Success
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    print(f"Failed after {max_retries} attempts: {e}")
                    raise
                else:
                    # Retry with exponential backoff
                    wait_time = 2 ** attempt
                    print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s")
                    time.sleep(wait_time)

    return results
```

## Common Patterns

### Database Migration

```python
def migrate_database_table(table_name: str, batch_size: int = 1000) -> None:
    """Migrate entire database table."""

    def fetch_records() -> Generator[ModelData]:
        """Generator to fetch records in batches."""
        offset = 0
        while True:
            records = database.query(
                f"SELECT * FROM {table_name} LIMIT ? OFFSET ?",
                batch_size, offset
            )

            if not records:
                break

            for record in records:
                yield record

            offset += batch_size

    # Migrate and update in place
    migrated_count = 0

    for user in manager.migrate_batch_streaming(
        fetch_records(),
        "User",
        "1.0.0",
        "2.0.0",
        chunk_size=batch_size
    ):
        database.update(
            f"UPDATE {table_name} SET first_name=?, last_name=? WHERE id=?",
            user.first_name, user.last_name, user.id
        )
        migrated_count += 1

        if migrated_count % 10000 == 0:
            print(f"Migrated {migrated_count} records...")

    print(f"Complete! Migrated {migrated_count} records total")
```

### File Processing

```python
import json
from pathlib import Path


def migrate_json_files(input_dir: Path, output_dir: Path) -> None:
    """Migrate all JSON files in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_dir.glob("*.json"):
        print(f"Processing {input_file.name}...")

        with open(input_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            migrated = manager.migrate_batch(
                data,
                "User",
                "1.0.0",
                "2.0.0"
            )
        else:
            # Single record
            migrated = manager.migrate(
                data,
                "User",
                "1.0.0",
                "2.0.0"
            )

        output_file = output_dir / input_file.name
        with open(output_file, "w") as f:
            if isinstance(migrated, list):
                json.dump([m.model_dump() for m in migrated], f, indent=2)
            else:
                json.dump(migrated.model_dump(), f, indent=2)

        print(f"  Saved to {output_file}")


migrate_json_files(Path("data/v1"), Path("data/v2"))
```

### CSV Processing

```python
import csv


def migrate_csv(input_path: str, output_path: str) -> None:
    """Migrate CSV file with streaming."""

    def read_csv() -> Generator[ModelData]:
        """Generator to read CSV rows."""
        with open(input_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row

    # Migrate with streaming
    with open(output_path, "w", newline="") as f:
        writer = None
        row_count = 0

        for migrated in manager.migrate_batch_streaming(
            read_csv(),
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=1000
        ):
            # Initialize writer with field names from first record
            if writer is None:
                fieldnames = list(migrated.model_dump().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            # Write migrated row
            writer.writerow(migrated.model_dump())
            row_count += 1

            if row_count % 10000 == 0:
                print(f"Processed {row_count} rows...")

    print(f"Complete! Migrated {row_count} rows")


migrate_csv("users_v1.csv", "users_v2.csv")
```

### Progress Tracking

```python
from tqdm import tqdm


def migrate_with_progress(
    data: Iterable[ModelData], desc: str = "Migrating"
) -> list[UserV2]:
    """Migrate with progress bar."""
    results = []

    # Convert to list to get length for progress bar
    data_list = list(data)

    for record in tqdm(data_list, desc=desc):
        migrated = manager.migrate(record, "User", "1.0.0", "2.0.0")
        results.append(migrated)

    return results


# Or with streaming
def migrate_streaming_with_progress(
    data: Iterable[ModelData], total: int, desc: str = "Migrating"
) -> list[UserV2]:
    """Migrate streaming data with progress bar."""
    results = []

    with tqdm(total=total, desc=desc) as pbar:
        for migrated in manager.migrate_batch_streaming(
            data,
            "User",
            "1.0.0",
            "2.0.0",
            chunk_size=1000
        ):
            results.append(migrated)
            pbar.update(1)


    return results
```

## Performance Tips

1. **Use streaming for large datasets** - Keeps memory constant
2. **Benchmark before parallelizing** - Overhead can exceed benefits
3. **Use processes for CPU-bound** - Threads for I/O-bound
4. **Tune chunk size** - Larger chunks = less overhead, more memory
5. **Profile migrations** - Identify bottlenecks in migration functions
6. **Consider raw dictionaries** - Skip validation if not needed
7. **Batch database operations** - Don't commit after every record

## Monitoring and Logging

Add instrumentation to track migration performance:

```python
import logging
import time

logger = logging.getLogger(__name__)


def migrate_with_metrics(
    data: Iterable[ModelData],
    model_name: str,
    from_version: str,
    to_version: str
) -> list[BaseModel]:
    """Migrate with detailed metrics."""
    start_time = time.time()
    record_count = 0
    error_count = 0

    try:
        results = []

        for record in data:
            try:
                migrated = manager.migrate(
                    record,
                    model_name,
                    from_version,
                    to_version
                )
                results.append(migrated)
                record_count += 1
            except Exception as e:
                error_count += 1
                logger.error(
                    f"Migration error",
                    extra={
                        "model": model_name,
                        "from_version": from_version,
                        "to_version": to_version,
                        "record": record,
                        "error": str(e)
                    }
                )

        duration = time.time() - start_time
        throughput = record_count / duration if duration > 0 else 0

        logger.info(
            f"Migration complete",
            extra={
                "model": model_name,
                "from_version": from_version,
                "to_version": to_version,
                "records": record_count,
                "errors": error_count,
                "duration_seconds": duration,
                "throughput": throughput
            }
        )

        return results

    except Exception as e:
        logger.exception("Migration failed catastrophically")
        raise
```

## Best Practices

1. **Test with production data samples** - Before bulk migration
2. **Start with small batches** - Verify correctness before scaling
3. **Monitor memory usage** - Use streaming if memory grows
4. **Log failures** - Track which records failed and why
5. **Have a rollback plan** - Backup data before bulk migrations
6. **Measure performance** - Benchmark different approaches
7. **Use type-safe variants** - When you need type checking

## Next Steps

Now that you understand batch processing:

**Continue learning:**

- [Nested Models](nested-models.md) - Batch migrate data with nested models
- [Discriminated Unions](../advanced/discriminated-unions.md) - Batch migrate
    polymorphic data
- [Schema Generation](schema-generation.md) - Generate schemas for batch
    operations

**Related topics:**

- [Writing Migrations](writing-migrations.md) - Optimize migration functions
    for performance
- [Testing Migrations](testing-migrations.md) - Test batch processing
    scenarios

**API Reference:**

- [`ModelManager` API](../reference/model-manager.md) - Complete
    `ModelManager` details
- [Exceptions](../reference/exceptions.md) - Exceptions pyrmute raises
- [Types](../reference/types.md) - Type alises exported by pyrmute
