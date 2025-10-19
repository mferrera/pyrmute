# Avro Schema Generation

pyrmute can generate [Apache Avro](https://avro.apache.org/) schemas for all
your model versions. Avro is widely used in data engineering, especially with
Apache Kafka, Hadoop, and data lakes. This guide covers Avro schema
generation, logical types, and integration with [Confluent Schema
Registry](https://github.com/confluentinc/schema-registry).

## Why Avro?

Apache Avro is used for:

- **Event streaming** - Kafka topics with schema evolution
- **Data lakes** - Efficient columnar storage (Parquet, ORC)
- **Big data processing** - Hadoop, Spark, Flink
- **Cross-language systems** - Language-agnostic serialization

**Avro vs JSON Schema:**

| Feature | JSON Schema | Avro |
|---------|-------------|------|
| Use case | API validation | Data serialization |
| Schema evolution | Manual | Built-in forward/backward compatibility |
| Size | Larger (self-describing) | Compact (schema separate) |
| Performance | Slower | Faster |
| Ecosystem | OpenAPI, Swagger | Kafka, Hadoop, Spark |

## Basic Avro Schema Generation

Generate an Avro schema for any registered model:

```python
from pydantic import BaseModel, Field
from pyrmute import ModelManager

manager = ModelManager()


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    """User account information."""
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    age: int = Field(ge=0, le=150, description="User's age in years")


# Generate Avro schema
schema = manager.get_avro_schema("User", "1.0.0", namespace="com.myapp")
print(json.dumps(schema, indent=2))
```

Output:
```json
{
  "type": "record",
  "name": "User",
  "namespace": "com.myapp.v1_0_0",
  "fields": [
    {
      "name": "name",
      "type": "string",
      "doc": "User's full name"
    },
    {
      "name": "email",
      "type": "string",
      "doc": "User's email address"
    },
    {
      "name": "age",
      "type": "int",
      "doc": "User's age in years"
    }
  ],
  "doc": "User account information."
}
```

## Type Mapping

### Basic Types

Python types are automatically mapped to Avro types:

| Python Type | Avro Type |
|-------------|-----------|
| `str` | `string` |
| `int` | `int` or `long`* |
| `float` | `double` |
| `bool` | `boolean` |
| `bytes` | `bytes` |

*Integer size is optimized based on constraints (see [Integer
Optimization](#integer-optimization))

```python
@manager.model("BasicTypes", "1.0.0")
class BasicTypesV1(BaseModel):
    name: str          # -> "string"
    count: int         # -> "int"
    price: float       # -> "double"
    active: bool       # -> "boolean"
    data: bytes        # -> "bytes"
```

### Logical Types

Special Python types use Avro logical types for precise semantics:

| Python Type | Avro Type | Logical Type |
|-------------|-----------|--------------|
| `datetime` | `long` | `timestamp-micros` |
| `date` | `int` | `date` |
| `time` | `long` | `time-micros` |
| `UUID` | `string` | `uuid` |
| `Decimal` | `bytes` | `decimal` |

```python
from datetime import datetime, date, time
from uuid import UUID
from decimal import Decimal


@manager.model("Event", "1.0.0")
class EventV1(BaseModel):
    event_id: UUID          # -> {"type": "string", "logicalType": "uuid"}
    timestamp: datetime     # -> {"type": "long", "logicalType": "timestamp-micros"}
    scheduled_date: date    # -> {"type": "int", "logicalType": "date"}
    scheduled_time: time    # -> {"type": "long", "logicalType": "time-micros"}
    amount: Decimal         # -> {"type": "bytes", "logicalType": "decimal"}
```

### Collection Types

Lists and dicts map to Avro arrays and maps:

```python
@manager.model("Collections", "1.0.0")
class CollectionsV1(BaseModel):
    tags: list[str]           # -> {"type": "array", "items": "string"}
    scores: list[int]         # -> {"type": "array", "items": "int"}
    metadata: dict[str, str]  # -> {"type": "map", "values": "string"}
    counts: dict[str, int]    # -> {"type": "map", "values": "int"}
```

### Optional Fields

Optional fields become unions with `null`:

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str                    # -> "string" (required)
    email: str | None = None     # -> ["null", "string"] (optional)
    age: int | None = None       # -> ["null", "int"] (optional)
```

Avro schema:
```json
{
  "fields": [
    {"name": "name", "type": "string"},
    {"name": "email", "type": ["null", "string"], "default": null},
    {"name": "age", "type": ["null", "int"], "default": null}
  ]
}
```

### Enum Types

Python Enums map to Avro enums:

```python
from enum import StrEnum


class Status(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"


@manager.model("Task", "1.0.0")
class TaskV1(BaseModel):
    name: str
    status: Status
```

Avro schema:
```json
{
  "fields": [
    {"name": "name", "type": "string"},
    {
      "name": "status",
      "type": {
        "type": "enum",
        "name": "Status",
        "symbols": ["pending", "active", "completed"]
      }
    }
  ]
}
```

### Union Types

Union types become Avro unions:

```python
@manager.model("Flexible", "1.0.0")
class FlexibleV1(BaseModel):
    value: str | int             # -> ["string", "int"]
    optional: str | int | None   # -> ["null", "string", "int"]
```

### Nested Models

Pydantic models become Avro records:

```python
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str
    city: str
    zip_code: str


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1
```

Avro schema:
```json
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "name", "type": "string"},
    {
      "name": "address",
      "type": {
        "type": "record",
        "name": "AddressV1",
        "fields": [
          {"name": "street", "type": "string"},
          {"name": "city", "type": "string"},
          {"name": "zip_code", "type": "string"}
        ]
      }
    }
  ]
}
```

## Avro Namespaces

Avro uses namespaces to organize schemas, similar to Java packages. pyrmute
automatically creates versioned namespaces:

```python
# Version is included in namespace
schema = manager.get_avro_schema("User", "1.0.0", namespace="com.mycompany")
# namespace: "com.mycompany.v1_0_0"

schema = manager.get_avro_schema("User", "2.0.0", namespace="com.mycompany")
# namespace: "com.mycompany.v2_0_0"
```

This ensures different versions don't conflict in schema registries.

**Best practices:**

- Use reverse domain notation: `com.company.domain`
- Be consistent across your organization
- Examples: `com.acme.events`, `com.acme.analytics`, `org.example.users`

## Exporting Avro Schemas

### Export All Schemas

Export Avro schemas for all registered models:

```python
manager.dump_avro_schemas(
    "schemas/avro/",
    namespace="com.mycompany.events"
)
```

Creates files like:
```
schemas/avro/
├── User_v1_0_0.avsc
├── User_v2_0_0.avsc
├── Order_v1_0_0.avsc
└── Product_v1_0_0.avsc
```

### Export Options

Customize the export:

```python
manager.dump_avro_schemas(
    "schemas/avro/",
    namespace="com.mycompany",
    indent=2,              # JSON indentation
    include_docs=True      # Include field descriptions
)
```

### Export Without Documentation

For production schemas without documentation overhead:

```python
manager.dump_avro_schemas(
    "schemas/avro/",
    namespace="com.mycompany",
    include_docs=False  # Omit doc fields
)
```

## Kafka Integration

### Basic Kafka Producer

Use Avro schemas with Kafka:

```python
from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer
import json


@manager.model("UserEvent", "1.0.0")
class UserEventV1(BaseModel):
    event_id: UUID
    user_id: str
    action: str
    timestamp: datetime


# Get Avro schema
schema = manager.get_avro_schema("UserEvent", "1.0.0", namespace="com.events")
value_schema = avro.loads(json.dumps(schema))

# Create producer
producer = AvroProducer({
    "bootstrap.servers": "localhost:9092",
    "schema.registry.url": "http://localhost:8081"
}, default_value_schema=value_schema)

# Send event
event = UserEventV1(
    event_id=uuid4(),
    user_id="user123",
    action="login",
    timestamp=datetime.now()
)

producer.produce(
    topic="user-events",
    value={
        "event_id": str(event.event_id),
        "user_id": event.user_id,
        "action": event.action,
        "timestamp": int(event.timestamp.timestamp() * 1_000_000)
    }
)
producer.flush()
```

### Kafka Schema Registry

Register schemas with Confluent Schema Registry:

```python
from confluent_kafka.schema_registry import SchemaRegistryClient, Schema


# Export schemas to files
manager.dump_avro_schemas("schemas/avro/", namespace="com.events")

# Register with Schema Registry
client = SchemaRegistryClient({"url": "http://localhost:8081"})

with open("schemas/avro/UserEvent_v1_0_0.avsc") as f:
    schema_str = f.read()
    schema = Schema(schema_str, schema_type="AVRO")
    client.register_schema(
        subject_name="user-events-value",
        schema=schema
    )
```

### Schema Evolution in Kafka

Manage schema evolution with proper compatibility:

```python
# Version 1: Original event
@manager.model("UserEvent", "1.0.0")
class UserEventV1(BaseModel):
    event_id: UUID
    user_id: str
    action: str


# Version 2: Add optional fields (backward compatible)
@manager.model("UserEvent", "2.0.0")
class UserEventV2(BaseModel):
    event_id: UUID
    user_id: str
    action: str
    metadata: dict[str, str] | None = None  # Optional - backward compatible
    timestamp: datetime | None = None       # Optional - backward compatible


# Register both versions
v1_schema = manager.get_avro_schema("UserEvent", "1.0.0", namespace="com.events")
v2_schema = manager.get_avro_schema("UserEvent", "2.0.0", namespace="com.events")

# V2 is backward compatible - old consumers can read new messages
# V2 is forward compatible - new consumers can read old messages (with defaults)
```

## fastavro Integration

Use schemas with the popular fastavro library:

```python
import fastavro
import io


@manager.model("Order", "1.0.0")
class OrderV1(BaseModel):
    order_id: str
    customer_name: str
    total: float
    items: list[str]


# Get Avro schema
schema = manager.get_avro_schema("Order", "1.0.0", namespace="com.orders")
parsed_schema = fastavro.parse_schema(schema)

# Write data
orders = [
    {
        "order_id": "O001",
        "customer_name": "Alice",
        "total": 99.99,
        "items": ["item1", "item2"]
    },
    {
        "order_id": "O002",
        "customer_name": "Bob",
        "total": 149.99,
        "items": ["item3"]
    }
]

output = io.BytesIO()
fastavro.writer(output, parsed_schema, orders)

# Read data back
output.seek(0)
for order in fastavro.reader(output):
    print(f"Order {order['order_id']}: ${order['total']}")
```

## Schema Evolution Best Practices

### Backward Compatible Changes

Add optional fields with defaults:

```python
# Version 1
@manager.model("Product", "1.0.0")
class ProductV1(BaseModel):
    id: str
    name: str
    price: float


# Version 2: Add optional fields
@manager.model("Product", "2.0.0")
class ProductV2(BaseModel):
    id: str
    name: str
    price: float
    description: str | None = None  # Backward compatible
    category: str | None = None     # Backward compatible
    in_stock: bool = True           # Has default
```

Old consumers can read new data (ignore new fields).

### Forward Compatible Changes

Don't remove required fields:

```python
# Version 2: Breaking change
@manager.model("Product", "2.0.0")
class ProductV2(BaseModel):
    id: str
    name: str
    # price: float  # Removed - NOT forward compatible
```

New consumers cannot read old data (missing field).

### Full Compatibility

Both add optional fields AND keep all existing fields:

```python
# Version 2: Fully compatible
@manager.model("Product", "2.0.0")
class ProductV2(BaseModel):
    id: str
    name: str
    price: float
    # New optional fields only
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
```

Works in both directions:
- Old consumers ← New producers ✓
- New consumers ← Old producers ✓

## Integer Optimization

pyrmute optimizes integer types based on constraints:

```python
from typing import Annotated


@manager.model("Optimized", "1.0.0")
class OptimizedV1(BaseModel):
    # Fits in 32-bit int
    small_number: Annotated[int, Field(ge=0, le=1000)] = 0
    # -> "int" (32-bit)

    # No constraints - uses safe default
    big_number: int
    # -> "int" (default, but could overflow)

    # Explicitly large range
    huge_number: Annotated[int, Field(ge=0, le=10_000_000_000)]
    # -> "long" (64-bit)
```

This produces more efficient schemas when possible while maintaining safety.

## Real-World Examples

### Event Sourcing

```python
from enum import Enum


class EventType(str, Enum):
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"


@manager.model("UserEvent", "1.0.0")
class UserEventV1(BaseModel):
    """User lifecycle event for event sourcing."""

    event_id: UUID = Field(description="Unique event identifier")
    event_type: EventType = Field(description="Type of event")
    aggregate_id: str = Field(description="User ID")
    timestamp: datetime = Field(description="When event occurred")
    payload: dict[str, str] = Field(description="Event-specific data")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# Export for event store
manager.dump_avro_schemas("event-store/schemas/", namespace="com.events")
```

### Data Lake Ingestion

```python
@manager.model("ClickEvent", "1.0.0")
class ClickEventV1(BaseModel):
    """Click tracking event for data lake."""

    session_id: UUID
    user_id: str | None = None  # Optional for anonymous users
    timestamp: datetime
    url: str
    referrer: str | None = None
    user_agent: str
    ip_address: str
    country_code: str
    device_type: str
    browser: str
    page_load_time_ms: int


# Write to data lake with Avro
schema = manager.get_avro_schema("ClickEvent", "1.0.0", namespace="com.analytics")

# Use with Spark, Parquet, etc.
```

### CDC (Change Data Capture)

```python
@manager.model("DatabaseChange", "1.0.0")
class DatabaseChangeV1(BaseModel):
    """Database change event from CDC."""

    operation: str  # INSERT, UPDATE, DELETE
    table_name: str
    timestamp: datetime
    before: dict[str, str] | None = None  # Old values
    after: dict[str, str] | None = None   # New values
    transaction_id: str


# Stream CDC events to Kafka with Avro
```

## Common Patterns

### Versioned Event Schemas

```python
# V1: Basic event
@manager.model("OrderEvent", "1.0.0")
class OrderEventV1(BaseModel):
    order_id: str
    status: str
    timestamp: datetime


# V2: Add customer info
@manager.model("OrderEvent", "2.0.0")
class OrderEventV2(BaseModel):
    order_id: str
    status: str
    timestamp: datetime
    customer_id: str | None = None
    customer_email: str | None = None


# Both versions coexist in Kafka
# Consumers choose which version to read
```

### Schema Testing

```python
import fastavro


def test_avro_schema_validity():
    """Test that generated schemas are valid Avro."""
    schema = manager.get_avro_schema("User", "1.0.0", namespace="com.test")

    # Should parse without errors
    parsed = fastavro.parse_schema(schema)
    assert parsed is not None


def test_avro_roundtrip():
    """Test data can be written and read back."""
    schema = manager.get_avro_schema("User", "1.0.0", namespace="com.test")
    parsed = fastavro.parse_schema(schema)

    test_data = [{"name": "Alice", "email": "alice@example.com", "age": 30}]

    output = io.BytesIO()
    fastavro.writer(output, parsed, test_data)

    output.seek(0)
    records = list(fastavro.reader(output))

    assert len(records) == 1
    assert records[0]["name"] == "Alice"
```

## Troubleshooting

### Schema Registry Errors

If schemas fail to register:

```python
# Ensure namespace follows conventions
schema = manager.get_avro_schema(
    "User",
    "1.0.0",
    namespace="com.mycompany"  # Use reverse domain notation
)

# Check schema is valid Avro
import fastavro
parsed = fastavro.parse_schema(schema)  # Should not raise
```

### Type Conversion Issues

For complex types, handle conversion explicitly:

```python
# When writing to Avro
event_data = {
    "event_id": str(event.event_id),  # UUID -> string
    "timestamp": int(event.timestamp.timestamp() * 1_000_000),  # datetime -> long
    "amount": float(event.amount),  # Decimal -> bytes (complex)
}
```

### Schema Compatibility

Check compatibility before deploying:

```python
# Use Schema Registry compatibility check
from confluent_kafka.schema_registry import SchemaRegistryClient

client = SchemaRegistryClient({"url": "http://localhost:8081"})

v1_schema = manager.get_avro_schema("User", "1.0.0", namespace="com.app")
v2_schema = manager.get_avro_schema("User", "2.0.0", namespace="com.app")

# Check if V2 is backward compatible with V1
is_compatible = client.test_compatibility(
    subject_name="user-value",
    schema=Schema(json.dumps(v2_schema), schema_type="AVRO")
)
```

## Comparison with JSON Schema

| Feature | JSON Schema | Avro |
|---------|-------------|------|
| Generation | `get_schema()` | `get_avro_schema()` |
| Export | `dump_schemas()` | `dump_avro_schemas()` |
| Transformers | ✓ Supported | ✗ Not supported |
| Modes | validation/serialization | N/A |
| Nested models | `$ref` or inline | Always inline (first use) |
| Namespaces | No | Yes (versioned) |
| Logical types | No | Yes |
| Use case | APIs, Data pipelines | Data pipelines |

## Best Practices

1. **Use consistent namespaces** - Follow your organization's naming conventions
2. **Version namespaces automatically** - Let pyrmute handle `v1_0_0` suffixes
3. **Add field descriptions** - Include documentation for data catalogs
4. **Test schema validity** - Use fastavro to validate before deploying
5. **Plan for evolution** - Make new fields optional with defaults
6. **Use logical types** - Leverage UUID, datetime, etc. for better semantics
7. **Export regularly** - Keep schema files in version control

## Next Steps

**Related topics:**

- [Schema Generation](schema-generation.md) - JSON Schema generation
- [Schema Transformers](../advanced/schema-transformers.md) - Schema evolution
      patterns

**External resources:**

- [Apache Avro Documentation](https://avro.apache.org/docs/current/)
- [Confluent Schema
      Registry](https://docs.confluent.io/platform/current/schema-registry/index.html)
- [fastavro Documentation](https://fastavro.readthedocs.io/)

**API Reference:**

- [`AvroSchemaGenerator`](../reference/avro-schema.md)
- [`ModelManager`](../reference/model-manager.md)
