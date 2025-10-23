# Protocol Buffer Schema Generation


pyrmute can generate [Protocol Buffer](https://protobuf.dev/) schemas for all
your model versions. Protocol Buffers (protobuf) are widely used for gRPC
services, microservices communication, and efficient binary serialization.
This guide covers protobuf schema generation, type mapping, and integration
with common protobuf tooling.

!!! note
    Protocol Buffers use `.proto` files to define message structures rather
    than separate schema files like JSON Schema or Avro. Throughout this
    guide, we use "schema" to refer to these `.proto` definitions for
    consistency with the rest of pyrmute's API and documentation.

## Why Protocol Buffers?

Protocol Buffers are used for:

- **gRPC services** - High-performance RPC framework
- **Microservices** - Language-agnostic service communication
- **Binary serialization** - Compact, efficient data representation
- **API definitions** - Strongly-typed service contracts

**Protocol Buffers vs JSON Schema:**

| Feature | JSON Schema | Protocol Buffers |
|---------|-------------|------------------|
| Use case | API validation | Service contracts, serialization |
| Schema evolution | Manual | Built-in backward compatibility |
| Size | Larger (text) | Compact (binary) |
| Performance | Slower | Faster |
| Ecosystem | OpenAPI, REST APIs | gRPC, microservices |
| Type safety | Runtime validation | Compile-time generation |

## Basic Protocol Buffer Schema Generation

Generate a Protocol Buffer schema for any registered model:

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


# Generate Protocol Buffer schema
proto_schema = manager.get_proto_schema("User", "1.0.0", package="com.myapp")
print(proto_schema)
```

Output:

```protobuf
syntax = "proto3";

package com.myapp;

// User account information.
message User {
  // User's full name
  string name = 1;
  // User's email address
  string email = 2;
  // User's age in years
  uint32 age = 3;
}
```

Note that the integer field was optimized to `uint32` based upon the Pydantic
constraints.

### Using the Generated Schema

The schema is returned as a string, ready to use:

```python
# Get schema as string
proto_schema = manager.get_proto_schema("User", "1.0.0", package="com.myapp")

# Write to file
from pathlib import Path
Path("user.proto").write_text(proto_schema)

# Or compile directly with protoc
import subprocess
subprocess.run(
    ["protoc", "--python_out=.", "user.proto"],
    check=True
)

# Or pass to stdin for dynamic compilation
result = subprocess.run(
    ["protoc", "--python_out=.", "--descriptor_set_out=user.desc", "-"],
    input=proto_schema.encode(),
    capture_output=True
)
```

## Type Mapping

### Basic Types

Python types are automatically mapped to Protocol Buffer types:

| Python Type | Proto2 Type | Proto3 Type |
|-------------|-------------|-------------|
| `str` | `string` | `string` |
| `int` | `int32` | `int32` |
| `float` | `double` | `double` |
| `bool` | `bool` | `bool` |
| `bytes` | `bytes` | `bytes` |

```python
@manager.model("BasicTypes", "1.0.0")
class BasicTypesV1(BaseModel):
    name: str          # -> string
    count: int         # -> int32
    price: float       # -> double
    active: bool       # -> bool
    data: bytes        # -> bytes
```

### Well-Known Types

Special Python types use Protocol Buffer well-known types:

| Python Type | Protobuf Type | Import Required |
|-------------|---------------|-----------------|
| `datetime` | `google.protobuf.Timestamp` | Yes |
| `UUID` | `string` | No |
| `Decimal` | `double` | No |

```python
from datetime import datetime
from uuid import UUID
from decimal import Decimal


@manager.model("Event", "1.0.0")
class EventV1(BaseModel):
    event_id: UUID          # -> string
    timestamp: datetime     # -> google.protobuf.Timestamp
    amount: Decimal         # -> double
```

Generated proto:

```protobuf
syntax = "proto3";

package com.myapp;

import "google/protobuf/timestamp.proto";

message Event {
  string event_id = 1;
  google.protobuf.Timestamp timestamp = 2;
  double amount = 3;
}
```

### Collection Types

Lists and maps are supported:

```python
@manager.model("Collections", "1.0.0")
class CollectionsV1(BaseModel):
    tags: list[str]           # -> repeated string
    scores: list[int]         # -> repeated int32
    metadata: dict[str, str]  # -> map<string, string>
    counts: dict[str, int]    # -> map<string, int32>
```

Generated proto:

```protobuf
message Collections {
  repeated string tags = 1;
  repeated int32 scores = 2;
  map<string, string> metadata = 3;
  map<string, int32> counts = 4;
}
```

### Optional Fields

**Proto3 behavior:**

Optional fields use the `optional` keyword for explicit presence tracking:

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str                 # -> string (required in Python, no label in proto3)
    email: str | None = None  # -> optional string (optional in Python)
    age: int | None = None    # -> optional int32 (optional in Python)
```

Generated proto3:

```protobuf
syntax = "proto3";

message User {
  string name = 1;
  optional string email = 2;
  optional int32 age = 3;
}
```

Fields without `optional` are implicitly optional but lack presence tracking
(can't distinguish between unset and default value).

**Proto2 behavior:**

Proto2 uses explicit `required` and `optional` labels:

```python
schema = manager.get_proto_schema("User", "1.0.0", use_proto3=False)
```

Generated proto2:

```protobuf
syntax = "proto2";

message User {
  required string name = 1;
  optional string email = 2;
  optional int32 age = 3;
}
```

**Presence tracking:** The `optional` keyword in proto3 enables field presence
detection, allowing you to distinguish between a field that was explicitly set
to its default value versus one that was never set.

### Enum Types

Python Enums map to Protocol Buffer enums:

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

Generated proto:

```protobuf
message Task {
  enum Status {
    PENDING = 0;
    ACTIVE = 1;
    COMPLETED = 2;
  }

  string name = 1;
  Status status = 2;
}
```

### Union Types

Union types become `oneof` in Protocol Buffers:

```python
@manager.model("Notification", "1.0.0")
class NotificationV1(BaseModel):
    notification_id: str
    content: str | int  # Union type
```

Generated proto:

```protobuf
message Notification {
  string notification_id = 1;
  oneof content_value {
    // content when type is str
    string content_string = 2;
    // content when type is int
    int32 content_int32 = 3;
  }
}
```

Optional unions:

```python
@manager.model("Flexible", "1.0.0")
class FlexibleV1(BaseModel):
    value: str | int | None  # Optional union
```

Generated proto:
```protobuf
message Flexible {
  oneof value_value {
    // value when type is str
    string value_string = 1;
    // value when type is int
    int32 value_int32 = 2;
  }
}
```

### Nested Messages

Pydantic models become nested messages:

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

Generated proto:
```protobuf
message User {
  message AddressV1 {
    string street = 1;
    string city = 2;
    string zip_code = 3;
  }

  string name = 1;
  AddressV1 address = 2;
}
```

## Protocol Buffer Packages

Protocol Buffers use packages to organize schemas, similar to namespaces in
other languages. Pyrmute uses a consistent package name across all versions.

```python
# Same package for all versions
schema_v1 = manager.get_proto_schema("User", "1.0.0", package="com.mycompany")
# package: "com.mycompany"

schema_v2 = manager.get_proto_schema("User", "2.0.0", package="com.mycompany")
# package: "com.mycompany"
```

**Best practices:**

- Use reverse domain notation: `com.company.service`
- Keep packages consistent across versions
- Use subpackages for logical grouping
- Examples: `com.acme.users`, `com.acme.orders`, `org.example.api`

**Note:** Unlike some other systems, protobuf packages should NOT include
version numbers. Versioning is handled through message names or file
organization.

## Proto2 vs Proto3

Choose the Protocol Buffer syntax version:

```python
# Proto3 (recommended for new projects)
schema = manager.get_proto_schema(
    "User", "1.0.0",
    package="com.myapp",
    use_proto3=True  # Default
)

# Proto2 (for legacy systems)
schema = manager.get_proto_schema(
    "User", "1.0.0",
    package="com.myapp",
    use_proto3=False
)
```

**Key differences:**

| Feature | Proto2 | Proto3 |
|---------|--------|--------|
| Required fields | Supported | Not supported |
| Default values | Custom defaults | Type defaults (0, "", false) |
| Presence tracking | Optional | Limited (use `optional` keyword) |
| Unknown fields | Preserved | Preserved |
| Recommendation | Legacy only | Modern projects |

**When to use proto2:**

- Maintaining existing proto2 services
- Need explicit `required` fields
- Custom default values required

**When to use proto3:**

- New projects (recommended)
- Simpler syntax
- Better forward compatibility

## Exporting Protocol Buffer Schemas

### Export All Schemas

Export protobuf schemas for all registered models:

```python
manager.dump_proto_schemas(
    "schemas/protos/",
    package="com.mycompany"
)
```

Creates files like:
```
schemas/protos/
├── User_v1_0_0.proto
├── User_v2_0_0.proto
├── Order_v1_0_0.proto
└── Product_v1_0_0.proto
```

### Export Options

Customize the export:

```python
manager.dump_proto_schemas(
    "schemas/protos/",
    package="com.mycompany.events",
    include_comments=True,   # Include field descriptions
    use_proto3=True          # Use proto3 syntax
)
```

### Export Without Documentation

For production schemas without documentation overhead:

```python
manager.dump_proto_schemas(
    "schemas/protos/",
    package="com.mycompany",
    include_comments=False  # Omit comments
)
```

## gRPC Integration

### Define a Service

```python
from pydantic import BaseModel


@manager.model("GetUserRequest", "1.0.0")
class GetUserRequestV1(BaseModel):
    user_id: str


@manager.model("GetUserResponse", "1.0.0")
class GetUserResponseV1(BaseModel):
    user_id: str
    name: str
    email: str


# Export schemas
proto_schemas = manager.dump_proto_schemas(
    "protos/", package="com.myapp.users"
)

# Or get individual schema as string
request_schema = manager.get_proto_schema(
    "GetUserRequest", "1.0.0", package="com.myapp.users"
)

# Write to file
Path("protos/user_service.proto").write_text(request_schema)
```

Manually add service definition to the generated proto:

```protobuf
syntax = "proto3";

package com.myapp.users;

message GetUserRequest {
  string user_id = 1;
}

message GetUserResponse {
  string user_id = 1;
  string name = 2;
  string email = 3;
}

// Add this service definition
service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
}
```

### Compile Protocol Buffers

Use `protoc` to generate code:

```bash
# Generate Go code
protoc --go_out=. --go-grpc_out=. protos/*.proto

# Generate Java code
protoc --java_out=. protos/*.proto
```

### Use Generated Code

```go
package main

import (
    "context"
    "log"

    pb "github.com/myapp/protos"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
)

func main() {
    // Connect to gRPC server
    conn, err := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()

    // Create client
    client := pb.NewUserServiceClient(conn)

    // Create request
    request := &pb.GetUserRequest{
        UserId: "123",
    }

    // Call service
    response, err := client.GetUser(context.Background(), request)
    if err != nil {
        log.Fatalf("GetUser failed: %v", err)
    }

    log.Printf("User: %s (%s)", response.Name, response.Email)
}
```

## Language Interoperability

Protocol Buffers work across languages:

### Python to Go

**Python (producer):**
```python
@manager.model("Event", "1.0.0")
class EventV1(BaseModel):
    event_id: str
    timestamp: datetime
    data: dict[str, str]

# Get schema as string
proto_schema = manager.get_proto_schema("Event", "1.0.0", package="com.events")

# Write to file for Go to compile
Path("protos/event.proto").write_text(proto_schema)
```

**Go (consumer):**
```go
// Generate Go code
// protoc --go_out=. protos/Event_v1_0_0.proto

import pb "github.com/myapp/protos"

event := &pb.Event{
    EventId: "evt_123",
    Timestamp: timestamppb.Now(),
    Data: map[string]string{"key": "value"},
}
```

### Python to Java

**Python:**
```python
manager.dump_proto_schemas("protos/", package="com.myapp")
```

**Java:**
```bash
# Generate Java code
protoc --java_out=src/main/java protos/*.proto
```

```java
import com.myapp.EventProtos.Event;

Event event = Event.newBuilder()
    .setEventId("evt_123")
    .setTimestamp(Timestamp.newBuilder().setSeconds(System.currentTimeMillis() / 1000))
    .putData("key", "value")
    .build();
```

## Schema Evolution Best Practices

### Backward Compatible Changes

Add new fields with field numbers that don't conflict:

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
```

Generated proto (v2):
```protobuf
message Product {
  string id = 1;
  string name = 2;
  double price = 3;
  optional string description = 4;  // New field
  optional string category = 5;     // New field
}
```

Old clients can read new messages (ignore unknown fields).

### Reserved Fields

When removing fields, reserve their numbers:

```python
# Don't do this - reuses field number
@manager.model("Product", "2.0.0")
class ProductV2(BaseModel):
    id: str
    name: str
    new_field: str  # Don't reuse the field number!
```

Instead, document reserved numbers in comments:

```protobuf
message Product {
  // reserved 3;  // Was: price (removed in v2)
  string id = 1;
  string name = 2;
  string new_field = 4;
}
```

### Field Numbering

**Best practices:**

- Never change field numbers
- Don't reuse field numbers
- Reserve 1-15 for frequently used fields (1-byte encoding)
- Use 16+ for less frequent fields

## Real-World Examples

### Microservice API

```python
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class OrderStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"


@manager.model("CreateOrderRequest", "1.0.0")
class CreateOrderRequestV1(BaseModel):
    """Request to create a new order."""
    customer_id: str
    items: list[str]
    total_amount: float


@manager.model("CreateOrderResponse", "1.0.0")
class CreateOrderResponseV1(BaseModel):
    """Response with created order details."""
    order_id: UUID
    status: OrderStatus
    created_at: datetime


# Export for gRPC service
manager.dump_proto_schemas("protos/", package="com.shop.orders")
```

### Event Streaming

```python
@manager.model("UserEvent", "1.0.0")
class UserEventV1(BaseModel):
    """User event for event streaming."""
    event_id: UUID
    user_id: str
    event_type: str
    timestamp: datetime
    payload: dict[str, str]
    metadata: dict[str, str] = Field(default_factory=dict)


# Export for Kafka with protobuf serialization
proto_schema = manager.get_proto_schema(
    "UserEvent", "1.0.0",
    package="com.events.users"
)

# Write to file for compilation
Path("user_event.proto").write_text(proto_schema)
```

### API Gateway

```python
@manager.model("ApiRequest", "1.0.0")
class ApiRequestV1(BaseModel):
    """Generic API request wrapper."""
    request_id: UUID
    timestamp: datetime
    endpoint: str
    method: str
    headers: dict[str, str]
    body: bytes | None = None


@manager.model("ApiResponse", "1.0.0")
class ApiResponseV1(BaseModel):
    """Generic API response wrapper."""
    request_id: UUID
    status_code: int
    headers: dict[str, str]
    body: bytes
    duration_ms: int
```

## Common Patterns

### Versioned Messages

```python
# V1: Basic order
@manager.model("Order", "1.0.0")
class OrderV1(BaseModel):
    order_id: str
    total: float


# V2: Add customer info
@manager.model("Order", "2.0.0")
class OrderV2(BaseModel):
    order_id: str
    total: float
    customer_id: str | None = None
    customer_email: str | None = None


# Both versions coexist
# Clients choose which version to use
```

### Polymorphic Messages

Use `oneof` for polymorphic data:

```python
@manager.model("Notification", "1.0.0")
class NotificationV1(BaseModel):
    notification_id: str
    timestamp: datetime
    # Use union for polymorphic content
    content: str | dict[str, str]  # Text or structured
```

### Pagination

```python
@manager.model("ListUsersRequest", "1.0.0")
class ListUsersRequestV1(BaseModel):
    page_size: int = 50
    page_token: str | None = None


@manager.model("ListUsersResponse", "1.0.0")
class ListUsersResponseV1(BaseModel):
    users: list[dict[str, str]]  # Simplified for example
    next_page_token: str | None = None
    total_count: int
```

## Schema Testing

### Validate Generated Schemas

```python
import subprocess


def test_proto_schema_validity() -> None:
    """Test that generated schemas are valid protobuf."""
    manager.dump_proto_schemas("test_protos/", package="com.test")

    # Validate with protoc
    result = subprocess.run(
        ["protoc", "--syntax_only", "test_protos/*.proto"],
        capture_output=True
    )

    assert result.returncode == 0, f"Invalid proto: {result.stderr}"


def test_proto_compilation() -> None:
    """Test that schemas can be compiled."""
    manager.dump_proto_schemas("test_protos/", package="com.test")

    # Compile to Python
    result = subprocess.run(
        ["protoc", "--python_out=.", "test_protos/*.proto"],
        capture_output=True
    )

    assert result.returncode == 0
```

### Test Serialization

```python
def test_protobuf_roundtrip() -> None:
    """Test data can be serialized and deserialized."""
    # Generate schema
    manager.dump_proto_schemas("test_protos/", package="com.test")

    # Compile
    subprocess.run(["protoc", "--python_out=.", "test_protos/User_v1_0_0.proto"])

    # Import generated code
    from test_protos import user_v1_0_0_pb2

    # Create message
    user = user_v1_0_0_pb2.User(
        name="Alice",
        email="alice@example.com",
        age=30
    )

    # Serialize
    serialized = user.SerializeToString()

    # Deserialize
    user2 = user_v1_0_0_pb2.User()
    user2.ParseFromString(serialized)

    assert user2.name == "Alice"
```

## Troubleshooting

### Import Errors

If well-known types aren't found:

```python
# Ensure google protobuf is installed
# pip install protobuf

# When compiling, include proto path
# protoc -I=/usr/include --python_out=. protos/*.proto
```

### Field Number Conflicts

Avoid reusing field numbers:

```python
# Bad: Reusing field number 3
@manager.model("Product", "2.0.0")
class ProductV2(BaseModel):
    id: str        # field 1
    name: str      # field 2
    category: str  # field 3 (was price in v1) - BAD!
```

### Compilation Errors

Check syntax version matches usage:

```bash
# If using proto3 features in proto2 file
protoc: syntax error

# Solution: Use consistent syntax
manager.get_proto_schema("Model", "1.0.0", use_proto3=True)
```

## Comparison with JSON Schema

| Feature | JSON Schema | Protocol Buffers |
|---------|-------------|------------------|
| Generation | `get_schema()` | `get_proto_schema()` |
| Export | `dump_schemas()` | `dump_proto_schemas()` |
| Syntax | JSON | Protobuf DSL |
| Code generation | No | Yes (protoc) |
| Binary format | No | Yes |
| Service definitions | No | Yes (with manual editing) |
| Use case | REST APIs | gRPC, microservices |

## Best Practices

1. **Use proto3 for new projects** - Modern syntax with better compatibility
2. **Keep packages consistent** - Don't version package names
3. **Add field descriptions** - Enable `include_comments=True`
4. **Never reuse field numbers** - Reserve removed field numbers
5. **Test compilation** - Validate schemas with `protoc`
6. **Version control schemas** - Keep `.proto` files in Git
7. **Document service contracts** - Add comments to generated files
8. **Use well-known types** - Leverage `google.protobuf.Timestamp`, etc.

## Next Steps

**Related topics:**

- [Schema Generation](schema-generation.md) - JSON Schema generation
- [Avro Schema Generation](avro-schemas.md) - Apache Avro schemas

**External resources:**

- [Protocol Buffers Documentation](https://protobuf.dev/)
- [gRPC Documentation](https://grpc.io/docs/)
- [Protocol Buffers Style Guide](https://protobuf.dev/programming-guides/style/)

**API Reference:**

- [`ProtoSchemaGenerator`](../reference/protobuf-schema.md)
- [`ModelManager`](../reference/model-manager.md)
