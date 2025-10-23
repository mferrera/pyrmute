"""Integration tests with real Avro libraries."""

import io
import json
import time
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

import avro  # type: ignore[import-untyped]
import avro.schema  # type: ignore[import-untyped]
import fastavro
from pydantic import BaseModel, Field

from pyrmute import ModelManager

# ruff: noqa: PLR2004


def test_avro_schema_validates_with_fastavro(manager: ModelManager) -> None:
    """Test that generated schemas are valid according to fastavro."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        id: UUID
        name: str
        email: str
        age: int
        created_at: datetime

    schema = manager.get_avro_schema("User", "1.0.0", namespace="com.test")

    parsed_schema = fastavro.parse_schema(dict(schema))
    assert parsed_schema is not None

    test_data = [
        {
            "id": str(uuid4()),
            "name": "Alice",
            "email": "alice@example.com",
            "age": 30,
            "created_at": int(datetime.now().timestamp() * 1_000_000),
        }
    ]

    output = io.BytesIO()
    fastavro.writer(output, parsed_schema, test_data)

    output.seek(0)
    records = list(fastavro.reader(output))
    assert len(records) == 1
    assert records[0]["name"] == "Alice"  # type: ignore[index,call-overload]


def test_avro_schema_with_nested_records_fastavro(manager: ModelManager) -> None:
    """Test nested record schemas work with fastavro."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: str
        zip_code: str

    @manager.model("Person", "1.0.0")
    class PersonV1(BaseModel):
        name: str
        address: AddressV1

    schema = manager.get_avro_schema("Person", "1.0.0", namespace="com.test")
    parsed_schema = fastavro.parse_schema(dict(schema))

    test_data = [
        {
            "name": "Bob",
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
                "zip_code": "12345",
            },
        }
    ]

    output = io.BytesIO()
    fastavro.writer(output, parsed_schema, test_data)

    output.seek(0)
    records = list(fastavro.reader(output))
    assert records[0]["address"]["city"] == "Springfield"  # type: ignore[index,call-overload]


def test_avro_schema_with_enums_fastavro(manager: ModelManager) -> None:
    """Test enum schemas work correctly with fastavro."""

    class Status(str, Enum):
        PENDING = "pending"
        ACTIVE = "active"
        COMPLETED = "completed"

    @manager.model("Task", "1.0.0")
    class TaskV1(BaseModel):
        name: str
        status: Status

    schema = manager.get_avro_schema("Task", "1.0.0", namespace="com.test")
    parsed_schema = fastavro.parse_schema(dict(schema))

    test_data = [
        {"name": "Task 1", "status": "pending"},
        {"name": "Task 2", "status": "active"},
    ]

    output = io.BytesIO()
    fastavro.writer(output, parsed_schema, test_data)

    output.seek(0)
    records = list(fastavro.reader(output))
    assert len(records) == 2
    assert records[0]["status"] == "pending"  # type: ignore[index,call-overload]
    assert records[1]["status"] == "active"  # type: ignore[index,call-overload]


def test_avro_schema_with_unions_fastavro(manager: ModelManager) -> None:
    """Test union types work correctly with fastavro."""

    @manager.model("Flexible", "1.0.0")
    class FlexibleV1(BaseModel):
        name: str
        value: str | int | None = None

    schema = manager.get_avro_schema("Flexible", "1.0.0", namespace="com.test")
    parsed_schema = fastavro.parse_schema(dict(schema))

    test_data = [
        {"name": "Item1", "value": None},
        {"name": "Item2", "value": "text"},
        {"name": "Item3", "value": 42},
    ]

    output = io.BytesIO()
    fastavro.writer(output, parsed_schema, test_data)

    output.seek(0)
    records = list(fastavro.reader(output))
    assert records[0]["value"] is None  # type: ignore[index,call-overload]
    assert records[1]["value"] == "text"  # type: ignore[index,call-overload]
    assert records[2]["value"] == 42  # type: ignore[index,call-overload]


def test_avro_schema_with_arrays_and_maps_fastavro(manager: ModelManager) -> None:
    """Test array and map types work with fastavro."""

    @manager.model("Config", "1.0.0")
    class ConfigV1(BaseModel):
        tags: list[str]
        settings: dict[str, str]

    schema = manager.get_avro_schema("Config", "1.0.0", namespace="com.test")
    parsed_schema = fastavro.parse_schema(dict(schema))

    test_data = [
        {
            "tags": ["tag1", "tag2", "tag3"],
            "settings": {"key1": "value1", "key2": "value2"},
        }
    ]

    output = io.BytesIO()
    fastavro.writer(output, parsed_schema, test_data)

    output.seek(0)
    records = list(fastavro.reader(output))
    assert len(records[0]["tags"]) == 3  # type: ignore[index,call-overload]
    assert records[0]["settings"]["key1"] == "value1"  # type: ignore[index,call-overload]


def test_avro_schema_logical_types_roundtrip_fastavro(manager: ModelManager) -> None:
    """Test logical types (UUID, datetime) serialize/deserialize correctly."""

    @manager.model("Event", "1.0.0")
    class EventV1(BaseModel):
        event_id: UUID
        timestamp: datetime
        event_date: date

    schema = manager.get_avro_schema("Event", "1.0.0", namespace="com.test")
    parsed_schema = fastavro.parse_schema(dict(schema))

    test_id = uuid4()
    test_time = datetime(2023, 6, 15, 14, 30, 45)
    test_date = date(2023, 6, 15)

    test_data = [
        {
            "event_id": str(test_id),
            "timestamp": int(test_time.timestamp() * 1_000_000),
            "event_date": (test_date - date(1970, 1, 1)).days,
        }
    ]

    output = io.BytesIO()
    fastavro.writer(output, parsed_schema, test_data)

    output.seek(0)
    records = list(fastavro.reader(output))
    assert records[0]["event_id"] == test_id  # type: ignore[index,call-overload]


def test_avro_schema_validates_with_avro_python(manager: ModelManager) -> None:
    """Test that generated schemas are valid according to official avro-python."""

    @manager.model("Product", "1.0.0")
    class ProductV1(BaseModel):
        name: str
        price: float
        in_stock: bool

    schema = manager.get_avro_schema("Product", "1.0.0", namespace="com.test")

    # Official avro library should parse the schema
    parsed_schema = avro.schema.parse(json.dumps(schema))
    assert parsed_schema is not None
    assert parsed_schema.type == "record"
    assert parsed_schema.name == "Product"


def test_avro_schema_complex_nested_validates(manager: ModelManager) -> None:
    """Test complex nested structures validate with official avro library."""

    @manager.model("Item", "1.0.0")
    class ItemV1(BaseModel):
        name: str
        quantity: int

    @manager.model("Order", "1.0.0")
    class OrderV1(BaseModel):
        order_id: str
        items: list[ItemV1]
        metadata: dict[str, str]

    schema = manager.get_avro_schema("Order", "1.0.0", namespace="com.test")
    parsed_schema = avro.schema.parse(json.dumps(schema))

    assert parsed_schema is not None
    assert len(parsed_schema.fields) == 3


# ============================================================================
# Kafka Schema Registry Integration Tests
# ============================================================================


def test_avro_schema_kafka_schema_registry_compatible(manager: ModelManager) -> None:
    """Test schema format is compatible with Kafka Schema Registry."""

    class EventType(str, Enum):
        USER_CREATED = "user_created"
        USER_UPDATED = "user_updated"

    @manager.model("UserEvent", "1.0.0")
    class UserEventV1(BaseModel):
        event_id: UUID
        event_type: EventType
        user_id: str
        timestamp: datetime

    schema = manager.get_avro_schema("UserEvent", "1.0.0", namespace="com.events")

    fastavro.parse_schema(dict(schema))
    assert schema["namespace"] == "com.events"

    schema_json = json.dumps(schema)
    assert len(schema_json) > 0

    reparsed = json.loads(schema_json)
    assert reparsed == schema


def test_avro_schema_multiple_versions_schema_registry(manager: ModelManager) -> None:
    """Test multiple versions can coexist in Schema Registry format."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        email: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str
        phone: str | None = None

    schema_v1 = manager.get_avro_schema(
        "User", "1.0.0", namespace="com.app", versioned_namespace=True
    )
    schema_v2 = manager.get_avro_schema(
        "User", "2.0.0", namespace="com.app", versioned_namespace=True
    )

    parsed_v1 = fastavro.parse_schema(dict(schema_v1))
    fastavro.parse_schema(dict(schema_v2))

    assert schema_v1["namespace"] != schema_v2["namespace"]
    assert "v1_0_0" in schema_v1["namespace"]
    assert "v2_0_0" in schema_v2["namespace"]

    v1_data = [{"name": "Alice", "email": "alice@example.com"}]

    output_v1 = io.BytesIO()
    fastavro.writer(output_v1, parsed_v1, v1_data)


def test_avro_schema_pydantic_to_avro_to_pydantic_roundtrip(
    manager: ModelManager,
) -> None:
    """Test complete roundtrip: Pydantic -> Avro -> Pydantic."""

    @manager.model("Order", "1.0.0")
    class OrderV1(BaseModel):
        order_id: UUID
        customer_name: str
        total: float
        items: list[str]

    original_order = OrderV1(
        order_id=uuid4(),
        customer_name="Bob Smith",
        total=99.99,
        items=["item1", "item2"],
    )

    schema = manager.get_avro_schema("Order", "1.0.0", namespace="com.orders")
    parsed_schema = fastavro.parse_schema(dict(schema))

    avro_data = {
        "order_id": original_order.order_id,
        "customer_name": original_order.customer_name,
        "total": original_order.total,
        "items": original_order.items,
    }

    output = io.BytesIO()
    fastavro.writer(output, parsed_schema, [avro_data])

    output.seek(0)
    records = list(fastavro.reader(output))

    reconstructed_order = OrderV1(
        order_id=records[0]["order_id"],  # type: ignore[index,call-overload,arg-type]
        customer_name=records[0]["customer_name"],  # type: ignore[index,call-overload]
        total=records[0]["total"],  # type: ignore[index,call-overload,arg-type]
        items=records[0]["items"],  # type: ignore[index,call-overload,arg-type]
    )

    assert reconstructed_order.order_id == original_order.order_id
    assert reconstructed_order.customer_name == original_order.customer_name
    assert reconstructed_order.total == original_order.total
    assert reconstructed_order.items == original_order.items


def test_avro_schema_batch_processing_integration(manager: ModelManager) -> None:
    """Test batch processing scenario with Avro serialization."""

    @manager.model("LogEntry", "1.0.0")
    class LogEntryV1(BaseModel):
        timestamp: datetime
        level: str
        message: str
        metadata: dict[str, str] = Field(default_factory=dict)

    schema = manager.get_avro_schema("LogEntry", "1.0.0", namespace="com.logs")
    parsed_schema = fastavro.parse_schema(dict(schema))

    log_entries = []
    for i in range(100):
        entry = LogEntryV1(
            timestamp=datetime.now(),
            level="INFO",
            message=f"Log message {i}",
            metadata={"source": "test", "batch": "1"},
        )
        log_entries.append(
            {
                "timestamp": int(entry.timestamp.timestamp() * 1_000_000),
                "level": entry.level,
                "message": entry.message,
                "metadata": entry.metadata,
            }
        )

    output = io.BytesIO()
    fastavro.writer(output, parsed_schema, log_entries)

    output.seek(0)
    records = list(fastavro.reader(output))

    assert len(records) == 100
    assert all(r["level"] == "INFO" for r in records)  # type: ignore[index,call-overload]


def test_avro_schema_schema_evolution_in_pipeline(manager: ModelManager) -> None:
    """Test schema evolution scenario in a data pipeline."""

    # Version 1: Original schema
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
        category: str | None = None
        in_stock: bool = True

    schema_v1 = manager.get_avro_schema("Product", "1.0.0", namespace="com.catalog")
    schema_v2 = manager.get_avro_schema("Product", "2.0.0", namespace="com.catalog")

    parsed_v1 = fastavro.parse_schema(dict(schema_v1))
    parsed_v2 = fastavro.parse_schema(dict(schema_v2))

    v1_data = [
        {"id": "P001", "name": "Widget", "price": 9.99},
        {"id": "P002", "name": "Gadget", "price": 19.99},
    ]

    output_v1 = io.BytesIO()
    fastavro.writer(output_v1, parsed_v1, v1_data)

    v2_data = [
        {
            "id": "P003",
            "name": "Gizmo",
            "price": 29.99,
            "category": "electronics",
            "in_stock": True,
        }
    ]

    output_v2 = io.BytesIO()
    fastavro.writer(output_v2, parsed_v2, v2_data)

    output_v1.seek(0)
    v1_records = list(fastavro.reader(output_v1))
    assert len(v1_records) == 2

    output_v2.seek(0)
    v2_records = list(fastavro.reader(output_v2))
    assert len(v2_records) == 1
    assert v2_records[0]["category"] == "electronics"  # type: ignore[index,call-overload]


def test_avro_schema_file_export_and_reload(
    manager: ModelManager, tmp_path: Path
) -> None:
    """Test exporting schemas to files and using them in external tools."""

    @manager.model("Event", "1.0.0")
    class EventV1(BaseModel):
        event_id: str
        event_type: str
        timestamp: datetime

    output_dir = tmp_path / "schemas"
    manager.dump_avro_schemas(output_dir, namespace="com.events")

    schema_file = output_dir / "Event_v1_0_0.avsc"
    assert schema_file.exists()

    with open(schema_file) as f:
        schema = json.load(f)

    parsed_schema = fastavro.parse_schema(schema)

    test_data = [
        {
            "event_id": "E001",
            "event_type": "user_login",
            "timestamp": int(datetime.now().timestamp() * 1_000_000),
        }
    ]

    output = io.BytesIO()
    fastavro.writer(output, parsed_schema, test_data)

    output.seek(0)
    records = list(fastavro.reader(output))
    assert records[0]["event_type"] == "user_login"  # type: ignore[index,call-overload]


def test_avro_schema_large_schema_generation_performance(
    manager: ModelManager,
) -> None:
    """Test performance with large complex schemas."""

    @manager.model("LargeModel", "1.0.0")
    class LargeModelV1(BaseModel):
        """Model with many fields for performance testing."""

        field_01: str
        field_02: int
        field_03: float
        field_04: bool
        field_05: datetime
        field_06: UUID
        field_07: list[str]
        field_08: dict[str, int]
        field_09: str | None = None
        field_10: int | None = None
        field_11: list[float] = Field(default_factory=list)
        field_12: dict[str, str] = Field(default_factory=dict)
        field_13: date
        field_14: None
        field_15: bytes
        field_16: Decimal
        field_17: str
        field_18: int
        field_19: float
        field_20: bool

    start = time.time()
    schema = manager.get_avro_schema("LargeModel", "1.0.0", namespace="com.test")
    duration = time.time() - start

    assert duration < 1.0
    assert len(schema["fields"]) == 20


def test_avro_schema_many_models_export_performance(
    manager: ModelManager, tmp_path: Path
) -> None:
    """Test exporting many models performs well."""
    for i in range(50):

        @manager.model(f"Model{i}", "1.0.0")
        class DynamicModel(BaseModel):
            name: str
            value: int

    start = time.time()
    output_dir = tmp_path / "many_schemas"
    manager.dump_avro_schemas(output_dir, namespace="com.test")
    duration = time.time() - start

    assert duration < 1.0  # Should complete in reasonable time
    assert len(list(output_dir.glob("*.avsc"))) == 50
