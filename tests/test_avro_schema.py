"""Tests for Avro schema generation."""

import json
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import pytest
from pydantic import AnyHttpUrl, BaseModel, Field

from pyrmute import ModelManager, ModelNotFoundError
from pyrmute.avro_schema import AvroExporter, AvroSchemaGenerator

# ruff: noqa: PLR2004
# mypy: disable-error-code="index,call-overload,typeddict-item"


@pytest.fixture
def avro_generator() -> AvroSchemaGenerator:
    """Create a fresh AvroSchemaGenerator instance for each test."""
    return AvroSchemaGenerator(namespace="com.test")


# ============================================================================
# Basic Type Tests
# ============================================================================


def test_avro_schema_basic_types(manager: ModelManager) -> None:
    """Test Avro schema generation for basic Python types."""

    @manager.model("BasicTypes", "1.0.0")
    class BasicTypesV1(BaseModel):
        name: str
        age: int
        height: float
        is_active: bool
        data: bytes

    schema = manager.get_avro_schema("BasicTypes", "1.0.0", namespace="com.test")

    assert schema["type"] == "record"
    assert schema["name"] == "BasicTypes"
    assert schema["namespace"] == "com.test"
    assert len(schema["fields"]) == 5

    field_types = {f["name"]: f["type"] for f in schema["fields"]}
    assert field_types["name"] == "string"
    assert field_types["age"] == "int"
    assert field_types["height"] == "double"
    assert field_types["is_active"] == "boolean"
    assert field_types["data"] == "bytes"


def test_avro_schema_optional_fields(manager: ModelManager) -> None:
    """Test Avro schema generation for optional fields."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        email: str | None = None
        age: int | None = None

    schema = manager.get_avro_schema("User", "1.0.0", namespace="com.test")

    fields = {f["name"]: f for f in schema["fields"]}

    assert fields["name"]["type"] == "string"
    assert "default" not in fields["name"]

    assert fields["email"]["type"] == ["null", "string"]
    assert fields["email"]["default"] is None

    assert fields["age"]["type"] == ["null", "int"]
    assert fields["age"]["default"] is None


def test_avro_schema_with_defaults(manager: ModelManager) -> None:
    """Test Avro schema generation for fields with default values."""

    @manager.model("Config", "1.0.0")
    class ConfigV1(BaseModel):
        timeout: int = 30
        retry_count: int = 3
        enabled: bool = True
        name: str = "default"

    schema = manager.get_avro_schema("Config", "1.0.0", namespace="com.test")

    fields = {f["name"]: f for f in schema["fields"]}

    assert fields["timeout"]["default"] == 30
    assert fields["retry_count"]["default"] == 3
    assert fields["enabled"]["default"] is True
    assert fields["name"]["default"] == "default"


# ============================================================================
# Documentation Tests
# ============================================================================


def test_avro_schema_with_documentation(manager: ModelManager) -> None:
    """Test Avro schema includes model and field documentation."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        """User account information."""

        name: str = Field(description="User's full name")
        age: int = Field(description="User's age in years")
        email: str = Field(description="Email address")

    schema = manager.get_avro_schema("User", "1.0.0", namespace="com.test")

    assert schema["doc"] == "User account information."

    fields = {f["name"]: f for f in schema["fields"]}
    assert fields["name"]["doc"] == "User's full name"
    assert fields["age"]["doc"] == "User's age in years"
    assert fields["email"]["doc"] == "Email address"


def test_avro_schema_without_documentation(manager: ModelManager) -> None:
    """Test Avro schema generation without documentation."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        """User model."""

        name: str = Field(description="User name")

    schema = manager.get_avro_schema(
        "User", "1.0.0", namespace="com.test", include_docs=False
    )

    assert "doc" not in schema
    fields = {f["name"]: f for f in schema["fields"]}
    assert "doc" not in fields["name"]


# ============================================================================
# Logical Type Tests
# ============================================================================


def test_avro_schema_datetime_logical_type(manager: ModelManager) -> None:
    """Test Avro schema uses logical types for datetime."""

    @manager.model("Event", "1.0.0")
    class EventV1(BaseModel):
        created_at: datetime
        scheduled_date: date
        scheduled_time: time

    schema = manager.get_avro_schema("Event", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    assert fields["created_at"]["type"] == {
        "type": "long",
        "logicalType": "timestamp-micros",
    }
    assert fields["scheduled_date"]["type"] == {"type": "int", "logicalType": "date"}
    assert fields["scheduled_time"]["type"] == {
        "type": "long",
        "logicalType": "time-micros",
    }


def test_avro_schema_uuid_logical_type(manager: ModelManager) -> None:
    """Test Avro schema uses logical type for UUID."""

    @manager.model("Resource", "1.0.0")
    class ResourceV1(BaseModel):
        id: UUID

    schema = manager.get_avro_schema("Resource", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    assert fields["id"]["type"] == {"type": "string", "logicalType": "uuid"}


def test_avro_schema_decimal_logical_type(manager: ModelManager) -> None:
    """Test Avro schema uses logical type for Decimal."""

    @manager.model("Price", "1.0.0")
    class PriceV1(BaseModel):
        amount: Decimal

    schema = manager.get_avro_schema("Price", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    decimal_type = fields["amount"]["type"]
    assert decimal_type["type"] == "bytes"
    assert decimal_type["logicalType"] == "decimal"
    assert decimal_type["precision"] == 10
    assert decimal_type["scale"] == 2


def test_avro_schema_optional_datetime(manager: ModelManager) -> None:
    """Test Avro schema for optional datetime fields."""

    @manager.model("Event", "1.0.0")
    class EventV1(BaseModel):
        name: str
        timestamp: datetime | None = None

    schema = manager.get_avro_schema("Event", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    timestamp_type = fields["timestamp"]["type"]
    assert isinstance(timestamp_type, list)
    assert "null" in timestamp_type
    assert {"type": "long", "logicalType": "timestamp-micros"} in timestamp_type
    assert fields["timestamp"]["default"] is None


# ============================================================================
# Enum Tests
# ============================================================================


def test_avro_schema_enum_type(manager: ModelManager) -> None:
    """Test Avro schema converts Python Enum to Avro enum."""

    class Status(str, Enum):
        PENDING = "pending"
        ACTIVE = "active"
        COMPLETED = "completed"

    @manager.model("Task", "1.0.0")
    class TaskV1(BaseModel):
        status: Status

    schema = manager.get_avro_schema("Task", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    status_type = fields["status"]["type"]
    assert status_type["type"] == "enum"
    assert status_type["name"] == "Status"
    assert status_type["symbols"] == ["pending", "active", "completed"]


def test_avro_schema_optional_enum(manager: ModelManager) -> None:
    """Test Avro schema for optional enum fields."""

    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    @manager.model("Task", "1.0.0")
    class TaskV1(BaseModel):
        name: str
        priority: Priority | None = None

    schema = manager.get_avro_schema("Task", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    priority_type = fields["priority"]["type"]
    assert isinstance(priority_type, list)
    assert "null" in priority_type

    # Find the enum in the union
    enum_type = next(
        t for t in priority_type if isinstance(t, dict) and t.get("type") == "enum"
    )
    assert enum_type["name"] == "Priority"
    assert enum_type["symbols"] == ["low", "medium", "high"]


def test_avro_schema_enum_with_default(manager: ModelManager) -> None:
    """Test Avro schema for enum fields with default values."""

    class Status(str, Enum):
        DRAFT = "draft"
        PUBLISHED = "published"

    @manager.model("Article", "1.0.0")
    class ArticleV1(BaseModel):
        title: str
        status: Status = Status.DRAFT

    schema = manager.get_avro_schema("Article", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    assert fields["status"]["default"] == "draft"


# ============================================================================
# Collection Type Tests
# ============================================================================


def test_avro_schema_list_type(manager: ModelManager) -> None:
    """Test Avro schema generation for list fields."""

    @manager.model("Order", "1.0.0")
    class OrderV1(BaseModel):
        items: list[str]
        quantities: list[int]
        prices: list[float]

    schema = manager.get_avro_schema("Order", "1.0.0", namespace="com.test")

    fields = {f["name"]: f for f in schema["fields"]}

    assert fields["items"]["type"] == {"type": "array", "items": "string"}
    assert fields["quantities"]["type"] == {"type": "array", "items": "int"}
    assert fields["prices"]["type"] == {"type": "array", "items": "double"}


def test_avro_schema_dict_type(manager: ModelManager) -> None:
    """Test Avro schema generation for dict fields."""

    @manager.model("Config", "1.0.0")
    class ConfigV1(BaseModel):
        settings: dict[str, str]
        counts: dict[str, int]
        flags: dict[str, bool]

    schema = manager.get_avro_schema("Config", "1.0.0", namespace="com.test")

    fields = {f["name"]: f for f in schema["fields"]}

    assert fields["settings"]["type"] == {"type": "map", "values": "string"}
    assert fields["counts"]["type"] == {"type": "map", "values": "int"}
    assert fields["flags"]["type"] == {"type": "map", "values": "boolean"}


def test_avro_schema_list_of_optional_items(manager: ModelManager) -> None:
    """Test Avro schema for list containing optional types."""

    @manager.model("Data", "1.0.0")
    class DataV1(BaseModel):
        values: list[str | None]

    schema = manager.get_avro_schema("Data", "1.0.0", namespace="com.test")

    fields = {f["name"]: f for f in schema["fields"]}
    array_schema = fields["values"]["type"]

    assert array_schema["type"] == "array"
    # Items should be union with null
    assert isinstance(array_schema["items"], list)
    assert "null" in array_schema["items"]
    assert "string" in array_schema["items"]


def test_avro_schema_default_factory(manager: ModelManager) -> None:
    """Test Avro schema handles default_factory."""

    @manager.model("Config", "1.0.0")
    class ConfigV1(BaseModel):
        tags: list[str] = Field(default_factory=list)
        metadata: dict[str, str] = Field(default_factory=dict)

    schema = manager.get_avro_schema("Config", "1.0.0", namespace="com.test")

    fields = {f["name"]: f for f in schema["fields"]}

    # Default factory should produce the default value
    assert fields["tags"]["default"] == []
    assert fields["metadata"]["default"] == {}


# ============================================================================
# Tuple Tests
# ============================================================================


def test_avro_schema_tuple_type(manager: ModelManager) -> None:
    """Test Avro schema handles tuple types."""

    @manager.model("Coordinates", "1.0.0")
    class CoordinatesV1(BaseModel):
        point: tuple[float, float]
        mixed: tuple[str, int, bool]

    schema = manager.get_avro_schema("Coordinates", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Tuple of same types becomes array of that type
    assert fields["point"]["type"] == {"type": "array", "items": "double"}

    # Tuple of mixed types becomes array of union
    mixed_type = fields["mixed"]["type"]
    assert mixed_type["type"] == "array"
    assert isinstance(mixed_type["items"], list)
    assert "string" in mixed_type["items"]
    assert "int" in mixed_type["items"]
    assert "boolean" in mixed_type["items"]


def test_avro_schema_empty_tuple(manager: ModelManager) -> None:
    """Test Avro schema for empty tuple."""

    @manager.model("Empty", "1.0.0")
    class EmptyV1(BaseModel):
        empty: tuple[()]

    schema = manager.get_avro_schema("Empty", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Empty tuple becomes array of string
    assert fields["empty"]["type"] == {"type": "array", "items": "string"}


# ============================================================================
# Union Type Tests
# ============================================================================


def test_avro_schema_union_types(manager: ModelManager) -> None:
    """Test Avro schema generation for union types."""

    @manager.model("Mixed", "1.0.0")
    class MixedV1(BaseModel):
        value: str | int
        optional_value: str | int | None = None

    schema = manager.get_avro_schema("Mixed", "1.0.0", namespace="com.test")

    fields = {f["name"]: f for f in schema["fields"]}

    # Non-optional union
    assert isinstance(fields["value"]["type"], list)
    assert "string" in fields["value"]["type"]
    assert "int" in fields["value"]["type"]

    # Optional union
    assert isinstance(fields["optional_value"]["type"], list)
    assert "null" in fields["optional_value"]["type"]
    assert "string" in fields["optional_value"]["type"]
    assert "int" in fields["optional_value"]["type"]


def test_avro_schema_union_with_logical_types(manager: ModelManager) -> None:
    """Test Avro schema for union with logical types."""

    @manager.model("Event", "1.0.0")
    class EventV1(BaseModel):
        identifier: str | UUID

    schema = manager.get_avro_schema("Event", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    id_type = fields["identifier"]["type"]
    assert isinstance(id_type, list)
    assert "string" in id_type
    assert {"type": "string", "logicalType": "uuid"} in id_type


# ============================================================================
# Nested Model Tests
# ============================================================================


def test_avro_schema_nested_model(manager: ModelManager) -> None:
    """Test Avro schema generation for nested Pydantic models."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        """Mailing address."""

        street: str
        city: str
        zip_code: str

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    schema = manager.get_avro_schema("User", "1.0.0", namespace="com.test")

    fields = {f["name"]: f for f in schema["fields"]}
    address_schema = fields["address"]["type"]

    assert address_schema["type"] == "record"
    assert address_schema["name"] == "AddressV1"
    assert address_schema["doc"] == "Mailing address."
    assert len(address_schema["fields"]) == 3

    nested_fields = {f["name"]: f["type"] for f in address_schema["fields"]}
    assert nested_fields["street"] == "string"
    assert nested_fields["city"] == "string"
    assert nested_fields["zip_code"] == "string"


def test_avro_schema_optional_nested_model(manager: ModelManager) -> None:
    """Test Avro schema generation for optional nested models."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: str

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        billing_address: AddressV1 | None = None

    schema = manager.get_avro_schema("User", "1.0.0", namespace="com.test")

    fields = {f["name"]: f for f in schema["fields"]}
    billing_type = fields["billing_address"]["type"]

    assert isinstance(billing_type, list)
    assert "null" in billing_type

    record_type = next(
        t for t in billing_type if isinstance(t, dict) and t.get("type") == "record"
    )
    assert record_type["name"] == "AddressV1"


def test_avro_schema_recursive_reference(manager: ModelManager) -> None:
    """Test Avro schema handles recursive type references."""

    @manager.model("Node", "1.0.0")
    class NodeV1(BaseModel):
        value: int
        children: list["NodeV1"]

    schema = manager.get_avro_schema("Node", "1.0.0", namespace="com.test")

    assert schema["type"] == "record"
    assert schema["name"] == "Node"

    fields = {f["name"]: f for f in schema["fields"]}
    children_type = fields["children"]["type"]

    assert children_type["type"] == "array"
    # Should be a string reference to Node, not a full nested definition
    assert children_type["items"] == "Node"


def test_avro_schema_multiple_references_same_type(manager: ModelManager) -> None:
    """Test that same nested type is referenced, not duplicated."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: str

    @manager.model("Company", "1.0.0")
    class CompanyV1(BaseModel):
        name: str
        headquarters: AddressV1
        billing_address: AddressV1

    schema = manager.get_avro_schema("Company", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # First occurrence should have full schema
    hq_type = fields["headquarters"]["type"]
    assert isinstance(hq_type, dict)
    assert hq_type["type"] == "record"
    assert hq_type["name"] == "AddressV1"

    # Second occurrence should just be a reference
    billing_type = fields["billing_address"]["type"]
    assert billing_type == "AddressV1"


# ============================================================================
# Complex Real-World Tests
# ============================================================================


def test_avro_schema_complex_example(manager: ModelManager) -> None:
    """Test Avro schema generation for a complex real-world model."""

    @manager.model("Order", "1.0.0")
    class OrderV1(BaseModel):
        """Customer order."""

        order_id: str = Field(description="Unique order identifier")
        customer_name: str = Field(description="Customer full name")
        items: list[str] = Field(description="List of item IDs")
        quantities: list[int] = Field(description="Quantity for each item")
        metadata: dict[str, str] = Field(
            default_factory=dict, description="Extra metadata"
        )
        total: float = Field(description="Order total amount")
        discount: float | None = Field(default=None, description="Discount amount")
        status: str = Field(default="pending", description="Order status")
        is_paid: bool = Field(default=False, description="Payment status")

    schema = manager.get_avro_schema("Order", "1.0.0", namespace="com.store")

    assert schema["type"] == "record"
    assert schema["name"] == "Order"
    assert schema["namespace"] == "com.store"
    assert schema["doc"] == "Customer order."
    assert len(schema["fields"]) == 9

    fields = {f["name"]: f for f in schema["fields"]}

    # Check required fields
    assert fields["order_id"]["type"] == "string"
    assert fields["customer_name"]["type"] == "string"
    assert fields["total"]["type"] == "double"

    # Check collections
    assert fields["items"]["type"] == {"type": "array", "items": "string"}
    assert fields["quantities"]["type"] == {"type": "array", "items": "int"}
    assert fields["metadata"]["type"] == {"type": "map", "values": "string"}

    # Check optional field
    assert fields["discount"]["type"] == ["null", "double"]
    assert fields["discount"]["default"] is None

    # Check fields with defaults
    assert fields["status"]["default"] == "pending"
    assert fields["is_paid"]["default"] is False


# ============================================================================
# Version and Namespace Tests
# ============================================================================


def test_avro_schema_version_in_namespace(manager: ModelManager) -> None:
    """Test that version is included in Avro namespace."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    schema_v1 = manager.get_avro_schema("User", "1.0.0", namespace="com.myapp")
    schema_v2 = manager.get_avro_schema(
        "User", "2.0.0", namespace="com.myapp", versioned_namespace=True
    )

    assert schema_v1["namespace"] == "com.myapp"
    assert schema_v2["namespace"] == "com.myapp.v2_0_0"


def test_avro_generator_namespace_formatting(
    avro_generator: AvroSchemaGenerator,
) -> None:
    """Test that namespace is properly formatted with version."""

    class TestModel(BaseModel):
        name: str

    schema = avro_generator.generate_schema(TestModel, "Test")
    schema_versioned = avro_generator.generate_schema(TestModel, "Test", "1.2.3")

    assert schema["namespace"] == "com.test"
    assert schema_versioned["namespace"] == "com.test.v1_2_3"


def test_avro_schema_multiple_versions_different_schemas(manager: ModelManager) -> None:
    """Test that different versions generate different Avro schemas."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str
        age: int

    schema_v1 = manager.get_avro_schema("User", "1.0.0", namespace="com.test")
    schema_v2 = manager.get_avro_schema("User", "2.0.0", namespace="com.test")

    assert len(schema_v1["fields"]) == 1
    assert len(schema_v2["fields"]) == 3
    assert schema_v1["namespace"] == "com.test"
    assert schema_v2["namespace"] == "com.test"


# ============================================================================
# File Export Tests
# ============================================================================


def test_dump_avro_schemas_creates_files(manager: ModelManager, tmp_path: Path) -> None:
    """Test that dump_avro_schemas creates .avsc files."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    @manager.model("Order", "1.0.0")
    class OrderV1(BaseModel):
        order_id: str

    output_dir = tmp_path / "avro_schemas"
    schemas = manager.dump_avro_schemas(output_dir, namespace="com.test")

    # Check return value structure
    assert "User" in schemas
    assert "Order" in schemas
    assert "1.0.0" in schemas["User"]
    assert "2.0.0" in schemas["User"]
    assert "1.0.0" in schemas["Order"]

    # Check files were created
    assert (output_dir / "User_v1_0_0.avsc").exists()
    assert (output_dir / "User_v2_0_0.avsc").exists()
    assert (output_dir / "Order_v1_0_0.avsc").exists()

    # Check file content
    user_v1_content = json.loads((output_dir / "User_v1_0_0.avsc").read_text())
    assert user_v1_content["name"] == "User"
    assert user_v1_content["namespace"] == "com.test"


def test_dump_avro_schemas_custom_indent(manager: ModelManager, tmp_path: Path) -> None:
    """Test dump_avro_schemas with custom indentation."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    output_dir = tmp_path / "schemas"
    manager.dump_avro_schemas(output_dir, namespace="com.test", indent=4)

    content = (output_dir / "User_v1_0_0.avsc").read_text()

    # Check that indentation is present (4 spaces)
    assert "    " in content

    # Verify it's valid JSON
    schema = json.loads(content)
    assert schema["name"] == "User"


def test_dump_avro_schemas_without_docs(manager: ModelManager, tmp_path: Path) -> None:
    """Test dump_avro_schemas without documentation."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        """User model."""

        name: str = Field(description="User name")

    output_dir = tmp_path / "schemas"
    manager.dump_avro_schemas(output_dir, namespace="com.test", include_docs=False)

    schema = json.loads((output_dir / "User_v1_0_0.avsc").read_text())

    assert "doc" not in schema
    assert all("doc" not in f for f in schema["fields"])


def test_avro_exporter_export_single_schema(
    manager: ModelManager, tmp_path: Path
) -> None:
    """Test AvroExporter can export a single schema."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        age: int

    exporter = AvroExporter(manager._registry, namespace="com.app")

    output_path = tmp_path / "user.avsc"
    schema = exporter.export_schema("User", "1.0.0", output_path)

    # Check return value
    assert schema["name"] == "User"
    assert schema["namespace"] == "com.app"
    assert len(schema["fields"]) == 2

    # Check file was created
    assert output_path.exists()

    # Check file content
    file_schema = json.loads(output_path.read_text())
    assert file_schema == schema


def test_avro_exporter_export_without_file(manager: ModelManager) -> None:
    """Test AvroExporter can generate schema without saving to file."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    exporter = AvroExporter(manager._registry, namespace="com.app")
    schema = exporter.export_schema("User", "1.0.0")

    assert schema["name"] == "User"
    assert schema["namespace"] == "com.app"


# ============================================================================
# Edge Cases
# ============================================================================


def test_avro_schema_empty_model(manager: ModelManager) -> None:
    """Test Avro schema generation for model with no fields."""

    @manager.model("Empty", "1.0.0")
    class EmptyV1(BaseModel):
        pass

    schema = manager.get_avro_schema("Empty", "1.0.0", namespace="com.test")

    assert schema["type"] == "record"
    assert schema["name"] == "Empty"
    assert schema["fields"] == []


def test_avro_schema_preserves_field_order(manager: ModelManager) -> None:
    """Test that Avro schema preserves field order from Pydantic model."""

    @manager.model("Ordered", "1.0.0")
    class OrderedV1(BaseModel):
        alpha: str
        beta: int
        gamma: float
        delta: bool

    schema = manager.get_avro_schema("Ordered", "1.0.0", namespace="com.test")

    field_names = [f["name"] for f in schema["fields"]]
    assert field_names == ["alpha", "beta", "gamma", "delta"]


def test_avro_schema_nested_collections(manager: ModelManager) -> None:
    """Test Avro schema for nested collection types."""

    @manager.model("Complex", "1.0.0")
    class ComplexV1(BaseModel):
        matrix: list[list[int]]
        nested_map: dict[str, list[str]]

    schema = manager.get_avro_schema("Complex", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # List of lists
    matrix_type = fields["matrix"]["type"]
    assert matrix_type["type"] == "array"
    assert matrix_type["items"]["type"] == "array"
    assert matrix_type["items"]["items"] == "int"

    # Map with list values
    nested_map_type = fields["nested_map"]["type"]
    assert nested_map_type["type"] == "map"
    assert nested_map_type["values"]["type"] == "array"
    assert nested_map_type["values"]["items"] == "string"


def test_avro_schema_model_without_docstring(manager: ModelManager) -> None:
    """Test Avro schema for model without docstring."""

    @manager.model("Simple", "1.0.0")
    class SimpleV1(BaseModel):
        name: str

    schema = manager.get_avro_schema("Simple", "1.0.0", namespace="com.test")

    # Should not have doc field if no docstring
    assert "doc" not in schema


def test_avro_schema_field_without_description(manager: ModelManager) -> None:
    """Test Avro schema for field without description."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        age: int = Field(description="User age")

    schema = manager.get_avro_schema("User", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # name has no description
    assert "doc" not in fields["name"]
    # age has description
    assert fields["age"]["doc"] == "User age"


# ============================================================================
# Integer Optimization Tests
# ============================================================================


def test_avro_schema_int_without_constraints(manager: ModelManager) -> None:
    """Test that int without constraints defaults to long for safety."""

    @manager.model("Data", "1.0.0")
    class DataV1(BaseModel):
        count: int

    schema = manager.get_avro_schema("Data", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Without constraints, should use long for safety
    assert fields["count"]["type"] == "int"  # Default in basic mapping


def test_avro_schema_int_with_constraints_fits_32bit(manager: ModelManager) -> None:
    """Test that constrained int uses int type when it fits in 32 bits."""

    @manager.model("Data", "1.0.0")
    class DataV1(BaseModel):
        small_number: Annotated[int, Field(ge=0, le=1000)]

    schema = manager.get_avro_schema("Data", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    assert fields["small_number"]["type"] == "int"


# ============================================================================
# Default Value Conversion Tests
# ============================================================================


def test_avro_schema_enum_default_value(manager: ModelManager) -> None:
    """Test that enum default values are converted to strings."""

    class Color(str, Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    @manager.model("Item", "1.0.0")
    class ItemV1(BaseModel):
        name: str
        color: Color = Color.RED

    schema = manager.get_avro_schema("Item", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Enum default should be converted to string value
    assert fields["color"]["default"] == "red"


def test_avro_schema_datetime_default_value(manager: ModelManager) -> None:
    """Test that datetime default values are converted properly."""

    @manager.model("Event", "1.0.0")
    class EventV1(BaseModel):
        name: str
        timestamp: datetime = Field(default_factory=datetime.now)

    schema = manager.get_avro_schema("Event", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Should have a default (produced by factory)
    assert "default" in fields["timestamp"]
    assert isinstance(fields["timestamp"]["default"], int)


def test_avro_schema_uuid_default_value(manager: ModelManager) -> None:
    """Test that UUID default values are converted to strings."""
    default_uuid = uuid4()

    @manager.model("Resource", "1.0.0")
    class ResourceV1(BaseModel):
        name: str
        id: UUID = default_uuid

    schema = manager.get_avro_schema("Resource", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # UUID should be converted to string
    assert fields["id"]["default"] == str(default_uuid)


def test_avro_schema_bytes_default_value(manager: ModelManager) -> None:
    """Test that bytes default values are handled."""

    @manager.model("Data", "1.0.0")
    class DataV1(BaseModel):
        content: bytes = b"hello"

    schema = manager.get_avro_schema("Data", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Bytes should be converted to string
    assert fields["content"]["default"] == "hello"


# ============================================================================
# Union Edge Cases
# ============================================================================


def test_avro_schema_union_duplicate_removal(manager: ModelManager) -> None:
    """Test that duplicate types in unions are removed."""

    @manager.model("Data", "1.0.0")
    class DataV1(BaseModel):
        # This creates a union that might have duplicates after processing
        value: str | int | str  # Duplicate str

    schema = manager.get_avro_schema("Data", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    value_type = fields["value"]["type"]
    # Should only have unique types
    assert isinstance(value_type, list)
    # Count occurrences of string
    string_count = sum(1 for t in value_type if t == "string")
    assert string_count == 1


def test_avro_schema_nested_union_flattening(manager: ModelManager) -> None:
    """Test that nested unions are flattened."""

    @manager.model("Data", "1.0.0")
    class DataV1(BaseModel):
        # Union of union should flatten
        value: str | int | None

    schema = manager.get_avro_schema("Data", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    value_type = fields["value"]["type"]
    assert isinstance(value_type, list)
    # Should be flattened list with null first (for optional)
    assert value_type[0] == "null"
    assert "string" in value_type
    assert "int" in value_type


# ============================================================================
# List and Array Edge Cases
# ============================================================================


def test_avro_schema_empty_list_annotation(manager: ModelManager) -> None:
    """Test Avro schema for list without type parameter."""

    @manager.model("Data", "1.0.0")
    class DataV1(BaseModel):
        items: list  # type: ignore[type-arg]

    schema = manager.get_avro_schema("Data", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # List without parameter should default to string items
    assert fields["items"]["type"] == {"type": "array", "items": "string"}


def test_avro_schema_list_of_enums(manager: ModelManager) -> None:
    """Test Avro schema for list of enum values."""

    class Status(str, Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    @manager.model("Report", "1.0.0")
    class ReportV1(BaseModel):
        statuses: list[Status]

    schema = manager.get_avro_schema("Report", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    array_type = fields["statuses"]["type"]
    assert array_type["type"] == "array"

    items_type = array_type["items"]
    assert items_type["type"] == "enum"
    assert items_type["name"] == "Status"
    assert items_type["symbols"] == ["active", "inactive"]


def test_avro_schema_list_of_nested_models(manager: ModelManager) -> None:
    """Test Avro schema for list of nested Pydantic models."""

    @manager.model("Item", "1.0.0")
    class ItemV1(BaseModel):
        name: str
        price: float

    @manager.model("Order", "1.0.0")
    class OrderV1(BaseModel):
        order_id: str
        items: list[ItemV1]

    schema = manager.get_avro_schema("Order", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    array_type = fields["items"]["type"]
    assert array_type["type"] == "array"

    items_type = array_type["items"]
    assert isinstance(items_type, dict)
    assert items_type["type"] == "record"
    assert items_type["name"] == "ItemV1"


# ============================================================================
# Map/Dict Edge Cases
# ============================================================================


def test_avro_schema_empty_dict_annotation(manager: ModelManager) -> None:
    """Test Avro schema for dict without type parameters."""

    @manager.model("Data", "1.0.0")
    class DataV1(BaseModel):
        metadata: dict  # type: ignore[type-arg]

    schema = manager.get_avro_schema("Data", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Dict without parameters should default to string values
    assert fields["metadata"]["type"] == {"type": "map", "values": "string"}


def test_avro_schema_dict_of_nested_models(manager: ModelManager) -> None:
    """Test Avro schema for dict with nested model values."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: str

    @manager.model("Company", "1.0.0")
    class CompanyV1(BaseModel):
        name: str
        offices: dict[str, AddressV1]

    schema = manager.get_avro_schema("Company", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    map_type = fields["offices"]["type"]
    assert map_type["type"] == "map"

    values_type = map_type["values"]
    assert isinstance(values_type, dict)
    assert values_type["type"] == "record"
    assert values_type["name"] == "AddressV1"


# ============================================================================
# Integration Tests
# ============================================================================


def test_avro_schema_kafka_integration_example(manager: ModelManager) -> None:
    """Test complete example suitable for Kafka schema registry."""

    class EventType(str, Enum):
        USER_CREATED = "user_created"
        USER_UPDATED = "user_updated"
        USER_DELETED = "user_deleted"

    @manager.model("UserEvent", "1.0.0")
    class UserEventV1(BaseModel):
        """User lifecycle event for Kafka topic."""

        event_id: UUID = Field(description="Unique event identifier")
        event_type: EventType = Field(description="Type of user event")
        timestamp: datetime = Field(description="When the event occurred")
        user_id: str = Field(description="User identifier")
        user_name: str = Field(description="User full name")
        email: str | None = Field(default=None, description="User email address")
        metadata: dict[str, str] = Field(
            default_factory=dict, description="Additional event metadata"
        )

    schema = manager.get_avro_schema("UserEvent", "1.0.0", namespace="com.myapp.events")

    # Verify it's a complete, valid Avro schema
    assert schema["type"] == "record"
    assert schema["name"] == "UserEvent"
    assert schema["namespace"] == "com.myapp.events"
    assert schema["doc"] == "User lifecycle event for Kafka topic."

    fields = {f["name"]: f for f in schema["fields"]}

    # Verify logical types
    assert fields["event_id"]["type"] == {"type": "string", "logicalType": "uuid"}
    assert fields["timestamp"]["type"] == {
        "type": "long",
        "logicalType": "timestamp-micros",
    }

    # Verify enum
    event_type = fields["event_type"]["type"]
    assert event_type["type"] == "enum"
    assert "user_created" in event_type["symbols"]

    # Verify optional field
    assert fields["email"]["type"] == ["null", "string"]
    assert fields["email"]["default"] is None

    # Verify all fields have documentation
    for field_name in ["event_id", "event_type", "timestamp", "user_id", "user_name"]:
        assert "doc" in fields[field_name]


def test_avro_schema_full_workflow(manager: ModelManager, tmp_path: Path) -> None:
    """Test complete workflow: define models, generate schemas, save to files."""

    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    @manager.model("Task", "1.0.0")
    class TaskV1(BaseModel):
        """Task tracking model."""

        task_id: UUID
        title: str
        description: str | None = None
        priority: Priority = Priority.MEDIUM
        created_at: datetime
        due_date: date | None = None
        tags: list[str] = Field(default_factory=list)

    @manager.model("Task", "2.0.0")
    class TaskV2(BaseModel):
        """Task tracking model with assignee."""

        task_id: UUID
        title: str
        description: str | None = None
        priority: Priority = Priority.MEDIUM
        created_at: datetime
        due_date: date | None = None
        tags: list[str] = Field(default_factory=list)
        assignee: str | None = None

    # Export all schemas
    output_dir = tmp_path / "kafka_schemas"
    schemas = manager.dump_avro_schemas(
        output_dir, namespace="com.taskapp", indent=2, versioned_namespace=True
    )

    # Verify both versions exported
    assert "Task" in schemas
    assert "1.0.0" in schemas["Task"]
    assert "2.0.0" in schemas["Task"]

    # Verify files created
    v1_file = output_dir / "Task_v1_0_0.avsc"
    v2_file = output_dir / "Task_v2_0_0.avsc"
    assert v1_file.exists()
    assert v2_file.exists()

    # Verify schemas are different
    v1_schema = json.loads(v1_file.read_text())
    v2_schema = json.loads(v2_file.read_text())

    v1_fields = {f["name"] for f in v1_schema["fields"]}
    v2_fields = {f["name"] for f in v2_schema["fields"]}

    assert "assignee" not in v1_fields
    assert "assignee" in v2_fields

    # Verify both are valid Avro schemas
    for schema in [v1_schema, v2_schema]:
        assert schema["type"] == "record"
        assert schema["name"] == "Task"
        assert "v1_0_0" in schema["namespace"] or "v2_0_0" in schema["namespace"]


def test_avro_schema_backwards_compatible_evolution(manager: ModelManager) -> None:
    """Test that schema evolution maintains backward compatibility patterns."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        user_id: str
        name: str
        email: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        user_id: str
        name: str
        email: str
        phone: str | None = None  # New optional field
        created_at: datetime | None = None  # New optional field

    schema_v1 = manager.get_avro_schema("User", "1.0.0", namespace="com.app")
    schema_v2 = manager.get_avro_schema("User", "2.0.0", namespace="com.app")

    fields_v1 = {f["name"] for f in schema_v1["fields"]}
    fields_v2 = {f["name"] for f in schema_v2["fields"]}

    # V2 is a superset of V1
    assert fields_v1.issubset(fields_v2)

    # New fields in V2 are optional with defaults
    v2_field_dict = {f["name"]: f for f in schema_v2["fields"]}
    assert v2_field_dict["phone"]["default"] is None
    assert v2_field_dict["created_at"]["default"] is None


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_avro_schema_model_not_found(manager: ModelManager) -> None:
    """Test error handling when model is not found."""
    with pytest.raises(ModelNotFoundError):
        manager.get_avro_schema("NonExistent", "1.0.0", namespace="com.test")


def test_avro_schema_version_not_found(manager: ModelManager) -> None:
    """Test error handling when version is not found."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    with pytest.raises(ModelNotFoundError):
        manager.get_avro_schema("User", "2.0.0", namespace="com.test")


# ============================================================================
# Performance and Scale Tests
# ============================================================================


def test_avro_schema_deeply_nested_models(manager: ModelManager) -> None:
    """Test Avro schema for deeply nested model structures."""

    @manager.model("Level3", "1.0.0")
    class Level3V1(BaseModel):
        value: str

    @manager.model("Level2", "1.0.0")
    class Level2V1(BaseModel):
        data: Level3V1

    @manager.model("Level1", "1.0.0")
    class Level1V1(BaseModel):
        nested: Level2V1

    @manager.model("Root", "1.0.0")
    class RootV1(BaseModel):
        deep: Level1V1

    schema = manager.get_avro_schema("Root", "1.0.0", namespace="com.test")

    # Should handle deep nesting
    assert schema["type"] == "record"
    fields = {f["name"]: f for f in schema["fields"]}

    # Verify deep nesting structure
    deep_type = fields["deep"]["type"]
    assert deep_type["type"] == "record"
    assert deep_type["name"] == "Level1V1"


def test_avro_exporter_multiple_models(manager: ModelManager, tmp_path: Path) -> None:
    """Test exporting multiple unrelated models."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("Product", "1.0.0")
    class ProductV1(BaseModel):
        name: str
        price: float

    @manager.model("Order", "1.0.0")
    class OrderV1(BaseModel):
        order_id: str

    output_dir = tmp_path / "schemas"
    schemas = manager.dump_avro_schemas(output_dir, namespace="com.store")

    # All models should be exported
    assert len(schemas) == 3
    assert "User" in schemas
    assert "Product" in schemas
    assert "Order" in schemas

    # All files should exist
    assert (output_dir / "User_v1_0_0.avsc").exists()
    assert (output_dir / "Product_v1_0_0.avsc").exists()
    assert (output_dir / "Order_v1_0_0.avsc").exists()


# ============================================================================
# JSON Schema Comparison Tests
# ============================================================================


def test_avro_schema_differs_from_json_schema(manager: ModelManager) -> None:
    """Test that Avro schema is different from JSON schema."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        age: int | None = None

    avro_schema = manager.get_avro_schema("User", "1.0.0", namespace="com.test")
    json_schema = manager.get_schema("User", "1.0.0")

    # Different formats
    assert avro_schema["type"] == "record"
    assert json_schema["type"] == "object"

    # Avro uses fields array, JSON Schema uses properties dict
    assert "fields" in avro_schema
    assert "properties" in json_schema

    # Avro has namespace, JSON Schema doesn't
    assert "namespace" in avro_schema
    assert "namespace" not in json_schema


# ============================================================================
# Default Factory Edge Cases
# ============================================================================


def test_avro_schema_default_factory_raises_exception(manager: ModelManager) -> None:
    """Test handling of default_factory that raises an exception."""

    def failing_factory() -> list[str]:
        raise ValueError("Factory failed")

    @manager.model("Config", "1.0.0")
    class ConfigV1(BaseModel):
        tags: list[str] = Field(default_factory=failing_factory)

    schema = manager.get_avro_schema("Config", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Should not have a default when factory fails
    assert "default" not in fields["tags"]


def test_avro_schema_optional_with_failing_factory(manager: ModelManager) -> None:
    """Test optional field with default_factory that raises exception."""

    def failing_factory() -> dict[str, str]:
        raise RuntimeError("Cannot create default")

    @manager.model("Data", "1.0.0")
    class DataV1(BaseModel):
        metadata: dict[str, str] | None = Field(default_factory=failing_factory)

    schema = manager.get_avro_schema("Data", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Optional field should default to None when factory fails
    assert fields["metadata"]["default"] is None
    assert fields["metadata"]["type"][0] == "null"  # type: ignore[literal-required]


def test_avro_schema_non_optional_with_failing_factory(manager: ModelManager) -> None:
    """Test non-optional field with default_factory that raises exception."""

    def failing_factory() -> list[int]:
        raise TypeError("Invalid factory")

    @manager.model("Numbers", "1.0.0")
    class NumbersV1(BaseModel):
        values: list[int] = Field(default_factory=failing_factory)

    schema = manager.get_avro_schema("Numbers", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Non-optional field should not have default when factory fails
    assert "default" not in fields["values"]


# ============================================================================
# Integer Constraint Optimization Tests
# ============================================================================


def test_avro_schema_int_with_ge_constraint(manager: ModelManager) -> None:
    """Test int optimization with greater-than-or-equal constraint."""

    @manager.model("Product", "1.0.0")
    class ProductV1(BaseModel):
        stock: Annotated[int, Field(ge=0, le=1000000)]

    schema = manager.get_avro_schema("Product", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    assert fields["stock"]["type"] == "int"


def test_avro_schema_int_with_gt_constraint(manager: ModelManager) -> None:
    """Test int optimization with greater-than constraint."""

    @manager.model("Counter", "1.0.0")
    class CounterV1(BaseModel):
        count: Annotated[int, Field(gt=-100, lt=100)]

    schema = manager.get_avro_schema("Counter", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    assert fields["count"]["type"] == "int"


def test_avro_schema_int_with_only_minimum(manager: ModelManager) -> None:
    """Test int type when only minimum constraint is provided."""

    @manager.model("PositiveNumber", "1.0.0")
    class PositiveNumberV1(BaseModel):
        value: Annotated[int, Field(ge=0)]

    schema = manager.get_avro_schema("PositiveNumber", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}
    assert fields["value"]["type"] == "int"


def test_avro_schema_int_with_only_maximum(manager: ModelManager) -> None:
    """Test int type when only maximum constraint is provided."""

    @manager.model("SmallNumber", "1.0.0")
    class SmallNumberV1(BaseModel):
        value: Annotated[int, Field(le=1000)]

    schema = manager.get_avro_schema("SmallNumber", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}
    assert fields["value"]["type"] == "int"


def test_avro_schema_int_exceeds_32bit_range(manager: ModelManager) -> None:
    """Test int type when constraints exceed 32-bit range."""

    @manager.model("BigNumber", "1.0.0")
    class BigNumberV1(BaseModel):
        large_value: Annotated[int, Field(ge=0, le=10_000_000_000)]

    schema = manager.get_avro_schema("BigNumber", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Exceeds 32-bit, should use long
    assert fields["large_value"]["type"] == "long"


def test_avro_schema_int_negative_range_exceeds_32bit(manager: ModelManager) -> None:
    """Test int type when negative range exceeds 32-bit."""

    @manager.model("NegativeRange", "1.0.0")
    class NegativeRangeV1(BaseModel):
        value: Annotated[int, Field(ge=-5_000_000_000, le=0)]

    schema = manager.get_avro_schema("NegativeRange", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Exceeds 32-bit on negative side, should use long
    assert fields["value"]["type"] == "long"


def test_avro_schema_int_without_field_info(manager: ModelManager) -> None:
    """Test int type falls back to basic mapping when no field_info."""
    generator = AvroSchemaGenerator(namespace="com.test")

    # Call _python_type_to_avro with None field_info
    avro_type = generator._python_type_to_avro(int, field_info=None)

    # Should return basic mapping default
    assert avro_type == "int"


def test_avro_schema_int_without_metadata(manager: ModelManager) -> None:
    """Test int optimization when field has no metadata."""

    @manager.model("Simple", "1.0.0")
    class SimpleV1(BaseModel):
        count: int

    schema = manager.get_avro_schema("Simple", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # No constraints means default to int from basic mapping
    assert fields["count"]["type"] == "int"


# ============================================================================
# Default Value Conversion Edge Cases
# ============================================================================


def test_avro_schema_pydantic_model_default(manager: ModelManager) -> None:
    """Test default value that is a Pydantic model instance."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: str

    default_address = AddressV1(street="123 Main St", city="Anytown")

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1 = default_address

    schema = manager.get_avro_schema("User", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # BaseModel should be converted to dict via model_dump
    assert isinstance(fields["address"]["default"], dict)
    assert fields["address"]["default"]["street"] == "123 Main St"
    assert fields["address"]["default"]["city"] == "Anytown"


def test_avro_schema_date_default_conversion(manager: ModelManager) -> None:
    """Test date default value is converted to days since epoch."""
    test_date = date(2023, 6, 15)

    @manager.model("Event", "1.0.0")
    class EventV1(BaseModel):
        name: str
        event_date: date = test_date

    schema = manager.get_avro_schema("Event", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Should be days since epoch
    epoch = date(1970, 1, 1)
    expected_days = (test_date - epoch).days
    assert fields["event_date"]["default"] == expected_days


def test_avro_schema_time_default_conversion(manager: ModelManager) -> None:
    """Test time default value is converted to microseconds since midnight."""
    test_time = time(14, 30, 45, 123456)

    @manager.model("Schedule", "1.0.0")
    class ScheduleV1(BaseModel):
        name: str
        start_time: time = test_time

    schema = manager.get_avro_schema("Schedule", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Should be microseconds since midnight
    expected_micros = (
        test_time.hour * 3600 + test_time.minute * 60 + test_time.second
    ) * 1_000_000 + test_time.microsecond
    assert fields["start_time"]["default"] == expected_micros


def test_avro_schema_decimal_default_conversion(manager: ModelManager) -> None:
    """Test Decimal default value is converted to float."""
    test_decimal = Decimal("123.45")

    @manager.model("Price", "1.0.0")
    class PriceV1(BaseModel):
        amount: Decimal = test_decimal

    schema = manager.get_avro_schema("Price", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Decimal should be converted to float
    assert fields["amount"]["default"] == 123.45
    assert isinstance(fields["amount"]["default"], float)


def test_avro_schema_bytes_with_decode_error(manager: ModelManager) -> None:
    """Test bytes default with invalid UTF-8 is handled gracefully."""
    invalid_bytes = b"\xff\xfe\xfd"

    @manager.model("Binary", "1.0.0")
    class BinaryV1(BaseModel):
        data: bytes = invalid_bytes

    schema = manager.get_avro_schema("Binary", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Should handle decode errors with ignore
    assert isinstance(fields["data"]["default"], str)


def test_avro_schema_nested_list_default(manager: ModelManager) -> None:
    """Test nested list default value conversion."""

    @manager.model("Matrix", "1.0.0")
    class MatrixV1(BaseModel):
        grid: list[list[int]] = Field(default_factory=lambda: [[1, 2], [3, 4]])

    schema = manager.get_avro_schema("Matrix", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Nested lists should be preserved
    assert fields["grid"]["default"] == [[1, 2], [3, 4]]


def test_avro_schema_nested_dict_default(manager: ModelManager) -> None:
    """Test nested dict default value conversion."""

    @manager.model("Config", "1.0.0")
    class ConfigV1(BaseModel):
        settings: dict[str, dict[str, int]] = Field(
            default_factory=lambda: {"db": {"port": 5432, "timeout": 30}}
        )

    schema = manager.get_avro_schema("Config", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # Nested dicts should be preserved
    assert fields["settings"]["default"] == {"db": {"port": 5432, "timeout": 30}}


def test_avro_schema_unknown_type_default(manager: ModelManager) -> None:
    """Test that unknown types in defaults are converted to string."""
    generator = AvroSchemaGenerator(namespace="com.test")

    class CustomClass:
        def __str__(self) -> str:
            return "custom_value"

    # Convert unknown type
    result = generator._convert_default_value(CustomClass())

    # Should fall back to string conversion
    assert result == "custom_value"


def test_avro_schema_nested_anyhttpurl_with_value(manager: ModelManager) -> None:
    """Tests that nested models with AnyHttpUrls value serializes correctly."""

    class Inner(BaseModel):
        url: AnyHttpUrl

    @manager.model("Outer", "1.0.0")
    class Outer(BaseModel):
        inner: Inner = Inner(url=AnyHttpUrl("http://test.com"))

    # Doesn't fail to serialize
    json.dumps(manager.get_avro_schema("Outer", "1.0.0"))


# ============================================================================
# Type Annotation Edge Cases
# ============================================================================


def test_avro_schema_none_type_annotation(manager: ModelManager) -> None:
    """Test handling of None as type annotation."""
    generator = AvroSchemaGenerator(namespace="com.test")

    # None annotation should return "null"
    avro_type = generator._python_type_to_avro(None)
    assert avro_type == "null"


def test_avro_schema_complex_union_with_none(manager: ModelManager) -> None:
    """Test complex union including None is handled correctly."""

    @manager.model("Flexible", "1.0.0")
    class FlexibleV1(BaseModel):
        value: str | int | float | None = None

    schema = manager.get_avro_schema("Flexible", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    value_type = fields["value"]["type"]
    assert isinstance(value_type, list)
    assert "null" in value_type
    assert "string" in value_type
    assert "int" in value_type
    assert "double" in value_type
    assert fields["value"]["default"] is None


def test_avro_schema_union_with_complex_types(manager: ModelManager) -> None:
    """Test union containing complex types like dict and list."""

    @manager.model("Mixed", "1.0.0")
    class MixedV1(BaseModel):
        data: list[str] | dict[str, int]

    schema = manager.get_avro_schema("Mixed", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    data_type = fields["data"]["type"]
    assert isinstance(data_type, list)

    # Should contain array and map schemas
    has_array = any(isinstance(t, dict) and t.get("type") == "array" for t in data_type)
    has_map = any(isinstance(t, dict) and t.get("type") == "map" for t in data_type)

    assert has_array
    assert has_map


def test_avro_schema_string_type_hints(manager: ModelManager) -> None:
    """Test handling of type hints as strings (forward references)."""
    generator = AvroSchemaGenerator(namespace="com.test")

    # Simulate string annotation by testing fallback logic
    # The code has fallback for string representations
    class FakeAnnotation:
        def __str__(self) -> str:
            return "typing.List[str]"

    # Should fallback to string when unable to parse
    result = generator._python_type_to_avro(FakeAnnotation())
    assert result == "string"


# ============================================================================
# Nested Model Reference Tests
# ============================================================================


def test_avro_schema_circular_reference_prevention(manager: ModelManager) -> None:
    """Test that circular references are handled by using type names."""

    @manager.model("Node", "1.0.0")
    class NodeV1(BaseModel):
        value: int
        parent: "NodeV1 | None" = None
        children: list["NodeV1"] = Field(default_factory=list)

    schema = manager.get_avro_schema("Node", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    parent_type = fields["parent"]["type"]
    assert isinstance(parent_type, list)
    assert "null" in parent_type

    children_type = fields["children"]["type"]
    assert children_type["type"] == "array"
    assert children_type["items"] == "Node"


def test_avro_schema_shared_nested_model_multiple_fields(
    manager: ModelManager,
) -> None:
    """Test that same nested model used in multiple fields creates reference."""

    @manager.model("Tag", "1.0.0")
    class TagV1(BaseModel):
        name: str
        color: str

    @manager.model("Article", "1.0.0")
    class ArticleV1(BaseModel):
        title: str
        primary_tag: TagV1
        secondary_tag: TagV1
        all_tags: list[TagV1]

    schema = manager.get_avro_schema("Article", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # First occurrence should have full schema
    primary_tag_type = fields["primary_tag"]["type"]
    assert isinstance(primary_tag_type, dict)
    assert primary_tag_type["type"] == "record"
    assert primary_tag_type["name"] == "TagV1"

    # Second occurrence should be reference
    secondary_tag_type = fields["secondary_tag"]["type"]
    assert secondary_tag_type == "TagV1"

    # Array should also use reference
    all_tags_type = fields["all_tags"]["type"]
    assert all_tags_type["items"] == "TagV1"


def test_avro_schema_deeply_nested_same_type(manager: ModelManager) -> None:
    """Test deeply nested structures with same type use references."""

    @manager.model("Category", "1.0.0")
    class CategoryV1(BaseModel):
        name: str
        subcategories: list["CategoryV1"] = Field(default_factory=list)

    schema = manager.get_avro_schema("Category", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    subcat_type = fields["subcategories"]["type"]
    assert subcat_type["type"] == "array"
    assert subcat_type["items"] == "Category"


# ============================================================================
# Tuple Edge Cases
# ============================================================================


def test_avro_schema_tuple_with_duplicate_types(manager: ModelManager) -> None:
    """Test tuple with duplicate types removes duplicates in union."""

    @manager.model("Data", "1.0.0")
    class DataV1(BaseModel):
        coords: tuple[float, float, float]

    schema = manager.get_avro_schema("Data", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    # All same type should collapse to single type
    coords_type = fields["coords"]["type"]
    assert coords_type == {"type": "array", "items": "double"}


def test_avro_schema_tuple_with_optional_elements(manager: ModelManager) -> None:
    """Test tuple containing optional types."""

    @manager.model("Record", "1.0.0")
    class RecordV1(BaseModel):
        data: tuple[str, int | None, bool]

    schema = manager.get_avro_schema("Record", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    data_type = fields["data"]["type"]
    assert data_type["type"] == "array"

    items = data_type["items"]
    assert isinstance(items, list)
    assert "null" in items
    assert "string" in items
    assert "int" in items
    assert "boolean" in items


# ============================================================================
# Enum Edge Cases
# ============================================================================


def test_avro_schema_int_enum(manager: ModelManager) -> None:
    """Test enum with integer values."""

    class Priority(int, Enum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    @manager.model("Task", "1.0.0")
    class TaskV1(BaseModel):
        priority: Priority

    with pytest.raises(ValueError, match="Unable to convert enum 'Priority' to Avro"):
        manager.get_avro_schema("Task", "1.0.0", namespace="com.test")


def test_avro_schema_enum_in_nested_union(manager: ModelManager) -> None:
    """Test enum within nested union structure."""

    class Status(str, Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    @manager.model("Report", "1.0.0")
    class ReportV1(BaseModel):
        status_or_count: Status | int

    schema = manager.get_avro_schema("Report", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    union_type = fields["status_or_count"]["type"]
    assert isinstance(union_type, list)

    has_enum = any(isinstance(t, dict) and t.get("type") == "enum" for t in union_type)
    has_int = "int" in union_type

    assert has_enum
    assert has_int


# ============================================================================
# Union Type Detection Tests
# ============================================================================


def test_avro_schema_python_310_union_syntax(manager: ModelManager) -> None:
    """Test that Python 3.10+ union syntax (X | Y) is handled."""

    @manager.model("Modern", "1.0.0")
    class ModernV1(BaseModel):
        value: str | int

    schema = manager.get_avro_schema("Modern", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    value_type = fields["value"]["type"]
    assert isinstance(value_type, list)
    assert "string" in value_type
    assert "int" in value_type


def test_avro_schema_complex_nested_union_flattening(manager: ModelManager) -> None:
    """Test that complex nested unions are properly flattened."""

    @manager.model("Complex", "1.0.0")
    class ComplexV1(BaseModel):
        # This creates nested unions that should be flattened
        value: str | int | (float | bool)

    schema = manager.get_avro_schema("Complex", "1.0.0", namespace="com.test")
    fields = {f["name"]: f for f in schema["fields"]}

    value_type = fields["value"]["type"]
    assert isinstance(value_type, list)

    assert "string" in value_type
    assert "int" in value_type
    assert "double" in value_type
    assert "boolean" in value_type


# ============================================================================
# Export and File Operations
# ============================================================================


def test_avro_exporter_creates_parent_directories(
    manager: ModelManager, tmp_path: Path
) -> None:
    """Test that exporter creates parent directories if they don't exist."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    # Use nested path that doesn't exist
    output_path = tmp_path / "deeply" / "nested" / "path" / "user.avsc"

    exporter = AvroExporter(manager._registry, namespace="com.app")
    exporter.export_schema("User", "1.0.0", output_path)

    # Should create all parent directories and file
    assert output_path.exists()
    assert output_path.parent.exists()


def test_avro_schema_model_with_no_fields_no_docstring(
    manager: ModelManager,
) -> None:
    """Test model with no fields and no docstring."""

    @manager.model("Empty", "1.0.0")
    class EmptyV1(BaseModel):
        pass

    schema = manager.get_avro_schema("Empty", "1.0.0", namespace="com.test")

    assert schema["type"] == "record"
    assert schema["name"] == "Empty"
    assert schema["fields"] == []
    assert "doc" not in schema


# ============================================================================
# Integration and Real-World Scenarios
# ============================================================================


def test_avro_schema_kafka_event_with_all_features(manager: ModelManager) -> None:
    """Test comprehensive Kafka event schema with all features combined."""

    class EventType(str, Enum):
        CREATE = "create"
        UPDATE = "update"
        DELETE = "delete"

    @manager.model("Metadata", "1.0.0")
    class MetadataV1(BaseModel):
        source: str
        tags: list[str] = Field(default_factory=list)

    @manager.model("KafkaEvent", "1.0.0")
    class KafkaEventV1(BaseModel):
        """Comprehensive Kafka event with all Avro features."""

        event_id: UUID
        event_type: EventType
        timestamp: datetime
        user_id: str | int
        payload: dict[str, str | int]
        metadata: MetadataV1 | None = None
        tags: list[str] = Field(default_factory=list)
        scheduled_date: date | None = None
        coordinates: tuple[float, float] = (0.0, 0.0)

    schema = manager.get_avro_schema("KafkaEvent", "1.0.0", namespace="com.events")

    assert schema["type"] == "record"
    assert schema["namespace"] == "com.events"
    assert schema["doc"] is not None

    fields = {f["name"]: f for f in schema["fields"]}

    assert fields["event_id"]["type"]["logicalType"] == "uuid"
    assert fields["event_type"]["type"]["type"] == "enum"
    assert fields["timestamp"]["type"]["logicalType"] == "timestamp-micros"
    assert isinstance(fields["user_id"]["type"], list)
    assert fields["payload"]["type"]["type"] == "map"
    assert "null" in fields["metadata"]["type"]
    assert fields["tags"]["default"] == []
    assert fields["coordinates"]["type"]["type"] == "array"


def test_avro_schema_evolution_compatibility(manager: ModelManager) -> None:
    """Test schema evolution maintains Avro compatibility rules."""

    @manager.model("Product", "1.0.0")
    class ProductV1(BaseModel):
        """Product catalog entry."""

        id: str
        name: str
        price: float

    @manager.model("Product", "2.0.0")
    class ProductV2(BaseModel):
        """Product catalog entry with optional fields."""

        id: str
        name: str
        price: float
        # New optional fields for forward compatibility
        description: str | None = None
        category: str | None = None
        in_stock: bool = True

    schema_v1 = manager.get_avro_schema("Product", "1.0.0", namespace="com.catalog")
    schema_v2 = manager.get_avro_schema("Product", "2.0.0", namespace="com.catalog")

    v1_fields = {f["name"] for f in schema_v1["fields"]}
    v2_fields = {f["name"]: f for f in schema_v2["fields"]}

    assert v1_fields.issubset(set(v2_fields.keys()))

    assert v2_fields["description"]["default"] is None
    assert v2_fields["category"]["default"] is None
    assert v2_fields["in_stock"]["default"] is True
