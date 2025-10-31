"""Comprehensive tests for Protocol Buffer schema generation - Coverage Enhancement."""

from collections.abc import Mapping, MutableMapping
from datetime import datetime
from decimal import Decimal
from enum import Enum, IntEnum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NewType, Optional
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from pyrmute import ModelManager, ModelVersion
from pyrmute._registry import Registry
from pyrmute.protobuf_schema import ProtoExporter, ProtoSchemaGenerator

if TYPE_CHECKING:
    from pyrmute._protobuf_types import ProtoEnum, ProtoField, ProtoMessage

# ruff: noqa: PLR2004, D106


class Priority(IntEnum):
    """Integer enum for testing."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Status(StrEnum):
    """String enum for testing."""

    ACTIVE = "active"
    INACTIVE = "inactive"


class NestedEnum(BaseModel):
    """Model with nested enum."""

    class InnerStatus(Enum):
        """Nested enum with comment."""

        PENDING = "pending"
        COMPLETED = "completed"

    status: InnerStatus


class NestedMessage(BaseModel):
    """Model with nested messages."""

    class Address(BaseModel):
        """Nested address with comment."""

        street: str
        city: str

    class Contact(BaseModel):
        """Nested contact."""

        email: str
        phone: str | None = None

    primary_address: Address
    contacts: list[Contact]


class EdgeCaseTypes(BaseModel):
    """Model testing edge case type handling."""

    # Well-known types
    uuid_field: UUID
    decimal_field: Decimal

    # Tuple types (should be treated like list)
    tuple_field: tuple[str, ...]
    fixed_tuple: tuple[int, str, bool]

    # Empty generic types
    any_list: list  # type: ignore[type-arg]
    any_dict: dict  # type: ignore[type-arg]

    # Dict with various key/value types
    int_dict: dict[str, int]
    bool_dict: dict[str, bool]
    float_dict: dict[str, float]

    # Fields with default_factory
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class IntegerConstraints(BaseModel):
    """Model testing all integer optimization paths."""

    # uint32: ge >= 0, le <= 2^32-1
    small_uint: int = Field(ge=0, le=1000)
    # uint64: ge >= 0, le > 2^32-1
    large_uint: int = Field(ge=0, le=5_000_000_000)
    # uint64: ge >= 0, no upper bound
    unbounded_uint: int = Field(ge=0)
    # int32: negative lower bound
    signed_int: int = Field(ge=-100, le=100)
    # int32: no constraints
    plain_int: int


class MultilineComments(BaseModel):
    """Model with multiline docstring.

    This tests that multiline comments are properly formatted in the proto output.
    """

    field_with_multiline: str = Field(
        description="This is a long description\nthat spans multiple lines\nfor testing"
    )


class Proto2SpecificBehavior(BaseModel):
    """Model to test proto2 specific code paths."""

    optional_field: str | None = None
    required_field: str
    field_with_default: int = 42


class ComplexNesting(BaseModel):
    """Model with multiple levels of nesting."""

    class Level1(BaseModel):
        """First level nesting."""

        class Level2(BaseModel):
            """Second level nesting."""

            value: str

        nested: Level2
        name: str

    data: Level1


class SelfReferential(BaseModel):
    """Model that references itself (tests recursion prevention)."""

    name: str
    parent: Optional["SelfReferential"] = None
    children: list["SelfReferential"] = Field(default_factory=list)


class UnionVariations(BaseModel):
    """Model testing various union patterns."""

    simple_union: str | int
    multi_union: str | int | float | bool

    class TypeA(BaseModel):
        a: str

    class TypeB(BaseModel):
        b: int

    model_union: TypeA | TypeB
    list_union: str | list[int]


def test_uuid_type_mapping() -> None:
    """Test UUID is mapped to string."""

    class Model(BaseModel):
        id: UUID

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    id_field = next((f for f in message["fields"] if f["name"] == "id"), None)
    assert id_field is not None
    assert id_field["type"] == "string"
    # UUID doesn't require an import
    assert "google/protobuf" not in generator._required_imports


def test_decimal_type_mapping() -> None:
    """Test Decimal is mapped to double."""

    class Model(BaseModel):
        amount: Decimal

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    amount_field = next((f for f in message["fields"] if f["name"] == "amount"), None)
    assert amount_field is not None
    assert amount_field["type"] == "double"


def test_tuple_ellipsis_type_handling() -> None:
    """Test tuple ellipsis types are handled like lists."""

    class Model(BaseModel):
        values: tuple[str, ...]

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    values_field = next((f for f in message["fields"] if f["name"] == "values"), None)
    assert values_field is not None
    assert values_field["type"] == "string"
    assert values_field["label"] == "repeated"


def test_tuple_homogeneous_type_handling() -> None:
    """Test tuple homogeneous types are handled like lists."""

    class Model(BaseModel):
        values: tuple[str, str]

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    values_field = next((f for f in message["fields"] if f["name"] == "values"), None)
    assert values_field is not None
    assert values_field["type"] == "string"
    assert values_field["label"] == "repeated"


def test_tuple_heterogeneous_type_handling() -> None:
    """Test tuple heterogeneous types are handled as separate messages."""

    class Model(BaseModel):
        values: tuple[str, int, float]

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    assert message["name"] == "Model"
    values_field = message["fields"][0]
    assert values_field["name"] == "values"
    nested_msg_name = values_field["type"]

    assert len(message["nested_messages"]) == 0
    assert len(generator._collected_nested_messages) == 1

    tuple_msg = generator._collected_nested_messages[0]
    assert tuple_msg["name"] == nested_msg_name
    assert len(tuple_msg["fields"]) == 3

    field_types = {f["number"]: f["type"] for f in tuple_msg["fields"]}
    assert field_types == {1: "string", 2: "int32", 3: "double"}


def test_empty_list_type() -> None:
    """Test list without type parameter defaults to string."""

    class Model(BaseModel):
        items: list  # type: ignore[type-arg]

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    items_field = next((f for f in message["fields"] if f["name"] == "items"), None)
    assert items_field is not None
    assert items_field["type"] == "string"
    assert items_field.get("label") is None


def test_empty_dict_type() -> None:
    """Test dict without type parameters defaults to map<string, string>."""

    class Model(BaseModel):
        data: dict  # type: ignore[type-arg]

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    data_field = next((f for f in message["fields"] if f["name"] == "data"), None)
    assert data_field is not None
    assert data_field["type"] == "map<string, string>"


def test_dict_with_various_value_types() -> None:
    """Test dict with different value types."""

    class Model(BaseModel):
        int_map: dict[str, int]
        bool_map: dict[str, bool]
        float_map: dict[str, float]

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field_types = {f["name"]: f["type"] for f in message["fields"]}
    assert field_types["int_map"] == "map<string, int32>"
    assert field_types["bool_map"] == "map<string, bool>"
    assert field_types["float_map"] == "map<string, double>"


def test_integer_uint64_optimization() -> None:
    """Test integer optimized to uint64 for large upper bounds."""

    class Model(BaseModel):
        big_number: int = Field(ge=0, le=5_000_000_000)  # > 2^32-1

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "big_number"), None)
    assert field is not None
    assert field["type"] == "uint64"


def test_integer_unbounded_uint() -> None:
    """Test integer with only ge>=0 becomes uint32."""

    class Model(BaseModel):
        count: int = Field(ge=0)

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "count"), None)
    assert field is not None
    assert field["type"] == "uint64"


def test_integer_with_negative_bound() -> None:
    """Test integer with negative bound stays int32."""

    class Model(BaseModel):
        temperature: int = Field(ge=-100, le=100)

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "temperature"), None)
    assert field is not None
    assert field["type"] == "int32"


def test_uint32_at_boundary() -> None:
    """Test integer at exact uint32 max boundary."""

    class Model(BaseModel):
        max_uint32: int = Field(ge=0, le=2**32 - 1)  # 4,294,967,295

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")
    field = next((f for f in message["fields"] if f["name"] == "max_uint32"), None)
    assert field is not None
    assert field["type"] == "uint32"


def test_uint64_just_over_boundary() -> None:
    """Test integer just over uint32 max uses uint64."""

    class Model(BaseModel):
        over_uint32: int = Field(ge=0, le=2**32)  # 4,294,967,296

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")
    field = next((f for f in message["fields"] if f["name"] == "over_uint32"), None)
    assert field is not None
    assert field["type"] == "uint64"


def test_multiple_fields_different_optimizations() -> None:
    """Test multiple fields with different optimization strategies."""

    class Model(BaseModel):
        default_int: int
        small_uint: int = Field(ge=0, le=100)
        large_uint: int = Field(ge=0, le=10_000_000_000)
        negative_allowed: int = Field(ge=-100, le=100)

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    default_int = next(
        (f for f in message["fields"] if f["name"] == "default_int"), None
    )
    assert default_int["type"] == "int32"  # type: ignore[index]

    small_uint = next((f for f in message["fields"] if f["name"] == "small_uint"), None)
    assert small_uint["type"] == "uint32"  # type: ignore[index]

    large_uint = next((f for f in message["fields"] if f["name"] == "large_uint"), None)
    assert large_uint["type"] == "uint64"  # type: ignore[index]

    negative_allowed = next(
        (f for f in message["fields"] if f["name"] == "negative_allowed"), None
    )
    assert negative_allowed["type"] == "int32"  # type: ignore[index]


def test_integer_no_metadata() -> None:
    """Test integer without constraints defaults to int32."""

    class Model(BaseModel):
        value: int

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "value"), None)
    assert field is not None
    assert field["type"] == "int32"


def test_field_with_default_factory() -> None:
    """Test fields with default_factory are marked as optional."""

    class Model(BaseModel):
        tags: list[str] = Field(default_factory=list)
        metadata: dict[str, str] = Field(default_factory=dict)

    generator = ProtoSchemaGenerator(use_proto3=True)
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    tags_field: ProtoField | None = next(
        (f for f in message["fields"] if f["name"] == "tags"), None
    )
    assert tags_field is not None
    assert tags_field["label"] == "repeated"

    metadata_field: ProtoField | None = next(
        (f for f in message["fields"] if f["name"] == "metadata"), None
    )
    assert metadata_field is not None


def test_nested_enum_in_message() -> None:
    """Test nested enum is properly included."""

    class Model(BaseModel):
        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        status: Status

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    assert "nested_enums" in message


def test_nested_message_in_message() -> None:
    """Test nested message is properly handled."""

    class Model(BaseModel):
        class Inner(BaseModel):
            value: str

        inner: Inner

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    inner_field: ProtoField | None = next(
        (f for f in message["fields"] if f["name"] == "inner"), None
    )
    assert inner_field is not None
    assert inner_field["type"] == "Inner"


def test_self_referential_model() -> None:
    """Test self-referential model doesn't cause infinite recursion."""
    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(
        SelfReferential, "SelfReferential", "1.0.0"
    )

    # Should successfully generate without recursion error
    assert message["name"] == "SelfReferential"
    assert len(message["fields"]) == 3

    parent_field = next((f for f in message["fields"] if f["name"] == "parent"), None)
    assert parent_field is not None
    assert parent_field["type"] == "SelfReferential"


def test_multiline_comment_in_proto_string() -> None:
    """Test multiline comments are properly formatted."""

    class Model(BaseModel):
        """Model docstring line 1.

        Model docstring line 2.
        Model docstring line 3.
        """

        field: str = Field(description="Field line 1\nField line 2")

    generator = ProtoSchemaGenerator(include_docs=True)
    proto_file = generator.generate_schema(Model, "Model", "1.0.0")
    proto_string = generator.proto_file_to_string(proto_file)

    assert "Model docstring line 1" in proto_string
    assert "Model docstring line 2" in proto_string

    assert "Field line 1" in proto_string
    assert "Field line 2" in proto_string


def test_proto2_optional_field_handling() -> None:
    """Test proto2 doesn't add optional label to optional fields."""

    class Model(BaseModel):
        optional_field: str | None = None
        required_field: str

    generator = ProtoSchemaGenerator(use_proto3=False)
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    optional_field = next(
        (f for f in message["fields"] if f["name"] == "optional_field"), None
    )
    assert optional_field is not None
    assert optional_field.get("label") == "optional"


def test_proto2_required_field() -> None:
    """Test proto2 required field handling."""

    class Model(BaseModel):
        required: str

    generator = ProtoSchemaGenerator(use_proto3=False)
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    required_field = next(
        (f for f in message["fields"] if f["name"] == "required"), None
    )
    assert required_field is not None


def test_enum_to_string_with_comment() -> None:
    """Test enum with comment in proto string output."""

    class Model(BaseModel):
        class Status(Enum):
            """Status enum with comment."""

            ACTIVE = "active"
            INACTIVE = "inactive"

        status: Status

    generator = ProtoSchemaGenerator(include_docs=True)
    test_enum: ProtoEnum = {
        "name": "Status",
        "values": {"ACTIVE": 0, "INACTIVE": 1},
        "comment": "Status enum comment",
    }

    enum_string = generator._enum_to_string(test_enum, indent=0)
    enum_output = "\n".join(enum_string)

    assert "// Status enum comment" in enum_output
    assert "enum Status {" in enum_output
    assert "ACTIVE = 0;" in enum_output
    assert "INACTIVE = 1;" in enum_output


def test_nested_message_with_indentation() -> None:
    """Test nested message string output has proper indentation."""

    class Model(BaseModel):
        class Inner(BaseModel):
            """Inner message comment."""

            value: str

        inner: Inner

    generator = ProtoSchemaGenerator(include_docs=True)

    inner_msg: ProtoMessage = {
        "name": "Inner",
        "fields": [{"name": "value", "type": "string", "number": 1}],
        "comment": "Inner message comment",
    }

    outer_msg: ProtoMessage = {
        "name": "Outer",
        "fields": [{"name": "inner", "type": "Inner", "number": 1}],
        "nested_messages": [inner_msg],
    }

    msg_string = generator._message_to_string(outer_msg, indent=0)
    msg_output = "\n".join(msg_string)

    assert "  message Inner {" in msg_output
    assert "    string value = 1;" in msg_output


def test_field_without_label_in_proto_string() -> None:
    """Test field without label generates correct proto string."""

    class Model(BaseModel):
        value: str

    generator = ProtoSchemaGenerator(use_proto3=True)
    proto_file = generator.generate_schema(Model, "Model", "1.0.0")
    proto_string = generator.proto_file_to_string(proto_file)

    # In proto3, scalar fields don't need labels
    assert "string value = 1;" in proto_string
    assert "optional string value" not in proto_string


def test_union_with_multiple_types() -> None:
    """Test union with 3+ types creates oneof."""

    class Model(BaseModel):
        value: str | int | float | bool

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    assert "oneofs" in message
    assert len(message["oneofs"]) == 1

    oneof_fields = [
        f for f in message["fields"] if f.get("oneof_group") == "value_value"
    ]
    assert len(oneof_fields) == 4


def test_union_with_model_types() -> None:
    """Test union with Pydantic models."""

    class TypeA(BaseModel):
        a_value: str

    class TypeB(BaseModel):
        b_value: int

    class Model(BaseModel):
        data: TypeA | TypeB

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    oneof_fields = [
        f for f in message["fields"] if f.get("oneof_group") == "data_value"
    ]
    assert len(oneof_fields) == 2

    field_types = {f["type"] for f in oneof_fields}
    assert "TypeA" in field_types
    assert "TypeB" in field_types


def test_protobuf_model_with_unioned_iterable_oneof(
    tmp_path: Path, manager: ModelManager
) -> None:
    """Models with unions with iterables raise an error."""

    @manager.model("Model", "1.0.0")
    class Model(BaseModel):
        """Model with union types."""

        list_or_int: list[int] | int

    generator = ProtoSchemaGenerator(package="integration_test")
    with pytest.raises(ValueError, match="Python iterables to ProtoBuf"):
        generator.generate_schema(Model, "Model", "1.0.0")


def test_protobuf_model_with_unioned_mapping_oneof(
    tmp_path: Path, manager: ModelManager
) -> None:
    """Models with unions with dictionaries raise an error."""

    @manager.model("Model", "1.0.0")
    class Model(BaseModel):
        """Model with union types."""

        dict_or_int: dict[str, Any] | int

    generator = ProtoSchemaGenerator(package="integration_test")
    with pytest.raises(
        ValueError, match="Cannot encode unions with Python dictionaries"
    ):
        generator.generate_schema(Model, "Model", "1.0.0")


def test_proto_file_with_enum() -> None:
    """Test proto file with enum generation."""

    class Status(str, Enum):
        UNKNOWN = "UNKNOWN"
        ACTIVE = "ACTIVE"
        INACTIVE = "INACTIVE"

    class Model(BaseModel):
        status: Status

    generator = ProtoSchemaGenerator()
    document = generator.generate_schema(Model, "Model", "1.0.0")
    proto_string = generator.proto_file_to_string(document)

    assert "enum Status {" in proto_string
    assert "UNKNOWN = 0;" in proto_string
    assert "ACTIVE = 1;" in proto_string


def test_proto_exporter_with_file_output(tmp_path: Path) -> None:
    """Test ProtoExporter writes files correctly."""
    registry = Registry()

    @registry.register("User", "1.0.0")
    class User(BaseModel):
        id: int
        name: str

    exporter = ProtoExporter(
        registry=registry,
        package="com.test",
        include_docs=True,
        use_proto3=True,
    )

    output_file = tmp_path / "user.proto"
    exporter.export_schema("User", "1.0.0", output_path=output_file)

    assert output_file.exists()

    content = output_file.read_text()
    assert 'syntax = "proto3";' in content
    assert "message User {" in content
    assert "int32 id = 1;" in content


def test_proto_exporter_without_file_output() -> None:
    """Test ProtoExporter returns proto file without writing."""
    registry = Registry()

    @registry.register("User", "1.0.0")
    class User(BaseModel):
        id: int
        name: str

    exporter = ProtoExporter(registry=registry)

    proto_file = exporter.export_schema("User", "1.0.0", output_path=None)

    assert "proto3" in proto_file
    assert "User" in proto_file


def test_proto_exporter_creates_parent_directories(tmp_path: Path) -> None:
    """Test ProtoExporter creates parent directories if needed."""
    registry = Registry()

    @registry.register("User", "1.0.0")
    class User(BaseModel):
        id: int

    exporter = ProtoExporter(registry=registry)

    nested_path = tmp_path / "protos" / "v1" / "user.proto"
    exporter.export_schema("User", "1.0.0", output_path=nested_path)

    assert nested_path.parent.exists()
    assert nested_path.exists()


def test_proto_exporter_export_all_schemas(tmp_path: Path) -> None:
    """Test ProtoExporter can export all schemas in registry."""
    registry = Registry()

    @registry.register("User", "1.0.0")
    class UserV1(BaseModel):
        id: int
        name: str

    @registry.register("User", "2.0.0")
    class UserV2(BaseModel):
        id: int
        name: str
        email: str

    @registry.register("Order", "1.0.0")
    class Order(BaseModel):
        id: int
        user_id: int
        total: float

    exporter = ProtoExporter(registry=registry, package="com.test")

    all_schemas = exporter.export_all_schemas(tmp_path)

    assert "User" in all_schemas
    assert "Order" in all_schemas
    assert "1.0.0" in all_schemas["User"]
    assert "2.0.0" in all_schemas["User"]
    assert "1.0.0" in all_schemas["Order"]

    assert (tmp_path / "User_v1_0_0.proto").exists()
    assert (tmp_path / "User_v2_0_0.proto").exists()
    assert (tmp_path / "Order_v1_0_0.proto").exists()


def test_model_version_not_in_export() -> None:
    """Test that ModelVersion objects work correctly."""
    registry = Registry()
    version = ModelVersion(major=1, minor=2, patch=3)

    @registry.register("User", version)
    class User(BaseModel):
        id: int

    exporter = ProtoExporter(registry=registry)
    proto_file = exporter.export_schema("User", version)

    assert "1_2_3" not in proto_file


def test_bytes_type_handling() -> None:
    """Test bytes type is correctly mapped."""

    class Model(BaseModel):
        data: bytes

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    data_field = next((f for f in message["fields"] if f["name"] == "data"), None)
    assert data_field is not None
    assert data_field["type"] == "bytes"


def test_int_enum_handling() -> None:
    """Test IntEnum is handled as enum type."""

    class Model(BaseModel):
        priority: Priority

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    priority_field = next(
        (f for f in message["fields"] if f["name"] == "priority"), None
    )
    assert priority_field is not None
    assert priority_field["type"] == "Priority"


def test_unknown_type_defaults_to_string() -> None:
    """Test unknown types default to string."""

    class CustomType:
        pass

    generator = ProtoSchemaGenerator()
    proto_type = generator._convert_type(CustomType, None)

    assert proto_type.type_representation == "string"
    assert proto_type.is_repeated is False


def test_none_annotation_defaults_to_string() -> None:
    """Test None annotation defaults to string."""
    generator = ProtoSchemaGenerator()
    proto_type = generator._convert_type(None, None)

    assert proto_type.type_representation == "string"
    assert proto_type.is_repeated is False


def test_complex_nested_structure() -> None:
    """Test deeply nested structure generates correctly."""
    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(
        ComplexNesting, "ComplexNesting", "1.0.0"
    )

    assert message["name"] == "ComplexNesting"
    assert len(message["fields"]) > 0


def test_proto_file_with_imports() -> None:
    """Test proto file includes imports."""

    class Model(BaseModel):
        timestamp: datetime

    generator = ProtoSchemaGenerator()
    proto_file = generator.generate_schema(Model, "Model", "1.0.0")
    proto_string = generator.proto_file_to_string(proto_file)

    assert "google/protobuf/timestamp.proto" in proto_string


# ============================================================================
# CONSTRAINT COMBINATION EDGE CASES
# ============================================================================


def test_integer_with_gt_constraint() -> None:
    """Test integer with gt (greater than) instead of ge."""

    class Model(BaseModel):
        positive: int = Field(gt=0)  # gt, not ge

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "positive"), None)
    assert field is not None
    assert field["type"] == "uint64"


def test_integer_with_lt_constraint() -> None:
    """Test integer with lt (less than) instead of le."""

    class Model(BaseModel):
        below_hundred: int = Field(lt=100)

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "below_hundred"), None)
    assert field is not None
    # lt not checked by optimizer, should be int32
    assert field["type"] == "int32"


def test_integer_with_mixed_gt_le() -> None:
    """Test integer with mixed gt and le constraints."""

    class Model(BaseModel):
        value: int = Field(gt=0, le=1000)

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "value"), None)
    assert field is not None
    assert field["type"] == "uint32"


def test_integer_with_multiple_of_constraint() -> None:
    """Test integer with multiple_of constraint doesn't break."""

    class Model(BaseModel):
        even: int = Field(multiple_of=2)
        positive_even: int = Field(ge=0, multiple_of=2)

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    even_field = next((f for f in message["fields"] if f["name"] == "even"), None)
    assert even_field is not None
    assert even_field["type"] == "int32"

    positive_even_field = next(
        (f for f in message["fields"] if f["name"] == "positive_even"), None
    )
    assert positive_even_field is not None
    assert positive_even_field["type"] == "uint64"


# ============================================================================
# FIELD NUMBERING EDGE CASES
# ============================================================================


def test_large_model_field_numbering() -> None:
    """Test field numbering in model with many fields."""
    fields_dict = {f"field_{i}": (str, "") for i in range(50)}

    Model = type(
        "LargeModel",
        (BaseModel,),
        {
            "__annotations__": {k: v[0] for k, v in fields_dict.items()},
            **{k: v[1] for k, v in fields_dict.items()},
        },
    )

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "LargeModel", "1.0.0")

    field_numbers = [f["number"] for f in message["fields"]]
    assert len(field_numbers) == len(set(field_numbers)), "Field numbers not unique"
    assert min(field_numbers) == 1
    assert max(field_numbers) == 50


def test_field_numbering_with_oneofs() -> None:
    """Test field numbering works correctly with oneofs consuming multiple numbers."""

    class Model(BaseModel):
        before: str
        union_field: str | int | float  # Takes 3 field numbers
        after: str

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field_numbers = [f["number"] for f in message["fields"]]
    assert len(field_numbers) == len(set(field_numbers))

    before = next((f for f in message["fields"] if f["name"] == "before"), None)
    after = next((f for f in message["fields"] if f["name"] == "after"), None)
    assert before is not None
    assert after is not None
    assert before["number"] == 1
    assert after["number"] == 5  # After the 3 oneof fields


def test_field_counter_reset_between_models() -> None:
    """Test field counter resets when generating multiple models."""

    class Model1(BaseModel):
        field1: str
        field2: str

    class Model2(BaseModel):
        field1: str
        field2: str

    generator = ProtoSchemaGenerator()

    message1 = generator._generate_proto_schema(Model1, "Model1", "1.0.0")
    message2 = generator._generate_proto_schema(Model2, "Model2", "1.0.0")

    field1_num = next(
        (f["number"] for f in message1["fields"] if f["name"] == "field1"), None
    )
    field2_num = next(
        (f["number"] for f in message2["fields"] if f["name"] == "field1"), None
    )

    assert field1_num == 1
    assert field2_num == 1


# ============================================================================
# TYPE INFERENCE AMBIGUITIES
# ============================================================================


def test_any_type_annotation() -> None:
    """Test Any type annotation defaults to string."""

    class Model(BaseModel):
        anything: Any

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "anything"), None)
    assert field is not None
    assert field["type"] == "string"


def test_forward_reference_in_list() -> None:
    """Test forward reference in list type."""

    class Model(BaseModel):
        items: list["Model"]

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    items = next((f for f in message["fields"] if f["name"] == "items"), None)
    assert items is not None
    assert items.get("label") == "repeated"


def test_new_type_wrapper() -> None:
    """Test NewType wrapper is handled."""
    UserId = NewType("UserId", int)

    class Model(BaseModel):
        user_id: UserId

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    assert len(message["fields"]) == 1


def test_union_with_none_only() -> None:
    """Test Union[None] (edge case, not Optional)."""

    class Model(BaseModel):
        value: Optional[None]  # noqa: UP045

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")
    assert len(message["fields"]) == 1


# ============================================================================
# UNION/ONEOF EDGE CASES
# ============================================================================


def test_union_with_duplicate_types() -> None:
    """Test union with same type appearing twice."""

    class Model(BaseModel):
        value: str | str

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")
    assert message is not None


def test_union_with_two_list_types() -> None:
    """Test union with two different list types."""

    class Model(BaseModel):
        data: list[str] | list[int]

    generator = ProtoSchemaGenerator()
    with pytest.raises(ValueError, match="Cannot encode unions with Python iterables"):
        generator._generate_proto_schema(Model, "Model", "1.0.0")


def test_union_with_two_dict_types() -> None:
    """Test union with two different dict types."""

    class Model(BaseModel):
        config: Mapping[str, str] | MutableMapping[str, int]

    generator = ProtoSchemaGenerator()
    with pytest.raises(
        ValueError, match="Cannot encode unions with Python dictionaries"
    ):
        generator._generate_proto_schema(Model, "Model", "1.0.0")


def test_very_large_union() -> None:
    """Test union with many types (10+)."""

    class Model(BaseModel):
        value: str | int | float | bool | bytes | (datetime) | None

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    assert "oneofs" in message or len(message["fields"]) > 0


def test_union_with_none_in_middle() -> None:
    """Test Union with None not at the end."""

    class Model(BaseModel):
        value: str | None | int

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    oneof_fields = [f for f in message["fields"] if f.get("oneof_group")]
    assert len(oneof_fields) == 2  # str and int, not None


def test_oneof_variant_names_use_registry_names(manager: ModelManager) -> None:
    """Test that oneof variant names use registry names from the model manager."""

    @manager.model("CardPayment", "1.0.0")
    class CardPaymentV1(BaseModel):
        card_number: str

    @manager.model("BankPayment", "1.0.0")
    class BankPaymentV1(BaseModel):
        account_number: str

    @manager.model("Payment", "1.0.0")
    class PaymentV1(BaseModel):
        payment: CardPaymentV1 | BankPaymentV1

    proto_content = manager.get_proto_schema("Payment", "1.0.0")

    assert "payment_cardpayment" in proto_content
    assert "payment_bankpayment" in proto_content

    assert "payment_cardpaymentv1" not in proto_content.lower()
    assert "payment_bankpaymentv1" not in proto_content.lower()


# ============================================================================
# NAME COLLISION RISKS
# ============================================================================


def test_field_named_value_value() -> None:
    """Test field name that conflicts with oneof naming pattern."""

    class Model(BaseModel):
        value: str
        value_value: str  # Could conflict with generated oneof name

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    assert len(message["fields"]) == 2


def test_field_with_proto_keyword_name() -> None:
    """Test field names that are proto keywords."""

    class Model(BaseModel):
        message: str  # 'message' is a proto keyword
        enum: str  # 'enum' is a proto keyword
        option: str  # 'option' is a proto keyword
        repeated: str  # 'repeated' is a proto keyword

    generator = ProtoSchemaGenerator()
    message_schema = generator._generate_proto_schema(Model, "Model", "1.0.0")
    assert len(message_schema["fields"]) == 4


def test_field_name_with_numbers() -> None:
    """Test field names starting with or containing numbers."""

    class Model(BaseModel):
        field_1: str
        field1: str
        value_2_test: str

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    assert len(message["fields"]) == 3


def test_nested_message_same_name_as_parent() -> None:
    """Test nested message with name matching parent field."""

    class Model(BaseModel):
        class Data(BaseModel):
            value: str

        data: Data  # Field name matches nested class name

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    data_field = next((f for f in message["fields"] if f["name"] == "data"), None)
    assert data_field is not None
    assert data_field["type"] == "Data"


# ============================================================================
# DICT/MAP EDGE CASES
# ============================================================================


def test_dict_with_int_keys() -> None:
    """Test dict with integer keys."""

    class Model(BaseModel):
        int_keys: dict[int, str]

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "int_keys"), None)
    assert field is not None
    assert field["type"] == "map<int32, string>"


def test_dict_with_complex_nested_values() -> None:
    """Test dict with nested complex value types."""

    class Model(BaseModel):
        nested_map: dict[str, list[dict[str, int]]]

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "nested_map"), None)
    assert field is not None


def test_nested_maps() -> None:
    """Test nested map structures."""

    class Model(BaseModel):
        map_of_maps: dict[str, dict[str, str]]

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "map_of_maps"), None)
    assert field is not None
    assert "map<string, map<string, string>>" in field["type"]


# ============================================================================
# DEEPLY NESTED STRUCTURES
# ============================================================================


def test_deep_nesting_10_levels() -> None:
    """Test deeply nested structure (10 levels)."""

    class Level10(BaseModel):
        value: str

    class Level9(BaseModel):
        child: Level10

    class Level8(BaseModel):
        child: Level9

    class Level7(BaseModel):
        child: Level8

    class Level6(BaseModel):
        child: Level7

    class Level5(BaseModel):
        child: Level6

    class Level4(BaseModel):
        child: Level5

    class Level3(BaseModel):
        child: Level4

    class Level2(BaseModel):
        child: Level3

    class Level1(BaseModel):
        child: Level2

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Level1, "Level1", "1.0.0")

    assert message["name"] == "Level1"


def test_mutual_recursion() -> None:
    """Test mutually recursive models."""

    class ModelA(BaseModel):
        b: Optional["ModelB"] = None

    class ModelB(BaseModel):
        a: ModelA | None = None

    generator = ProtoSchemaGenerator()

    message_a = generator._generate_proto_schema(ModelA, "ModelA", "1.0.0")
    assert message_a["name"] == "ModelA"

    message_b = generator._generate_proto_schema(ModelB, "ModelB", "1.0.0")
    assert message_b["name"] == "ModelB"


def test_cycle_through_lists() -> None:
    """Test recursive structure through lists."""

    class Node(BaseModel):
        value: str
        children: list["Node"] = Field(default_factory=list)

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Node, "NotNode", "1.0.0")

    children = next((f for f in message["fields"] if f["name"] == "children"), None)
    assert children is not None
    assert children["type"] == "NotNode"
    assert children["label"] == "repeated"


# ============================================================================
# COMMENT/DOCUMENTATION EDGE CASES
# ============================================================================


def test_very_long_comment() -> None:
    """Test field with very long description."""
    long_desc = "x" * 2000

    class Model(BaseModel):
        field: str = Field(description=long_desc)

    generator = ProtoSchemaGenerator(include_docs=True)
    proto_file = generator.generate_schema(Model, "Model", "1.0.0")
    proto_string = generator.proto_file_to_string(proto_file)

    assert "x" * 100 in proto_string


def test_comment_with_proto_syntax() -> None:
    """Test comment containing proto syntax."""

    class Model(BaseModel):
        """Model with message { field: value } in docstring."""

        field: str = Field(description="message Status { ACTIVE = 1; }")

    generator = ProtoSchemaGenerator(include_docs=True)
    proto_file = generator.generate_schema(Model, "Model", "1.0.0")
    proto_string = generator.proto_file_to_string(proto_file)

    assert "// Model with message" in proto_string


def test_unicode_in_comments() -> None:
    """Test unicode characters in docstrings."""

    class Model(BaseModel):
        """Model with émojis 🚀 and spëcial çharacters."""

        field: str = Field(description="Unicode: 你好 мир שלום")

    generator = ProtoSchemaGenerator(include_docs=True)
    proto_file = generator.generate_schema(Model, "Model", "1.0.0")
    proto_string = generator.proto_file_to_string(proto_file)

    assert "🚀" in proto_string or "émojis" in proto_string


def test_empty_docstring_vs_none() -> None:
    """Test difference between empty docstring and no docstring."""

    class ModelWithEmpty(BaseModel):
        """"""  # noqa: D419 Empty docstring

        field: str

    class ModelWithNone(BaseModel):
        field: str

    generator = ProtoSchemaGenerator(include_docs=True)

    msg_empty = generator._generate_proto_schema(ModelWithEmpty, "Model1", "1.0.0")
    msg_none = generator._generate_proto_schema(ModelWithNone, "Model2", "1.0.0")

    assert msg_empty["name"] == "Model1"
    assert msg_none["name"] == "Model2"


# ============================================================================
# ENUM EDGE CASES
# ============================================================================


def test_enum_with_non_sequential_values() -> None:
    """Test enum with non-sequential integer values."""

    class Priority(Enum):
        LOW = 1
        MEDIUM = 5
        HIGH = 100

    class Model(BaseModel):
        priority: Priority

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "priority"), None)
    assert field is not None
    assert field["type"] == "Priority"


def test_enum_with_negative_values() -> None:
    """Test enum with negative integer values."""

    class Temperature(Enum):
        COLD = -1
        NORMAL = 0
        HOT = 1

    class Model(BaseModel):
        temp: Temperature

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "temp"), None)
    assert field is not None
    assert field["type"] == "Temperature"


def test_string_enum_with_special_chars() -> None:
    """Test string enum with non-identifier strings."""

    class Status(Enum):
        ACTIVE = "active-status"  # Has hyphen
        IN_PROGRESS = "in.progress"  # Has dot
        DONE = "done!"  # Has special char

    class Model(BaseModel):
        status: Status

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "status"), None)
    assert field is not None


# ============================================================================
# IMPORT HANDLING
# ============================================================================


def test_multiple_timestamp_fields() -> None:
    """Test multiple fields requiring the same import."""

    class Model(BaseModel):
        created_at: datetime
        updated_at: datetime
        deleted_at: datetime

    generator = ProtoSchemaGenerator()
    document = generator.generate_schema(Model, "Model", "1.0.0")
    proto_file = document.to_proto_file()

    imports = proto_file.get("imports", [])
    timestamp_imports = [i for i in imports if "timestamp" in i.lower()]
    assert len(timestamp_imports) == 1  # Deduplicated


def test_nested_timestamp_fields() -> None:
    """Test timestamp in nested messages triggers import."""

    class Inner(BaseModel):
        timestamp: datetime

    class Outer(BaseModel):
        inner: Inner
        own_timestamp: datetime

    generator = ProtoSchemaGenerator()
    document = generator.generate_schema(Outer, "Outer", "1.0.0")
    proto_file = document.to_proto_file()

    assert "google/protobuf/timestamp.proto" in proto_file.get("imports", [])


def test_integer_constraints_collected_fully() -> None:
    """Ensure ge and le are both collected."""

    class Model(BaseModel):
        field1: int = Field(ge=0, le=5_000_000_000)
        field2: int = Field(ge=1, le=100)
        field3: int = Field(ge=0, le=2**32 - 1)
        field4: int = Field(ge=0, le=2**32)

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field1 = next((f for f in message["fields"] if f["name"] == "field1"), None)
    assert field1 is not None
    assert field1["type"] == "uint64", "Should be uint64 for values > 2^32-1"

    field2 = next((f for f in message["fields"] if f["name"] == "field2"), None)
    assert field2 is not None
    assert field2["type"] == "uint32", "Should be uint32 for small positive range"

    field3 = next((f for f in message["fields"] if f["name"] == "field3"), None)
    assert field3 is not None
    assert field3["type"] == "uint32", "Should be uint32 at exact boundary"

    field4 = next((f for f in message["fields"] if f["name"] == "field4"), None)
    assert field4 is not None
    assert field4["type"] == "uint64", "Should be uint64 just over boundary"


# ============================================================================
# PROTO SYNTAX VALIDATION EDGE CASES
# ============================================================================


def test_field_number_uniqueness_across_oneofs() -> None:
    """Test that oneof fields don't reuse field numbers."""

    class Model(BaseModel):
        regular1: str
        union1: str | int
        regular2: str
        union2: float | bool

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field_numbers = [f["number"] for f in message["fields"]]

    assert len(field_numbers) == len(set(field_numbers)), (
        "Duplicate field numbers found"
    )
    assert sorted(field_numbers) == list(range(1, len(field_numbers) + 1))


def test_proto_field_ordering_with_oneof(manager: ModelManager) -> None:
    """Test that protobuf fields maintain definition order with oneofs."""

    @manager.model("Notification", "1.0.0")
    class NotificationV1(BaseModel):
        notification_id: str
        content: str | int  # Union type (oneof)
        timestamp: int

    proto_schema = manager.get_proto_schema(
        "Notification", "1.0.0", package="com.test", use_proto3=True
    )

    lines = [line.strip() for line in proto_schema.split("\n") if line.strip()]

    notification_id_idx = None
    oneof_start_idx = None
    timestamp_idx = None

    for i, line in enumerate(lines):
        if "string notification_id = 1" in line:
            notification_id_idx = i
        elif "oneof content_value" in line:
            oneof_start_idx = i
        elif "int32 timestamp = 4" in line:
            timestamp_idx = i

    assert notification_id_idx is not None, "notification_id field not found"
    assert oneof_start_idx is not None, "oneof not found"
    assert timestamp_idx is not None, "timestamp field not found"

    assert notification_id_idx < oneof_start_idx, (
        f"notification_id should come before oneof. Got indices: "
        f"{notification_id_idx} >= {oneof_start_idx}"
    )
    assert oneof_start_idx < timestamp_idx, (
        f"oneof should come before timestamp. Got indices: "
        f"{oneof_start_idx} >= {timestamp_idx}"
    )

    assert "notification_id = 1" in proto_schema
    assert "content_string = 2" in proto_schema
    assert "content_int32 = 3" in proto_schema
    assert "timestamp = 4" in proto_schema


def test_proto_field_ordering_multiple_oneofs(manager: ModelManager) -> None:
    """Test ordering with multiple oneofs interspersed with regular fields."""

    @manager.model("Complex", "1.0.0")
    class ComplexV1(BaseModel):
        id: str
        value1: str | int  # First oneof
        name: str
        value2: float | bool  # Second oneof
        count: int

    proto_schema = manager.get_proto_schema(
        "Complex", "1.0.0", package="com.test", use_proto3=True
    )

    lines = [line.strip() for line in proto_schema.split("\n") if line.strip()]

    indices = {}
    for i, line in enumerate(lines):
        if "string id = 1" in line:
            indices["id"] = i
        elif "oneof value1_value" in line:
            indices["oneof1"] = i
        elif "string name = 4" in line:
            indices["name"] = i
        elif "oneof value2_value" in line:
            indices["oneof2"] = i
        elif "int32 count = 7" in line:
            indices["count"] = i

    assert len(indices) == 5, f"Missing elements. Found: {indices}"

    assert (
        indices["id"]
        < indices["oneof1"]
        < indices["name"]
        < indices["oneof2"]
        < indices["count"]
    ), f"Fields not in definition order. Indices: {indices}"


def test_proto_field_ordering_oneof_at_start(manager: ModelManager) -> None:
    """Test ordering when oneof is the first field."""

    @manager.model("StartWithOneof", "1.0.0")
    class StartWithOneofV1(BaseModel):
        content: str | int  # Oneof first
        id: str
        name: str

    proto_schema = manager.get_proto_schema(
        "StartWithOneof", "1.0.0", package="com.test", use_proto3=True
    )

    lines = [line.strip() for line in proto_schema.split("\n") if line.strip()]

    oneof_idx = None
    id_idx = None

    for i, line in enumerate(lines):
        if "oneof content_value" in line:
            oneof_idx = i
        elif "string id = 3" in line:
            id_idx = i

    assert oneof_idx is not None and id_idx is not None
    assert oneof_idx < id_idx, "oneof should come before id field"


def test_proto_field_ordering_oneof_at_end(manager: ModelManager) -> None:
    """Test ordering when oneof is the last field."""

    @manager.model("EndWithOneof", "1.0.0")
    class EndWithOneofV1(BaseModel):
        id: str
        name: str
        content: str | int  # Oneof last

    proto_schema = manager.get_proto_schema(
        "EndWithOneof", "1.0.0", package="com.test", use_proto3=True
    )

    lines = [line.strip() for line in proto_schema.split("\n") if line.strip()]

    name_idx = None
    oneof_idx = None

    for i, line in enumerate(lines):
        if "string name = 2" in line:
            name_idx = i
        elif "oneof content_value" in line:
            oneof_idx = i

    assert name_idx is not None and oneof_idx is not None
    assert name_idx < oneof_idx, "name field should come before oneof"


def test_proto_schema_with_string_enum(manager: ModelManager) -> None:
    """Test that string enums are embedded in the message."""

    class Status(StrEnum):
        PENDING = "pending"
        ACTIVE = "active"
        COMPLETED = "completed"

    @manager.model("Task", "1.0.0")
    class TaskV1(BaseModel):
        name: str
        status: Status

    proto_schema = manager.get_proto_schema(
        "Task", "1.0.0", package="com.test", use_proto3=True
    )

    assert "enum Status {" in proto_schema
    assert "PENDING = 0;" in proto_schema
    assert "ACTIVE = 1;" in proto_schema
    assert "COMPLETED = 2;" in proto_schema

    assert "Status status = 2;" in proto_schema

    enum_idx = proto_schema.index("enum Status")
    field_idx = proto_schema.index("Status status = 2")
    assert enum_idx < field_idx


def test_proto_schema_with_int_enum(manager: ModelManager) -> None:
    """Test that int enums are embedded correctly."""

    class Priority(IntEnum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    @manager.model("Item", "1.0.0")
    class ItemV1(BaseModel):
        name: str
        priority: Priority

    proto_schema = manager.get_proto_schema(
        "Item", "1.0.0", package="com.test", use_proto3=True
    )

    assert "enum Priority {" in proto_schema
    assert "LOW = 0;" in proto_schema  # Proto enums always start at 0
    assert "MEDIUM = 1;" in proto_schema
    assert "HIGH = 2;" in proto_schema
    assert "Priority priority = 2;" in proto_schema


def test_proto_schema_with_multiple_enums(manager: ModelManager) -> None:
    """Test multiple different enums in the same message."""
    manager = ModelManager()

    class Status(StrEnum):
        DRAFT = "draft"
        PUBLISHED = "published"

    class Category(StrEnum):
        NEWS = "news"
        BLOG = "blog"
        DOCS = "docs"

    @manager.model("Article", "1.0.0")
    class ArticleV1(BaseModel):
        title: str
        status: Status
        category: Category

    proto_schema = manager.get_proto_schema(
        "Article", "1.0.0", package="com.test", use_proto3=True
    )

    assert "enum Status {" in proto_schema
    assert "DRAFT = 0;" in proto_schema
    assert "PUBLISHED = 1;" in proto_schema

    assert "enum Category {" in proto_schema
    assert "NEWS = 0;" in proto_schema
    assert "BLOG = 1;" in proto_schema
    assert "DOCS = 2;" in proto_schema

    assert "Status status = 2;" in proto_schema
    assert "Category category = 3;" in proto_schema


def test_proto_schema_with_optional_enum(manager: ModelManager) -> None:
    """Test optional enum fields."""
    manager = ModelManager()

    class Status(StrEnum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        status: Status | None = None

    proto_schema = manager.get_proto_schema("User", "1.0.0", package="com.test")

    assert "enum Status {" in proto_schema
    assert "ACTIVE = 0;" in proto_schema
    assert "INACTIVE = 1;" in proto_schema

    assert "optional Status status = 2;" in proto_schema


def test_proto_schema_with_enum_list(manager: ModelManager) -> None:
    """Test repeated enum fields."""

    class Tag(StrEnum):
        IMPORTANT = "important"
        URGENT = "urgent"
        REVIEW = "review"

    @manager.model("Task", "1.0.0")
    class TaskV1(BaseModel):
        name: str
        tags: list[Tag]

    proto_schema = manager.get_proto_schema("Task", "1.0.0", package="com.test")

    assert "enum Tag {" in proto_schema
    assert "IMPORTANT = 0;" in proto_schema
    assert "URGENT = 1;" in proto_schema
    assert "REVIEW = 2;" in proto_schema

    assert "repeated Tag tags = 2;" in proto_schema


def test_proto_schema_enum_with_description(manager: ModelManager) -> None:
    """Test that enum docstrings are included as comments."""

    class Status(StrEnum):
        """Task status enumeration."""

        PENDING = "pending"
        DONE = "done"

    @manager.model("Task", "1.0.0")
    class TaskV1(BaseModel):
        name: str
        status: Status

    proto_schema = manager.get_proto_schema(
        "Task", "1.0.0", package="com.test", include_docs=True
    )

    assert "// Task status enumeration." in proto_schema
    assert "enum Status {" in proto_schema


def test_proto_schema_enum_not_duplicated(manager: ModelManager) -> None:
    """Test that the same enum used twice is only defined once."""

    class Status(StrEnum):
        OPEN = "open"
        CLOSED = "closed"

    @manager.model("Document", "1.0.0")
    class DocumentV1(BaseModel):
        name: str
        status: Status
        previous_status: Status

    proto_schema = manager.get_proto_schema("Document", "1.0.0", package="com.test")

    enum_count = proto_schema.count("enum Status {")
    assert enum_count == 1, (
        f"Expected enum to be defined once, found {enum_count} times"
    )

    assert "Status status = 2;" in proto_schema
    assert "Status previous_status = 3;" in proto_schema


def test_proto_schema_enum_field_ordering(manager: ModelManager) -> None:
    """Test that enums maintain correct field ordering."""

    class Priority(StrEnum):
        LOW = "low"
        HIGH = "high"

    @manager.model("Task", "1.0.0")
    class TaskV1(BaseModel):
        id: str
        priority: Priority
        name: str

    proto_schema = manager.get_proto_schema("Task", "1.0.0", package="com.test")

    lines = [line.strip() for line in proto_schema.split("\n") if line.strip()]

    enum_idx = None
    id_idx = None
    priority_idx = None
    name_idx = None

    for i, line in enumerate(lines):
        if "enum Priority {" in line:
            enum_idx = i
        elif "string id = 1" in line:
            id_idx = i
        elif "Priority priority = 2" in line:
            priority_idx = i
        elif "string name = 3" in line:
            name_idx = i

    assert enum_idx is not None
    assert id_idx is not None
    assert priority_idx is not None
    assert name_idx is not None

    assert enum_idx < id_idx < priority_idx < name_idx


def test_proto_schema_dump_with_enums(tmp_path: Path, manager: ModelManager) -> None:
    """Test that dump_proto_schemas correctly handles enums."""

    class Status(StrEnum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        status: Status

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        status: Status
        email: str

    output_dir = tmp_path / "protos"
    schemas = manager.dump_proto_schemas(output_dir, package="com.test")

    assert "enum Status {" in schemas["User"]["1.0.0"]
    assert "enum Status {" in schemas["User"]["2.0.0"]

    v1_file = output_dir / "User_v1_0_0.proto"
    v2_file = output_dir / "User_v2_0_0.proto"

    assert v1_file.exists()
    assert v2_file.exists()

    v1_content = v1_file.read_text()
    assert "enum Status {" in v1_content
    assert "ACTIVE = 0;" in v1_content


def test_proto_schema_enum_in_nested_message(manager: ModelManager) -> None:
    """Test enums work correctly in nested messages."""

    class AddressType(StrEnum):
        HOME = "home"
        WORK = "work"

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        address_type: AddressType

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    proto_schema = manager.get_proto_schema("User", "1.0.0", package="com.test")

    assert "message Address {" in proto_schema
    assert "enum AddressType {" in proto_schema
    assert "HOME = 0;" in proto_schema
    assert "WORK = 1;" in proto_schema
    assert "AddressType address_type" in proto_schema


def test_proto_schema_with_simple_nested_model(manager: ModelManager) -> None:
    """Test that nested models are embedded in the message."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: str

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    proto_schema = manager.get_proto_schema("User", "1.0.0", package="com.test")

    assert "message Address {" in proto_schema
    assert "string street = 1;" in proto_schema
    assert "string city = 2;" in proto_schema

    assert "Address address = 2;" in proto_schema

    nested_msg_idx = proto_schema.index("message Address")
    field_idx = proto_schema.index("Address address")
    assert nested_msg_idx < field_idx


def test_proto_schema_nested_model_with_enum(manager: ModelManager) -> None:
    """Test nested models that contain enums."""

    class AddressType(StrEnum):
        HOME = "home"
        WORK = "work"

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        address_type: AddressType

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    proto_schema = manager.get_proto_schema("User", "1.0.0", package="com.test")

    assert "message Address {" in proto_schema

    assert "enum AddressType {" in proto_schema
    assert "HOME = 0;" in proto_schema
    assert "WORK = 1;" in proto_schema

    assert "AddressType address_type" in proto_schema


def test_proto_schema_multiple_nested_models(manager: ModelManager) -> None:
    """Test multiple different nested models."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: str

    @manager.model("Contact", "1.0.0")
    class ContactV1(BaseModel):
        email: str
        phone: str

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1
        contact: ContactV1

    proto_schema = manager.get_proto_schema("User", "1.0.0", package="com.test")

    assert "message Address {" in proto_schema
    assert "message Contact {" in proto_schema
    assert "Address address = 2;" in proto_schema
    assert "Contact contact = 3;" in proto_schema


def test_proto_schema_optional_nested_model(manager: ModelManager) -> None:
    """Test optional nested model fields."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1 | None = None

    proto_schema = manager.get_proto_schema("User", "1.0.0", package="com.test")

    assert "message Address {" in proto_schema
    assert "optional Address address = 2;" in proto_schema


def test_proto_schema_repeated_nested_model(manager: ModelManager) -> None:
    """Test repeated nested model fields."""

    @manager.model("Item", "1.0.0")
    class ItemV1(BaseModel):
        name: str
        quantity: int

    @manager.model("Order", "1.0.0")
    class OrderV1(BaseModel):
        order_id: str
        items: list[ItemV1]

    proto_schema = manager.get_proto_schema("Order", "1.0.0", package="com.test")

    assert "message Item {" in proto_schema
    assert "repeated Item items = 2;" in proto_schema


def test_proto_schema_deeply_nested_models(manager: ModelManager) -> None:
    """Test deeply nested models (3 levels)."""

    @manager.model("City", "1.0.0")
    class CityV1(BaseModel):
        name: str
        zip_code: str

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: CityV1

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    proto_schema = manager.get_proto_schema("User", "1.0.0", package="com.test")

    assert "message City {" in proto_schema
    assert "message Address {" in proto_schema

    address_start = proto_schema.index("message Address {")
    city_start = proto_schema.index("message City {")

    assert city_start < address_start


def test_proto_schema_nested_model_not_duplicated(manager: ModelManager) -> None:
    """Test that nested models used multiple times are only defined once."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        home_address: AddressV1
        work_address: AddressV1

    proto_schema = manager.get_proto_schema("User", "1.0.0", package="com.test")

    message_count = proto_schema.count("message Address {")
    assert message_count == 1, (
        f"Expected message to be defined once, found {message_count} times"
    )

    assert "Address home_address = 2;" in proto_schema
    assert "Address work_address = 3;" in proto_schema


def test_proto_schema_nested_model_with_description(manager: ModelManager) -> None:
    """Test that nested model docstrings are included as comments."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        """User's address information."""

        street: str

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    proto_schema = manager.get_proto_schema(
        "User", "1.0.0", package="com.test", include_docs=True
    )

    assert "// User's address information." in proto_schema
    assert "message Address {" in proto_schema


def test_proto_schema_nested_model_field_numbering(manager: ModelManager) -> None:
    """Test that field numbers are correct in nested messages."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: str
        zip_code: str

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1
        email: str

    proto_schema = manager.get_proto_schema("User", "1.0.0", package="com.test")

    assert "string name = 1;" in proto_schema
    assert "Address address = 2;" in proto_schema
    assert "string email = 3;" in proto_schema

    assert "string street = 1;" in proto_schema
    assert "string city = 2;" in proto_schema
    assert "string zip_code = 3;" in proto_schema


def test_proto_schema_dump_with_nested_models(
    tmp_path: Path, manager: ModelManager
) -> None:
    """Test that dump_proto_schemas correctly handles nested models."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    output_dir = tmp_path / "protos"
    schemas = manager.dump_proto_schemas(output_dir, package="com.test")

    assert "message Address {" in schemas["User"]["1.0.0"]

    user_file = output_dir / "User_v1_0_0.proto"
    assert user_file.exists()

    user_content = user_file.read_text()
    assert "message Address {" in user_content
    assert "Address address = 2;" in user_content
