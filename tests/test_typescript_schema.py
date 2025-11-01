"""Tests for TypeScript schema generation from Pydantic models."""

from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum, IntEnum, StrEnum
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeVar
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, Field, computed_field, conint, constr

from pyrmute._registry import Registry
from pyrmute.typescript_schema import (
    TypeScriptConfig,
    TypeScriptExporter,
    TypeScriptSchemaGenerator,
)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


@pytest.fixture
def generator() -> TypeScriptSchemaGenerator:
    """Create a TypeScript schema generator."""
    return TypeScriptSchemaGenerator(style="interface")


@pytest.fixture
def zod_generator() -> TypeScriptSchemaGenerator:
    """Create a Zod schema generator."""
    return TypeScriptSchemaGenerator(style="zod")


@pytest.fixture
def type_generator() -> TypeScriptSchemaGenerator:
    """Create a TypeScript type alias generator."""
    return TypeScriptSchemaGenerator(style="type")


# ============================================================================
# Basic Type Tests
# ============================================================================


def test_typescript_interface_basic_types(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript interface generation for basic Python types."""

    class BasicTypes(BaseModel):
        name: str
        age: int
        height: float
        is_active: bool
        data: bytes

    module = generator.generate_schema(BasicTypes, "BasicTypes", "1.0.0")
    schema = module.to_string()

    assert "export interface BasicTypes {" in schema
    assert "name: string;" in schema
    assert "age: number;" in schema
    assert "height: number;" in schema
    assert "is_active: boolean;" in schema
    assert "data: string;" in schema


def test_typescript_type_alias_basic_types(
    type_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript type alias generation for basic Python types."""

    class BasicTypes(BaseModel):
        name: str
        age: int
        is_active: bool

    module = type_generator.generate_schema(BasicTypes, "BasicTypes", "1.0.0")
    schema = module.to_string()

    assert "export type BasicTypes = {" in schema
    assert "name: string;" in schema
    assert "age: number;" in schema
    assert "is_active: boolean;" in schema


def test_typescript_zod_basic_types(zod_generator: TypeScriptSchemaGenerator) -> None:
    """Test Zod schema generation for basic Python types."""

    class BasicTypes(BaseModel):
        name: str
        age: int
        height: float
        is_active: bool

    module = zod_generator.generate_schema(BasicTypes, "BasicTypes", "1.0.0")
    schema = module.to_string()

    assert "import { z } from 'zod';" in schema
    assert "export const BasicTypesSchema = z.object({" in schema
    assert "name: z.string()," in schema
    assert "age: z.number().int()," in schema
    assert "height: z.number()," in schema
    assert "is_active: z.boolean()," in schema
    assert "export type BasicTypes = z.infer<typeof BasicTypesSchema>;" in schema


# ============================================================================
# Optional Field Tests
# ============================================================================


def test_typescript_interface_optional_fields(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface generation for optional fields."""

    class User(BaseModel):
        name: str
        email: str | None = None
        age: int | None = None

    module = generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "name: string;" in schema
    assert "email?: string;" in schema
    assert "age?: number;" in schema


def test_typescript_zod_optional_fields(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema generation for optional fields."""

    class User(BaseModel):
        name: str
        email: str | None = None
        age: int | None = None

    module = zod_generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "name: z.string()," in schema
    assert "email: z.string().optional()," in schema
    assert "age: z.number().int().optional()," in schema


def test_typescript_interface_with_defaults(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface generation for fields with default values."""

    class Config(BaseModel):
        timeout: int = 30
        retry_count: int = 3
        enabled: bool = True
        name: str = "default"

    module = generator.generate_schema(Config, "Config", "1.0.0")
    schema = module.to_string()

    assert "timeout?: number;" in schema
    assert "retry_count?: number;" in schema
    assert "enabled?: boolean;" in schema
    assert "name?: string;" in schema


# ============================================================================
# Documentation Tests
# ============================================================================


def test_typescript_interface_with_documentation(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface includes model and field documentation."""

    class User(BaseModel):
        """User account information."""

        name: str = Field(description="User's full name")
        age: int = Field(description="User's age in years")
        email: str = Field(description="Email address")

    module = generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "/**" in schema
    assert " * User account information." in schema
    assert " */" in schema
    assert "/** User's full name */" in schema
    assert "/** User's age in years */" in schema
    assert "/** Email address */" in schema


# ============================================================================
# Date/Time Type Tests
# ============================================================================


def test_typescript_interface_datetime_as_string(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface uses string for datetime by default."""

    class Event(BaseModel):
        created_at: datetime
        scheduled_date: date
        scheduled_time: time

    module = generator.generate_schema(Event, "Event", "1.0.0")
    schema = module.to_string()

    assert "created_at: string;" in schema
    assert "scheduled_date: string;" in schema
    assert "scheduled_time: string;" in schema


def test_typescript_interface_datetime_as_timestamp() -> None:
    """Test TypeScript interface uses number for datetime with timestamp config."""

    class Event(BaseModel):
        created_at: datetime
        scheduled_date: date

    config = TypeScriptConfig(date_format="timestamp")
    generator = TypeScriptSchemaGenerator(style="interface", config=config)
    module = generator.generate_schema(Event, "Event", "1.0.0")
    schema = module.to_string()

    assert "created_at: number;" in schema
    assert "scheduled_date: number;" in schema


def test_typescript_zod_datetime_validators(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema uses datetime validators."""

    class Event(BaseModel):
        created_at: datetime
        scheduled_date: date
        scheduled_time: time

    module = zod_generator.generate_schema(Event, "Event", "1.0.0")
    schema = module.to_string()

    assert "created_at: z.string().datetime()," in schema
    assert "scheduled_date: z.string().date()," in schema
    assert "scheduled_time: z.string().time()," in schema


def test_typescript_interface_uuid(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript interface uses string for UUID."""

    class Resource(BaseModel):
        id: UUID

    module = generator.generate_schema(Resource, "Resource", "1.0.0")
    schema = module.to_string()

    assert "id: string;" in schema


def test_typescript_zod_uuid_validator(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema uses UUID validator."""

    class Resource(BaseModel):
        id: UUID

    module = zod_generator.generate_schema(Resource, "Resource", "1.0.0")
    schema = module.to_string()

    assert "id: z.string().uuid()," in schema


def test_typescript_interface_decimal(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript interface uses number for Decimal."""

    class Price(BaseModel):
        amount: Decimal

    module = generator.generate_schema(Price, "Price", "1.0.0")
    schema = module.to_string()

    assert "amount: number;" in schema


def test_typescript_interface_optional_datetime(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface for optional datetime fields."""

    class Event(BaseModel):
        name: str
        timestamp: datetime | None = None

    module = generator.generate_schema(Event, "Event", "1.0.0")
    schema = module.to_string()

    assert "name: string;" in schema
    assert "timestamp?: string;" in schema


def test_typescript_zod_optional_datetime(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema for optional datetime fields."""

    class Event(BaseModel):
        name: str
        timestamp: datetime | None = None

    module = zod_generator.generate_schema(Event, "Event", "1.0.0")
    schema = module.to_string()

    assert "name: z.string()," in schema
    assert "timestamp: z.string().datetime().optional()," in schema


# ============================================================================
# Collection Type Tests
# ============================================================================


def test_typescript_interface_list_types(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript interface generation for list types."""

    class Container(BaseModel):
        tags: list[str]
        scores: list[int]
        metadata: list[dict[str, str]]

    module = generator.generate_schema(Container, "Container", "1.0.0")
    schema = module.to_string()

    assert "tags: string[];" in schema
    assert "scores: number[];" in schema
    assert "metadata: Record<string, string>[];" in schema


def test_typescript_interface_dict_types(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript interface generation for dict types."""

    class Container(BaseModel):
        metadata: dict[str, str]
        counts: dict[str, int]
        nested: dict[str, dict[str, int]]

    module = generator.generate_schema(Container, "Container", "1.0.0")
    schema = module.to_string()

    assert "metadata: Record<string, string>;" in schema
    assert "counts: Record<string, number>;" in schema
    assert "nested: Record<string, Record<string, number>>;" in schema


def test_typescript_zod_list_types(zod_generator: TypeScriptSchemaGenerator) -> None:
    """Test Zod schema generation for list types."""

    class Container(BaseModel):
        tags: list[str]
        scores: list[int]

    module = zod_generator.generate_schema(Container, "Container", "1.0.0")
    schema = module.to_string()

    assert "tags: z.array(z.string())," in schema
    assert "scores: z.array(z.number().int())," in schema


def test_typescript_zod_dict_types(zod_generator: TypeScriptSchemaGenerator) -> None:
    """Test Zod schema generation for dict types."""

    class Container(BaseModel):
        metadata: dict[str, str]
        counts: dict[str, int]

    module = zod_generator.generate_schema(Container, "Container", "1.0.0")
    schema = module.to_string()

    assert "metadata: z.record(z.string())," in schema
    assert "counts: z.record(z.number().int())," in schema


def test_typescript_interface_set_types(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript interface converts set to array."""

    class Container(BaseModel):
        unique_tags: set[str]
        unique_ids: set[int]

    module = generator.generate_schema(Container, "Container", "1.0.0")
    schema = module.to_string()

    assert "unique_tags: string[];" in schema
    assert "unique_ids: number[];" in schema


def test_typescript_interface_tuple_types(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript interface generation for tuple types."""

    class Container(BaseModel):
        coordinates: tuple[float, float]
        point: tuple[int, int, int]

    module = generator.generate_schema(Container, "Container", "1.0.0")
    schema = module.to_string()

    assert "coordinates: [number, number];" in schema
    assert "point: [number, number, number];" in schema


def test_typescript_zod_tuple_types(zod_generator: TypeScriptSchemaGenerator) -> None:
    """Test Zod schema generation for tuple types."""

    class Container(BaseModel):
        coordinates: tuple[float, float]

    module = zod_generator.generate_schema(Container, "Container", "1.0.0")
    schema = module.to_string()

    assert "coordinates: z.tuple([z.number(), z.number()])," in schema


# ============================================================================
# Enum Tests
# ============================================================================


def test_typescript_interface_enum_as_union(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface converts Python Enum to union type by default."""

    class Status(str, Enum):
        PENDING = "pending"
        ACTIVE = "active"
        COMPLETED = "completed"

    class Task(BaseModel):
        status: Status

    module = generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    assert "status: 'pending' | 'active' | 'completed';" in schema


def test_typescript_interface_enum_as_enum_reference() -> None:
    """Test TypeScript interface can reference enum type name."""

    class Status(str, Enum):
        PENDING = "pending"
        ACTIVE = "active"
        COMPLETED = "completed"

    class Task(BaseModel):
        status: Status

    config = TypeScriptConfig(enum_style="enum")
    generator = TypeScriptSchemaGenerator(style="interface", config=config)
    module = generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    assert "status: Status;" in schema


def test_typescript_interface_optional_enum(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface for optional enum fields."""

    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class Task(BaseModel):
        name: str
        priority: Priority | None = None

    module = generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    assert "name: string;" in schema
    assert "priority?: 'low' | 'medium' | 'high';" in schema


def test_typescript_zod_enum(zod_generator: TypeScriptSchemaGenerator) -> None:
    """Test Zod schema for enum fields."""

    class Status(str, Enum):
        DRAFT = "draft"
        PUBLISHED = "published"

    class Article(BaseModel):
        status: Status

    module = zod_generator.generate_schema(Article, "Article", "1.0.0")
    schema = module.to_string()

    assert "status: z.enum(['draft', 'published'])," in schema


def test_typescript_zod_optional_enum(zod_generator: TypeScriptSchemaGenerator) -> None:
    """Test Zod schema for optional enum fields."""

    class Priority(str, Enum):
        LOW = "low"
        HIGH = "high"

    class Task(BaseModel):
        priority: Priority | None = None

    module = zod_generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    assert "priority: z.enum(['low', 'high']).optional()," in schema


def test_typescript_interface_enum_with_default(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface for enum fields with default values."""

    class Status(StrEnum):
        DRAFT = "draft"
        PUBLISHED = "published"

    class Article(BaseModel):
        title: str
        status: Status = Status.DRAFT

    module = generator.generate_schema(Article, "Article", "1.0.0")
    schema = module.to_string()

    assert "title: string;" in schema
    # Field with default should be optional
    assert "status?: 'draft' | 'published';" in schema


def test_typescript_interface_int_enum(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript interface for integer enum."""

    class Level(int, Enum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    class Task(BaseModel):
        level: Level

    module = generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    assert "level: 1 | 2 | 3;" in schema


# ============================================================================
# Union Type Tests
# ============================================================================


def test_typescript_interface_union_types(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript interface for union types."""

    class Response(BaseModel):
        data: str | int
        result: bool | dict[str, str]

    module = generator.generate_schema(Response, "Response", "1.0.0")
    schema = module.to_string()

    assert "data: string | number;" in schema
    assert "result: boolean | Record<string, string>;" in schema


def test_typescript_zod_union_types(zod_generator: TypeScriptSchemaGenerator) -> None:
    """Test Zod schema for union types."""

    class Response(BaseModel):
        data: str | int

    module = zod_generator.generate_schema(Response, "Response", "1.0.0")
    schema = module.to_string()

    assert "data: z.union([z.string(), z.number().int()])," in schema


def test_typescript_interface_complex_union(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface for complex union with None."""

    class Data(BaseModel):
        value: str | int | None = None

    module = generator.generate_schema(Data, "Data", "1.0.0")
    schema = module.to_string()

    assert "value?: string | number;" in schema


def test_typescript_zod_complex_union(zod_generator: TypeScriptSchemaGenerator) -> None:
    """Test Zod schema for complex union with None."""

    class Data(BaseModel):
        value: str | int | None = None

    module = zod_generator.generate_schema(Data, "Data", "1.0.0")
    schema = module.to_string()

    assert "value: z.union([z.string(), z.number().int()]).optional()," in schema


# ============================================================================
# Nested Model Tests
# ============================================================================


def test_typescript_interface_nested_model(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface for nested Pydantic models."""

    class Address(BaseModel):
        street: str
        city: str

    class User(BaseModel):
        name: str
        address: Address

    module = generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "street: string;" in schema
    assert "name: string;" in schema
    assert "address: Address;" in schema


def test_typescript_zod_nested_model(zod_generator: TypeScriptSchemaGenerator) -> None:
    """Test Zod schema for nested Pydantic models."""

    class Address(BaseModel):
        street: str
        city: str

    class User(BaseModel):
        name: str
        address: Address

    module = zod_generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "name: z.string()," in schema
    assert "address: AddressSchema," in schema


def test_typescript_interface_optional_nested_model(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface for optional nested models."""

    class Address(BaseModel):
        street: str

    class User(BaseModel):
        name: str
        address: Address | None = None

    module = generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "name: string;" in schema
    assert "address?: Address;" in schema


def test_typescript_interface_list_of_nested_models(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface for list of nested models."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    module = generator.generate_schema(Order, "Order", "1.0.0")
    schema = module.to_string()

    assert "items: Item[];" in schema


# ============================================================================
# Validation Constraint Tests (Zod)
# ============================================================================


def test_typescript_zod_string_constraints(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema applies string constraints."""

    class User(BaseModel):
        username: str = Field(min_length=3, max_length=20)
        email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

    module = zod_generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "username: z.string().min(3).max(20)," in schema
    assert "email: z.string().regex(/^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$/)," in schema


def test_typescript_zod_numeric_constraints(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema applies numeric constraints."""

    class Config(BaseModel):
        port: int = Field(ge=1024, le=65535)
        temperature: float = Field(gt=0, lt=100)

    module = zod_generator.generate_schema(Config, "Config", "1.0.0")
    schema = module.to_string()

    assert "port: z.number().int().gte(1024).lte(65535)," in schema
    assert "temperature: z.number().gt(0).lt(100)," in schema


# ============================================================================
# Version Naming Tests
# ============================================================================


def test_typescript_interface_version_in_name(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface includes version in type name."""

    class User(BaseModel):
        name: str

    module = generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "export interface User {" in schema


def test_typescript_interface_multiple_versions(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface generation for multiple versions."""

    class UserV1(BaseModel):
        name: str

    class UserV2(BaseModel):
        name: str
        email: str

    schema_v1 = generator.generate_schema(UserV1, "User", "1.0.0").to_string()
    schema_v2 = generator.generate_schema(UserV2, "User", "2.0.0").to_string()

    assert "export interface User {" in schema_v1
    assert "export interface User {" in schema_v2
    assert "email: string;" in schema_v2
    assert "email: string;" not in schema_v1


def test_typescript_zod_version_in_name(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema includes version in schema name."""

    class User(BaseModel):
        name: str

    module = zod_generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "export const UserSchema = z.object({" in schema
    assert "export type User = z.infer<typeof UserSchema>;" in schema


# ============================================================================
# Export Tests
# ============================================================================


def test_typescript_export_single_schema(tmp_path: Path) -> None:
    """Test exporting a single TypeScript schema to file."""

    class User(BaseModel):
        name: str
        age: int

    registry = Registry()
    registry.register("User", "1.0.0")(User)

    exporter = TypeScriptExporter(registry, style="interface")
    output_dir = tmp_path / "schemas"
    schema = exporter.export_schema("User", "1.0.0", output_path=output_dir)

    output_file = output_dir / "User.v1.0.0.ts"
    assert output_file.exists()
    content = output_file.read_text()
    assert "export interface User {" in content
    assert schema == content


def test_typescript_export_all_schemas(tmp_path: Path) -> None:
    """Test exporting all TypeScript schemas to directory."""

    class UserV1(BaseModel):
        name: str

    class UserV2(BaseModel):
        name: str
        email: str

    class Order(BaseModel):
        id: int

    registry = Registry()
    registry.register("User", "1.0.0")(UserV1)
    registry.register("User", "2.0.0")(UserV2)
    registry.register("Order", "1.0.0")(Order)

    exporter = TypeScriptExporter(registry, style="interface")
    output_dir = tmp_path / "types"
    schemas = exporter.export_all_schemas(output_dir)

    assert (output_dir / "User.v1.0.0.ts").exists()
    assert (output_dir / "User.v2.0.0.ts").exists()
    assert (output_dir / "Order.v1.0.0.ts").exists()

    assert "User" in schemas
    assert "Order" in schemas
    assert "1.0.0" in schemas["User"]
    assert "2.0.0" in schemas["User"]


def test_typescript_export_zod_schemas(tmp_path: Path) -> None:
    """Test exporting Zod schemas to directory."""

    class User(BaseModel):
        name: str

    registry = Registry()
    registry.register("User", "1.0.0")(User)

    exporter = TypeScriptExporter(registry, style="zod")
    output_dir = tmp_path / "schemas"
    exporter.export_all_schemas(output_dir)

    schema_file = output_dir / "User.v1.0.0.ts"
    assert schema_file.exists()

    content = schema_file.read_text()
    assert "import { z } from 'zod';" in content
    assert "export const UserSchema = z.object({" in content
    assert "export type User = z.infer<typeof UserSchema>;" in content


def test_typescript_export_by_major_version(tmp_path: Path) -> None:
    """Test exporting schemas organized by major version."""

    class UserV1(BaseModel):
        name: str

    class UserV1_1(BaseModel):
        name: str
        email: str | None = None

    class UserV2(BaseModel):
        name: str
        email: str

    registry = Registry()
    registry.register("User", "1.0.0")(UserV1)
    registry.register("User", "1.1.0")(UserV1_1)
    registry.register("User", "2.0.0")(UserV2)

    exporter = TypeScriptExporter(registry, style="interface")
    output_dir = tmp_path / "types"
    schemas = exporter.export_all_schemas(output_dir, organization="major_version")

    assert (output_dir / "v1" / "User.v1.0.0.ts").exists()
    assert (output_dir / "v1" / "User.v1.1.0.ts").exists()
    assert (output_dir / "v2" / "User.v2.0.0.ts").exists()

    assert "User" in schemas
    assert "1.0.0" in schemas["User"]
    assert "1.1.0" in schemas["User"]
    assert "2.0.0" in schemas["User"]


def test_typescript_export_by_model(tmp_path: Path) -> None:
    """Test exporting schemas organized by model."""

    class User(BaseModel):
        name: str

    class Order(BaseModel):
        id: int

    registry = Registry()
    registry.register("User", "1.0.0")(User)
    registry.register("User", "2.0.0")(User)
    registry.register("Order", "1.0.0")(Order)

    exporter = TypeScriptExporter(registry, style="interface")
    output_dir = tmp_path / "types"
    schemas = exporter.export_all_schemas(output_dir, organization="model")

    assert (output_dir / "User" / "1.0.0.ts").exists()
    assert (output_dir / "User" / "2.0.0.ts").exists()
    assert (output_dir / "Order" / "1.0.0.ts").exists()

    assert "User" in schemas
    assert "Order" in schemas
    assert "1.0.0" in schemas["User"]
    assert "2.0.0" in schemas["User"]


def test_typescript_export_barrel_exports_by_major_version(tmp_path: Path) -> None:
    """Test barrel exports generation for by-major-version organization."""

    class UserV1(BaseModel):
        name: str

    class UserV2(BaseModel):
        name: str
        email: str

    class OrderV1(BaseModel):
        id: int

    registry = Registry()
    registry.register("User", "1.0.0")(UserV1)
    registry.register("User", "2.0.0")(UserV2)
    registry.register("Order", "1.0.0")(OrderV1)

    exporter = TypeScriptExporter(registry, style="interface")
    output_dir = tmp_path / "types"
    exporter.export_all_schemas(
        output_dir, organization="major_version", include_barrel_exports=True
    )

    assert (output_dir / "v1" / "index.ts").exists()
    assert (output_dir / "v2" / "index.ts").exists()
    assert (output_dir / "index.ts").exists()

    v1_index = (output_dir / "v1" / "index.ts").read_text()
    assert "export * from './Order.v1.0.0';" in v1_index
    assert "export * from './User.v1.0.0';" in v1_index

    v2_index = (output_dir / "v2" / "index.ts").read_text()
    assert "export * from './User.v2.0.0';" in v2_index

    root_index = (output_dir / "index.ts").read_text()
    assert "export * from './v2';" in root_index


def test_typescript_export_barrel_exports_by_model(tmp_path: Path) -> None:
    """Test barrel exports generation for by-model organization."""

    class User(BaseModel):
        name: str

    class Order(BaseModel):
        id: int

    registry = Registry()
    registry.register("User", "1.0.0")(User)
    registry.register("User", "2.0.0")(User)
    registry.register("Order", "1.0.0")(Order)
    registry.register("Order", "1.5.0")(Order)

    exporter = TypeScriptExporter(registry, style="interface")
    output_dir = tmp_path / "types"
    exporter.export_all_schemas(
        output_dir, organization="model", include_barrel_exports=True
    )

    assert (output_dir / "User" / "index.ts").exists()
    assert (output_dir / "Order" / "index.ts").exists()
    assert (output_dir / "index.ts").exists()

    user_index = (output_dir / "User" / "index.ts").read_text()
    assert "export * from './2.0.0';" in user_index

    order_index = (output_dir / "Order" / "index.ts").read_text()
    assert "export * from './1.5.0';" in order_index

    root_index = (output_dir / "index.ts").read_text()
    assert "export * from './Order';" in root_index
    assert "export * from './User';" in root_index


def test_typescript_export_no_barrel_exports_for_flat(tmp_path: Path) -> None:
    """Test that barrel exports are not generated for flat organization."""

    class User(BaseModel):
        name: str

    registry = Registry()
    registry.register("User", "1.0.0")(User)

    exporter = TypeScriptExporter(registry, style="interface")
    output_dir = tmp_path / "types"
    exporter.export_all_schemas(
        output_dir,
        organization="flat",
        include_barrel_exports=True,  # Should be ignored
    )

    assert not (output_dir / "index.ts").exists()
    assert (output_dir / "User.v1.0.0.ts").exists()


def test_typescript_export_disable_barrel_exports(tmp_path: Path) -> None:
    """Test disabling barrel exports generation."""

    class User(BaseModel):
        name: str

    registry = Registry()
    registry.register("User", "1.0.0")(User)
    registry.register("User", "2.0.0")(User)

    exporter = TypeScriptExporter(registry, style="interface")
    output_dir = tmp_path / "types"
    exporter.export_all_schemas(
        output_dir, organization="major_version", include_barrel_exports=False
    )

    assert (output_dir / "v1" / "User.v1.0.0.ts").exists()
    assert (output_dir / "v2" / "User.v2.0.0.ts").exists()
    assert not (output_dir / "v1" / "index.ts").exists()
    assert not (output_dir / "v2" / "index.ts").exists()
    assert not (output_dir / "index.ts").exists()


def test_typescript_export_single_schema_with_organization(tmp_path: Path) -> None:
    """Test exporting a single schema with different organization styles."""

    class User(BaseModel):
        name: str
        email: str

    registry = Registry()
    registry.register("User", "1.2.3")(User)

    exporter = TypeScriptExporter(registry, style="interface")

    flat_dir = tmp_path / "flat"
    exporter.export_schema("User", "1.2.3", flat_dir, organization="flat")
    assert (flat_dir / "User.v1.2.3.ts").exists()

    major_dir = tmp_path / "major"
    exporter.export_schema("User", "1.2.3", major_dir, organization="major_version")
    assert (major_dir / "v1" / "User.v1.2.3.ts").exists()

    model_dir = tmp_path / "model"
    exporter.export_schema("User", "1.2.3", model_dir, organization="model")
    assert (model_dir / "User" / "1.2.3.ts").exists()


def test_typescript_clean_type_names(tmp_path: Path) -> None:
    """Test that exported schemas use clean type names without version suffixes."""

    class User(BaseModel):
        name: str
        email: str

    registry = Registry()
    registry.register("User", "1.0.0")(User)

    exporter = TypeScriptExporter(registry, style="interface")
    output_dir = tmp_path / "types"
    exporter.export_all_schemas(output_dir)

    schema_file = output_dir / "User.v1.0.0.ts"
    content = schema_file.read_text()

    assert "export interface User {" in content
    assert "UserV1_0_0" not in content
    assert "name: string;" in content
    assert "email: string;" in content


def test_typescript_zod_clean_type_names(tmp_path: Path) -> None:
    """Test that Zod schemas use clean type names."""

    class User(BaseModel):
        name: str
        age: int

    registry = Registry()
    registry.register("User", "2.1.0")(User)

    exporter = TypeScriptExporter(registry, style="zod")
    output_dir = tmp_path / "schemas"
    exporter.export_all_schemas(output_dir)

    schema_file = output_dir / "User.v2.1.0.ts"
    content = schema_file.read_text()

    assert "export const UserSchema = z.object({" in content
    assert "export type User = z.infer<typeof UserSchema>;" in content
    assert "UserV2_1_0" not in content


# ============================================================================
# More complex
# ============================================================================


def test_typescript_discriminated_union(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for discriminated unions."""

    class Cat(BaseModel):
        type: Literal["cat"]
        meow_volume: int

    class Dog(BaseModel):
        type: Literal["dog"]
        bark_volume: int

    class Pet(BaseModel):
        pet: Cat | Dog

    module = generator.generate_schema(Pet, "Pet", "1.0.0")
    schema = module.to_string()

    assert "pet:" in schema
    assert "Cat" in schema or "Dog" in schema or "|" in schema


def test_typescript_deeply_nested_models(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for deeply nested models."""

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class Company(BaseModel):
        name: str
        address: Address

    class Employee(BaseModel):
        name: str
        company: Company

    class Department(BaseModel):
        name: str
        employees: list[Employee]

    class Organization(BaseModel):
        name: str
        departments: list[Department]

    module = generator.generate_schema(Organization, "Organization", "1.0.0")
    schema = module.to_string()

    assert "name: string;" in schema
    assert "departments: Department[];" in schema


def test_typescript_recursive_model(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for recursive/self-referential models."""

    class TreeNode(BaseModel):
        value: int
        children: list["TreeNode"] = []

    module = generator.generate_schema(TreeNode, "TreeNode", "1.0.0")
    schema = module.to_string()

    assert "value: number;" in schema
    assert "children?: TreeNode[];" in schema


def test_typescript_recursive_model_version_same(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation for recursive/self-referential models remains so."""

    class TreeNodeV1(BaseModel):
        value: int
        children: list["TreeNodeV1"] = []

    module = generator.generate_schema(TreeNodeV1, "TreeNode", "1.0.0")
    schema = module.to_string()

    assert "value: number;" in schema
    assert "children?: TreeNode[];" in schema


def test_typescript_recursive_model_enum_mode(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation for recursive/self-referential models enum style."""

    class TreeNode(BaseModel):
        value: int
        children: list["TreeNode"] = []

    config = TypeScriptConfig(enum_style="enum")
    generator.config = config
    module = generator.generate_schema(TreeNode, "TreeNode", "1.0.0")
    schema = module.to_string()

    assert "value: number;" in schema
    assert "children?: TreeNode[];" in schema


def test_typescript_complex_nested_collections(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation for complex nested collections."""

    class ComplexModel(BaseModel):
        matrix: list[list[int]]
        nested_config: dict[str, dict[str, str]]
        records: list[dict[str, int]]
        groups: dict[str, list[str]]
        coordinates: list[tuple[float, float]]

    module = generator.generate_schema(ComplexModel, "ComplexModel", "1.0.0")
    schema = module.to_string()

    assert "matrix: number[][];" in schema
    assert "nested_config: Record<string, Record<string, string>>;" in schema
    assert "records: Record<string, number>[];" in schema
    assert "groups: Record<string, string[]>;" in schema
    assert "coordinates: [number, number][];" in schema


def test_typescript_literal_types(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for Literal types."""

    class Config(BaseModel):
        environment: Literal["dev", "staging", "prod"]
        log_level: Literal["debug", "info", "warning", "error"]
        enabled: Literal[True]

    module = generator.generate_schema(Config, "Config", "1.0.0")
    schema = module.to_string()

    assert "environment:" in schema
    assert "log_level:" in schema
    assert "enabled:" in schema


def test_typescript_annotated_types(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for Annotated types with constraints."""

    class User(BaseModel):
        username: Annotated[str, constr(min_length=3, max_length=20)]
        age: Annotated[int, conint(ge=0, le=150)]
        email: Annotated[str, constr(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")]

    module = generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "username: string;" in schema
    assert "age: number;" in schema
    assert "email: string;" in schema


def test_typescript_zod_annotated_constraints(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema generation preserves Annotated constraints."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3, max_length=20)]
        age: Annotated[int, Field(ge=0, le=150)]

    module = zod_generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "username: z.string().min(3).max(20)," in schema
    assert "age: z.number().int().gte(0).lte(150)," in schema


def test_typescript_mixed_optional_and_required(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation with mix of optional and required fields."""

    class MixedModel(BaseModel):
        required_field: str
        optional_with_none: str | None = None
        optional_with_default: int = 42
        required_but_nullable: str | None
        optional_complex: dict[str, list[int]] | None = None

    module = generator.generate_schema(MixedModel, "MixedModel", "1.0.0")
    schema = module.to_string()

    assert "required_field: string;" in schema
    assert "optional_with_none?: string;" in schema
    assert "optional_with_default?: number;" in schema
    assert "required_but_nullable" in schema
    assert "optional_complex?: Record<string, number[]>;" in schema


def test_typescript_all_primitive_types(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for all supported primitive types."""

    class AllTypes(BaseModel):
        str_field: str
        int_field: int
        float_field: float
        bool_field: bool
        bytes_field: bytes
        datetime_field: datetime
        date_field: date
        time_field: time
        uuid_field: UUID
        decimal_field: Decimal
        none_field: None

    module = generator.generate_schema(AllTypes, "AllTypes", "1.0.0")
    schema = module.to_string()

    assert "str_field: string;" in schema
    assert "int_field: number;" in schema
    assert "float_field: number;" in schema
    assert "bool_field: boolean;" in schema
    assert "bytes_field: string;" in schema
    assert "datetime_field: string;" in schema
    assert "date_field: string;" in schema
    assert "time_field: string;" in schema
    assert "uuid_field: string;" in schema
    assert "decimal_field: number;" in schema
    assert "none_field: null;" in schema


def test_typescript_empty_model(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for empty models."""

    class Empty(BaseModel):
        pass

    module = generator.generate_schema(Empty, "Empty", "1.0.0")
    schema = module.to_string()

    assert "export interface Empty {" in schema
    assert "}" in schema


def test_typescript_model_with_forward_reference(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation with forward references."""

    class Node(BaseModel):
        value: int
        next: "Node | None" = None

    module = generator.generate_schema(Node, "Node", "1.0.0")
    schema = module.to_string()

    assert "value: number;" in schema
    assert "next?: Node;" in schema


def test_typescript_multiple_inheritance_base(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation for models with base classes."""

    class Base(BaseModel):
        id: int
        created_at: datetime

    class Extended(Base):
        name: str
        description: str

    module = generator.generate_schema(Extended, "Extended", "1.0.0")
    schema = module.to_string()

    # Should include all fields from base and extended
    assert "id: number;" in schema
    assert "created_at: string;" in schema
    assert "name: string;" in schema
    assert "description: string;" in schema


# ============================================================================
# TypeScript Validation Tests
# ============================================================================


def test_typescript_very_long_model_name(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation with very long model names."""
    long_name = "A" * 200

    class VeryLongNamedModel(BaseModel):
        field: str

    module = generator.generate_schema(VeryLongNamedModel, long_name, "1.0.0")
    schema = module.to_string()

    assert "export interface" in schema
    assert "field: string;" in schema


def test_typescript_unicode_in_docstrings(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation with Unicode characters in documentation."""

    class UnicodeModel(BaseModel):
        """Model with emoji 🎉 and Chinese 中文 characters."""

        field: str = Field(description="Field with emoji 🚀 and symbols ±∞")

    module = generator.generate_schema(UnicodeModel, "UnicodeModel", "1.0.0")
    schema = module.to_string()

    assert "🎉" in schema or "emoji" in schema
    assert "field: string;" in schema


def test_typescript_circular_reference(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation with circular references."""

    class Person(BaseModel):
        name: str
        spouse: "Person | None" = None
        friends: list["Person"] = []

    module = generator.generate_schema(Person, "Person", "1.0.0")
    schema = module.to_string()

    assert "name: string;" in schema
    assert "spouse?: Person;" in schema
    assert "friends" in schema


# ============================================================================
# Configuration Tests
# ============================================================================


def test_typescript_config_date_format_timestamp() -> None:
    """Test TypeScript generation with timestamp date format."""
    config = TypeScriptConfig(date_format="timestamp")
    generator = TypeScriptSchemaGenerator(style="interface", config=config)

    class Event(BaseModel):
        timestamp: datetime
        date: date

    module = generator.generate_schema(Event, "Event", "1.0.0")
    schema = module.to_string()

    assert "timestamp: number;" in schema
    assert "date: number;" in schema


def test_typescript_config_enum_style_enum() -> None:
    """Test TypeScript generation with enum style."""
    config = TypeScriptConfig(enum_style="enum")
    generator = TypeScriptSchemaGenerator(style="interface", config=config)

    class Status(str, Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    class Model(BaseModel):
        status: Status

    module = generator.generate_schema(Model, "Model", "1.0.0")
    schema = module.to_string()

    assert "status: Status;" in schema


# ============================================================================
# Comparison Tests
# ============================================================================


def test_typescript_interface_vs_type_output() -> None:
    """Test that interface and type alias generate similar output."""

    class User(BaseModel):
        name: str
        age: int

    interface_gen = TypeScriptSchemaGenerator(style="interface")
    type_gen = TypeScriptSchemaGenerator(style="type")

    interface_module = interface_gen.generate_schema(User, "User", "1.0.0")
    interface_schema = interface_module.to_string()
    type_module = type_gen.generate_schema(User, "User", "1.0.0")
    type_schema = type_module.to_string()

    assert "name: string;" in interface_schema
    assert "name: string;" in type_schema
    assert "age: number;" in interface_schema
    assert "age: number;" in type_schema

    assert "interface" in interface_schema
    assert "type" in type_schema


def test_typescript_zod_generates_both_schema_and_type(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test that Zod generation creates both schema and inferred type."""

    class User(BaseModel):
        name: str

    module = zod_generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "export const UserSchema = z.object({" in schema
    assert "export type User = z.infer<typeof UserSchema>;" in schema


def test_typescript_generic_model_abstract(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation for abstract generic models."""

    class ApiResponse(BaseModel, Generic[T]):
        """Generic API response."""

        data: T | None = None
        error: str | None = None
        success: bool = True

    module = generator.generate_schema(ApiResponse, "ApiResponse", "1.0.0")
    schema = module.to_string()

    # Should generate a generic interface with type parameter
    # Expected: export interface ApiResponse<T> {
    assert "export interface ApiResponse" in schema
    # The generic parameter might be handled in different ways
    # At minimum, the fields should be present
    assert "data?: T;" in schema or "data?: any;" in schema
    assert "error?: string;" in schema
    assert "success?: boolean;" in schema


def test_typescript_generic_model_concrete(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation for concrete generic model instantiations."""

    class User(BaseModel):
        name: str
        email: str

    class ApiResponse(BaseModel, Generic[T]):
        data: T | None = None
        error: str | None = None

    UserResponse = ApiResponse[User]

    module = generator.generate_schema(UserResponse, "UserResponse", "1.0.0")
    schema = module.to_string()

    assert "export interface UserResponse" in schema
    assert "data?: User | null;" in schema or "data?:" in schema
    assert "error?: string;" in schema


def test_typescript_multiple_type_params(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for models with multiple type parameters."""

    class KeyValuePair(BaseModel, Generic[K, V]):
        key: K
        value: V

    module = generator.generate_schema(KeyValuePair, "KeyValuePair", "1.0.0")
    schema = module.to_string()

    assert "export interface KeyValuePair" in schema
    assert "key:" in schema
    assert "value:" in schema


def test_typescript_nested_generics(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for nested generic types."""

    class Wrapper(BaseModel, Generic[T]):
        value: T
        items: list[T]

    module = generator.generate_schema(Wrapper, "Wrapper", "1.0.0")
    schema = module.to_string()

    assert "export interface Wrapper" in schema
    assert "value:" in schema
    assert "items:" in schema


def test_typescript_zod_generic_model(zod_generator: TypeScriptSchemaGenerator) -> None:
    """Test Zod schema generation for generic models."""

    class Container(BaseModel, Generic[T]):
        content: T

    module = zod_generator.generate_schema(Container, "Container", "1.0.0")
    schema = module.to_string()

    # Zod doesn't really support generics in the same way
    assert "export const ContainerSchema" in schema
    assert "content:" in schema


def test_typescript_bounded_type_var(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation with bounded TypeVars."""

    class Animal(BaseModel):
        name: str

    class Dog(Animal):
        breed: str

    AnimalT = TypeVar("AnimalT", bound=Animal)

    class Shelter(BaseModel, Generic[AnimalT]):
        animals: list[AnimalT]

    module = generator.generate_schema(Shelter, "Shelter", "1.0.0")
    schema = module.to_string()

    assert "export interface Shelter" in schema
    assert "animals:" in schema


def test_typescript_constrained_type_var(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation with constrained TypeVars."""
    NumberT = TypeVar("NumberT", int, float)

    class NumericContainer(BaseModel, Generic[NumberT]):
        value: NumberT

    module = generator.generate_schema(NumericContainer, "NumericContainer", "1.0.0")
    schema = module.to_string()

    assert "export interface NumericContainer" in schema
    assert "value:" in schema


def test_typescript_computed_field_basic(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for basic computed fields."""

    class Person(BaseModel):
        first_name: str
        last_name: str

        @computed_field  # type: ignore[prop-decorator]
        @property
        def full_name(self) -> str:
            return f"{self.first_name} {self.last_name}"

    module = generator.generate_schema(Person, "Person", "1.0.0")
    schema = module.to_string()

    assert "first_name: string;" in schema
    assert "last_name: string;" in schema
    assert "full_name" in schema
    assert "readonly full_name: string;" in schema or "full_name: string;" in schema


def test_typescript_computed_field_complex_type(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation for computed fields with complex types."""

    class Order(BaseModel):
        items: list[dict[str, float]]
        tax_rate: float = 0.1

        @computed_field  # type: ignore[prop-decorator]
        @property
        def total(self) -> float:
            subtotal = sum(item["price"] for item in self.items)
            return subtotal * (1 + self.tax_rate)

        @computed_field  # type: ignore[prop-decorator]
        @property
        def item_count(self) -> int:
            return len(self.items)

    module = generator.generate_schema(Order, "Order", "1.0.0")
    schema = module.to_string()

    assert "items: Record<string, number>[];" in schema
    assert "tax_rate?: number;" in schema
    assert "total" in schema
    assert "item_count" in schema
    assert ": number;" in schema  # for total and item_count


def test_typescript_computed_field_optional(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation for optional computed fields."""

    class User(BaseModel):
        email: str
        phone: str | None = None

        @computed_field  # type: ignore[prop-decorator]
        @property
        def contact_method(self) -> str | None:
            return self.phone or self.email

    module = generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "email: string;" in schema
    assert "phone?: string;" in schema
    assert "contact_method" in schema


def test_typescript_computed_field_nested_model(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation for computed fields returning models."""

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class Person(BaseModel):
        name: str
        street: str
        city: str
        country: str

        @computed_field  # type: ignore[prop-decorator]
        @property
        def address(self) -> Address:
            return Address(street=self.street, city=self.city, country=self.country)

    module = generator.generate_schema(Person, "Person", "1.0.0")
    schema = module.to_string()

    assert "name: string;" in schema
    assert "address" in schema
    assert "Address" in schema or "address:" in schema


def test_typescript_zod_computed_field(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema generation for computed fields."""

    class Rectangle(BaseModel):
        width: float
        height: float

        @computed_field  # type: ignore[prop-decorator]
        @property
        def area(self) -> float:
            return self.width * self.height

    module = zod_generator.generate_schema(Rectangle, "Rectangle", "1.0.0")
    schema = module.to_string()

    assert "width: z.number()," in schema
    assert "height: z.number()," in schema
    assert "area" in schema


def test_typescript_computed_field_list(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for computed fields returning lists."""

    class Team(BaseModel):
        members: list[str]

        @computed_field  # type: ignore[prop-decorator]
        @property
        def member_count(self) -> int:
            return len(self.members)

        @computed_field  # type: ignore[prop-decorator]
        @property
        def uppercase_members(self) -> list[str]:
            return [m.upper() for m in self.members]

    module = generator.generate_schema(Team, "Team", "1.0.0")
    schema = module.to_string()

    assert "members: string[];" in schema
    assert "member_count" in schema
    assert "uppercase_members" in schema


def test_typescript_computed_field_cached(generator: TypeScriptSchemaGenerator) -> None:
    """Test TypeScript generation for cached computed fields."""

    class ExpensiveComputation(BaseModel):
        data: list[int]

        @computed_field  # type: ignore[prop-decorator]
        @property
        def expensive_result(self) -> int:
            return sum(self.data)

    module = generator.generate_schema(
        ExpensiveComputation, "ExpensiveComputation", "1.0.0"
    )
    schema = module.to_string()

    assert "data: number[];" in schema
    assert "expensive_result" in schema


def test_typescript_computed_field_serialization_alias(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation for computed fields with serialization aliases."""

    class Model(BaseModel):
        internal_value: int

        @computed_field  # type: ignore[prop-decorator]
        @property
        def computed_value(self) -> int:
            return self.internal_value * 2

    module = generator.generate_schema(Model, "Model", "1.0.0")
    schema = module.to_string()

    assert "internal_value: number;" in schema
    # Should use alias if provided
    assert "externalValue" in schema or "computed_value" in schema


def test_typescript_multiple_computed_fields(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation with multiple computed fields."""

    class Product(BaseModel):
        name: str
        price: float
        quantity: int
        tax_rate: float = 0.2

        @computed_field  # type: ignore[prop-decorator]
        @property
        def subtotal(self) -> float:
            return self.price * self.quantity

        @computed_field  # type: ignore[prop-decorator]
        @property
        def tax(self) -> float:
            return self.subtotal * self.tax_rate

        @computed_field  # type: ignore[prop-decorator]
        @property
        def total(self) -> float:
            return self.subtotal + self.tax

    module = generator.generate_schema(Product, "Product", "1.0.0")
    schema = module.to_string()

    assert "name: string;" in schema
    assert "price: number;" in schema
    assert "quantity: number;" in schema
    assert "tax_rate?: number;" in schema
    assert "subtotal" in schema
    assert "tax" in schema
    assert "total" in schema


# ============================================================================
# Configuration Tests
# ============================================================================


def test_typescript_computed_field_readonly_option() -> None:
    """Test configuration option for marking computed fields as readonly."""
    config = TypeScriptConfig(mark_computed_readonly=True)
    generator = TypeScriptSchemaGenerator(style="interface", config=config)

    class Model(BaseModel):
        value: int

        @computed_field  # type: ignore[prop-decorator]
        @property
        def doubled(self) -> int:
            return self.value * 2

    module = generator.generate_schema(Model, "Model", "1.0.0")
    schema = module.to_string()

    assert "readonly doubled: number;" in schema


def test_typescript_computed_field_exclude_option() -> None:
    """Test configuration option to exclude computed fields."""
    config = TypeScriptConfig(include_computed_fields=False)
    generator = TypeScriptSchemaGenerator(style="interface", config=config)

    class Model(BaseModel):
        value: int

        @computed_field  # type: ignore[prop-decorator]
        @property
        def doubled(self) -> int:
            return self.value * 2

    module = generator.generate_schema(Model, "Model", "1.0.0")
    schema = module.to_string()

    assert "doubled" not in schema
    assert "value: number;" in schema


# ============================================================================
# Documentation Tests
# ============================================================================


def test_typescript_computed_field_with_docstring(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript generation includes computed field docstrings."""

    class Model(BaseModel):
        value: int

        @computed_field  # type: ignore[prop-decorator]
        @property
        def computed(self) -> int:
            """This is a computed field."""
            return self.value * 2

    module = generator.generate_schema(Model, "Model", "1.0.0")
    schema = module.to_string()

    assert "computed" in schema
    if "/**" in schema:
        assert "computed field" in schema.lower()


# ============================================================================
# Edge Cases
# ============================================================================


def test_typescript_computed_field_depends_on_other_computed(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test computed fields that depend on other computed fields."""

    class Chain(BaseModel):
        base: int

        @computed_field  # type: ignore[prop-decorator]
        @property
        def step1(self) -> int:
            return self.base * 2

        @computed_field  # type: ignore[prop-decorator]
        @property
        def step2(self) -> int:
            return self.step1 * 2

        @computed_field  # type: ignore[prop-decorator]
        @property
        def step3(self) -> int:
            return self.step2 * 2

    module = generator.generate_schema(Chain, "Chain", "1.0.0")
    schema = module.to_string()

    assert "base: number;" in schema
    assert "step1" in schema
    assert "step2" in schema
    assert "step3" in schema


@pytest.fixture
def union_generator() -> TypeScriptSchemaGenerator:
    """Generator with union style (default)."""
    config = TypeScriptConfig(enum_style="union")
    return TypeScriptSchemaGenerator(style="interface", config=config)


@pytest.fixture
def enum_generator() -> TypeScriptSchemaGenerator:
    """Generator with enum style."""
    config = TypeScriptConfig(enum_style="enum")
    return TypeScriptSchemaGenerator(style="interface", config=config)


@pytest.fixture
def enum_type_generator() -> TypeScriptSchemaGenerator:
    """Generator with enum style and type alias."""
    config = TypeScriptConfig(enum_style="enum")
    return TypeScriptSchemaGenerator(style="type", config=config)


# ============================================================================
# Union Style Tests (Default)
# ============================================================================


def test_enum_union_style_default(union_generator: TypeScriptSchemaGenerator) -> None:
    """Test default union style for enums."""

    class Status(str, Enum):
        PENDING = "pending"
        ACTIVE = "active"
        COMPLETED = "completed"

    class Task(BaseModel):
        name: str
        status: Status

    module = union_generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    # Should generate inline union type
    assert "status: 'pending' | 'active' | 'completed';" in schema
    assert "export enum Status" not in schema
    assert "enum Status {" not in schema


def test_enum_union_style_explicit(union_generator: TypeScriptSchemaGenerator) -> None:
    """Test explicit union style configuration."""

    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class Task(BaseModel):
        priority: Priority

    module = union_generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    assert "priority: 'low' | 'medium' | 'high';" in schema
    assert "export enum Priority" not in schema


# ============================================================================
# Enum Style Tests
# ============================================================================


def test_enum_style_basic(enum_generator: TypeScriptSchemaGenerator) -> None:
    """Test basic enum style generation."""

    class Status(str, Enum):
        PENDING = "pending"
        ACTIVE = "active"
        COMPLETED = "completed"

    class Task(BaseModel):
        name: str
        status: Status

    module = enum_generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    assert "export enum Status {" in schema
    assert "PENDING = 'pending'," in schema
    assert "ACTIVE = 'active'," in schema
    assert "COMPLETED = 'completed'," in schema

    assert "status: Status;" in schema

    assert "'pending' | 'active'" not in schema


def test_enum_style_multiple_enums(enum_generator: TypeScriptSchemaGenerator) -> None:
    """Test generation with multiple enums."""

    class Status(str, Enum):
        DRAFT = "draft"
        PUBLISHED = "published"

    class Priority(str, Enum):
        LOW = "low"
        HIGH = "high"

    class Article(BaseModel):
        title: str
        status: Status
        priority: Priority

    module = enum_generator.generate_schema(Article, "Article", "1.0.0")
    schema = module.to_string()

    assert "export enum Status {" in schema
    assert "DRAFT = 'draft'," in schema
    assert "PUBLISHED = 'published'," in schema

    assert "export enum Priority {" in schema
    assert "LOW = 'low'," in schema
    assert "HIGH = 'high'," in schema

    assert "status: Status;" in schema
    assert "priority: Priority;" in schema


def test_enum_style_int_enum(enum_generator: TypeScriptSchemaGenerator) -> None:
    """Test enum style with integer enums."""

    class ErrorCode(IntEnum):
        SUCCESS = 0
        NOT_FOUND = 404
        SERVER_ERROR = 500

    class Response(BaseModel):
        code: ErrorCode
        message: str

    module = enum_generator.generate_schema(Response, "Response", "1.0.0")
    schema = module.to_string()

    assert "export enum ErrorCode {" in schema
    assert "SUCCESS = 0," in schema
    assert "NOT_FOUND = 404," in schema
    assert "SERVER_ERROR = 500," in schema

    assert "code: ErrorCode;" in schema


def test_enum_style_optional_enum(enum_generator: TypeScriptSchemaGenerator) -> None:
    """Test enum style with optional enum fields."""

    class Status(str, Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    class User(BaseModel):
        name: str
        status: Status | None = None

    module = enum_generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "export enum Status {" in schema
    assert "ACTIVE = 'active'," in schema
    assert "INACTIVE = 'inactive'," in schema

    assert "status?: Status;" in schema


def test_enum_style_with_default(enum_generator: TypeScriptSchemaGenerator) -> None:
    """Test enum style with default enum values."""

    class Status(str, Enum):
        DRAFT = "draft"
        PUBLISHED = "published"

    class Article(BaseModel):
        title: str
        status: Status = Status.DRAFT

    module = enum_generator.generate_schema(Article, "Article", "1.0.0")
    schema = module.to_string()

    assert "export enum Status {" in schema
    assert "status?: Status;" in schema  # Has default, so optional


def test_enum_style_enum_in_list(enum_generator: TypeScriptSchemaGenerator) -> None:
    """Test enum style with list of enums."""

    class Tag(str, Enum):
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        TYPESCRIPT = "typescript"

    class Project(BaseModel):
        name: str
        tags: list[Tag]

    module = enum_generator.generate_schema(Project, "Project", "1.0.0")
    schema = module.to_string()

    assert "export enum Tag {" in schema
    assert "PYTHON = 'python'," in schema

    assert "tags: Tag[];" in schema


def test_enum_style_enum_in_dict(enum_generator: TypeScriptSchemaGenerator) -> None:
    """Test enum style with dict containing enums."""

    class Status(str, Enum):
        PASS = "pass"
        FAIL = "fail"

    class TestResults(BaseModel):
        results: dict[str, Status]

    module = enum_generator.generate_schema(TestResults, "TestResults", "1.0.0")
    schema = module.to_string()

    assert "export enum Status {" in schema
    assert "results: Record<string, Status>;" in schema


def test_enum_style_nested_model_with_enum(
    enum_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test enum style with nested models containing enums."""

    class Status(str, Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    class Address(BaseModel):
        street: str
        status: Status

    class User(BaseModel):
        name: str
        address: Address

    module = enum_generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "export enum Status {" in schema
    assert "ACTIVE = 'active'," in schema

    assert "status: Status;" in schema


def test_enum_style_deeply_nested_enums(
    enum_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test enum style with deeply nested models containing multiple enums."""

    class Priority(str, Enum):
        LOW = "low"
        HIGH = "high"

    class Status(str, Enum):
        OPEN = "open"
        CLOSED = "closed"

    class Task(BaseModel):
        name: str
        priority: Priority

    class Project(BaseModel):
        title: str
        status: Status
        tasks: list[Task]

    class Organization(BaseModel):
        name: str
        projects: list[Project]

    module = enum_generator.generate_schema(Organization, "Organization", "1.0.0")
    schema = module.to_string()

    assert "export enum Priority {" in schema
    assert "LOW = 'low'," in schema
    assert "HIGH = 'high'," in schema

    assert "export enum Status {" in schema
    assert "OPEN = 'open'," in schema
    assert "CLOSED = 'closed'," in schema

    assert "projects: Project[];" in schema or "projects:" in schema


def test_enum_style_same_enum_multiple_fields(
    enum_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test enum style when same enum is used in multiple fields."""

    class Status(str, Enum):
        DRAFT = "draft"
        PUBLISHED = "published"

    class Workflow(BaseModel):
        current_status: Status
        previous_status: Status | None = None
        target_status: Status

    module = enum_generator.generate_schema(Workflow, "Workflow", "1.0.0")
    schema = module.to_string()

    enum_count = schema.count("export enum Status {")
    assert enum_count == 1

    assert "current_status: Status;" in schema
    assert "previous_status?: Status;" in schema
    assert "target_status: Status;" in schema


def test_enum_style_enum_order(enum_generator: TypeScriptSchemaGenerator) -> None:
    """Test that enum declarations appear before interface."""

    class Priority(str, Enum):
        LOW = "low"
        HIGH = "high"

    class Task(BaseModel):
        name: str
        priority: Priority

    module = enum_generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    enum_pos = schema.find("export enum Priority")
    interface_pos = schema.find("export interface Task")

    assert enum_pos < interface_pos
    assert enum_pos >= 0
    assert interface_pos >= 0


# ============================================================================
# Type Alias Tests with Enum
# ============================================================================


def test_enum_style_type_alias(enum_type_generator: TypeScriptSchemaGenerator) -> None:
    """Test enum style with type alias output."""

    class Status(str, Enum):
        PENDING = "pending"
        ACTIVE = "active"

    class Task(BaseModel):
        status: Status

    module = enum_type_generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    assert "export enum Status {" in schema
    assert "PENDING = 'pending'," in schema

    assert "export type Task = {" in schema
    assert "status: Status;" in schema


# ============================================================================
# Edge Cases
# ============================================================================


def test_enum_style_empty_enum(enum_generator: TypeScriptSchemaGenerator) -> None:
    """Test handling of empty enum (edge case)."""

    # This is technically invalid but test graceful handling
    class EmptyEnum(str, Enum):
        pass

    class Model(BaseModel):
        value: str

    module = enum_generator.generate_schema(Model, "Model", "1.0.0")
    schema = module.to_string()
    assert "export interface Model" in schema


def test_enum_style_special_characters(
    enum_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test enum values with special characters."""

    class Status(str, Enum):
        IN_PROGRESS = "in-progress"
        WAITING_FOR_REVIEW = "waiting-for-review"

    class Task(BaseModel):
        status: Status

    module = enum_generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    assert "export enum Status {" in schema
    assert "IN_PROGRESS = 'in-progress'," in schema
    assert "WAITING_FOR_REVIEW = 'waiting-for-review'," in schema


def test_enum_style_numeric_string_values(
    enum_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test enum with numeric string values."""

    class Version(str, Enum):
        V1 = "1.0"
        V2 = "2.0"
        V3 = "3.0"

    class Config(BaseModel):
        version: Version

    module = enum_generator.generate_schema(Config, "Config", "1.0.0")
    schema = module.to_string()

    assert "export enum Version {" in schema
    assert "V1 = '1.0'," in schema
    assert "V2 = '2.0'," in schema


def test_enum_style_with_documentation(
    enum_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test that model documentation is preserved with enum style."""

    class Status(str, Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    class User(BaseModel):
        """User account model."""

        name: str
        status: Status

    module = enum_generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    assert "export enum Status {" in schema
    assert "/**" in schema
    assert "User account model" in schema


# ============================================================================
# Mixed Enum and Union Types
# ============================================================================


def test_enum_style_with_literal_types(
    enum_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test enum style alongside literal types."""

    class Status(str, Enum):
        DRAFT = "draft"
        PUBLISHED = "published"

    class Article(BaseModel):
        status: Status
        environment: Literal["dev", "prod"]

    module = enum_generator.generate_schema(Article, "Article", "1.0.0")
    schema = module.to_string()

    assert "export enum Status {" in schema
    assert "status: Status;" in schema

    assert "environment: 'dev' | 'prod';" in schema


def test_enum_style_with_union_types(enum_generator: TypeScriptSchemaGenerator) -> None:
    """Test enum style with union types."""

    class Status(str, Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    class Model(BaseModel):
        status: Status
        value: str | int

    module = enum_generator.generate_schema(Model, "Model", "1.0.0")
    schema = module.to_string()

    assert "export enum Status {" in schema
    assert "status: Status;" in schema
    assert "value: string | number;" in schema


# ============================================================================
# Configuration Edge Cases
# ============================================================================


def test_no_config_defaults_to_union() -> None:
    """Test that no config defaults to union style."""
    generator = TypeScriptSchemaGenerator(style="interface")

    class Status(str, Enum):
        ACTIVE = "active"

    class Model(BaseModel):
        status: Status

    module = generator.generate_schema(Model, "Model", "1.0.0")
    schema = module.to_string()

    assert "'active'" in schema
    assert "export enum Status" not in schema


def test_empty_config_defaults_to_union() -> None:
    """Test that empty config defaults to union style."""
    config = TypeScriptConfig()
    generator = TypeScriptSchemaGenerator(style="interface", config=config)

    class Status(str, Enum):
        ACTIVE = "active"

    class Model(BaseModel):
        status: Status

    module = generator.generate_schema(Model, "Model", "1.0.0")
    schema = module.to_string()

    assert "'active'" in schema
    assert "export enum Status" not in schema


# ============================================================================
# Generator State Tests
# ============================================================================


def test_enum_tracking_resets_between_calls(
    enum_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test that enum tracking is reset between schema generations."""

    class Status1(str, Enum):
        ACTIVE = "active"

    class Model1(BaseModel):
        status: Status1

    class Status2(str, Enum):
        PENDING = "pending"

    class Model2(BaseModel):
        status: Status2

    module1 = enum_generator.generate_schema(Model1, "Model1", "1.0.0")
    schema1 = module1.to_string()
    assert "export enum Status1 {" in schema1
    assert "export enum Status2" not in schema1

    module2 = enum_generator.generate_schema(Model2, "Model2", "1.0.0")
    schema2 = module2.to_string()
    assert "export enum Status2 {" in schema2
    assert "export enum Status1" not in schema2


# ============================================================================
# Integration Tests
# ============================================================================


def test_enum_style_complex_model(enum_generator: TypeScriptSchemaGenerator) -> None:
    """Test enum style with complex model."""

    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class Status(str, Enum):
        TODO = "todo"
        IN_PROGRESS = "in_progress"
        DONE = "done"

    class Task(BaseModel):
        """A task in the system."""

        title: str
        description: str | None = None
        priority: Priority
        status: Status = Status.TODO
        tags: list[str] = []
        created_at: datetime

    module = enum_generator.generate_schema(Task, "Task", "1.0.0")
    schema = module.to_string()

    assert "export enum Priority {" in schema
    assert "export enum Status {" in schema

    assert "export interface Task {" in schema
    assert "priority: Priority;" in schema
    assert "status?: Status;" in schema  # Has default

    priority_pos = schema.find("export enum Priority")
    status_pos = schema.find("export enum Status")
    interface_pos = schema.find("export interface")

    assert priority_pos < interface_pos
    assert status_pos < interface_pos


# ============================================================================
# Field Aliases Tests
# ============================================================================


def test_typescript_interface_field_aliases(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface generation with field aliases."""

    class UserProfile(BaseModel):
        user_id: str = Field(alias="userId")
        first_name: str = Field(alias="firstName")
        last_name: str = Field(alias="lastName")
        email_address: str = Field(alias="emailAddress")

    module = generator.generate_schema(UserProfile, "UserProfile", "1.0.0")
    schema = module.to_string()

    assert "export interface UserProfile {" in schema
    assert "userId: string;" in schema
    assert "firstName: string;" in schema
    assert "lastName: string;" in schema
    assert "emailAddress: string;" in schema
    assert "user_id" not in schema
    assert "first_name" not in schema
    assert "last_name" not in schema
    assert "email_address" not in schema


def test_typescript_type_alias_field_aliases(
    type_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript type alias generation with field aliases."""

    class APIRequest(BaseModel):
        request_id: str = Field(alias="requestId")
        created_at: str = Field(alias="createdAt")
        user_agent: str | None = Field(default=None, alias="userAgent")

    module = type_generator.generate_schema(APIRequest, "APIRequest", "1.0.0")
    schema = module.to_string()

    assert "export type APIRequest = {" in schema
    assert "requestId: string;" in schema
    assert "createdAt: string;" in schema
    assert "userAgent?: string;" in schema


def test_typescript_zod_field_aliases(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema generation with field aliases."""

    class Product(BaseModel):
        product_id: str = Field(alias="productId")
        product_name: str = Field(alias="productName")
        price_cents: int = Field(alias="priceCents")

    module = zod_generator.generate_schema(Product, "Product", "1.0.0")
    schema = module.to_string()

    assert "export const ProductSchema = z.object({" in schema
    assert "productId: z.string()," in schema
    assert "productName: z.string()," in schema
    assert "priceCents: z.number().int()," in schema
    assert "product_id" not in schema
    assert "product_name" not in schema
    assert "price_cents" not in schema


def test_typescript_interface_mixed_aliases_and_regular_fields(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface with both aliased and non-aliased fields."""

    class MixedFields(BaseModel):
        id: str
        created_at: str = Field(alias="createdAt")
        name: str
        is_active: bool = Field(alias="isActive")

    module = generator.generate_schema(MixedFields, "MixedFields", "1.0.0")
    schema = module.to_string()

    assert "id: string;" in schema
    assert "createdAt: string;" in schema
    assert "name: string;" in schema
    assert "isActive: boolean;" in schema


def test_typescript_interface_optional_aliased_fields(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface with optional aliased fields."""

    class OptionalAliased(BaseModel):
        user_id: str = Field(alias="userId")
        middle_name: str | None = Field(default=None, alias="middleName")
        phone_number: str | None = Field(default=None, alias="phoneNumber")

    module = generator.generate_schema(OptionalAliased, "OptionalAliased", "1.0.0")
    schema = module.to_string()

    assert "userId: string;" in schema
    assert "middleName?: string;" in schema
    assert "phoneNumber?: string;" in schema


def test_typescript_zod_optional_aliased_fields(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema with optional aliased fields."""

    class OptionalAliased(BaseModel):
        user_id: str = Field(alias="userId")
        middle_name: str | None = Field(default=None, alias="middleName")

    module = zod_generator.generate_schema(OptionalAliased, "OptionalAliased", "1.0.0")
    schema = module.to_string()

    assert "userId: z.string()," in schema
    assert "middleName: z.string().optional()," in schema


def test_typescript_interface_serialization_alias(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface with serialization_alias."""

    class SerializedModel(BaseModel):
        internal_name: str = Field(serialization_alias="externalName")
        internal_id: int = Field(serialization_alias="externalId")

    module = generator.generate_schema(SerializedModel, "SerializedModel", "1.0.0")
    schema = module.to_string()

    assert "externalName: string;" in schema
    assert "externalId: number;" in schema


def test_typescript_interface_nested_model_with_aliases(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface with nested models containing aliases."""

    class Address(BaseModel):
        street_name: str = Field(alias="streetName")
        postal_code: str = Field(alias="postalCode")

    class Person(BaseModel):
        first_name: str = Field(alias="firstName")
        home_address: Address = Field(alias="homeAddress")

    module = generator.generate_schema(Person, "Person", "1.0.0")
    schema = module.to_string()

    assert "firstName: string;" in schema
    assert "homeAddress: Address;" in schema
    assert "streetName: string;" in schema
    assert "postalCode: string;" in schema


# ============================================================================
# Discriminated Unions Tests
# ============================================================================


def test_typescript_interface_discriminated_union(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface generation for discriminated unions."""

    class ClickEvent(BaseModel):
        event_type: Literal["click"] = "click"
        element_id: str = Field(alias="elementId")

    class ViewEvent(BaseModel):
        event_type: Literal["view"] = "view"
        page_url: str = Field(alias="pageUrl")

    class EventContainer(BaseModel):
        event: Annotated[ClickEvent | ViewEvent, Field(discriminator="event_type")]

    module = generator.generate_schema(EventContainer, "EventContainer", "1.0.0")
    schema = module.to_string()

    assert "export interface ClickEvent {" in schema
    assert "event_type: 'click';" in schema
    assert "elementId: string;" in schema
    assert "export interface ViewEvent {" in schema
    assert "event_type: 'view';" in schema
    assert "pageUrl: string;" in schema
    assert "event: ClickEvent | ViewEvent;" in schema


def test_typescript_type_alias_discriminated_union(
    type_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript type alias for discriminated unions."""

    class SuccessResponse(BaseModel):
        status: Literal["success"] = "success"
        data: dict[str, Any]

    class ErrorResponse(BaseModel):
        status: Literal["error"] = "error"
        error_message: str = Field(alias="errorMessage")

    class Response(BaseModel):
        result: Annotated[
            SuccessResponse | ErrorResponse, Field(discriminator="status")
        ]

    module = type_generator.generate_schema(Response, "Response", "1.0.0")
    schema = module.to_string()

    assert "status: 'success';" in schema
    assert "data: Record<string, any>;" in schema
    assert "status: 'error';" in schema
    assert "errorMessage: string;" in schema
    assert "result: SuccessResponse | ErrorResponse;" in schema


def test_typescript_zod_discriminated_union(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema generation for discriminated unions."""

    class NotificationEmail(BaseModel):
        type: Literal["email"] = "email"
        email_address: str = Field(alias="emailAddress")

    class NotificationSMS(BaseModel):
        type: Literal["sms"] = "sms"
        phone_number: str = Field(alias="phoneNumber")

    class Notification(BaseModel):
        notification: Annotated[
            NotificationEmail | NotificationSMS, Field(discriminator="type")
        ]

    module = zod_generator.generate_schema(Notification, "Notification", "1.0.0")
    schema = module.to_string()

    assert "export const NotificationEmailSchema = z.object({" in schema
    assert "type: z.literal('email')," in schema
    assert "emailAddress: z.string()," in schema
    assert "export const NotificationSMSSchema = z.object({" in schema
    assert "type: z.literal('sms')," in schema
    assert "phoneNumber: z.string()," in schema
    assert (
        "notification: z.union([NotificationEmailSchema, NotificationSMSSchema]),"
        in schema
    )


def test_typescript_interface_discriminated_union_with_enum(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test discriminated union using enum values."""

    class PaymentType(str, Enum):
        CARD = "card"
        BANK = "bank"

    class CardPayment(BaseModel):
        payment_type: Literal[PaymentType.CARD] = PaymentType.CARD
        card_number: str = Field(alias="cardNumber")

    class BankPayment(BaseModel):
        payment_type: Literal[PaymentType.BANK] = PaymentType.BANK
        account_number: str = Field(alias="accountNumber")

    class Payment(BaseModel):
        payment: Annotated[
            CardPayment | BankPayment, Field(discriminator="payment_type")
        ]

    module = generator.generate_schema(Payment, "Payment", "1.0.0")
    schema = module.to_string()

    assert "payment_type: 'card';" in schema
    assert "cardNumber: string;" in schema
    assert "payment_type: 'bank';" in schema
    assert "accountNumber: string;" in schema


def test_typescript_interface_discriminated_union_complex(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test discriminated union with multiple fields and optional values."""

    class CreateAction(BaseModel):
        action: Literal["create"] = "create"
        resource_type: str = Field(alias="resourceType")
        initial_data: dict[str, Any] = Field(alias="initialData")

    class UpdateAction(BaseModel):
        action: Literal["update"] = "update"
        resource_id: str = Field(alias="resourceId")
        changed_fields: dict[str, Any] = Field(alias="changedFields")

    class DeleteAction(BaseModel):
        action: Literal["delete"] = "delete"
        resource_id: str = Field(alias="resourceId")
        soft_delete: bool = Field(default=False, alias="softDelete")

    class ActionContainer(BaseModel):
        action: Annotated[
            CreateAction | UpdateAction | DeleteAction,
            Field(discriminator="action"),
        ]

    module = generator.generate_schema(ActionContainer, "ActionContainer", "1.0.0")
    schema = module.to_string()

    assert "action: 'create';" in schema
    assert "resourceType: string;" in schema
    assert "initialData: Record<string, any>;" in schema
    assert "action: 'update';" in schema
    assert "resourceId: string;" in schema
    assert "changedFields: Record<string, any>;" in schema
    assert "action: 'delete';" in schema
    assert "softDelete?: boolean;" in schema


# ============================================================================
# extra='allow' Tests
# ============================================================================


def test_typescript_interface_extra_allow(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface with extra='allow'."""

    class DynamicConfig(BaseModel):
        model_config = ConfigDict(extra="allow")
        config_name: str = Field(alias="configName")
        enabled: bool = True

    module = generator.generate_schema(DynamicConfig, "DynamicConfig", "1.0.0")
    schema = module.to_string()

    assert "export interface DynamicConfig {" in schema
    assert "configName: string;" in schema
    assert "enabled?: boolean;" in schema
    assert "[key: string]: any;" in schema


def test_typescript_type_alias_extra_allow(
    type_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript type alias with extra='allow'."""

    class FlexibleModel(BaseModel):
        model_config = ConfigDict(extra="allow")
        id: str
        name: str

    module = type_generator.generate_schema(FlexibleModel, "FlexibleModel", "1.0.0")
    schema = module.to_string()

    assert "export type FlexibleModel = {" in schema
    assert "id: string;" in schema
    assert "name: string;" in schema
    assert "[key: string]: any;" in schema


def test_typescript_zod_extra_allow(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema with extra='allow'."""

    class ExtensibleData(BaseModel):
        model_config = ConfigDict(extra="allow")
        core_field: str = Field(alias="coreField")
        version: int = 1

    module = zod_generator.generate_schema(ExtensibleData, "ExtensibleData", "1.0.0")
    schema = module.to_string()

    assert "export const ExtensibleDataSchema = z.object({" in schema
    assert "coreField: z.string()," in schema
    assert "version: z.number().int()," in schema
    assert "}).passthrough();" in schema


def test_typescript_interface_extra_allow_empty_model(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface with extra='allow' and no defined fields."""

    class EmptyDynamic(BaseModel):
        model_config = ConfigDict(extra="allow")

    module = generator.generate_schema(EmptyDynamic, "EmptyDynamic", "1.0.0")
    schema = module.to_string()

    assert "export interface EmptyDynamic {" in schema
    assert "[key: string]: any;" in schema


def test_typescript_interface_extra_allow_with_optional_fields(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface with extra='allow' and optional fields."""

    class PartialDynamic(BaseModel):
        model_config = ConfigDict(extra="allow")
        required_field: str = Field(alias="requiredField")
        optional_field: str | None = Field(default=None, alias="optionalField")

    module = generator.generate_schema(PartialDynamic, "PartialDynamic", "1.0.0")
    schema = module.to_string()

    assert "requiredField: string;" in schema
    assert "optionalField?: string;" in schema
    assert "[key: string]: any;" in schema


def test_typescript_zod_extra_allow_with_discriminator(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema with extra='allow' on union members."""

    class FlexibleEventA(BaseModel):
        model_config = ConfigDict(extra="allow")
        type: Literal["flexible_a"] = "flexible_a"
        timestamp: int

    class FlexibleEventB(BaseModel):
        model_config = ConfigDict(extra="allow")
        type: Literal["flexible_b"] = "flexible_b"
        count: int

    class Container(BaseModel):
        event: Annotated[FlexibleEventA | FlexibleEventB, Field(discriminator="type")]

    module = zod_generator.generate_schema(Container, "Container", "1.0.0")
    schema = module.to_string()

    assert "export const FlexibleEventASchema = z.object({" in schema
    assert "type: z.literal('flexible_a')," in schema
    assert "timestamp: z.number().int()," in schema
    assert "}).passthrough();" in schema
    assert "export const FlexibleEventBSchema = z.object({" in schema
    assert "type: z.literal('flexible_b')," in schema
    assert "count: z.number().int()," in schema


def test_typescript_interface_without_extra_allow(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface without extra='allow' has no index signature."""

    class StrictModel(BaseModel):
        id: str
        name: str

    module = generator.generate_schema(StrictModel, "StrictModel", "1.0.0")
    schema = module.to_string()

    assert "export interface StrictModel {" in schema
    assert "id: string;" in schema
    assert "name: string;" in schema
    assert "[key: string]: any;" not in schema


def test_typescript_zod_without_extra_allow(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema without extra='allow' has no passthrough."""

    class StrictModel(BaseModel):
        id: str
        name: str

    module = zod_generator.generate_schema(StrictModel, "StrictModel", "1.0.0")
    schema = module.to_string()

    assert "export const StrictModelSchema = z.object({" in schema
    assert "}).passthrough()" not in schema
    assert ".passthrough()" not in schema


def test_typescript_interface_all_features_combined(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript interface with aliases and extra='allow'."""

    class FlexibleEvent(BaseModel):
        model_config = ConfigDict(extra="allow")
        event_type: Literal["custom"] = "custom"
        event_id: str = Field(alias="eventId")
        created_at: str = Field(alias="createdAt")
        user_id: str | None = Field(default=None, alias="userId")

    module = generator.generate_schema(FlexibleEvent, "FlexibleEvent", "1.0.0")
    schema = module.to_string()

    assert "export interface FlexibleEvent {" in schema
    assert "event_type: 'custom';" in schema
    assert "eventId: string;" in schema
    assert "createdAt: string;" in schema
    assert "userId?: string;" in schema
    assert "[key: string]: any;" in schema


def test_typescript_type_alias_all_features_combined(
    type_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test TypeScript type alias with aliases and extra='allow'."""

    class DynamicResponse(BaseModel):
        model_config = ConfigDict(extra="allow")
        response_type: Literal["dynamic"] = "dynamic"
        request_id: str = Field(alias="requestId")
        timestamp_ms: int = Field(alias="timestampMs")

    module = type_generator.generate_schema(DynamicResponse, "DynamicResponse", "1.0.0")
    schema = module.to_string()

    assert "export type DynamicResponse = {" in schema
    assert "response_type: 'dynamic';" in schema
    assert "requestId: string;" in schema
    assert "timestampMs: number;" in schema
    assert "[key: string]: any;" in schema


def test_typescript_zod_all_features_combined(
    zod_generator: TypeScriptSchemaGenerator,
) -> None:
    """Test Zod schema with aliases and extra='allow'."""

    class ExtensibleAction(BaseModel):
        model_config = ConfigDict(extra="allow")
        action_type: Literal["extensible"] = "extensible"
        action_id: str = Field(alias="actionId")
        performed_by: str = Field(alias="performedBy")

    module = zod_generator.generate_schema(
        ExtensibleAction, "ExtensibleAction", "1.0.0"
    )
    schema = module.to_string()

    assert "export const ExtensibleActionSchema = z.object({" in schema
    assert "action_type: z.literal('extensible')," in schema
    assert "actionId: z.string()," in schema
    assert "performedBy: z.string()," in schema
    assert "}).passthrough();" in schema


def test_typescript_interface_discriminated_union_with_extra_allow(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test discriminated union where members have extra='allow'."""

    class FlexibleEventA(BaseModel):
        model_config = ConfigDict(extra="allow")
        type: Literal["a"] = "a"
        field_a: str = Field(alias="fieldA")

    class FlexibleEventB(BaseModel):
        model_config = ConfigDict(extra="allow")
        type: Literal["b"] = "b"
        field_b: int = Field(alias="fieldB")

    class EventWrapper(BaseModel):
        event: Annotated[FlexibleEventA | FlexibleEventB, Field(discriminator="type")]

    module = generator.generate_schema(EventWrapper, "EventWrapper", "1.0.0")
    schema = module.to_string()

    assert "export interface FlexibleEventA {" in schema
    assert "type: 'a';" in schema
    assert "fieldA: string;" in schema
    assert "[key: string]: any;" in schema
    assert "export interface FlexibleEventB {" in schema
    assert "type: 'b';" in schema
    assert "fieldB: number;" in schema


def test_typescript_interface_nested_with_all_features(
    generator: TypeScriptSchemaGenerator,
) -> None:
    """Test nested models with aliases and extra='allow'."""

    class MetadataModel(BaseModel):
        model_config = ConfigDict(extra="allow")
        created_at: str = Field(alias="createdAt")
        updated_at: str = Field(alias="updatedAt")

    class MainModel(BaseModel):
        model_config = ConfigDict(extra="allow")
        main_id: str = Field(alias="mainId")
        metadata: MetadataModel

    module = generator.generate_schema(MainModel, "MainModel", "1.0.0")
    schema = module.to_string()

    assert "export interface MetadataModel {" in schema
    assert "createdAt: string;" in schema
    assert "updatedAt: string;" in schema
    assert "export interface MainModel {" in schema
    assert "mainId: string;" in schema
    assert "metadata: MetadataModel;" in schema
    lines = schema.split("\n")
    metadata_block = False
    main_block = False
    for line in lines:
        if "interface MetadataModel" in line:
            metadata_block = True
            main_block = False
        elif "interface MainModel" in line:
            metadata_block = False
            main_block = True
        elif "[key: string]: any;" in line:
            assert metadata_block or main_block
