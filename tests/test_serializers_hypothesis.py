"""Hypothesis strategies for generating complex Pydantic models."""

import json
import shutil
import subprocess
import tempfile
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from enum import IntEnum, StrEnum
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Self,
)
from uuid import UUID

import fastavro
import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from hypothesis.strategies import DrawFn
from pydantic import (
    AwareDatetime,
    BaseModel,
    ByteSize,
    ConfigDict,
    Discriminator,
    Field,
    FutureDate,
    IPvAnyAddress,
    IPvAnyNetwork,
    Json,
    NaiveDatetime,
    NegativeFloat,
    NegativeInt,
    NonNegativeFloat,
    NonNegativeInt,
    NonPositiveFloat,
    NonPositiveInt,
    PastDate,
    PositiveFloat,
    PositiveInt,
    SecretBytes,
    SecretStr,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    Tag,
    computed_field,
    conbytes,
    confloat,
    confrozenset,
    conint,
    conlist,
    conset,
    constr,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.types import StringConstraints

from pyrmute import ModelManager

# ruff: noqa: PLR2004


class ColorEnum(StrEnum):
    """Example string enum."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"


class PriorityIntEnum(IntEnum):
    """Example int enum."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# Basic types
st_none = st.none()
st_bool = st.booleans()
st_int = st.integers(min_value=-1000000, max_value=1000000)
st_float = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
st_str = st.text(min_size=0, max_size=100)
st_bytes = st.binary(min_size=0, max_size=100)

# Constrained types
st_positive_int = st.integers(min_value=1, max_value=1000000)
st_negative_int = st.integers(min_value=-1000000, max_value=-1)
st_non_negative_int = st.integers(min_value=0, max_value=1000000)
st_non_positive_int = st.integers(min_value=-1000000, max_value=0)

st_positive_float = st.floats(min_value=0.01, max_value=1e6, allow_nan=False)
st_negative_float = st.floats(min_value=-1e6, max_value=-0.01, allow_nan=False)
st_non_negative_float = st.floats(min_value=0, max_value=1e6, allow_nan=False)
st_non_positive_float = st.floats(min_value=-1e6, max_value=0, allow_nan=False)

# String with constraints
st_constrained_str = st.text(
    min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))
)
st_pattern_str = st.from_regex(r"^[A-Z]{3}-\d{4}$", fullmatch=True)

# Collections
st_list_int = st.lists(st_int, min_size=0, max_size=10)
st_list_str = st.lists(st_str, min_size=0, max_size=10)
st_set_str = st.sets(st_str, min_size=0, max_size=10)
st_frozenset_int = st.frozensets(st_int, min_size=0, max_size=10)
st_tuple_mixed = st.tuples(st_int, st_str, st_bool)
st_dict_str_int = st.dictionaries(st_str, st_int, min_size=0, max_size=10)

# Date and time
st_datetime = st.datetimes(
    min_value=datetime(2000, 1, 1), max_value=datetime(2030, 12, 31)
)
st_date = st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31))
st_time = st.times()
st_timedelta = st.timedeltas(
    min_value=timedelta(seconds=0), max_value=timedelta(days=365)
)

# Special types
st_uuid = st.uuids()
st_decimal = st.decimals(
    min_value=Decimal("-1000000"),
    max_value=Decimal("1000000"),
    allow_nan=False,
    allow_infinity=False,
    places=2,
)

# Network types
st_ipv4 = st.from_type(IPv4Address)
st_ipv6 = st.from_type(IPv6Address)
st_ipv4_network = st.from_type(IPv4Network)
st_ipv6_network = st.from_type(IPv6Network)

# URLs - simplified for testing
st_url = st.builds(
    lambda: "https://example.com/path",
)
st_email = st.emails()

# Enums
st_color_enum = st.sampled_from(ColorEnum)
st_priority_enum = st.sampled_from(PriorityIntEnum)

# Literal
st_literal = st.sampled_from(["option1", "option2", "option3"])

# Path
st_path = st.builds(Path, st.text(min_size=1, max_size=50))


class NestedModel(BaseModel):
    """Simple nested model."""

    name: str
    value: int
    tags: list[str] = Field(default_factory=list)


class ComplexNestedModel(BaseModel):
    """More complex nested model with various field types."""

    id: UUID
    created_at: datetime
    metadata: dict[str, Any]
    scores: list[float]

    model_config = ConfigDict(str_strip_whitespace=True)


class ModelWithAllBasicTypes(BaseModel):
    """Model with all basic Python types."""

    model_type: Literal["all_basic"] = "all_basic"

    # Basic types
    field_int: int
    field_float: float
    field_str: str
    field_bool: bool
    field_bytes: bytes
    field_none: None

    # Optional
    field_optional_int: int | None = None
    field_optional_str: str | None = None

    # Collections
    field_list: list[int]
    field_set: set[str]
    field_frozenset: frozenset[int]
    field_tuple: tuple[int, str, bool]
    field_dict: dict[str, int]

    # Nested collections
    field_list_of_lists: list[list[int]]
    field_dict_of_lists: dict[str, list[float]]


class ModelWithConstrainedTypes(BaseModel):
    """Model with constrained numeric types."""

    model_type: Literal["constrained"] = "constrained"

    # Constrained integers
    positive_int: PositiveInt
    negative_int: NegativeInt
    non_negative_int: NonNegativeInt
    non_positive_int: NonPositiveInt

    # Constrained floats
    positive_float: PositiveFloat
    negative_float: NegativeFloat
    non_negative_float: NonNegativeFloat
    non_positive_float: NonPositiveFloat

    # Strict types
    strict_int: StrictInt
    strict_float: StrictFloat
    strict_str: StrictStr
    strict_bool: StrictBool

    # Conint/confloat with custom constraints
    bounded_int: conint(ge=10, le=100)  # type: ignore[valid-type]
    bounded_float: confloat(gt=0.0, lt=1.0)  # type: ignore[valid-type]
    multiple_of_five: conint(multiple_of=5)  # type: ignore[valid-type]


class ModelWithStringConstraints(BaseModel):
    """Model with various string constraints."""

    # String length constraints
    min_length_str: constr(min_length=5)  # type: ignore[valid-type]
    max_length_str: constr(max_length=20)  # type: ignore[valid-type]
    length_range_str: constr(min_length=5, max_length=20)  # type: ignore[valid-type]

    # Pattern constraints
    pattern_str: constr(pattern=r"^[A-Z]{3}-\d{4}$")  # type: ignore[valid-type]

    # Annotated with StringConstraints
    annotated_str: Annotated[str, StringConstraints(min_length=3, max_length=10)]

    # Strip whitespace
    stripped_str: constr(strip_whitespace=True)  # type: ignore[valid-type]

    # Uppercase/lowercase
    upper_str: constr(to_upper=True)  # type: ignore[valid-type]
    lower_str: constr(to_lower=True)  # type: ignore[valid-type]


class ModelWithCollectionConstraints(BaseModel):
    """Model with constrained collections."""

    # List constraints
    min_length_list: conlist(int, min_length=2)  # type: ignore[valid-type]
    max_length_list: conlist(str, max_length=5)  # type: ignore[valid-type]
    bounded_list: conlist(float, min_length=1, max_length=10)  # type: ignore[valid-type]

    # Set constraints
    min_set: conset(int, min_length=1)  # type: ignore[valid-type]
    max_set: conset(str, max_length=5)  # type: ignore[valid-type]

    # Frozenset constraints
    min_frozenset: confrozenset(int, min_length=1)  # type: ignore[valid-type]

    # Bytes constraints
    min_bytes: conbytes(min_length=5)  # type: ignore[valid-type]
    max_bytes: conbytes(max_length=100)  # type: ignore[valid-type]


class ModelWithDateTimeTypes(BaseModel):
    """Model with all datetime-related types."""

    field_datetime: datetime
    field_date: date
    field_time: time
    field_timedelta: timedelta

    # Past/Future dates
    past_date: PastDate
    future_date: FutureDate

    # Aware/Naive datetime
    aware_dt: AwareDatetime
    naive_dt: NaiveDatetime


class ModelWithSpecialTypes(BaseModel):
    """Model with special Pydantic types."""

    model_type: Literal["special"] = "special"

    field_uuid: UUID
    field_decimal: Decimal
    field_path: Path

    # IP addresses
    ipv4_addr: IPv4Address
    ipv6_addr: IPv6Address
    any_ip_addr: IPvAnyAddress

    # Networks
    ipv4_net: IPv4Network
    ipv6_net: IPv6Network
    any_ip_net: IPvAnyNetwork

    # Secret types
    secret_str: SecretStr
    secret_bytes: SecretBytes

    # Byte size
    byte_size: ByteSize


class ModelWithEnumsAndLiteralsNoInt(BaseModel):
    """Model with enums and literal types."""

    color: ColorEnum

    # Literal types
    status: Literal["active", "inactive", "pending"]
    version: Literal[1, 2, 3]

    # Optional enum
    optional_color: ColorEnum | None = None


class ModelWithEnumsAndLiterals(ModelWithEnumsAndLiteralsNoInt):
    """Model with enums, literal types, and int enums."""

    priority: PriorityIntEnum


class ModelWithUnionsWithoutListDict(BaseModel):
    """Model with union types."""

    int_or_str: int | str
    multiple_union: int | float | str | bool

    # Optional unions
    optional_union: int | str | None = None


class ModelWithUnions(ModelWithUnionsWithoutListDict):
    """Model with union types."""

    list_or_dict: list[int] | dict[str, int]


class ModelWithNestedModels(BaseModel):
    """Model with nested Pydantic models."""

    simple_nested: NestedModel
    complex_nested: ComplexNestedModel
    optional_nested: NestedModel | None = None

    # Collections of nested models
    list_of_nested: list[NestedModel]
    dict_of_nested: dict[str, NestedModel]


class ModelWithFieldFeatures(BaseModel):
    """Model demonstrating Field features."""

    # Default values
    with_default: int = Field(default=42)
    with_default_factory: list[int] = Field(default_factory=list)

    # Aliases
    field_with_alias: str = Field(alias="fieldAlias")

    # Validation alias
    field_with_validation_alias: int = Field(validation_alias="valAlias")

    # Serialization alias
    field_with_serialization_alias: str = Field(serialization_alias="serAlias")

    # Description and examples
    documented_field: str = Field(
        description="A well-documented field", examples=["example1", "example2"]
    )

    # Exclude from serialization
    internal_field: int = Field(exclude=True)

    # Constraints via Field
    constrained_via_field: int = Field(ge=0, le=100)
    str_constrained_via_field: str = Field(min_length=5, max_length=50)

    # Deprecated field
    deprecated_field: str | None = Field(default=None, deprecated=True)

    # Frozen field
    frozen_field: int = Field(default=10, frozen=True)


class ModelWithValidators(BaseModel):
    """Model with field and model validators."""

    email: str
    age: int
    password: str
    confirm_password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Email validator."""
        if "@" not in v:
            raise ValueError("Invalid email")
        return v.lower()

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        """Age validator."""
        if v < 0 or v > 150:
            raise ValueError("Age must be between 0 and 150")
        return v

    @model_validator(mode="after")
    def validate_passwords_match(self: Self) -> Self:
        """After validator."""
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class ModelWithComputedFields(BaseModel):
    """Model with computed fields."""

    first_name: str
    last_name: str
    birth_year: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def full_name(self: Self) -> str:
        """Computed string field."""
        return f"{self.first_name} {self.last_name}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def age_estimate(self: Self) -> int:
        """Computed int field."""
        return 2025 - self.birth_year


class ModelWithSerializers(BaseModel):
    """Model with custom serializers."""

    value: int
    timestamp: datetime

    @field_serializer("value")
    def serialize_value(self: Self, value: int) -> str:
        """String serializer method."""
        return f"VALUE:{value}"

    @field_serializer("timestamp")
    def serialize_timestamp(self: Self, dt: datetime) -> int:
        """String serializer method."""
        return int(dt.timestamp())


class ModelWithConfig(BaseModel):
    """Model with various config options."""

    name: str
    value: int

    model_config = ConfigDict(
        str_strip_whitespace=True,
        str_min_length=1,
        str_max_length=100,
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
        frozen=False,
        populate_by_name=True,
        use_enum_values=True,
    )


class ModelWithJsonType(BaseModel):
    """Model with JSON type."""

    json_data: Json  # type: ignore[type-arg]
    json_list: Json[list[int]]
    json_dict: Json[dict[str, Any]]


class RecursiveModel(BaseModel):
    """Model with recursive structure."""

    name: str
    value: int
    children: list["RecursiveModel"] = Field(default_factory=list)


class BasicWrapper(BaseModel):
    """Basic wrapper for discriminated union."""

    model_type: Literal["basic"] = "basic"
    data: ModelWithAllBasicTypes


class ConstrainedWrapper(BaseModel):
    """Constrained wrapper for discriminated union."""

    model_type: Literal["constrained"] = "constrained"
    data: ModelWithConstrainedTypes


class SpecialWrapper(BaseModel):
    """Special wrapper for discriminated union."""

    model_type: Literal["special"] = "special"
    data: ModelWithSpecialTypes


class ProtoBufMegaNestedModel(BaseModel):
    """Model serialiazble as ProtoBuf (excludes unions lists/dicts)."""

    basic_types: ModelWithAllBasicTypes
    constrained: ModelWithConstrainedTypes
    strings: ModelWithStringConstraints
    collections: ModelWithCollectionConstraints
    datetimes: ModelWithDateTimeTypes
    special: ModelWithSpecialTypes
    enums: ModelWithEnumsAndLiterals
    unions: ModelWithUnionsWithoutListDict
    nested: ModelWithNestedModels
    field_features: ModelWithFieldFeatures
    recursive: RecursiveModel
    optional_basic: ModelWithAllBasicTypes | None = None
    optional_constrained: ModelWithConstrainedTypes | None = None
    list_of_basic: list[ModelWithAllBasicTypes]
    list_of_special: list[ModelWithSpecialTypes]
    dict_of_enums: dict[str, ModelWithEnumsAndLiterals]
    union_with_none: ModelWithAllBasicTypes | None
    union_primitive_model: int | str | ModelWithEnumsAndLiterals
    model_union: Annotated[
        Annotated[ModelWithAllBasicTypes, Tag("basic")]
        | Annotated[ModelWithConstrainedTypes, Tag("constrained")]
        | Annotated[ModelWithSpecialTypes, Tag("special")],
        Discriminator("model_type"),
    ] = Field(discriminator="model_type")
    nested_discriminated: (
        Annotated["MegaNestedModel", Tag("mega")]
        | Annotated[ModelWithAllBasicTypes, Tag("basic")]
    )
    optional_union_chain: str | int | ModelWithSpecialTypes | None


class AvroMegaNestedModel(BaseModel):
    """Model serialiazble as ProtoBuf (excludes unions lists/dicts)."""

    basic_types: ModelWithAllBasicTypes
    constrained: ModelWithConstrainedTypes
    strings: ModelWithStringConstraints
    collections: ModelWithCollectionConstraints
    datetimes: ModelWithDateTimeTypes
    special: ModelWithSpecialTypes
    enums: ModelWithEnumsAndLiteralsNoInt
    unions: ModelWithUnions
    nested: ModelWithNestedModels
    field_features: ModelWithFieldFeatures
    recursive: RecursiveModel
    optional_basic: ModelWithAllBasicTypes | None = None
    optional_constrained: ModelWithConstrainedTypes | None = None
    list_of_basic: list[ModelWithAllBasicTypes]
    list_of_special: list[ModelWithSpecialTypes]
    dict_of_enums: dict[str, ModelWithEnumsAndLiteralsNoInt]
    union_with_none: ModelWithAllBasicTypes | None
    union_primitive_model: int | str | ModelWithEnumsAndLiteralsNoInt
    model_union: Annotated[
        Annotated[ModelWithAllBasicTypes, Tag("basic")]
        | Annotated[ModelWithConstrainedTypes, Tag("constrained")]
        | Annotated[ModelWithSpecialTypes, Tag("special")],
        Discriminator("model_type"),
    ] = Field(discriminator="model_type")
    nested_discriminated: (
        Annotated["MegaNestedModel", Tag("mega")]
        | Annotated[ModelWithAllBasicTypes, Tag("basic")]
    )
    optional_union_chain: str | int | ModelWithSpecialTypes | None
    list_of_discriminated_unions: list[BasicWrapper | ConstrainedWrapper]
    deeply_optional: dict[str, list[ModelWithNestedModels | None]] | None
    union_of_lists: list[int] | list[ModelWithAllBasicTypes]


class MegaNestedModel(ProtoBufMegaNestedModel):
    """Put it all together."""

    unions: ModelWithUnions
    list_of_discriminated_unions: list[BasicWrapper | ConstrainedWrapper]
    deeply_optional: dict[str, list[ModelWithNestedModels | None]] | None
    union_of_lists: list[int] | list[ModelWithAllBasicTypes]


# Forward reference resolution
RecursiveModel.model_rebuild()


@st.composite
def nested_model_strategy(draw: DrawFn) -> NestedModel:
    """Creates a nested model."""
    return NestedModel(
        name=draw(st_str),
        value=draw(st_int),
        tags=draw(st_list_str).copy() if draw(st_bool) else [],
    )


@st.composite
def complex_nested_model_strategy(draw: DrawFn) -> ComplexNestedModel:
    """Creates a complex nested model."""
    return ComplexNestedModel(
        id=draw(st_uuid),
        created_at=draw(st_datetime),
        metadata=draw(st_dict_str_int),
        scores=draw(st.lists(st_positive_float, min_size=0, max_size=5)),
    )


@st.composite
def model_with_all_basic_types_strategy(draw: DrawFn) -> ModelWithAllBasicTypes:
    """Creates a model with all basic types."""
    return ModelWithAllBasicTypes(
        field_int=draw(st_int),
        field_float=draw(st_float),
        field_str=draw(st_str),
        field_bool=draw(st_bool),
        field_bytes=draw(st_bytes),
        field_none=None,
        field_optional_int=draw(st.one_of(st_int, st_none)),
        field_optional_str=draw(st.one_of(st_str, st_none)),
        field_list=draw(st_list_int),
        field_set=draw(st_set_str),
        field_frozenset=draw(st_frozenset_int),
        field_tuple=draw(st_tuple_mixed),
        field_dict=draw(st_dict_str_int),
        field_list_of_lists=draw(st.lists(st_list_int, min_size=0, max_size=3)),
        field_dict_of_lists=draw(
            st.dictionaries(
                st_str,
                st.lists(st_float, min_size=0, max_size=5),
                min_size=0,
                max_size=5,
            )
        ),
    )


@st.composite
def model_with_constrained_types_strategy(draw: DrawFn) -> ModelWithConstrainedTypes:
    """Creates a model with constrained types."""
    return ModelWithConstrainedTypes(
        positive_int=draw(st_positive_int),
        negative_int=draw(st_negative_int),
        non_negative_int=draw(st_non_negative_int),
        non_positive_int=draw(st_non_positive_int),
        positive_float=draw(st_positive_float),
        negative_float=draw(st_negative_float),
        non_negative_float=draw(st_non_negative_float),
        non_positive_float=draw(st_non_positive_float),
        strict_int=draw(st_int),
        strict_float=draw(st_float),
        strict_str=draw(st_str),
        strict_bool=draw(st_bool),
        bounded_int=draw(st.integers(min_value=10, max_value=100)),
        bounded_float=draw(
            st.floats(min_value=0.001, max_value=0.999, allow_nan=False)
        ),
        multiple_of_five=draw(
            st.integers(min_value=-100, max_value=100).map(lambda x: (x // 5) * 5)
        ),
    )


@st.composite
def model_with_string_constraints_strategy(draw: DrawFn) -> ModelWithStringConstraints:
    """Creates a model with string constraints."""
    return ModelWithStringConstraints(
        min_length_str=draw(st.text(min_size=5, max_size=50)),
        max_length_str=draw(st.text(min_size=0, max_size=20)),
        length_range_str=draw(st.text(min_size=5, max_size=20)),
        pattern_str=draw(st_pattern_str),
        annotated_str=draw(st.text(min_size=3, max_size=10)),
        stripped_str=draw(st_str),
        upper_str=draw(st_str),
        lower_str=draw(st_str),
    )


@st.composite
def model_with_collection_constraints_strategy(
    draw: DrawFn,
) -> ModelWithCollectionConstraints:
    """Creates a model with collection constraints."""
    return ModelWithCollectionConstraints(
        min_length_list=draw(st.lists(st_int, min_size=2, max_size=10)),
        max_length_list=draw(st.lists(st_str, min_size=0, max_size=5)),
        bounded_list=draw(st.lists(st_float, min_size=1, max_size=10)),
        min_set=draw(st.sets(st_int, min_size=1, max_size=10)),
        max_set=draw(st.sets(st_str, min_size=0, max_size=5)),
        min_frozenset=draw(st.frozensets(st_int, min_size=1, max_size=10)),
        min_bytes=draw(st.binary(min_size=5, max_size=50)),
        max_bytes=draw(st.binary(min_size=0, max_size=100)),
    )


@st.composite
def model_with_datetime_types_strategy(draw: DrawFn) -> ModelWithDateTimeTypes:
    """Creates a model with datetime types."""
    past = draw(st.dates(min_value=date(1900, 1, 1), max_value=date.today()))
    future = draw(
        st.dates(
            min_value=date.today() + timedelta(days=7),
            max_value=date(2100, 12, 31),
        )
    )

    return ModelWithDateTimeTypes(
        field_datetime=draw(st_datetime),
        field_date=draw(st_date),
        field_time=draw(st_time),
        field_timedelta=draw(st_timedelta),
        past_date=past,
        future_date=future,
        aware_dt=draw(st_datetime).replace(tzinfo=UTC),
        naive_dt=draw(st_datetime).replace(tzinfo=None),
    )


@st.composite
def model_with_special_types_strategy(draw: DrawFn) -> ModelWithSpecialTypes:
    """Creates a model with special types."""
    return ModelWithSpecialTypes(
        field_uuid=draw(st_uuid),
        field_decimal=draw(st_decimal),
        field_path=draw(st_path),
        ipv4_addr=draw(st_ipv4),
        ipv6_addr=draw(st_ipv6),
        any_ip_addr=draw(st.one_of(st_ipv4, st_ipv6)),
        ipv4_net=draw(st_ipv4_network),
        ipv6_net=draw(st_ipv6_network),
        any_ip_net=draw(st.one_of(st_ipv4_network, st_ipv6_network)),
        secret_str=SecretStr(draw(st_str)),
        secret_bytes=SecretBytes(draw(st_bytes)),
        byte_size=draw(st_positive_int),  # type: ignore[arg-type]
    )


@st.composite
def model_with_enums_and_literals_no_int_strategy(
    draw: DrawFn,
) -> ModelWithEnumsAndLiteralsNoInt:
    """Creates a model with enums and literals."""
    return ModelWithEnumsAndLiteralsNoInt(
        color=draw(st_color_enum),
        status=draw(st.sampled_from(["active", "inactive", "pending"])),
        version=draw(st.sampled_from([1, 2, 3])),
        optional_color=draw(st.one_of(st_color_enum, st_none)),
    )


@st.composite
def model_with_enums_and_literals_strategy(draw: DrawFn) -> ModelWithEnumsAndLiterals:
    """Creates a model with enums and literals."""
    return ModelWithEnumsAndLiterals(
        color=draw(st_color_enum),
        priority=draw(st_priority_enum),
        status=draw(st.sampled_from(["active", "inactive", "pending"])),
        version=draw(st.sampled_from([1, 2, 3])),
        optional_color=draw(st.one_of(st_color_enum, st_none)),
    )


@st.composite
def model_with_unions_without_list_dict_strategy(
    draw: DrawFn,
) -> ModelWithUnionsWithoutListDict:
    """Creates a model with unions, without list or dicts."""
    return ModelWithUnionsWithoutListDict(
        int_or_str=draw(st.one_of(st_int, st_str)),
        multiple_union=draw(st.one_of(st_int, st_float, st_str, st_bool)),
        optional_union=draw(st.one_of(st_int, st_str, st_none)),
    )


@st.composite
def model_with_unions_strategy(draw: DrawFn) -> ModelWithUnions:
    """Creates a model with unions."""
    return ModelWithUnions(
        int_or_str=draw(st.one_of(st_int, st_str)),
        list_or_dict=draw(st.one_of(st_list_int, st_dict_str_int)),
        multiple_union=draw(st.one_of(st_int, st_float, st_str, st_bool)),
        optional_union=draw(st.one_of(st_int, st_str, st_none)),
    )


@st.composite
def model_with_nested_models_strategy(draw: DrawFn) -> ModelWithNestedModels:
    """Creates a model with nested models."""
    return ModelWithNestedModels(
        simple_nested=draw(nested_model_strategy()),
        complex_nested=draw(complex_nested_model_strategy()),
        optional_nested=draw(st.one_of(nested_model_strategy(), st_none)),
        list_of_nested=draw(st.lists(nested_model_strategy(), min_size=0, max_size=3)),
        dict_of_nested=draw(
            st.dictionaries(st_str, nested_model_strategy(), min_size=0, max_size=3)
        ),
    )


@st.composite
def model_with_field_features_strategy(draw: DrawFn) -> ModelWithFieldFeatures:
    """Creates a model with field features."""
    return ModelWithFieldFeatures(  # type: ignore[call-arg]
        fieldAlias=draw(st_str),  # Using alias
        valAlias=draw(st_int),  # Using validation alias
        field_with_serialization_alias=draw(st_str),  # Using serialization alias
        documented_field=draw(st_str),
        internal_field=draw(st_int),
        constrained_via_field=draw(st.integers(min_value=0, max_value=100)),
        str_constrained_via_field=draw(st.text(min_size=5, max_size=50)),
        deprecated_field=draw(st.one_of(st_str, st_none)),
    )


@st.composite
def recursive_model_strategy(draw: DrawFn, max_depth: int = 2) -> RecursiveModel:
    """Creates a recursive model."""
    name = draw(st_str)
    value = draw(st_int)

    if max_depth <= 0:
        return RecursiveModel(name=name, value=value, children=[])

    num_children = draw(st.integers(min_value=0, max_value=2))
    children = [
        draw(recursive_model_strategy(max_depth=max_depth - 1))
        for _ in range(num_children)
    ]

    return RecursiveModel(name=name, value=value, children=children)


@st.composite
def nested_discriminated_field_strategy(
    draw: DrawFn, current_depth: int
) -> MegaNestedModel | ModelWithAllBasicTypes:
    """Creates a nested discriminator."""
    if current_depth <= 0:
        return draw(model_with_all_basic_types_strategy())
    return draw(
        st.one_of(
            mega_nested_model_strategy(max_depth=current_depth - 1),
            model_with_all_basic_types_strategy(),
        )
    )


protobuf_model_strategies = st.one_of(
    model_with_all_basic_types_strategy(),
    model_with_constrained_types_strategy(),
    model_with_string_constraints_strategy(),
    model_with_collection_constraints_strategy(),
    model_with_datetime_types_strategy(),
    model_with_special_types_strategy(),
    model_with_enums_and_literals_strategy(),
    model_with_unions_without_list_dict_strategy(),
    model_with_nested_models_strategy(),
    model_with_field_features_strategy(),
    recursive_model_strategy(),
)

avro_model_strategies = st.one_of(
    model_with_all_basic_types_strategy(),
    model_with_constrained_types_strategy(),
    model_with_string_constraints_strategy(),
    model_with_collection_constraints_strategy(),
    model_with_datetime_types_strategy(),
    model_with_special_types_strategy(),
    model_with_enums_and_literals_no_int_strategy(),
    model_with_unions_without_list_dict_strategy(),
    model_with_nested_models_strategy(),
    model_with_field_features_strategy(),
    recursive_model_strategy(),
)

all_model_strategies = st.one_of(
    model_with_all_basic_types_strategy(),
    model_with_constrained_types_strategy(),
    model_with_string_constraints_strategy(),
    model_with_collection_constraints_strategy(),
    model_with_datetime_types_strategy(),
    model_with_special_types_strategy(),
    model_with_enums_and_literals_strategy(),
    model_with_unions_strategy(),
    model_with_nested_models_strategy(),
    model_with_field_features_strategy(),
    recursive_model_strategy(),
)


@st.composite
def protobuf_mega_nested_model_strategy(
    draw: DrawFn, max_depth: int = 3
) -> ProtoBufMegaNestedModel:
    """Generate a massive nested model containing all other model types."""
    return ProtoBufMegaNestedModel(
        basic_types=draw(model_with_all_basic_types_strategy()),
        constrained=draw(model_with_constrained_types_strategy()),
        strings=draw(model_with_string_constraints_strategy()),
        collections=draw(model_with_collection_constraints_strategy()),
        datetimes=draw(model_with_datetime_types_strategy()),
        special=draw(model_with_special_types_strategy()),
        enums=draw(model_with_enums_and_literals_strategy()),
        unions=draw(model_with_unions_without_list_dict_strategy()),
        nested=draw(model_with_nested_models_strategy()),
        field_features=draw(model_with_field_features_strategy()),
        recursive=draw(recursive_model_strategy(max_depth=max_depth)),
        optional_basic=draw(st.one_of(model_with_all_basic_types_strategy(), st_none)),
        optional_constrained=draw(
            st.one_of(model_with_constrained_types_strategy(), st_none)
        ),
        list_of_basic=draw(
            st.lists(model_with_all_basic_types_strategy(), min_size=0, max_size=2)
        ),
        list_of_special=draw(
            st.lists(model_with_special_types_strategy(), min_size=0, max_size=2)
        ),
        dict_of_enums=draw(
            st.dictionaries(
                st_str, model_with_enums_and_literals_strategy(), min_size=0, max_size=2
            )
        ),
        model_union=draw(
            st.one_of(
                model_with_all_basic_types_strategy(),
                model_with_constrained_types_strategy(),
                model_with_special_types_strategy(),
            )
        ),
        nested_discriminated=draw(nested_discriminated_field_strategy(max_depth - 1)),
        union_with_none=draw(st.one_of(model_with_all_basic_types_strategy(), st_none)),
        union_primitive_model=draw(
            st.one_of(st_int, st_str, model_with_enums_and_literals_strategy())
        ),
        optional_union_chain=draw(
            st.one_of(st_none, st_str, st_int, model_with_special_types_strategy())
        ),
    )


@st.composite
def avro_mega_nested_model_strategy(
    draw: DrawFn, max_depth: int = 3
) -> AvroMegaNestedModel:
    """Generate a massive nested model containing all other model types."""
    return AvroMegaNestedModel(
        basic_types=draw(model_with_all_basic_types_strategy()),
        constrained=draw(model_with_constrained_types_strategy()),
        strings=draw(model_with_string_constraints_strategy()),
        collections=draw(model_with_collection_constraints_strategy()),
        datetimes=draw(model_with_datetime_types_strategy()),
        special=draw(model_with_special_types_strategy()),
        enums=draw(model_with_enums_and_literals_no_int_strategy()),
        unions=draw(model_with_unions_strategy()),
        nested=draw(model_with_nested_models_strategy()),
        field_features=draw(model_with_field_features_strategy()),
        recursive=draw(recursive_model_strategy(max_depth=max_depth)),
        optional_basic=draw(st.one_of(model_with_all_basic_types_strategy(), st_none)),
        optional_constrained=draw(
            st.one_of(model_with_constrained_types_strategy(), st_none)
        ),
        list_of_basic=draw(
            st.lists(model_with_all_basic_types_strategy(), min_size=0, max_size=2)
        ),
        list_of_special=draw(
            st.lists(model_with_special_types_strategy(), min_size=0, max_size=2)
        ),
        dict_of_enums=draw(
            st.dictionaries(
                st_str,
                model_with_enums_and_literals_no_int_strategy(),
                min_size=0,
                max_size=2,
            )
        ),
        model_union=draw(
            st.one_of(
                model_with_all_basic_types_strategy(),
                model_with_constrained_types_strategy(),
                model_with_special_types_strategy(),
            )
        ),
        nested_discriminated=draw(nested_discriminated_field_strategy(max_depth - 1)),
        union_with_none=draw(st.one_of(model_with_all_basic_types_strategy(), st_none)),
        union_primitive_model=draw(
            st.one_of(st_int, st_str, model_with_enums_and_literals_no_int_strategy())
        ),
        list_of_discriminated_unions=draw(
            st.lists(
                st.one_of(
                    st.builds(BasicWrapper, data=model_with_all_basic_types_strategy()),
                    st.builds(
                        ConstrainedWrapper, data=model_with_constrained_types_strategy()
                    ),
                ),
                min_size=0,
                max_size=2,
            )
        ),
        deeply_optional=draw(
            st.one_of(
                st.none(),
                st.dictionaries(
                    st_str,
                    st.lists(
                        st.one_of(model_with_nested_models_strategy(), st_none),
                        min_size=0,
                        max_size=2,
                    ),
                    min_size=0,
                    max_size=2,
                ),
            )
        ),
        union_of_lists=draw(
            st.one_of(
                st_list_int,
                st.lists(model_with_all_basic_types_strategy(), min_size=0, max_size=2),
            )
        ),
        optional_union_chain=draw(
            st.one_of(st_none, st_str, st_int, model_with_special_types_strategy())
        ),
    )


@st.composite
def mega_nested_model_strategy(draw: DrawFn, max_depth: int = 3) -> MegaNestedModel:
    """Generate a massive nested model containing all other model types."""
    return MegaNestedModel(
        basic_types=draw(model_with_all_basic_types_strategy()),
        constrained=draw(model_with_constrained_types_strategy()),
        strings=draw(model_with_string_constraints_strategy()),
        collections=draw(model_with_collection_constraints_strategy()),
        datetimes=draw(model_with_datetime_types_strategy()),
        special=draw(model_with_special_types_strategy()),
        enums=draw(model_with_enums_and_literals_strategy()),
        unions=draw(model_with_unions_strategy()),
        nested=draw(model_with_nested_models_strategy()),
        field_features=draw(model_with_field_features_strategy()),
        recursive=draw(recursive_model_strategy(max_depth=max_depth)),
        optional_basic=draw(st.one_of(model_with_all_basic_types_strategy(), st_none)),
        optional_constrained=draw(
            st.one_of(model_with_constrained_types_strategy(), st_none)
        ),
        list_of_basic=draw(
            st.lists(model_with_all_basic_types_strategy(), min_size=0, max_size=2)
        ),
        list_of_special=draw(
            st.lists(model_with_special_types_strategy(), min_size=0, max_size=2)
        ),
        dict_of_enums=draw(
            st.dictionaries(
                st_str, model_with_enums_and_literals_strategy(), min_size=0, max_size=2
            )
        ),
        model_union=draw(
            st.one_of(
                model_with_all_basic_types_strategy(),
                model_with_constrained_types_strategy(),
                model_with_special_types_strategy(),
            )
        ),
        nested_discriminated=draw(nested_discriminated_field_strategy(max_depth - 1)),
        union_with_none=draw(st.one_of(model_with_all_basic_types_strategy(), st_none)),
        union_primitive_model=draw(
            st.one_of(st_int, st_str, model_with_enums_and_literals_strategy())
        ),
        list_of_discriminated_unions=draw(
            st.lists(
                st.one_of(
                    st.builds(BasicWrapper, data=model_with_all_basic_types_strategy()),
                    st.builds(
                        ConstrainedWrapper, data=model_with_constrained_types_strategy()
                    ),
                ),
                min_size=0,
                max_size=2,
            )
        ),
        deeply_optional=draw(
            st.one_of(
                st.none(),
                st.dictionaries(
                    st_str,
                    st.lists(
                        st.one_of(model_with_nested_models_strategy(), st_none),
                        min_size=0,
                        max_size=2,
                    ),
                    min_size=0,
                    max_size=2,
                ),
            )
        ),
        union_of_lists=draw(
            st.one_of(
                st_list_int,
                st.lists(model_with_all_basic_types_strategy(), min_size=0, max_size=2),
            )
        ),
        optional_union_chain=draw(
            st.one_of(st_none, st_str, st_int, model_with_special_types_strategy())
        ),
    )


@pytest.mark.integration
@given(model=all_model_strategies)
@settings(max_examples=20)
def test_json_schema_export_model_all_strats(model: BaseModel) -> None:
    """Test JSON Schema serialization with any complex model type."""
    manager = ModelManager()
    manager.model("Model", "1.0.0")(type(model))
    schema = manager.get_schema("Model", "1.0.0")

    json.dumps(schema, indent=2)


@pytest.mark.integration
@given(model=avro_model_strategies)
@settings(max_examples=20)
def test_avro_export_model_all_strats(model: BaseModel) -> None:
    """Test Avro serialization with any complex model type."""
    manager = ModelManager()
    manager.model("Model", "1.0.0")(type(model))
    schema = manager.get_avro_schema("Model", "1.0.0")

    json.dumps(schema, indent=2)
    fastavro.parse_schema(dict(schema))


@pytest.mark.integration
@given(model=protobuf_model_strategies)
@settings(max_examples=20)
def test_protobuf_schema_export_model_all_strats(model: BaseModel) -> None:
    """Test Protobuf serialization with any complex model type."""
    manager = ModelManager()
    manager.model("Model", "1.0.0")(type(model))

    proto_content = manager.get_proto_schema("Model", "1.0.0")

    if not shutil.which("protoc"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        proto_path = Path(tmpdir) / "complex.proto"
        proto_path.write_text(proto_content)

        result = subprocess.run(
            [
                "protoc",
                f"--proto_path={tmpdir}",
                f"--python_out={tmpdir}",
                "complex.proto",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, (
            f"protoc failed to compile generated proto.\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}\n"
            f"Proto:\n{proto_content}"
        )


@pytest.mark.integration
@given(model=all_model_strategies)
@settings(max_examples=20, deadline=5000)
def test_typescript_schema_export_model_all_strats(model: BaseModel) -> None:
    """Test TypeScript serialization with any complex model type."""
    manager = ModelManager()
    manager.model("Model", "1.0.0")(type(model))
    schema = manager.get_typescript_schema("Model", "1.0.0")

    if not shutil.which("tsc"):
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        f.write(schema)
        temp_path = f.name

    result = subprocess.run(
        ["tsc", "--noEmit", "--strict", temp_path],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"TypeScript validation failed: {result.stderr}"


@pytest.mark.integration
@given(model=st.one_of(mega_nested_model_strategy()))
@settings(max_examples=20)
def test_json_schema_export_model(model: BaseModel) -> None:
    """Test JSON Schema serialization with any complex model type."""
    manager = ModelManager()
    manager.model("Model", "1.0.0")(type(model))
    schema = manager.get_schema("Model", "1.0.0")

    json.dumps(schema, indent=2)


@pytest.mark.integration
@given(model=st.one_of(avro_mega_nested_model_strategy()))
@settings(max_examples=20)
def test_avro_export_model(model: BaseModel) -> None:
    """Test Avro serialization with any complex model type."""
    manager = ModelManager()
    manager.model("Model", "1.0.0")(type(model))
    schema = manager.get_avro_schema("Model", "1.0.0")
    json.dumps(schema, indent=2)
    fastavro.parse_schema(dict(schema))


@pytest.mark.integration
@given(model=st.one_of(protobuf_mega_nested_model_strategy()))
@settings(max_examples=20)
def test_protobuf_schema_export_model(model: BaseModel) -> None:
    """Test Protobuf serialization with any complex model type."""
    manager = ModelManager()
    manager.model("Model", "1.0.0")(type(model))

    proto_content = manager.get_proto_schema("Model", "1.0.0")

    if not shutil.which("protoc"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        proto_path = Path(tmpdir) / "complex.proto"
        proto_path.write_text(proto_content)

        result = subprocess.run(
            [
                "protoc",
                f"--proto_path={tmpdir}",
                f"--python_out={tmpdir}",
                "complex.proto",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, (
            f"protoc failed to compile generated proto.\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}\n"
            f"Proto:\n{proto_content}"
        )


@pytest.mark.integration
@given(model=st.one_of(mega_nested_model_strategy()))
@settings(max_examples=20, deadline=5000)
def test_typescript_schema_export_model(model: BaseModel) -> None:
    """Test TypeScript serialization with any complex model type."""
    manager = ModelManager()
    manager.model("Model", "1.0.0")(type(model))

    schema = manager.get_typescript_schema("Model", "1.0.0")

    if not shutil.which("tsc"):
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        f.write(schema)
        temp_path = f.name

    result = subprocess.run(
        ["tsc", "--noEmit", "--strict", temp_path],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"TypeScript validation failed: {result.stderr}"
