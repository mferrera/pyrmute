"""Tests for TypeInspector utility class."""

from collections.abc import Mapping, MutableMapping
from enum import Enum
from typing import Any, Union, get_origin

import pytest
from pydantic import BaseModel, Field, RootModel

from pyrmute._type_inspector import TypeInspector

# ruff: noqa: PLR2004


class Color(str, Enum):
    """Test enum for color values."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Status(int, Enum):
    """Test enum for status codes."""

    PENDING = 0
    ACTIVE = 1
    COMPLETED = 2


class Address(BaseModel):
    """Test model for addresses."""

    street: str
    city: str
    zip_code: str


class Person(BaseModel):
    """Test model for person with nested Address."""

    name: str
    age: int
    address: Address


class RecursiveNode(BaseModel):
    """Test model with self-reference."""

    value: int
    children: list["RecursiveNode"] = []


@pytest.mark.parametrize(
    "origin, expected",
    [
        (Union, True),
        (list, False),
        (dict, False),
        (None, False),
        (str, False),
    ],
)
def test_is_union_type(origin: Any, expected: bool) -> None:
    """Test that is_union_type correctly identifies Union types."""
    assert TypeInspector.is_union_type(origin) == expected


def test_is_union_type_with_pipe_syntax() -> None:
    """Test that is_union_type handles Python 3.10+ pipe syntax (str | int)."""
    annotation = str | int
    origin = get_origin(annotation)
    assert TypeInspector.is_union_type(origin) is True


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (str | None, True),
        (int | None, True),
        (str | int | None, True),
        (str, False),
        (int, False),
        (str | int, False),
    ],
)
def test_is_optional_type(annotation: Any, expected: bool) -> None:
    """Test that is_optional_type correctly identifies Optional types."""
    assert TypeInspector.is_optional_type(annotation) == expected


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (str | None, ["str"]),
        (str | int, ["str", "int"]),
        (str | int | None, ["str", "int"]),
        (str | int | bool | None, ["str", "int", "bool"]),
    ],
)
def test_get_non_none_union_args(annotation: Any, expected: list[str]) -> None:
    """Test that get_non_none_union_args extracts non-None types from unions."""
    result = TypeInspector.get_non_none_union_args(annotation)
    result_names = [t.__name__ for t in result]
    assert result_names == expected


def test_get_non_none_union_args_non_union() -> None:
    """Test that get_non_none_union_args returns the type itself for non-unions."""
    result = TypeInspector.get_non_none_union_args(str)
    assert result == [str]


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (str | int, True),
        (str | int | bool, True),
        (str | None, False),
        (int | None, False),
        (str, False),
    ],
)
def test_is_union_requiring_oneof(annotation: Any, expected: bool) -> None:
    """Test is_union_requiring_oneof identifies multi-type unions (not Optional)."""
    assert TypeInspector.is_union_requiring_oneof(annotation) == expected


@pytest.mark.parametrize(
    "origin, expected",
    [
        (list, True),
        (set, True),
        (frozenset, True),
        (tuple, False),
        (dict, False),
        (None, False),
    ],
)
def test_is_list_like(origin: Any, expected: bool) -> None:
    """Test that is_list_like identifies list-like container types."""
    assert TypeInspector.is_list_like(origin) == expected


def test_is_list_like_with_generic() -> None:
    """Test that is_list_like handles generic list types."""
    annotation = list[str]
    origin = get_origin(annotation)
    assert TypeInspector.is_list_like(origin) is True


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (dict, True),
        (dict[str, int], True),
        (Mapping[str, int], True),
        (MutableMapping[str, int], True),
        (Mapping, True),
        (MutableMapping, True),
        (list, False),
        (str, False),
    ],
)
def test_is_dict_like(annotation: Any, expected: bool) -> None:
    """Test that is_dict_like identifies dict-like mapping types."""
    origin = get_origin(annotation)
    # Origin will be None if list, str
    assert TypeInspector.is_dict_like(origin or annotation, annotation) == expected


@pytest.mark.parametrize(
    "python_type, expected",
    [
        (Address, True),
        (Person, True),
        (RecursiveNode, True),
        (str, False),
        (int, False),
        (Color, False),
    ],
)
def test_is_base_model(python_type: Any, expected: bool) -> None:
    """Test that is_base_model correctly identifies Pydantic BaseModel types."""
    assert TypeInspector.is_base_model(python_type) == expected


@pytest.mark.parametrize(
    "python_type, expected",
    [
        (Color, True),
        (Status, True),
        (str, False),
        (int, False),
        (Address, False),
    ],
)
def test_is_enum(python_type: Any, expected: bool) -> None:
    """Test that is_enum correctly identifies Enum types."""
    assert TypeInspector.is_enum(python_type) == expected


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (tuple[str, ...], True),
        (tuple[int, ...], True),
        (tuple[str, int], False),
        (tuple[str, int, bool], False),
        (tuple, False),
        (str, False),
    ],
)
def test_is_variable_length_tuple(annotation: Any, expected: bool) -> None:
    """Test that is_variable_length_tuple identifies tuple[T, ...] patterns."""
    assert TypeInspector.is_variable_length_tuple(annotation) == expected


@pytest.mark.parametrize(
    "annotation, expected_count, expected_types",
    [
        (tuple[str, ...], 1, [str]),
        (tuple[int, ...], 1, [int]),
        (tuple[str, int], 2, [str, int]),
        (tuple[str, int, bool], 3, [str, int, bool]),
        (tuple, 0, []),
    ],
)
def test_get_tuple_element_types(
    annotation: Any, expected_count: int, expected_types: list[type]
) -> None:
    """Test that get_tuple_element_types extracts element types from tuples."""
    result = TypeInspector.get_tuple_element_types(annotation)
    assert len(result) == expected_count
    assert result == expected_types


def test_get_tuple_element_types_non_tuple() -> None:
    """Test that get_tuple_element_types returns empty list for non-tuple types."""
    assert TypeInspector.get_tuple_element_types(str) == []


def test_collect_nested_models_simple() -> None:
    """Test that collect_nested_models finds directly nested models."""
    result = TypeInspector.collect_nested_models(Person)
    assert "Address" in result
    assert result["Address"] == Address


def test_collect_nested_models_no_nesting() -> None:
    """Test that collect_nested_models returns empty dict for models with no nesting."""
    result = TypeInspector.collect_nested_models(Address)
    assert result == {}


def test_collect_nested_models_recursive() -> None:
    """Test that collect_nested_models handles self-referencing models."""
    result = TypeInspector.collect_nested_models(RecursiveNode)
    assert "RecursiveNode" in result


def test_collect_nested_models_in_collections() -> None:
    """Test that collect_nested_models finds models inside collections."""

    class Container(BaseModel):
        addresses: list[Address]
        person_map: dict[str, Person]

    result = TypeInspector.collect_nested_models(Container)
    assert "Address" in result
    assert "Person" in result


def test_collect_nested_models_in_unions() -> None:
    """Test that collect_nested_models finds models inside unions."""

    class Document(BaseModel):
        author: Person | str

    result = TypeInspector.collect_nested_models(Document)
    assert "Person" in result
    assert "Address" in result


def test_get_numeric_constraints_all_present() -> None:
    """Test that get_numeric_constraints extracts all numeric constraint types."""

    class Constrained(BaseModel):
        value: int = Field(ge=0, le=100)

    field_info = Constrained.model_fields["value"]
    constraints = TypeInspector.get_numeric_constraints(field_info)

    assert constraints["ge"] == 0
    assert constraints["le"] == 100
    assert constraints["gt"] is None
    assert constraints["lt"] is None


def test_get_numeric_constraints_gt_lt() -> None:
    """Test that get_numeric_constraints handles gt/lt (exclusive bounds)."""

    class Constrained(BaseModel):
        value: int = Field(gt=0, lt=100)

    field_info = Constrained.model_fields["value"]
    constraints = TypeInspector.get_numeric_constraints(field_info)

    assert constraints["gt"] == 0
    assert constraints["lt"] == 100
    assert constraints["ge"] is None
    assert constraints["le"] is None


def test_get_numeric_constraints_none() -> None:
    """Test that get_numeric_constraints returns None values when no constraints."""

    class Unconstrained(BaseModel):
        value: int

    field_info = Unconstrained.model_fields["value"]
    constraints = TypeInspector.get_numeric_constraints(field_info)

    assert constraints["ge"] is None
    assert constraints["gt"] is None
    assert constraints["le"] is None
    assert constraints["lt"] is None


def test_get_string_constraints_all_present() -> None:
    """Test that get_string_constraints extracts all string constraint types."""

    class Constrained(BaseModel):
        value: str = Field(pattern=r"^\d+$", min_length=1, max_length=10)

    field_info = Constrained.model_fields["value"]
    constraints = TypeInspector.get_string_constraints(field_info)

    assert constraints["pattern"] == r"^\d+$"
    assert constraints["min_length"] == 1
    assert constraints["max_length"] == 10


def test_get_string_constraints_none() -> None:
    """Test that get_string_constraints returns None values when no constraints."""

    class Unconstrained(BaseModel):
        value: str

    field_info = Unconstrained.model_fields["value"]
    constraints = TypeInspector.get_string_constraints(field_info)

    assert constraints["pattern"] is None
    assert constraints["min_length"] is None
    assert constraints["max_length"] is None


def test_can_fit_in_32bit_int_true() -> None:
    """Test that can_fit_in_32bit_int returns True for values within 32-bit range."""

    class Model(BaseModel):
        value: int = Field(ge=-100, le=100)

    field_info = Model.model_fields["value"]
    assert TypeInspector.can_fit_in_32bit_int(field_info) is True


def test_can_fit_in_32bit_int_false_too_large() -> None:
    """Test that can_fit_in_32bit_int returns False when max exceeds 32-bit."""

    class Model(BaseModel):
        value: int = Field(ge=0, le=5_000_000_000)

    field_info = Model.model_fields["value"]
    assert TypeInspector.can_fit_in_32bit_int(field_info) is False


def test_can_fit_in_32bit_int_false_too_small() -> None:
    """Test that can_fit_in_32bit_int returns False when min below 32-bit."""

    class Model(BaseModel):
        value: int = Field(ge=-5_000_000_000, le=0)

    field_info = Model.model_fields["value"]
    assert TypeInspector.can_fit_in_32bit_int(field_info) is False


def test_can_fit_in_32bit_int_false_no_constraints() -> None:
    """Test that can_fit_in_32bit_int returns False when constraints missing."""

    class Model(BaseModel):
        value: int

    field_info = Model.model_fields["value"]
    assert TypeInspector.can_fit_in_32bit_int(field_info) is False


def test_can_fit_in_32bit_int_false_partial_constraints() -> None:
    """Test that can_fit_in_32bit_int returns False with only one constraint."""

    class Model(BaseModel):
        value: int = Field(ge=0)

    field_info = Model.model_fields["value"]
    assert TypeInspector.can_fit_in_32bit_int(field_info) is False


def test_can_fit_in_32bit_int_boundary_values() -> None:
    """Test that can_fit_in_32bit_int handles exact 32-bit boundaries."""

    class Model(BaseModel):
        value: int = Field(ge=-(2**31), le=(2**31 - 1))

    field_info = Model.model_fields["value"]
    assert TypeInspector.can_fit_in_32bit_int(field_info) is True


def test_is_unsigned_int_true() -> None:
    """Test that is_unsigned_int returns True for non-negative constraints."""

    class Model(BaseModel):
        value: int = Field(ge=0)

    field_info = Model.model_fields["value"]
    assert TypeInspector.is_unsigned_int(field_info) is True


def test_is_unsigned_int_false() -> None:
    """Test that is_unsigned_int returns False for potentially negative values."""

    class Model(BaseModel):
        value: int = Field(ge=-10)

    field_info = Model.model_fields["value"]
    assert TypeInspector.is_unsigned_int(field_info) is False


def test_is_unsigned_int_false_no_constraints() -> None:
    """Test that is_unsigned_int returns False when no constraints present."""

    class Model(BaseModel):
        value: int

    field_info = Model.model_fields["value"]
    assert TypeInspector.is_unsigned_int(field_info) is False


def test_can_fit_in_32bit_uint_true() -> None:
    """Test can_fit_in_32bit_uint returns True for values within unsigned 32-bit."""

    class Model(BaseModel):
        value: int = Field(ge=0, le=1000)

    field_info = Model.model_fields["value"]
    assert TypeInspector.can_fit_in_32bit_uint(field_info) is True


def test_can_fit_in_32bit_uint_false_too_large() -> None:
    """Test that can_fit_in_32bit_uint returns False when max exceeds uint32."""

    class Model(BaseModel):
        value: int = Field(ge=0, le=5_000_000_000)

    field_info = Model.model_fields["value"]
    assert TypeInspector.can_fit_in_32bit_uint(field_info) is False


def test_can_fit_in_32bit_uint_false_negative() -> None:
    """Test that can_fit_in_32bit_uint returns False for signed integers."""

    class Model(BaseModel):
        value: int = Field(ge=-100, le=100)

    field_info = Model.model_fields["value"]
    assert TypeInspector.can_fit_in_32bit_uint(field_info) is False


def test_can_fit_in_32bit_uint_boundary() -> None:
    """Test that can_fit_in_32bit_uint handles exact uint32 boundary."""

    class Model(BaseModel):
        value: int = Field(ge=0, le=(2**32 - 1))

    field_info = Model.model_fields["value"]
    assert TypeInspector.can_fit_in_32bit_uint(field_info) is True


def test_collect_nested_models_with_optional() -> None:
    """Test that collect_nested_models finds models in Optional fields."""

    class Document(BaseModel):
        author: Person | None = None

    result = TypeInspector.collect_nested_models(Document)
    assert "Person" in result
    assert "Address" in result


def test_collect_nested_models_deeply_nested() -> None:
    """Test that collect_nested_models handles multiple levels of nesting."""

    class Company(BaseModel):
        ceo: Person
        employees: list[Person]

    result = TypeInspector.collect_nested_models(Company)
    assert "Person" in result
    assert "Address" in result


def test_get_numeric_constraints_with_gt_and_ge() -> None:
    """Test get_numeric_constraints handles both gt and ge (ge takes precedence)."""

    class Model(BaseModel):
        value: int = Field(ge=10, gt=5)

    field_info = Model.model_fields["value"]
    constraints = TypeInspector.get_numeric_constraints(field_info)

    assert constraints["ge"] == 10
    assert constraints["gt"] == 5


def test_is_variable_length_tuple_with_bare_ellipsis() -> None:
    """Test that is_variable_length_tuple only matches tuple[T, ...] pattern."""
    annotation = tuple[str, int]
    assert TypeInspector.is_variable_length_tuple(annotation) is False


def test_is_root_model() -> None:
    """Tests that is_root_model correctly detects."""

    class A(BaseModel):
        val: int

    class B(RootModel[list[int]]):
        root: list[int]

    class C:
        pass

    assert TypeInspector.is_root_model(A) is False
    assert TypeInspector.is_root_model(B) is True
    assert TypeInspector.is_root_model(C) is False  # type: ignore[arg-type]


def test_get_root_annotation_not_rootmodel() -> None:
    """Tests getting a RootModel annotation from a not-RootModel."""

    class A(BaseModel):
        val: int

    with pytest.raises(ValueError, match="not a RootModel"):
        TypeInspector.get_root_annotation(A)


def test_get_root_annotation_rootmodel() -> None:
    """Tests getting a RootModel annotation from a not-RootModel."""

    class B(RootModel[list[int]]):
        root: list[int]

    assert TypeInspector.get_root_annotation(B) == list[int]
