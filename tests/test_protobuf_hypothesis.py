"""Property-based fuzz tests for Protocol Buffer schema generation."""

import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pytest
from hypothesis import assume, given, settings, strategies as st
from pydantic import BaseModel, Field, create_model

from pyrmute.protobuf_schema import ProtoSchemaGenerator


@st.composite
def field_names(draw: Any) -> str:
    """Generate valid Pydantic field names."""
    first_char = draw(st.sampled_from("abcdefghijklmnopqrstuvwxyz"))
    rest = draw(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
            min_size=0,
            max_size=20,
        )
    )
    name = first_char + rest
    return str(name)


@st.composite
def integer_constraints(draw: Any) -> dict[str, int]:
    """Generate valid integer constraint combinations."""
    constraints: dict[str, int] = {}

    # Generate ge/gt
    if draw(st.booleans()):
        use_ge = draw(st.booleans())
        value = draw(st.integers(min_value=-1000, max_value=1000))
        constraints["ge" if use_ge else "gt"] = value

    # Generate le/lt
    if draw(st.booleans()):
        use_le = draw(st.booleans())
        value = draw(st.integers(min_value=-1000, max_value=10_000_000_000))
        # Ensure le/lt is greater than ge/gt if both exist
        if "ge" in constraints:
            assume(value > constraints["ge"])
        if "gt" in constraints:
            assume(value > constraints["gt"] + 1)
        constraints["le" if use_le else "lt"] = value

    return constraints


@st.composite
def simple_types(draw: Any) -> type:
    """Generate simple Python types."""
    return type(
        draw(
            st.sampled_from(
                [
                    str,
                    int,
                    float,
                    bool,
                    bytes,
                ]
            )
        )
    )


@st.composite
def complex_type(draw: Any, max_depth: int = 2, current_depth: int = 0) -> Any:
    """Recursively generate Python types usable in Pydantic models."""
    simple = simple_types()

    if current_depth >= max_depth:
        return draw(simple)

    container_strategy = st.one_of(
        simple,
        st.lists(st.deferred(lambda: complex_type(max_depth, current_depth + 1))),
        st.dictionaries(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5),
            st.deferred(lambda: complex_type(max_depth, current_depth + 1)),
        ),
        st.tuples(simple, simple),
    )
    return draw(container_strategy)


@st.composite
def complex_model(draw: Any, max_depth: int = 3, current_depth: int = 0) -> Any:
    """Generate a random Pydantic model with nested fields and complex types."""
    field_count = draw(st.integers(min_value=1, max_value=5))
    fields: dict[str, Any] = {}

    for _ in range(field_count):
        field_name = draw(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz",
                min_size=1,
                max_size=10,
            )
        )
        assume(field_name.isidentifier() and not field_name[0].isdigit())

        field_type = draw(
            complex_type(max_depth=max_depth, current_depth=current_depth)
        )

        if isinstance(field_type, tuple):
            field_type = tuple[*field_type]  # type: ignore[valid-type]

        if draw(st.booleans()):
            field_type = Optional[field_type]  # noqa: UP045
        if draw(st.booleans()):
            fields[field_name] = (field_type, Field(default=None))
        else:
            fields[field_name] = (field_type, ...)

    model_name = f"GeneratedModel_{current_depth}_{draw(st.integers(0, 9999))}"
    return create_model(
        model_name,
        **dict.fromkeys(fields, (str, ...)),
    )  # type: ignore[call-overload]


@given(
    field_count=st.integers(min_value=1, max_value=50),
    field_name_strategy=st.lists(field_names(), min_size=1, max_size=50, unique=True),
)
@settings(max_examples=50, deadline=None)
def test_field_numbering_uniqueness(
    field_count: int, field_name_strategy: list[str]
) -> None:
    """Property: All field numbers must be unique and sequential."""
    field_names_list = field_name_strategy[:field_count]

    Model = create_model(  # type: ignore[call-overload]
        "TestModel",
        **dict.fromkeys(field_names_list, (str, ...)),
    )

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "TestModel", "1.0.0")

    field_numbers = [f["number"] for f in message["fields"]]

    assert len(field_numbers) == len(set(field_numbers)), (
        f"Duplicate field numbers: {field_numbers}"
    )
    assert all(n > 0 for n in field_numbers), (
        f"Non-positive field numbers: {field_numbers}"
    )
    assert sorted(field_numbers) == list(range(1, len(field_numbers) + 1)), (
        f"Non-sequential field numbers: {sorted(field_numbers)}"
    )


@given(
    constraints=integer_constraints(),
)
@settings(max_examples=100, deadline=None)
def test_integer_optimization_never_crashes(constraints: dict[str, int]) -> None:
    """Property: Integer optimization should never crash regardless of constraints."""
    if constraints:
        Model = create_model(
            "TestModel",
            value=(int, Field(**constraints)),  # type: ignore[call-overload]
        )
    else:
        Model = create_model("TestModel", value=(int, ...))

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "TestModel", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "value"), None)
    assert field is not None
    assert field["type"] in ("int32", "uint32", "uint64", "int64")


@given(
    type_choice=simple_types(),
    field_name=field_names(),
)
@settings(max_examples=50, deadline=None)
def test_basic_type_mapping_always_works(type_choice: type, field_name: str) -> None:
    """Property: All basic types should map to valid proto types."""
    Model = create_model(
        "TestModel",
        **{field_name: (type_choice, ...)},
    )  # type: ignore[call-overload]

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "TestModel", "1.0.0")

    assert len(message["fields"]) == 1

    field = message["fields"][0]
    assert field["name"] == field_name

    assert isinstance(field["type"], str)
    assert len(field["type"]) > 0


@given(
    list_depth=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=20, deadline=None)
def test_nested_list_handling(list_depth: int) -> None:
    """Property: Nested lists should be handled gracefully."""
    current_type: Any = str
    for _ in range(list_depth):
        current_type = list[current_type]

    Model = create_model("TestModel", nested=(current_type, ...))

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "TestModel", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "nested"), None)
    assert field is not None


@given(
    key_type=st.sampled_from([str, int]),
    value_type=simple_types(),
    field_name=field_names(),
)
@settings(max_examples=50, deadline=None)
def test_dict_type_mapping(key_type: type, value_type: type, field_name: str) -> None:
    """Property: Dict types should always produce valid map syntax."""
    Model = create_model(  # type: ignore[call-overload]
        "TestModel",
        **{field_name: (dict[key_type, value_type], ...)},  # type: ignore[valid-type]
    )

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "TestModel", "1.0.0")
    field = next((f for f in message["fields"] if f["name"] == field_name), None)
    assert field is not None

    assert field["type"].startswith("map<")
    assert field["type"].endswith(">")
    assert "," in field["type"]


@given(
    union_size=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=30, deadline=None)
def test_union_creates_oneof(union_size: int) -> None:
    """Property: Unions with 2+ non-None types should create oneofs."""
    types_pool = [str, int, float, bool, bytes][:union_size]

    union_type = Union[tuple(types_pool)]  # type: ignore # noqa: UP007
    Model = create_model("TestModel", value=(union_type, ...))

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "TestModel", "1.0.0")

    assert "oneofs" in message
    assert len(message["oneofs"]) >= 1

    oneof_fields = [f for f in message["fields"] if f.get("oneof_group")]
    assert len(oneof_fields) == union_size


@given(
    model_name=st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        min_size=1,
        max_size=30,
    ),
)
@settings(max_examples=50, deadline=None)
def test_model_name_preserved(model_name: str) -> None:
    """Property: Generated message name should match input name."""
    Model = create_model(model_name, field=(str, ...))

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, model_name, "1.0.0")

    assert message["name"] == model_name


@given(
    description_length=st.integers(min_value=0, max_value=500),
)
@settings(max_examples=30, deadline=None)
def test_long_descriptions_handled(description_length: int) -> None:
    """Property: Long field descriptions should not cause errors."""
    description = "x" * description_length

    class Model(BaseModel):
        field: str = Field(description=description)

    generator = ProtoSchemaGenerator(include_docs=True)
    proto_file = generator.generate_proto_file(Model, "Model", "1.0.0")
    proto_string = generator.proto_file_to_string(proto_file)

    assert isinstance(proto_string, str)
    assert len(proto_string) > 0


@given(field_count=st.integers(min_value=1, max_value=20))
@settings(max_examples=50, deadline=None)
def test_invariant_no_field_number_zero(field_count: int) -> None:
    """Invariant: Proto field numbers must never be 0."""
    Model = create_model(  # type: ignore[call-overload]
        "TestModel", **{f"field{i}": (str, ...) for i in range(field_count)}
    )

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "TestModel", "1.0.0")

    field_numbers = [f["number"] for f in message["fields"]]
    assert 0 not in field_numbers, "Field number 0 is invalid in protobuf"


@given(
    ge_value=st.integers(min_value=0, max_value=1000),
    le_value=st.integers(min_value=0, max_value=2**40),
)
@settings(max_examples=100, deadline=None)
def test_invariant_uint_types_have_ge_zero(ge_value: int, le_value: int) -> None:
    """Invariant: uint32/uint64 types should only be used when ge >= 0."""
    assume(le_value > ge_value)

    class Model(BaseModel):
        value: int = Field(ge=ge_value, le=le_value)

    generator = ProtoSchemaGenerator()
    message = generator._generate_proto_schema(Model, "Model", "1.0.0")

    field = next((f for f in message["fields"] if f["name"] == "value"), None)
    assert field is not None

    if field["type"] in ("uint32", "uint64"):
        assert ge_value >= 0, f"uint type used but ge={ge_value} < 0"


@given(model_count=st.integers(min_value=1, max_value=10))
@settings(max_examples=20, deadline=None)
def test_invariant_field_counter_resets(model_count: int) -> None:
    """Invariant: Field counter should reset for each new model."""
    generator = ProtoSchemaGenerator()
    first_fields = []

    for i in range(model_count):
        Model = create_model(f"Model{i}", field1=(str, ...), field2=(int, ...))
        message = generator._generate_proto_schema(Model, f"Model{i}", "1.0.0")
        first_field_num = message["fields"][0]["number"]
        first_fields.append(first_field_num)

    assert all(num == 1 for num in first_fields), (
        f"Field counter not resetting: {first_fields}"
    )


@pytest.mark.integration
def test_proto_compiler_validation_sample() -> None:
    """Validate generated proto compiles with protoc.

    This test requires protoc (Protocol Buffer compiler) to be installed.
    Skip if not available.

    Install protoc:
      - macOS: brew install protobuf
      - Ubuntu: apt-get install protobuf-compiler
      - Or download from: https://github.com/protocolbuffers/protobuf/releases
    """
    if not shutil.which("protoc"):
        pytest.skip("protoc not installed")

    class TestModel(BaseModel):
        id: int = Field(ge=0)
        name: str
        tags: list[str]
        metadata: dict[str, str]
        created_at: datetime

    generator = ProtoSchemaGenerator(package="test")
    proto_file = generator.generate_proto_file(TestModel, "TestModel", "1.0.0")
    proto_content = generator.proto_file_to_string(proto_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        proto_path = Path(tmpdir) / "test.proto"
        proto_path.write_text(proto_content)

        result = subprocess.run(
            [
                "protoc",
                f"--proto_path={tmpdir}",
                f"--python_out={tmpdir}",
                "test.proto",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"protoc compilation failed:\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}\n"
            f"Proto content:\n{proto_content}"
        )


@pytest.mark.integration
@given(
    field_count=st.integers(min_value=1, max_value=5),
    field_name_strategy=st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz",
            min_size=1,
            max_size=10,
        ),
        min_size=1,
        max_size=5,
        unique=True,
    ),
)
@settings(max_examples=5, deadline=None)
def test_generated_proto_compiles_with_protoc(
    field_count: int, field_name_strategy: str
) -> None:
    """Integration test: Generated protobuf schema compiles with protoc."""
    if not shutil.which("protoc"):
        pytest.skip("protoc not installed")

    fields = dict.fromkeys(field_name_strategy[:field_count], (str, ...))
    Model = create_model("GeneratedModel", **fields)  # type: ignore[call-overload]

    generator = ProtoSchemaGenerator(package="integration_test")
    proto_file = generator.generate_proto_file(Model, "GeneratedModel", "1.0.0")
    proto_content = generator.proto_file_to_string(proto_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        proto_path = Path(tmpdir) / "generated.proto"
        proto_path.write_text(proto_content)

        result = subprocess.run(
            [
                "protoc",
                f"--proto_path={tmpdir}",
                f"--python_out={tmpdir}",
                "generated.proto",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"protoc failed to compile generated proto.\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}\n"
            f"Proto content:\n{proto_content}"
        )


@pytest.mark.integration
@given(model=complex_model(max_depth=2))
@settings(max_examples=5, deadline=None)
def test_complex_generated_proto_compiles_with_protoc(model: Any) -> None:
    """Integration: even complex nested models produce valid .proto files."""
    if not shutil.which("protoc"):
        pytest.skip("protoc not installed")

    generator = ProtoSchemaGenerator(package="integration_test")
    proto_file = generator.generate_proto_file(model, model.__name__, "1.0.0")
    proto_content = generator.proto_file_to_string(proto_file)

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
        )

        assert result.returncode == 0, (
            f"protoc failed to compile generated proto.\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}\n"
            f"Proto:\n{proto_content}"
        )
