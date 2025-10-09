"""Hypothesis-based property tests for ModelManager using pytest functions."""

import string
from collections.abc import Callable
from typing import Any, Self

from hypothesis import HealthCheck, assume, given, settings, strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule
from pydantic import BaseModel, Field

from pyrmute import MigrationError, ModelManager, ModelVersion


# Strategies
@st.composite
def semantic_version(draw: st.DrawFn) -> str:
    """Generate valid semantic versions."""
    major = draw(st.integers(min_value=0, max_value=5))
    minor = draw(st.integers(min_value=0, max_value=10))
    patch = draw(st.integers(min_value=0, max_value=20))
    return f"{major}.{minor}.{patch}"


@st.composite
def model_name(draw: st.DrawFn) -> str:
    """Generate valid model names."""
    return draw(st.sampled_from(["User", "Product", "Order", "Address", "Item"]))


@st.composite
def field_name(draw: st.DrawFn) -> str:
    """Generate valid Python field names."""
    return draw(
        st.sampled_from(
            [
                "name",
                "email",
                "age",
                "value",
                "status",
                "count",
                "data",
                "metadata",
                "field1",
                "field2",
                "field3",
            ]
        )
    )


@st.composite
def simple_field_value(draw: st.DrawFn) -> Any:
    """Generate simple field values (faster generation)."""
    return draw(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(min_value=-100, max_value=100),
            st.text(alphabet=string.ascii_letters, max_size=20),
        )
    )


@st.composite
def field_value(draw: st.DrawFn) -> Any:
    """Generate valid field values."""
    return draw(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(min_value=-100, max_value=100),
            st.floats(
                allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000
            ),
            st.text(alphabet=string.ascii_letters, max_size=30),
            st.lists(st.integers(min_value=0, max_value=100), max_size=5),
            st.dictionaries(
                st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=5),
                st.integers(min_value=0, max_value=100),
                max_size=3,
            ),
        )
    )


# Simple property tests
@given(
    name=model_name(),
    version=semantic_version(),
    data=st.dictionaries(field_name(), simple_field_value(), min_size=1, max_size=5),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_migrate_to_same_version_is_identity(
    name: str, version: str, data: dict[str, Any]
) -> None:
    """Migrating to the same version should return identical data."""
    manager = ModelManager()

    class_name = f"{name}V1"
    annotations: dict[str, Any] = {}
    defaults: dict[str, Any] = {}

    for k, v in data.items():
        if v is None:
            annotations[k] = str | None
            defaults[k] = None
        else:
            annotations[k] = type(v)
            defaults[k] = v

    DynamicModel = type(
        class_name, (BaseModel,), {"__annotations__": annotations, **defaults}
    )

    manager.model(name, version, backward_compatible=True)(DynamicModel)
    result = manager._migration_manager.migrate(data, name, version, version)

    assert result == data


@given(
    name=model_name(),
    v1=semantic_version(),
    v2=semantic_version(),
    v3=semantic_version(),
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_migration_transitivity(name: str, v1: str, v2: str, v3: str) -> None:
    """Migration should be transitive: v1→v2→v3 == v1→v3."""
    versions = sorted([ModelVersion.parse(v) for v in [v1, v2, v3]])
    assume(len(set(versions)) == 3)  # noqa: PLR2004

    v1, v2, v3 = [str(v) for v in versions]

    manager = ModelManager()

    @manager.model(name, v1, backward_compatible=True)
    class ModelV1(BaseModel):
        field1: str = "default1"

    @manager.model(name, v2, backward_compatible=True)
    class ModelV2(BaseModel):
        field1: str = "default1"
        field2: str = "default2"

    @manager.model(name, v3, backward_compatible=True)
    class ModelV3(BaseModel):
        field1: str = "default1"
        field2: str = "default2"
        field3: str = "default3"

    data = {"field1": "test"}

    # Direct migration
    direct = manager._migration_manager.migrate(data, name, v1, v3)

    # Transitive migration
    intermediate = manager._migration_manager.migrate(data, name, v1, v2)
    transitive = manager._migration_manager.migrate(intermediate, name, v2, v3)

    assert direct == transitive


@given(
    name=model_name(),
    version=semantic_version(),
    data=st.dictionaries(field_name(), simple_field_value(), min_size=1, max_size=3),
    extra_fields=st.dictionaries(
        st.sampled_from(["extra1", "extra2", "extra3", "metadata", "custom"]),
        simple_field_value(),
        min_size=1,
        max_size=3,
    ),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_extra_fields_preserved(
    name: str, version: str, data: dict[str, Any], extra_fields: dict[str, Any]
) -> None:
    """Extra fields not in schema should be preserved during migration."""
    # Ensure no overlap between data and extra_fields
    assume(not set(data.keys()).intersection(set(extra_fields.keys())))

    manager = ModelManager()
    annotations: dict[str, Any] = {}
    defaults: dict[str, Any] = {}

    for k, v in data.items():
        if v is None:
            annotations[k] = str | None
            defaults[k] = None
        else:
            annotations[k] = type(v)
            defaults[k] = v

    DynamicModel = type(
        f"{name}V1", (BaseModel,), {"__annotations__": annotations, **defaults}
    )

    manager.model(name, version, backward_compatible=True)(DynamicModel)

    full_data = {**data, **extra_fields}
    result = manager._migration_manager.migrate(full_data, name, version, version)

    for key, value in extra_fields.items():
        assert key in result
        assert result[key] == value


@given(name=model_name(), v1=semantic_version(), v2=semantic_version())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_version_ordering_independence(name: str, v1: str, v2: str) -> None:
    """Migration should work in both directions for auto-migrate."""
    versions = sorted([ModelVersion.parse(v) for v in [v1, v2]])
    assume(versions[0] != versions[1])

    v1, v2 = str(versions[0]), str(versions[1])

    manager = ModelManager()

    @manager.model(name, v1, backward_compatible=True)
    class ModelV1(BaseModel):
        field1: str = "default1"

    @manager.model(name, v2, backward_compatible=True)
    class ModelV2(BaseModel):
        field1: str = "default1"
        field2: str = "default2"

    data = {"field1": "test"}
    forward = manager._migration_manager.migrate(data, name, v1, v2)
    assert "field1" in forward
    assert "field2" in forward

    backward = manager._migration_manager.migrate(forward, name, v2, v1)
    assert "field1" in backward


@given(
    name=model_name(),
    versions=st.lists(semantic_version(), min_size=2, max_size=4, unique=True),
)
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_migration_chain_consistency(name: str, versions: list[str]) -> None:
    """Migrating through a chain should be consistent."""
    sorted_versions = sorted([ModelVersion.parse(v) for v in versions])
    version_strs = [str(v) for v in sorted_versions]

    manager = ModelManager()

    for i, version in enumerate(version_strs):
        fields_dict: dict[str, Any] = {
            "__annotations__": {"field1": str},
            "field1": "default1",
        }

        for j in range(i):
            field_name = f"field{j + 2}"
            fields_dict["__annotations__"][field_name] = str
            fields_dict[field_name] = f"default{j + 2}"

        ModelClass = type(f"{name}V{i}", (BaseModel,), fields_dict)
        manager.model(name, version, backward_compatible=True)(ModelClass)

    data = {"field1": "test"}
    result = manager._migration_manager.migrate(
        data, name, version_strs[0], version_strs[-1]
    )

    assert "field1" in result
    assert result["field1"] == "test"


@given(
    data=st.dictionaries(
        field_name(),
        st.one_of(
            st.text(alphabet=string.ascii_letters, max_size=10),
            st.integers(min_value=0, max_value=100),
        ),
        min_size=1,
        max_size=5,
    )
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_data_types_preserved(data: dict[str, Any]) -> None:
    """Data types should be preserved through migration."""
    manager = ModelManager()
    name = "TypeTest"
    version = "1.0.0"

    annotations = {}
    defaults = {}

    for k, v in data.items():
        annotations[k] = type(v)
        defaults[k] = v

    DynamicModel = type(
        "TypeTestV1", (BaseModel,), {"__annotations__": annotations, **defaults}
    )

    manager.model(name, version, backward_compatible=True)(DynamicModel)

    result = manager._migration_manager.migrate(data, name, version, version)

    for key, value in data.items():
        assert key in result
        assert type(result[key]) is type(value)
        assert result[key] == value


@given(
    num_fields=st.integers(min_value=1, max_value=10),
    version=semantic_version(),
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_many_fields_handled_correctly(num_fields: int, version: str) -> None:
    """Models with many fields should be handled correctly."""
    manager = ModelManager()
    name = "ManyFields"

    annotations = {f"field_{i}": str for i in range(num_fields)}
    defaults = {f"field_{i}": f"default_{i}" for i in range(num_fields)}

    ManyFieldsModel = type(
        "ManyFieldsV1", (BaseModel,), {"__annotations__": annotations, **defaults}
    )

    manager.model(name, version, backward_compatible=True)(ManyFieldsModel)

    data = {f"field_{i}": f"value_{i}" for i in range(num_fields)}

    result = manager._migration_manager.migrate(data, name, version, version)

    # All fields should be present
    assert len(result) >= num_fields
    for i in range(num_fields):
        assert f"field_{i}" in result
        assert result[f"field_{i}"] == f"value_{i}"


@given(
    name=model_name(),
    v1=semantic_version(),
    v2=semantic_version(),
    field_names=st.lists(
        st.sampled_from(["new_field1", "new_field2", "new_field3"]),
        min_size=1,
        max_size=3,
        unique=True,
    ),
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_field_addition_with_defaults(
    name: str, v1: str, v2: str, field_names: list[str]
) -> None:
    """Adding fields with defaults should not break existing data."""
    versions = sorted([ModelVersion.parse(v) for v in [v1, v2]])
    assume(versions[0] != versions[1])

    v1, v2 = str(versions[0]), str(versions[1])

    manager = ModelManager()

    @manager.model(name, v1, backward_compatible=True)
    class ModelV1(BaseModel):
        base_field: str = "base"

    v2_annotations = {"base_field": str}
    v2_defaults = {"base_field": "base"}
    for fname in field_names:
        v2_annotations[fname] = str
        v2_defaults[fname] = f"default_{fname}"

    ModelV2 = type(
        f"{name}V2", (BaseModel,), {"__annotations__": v2_annotations, **v2_defaults}
    )
    manager.model(name, v2, backward_compatible=True)(ModelV2)

    data = {"base_field": "test"}
    result = manager._migration_manager.migrate(data, name, v1, v2)

    assert result["base_field"] == "test"
    for fname in field_names:
        assert fname in result


@given(
    name=model_name(),
    version=semantic_version(),
    null_fields=st.lists(
        st.sampled_from(["field1", "field2", "field3"]),
        min_size=1,
        max_size=3,
        unique=True,
    ),
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_null_values_handled(name: str, version: str, null_fields: list[str]) -> None:
    """Null values should be handled correctly."""
    manager = ModelManager()

    annotations = dict.fromkeys(null_fields, str | None)
    defaults = dict.fromkeys(null_fields)

    NullableModel = type(
        f"{name}V1", (BaseModel,), {"__annotations__": annotations, **defaults}
    )

    manager.model(name, version, backward_compatible=True)(NullableModel)

    data = dict.fromkeys(null_fields)

    result = manager._migration_manager.migrate(data, name, version, version)

    for fname in null_fields:
        assert fname in result
        assert result[fname] is None


# Edge cases
@given(depth=st.integers(min_value=2, max_value=4))
@settings(
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_deeply_nested_models(depth: int) -> None:
    """Test migration with deeply nested model structures."""
    manager = ModelManager()

    models = []
    for i in range(depth):
        if i == 0:

            @manager.model(f"Level{i}", "1.0.0", backward_compatible=True)
            class BaseLevel(BaseModel):
                value: str = "base"

            models.append(BaseLevel)
        else:
            prev_model_class = models[-1]

            def make_next_level(prev_cls: type) -> Any:
                @manager.model(f"Level{i}", "1.0.0", backward_compatible=True)  # noqa: B023
                class NextLevel(BaseModel):
                    value: str = "nested"
                    nested: prev_cls = Field(  # type: ignore
                        default_factory=lambda: prev_cls(value="default")
                    )

                return NextLevel

            models.append(make_next_level(prev_model_class))

    nested_data: dict[str, Any] = {"value": "test"}
    for i in range(1, depth):
        nested_data = {"value": f"level{i}", "nested": nested_data}

    result = manager._migration_manager.migrate(
        nested_data, f"Level{depth - 1}", "1.0.0", "1.0.0"
    )
    assert isinstance(result, dict)


@given(
    num_versions=st.integers(min_value=2, max_value=6),
)
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_long_migration_chains(num_versions: int) -> None:
    """Test migration through long version chains."""
    manager = ModelManager()
    name = "ChainModel"

    for i in range(num_versions):
        version = f"1.0.{i}"

        @manager.model(name, version, backward_compatible=True)
        class ChainModel(BaseModel):
            field1: str = "default"
            version_num: int = i

        if i > 0:
            prev_version = f"1.0.{i - 1}"

            def make_migration(
                version_number: int,
            ) -> Callable[[dict[str, Any]], dict[str, Any]]:
                def migrate(data: dict[str, Any]) -> dict[str, Any]:
                    return {**data, "version_num": version_number}

                return migrate

            manager.migration(name, prev_version, version)(make_migration(i))

    data = {"field1": "value"}
    result = manager._migration_manager.migrate(
        data, name, "1.0.0", f"1.0.{num_versions - 1}"
    )

    assert result["version_num"] == num_versions - 1


@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(data=st.dictionaries(field_name(), simple_field_value(), min_size=0, max_size=3))
def test_migration_with_all_optional_fields(data: dict[str, Any]) -> None:
    """Test migration when all fields have defaults."""
    assume("field3" not in data)

    manager = ModelManager()

    @manager.model("Optional", "1.0.0", backward_compatible=True)
    class OptionalV1(BaseModel):
        field1: str = "default1"
        field2: int = 0

    @manager.model("Optional", "2.0.0", backward_compatible=True)
    class OptionalV2(BaseModel):
        field1: str = "default1"
        field2: int = 0
        field3: list[str] = Field(default_factory=list)

    result = manager._migration_manager.migrate(data, "Optional", "1.0.0", "2.0.0")
    assert "field3" in result
    assert result["field3"] == []


@given(num_extra_fields=st.integers(min_value=1, max_value=10))
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_many_extra_fields_preserved(num_extra_fields: int) -> None:
    """Test that large numbers of extra fields are preserved."""
    manager = ModelManager()

    @manager.model("ManyFields", "1.0.0", backward_compatible=True)
    class ManyFieldsModel(BaseModel):
        required_field: str = "required"

    data = {"required_field": "test"}
    for i in range(num_extra_fields):
        data[f"extra_{i}"] = f"value_{i}"

    result = manager._migration_manager.migrate(data, "ManyFields", "1.0.0", "1.0.0")

    for i in range(num_extra_fields):
        assert f"extra_{i}" in result
        assert result[f"extra_{i}"] == f"value_{i}"


@given(name=model_name(), version=semantic_version())
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_empty_data_migration(name: str, version: str) -> None:
    """Empty data should migrate successfully with defaults."""
    manager = ModelManager()

    @manager.model(name, version, backward_compatible=True)
    class EmptyModel(BaseModel):
        field1: str = "default1"
        field2: int = 0
        field3: list[str] = Field(default_factory=list)

    result = manager._migration_manager.migrate({}, name, version, version)

    assert isinstance(result, dict)


@given(name=model_name(), v1=semantic_version(), v2=semantic_version())
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_migration_with_no_changes(name: str, v1: str, v2: str) -> None:
    """Migration between versions with identical schemas should work."""
    versions = sorted([ModelVersion.parse(v) for v in [v1, v2]])
    assume(versions[0] != versions[1])

    v1, v2 = str(versions[0]), str(versions[1])

    manager = ModelManager()

    @manager.model(name, v1, backward_compatible=True)
    class ModelV1(BaseModel):
        field1: str = "default"
        field2: int = 0

    @manager.model(name, v2, backward_compatible=True)
    class ModelV2(BaseModel):
        field1: str = "default"
        field2: int = 0

    data = {"field1": "test", "field2": 42}
    result = manager._migration_manager.migrate(data, name, v1, v2)

    assert result == data


@given(
    name=model_name(),
    version=semantic_version(),
    list_size=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_list_fields_preserved(name: str, version: str, list_size: int) -> None:
    """List fields should be preserved correctly."""
    manager = ModelManager()

    @manager.model(name, version, backward_compatible=True)
    class ListModel(BaseModel):
        items: list[int] = Field(default_factory=list)

    data = {"items": list(range(list_size))}
    result = manager._migration_manager.migrate(data, name, version, version)

    assert "items" in result
    assert result["items"] == list(range(list_size))
    assert len(result["items"]) == list_size


@given(
    name=model_name(),
    version=semantic_version(),
    dict_size=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_dict_fields_preserved(name: str, version: str, dict_size: int) -> None:
    """Dictionary fields should be preserved correctly."""
    manager = ModelManager()

    @manager.model(name, version, backward_compatible=True)
    class DictModel(BaseModel):
        metadata: dict[str, int] = Field(default_factory=dict)

    data = {"metadata": {f"key_{i}": i for i in range(dict_size)}}
    result = manager._migration_manager.migrate(data, name, version, version)

    assert "metadata" in result
    assert result["metadata"] == {f"key_{i}": i for i in range(dict_size)}
    assert len(result["metadata"]) == dict_size


# Stateful
class ModelManagerStateMachine(RuleBasedStateMachine):
    """Stateful testing for ModelManager to find complex interaction bugs."""

    def __init__(self: Self) -> None:
        """Init."""
        super().__init__()
        self.manager = ModelManager()
        self.registered_models: dict[str, list[str]] = {}  # name -> [versions]
        self.model_classes: dict[tuple[str, str], type[BaseModel]] = {}
        self.migrations: set[tuple[str, str, str]] = set()  # (name, from_v, to_v)

    models = Bundle("models")

    @rule(
        target=models,
        name=model_name(),
        version=semantic_version(),
        backward_compatible=st.booleans(),
    )
    def register_model(
        self: Self, name: str, version: str, backward_compatible: bool
    ) -> tuple[str, str]:
        """Register a new model version."""
        assume((name, version) not in self.model_classes)

        @self.manager.model(name, version, backward_compatible=backward_compatible)
        class DynamicModel(BaseModel):
            name: str = "default"
            value: int = 0

        if name not in self.registered_models:
            self.registered_models[name] = []
        self.registered_models[name].append(version)
        self.model_classes[(name, version)] = DynamicModel

        return (name, version)

    @rule(model1=models, model2=models)
    def register_migration_between_models(
        self: Self, model1: tuple[str, str], model2: tuple[str, str]
    ) -> None:
        """Register a migration between two model versions."""
        name1, v1 = model1
        name2, v2 = model2

        # Only migrate within the same model
        assume(name1 == name2)
        # Don't migrate to same version
        assume(v1 != v2)
        # Don't duplicate migrations
        assume((name1, v1, v2) not in self.migrations)

        @self.manager.migration(name1, v1, v2)
        def migrate(data: dict[str, Any]) -> dict[str, Any]:
            return {**data, "migrated": True}

        self.migrations.add((name1, v1, v2))

    @rule(
        model=models,
        data=st.dictionaries(field_name(), simple_field_value(), max_size=3),
    )
    def migrate_to_same_version(
        self: Self, model: tuple[str, str], data: dict[str, Any]
    ) -> None:
        """Migrating to same version should preserve data."""
        name, version = model

        result = self.manager._migration_manager.migrate(data, name, version, version)

        for key, value in data.items():
            if key in result:
                assert result[key] == value

    @rule(model1=models, model2=models)
    def test_migration_path_exists_or_fails_gracefully(
        self: Self, model1: tuple[str, str], model2: tuple[str, str]
    ) -> None:
        """Migration should either succeed or raise appropriate error."""
        name1, v1 = model1
        name2, v2 = model2

        assume(name1 == name2)
        assume(v1 != v2)

        data = {"name": "test", "value": 42}

        try:
            result = self.manager._migration_manager.migrate(data, name1, v1, v2)
            assert isinstance(result, dict)
        except MigrationError as e:
            assert "migration path" in str(e).lower() or "not found" in str(e).lower()

    @invariant()
    def all_registered_models_retrievable(self: Self) -> None:
        """All registered models should be retrievable."""
        for name, versions in self.registered_models.items():
            for version in versions:
                model = self.manager.get(name, version)
                assert model is not None
                assert issubclass(model, BaseModel)

    @invariant()
    def latest_version_is_max(self: Self) -> None:
        """Latest version should be the maximum version."""
        for name, versions in self.registered_models.items():
            if versions:
                latest = self.manager.get(name, None)
                version_objects = sorted([ModelVersion.parse(v) for v in versions])
                expected_latest = self.manager.get(name, str(version_objects[-1]))
                assert latest == expected_latest


TestModelManagerStateful = ModelManagerStateMachine.TestCase
