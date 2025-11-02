"""Tests for model builder utilities."""

from typing import Literal, Self

import pytest
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pyrmute import ModelManager
from pyrmute._model_builder import (
    auto_detect_nested_pins,
    merge_nested_pins,
    normalize_pin_specs,
)
from pyrmute.model_version import ModelVersion

# ruff: noqa: PLR2004


def test_auto_detect_simple_nested_model() -> None:
    """Test auto-detection of single nested model."""

    class Address(BaseModel):
        street: str

    class User(BaseModel):
        name: str
        address: Address

    def get_model_info(cls: type) -> tuple[str, ModelVersion] | None:
        if cls is Address:
            return ("Address", ModelVersion(1, 0, 0))
        return None

    pins = auto_detect_nested_pins(User, get_model_info)
    assert pins == {"address": "Address:1.0.0"}


def test_auto_detect_multiple_nested_models() -> None:
    """Test auto-detection with multiple nested models."""

    class Address(BaseModel):
        street: str

    class Profile(BaseModel):
        bio: str

    class User(BaseModel):
        name: str
        address: Address
        profile: Profile

    def get_model_info(cls: type) -> tuple[str, ModelVersion] | None:
        if cls is Address:
            return ("Address", ModelVersion(1, 0, 0))
        if cls is Profile:
            return ("Profile", ModelVersion(2, 0, 0))
        return None

    pins = auto_detect_nested_pins(User, get_model_info)
    assert pins == {
        "address": "Address:1.0.0",
        "profile": "Profile:2.0.0",
    }


def test_auto_detect_deep_nesting() -> None:
    """Test auto-detection with deep nesting (3+ levels)."""

    class City(BaseModel):
        name: str

    class Address(BaseModel):
        street: str
        city: City

    class User(BaseModel):
        name: str
        address: Address

    def get_model_info(cls: type) -> tuple[str, ModelVersion] | None:
        if cls is City:
            return ("City", ModelVersion(1, 0, 0))
        if cls is Address:
            return ("Address", ModelVersion(1, 0, 0))
        return None

    pins = auto_detect_nested_pins(User, get_model_info)
    assert "address" in pins
    assert "address.city" in pins
    assert pins["address"] == "Address:1.0.0"
    assert pins["address.city"] == "City:1.0.0"


def test_auto_detect_optional_nested_model() -> None:
    """Test auto-detection with Optional[Model]."""

    class Address(BaseModel):
        street: str

    class User(BaseModel):
        name: str
        address: Address | None = None

    def get_model_info(cls: type) -> tuple[str, ModelVersion] | None:
        if cls is Address:
            return ("Address", ModelVersion(1, 0, 0))
        return None

    pins = auto_detect_nested_pins(User, get_model_info)
    assert pins == {"address": "Address:1.0.0"}


def test_auto_detect_list_of_nested_models() -> None:
    """Test auto-detection with list[Model]."""

    class Tag(BaseModel):
        name: str

    class Post(BaseModel):
        title: str
        tags: list[Tag]

    def get_model_info(cls: type) -> tuple[str, ModelVersion] | None:
        if cls is Tag:
            return ("Tag", ModelVersion(1, 0, 0))
        return None

    pins = auto_detect_nested_pins(Post, get_model_info)
    assert pins == {"tags": "Tag:1.0.0"}


def test_auto_detect_dict_of_nested_models() -> None:
    """Test auto-detection with dict[str, Model]."""

    class Permission(BaseModel):
        can_read: bool

    class User(BaseModel):
        name: str
        permissions: dict[str, Permission]

    def get_model_info(cls: type) -> tuple[str, ModelVersion] | None:
        if cls is Permission:
            return ("Permission", ModelVersion(1, 0, 0))
        return None

    pins = auto_detect_nested_pins(User, get_model_info)
    assert pins == {"permissions": "Permission:1.0.0"}


# ============================================================================
# Tests for normalize_pin_specs
# ============================================================================


def test_normalize_shorthand_version() -> None:
    """Test normalization of shorthand version."""

    class Address(BaseModel):
        street: str

    class User(BaseModel):
        name: str
        address: Address

    def get_model_info(cls: type) -> tuple[str, ModelVersion] | None:
        if cls is Address:
            return ("Address", ModelVersion(1, 0, 0))
        return None

    pins = {"address": "2.0.0"}
    normalized = normalize_pin_specs(User, pins, get_model_info)
    assert normalized == {"address": "Address:2.0.0"}


def test_normalize_already_normalized() -> None:
    """Test that already normalized pins are unchanged."""

    class Address(BaseModel):
        street: str

    class User(BaseModel):
        name: str
        address: Address

    def get_model_info(cls: type) -> tuple[str, ModelVersion] | None:
        if cls is Address:
            return ("Address", ModelVersion(1, 0, 0))
        return None

    pins = {"address": "Address:3.0.0"}
    normalized = normalize_pin_specs(User, pins, get_model_info)
    assert normalized == {"address": "Address:3.0.0"}


def test_normalize_nested_path_shorthand() -> None:
    """Test shorthand for nested paths."""

    class City(BaseModel):
        name: str

    class Address(BaseModel):
        street: str
        city: City

    class User(BaseModel):
        name: str
        address: Address

    def get_model_info(cls: type) -> tuple[str, ModelVersion] | None:
        if cls is City:
            return ("City", ModelVersion(1, 0, 0))
        if cls is Address:
            return ("Address", ModelVersion(1, 0, 0))
        return None

    pins = {"address.city": "2.0.0"}
    normalized = normalize_pin_specs(User, pins, get_model_info)
    assert normalized == {"address.city": "City:2.0.0"}


# ============================================================================
# Tests for merge_nested_pins
# ============================================================================


def test_merge_explicit_overrides_detected() -> None:
    """Test that explicit pins override detected pins."""
    inherited: dict[str, tuple[str, ModelVersion]] = {}
    detected = {"address": "Address:1.0.0"}
    explicit = {"address": "Address:2.0.0"}

    merged = merge_nested_pins(inherited, detected, explicit)
    assert merged["address"] == ("Address", ModelVersion(2, 0, 0))


def test_merge_explicit_overrides_inherited() -> None:
    """Test that explicit pins override inherited pins."""
    inherited = {"address": ("Address", ModelVersion(1, 0, 0))}
    detected: dict[str, str] = {}
    explicit = {"address": "Address:3.0.0"}

    merged = merge_nested_pins(inherited, detected, explicit)
    assert merged["address"] == ("Address", ModelVersion(3, 0, 0))


def test_merge_detected_overrides_inherited() -> None:
    """Test that detected pins override inherited pins."""
    inherited = {"address": ("Address", ModelVersion(1, 0, 0))}
    detected = {"address": "Address:2.0.0"}
    explicit: dict[str, str] = {}

    merged = merge_nested_pins(inherited, detected, explicit)
    assert merged["address"] == ("Address", ModelVersion(2, 0, 0))


def test_merge_all_sources() -> None:
    """Test merging pins from all three sources."""
    inherited = {
        "address": ("Address", ModelVersion(1, 0, 0)),
        "profile": ("Profile", ModelVersion(1, 0, 0)),
    }
    detected = {
        "address": "Address:2.0.0",
        "avatar": "Avatar:1.0.0",
    }
    explicit = {"address": "Address:3.0.0"}

    merged = merge_nested_pins(inherited, detected, explicit)

    assert merged["address"] == ("Address", ModelVersion(3, 0, 0))
    assert merged["profile"] == ("Profile", ModelVersion(1, 0, 0))
    assert merged["avatar"] == ("Avatar", ModelVersion(1, 0, 0))


# ============================================================================
# Tests for field preservation
# ============================================================================


def test_preserves_field_constraints(manager: ModelManager) -> None:
    """Test that field constraints are preserved."""

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str = Field(min_length=1, max_length=100)
        age: int = Field(ge=0, le=150)
        email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

    user_model = manager.get("User", "1.0.0")

    # Test constraints work
    with pytest.raises(ValueError):
        user_model(name="", age=25, email="test@example.com")

    with pytest.raises(ValueError):
        user_model(name="John", age=-1, email="test@example.com")

    with pytest.raises(ValueError):
        user_model(name="John", age=25, email="invalid-email")

    # Valid data should work
    user = user_model(name="John", age=25, email="test@example.com")
    assert user.name == "John"  # type: ignore[attr-defined]


def test_preserves_field_metadata(manager: ModelManager) -> None:
    """Test that field metadata (title, description, etc.) is preserved."""

    @manager.composite("Product", "1.0.0")
    class ProductV1(BaseModel):
        name: str = Field(
            title="Product Name",
            description="The name of the product",
            examples=["Widget", "Gadget"],
        )
        price: float = Field(
            title="Price",
            description="Price in USD",
            gt=0,
        )

    product_model = manager.get("Product", "1.0.0")

    name_field = product_model.model_fields["name"]
    assert name_field.title == "Product Name"
    assert name_field.description == "The name of the product"
    assert name_field.examples == ["Widget", "Gadget"]

    price_field = product_model.model_fields["price"]
    assert price_field.title == "Price"
    assert price_field.description == "Price in USD"


def test_preserves_field_validators(manager: ModelManager) -> None:
    """Test that field validators are preserved."""

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        email: str

        @field_validator("email")
        @classmethod
        def validate_email(cls, v: str) -> str:
            if "@" not in v:
                raise ValueError("Invalid email")
            return v.lower()

    user_model = manager.get("User", "1.0.0")

    user = user_model(email="Test@Example.com")
    assert user.email == "test@example.com"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="Invalid email"):
        user_model(email="invalid")


def test_preserves_model_validators(manager: ModelManager) -> None:
    """Test that model validators are preserved."""

    @manager.composite("DateRange", "1.0.0")
    class DateRangeV1(BaseModel):
        start: int
        end: int

        @model_validator(mode="after")
        def check_dates(self: Self) -> Self:
            if self.start >= self.end:
                raise ValueError("start must be before end")
            return self

    model = manager.get("DateRange", "1.0.0")

    date_range = model(start=1, end=10)
    assert date_range.start == 1  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="start must be before end"):
        model(start=10, end=5)


def test_preserves_default_values(manager: ModelManager) -> None:
    """Test that default values are preserved."""

    @manager.composite("Config", "1.0.0")
    class ConfigV1(BaseModel):
        timeout: int = 30
        retries: int = 3
        debug: bool = False

    config_model = manager.get("Config", "1.0.0")

    config = config_model()
    assert config.timeout == 30  # type: ignore[attr-defined]
    assert config.retries == 3  # type: ignore[attr-defined]
    assert config.debug is False  # type: ignore[attr-defined]

    config = config_model(timeout=60, debug=True)
    assert config.timeout == 60  # type: ignore[attr-defined]
    assert config.debug is True  # type: ignore[attr-defined]


def test_preserves_default_factory(manager: ModelManager) -> None:
    """Test that default_factory is preserved."""

    @manager.composite("Container", "1.0.0")
    class ContainerV1(BaseModel):
        items: list[str] = Field(default_factory=list)
        metadata: dict[str, str] = Field(default_factory=dict)

    container_model = manager.get("Container", "1.0.0")

    c1 = container_model()
    c2 = container_model()

    c1.items.append("item1")  # type: ignore[attr-defined]
    assert len(c1.items) == 1  # type: ignore[attr-defined]
    assert len(c2.items) == 0  # type: ignore[attr-defined]


def test_preserves_field_alias(manager: ModelManager) -> None:
    """Test that field aliases are preserved."""

    @manager.composite("ApiResponse", "1.0.0")
    class ApiResponseV1(BaseModel):
        user_name: str = Field(alias="userName")
        user_id: int = Field(alias="userId")

    response_model = manager.get("ApiResponse", "1.0.0")

    response = response_model.model_validate({"userName": "Alice", "userId": 123})
    assert response.user_name == "Alice"  # type: ignore[attr-defined]
    assert response.user_id == 123  # type: ignore[attr-defined]


def test_preserves_discriminated_union(manager: ModelManager) -> None:
    """Test that discriminated unions are preserved."""

    @manager.composite("Cat", "1.0.0")
    class CatV1(BaseModel):
        pet_type: Literal["cat"] = "cat"
        meow: str

    @manager.composite("Dog", "1.0.0")
    class DogV1(BaseModel):
        pet_type: Literal["dog"] = "dog"
        bark: str

    @manager.composite("Pet", "1.0.0")
    class PetV1(BaseModel):
        animal: CatV1 | DogV1 = Field(discriminator="pet_type")

    pet_model = manager.get("Pet", "1.0.0")

    pet_cat = pet_model.model_validate({"animal": {"pet_type": "cat", "meow": "meow"}})
    assert isinstance(pet_cat.animal, CatV1)  # type: ignore[attr-defined]

    pet_dog = pet_model.model_validate({"animal": {"pet_type": "dog", "bark": "woof"}})
    assert isinstance(pet_dog.animal, DogV1)  # type: ignore[attr-defined]


def test_preserves_model_config(manager: ModelManager) -> None:
    """Test that model_config is preserved."""

    @manager.composite("StrictModel", "1.0.0")
    class StrictModelV1(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            str_strip_whitespace=True,
        )

        name: str
        age: int

    model = manager.get("StrictModel", "1.0.0")

    with pytest.raises(ValueError):
        model(name="Alice", age=25, extra="not_allowed")

    obj = model(name="  Alice  ", age=25)
    assert obj.name == "Alice"  # type: ignore[attr-defined]

    obj = model(name="Alice", age=25)
    with pytest.raises(ValueError):
        obj.age = "not_an_int"  # type: ignore[attr-defined]


def test_preserves_computed_field(manager: ModelManager) -> None:
    """Test that computed fields work correctly."""

    @manager.composite("Person", "1.0.0")
    class PersonV1(BaseModel):
        first_name: str
        last_name: str

        @property
        def full_name(self) -> str:
            return f"{self.first_name} {self.last_name}"

    person_model = manager.get("Person", "1.0.0")

    person = person_model(first_name="John", last_name="Doe")
    assert person.full_name == "John Doe"  # type: ignore[attr-defined]


def test_preserves_json_schema_extra(manager: ModelManager) -> None:
    """Test that json_schema_extra is preserved."""

    @manager.composite("ApiModel", "1.0.0")
    class ApiModelV1(BaseModel):
        name: str = Field(
            json_schema_extra={
                "x-custom": "value",
                "x-validation-rule": "special",
            }
        )

    model = manager.get("ApiModel", "1.0.0")
    schema = model.model_json_schema()

    name_schema = schema["properties"]["name"]
    assert name_schema.get("x-custom") == "value"
    assert name_schema.get("x-validation-rule") == "special"


def test_preserves_frozen_fields(manager: ModelManager) -> None:
    """Test that frozen fields are preserved."""

    @manager.composite("ImmutableModel", "1.0.0")
    class ImmutableModelV1(BaseModel):
        model_config = ConfigDict(frozen=True)

        id: str
        value: int

    model = manager.get("ImmutableModel", "1.0.0")

    obj = model(id="123", value=42)

    with pytest.raises(ValueError):
        obj.value = 100  # type: ignore[attr-defined]


def test_preserves_field_repr(manager: ModelManager) -> None:
    """Test that field repr setting is preserved."""

    @manager.composite("SecretModel", "1.0.0")
    class SecretModelV1(BaseModel):
        username: str
        password: str = Field(repr=False)

    model = manager.get("SecretModel", "1.0.0")
    obj = model(username="admin", password="secret123")

    repr_str = repr(obj)
    assert "admin" in repr_str
    assert "secret123" not in repr_str


def test_preserves_multiple_validators(manager: ModelManager) -> None:
    """Test that multiple validators on the same field are preserved."""

    @manager.composite("ValidatedModel", "1.0.0")
    class ValidatedModelV1(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def check_positive(cls, v: int) -> int:
            if v <= 0:
                raise ValueError("Must be positive")
            return v

        @field_validator("value")
        @classmethod
        def check_even(cls, v: int) -> int:
            if v % 2 != 0:
                raise ValueError("Must be even")
            return v

    model = manager.get("ValidatedModel", "1.0.0")

    # Should pass both validators
    obj = model(value=4)
    assert obj.value == 4  # type: ignore[attr-defined]

    # Should fail first validator
    with pytest.raises(ValueError, match="Must be positive"):
        model(value=-2)

    # Should fail second validator
    with pytest.raises(ValueError, match="Must be even"):
        model(value=3)


def test_preserves_field_exclude(manager: ModelManager) -> None:
    """Test that field exclude setting is preserved."""

    @manager.composite("ModelWithExclude", "1.0.0")
    class ModelWithExcludeV1(BaseModel):
        public: str
        private: str = Field(exclude=True)

    model = manager.get("ModelWithExclude", "1.0.0")
    obj = model(public="visible", private="hidden")

    data = obj.model_dump()
    assert "public" in data
    assert "private" not in data


# ============================================================================
# Integration tests for composite models
# ============================================================================


def test_composite_with_nested_versions(manager: ModelManager) -> None:
    """Test composite model with different nested versions."""

    @manager.composite("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: str

    @manager.composite("Address", "2.0.0")
    class AddressV2(BaseModel):
        street: str
        city: str
        postal_code: str

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    @manager.composite("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        address: AddressV2

    user_v1 = manager.get("User", "1.0.0")
    u1 = user_v1(name="Alice", address={"street": "123 Main", "city": "NYC"})
    assert hasattr(u1.address, "street")  # type: ignore[attr-defined]
    assert hasattr(u1.address, "city")  # type: ignore[attr-defined]
    assert not hasattr(u1.address, "postal_code")  # type: ignore[attr-defined]

    user_v2 = manager.get("User", "2.0.0")
    u2 = user_v2(
        name="Bob",
        address={"street": "456 Elm", "city": "LA", "postal_code": "90001"},
    )
    assert hasattr(u2.address, "street")  # type: ignore[attr-defined]
    assert hasattr(u2.address, "city")  # type: ignore[attr-defined]
    assert hasattr(u2.address, "postal_code")  # type: ignore[attr-defined]


def test_composite_with_field_validators_on_nested(manager: ModelManager) -> None:
    """Test that validators work on nested models in composites."""

    @manager.composite("Email", "1.0.0")
    class EmailV1(BaseModel):
        address: str

        @field_validator("address")
        @classmethod
        def validate_email(cls, v: str) -> str:
            if "@" not in v:
                raise ValueError("Invalid email")
            return v

    @manager.composite("Contact", "1.0.0")
    class ContactV1(BaseModel):
        name: str
        email: EmailV1

    contact_model = manager.get("Contact", "1.0.0")

    contact = contact_model(name="Alice", email={"address": "alice@example.com"})
    assert contact.email.address == "alice@example.com"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="Invalid email"):
        contact_model(name="Bob", email={"address": "invalid"})


def test_composite_inheritance_with_validators(manager: ModelManager) -> None:
    """Test that validators are preserved through inheritance."""

    @manager.composite("Base", "1.0.0")
    class BaseV1(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def check_positive(cls, v: int) -> int:
            if v <= 0:
                raise ValueError("Value must be positive")
            return v

    @manager.composite("Base", "2.0.0", inherit_from="1.0.0")
    class BaseV2(BaseModel):
        value: int
        extra: str = "default"

        @field_validator("value")
        @classmethod
        def check_positive(cls, v: int) -> int:
            if v <= 0:
                raise ValueError("Value must be positive")
            return v

    model_v2 = manager.get("Base", "2.0.0")

    # Validator should still work
    with pytest.raises(ValueError, match="Value must be positive"):
        model_v2(value=-5)

    # Valid value
    obj = model_v2(value=10)
    assert obj.value == 10  # type: ignore[attr-defined]


def test_composite_with_list_of_nested_validators(manager: ModelManager) -> None:
    """Test validators work on lists of nested models."""

    @manager.composite("Tag", "1.0.0")
    class TagV1(BaseModel):
        name: str

        @field_validator("name")
        @classmethod
        def validate_name(cls, v: str) -> str:
            if len(v) < 2:
                raise ValueError("Tag name too short")
            return v.lower()

    @manager.composite("Post", "1.0.0")
    class PostV1(BaseModel):
        title: str
        tags: list[TagV1]

    post_model = manager.get("Post", "1.0.0")

    post = post_model(title="My Post", tags=[{"name": "Python"}, {"name": "Testing"}])
    assert post.tags[0].name == "python"  # type: ignore[attr-defined]
    assert post.tags[1].name == "testing"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="Tag name too short"):
        post_model(title="My Post", tags=[{"name": "x"}])


def test_composite_with_optional_nested_validators(manager: ModelManager) -> None:
    """Test validators work on optional nested models."""

    @manager.composite("Profile", "1.0.0")
    class ProfileV1(BaseModel):
        bio: str

        @field_validator("bio")
        @classmethod
        def validate_bio(cls, v: str) -> str:
            if len(v) > 500:
                raise ValueError("Bio too long")
            return v

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        profile: ProfileV1 | None = None

    user_model = manager.get("User", "1.0.0")

    user = user_model(name="Alice")
    assert user.profile is None  # type: ignore[attr-defined]

    user = user_model(name="Bob", profile={"bio": "Short bio"})
    assert user.profile.bio == "Short bio"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="Bio too long"):
        user_model(name="Charlie", profile={"bio": "x" * 501})


def test_composite_with_dict_of_nested_validators(manager: ModelManager) -> None:
    """Test validators work on dicts of nested models."""

    @manager.composite("Permission", "1.0.0")
    class PermissionV1(BaseModel):
        level: int

        @field_validator("level")
        @classmethod
        def validate_level(cls, v: int) -> int:
            if v < 0 or v > 10:
                raise ValueError("Level must be 0-10")
            return v

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        permissions: dict[str, PermissionV1]

    user_model = manager.get("User", "1.0.0")

    # Valid permissions
    user = user_model(
        name="Alice",
        permissions={"read": {"level": 5}, "write": {"level": 8}},
    )
    assert user.permissions["read"].level == 5  # type: ignore[attr-defined]

    # Invalid permission
    with pytest.raises(ValueError, match="Level must be 0-10"):
        user_model(name="Bob", permissions={"admin": {"level": 15}})


def test_composite_deep_nesting_with_validators(manager: ModelManager) -> None:
    """Test validators work through deep nesting."""

    @manager.composite("City", "1.0.0")
    class CityV1(BaseModel):
        name: str

        @field_validator("name")
        @classmethod
        def validate_name(cls, v: str) -> str:
            if len(v) < 2:
                raise ValueError("City name too short")
            return v.title()

    @manager.composite("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: CityV1

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    user_model = manager.get("User", "1.0.0")

    # Valid deep nesting
    user = user_model(
        name="Alice",
        address={"street": "123 Main", "city": {"name": "new york"}},
    )
    assert user.address.city.name == "New York"  # type: ignore[attr-defined]

    # Invalid deep nested field
    with pytest.raises(ValueError, match="City name too short"):
        user_model(name="Bob", address={"street": "456 Elm", "city": {"name": "x"}})
