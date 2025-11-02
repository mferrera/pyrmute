"""Tests for type stub generation."""

from pathlib import Path

from pydantic import BaseModel

from pyrmute import ModelManager


def test_generate_basic_stub(tmp_path: Path, manager: ModelManager) -> None:
    """Test generating a basic stub file."""
    manager = ModelManager()

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: str

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        email: str
        address: AddressV1

    manager.export_type_stubs(tmp_path, package_name="testapp")

    stub_file = tmp_path / "__init__.pyi"
    assert stub_file.exists()

    py_typed = tmp_path / "py.typed"
    assert py_typed.exists()

    content = stub_file.read_text()
    assert "class UserV1_0_0(BaseModel):" in content
    assert "name: str" in content
    assert "email: str" in content
    assert "address: AddressV1_0_0" in content
    assert "class AddressV1_0_0(BaseModel):" in content
    assert "User = UserV1_0_0  # Latest version" in content


def test_stub_with_optional_fields(tmp_path: Path, manager: ModelManager) -> None:
    """Test stub generation with optional fields."""

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        email: str | None = None
        age: int | None = None

    manager.export_type_stubs(tmp_path, package_name="testapp")

    content = (tmp_path / "__init__.pyi").read_text()
    assert "email: str | None" in content
    assert "age: int | None" in content


def test_stub_with_collections(tmp_path: Path, manager: ModelManager) -> None:
    """Test stub generation with list/dict types."""

    @manager.model("Tag", "1.0.0")
    class TagV1(BaseModel):
        name: str

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        tags: list[TagV1]
        metadata: dict[str, str]

    manager.export_type_stubs(tmp_path, package_name="testapp")

    content = (tmp_path / "__init__.pyi").read_text()
    assert "tags: list[TagV1_0_0]" in content
    assert "metadata: dict[str, str]" in content


def test_stub_with_deep_nesting(tmp_path: Path, manager: ModelManager) -> None:
    """Test that nested dependencies are included."""

    @manager.composite("City", "1.0.0")
    class CityV1(BaseModel):
        name: str

    @manager.composite("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: CityV1

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    manager.export_type_stubs(
        tmp_path,
        package_name="testapp",
        models=[("User", "1.0.0")],
        include_nested=True,
    )

    content = (tmp_path / "__init__.pyi").read_text()
    assert "class UserV1_0_0(BaseModel):" in content
    assert "class AddressV1_0_0(BaseModel):" in content
    assert "class CityV1_0_0(BaseModel):" in content


def test_stub_multiple_versions(tmp_path: Path, manager: ModelManager) -> None:
    """Test stub generation with multiple versions."""

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.composite("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    manager.export_type_stubs(tmp_path, package_name="testapp")

    content = (tmp_path / "__init__.pyi").read_text()
    assert "class UserV1_0_0(BaseModel):" in content
    assert "class UserV2_0_0(BaseModel):" in content
    assert "User = UserV2_0_0  # Latest version" in content


def test_stub_with_docstrings(tmp_path: Path, manager: ModelManager) -> None:
    """Test that docstrings are preserved in stubs."""

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        """A user in the system."""

        name: str

    manager.export_type_stubs(tmp_path, package_name="testapp")

    content = (tmp_path / "__init__.pyi").read_text()
    assert "A user in the system." in content


def test_export_by_module(tmp_path: Path, manager: ModelManager) -> None:
    """Test exporting stubs organized by module."""
    manager = ModelManager()

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.composite("Order", "1.0.0")
    class OrderV1(BaseModel):
        total: float

    manager.export_type_stubs_by_module(
        tmp_path,
        "testapp.models",
        {
            "user": [("User", "1.0.0")],
            "order": [("Order", "1.0.0")],
        },
    )

    assert (tmp_path / "user.pyi").exists()
    assert (tmp_path / "order.pyi").exists()

    init_content = (tmp_path / "__init__.pyi").read_text()
    assert "from .user import *" in init_content
    assert "from .order import *" in init_content

    user_content = (tmp_path / "user.pyi").read_text()
    assert "class UserV1_0_0(BaseModel):" in user_content

    order_content = (tmp_path / "order.pyi").read_text()
    assert "class OrderV1_0_0(BaseModel):" in order_content
