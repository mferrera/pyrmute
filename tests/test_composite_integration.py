"""Full integration tests for composite versioning feature."""

from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from pyrmute import InvalidVersionError, ModelData, ModelManager


@pytest.mark.xfail(reason="Nested model migrations not yet added")
def test_full_workflow_simple(manager: ModelManager) -> None:
    """Test complete workflow: register, migrate, export."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: str

    @manager.model("Address", "2.0.0")
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

    @manager.migration("Address", "1.0.0", "2.0.0")
    def upgrade_address(data: ModelData) -> ModelData:
        return {**data, "postal_code": "00000"}

    @manager.migration("User", "1.0.0", "2.0.0")
    def upgrade_user(data: ModelData) -> ModelData:
        return data

    user_v1_data = {"name": "Alice", "address": {"street": "123 Main", "city": "NYC"}}
    user_v2 = manager.migrate(user_v1_data, "User", "1.0.0", "2.0.0")

    assert user_v2.name == "Alice"  # type: ignore[attr-defined]
    assert user_v2.address.postal_code == "00000"  # type: ignore[attr-defined]

    pins = manager.get_nested_version_pins("User", "2.0.0")
    assert pins["address"] == ("Address", "2.0.0")

    schema = manager.get_schema("User", "2.0.0")
    assert "properties" in schema
    assert "address" in schema["properties"]  # type: ignore[operator]


def test_full_workflow_deep_nesting(manager: ModelManager) -> None:
    """Test workflow with 6 levels of nesting."""

    @manager.composite("Person", "1.0.0")
    class PersonV1(BaseModel):
        """A person."""

        name: str
        email: str

    @manager.composite("Assignment", "1.0.0")
    class AssignmentV1(BaseModel):
        """Task assignment."""

        person: PersonV1
        role: str

    @manager.composite("Task", "1.0.0")
    class TaskV1(BaseModel):
        """A task."""

        title: str
        assignment: AssignmentV1

    @manager.composite("Project", "1.0.0")
    class ProjectV1(BaseModel):
        """A project."""

        name: str
        task: TaskV1

    @manager.composite("Team", "1.0.0")
    class TeamV1(BaseModel):
        """A team."""

        name: str
        project: ProjectV1

    @manager.composite("Department", "1.0.0")
    class DepartmentV1(BaseModel):
        """A department."""

        name: str
        team: TeamV1

    @manager.composite("Company", "1.0.0")
    class CompanyV1(BaseModel):
        """A company."""

        name: str
        department: DepartmentV1

    company_pins = manager.get_nested_version_pins("Company", "1.0.0")
    assert len(company_pins) == 6  # noqa: PLR2004
    assert "department" in company_pins
    assert "department.team.project.task.assignment.person" in company_pins

    company_data = {
        "name": "Acme Corp",
        "department": {
            "name": "Engineering",
            "team": {
                "name": "Platform",
                "project": {
                    "name": "Migration",
                    "task": {
                        "title": "Database Schema",
                        "assignment": {
                            "person": {"name": "Alice", "email": "alice@acme.com"},
                            "role": "Lead",
                        },
                    },
                },
            },
        },
    }

    company = CompanyV1.model_validate(company_data)
    assert company.department.team.project.task.assignment.person.name == "Alice"

    @manager.composite("Person", "2.0.0")
    class PersonV2(BaseModel):
        """A person with phone."""

        name: str
        email: str
        phone: str | None = None

    @manager.composite("Assignment", "2.0.0")
    class AssignmentV2(BaseModel):
        """Task assignment with updated person."""

        person: PersonV2
        role: str

    @manager.composite("Task", "2.0.0")
    class TaskV2(BaseModel):
        """A task with updated assignment."""

        title: str
        assignment: AssignmentV2

    @manager.composite("Project", "2.0.0")
    class ProjectV2(BaseModel):
        """A project with updated task."""

        name: str
        task: TaskV2

    @manager.composite("Team", "2.0.0")
    class TeamV2(BaseModel):
        """A team with updated project."""

        name: str
        project: ProjectV2

    @manager.composite("Department", "2.0.0")
    class DepartmentV2(BaseModel):
        """A department with updated team."""

        name: str
        team: TeamV2

    @manager.composite("Company", "2.0.0")
    class CompanyV2(BaseModel):
        """A company with updated department."""

        name: str
        department: DepartmentV2

    company_v2_pins = manager.get_nested_version_pins("Company", "2.0.0")
    assert company_v2_pins["department.team.project.task.assignment.person"] == (
        "Person",
        "2.0.0",
    )

    @manager.migration("Person", "1.0.0", "2.0.0")
    def add_phone(data: ModelData) -> ModelData:
        return {**data, "phone": None}

    for model_name in [
        "Assignment",
        "Task",
        "Project",
        "Team",
        "Department",
        "Company",
    ]:

        @manager.migration(model_name, "1.0.0", "2.0.0")
        def passthrough(data: ModelData) -> ModelData:
            return data

    company_v2 = manager.migrate(company_data, "Company", "1.0.0", "2.0.0")
    assert company_v2.department.team.project.task.assignment.person.phone is None  # type: ignore[attr-defined]


def test_full_workflow_with_inheritance(manager: ModelManager) -> None:
    """Test workflow using version inheritance."""

    @manager.model("City", "1.0.0")
    class CityV1(BaseModel):
        name: str

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str
        city: CityV1

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        address: AddressV1

    @manager.model("City", "2.0.0")
    class CityV2(BaseModel):
        name: str
        country: str = "US"

    @manager.model("Address", "2.0.0")
    class AddressV2(BaseModel):
        street: str
        city: CityV2

    @manager.composite("User", "2.0.0", inherit_from="1.0.0")
    class UserV2(BaseModel):
        name: str
        address: AddressV2

    manager.get_nested_version_pins("User", "1.0.0")
    pins_v2 = manager.get_nested_version_pins("User", "2.0.0")

    assert pins_v2["address"] == ("Address", "2.0.0")
    assert pins_v2["address.city"] == ("City", "2.0.0")


def test_full_workflow_with_collections(manager: ModelManager) -> None:
    """Test workflow with lists and dicts."""

    @manager.composite("Tag", "1.0.0")
    class TagV1(BaseModel):
        name: str
        color: str

    @manager.composite("Comment", "1.0.0")
    class CommentV1(BaseModel):
        text: str
        author: str

    @manager.composite("Post", "1.0.0")
    class PostV1(BaseModel):
        title: str
        tags: list[TagV1]
        comments: list[CommentV1]
        metadata: dict[str, str]

    post_data = {
        "title": "Hello World",
        "tags": [
            {"name": "python", "color": "blue"},
            {"name": "tutorial", "color": "green"},
        ],
        "comments": [{"text": "Great post!", "author": "Alice"}],
        "metadata": {"category": "tech", "lang": "en"},
    }

    post = PostV1.model_validate(post_data)
    assert len(post.tags) == 2  # noqa: PLR2004
    assert post.tags[0].name == "python"
    assert len(post.comments) == 1

    pins = manager.get_nested_version_pins("Post", "1.0.0")
    assert ("Tag", "1.0.0") in [tuple(v) for v in pins.values()]
    assert ("Comment", "1.0.0") in [tuple(v) for v in pins.values()]


def test_full_workflow_with_optional_fields(manager: ModelManager) -> None:
    """Test workflow with optional nested models."""

    @manager.composite("Avatar", "1.0.0")
    class AvatarV1(BaseModel):
        url: str
        size: int

    @manager.composite("Profile", "1.0.0")
    class ProfileV1(BaseModel):
        bio: str
        avatar: AvatarV1 | None = None

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        profile: ProfileV1 | None = None

    user1 = UserV1(name="Alice", profile=None)
    assert user1.profile is None

    user2 = UserV1(
        name="Bob",
        profile={  # type: ignore[arg-type]
            "bio": "Developer",
            "avatar": None,
        },
    )
    assert user2.profile.avatar is None  # type: ignore[union-attr]

    user3 = UserV1(
        name="Charlie",
        profile={  # type: ignore[arg-type]
            "bio": "Designer",
            "avatar": {"url": "http://example.com/avatar.jpg", "size": 100},
        },
    )
    assert user3.profile.avatar.url == "http://example.com/avatar.jpg"  # type: ignore[union-attr]


def test_export_stubs_integration(tmp_path: Path, manager: ModelManager) -> None:
    """Test full stub export workflow."""

    @manager.composite("Address", "1.0.0")
    class AddressV1(BaseModel):
        """An address."""

        street: str
        city: str

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        """A user in the system."""

        name: str
        email: str
        address: AddressV1

    @manager.composite("User", "2.0.0")
    class UserV2(BaseModel):
        """A user with additional fields."""

        name: str
        email: str
        address: AddressV1
        age: int | None = None

    manager.export_type_stubs(tmp_path, package_name="testapp")

    assert (tmp_path / "__init__.pyi").exists()
    assert (tmp_path / "py.typed").exists()

    content = (tmp_path / "__init__.pyi").read_text()

    assert "Type stubs for testapp" in content
    assert "auto-generated" in content

    assert "from pydantic import BaseModel" in content

    assert "class AddressV1_0_0(BaseModel):" in content
    assert "class UserV1_0_0(BaseModel):" in content
    assert "class UserV2_0_0(BaseModel):" in content

    assert "An address." in content
    assert "A user in the system." in content

    assert "street: str" in content
    assert "city: str" in content
    assert "name: str" in content
    assert "email: str" in content
    assert "address: AddressV1_0_0" in content
    assert "age: int | None" in content

    assert "User = UserV2_0_0  # Latest version" in content
    assert "Address = AddressV1_0_0  # Latest version" in content


def test_schema_export_with_composite(manager: ModelManager) -> None:
    """Test that schema export works with composite models."""

    @manager.composite("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str = Field(description="Street address")
        city: str = Field(description="City name")

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str = Field(description="User's full name")
        address: AddressV1

    schema = manager.get_schema("User", "1.0.0")

    assert schema["type"] == "object"
    assert "properties" in schema
    assert "name" in schema["properties"]  # type: ignore[operator]
    assert "address" in schema["properties"]  # type: ignore[operator]

    assert schema["properties"]["name"]["description"] == "User's full name"  # type: ignore[index,call-overload]

    address_schema = schema["properties"]["address"]  # type: ignore[index,call-overload]
    assert "properties" in address_schema or "$ref" in address_schema  # type: ignore[operator]


def test_error_handling(manager: ModelManager) -> None:
    """Test error handling in composite versioning."""

    @manager.model("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str

    with pytest.raises(InvalidVersionError, match="Invalid version string: 'invalid'"):

        @manager.composite("User", "1.0.0", address="invalid")
        class UserV1(BaseModel):
            name: str
            address: AddressV1


def test_compatibility_with_regular_models(manager: ModelManager) -> None:
    """Test that composite models work alongside regular models."""

    @manager.model("Simple", "1.0.0")
    class SimpleV1(BaseModel):
        value: str

    @manager.composite("Address", "1.0.0")
    class AddressV1(BaseModel):
        street: str

    @manager.composite("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        simple: SimpleV1
        address: AddressV1

    user = UserV1(
        name="Alice",
        simple={"value": "test"},  # type: ignore[arg-type]
        address={"street": "123 Main"},  # type: ignore[arg-type]
    )

    assert user.simple.value == "test"
    assert user.address.street == "123 Main"

    pins = manager.get_nested_version_pins("User", "1.0.0")
    assert ("Simple", "1.0.0") in [tuple(v) for v in pins.values()]
    assert ("Address", "1.0.0") in [tuple(v) for v in pins.values()]
