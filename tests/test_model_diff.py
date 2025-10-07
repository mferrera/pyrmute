"""Tests the ModelDiff class."""

from typing import Any

from pydantic import BaseModel, Field

from pyrmute import ModelManager, ModelVersion


def multiple_breaking_warnings(markdown: str) -> bool:
    """Check if there are multiple breaking change warnings."""
    return markdown.count("⚠️") > 1


# Markdown generation tests
def test_diff_to_markdown_default_depth(manager: ModelManager) -> None:
    """Test diff markdown generation with default header depth."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "# User: 1.0.0 → 2.0.0" in markdown
    assert "## Added Fields" in markdown
    assert "## Removed Fields" in markdown
    assert "## Modified Fields" in markdown
    assert "## Breaking Changes" in markdown


def test_diff_to_markdown_custom_depth(manager: ModelManager) -> None:
    """Test diff markdown generation with custom header depth."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    diff = manager.diff("User", "1.0.0", "2.0.0")

    markdown2 = diff.to_markdown(header_depth=2)
    assert "## User: 1.0.0 → 2.0.0" in markdown2
    assert "### Added Fields" in markdown2
    assert "### Removed Fields" in markdown2
    assert "### Modified Fields" in markdown2

    markdown3 = diff.to_markdown(header_depth=3)
    assert "### User: 1.0.0 → 2.0.0" in markdown3
    assert "#### Added Fields" in markdown3
    assert "#### Removed Fields" in markdown3
    assert "#### Modified Fields" in markdown3


def test_diff_to_markdown_depth_clamping(manager: ModelManager) -> None:
    """Test that header depth is clamped to valid range (1-6)."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    diff = manager.diff("User", "1.0.0", "1.0.0")

    markdown_low = diff.to_markdown(header_depth=0)
    assert markdown_low.startswith("# User:")

    markdown_high = diff.to_markdown(header_depth=10)
    assert markdown_high.startswith("###### User:")


def test_diff_to_markdown_shows_added_fields(manager: ModelManager) -> None:
    """Test that markdown shows added fields with type info."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str
        age: int | None = None

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "`email: str` (required)" in markdown
    assert "`age:" in markdown
    assert "(optional)" in markdown


def test_diff_to_markdown_shows_removed_fields(manager: ModelManager) -> None:
    """Test that markdown shows removed fields."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        username: str
        legacy_id: int

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "`username`" in markdown
    assert "`legacy_id`" in markdown


def test_diff_to_markdown_shows_modified_fields(manager: ModelManager) -> None:
    """Test that markdown shows modified fields with details."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        age: int

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        age: str

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "`age`" in markdown
    assert "type:" in markdown
    assert "`int` → `str`" in markdown or "int` → `str" in markdown


def test_diff_to_markdown_shows_requirement_changes(manager: ModelManager) -> None:
    """Test that markdown shows required/optional changes."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        email: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        email: str | None = None

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "`email`" in markdown
    assert "now optional" in markdown


def test_diff_to_markdown_shows_default_changes(manager: ModelManager) -> None:
    """Test that markdown shows default value changes."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        status: str = "active"

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        status: str = "pending"

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "`status`" in markdown
    assert "default:" in markdown
    assert "`active` → `pending`" in markdown


def test_diff_to_markdown_shows_default_added(manager: ModelManager) -> None:
    """Test that markdown shows when default is added."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        status: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        status: str = "pending"

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "`status`" in markdown
    assert "default added:" in markdown
    assert "`pending`" in markdown


def test_diff_to_markdown_shows_default_removed(manager: ModelManager) -> None:
    """Test that markdown shows when default is removed."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        status: str = "active"

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        status: str

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "`status`" in markdown
    assert "default removed" in markdown
    assert "`active`" in markdown


def test_diff_to_markdown_shows_breaking_changes(manager: ModelManager) -> None:
    """Test that markdown identifies breaking changes."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "## Breaking Changes" in markdown
    assert "⚠️" in markdown
    assert "email" in markdown
    assert "will fail for existing data" in markdown


def test_diff_to_markdown_breaking_removed_fields(manager: ModelManager) -> None:
    """Test that markdown warns about removed fields."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        username: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "⚠️" in markdown
    assert "username" in markdown
    assert "will be lost" in markdown


def test_diff_to_markdown_breaking_optional_to_required(
    manager: ModelManager,
) -> None:
    """Test that markdown warns about optional to required changes."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        email: str | None = None

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        email: str

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "⚠️" in markdown
    assert "email" in markdown
    assert "optional to required" in markdown


def test_diff_to_markdown_breaking_type_changes(manager: ModelManager) -> None:
    """Test that markdown warns about type changes."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        age: int

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        age: str

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "⚠️" in markdown
    assert "age" in markdown
    assert "type changed" in markdown


def test_diff_to_markdown_no_changes(manager: ModelManager) -> None:
    """Test markdown when there are no changes."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        email: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "## Added Fields" in markdown
    assert "None" in markdown
    assert "## Removed Fields" in markdown

    # Breaking changes section should not appear if no breaking changes
    breaking_count = markdown.count("## Breaking Changes")
    assert breaking_count == 0


def test_diff_to_markdown_complex_scenario(manager: ModelManager) -> None:
    """Test markdown with a complex mix of changes."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str
        username: str
        age: int
        status: str = "active"

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str
        age: str | None = None
        role: str = "user"

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    # Added fields
    assert "`email:" in markdown
    assert "`role:" in markdown

    # Removed fields
    assert "`username`" in markdown
    assert "`status`" in markdown

    # Modified fields
    assert "`age`" in markdown
    assert "type:" in markdown

    # Breaking changes
    assert "⚠️" in markdown
    assert multiple_breaking_warnings(markdown)


def test_diff_to_markdown_sorted_fields(manager: ModelManager) -> None:
    """Test that fields are sorted alphabetically in markdown."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        zebra: str
        apple: str
        middle: str

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    # Find positions of field names in the markdown
    apple_pos = markdown.find("`apple")
    middle_pos = markdown.find("`middle")
    zebra_pos = markdown.find("`zebra")

    assert apple_pos < middle_pos < zebra_pos


def test_diff_to_markdown_with_field_validators(manager: ModelManager) -> None:
    """Test markdown works with Pydantic Field validators."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        age: int = Field(ge=0)

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        age: int = Field(ge=0, le=120)

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    # Should not show as modified since the type and requirement didn't change
    assert "## Modified Fields" in markdown
    assert isinstance(markdown, str)
    assert len(markdown) > 0


def test_diff_to_markdown_with_model_versions(manager: ModelManager) -> None:
    """Test markdown generation with ModelVersion objects."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    diff = manager.diff("User", ModelVersion(1, 0, 0), ModelVersion(2, 0, 0))
    markdown = diff.to_markdown()

    assert "# User: 1.0.0 → 2.0.0" in markdown
    assert "`email:" in markdown


def test_diff_to_markdown_multiple_changes_same_field(
    manager: ModelManager,
) -> None:
    """Test markdown shows all changes to a single field."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        age: int

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        age: str | None = "0"

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "`age`" in markdown
    assert "type:" in markdown
    assert "now optional" in markdown
    assert "default added:" in markdown


def test_diff_to_markdown_formats_types_cleanly(manager: ModelManager) -> None:
    """Test that type annotations are formatted cleanly."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        data: dict[str, Any]

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        data: list[str]

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    assert "`data`" in markdown
    assert "type:" in markdown


def test_diff_markdown_is_valid_markdown(manager: ModelManager) -> None:
    """Test that generated markdown is valid and well-formed."""

    @manager.model("User", "1.0.0")
    class UserV1(BaseModel):
        name: str

    @manager.model("User", "2.0.0")
    class UserV2(BaseModel):
        name: str
        email: str

    diff = manager.diff("User", "1.0.0", "2.0.0")
    markdown = diff.to_markdown()

    lines = markdown.split("\n")
    assert any(line.startswith("# ") for line in lines)
    assert any(line.startswith("## ") for line in lines)

    assert "" in lines
    assert any(line.startswith("- ") for line in lines)
