"""Tests for ModelVersion."""

# ruff: noqa: PLR2004

import pytest

from pyrmute import InvalidVersionError, ModelVersion


def test_version_creation() -> None:
    """Test creating a version with valid values."""
    version = ModelVersion(1, 2, 3)
    assert version.major == 1
    assert version.minor == 2
    assert version.patch == 3


def test_version_immutable() -> None:
    """Test that version is immutable (frozen dataclass)."""
    version = ModelVersion(1, 0, 0)
    with pytest.raises(AttributeError):
        version.major = 2  # type: ignore


def test_parse_valid_version() -> None:
    """Test parsing a valid semantic version string."""
    version = ModelVersion.parse("1.2.3")
    assert version.major == 1
    assert version.minor == 2
    assert version.patch == 3


def test_parse_zero_version() -> None:
    """Test parsing version with zero values."""
    version = ModelVersion.parse("0.0.0")
    assert version.major == 0
    assert version.minor == 0
    assert version.patch == 0


def test_parse_large_numbers() -> None:
    """Test parsing version with large numbers."""
    version = ModelVersion.parse("100.200.300")
    assert version.major == 100
    assert version.minor == 200
    assert version.patch == 300


def test_parse_invalid_too_few_parts() -> None:
    """Test parsing fails with too few version parts."""
    with pytest.raises(InvalidVersionError, match=r"Invalid version string: '1.2'"):
        ModelVersion.parse("1.2")


def test_parse_invalid_too_many_parts() -> None:
    """Test parsing fails with too many version parts."""
    with pytest.raises(InvalidVersionError, match=r"Invalid version string: '1.2.3.4'"):
        ModelVersion.parse("1.2.3.4")


def test_parse_invalid_non_numeric() -> None:
    """Test parsing fails with non-numeric parts."""
    with pytest.raises(InvalidVersionError, match=r"Invalid version string: '1.2.x'"):
        ModelVersion.parse("1.2.x")


def test_parse_invalid_negative_numbers() -> None:
    """Test parsing fails with negative numbers."""
    with pytest.raises(InvalidVersionError, match=r"Invalid version string: '1\.-2.3'"):
        ModelVersion.parse("1.-2.3")


def test_parse_invalid_empty_string() -> None:
    """Test parsing fails with empty string."""
    with pytest.raises(InvalidVersionError, match="Invalid version string: ''"):
        ModelVersion.parse("")


def test_parse_invalid_float() -> None:
    """Test parsing fails with float values."""
    with pytest.raises(InvalidVersionError, match=r"Invalid version string: '1.2.3.5'"):
        ModelVersion.parse("1.2.3.5")


def test_str_representation() -> None:
    """Test string representation of version."""
    version = ModelVersion(1, 2, 3)
    assert str(version) == "1.2.3"


def test_str_with_zeros() -> None:
    """Test string representation preserves zeros."""
    version = ModelVersion(0, 0, 1)
    assert str(version) == "0.0.1"


def test_repr_representation() -> None:
    """Test detailed representation of version."""
    version = ModelVersion(1, 2, 3)
    assert repr(version) == "ModelVersion(1, 2, 3)"


def test_version_ordering_equal() -> None:
    """Test versions with same values are equal."""
    v1 = ModelVersion(1, 2, 3)
    v2 = ModelVersion(1, 2, 3)
    assert v1 == v2


def test_version_ordering_major() -> None:
    """Test ordering by major version."""
    v1 = ModelVersion(1, 0, 0)
    v2 = ModelVersion(2, 0, 0)
    assert v1 < v2
    assert v2 > v1


def test_version_ordering_minor() -> None:
    """Test ordering by minor version when major is equal."""
    v1 = ModelVersion(1, 1, 0)
    v2 = ModelVersion(1, 2, 0)
    assert v1 < v2
    assert v2 > v1


def test_version_ordering_patch() -> None:
    """Test ordering by patch version when major and minor are equal."""
    v1 = ModelVersion(1, 2, 3)
    v2 = ModelVersion(1, 2, 4)
    assert v1 < v2
    assert v2 > v1


def test_version_ordering_complex() -> None:
    """Test complex ordering scenarios."""
    versions = [
        ModelVersion(2, 0, 0),
        ModelVersion(1, 3, 0),
        ModelVersion(1, 2, 5),
        ModelVersion(1, 2, 3),
    ]
    sorted_versions = sorted(versions)
    assert sorted_versions == [
        ModelVersion(1, 2, 3),
        ModelVersion(1, 2, 5),
        ModelVersion(1, 3, 0),
        ModelVersion(2, 0, 0),
    ]


def test_parse_and_str_roundtrip() -> None:
    """Test that parsing and converting to string are inverse operations."""
    original = "1.2.3"
    version = ModelVersion.parse(original)
    assert str(version) == original


@pytest.mark.parametrize(
    "version_str,expected_major,expected_minor,expected_patch",
    [
        ("0.0.1", 0, 0, 1),
        ("1.0.0", 1, 0, 0),
        ("1.2.3", 1, 2, 3),
        ("10.20.30", 10, 20, 30),
        ("999.999.999", 999, 999, 999),
    ],
)
def test_parse_multiple_valid_versions(
    version_str: str,
    expected_major: int,
    expected_minor: int,
    expected_patch: int,
) -> None:
    """Test parsing various valid version strings."""
    version = ModelVersion.parse(version_str)
    assert version.major == expected_major
    assert version.minor == expected_minor
    assert version.patch == expected_patch


@pytest.mark.parametrize(
    "invalid_version",
    [
        "1",
        "1.2",
        "1.2.3.4",
        "a.b.c",
        "1.2.x",
        "1.x.3",
        "x.2.3",
        "1.2.3-beta",
        "v1.2.3",
        "",
        "1..3",
        ".1.2.3",
        "1.2.3.",
    ],
)
def test_parse_invalid_versions(invalid_version: str) -> None:
    """Test that invalid version strings raise ValueError."""
    with pytest.raises(
        InvalidVersionError, match=f"Invalid version string: '{invalid_version}'"
    ):
        ModelVersion.parse(invalid_version)
