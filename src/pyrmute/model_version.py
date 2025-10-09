"""Models the version provided to managers."""

from dataclasses import dataclass
from typing import Self

from .exceptions import InvalidVersionError


@dataclass(frozen=True, order=True)
class ModelVersion:
    """Semantic version representation.

    Attributes:
        major: Major version number (breaking changes).
        minor: Minor version number (backward-compatible features).
        patch: Patch version number (backward-compatible fixes).
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version_str: str) -> Self:
        """Parse a semantic version string.

        Args:
            version_str: Version string in format "major.minor.patch".

        Returns:
            Parsed Version instance.

        Raises:
            InvalidVersionError: If version string format is invalid.
        """
        try:
            parts = version_str.split(".")
            if len(parts) != 3:  # noqa: PLR2004
                raise InvalidVersionError(
                    version_str, "Version must have exactly 3 parts (major.minor.patch)"
                )

            major, minor, patch = map(int, parts)
            if major < 0 or minor < 0 or patch < 0:
                raise InvalidVersionError(
                    version_str, "Version parts must be positive integers"
                )
            return cls(major, minor, patch)
        except ValueError as e:
            raise InvalidVersionError(
                version_str, f"Version parts must be integers: {e}"
            ) from e

    def __str__(self: Self) -> str:
        """Return string representation of version.

        Returns:
            Version string in format "major.minor.patch".
        """
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self: Self) -> str:
        """Return detailed string representation.

        Returns:
            Detailed version representation.
        """
        return f"ModelVersion({self.major}, {self.minor}, {self.patch})"
