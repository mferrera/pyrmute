"""Exceptions."""

from typing import Self


class VersionedModelError(Exception):
    """Base exception for all versioned model errors."""


class ModelNotFoundError(VersionedModelError):
    """Raised when a model or version cannot be found in the registry."""

    def __init__(self: Self, name: str, version: str | None = None) -> None:
        """Initializes ModelNotFoundError."""
        self.name = name
        self.version = version
        if version:
            msg = f"Model '{name}' version '{version}' not found in registry"
        else:
            msg = f"Model '{name}' not found in registry"
        super().__init__(msg)


class MigrationError(VersionedModelError):
    """Raised when a migration fails or cannot be found."""

    def __init__(
        self: Self,
        name: str,
        from_version: str,
        to_version: str,
        reason: str | None = None,
    ) -> None:
        """Initializes MigrationError."""
        self.name = name
        self.from_version = from_version
        self.to_version = to_version
        self.reason = reason

        msg = f"Migration failed for '{name}': {from_version} → {to_version}"
        if reason:
            msg += f"\nReason: {reason}"
        super().__init__(msg)


class InvalidVersionError(VersionedModelError):
    """Raised when a version string cannot be parsed."""

    def __init__(self: Self, version_string: str, reason: str | None = None) -> None:
        """Initializes InvalidVersionError."""
        self.version_string = version_string
        msg = f"Invalid version string: '{version_string}'"
        if reason:
            msg += f"\n{reason}"
        super().__init__(msg)
