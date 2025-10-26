"""Model diff class."""

import json
from dataclasses import asdict, dataclass
from types import GenericAlias, UnionType
from typing import Any, Self, cast

from pydantic import BaseModel
from pydantic_core import PydanticUndefined


@dataclass
class ModelDiff:
    """Contains the difference between two models."""

    model_name: str
    from_version: str
    to_version: str
    added_fields: list[str]
    removed_fields: list[str]
    modified_fields: dict[str, Any]
    added_field_info: dict[str, Any]
    unchanged_fields: list[str]

    def to_markdown(self: Self, header_depth: int = 1) -> str:
        """Generate a markdown representation of the diff.

        Args:
            header_depth: Base header level (1-6). All headers are relative to this.
                For example, header_depth=2 makes the title "##" and subsections "###".

        Returns:
            Formatted markdown string showing the differences.

        Example:
            ```python
            diff.to_markdown(header_depth=1)  # Default: # Title, ## Sections
            diff.to_markdown(header_depth=2)  # ## Title, ### Sections
            diff.to_markdown(header_depth=3)  # ### Title, #### Sections
            ```
        """
        header_depth = max(1, min(6, header_depth))

        h1 = "#" * header_depth
        h2 = "#" * (header_depth + 1)

        lines = [
            f"{h1} {self.model_name}: {self.from_version} → {self.to_version}",
            "",
        ]

        lines.append(f"{h2} Added Fields")
        lines.append("")
        if self.added_fields:
            for field_name in sorted(self.added_fields):
                field_desc = self._format_field_description(field_name, "added")
                lines.append(f"- {field_desc}")
        else:
            lines.append("None")
        lines.append("")

        lines.append(f"{h2} Removed Fields")
        lines.append("")
        if self.removed_fields:
            for field_name in sorted(self.removed_fields):
                field_desc = self._format_field_description(field_name, "removed")
                lines.append(f"- {field_desc}")
        else:
            lines.append("None")
        lines.append("")

        lines.append(f"{h2} Modified Fields")
        lines.append("")
        if self.modified_fields:
            for field_name in sorted(self.modified_fields.keys()):
                changes = self.modified_fields[field_name]
                field_desc = self._format_modified_field(field_name, changes)
                lines.append(f"- {field_desc}")
        else:
            lines.append("None")
        lines.append("")

        breaking_changes = self._identify_breaking_changes()
        if breaking_changes:
            lines.append(f"{h2} Breaking Changes")
            lines.append("")
            lines.extend(f"-⚠️  {warning}" for warning in breaking_changes)
            lines.append("")

        return "\n".join(lines)

    def to_dict(self: Self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary.

        Converts type objects to their string representation for JSON compatibility.

        Returns:
            Dictionary with all type objects converted to strings.

        Example:
            ```python
            diff_dict = diff.to_dict()
            json.dumps(diff_dict, indent=2)
            ```
        """
        diff_dict = asdict(self)
        return cast("dict[str, Any]", self._serialize_for_json(diff_dict))

    def _serialize_for_json(self: Self, obj: Any) -> Any:  # noqa: PLR0911
        """Convert an object to JSON-serializable format, handling types."""
        if isinstance(obj, type):
            return obj.__name__
        if hasattr(obj, "__origin__"):
            return str(obj)
        if isinstance(obj, UnionType):
            return str(obj)
        if isinstance(obj, GenericAlias):
            return str(obj)
        if isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        try:
            # Maybe it's already json serializable
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    def _format_field_description(self: Self, field_name: str, context: str) -> str:
        """Format a field for display."""
        if context == "added" and field_name in self.added_field_info:
            info = self.added_field_info[field_name]
            type_str = self._format_type(info["type"])
            req_str = "required" if info["required"] else "optional"
            return f"`{field_name}: {type_str}` ({req_str})"

        return f"`{field_name}`"

    def _format_modified_field(
        self: Self, field_name: str, changes: dict[str, Any]
    ) -> str:
        """Format a modified field with its changes."""
        parts = [f"`{field_name}`"]

        if "type_changed" in changes:
            from_type = self._format_type(changes["type_changed"]["from"])
            to_type = self._format_type(changes["type_changed"]["to"])
            parts.append(f"type: `{from_type}` → `{to_type}`")

        if "required_changed" in changes:
            req_change = changes["required_changed"]
            if req_change["from"] and not req_change["to"]:
                parts.append("now optional")
            elif not req_change["from"] and req_change["to"]:
                parts.append("now required")

        if "default_changed" in changes:
            from_val = changes["default_changed"]["from"]
            to_val = changes["default_changed"]["to"]
            parts.append(f"default: `{from_val}` → `{to_val}`")

        if "default_added" in changes:
            parts.append(f"default added: `{changes['default_added']}`")

        if "default_removed" in changes:
            parts.append(f"default removed (was `{changes['default_removed']}`)")

        return " - ".join(parts)

    def _format_type(self: Self, type_annotation: Any) -> str:
        """Format a type annotation for display."""
        if hasattr(type_annotation, "__name__"):
            return str(type_annotation.__name__)

        type_str = str(type_annotation)
        type_str = type_str.replace("typing.", "")
        return type_str.replace("typing_extensions.", "")

    def _identify_breaking_changes(self: Self) -> list[str]:
        """Identify breaking changes that could cause issues."""
        warnings = []

        for field_name in self.added_fields:
            if field_name in self.added_field_info:
                info = self.added_field_info[field_name]
                is_required = info["required"] and info["default"] is None

                if is_required:
                    warnings.append(
                        f"New required field '{field_name}' will fail for existing "
                        "data without defaults"
                    )

        if self.removed_fields:
            fields_str = ", ".join(f"'{f}'" for f in sorted(self.removed_fields))
            warnings.append(
                f"Removed fields {fields_str} will be lost during migration"
            )

        for field_name, changes in self.modified_fields.items():
            if "required_changed" in changes:
                req_change = changes["required_changed"]
                if not req_change["from"] and req_change["to"]:
                    warnings.append(
                        f"Field '{field_name}' changed from optional to required"
                    )

            if "type_changed" in changes:
                warnings.append(
                    f"Field '{field_name}' type changed - may cause validation errors"
                )

        return warnings

    @classmethod
    def from_models(
        cls,
        name: str,
        from_model: type[BaseModel],
        to_model: type[BaseModel],
        from_version: str,
        to_version: str,
    ) -> Self:
        """Create a ModelDiff by comparing two Pydantic models.

        Args:
            name: Name of the model.
            from_model: Source model class.
            to_model: Target model class.
            from_version: Source version string.
            to_version: Target version string.

        Returns:
            ModelDiff instance with computed differences.
        """
        from_fields = from_model.model_fields
        to_fields = to_model.model_fields

        from_keys = set(from_fields.keys())
        to_keys = set(to_fields.keys())

        added = list(to_keys - from_keys)
        removed = list(from_keys - to_keys)
        common = from_keys & to_keys

        modified = {}
        unchanged = []

        for field_name in common:
            from_field = from_fields[field_name]
            to_field = to_fields[field_name]

            changes: dict[str, Any] = {}

            if from_field.annotation != to_field.annotation:
                changes["type_changed"] = {
                    "from": from_field.annotation,
                    "to": to_field.annotation,
                }

            from_required = from_field.is_required()
            to_required = to_field.is_required()
            if from_required != to_required:
                changes["required_changed"] = {
                    "from": from_required,
                    "to": to_required,
                }

            from_default = from_field.default
            to_default = to_field.default

            if from_default != to_default and not (
                from_default is PydanticUndefined and to_default is PydanticUndefined
            ):
                if (
                    from_default is not PydanticUndefined
                    and to_default is not PydanticUndefined
                ):
                    changes["default_changed"] = {
                        "from": from_default,
                        "to": to_default,
                    }
                elif from_default is PydanticUndefined:
                    changes["default_added"] = to_default
                else:
                    changes["default_removed"] = from_default

            if changes:
                modified[field_name] = changes
            else:
                unchanged.append(field_name)

        added_field_info = {}
        for field_name in added:
            to_field = to_fields[field_name]
            added_field_info[field_name] = {
                "type": to_field.annotation,
                "required": to_field.is_required(),
                "default": to_field.default
                if to_field.default is not PydanticUndefined
                else None,
            }

        return cls(
            model_name=name,
            from_version=from_version,
            to_version=to_version,
            added_fields=added,
            removed_fields=removed,
            modified_fields=modified,
            unchanged_fields=unchanged,
            added_field_info=added_field_info,
        )
