"""Schema configuration for customized schema generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from pydantic.json_schema import GenerateJsonSchema

if TYPE_CHECKING:
    from .types import JsonSchemaGenerator, JsonSchemaMode


@dataclass
class SchemaConfig:
    """Configuration for JSON schema generation.

    This class provides fine-grained control over how Pydantic generates JSON schemas,
    supporting both callable generators and Pydantic's GenerateJsonSchema classes.

    Attributes:
        schema_generator: Custom schema generator. Can be either:
            - A callable taking (type[BaseModel]) -> JsonSchema
            - A subclass of pydantic.json_schema.GenerateJsonSchema
        mode: Schema generation mode - 'validation' for input validation or
            'serialization' for output serialization.
        by_alias: Whether to use field aliases in the schema.
        ref_template: Template for JSON schema $ref URIs.
        extra_kwargs: Additional arguments to pass to model_json_schema().

    Example (Callable Generator):
        >>> def custom_generator(model: type[BaseModel]) -> JsonSchema:
        ...     schema = model.model_json_schema()
        ...     schema["x-custom"] = "metadata"
        ...     return schema
        >>>
        >>> config = SchemaConfig(
        ...     schema_generator=custom_generator,
        ...     mode="validation"
        ... )

    Example (GenerateJsonSchema Class):
        >>> from pydantic.json_schema import GenerateJsonSchema
        >>>
        >>> class CustomSchemaGenerator(GenerateJsonSchema):
        ...     def generate(
        ...         self,
        ...         schema: Mapping[str, Any],
        ...         mode: JsonSchemaMode = "validation"
        ...     ) -> JsonSchema:
        ...         json_schema = super().generate(schema, mode=mode)
        ...         json_schema["x-custom"] = "metadata"
        ...         return json_schema
        >>>
        >>> config = SchemaConfig(
        ...     schema_generator=CustomSchemaGenerator,
        ...     mode="validation",
        ...     by_alias=True
        ... )
    """

    schema_generator: JsonSchemaGenerator | type[GenerateJsonSchema] | None = None
    mode: JsonSchemaMode = "validation"
    by_alias: bool = True
    ref_template: str = "#/$defs/{model}"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def merge_with(self: Self, other: SchemaConfig | None) -> SchemaConfig:
        """Merge this config with another, with other taking precedence.

        Args:
            other: Configuration to merge with (overrides this config).

        Returns:
            New SchemaConfig with merged values.
        """
        if other is None:
            return self

        return SchemaConfig(
            schema_generator=other.schema_generator or self.schema_generator,
            mode=other.mode if other.mode != "validation" else self.mode,
            by_alias=other.by_alias if not other.by_alias else self.by_alias,
            ref_template=other.ref_template
            if other.ref_template != "#/$defs/{model}"
            else self.ref_template,
            extra_kwargs={**self.extra_kwargs, **other.extra_kwargs},
        )

    def to_kwargs(self: Self) -> dict[str, Any]:
        """Convert config to kwargs for model_json_schema().

        Note: If schema_generator is a callable (JsonSchemaGenerator type), it cannot be
        passed to model_json_schema() and must be handled separately by calling it
        directly.

        Returns:
            Dictionary of arguments for Pydantic's model_json_schema(). If
            schema_generator is a callable, it will NOT be included.
        """
        kwargs = {
            "mode": self.mode,
            "by_alias": self.by_alias,
            "ref_template": self.ref_template,
            **self.extra_kwargs,
        }

        # Only add schema_generator if it's a GenerateJsonSchema class
        # Callable generators are handled separately
        if (
            self.schema_generator is not None
            and isinstance(self.schema_generator, type)
            and issubclass(self.schema_generator, GenerateJsonSchema)
        ):
            kwargs["schema_generator"] = self.schema_generator

        return kwargs

    def is_callable_generator(self: Self) -> bool:
        """Check if schema_generator is a callable function.

        Returns:
            True if schema_generator is a callable (not a class).
        """
        if self.schema_generator is None:
            return False

        return callable(self.schema_generator) and not isinstance(
            self.schema_generator, type
        )
