# Custom Schema Generators

Custom schema generators give you complete control over JSON schema generation. This guide covers creating custom generators, advanced patterns, and integration with schema generation tools.

## Understanding Schema Generators

Pydantic's `GenerateJsonSchema` class controls how models are converted to JSON schemas. By subclassing it, you can customize every aspect of schema generation:

```python
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode
from typing import Any
from pyrmute import ModelManager, SchemaConfig

manager = ModelManager()

class CustomGenerator(GenerateJsonSchema):
    """Custom schema generator with additional metadata."""

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        """Generate schema with custom modifications.

        Args:
            schema: Core schema dict from Pydantic
            mode: 'validation' or 'serialization'

        Returns:
            Modified JSON schema
        """
        # Call parent to get base schema
        json_schema = super().generate(schema, mode=mode)

        # Add custom metadata
        json_schema["x-custom"] = "metadata"
        json_schema["$schema"] = self.schema_dialect

        return json_schema

# Use custom generator
config = SchemaConfig(schema_generator=CustomGenerator)
manager = ModelManager(default_schema_config=config)

@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    email: str

schema = manager.get_schema("User", "1.0.0")
print(schema["x-custom"])  # "metadata"
```

## Basic Customizations

### Adding Global Metadata

Add metadata to all generated schemas:

```python
class MetadataGenerator(GenerateJsonSchema):
    """Add organization metadata to all schemas."""

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Add company information
        json_schema["x-organization"] = "Acme Corp"
        json_schema["x-contact"] = "api@acme.com"
        json_schema["x-generated-by"] = "pyrmute"

        # Add JSON Schema version
        json_schema["$schema"] = self.schema_dialect

        return json_schema
```

### Custom Schema Dialect

Specify a different JSON Schema version:

```python
class Draft2020Generator(GenerateJsonSchema):
    """Use JSON Schema Draft 2020-12."""

    # Override class attribute
    schema_dialect = "https://json-schema.org/draft/2020-12/schema"

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)
        json_schema["$schema"] = self.schema_dialect
        return json_schema
```

### Adding Examples

Automatically add examples to all schemas:

```python
class ExampleGenerator(GenerateJsonSchema):
    """Add example data to schemas."""

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Add examples if not already present
        if "examples" not in json_schema:
            json_schema["examples"] = self._generate_examples(json_schema)

        return json_schema

    def _generate_examples(self, schema: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate example data based on schema."""
        examples = []
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if properties:
            example = {}
            for prop_name, prop_schema in properties.items():
                if prop_name in required:
                    # Generate example value based on type
                    example[prop_name] = self._example_value(prop_schema)

            if example:
                examples.append(example)

        return examples

    def _example_value(self, prop_schema: dict[str, Any]) -> Any:
        """Generate example value for a property."""
        prop_type = prop_schema.get("type")

        if prop_type == "string":
            return prop_schema.get("default", "example")
        elif prop_type == "integer":
            return prop_schema.get("default", 42)
        elif prop_type == "number":
            return prop_schema.get("default", 3.14)
        elif prop_type == "boolean":
            return prop_schema.get("default", True)
        elif prop_type == "array":
            return []
        elif prop_type == "object":
            return {}
        else:
            return None
```

## Advanced Patterns

### OpenAPI Extensions

Add OpenAPI-specific extensions:

```python
class OpenAPIGenerator(GenerateJsonSchema):
    """Generate schemas with OpenAPI extensions."""

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Add OpenAPI extensions
        json_schema["x-tags"] = self._extract_tags(schema)
        json_schema["x-security"] = self._determine_security(schema)

        # Add response codes for serialization mode
        if mode == "serialization":
            json_schema["x-response-code"] = 200

        return json_schema

    def _extract_tags(self, schema: dict[str, Any]) -> list[str]:
        """Extract tags from model docstring or metadata."""
        # Extract from title or add default
        title = schema.get("title", "")
        return [title.lower()] if title else ["general"]

    def _determine_security(self, schema: dict[str, Any]) -> list[dict[str, Any]]:
        """Determine security requirements."""
        # Check if schema has sensitive fields
        properties = schema.get("properties", {})
        has_auth = any(
            "password" in name or "token" in name or "secret" in name
            for name in properties.keys()
        )

        if has_auth:
            return [{"bearerAuth": []}]
        return []
```

### Field-Level Customization

Modify specific field schemas:

```python
class FieldCustomizer(GenerateJsonSchema):
    """Customize specific field types."""

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Process all properties
        if "properties" in json_schema:
            for field_name, field_schema in json_schema["properties"].items():
                self._customize_field(field_name, field_schema)

        return json_schema

    def _customize_field(self, name: str, field_schema: dict[str, Any]) -> None:
        """Customize individual field schema."""
        # Add format hints for common field names
        if "email" in name.lower():
            field_schema["format"] = "email"
            field_schema["x-validation"] = "email"

        elif "url" in name.lower() or "link" in name.lower():
            field_schema["format"] = "uri"
            field_schema["x-validation"] = "url"

        elif "phone" in name.lower():
            field_schema["format"] = "phone"
            field_schema["pattern"] = r"^\+?[1-9]\d{1,14}$"

        elif "date" in name.lower() and field_schema.get("type") == "string":
            field_schema["format"] = "date"

        # Add UI hints
        if "password" in name.lower():
            field_schema["x-ui"] = {"widget": "password"}

        elif "description" in name.lower() or "bio" in name.lower():
            field_schema["x-ui"] = {"widget": "textarea"}
```

### Conditional Schema Generation

Generate different schemas based on conditions:

```python
import os

class ConditionalGenerator(GenerateJsonSchema):
    """Generate schemas with environment-specific features."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment = os.getenv("ENVIRONMENT", "production")
        self.include_debug_info = self.environment == "development"

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Add debug information in development
        if self.include_debug_info:
            json_schema["x-debug"] = {
                "environment": self.environment,
                "mode": mode,
                "generated_at": self._get_timestamp(),
            }

        # Add stricter validation in production
        if self.environment == "production":
            self._add_strict_validation(json_schema)

        return json_schema

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

    def _add_strict_validation(self, schema: dict[str, Any]) -> None:
        """Add stricter validation rules for production."""
        # Disable additional properties
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        # Make all string fields non-empty
        if "properties" in schema:
            for field_schema in schema["properties"].values():
                if field_schema.get("type") == "string":
                    field_schema["minLength"] = 1
```

### Version-Aware Schema Generation

Include version information in schemas:

```python
class VersionedGenerator(GenerateJsonSchema):
    """Add version information to schemas."""

    def __init__(self, model_name: str = None, version: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.version = version

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Add version metadata
        if self.version:
            json_schema["x-version"] = self.version
            json_schema["x-model-name"] = self.model_name

        # Add $id with version
        if self.model_name and self.version:
            json_schema["$id"] = f"https://api.example.com/schemas/{self.model_name}/v{self.version}"

        return json_schema

# Usage with pyrmute
class VersionedSchemaConfig(SchemaConfig):
    """SchemaConfig that passes version info to generator."""

    def __init__(self, model_name: str, version: str, **kwargs):
        self.model_name = model_name
        self.version = version
        super().__init__(**kwargs)

    def to_kwargs(self) -> dict[str, Any]:
        kwargs = super().to_kwargs()
        # Create generator instance with version info
        if self.schema_generator is VersionedGenerator:
            kwargs["schema_generator"] = VersionedGenerator(
                model_name=self.model_name,
                version=self.version
            )
        return kwargs
```

## Handling Definitions

### Custom Definition Processing

Control how `$defs` are generated:

```python
class DefinitionCustomizer(GenerateJsonSchema):
    """Customize how definitions are structured."""

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Process definitions
        if "$defs" in json_schema:
            self._customize_definitions(json_schema["$defs"])

        return json_schema

    def _customize_definitions(self, defs: dict[str, Any]) -> None:
        """Customize each definition."""
        for def_name, def_schema in defs.items():
            # Add metadata to each definition
            def_schema["x-definition-name"] = def_name

            # Add examples if not present
            if "examples" not in def_schema and "properties" in def_schema:
                def_schema["examples"] = [
                    self._generate_example(def_schema)
                ]

    def _generate_example(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Generate example from schema."""
        example = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            prop_type = prop_schema.get("type")
            if prop_type == "string":
                example[prop_name] = "example"
            elif prop_type == "integer":
                example[prop_name] = 0
            elif prop_type == "boolean":
                example[prop_name] = True
        return example
```

### Flattening Definitions

Inline all definitions instead of using `$ref`:

```python
class FlatteningGenerator(GenerateJsonSchema):
    """Flatten all $ref into inline definitions."""

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Get definitions
        defs = json_schema.pop("$defs", {})

        # Replace all $ref with actual definitions
        if defs:
            self._inline_refs(json_schema, defs)

        return json_schema

    def _inline_refs(self, schema: dict[str, Any], defs: dict[str, Any]) -> None:
        """Recursively replace $ref with inline definitions."""
        if isinstance(schema, dict):
            if "$ref" in schema:
                # Extract definition name
                ref = schema["$ref"]
                if ref.startswith("#/$defs/"):
                    def_name = ref.split("/")[-1]
                    if def_name in defs:
                        # Replace $ref with inline definition
                        definition = defs[def_name].copy()
                        schema.clear()
                        schema.update(definition)

            # Recursively process nested schemas
            for value in schema.values():
                if isinstance(value, dict):
                    self._inline_refs(value, defs)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._inline_refs(item, defs)
```

## Integration Patterns

### Combining with Schema Transformers

Use generators for broad changes, transformers for model-specific:

```python
class BaseGenerator(GenerateJsonSchema):
    """Base generator for all schemas."""

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Global changes
        json_schema["$schema"] = self.schema_dialect
        json_schema["x-generated-by"] = "pyrmute"

        return json_schema

# Use with transformers for specific models
config = SchemaConfig(schema_generator=BaseGenerator)
manager = ModelManager(default_schema_config=config)

@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str

@manager.schema_transformer("User", "1.0.0")
def add_user_specific_metadata(schema):
    """Model-specific customization."""
    schema["x-requires-auth"] = True
    schema["x-rate-limit"] = 100
    return schema
```

### Per-Model Generators

Different generators for different models:

```python
class PublicAPIGenerator(GenerateJsonSchema):
    """Generator for public API schemas."""

    def generate(self, schema, mode="validation"):
        json_schema = super().generate(schema, mode=mode)
        json_schema["x-visibility"] = "public"
        json_schema["x-rate-limited"] = True
        return json_schema

class InternalAPIGenerator(GenerateJsonSchema):
    """Generator for internal API schemas."""

    def generate(self, schema, mode="validation"):
        json_schema = super().generate(schema, mode=mode)
        json_schema["x-visibility"] = "internal"
        json_schema["x-rate-limited"] = False
        return json_schema

# Use different generators per call
public_schema = manager.get_schema(
    "User",
    "1.0.0",
    config=SchemaConfig(schema_generator=PublicAPIGenerator)
)

internal_schema = manager.get_schema(
    "User",
    "1.0.0",
    config=SchemaConfig(schema_generator=InternalAPIGenerator)
)
```

## Real-World Examples

### Swagger/OpenAPI Generator

Complete OpenAPI-compatible schema generator:

```python
class SwaggerGenerator(GenerateJsonSchema):
    """Generate OpenAPI 3.0 compatible schemas."""

    schema_dialect = "http://json-schema.org/draft-07/schema#"

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Convert Pydantic schema to OpenAPI format
        self._convert_to_openapi(json_schema)

        return json_schema

    def _convert_to_openapi(self, schema: dict[str, Any]) -> None:
        """Convert schema to OpenAPI 3.0 format."""
        # Remove fields not supported in OpenAPI
        schema.pop("$schema", None)

        # Convert $defs to components/schemas format
        if "$defs" in schema:
            defs = schema.pop("$defs")
            schema["components"] = {"schemas": defs}

        # Update $ref format
        self._update_refs(schema)

        # Add OpenAPI-specific fields
        if "title" in schema:
            schema["x-swagger-router-model"] = schema["title"]

    def _update_refs(self, schema: dict[str, Any]) -> None:
        """Update $ref format for OpenAPI."""
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref = schema["$ref"]
                # Convert #/$defs/Model to #/components/schemas/Model
                if ref.startswith("#/$defs/"):
                    schema["$ref"] = ref.replace(
                        "#/$defs/",
                        "#/components/schemas/"
                    )

            for value in schema.values():
                self._update_refs(value)
        elif isinstance(schema, list):
            for item in schema:
                self._update_refs(item)
```

### Documentation Generator

Generate schemas optimized for documentation:

```python
class DocumentationGenerator(GenerateJsonSchema):
    """Generate schemas optimized for documentation."""

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Enhance for documentation
        self._add_human_readable_descriptions(json_schema)
        self._add_examples(json_schema)
        self._add_constraints_descriptions(json_schema)

        return json_schema

    def _add_human_readable_descriptions(self, schema: dict[str, Any]) -> None:
        """Add readable descriptions based on field names."""
        if "properties" in schema:
            for field_name, field_schema in schema["properties"].items():
                if "description" not in field_schema:
                    # Generate description from field name
                    field_schema["description"] = (
                        field_name.replace("_", " ").title()
                    )

    def _add_examples(self, schema: dict[str, Any]) -> None:
        """Add realistic examples."""
        if "properties" in schema and "examples" not in schema:
            schema["examples"] = [self._generate_realistic_example(schema)]

    def _generate_realistic_example(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Generate realistic example data."""
        example = {}
        properties = schema.get("properties", {})

        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type")

            # Generate realistic values based on field name
            if "email" in field_name.lower():
                example[field_name] = "user@example.com"
            elif "name" in field_name.lower():
                example[field_name] = "John Doe"
            elif "age" in field_name.lower():
                example[field_name] = 30
            elif field_type == "string":
                example[field_name] = f"Example {field_name}"
            elif field_type == "integer":
                example[field_name] = 42
            elif field_type == "boolean":
                example[field_name] = True

        return example

    def _add_constraints_descriptions(self, schema: dict[str, Any]) -> None:
        """Add descriptions for validation constraints."""
        if "properties" in schema:
            for field_schema in schema["properties"].values():
                constraints = []

                if "minLength" in field_schema:
                    constraints.append(
                        f"minimum length: {field_schema['minLength']}"
                    )
                if "maxLength" in field_schema:
                    constraints.append(
                        f"maximum length: {field_schema['maxLength']}"
                    )
                if "minimum" in field_schema:
                    constraints.append(
                        f"minimum value: {field_schema['minimum']}"
                    )
                if "maximum" in field_schema:
                    constraints.append(
                        f"maximum value: {field_schema['maximum']}"
                    )
                if "pattern" in field_schema:
                    constraints.append(
                        f"pattern: {field_schema['pattern']}"
                    )

                if constraints:
                    constraint_text = ", ".join(constraints)
                    current_desc = field_schema.get("description", "")
                    field_schema["description"] = (
                        f"{current_desc} ({constraint_text})"
                        if current_desc
                        else f"Constraints: {constraint_text}"
                    )
```

## Testing Custom Generators

Always test your custom generators:

```python
def test_custom_generator():
    """Test custom generator produces expected output."""
    from pydantic import BaseModel, Field

    class TestModel(BaseModel):
        name: str = Field(description="User's name")
        age: int = Field(ge=0, le=150)

    # Create generator
    generator = CustomGenerator()

    # Generate schema
    schema = generator.generate_schema(
        TestModel.__pydantic_core_schema__,
        mode="validation"
    )

    # Verify custom fields
    assert "x-custom" in schema
    assert schema["x-custom"] == "metadata"

    # Verify standard fields still present
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]

def test_generator_with_manager():
    """Test generator integration with ModelManager."""
    config = SchemaConfig(schema_generator=CustomGenerator)
    manager = ModelManager(default_schema_config=config)

    @manager.model("Test", "1.0.0")
    class TestV1(BaseModel):
        value: str

    schema = manager.get_schema("Test", "1.0.0")

    assert "x-custom" in schema
    assert schema["properties"]["value"]["type"] == "string"
```

## Best Practices

1. **Call super().generate()** - Always call parent implementation first
2. **Don't mutate input** - Work with the returned schema
3. **Handle all modes** - Test both validation and serialization modes
4. **Document behavior** - Explain what customizations you're adding
5. **Test thoroughly** - Verify generated schemas are valid
6. **Consider performance** - Generators run on every schema generation
7. **Use transformers for simple cases** - Reserve generators for complex logic

## Common Pitfalls

### Not Calling Super

```python
# ❌ BAD - Doesn't generate base schema
class BadGenerator(GenerateJsonSchema):
    def generate(self, schema, mode="validation"):
        return {"x-custom": "only"}  # Missing all Pydantic fields!

# ✅ GOOD - Builds on base schema
class GoodGenerator(GenerateJsonSchema):
    def generate(self, schema, mode="validation"):
        json_schema = super().generate(schema, mode=mode)
        json_schema["x-custom"] = "added"
        return json_schema
```

### Mutating Input

```python
# ❌ BAD - Mutates input schema
class BadGenerator(GenerateJsonSchema):
    def generate(self, schema, mode="validation"):
        schema["x-custom"] = "value"  # Mutates input!
        return schema

# ✅ GOOD - Works with returned schema
class GoodGenerator(GenerateJsonSchema):
    def generate(self, schema, mode="validation"):
        json_schema = super().generate(schema, mode=mode)
        json_schema["x-custom"] = "value"
        return json_schema
```

### Ignoring Mode

```python
# ❌ BAD - Ignores mode parameter
class BadGenerator(GenerateJsonSchema):
    def generate(self, schema, mode="validation"):
        json_schema = super().generate(schema, mode=mode)
        json_schema["x-mode"] = "always-validation"  # Wrong!
        return json_schema

# ✅ GOOD - Respects mode
class GoodGenerator(GenerateJsonSchema):
    def generate(self, schema, mode="validation"):
        json_schema = super().generate(schema, mode=mode)
        json_schema["x-mode"] = mode  # Correct!
        return json_schema
```

## Next Steps

Now that you understand custom schema generators:

**Related topics:**

- [Schema Transformers](schema-transformers.md) - Simpler alternative for
    basic customization
- [Schema Generation](../user-guide/schema-generation.md) - Using custom
    generators

**API Reference:**

- [SchemaConfig API](../reference/schema-config.md) - Complete SchemaConfig
    details
