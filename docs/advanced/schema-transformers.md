# Schema Transformers

Schema transformers are simple functions that modify schemas after generation.
This guide covers transformer patterns, composition strategies, and best
practices for post-processing schemas.

## What are Schema Transformers?

Transformers are functions that take a JSON schema and return a modified
version. They run **after** Pydantic generates the base schema, making them
perfect for simple, model-specific customizations:

```python
from pydantic import BaseModel
from pyrmute import ModelManager, JsonSchema

manager = ModelManager()


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    email: str


@manager.schema_transformer("User", "1.0.0")
def add_examples(schema: JsonSchema) -> JsonSchema:
    """Add example data to User schema."""
    schema["examples"] = [
        {"name": "Alice Smith", "email": "alice@example.com"},
        {"name": "Bob Jones", "email": "bob@example.com"}
    ]
    return schema

# Transformer runs automatically
schema = manager.get_schema("User", "1.0.0")
print(schema["examples"])
# [{"name": "Alice Smith", "email": "alice@example.com"}, ...]
```

**Key characteristics:**

- Simple functions: `(JsonSchema) -> JsonSchema`
- Run after schema generation
- Model-specific customization
- Easy to compose multiple transformers
- No need to subclass anything

## Basic Transformers

### Adding Metadata

Add custom fields to schemas:

```python
@manager.schema_transformer("User", "1.0.0")
def add_metadata(schema: JsonSchema) -> JsonSchema:
    """Add custom metadata fields."""
    schema["x-version"] = "1.0.0"
    schema["x-deprecated"] = False
    schema["x-owner"] = "user-team"
    schema["x-last-updated"] = "2024-01-15"
    return schema
```

### Adding Examples

Provide example data for documentation:

```python
@manager.schema_transformer("Product", "1.0.0")
def add_product_examples(schema: JsonSchema) -> JsonSchema:
    """Add realistic product examples."""
    schema["examples"] = [
        {
            "id": 1,
            "name": "Widget",
            "price": 9.99,
            "in_stock": True
        },
        {
            "id": 2,
            "name": "Gadget",
            "price": 19.99,
            "in_stock": False
        }
    ]
    return schema
```

### Adding Descriptions

Enhance field descriptions:

```python
@manager.schema_transformer("Order", "1.0.0")
def enhance_descriptions(schema: JsonSchema) -> JsonSchema:
    """Add detailed field descriptions."""
    properties = schema.get("properties", {})

    if "order_id" in properties:
        properties["order_id"]["description"] = (
            "Unique identifier for the order. "
            "Format: ORD-{timestamp}-{random}"
        )

    if "status" in properties:
        properties["status"]["description"] = (
            "Current order status. "
            "Possible values: pending, processing, shipped, delivered, cancelled"
        )

    return schema
```

### Adding Validation Rules

Add additional constraints:

```python
@manager.schema_transformer("Email", "1.0.0")
def add_email_validation(schema: JsonSchema) -> JsonSchema:
    """Add email validation pattern."""
    properties = schema.get("properties", {})

    for field_name, field_schema in properties.items():
        if "email" in field_name.lower():
            field_schema["format"] = "email"
            field_schema["pattern"] = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    return schema
```

## Multiple Transformers

Register multiple transformers for the same model:

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    email: str
    age: int


@manager.schema_transformer("User", "1.0.0")
def add_examples(schema: JsonSchema) -> JsonSchema:
    """First transformer: add examples."""
    schema["examples"] = [
        {"name": "Alice", "email": "alice@example.com", "age": 30}
    ]
    return schema


@manager.schema_transformer("User", "1.0.0")
def add_metadata(schema: JsonSchema) -> JsonSchema:
    """Second transformer: add metadata."""
    schema["x-requires-auth"] = True
    schema["x-rate-limit"] = 100
    return schema


@manager.schema_transformer("User", "1.0.0")
def add_constraints(schema: JsonSchema) -> JsonSchema:
    """Third transformer: add constraints."""
    schema["additionalProperties"] = False
    return schema


# All three transformers run in registration order
schema = manager.get_schema("User", "1.0.0")
# Has examples, metadata, and constraints
```

**Execution order:** Transformers run in the order they were registered.

## Advanced Patterns

### Conditional Transformations

Apply transformations based on schema content:

```python
@manager.schema_transformer("Payment", "1.0.0")
def add_security_requirements(schema: JsonSchema) -> JsonSchema:
    """Add security metadata for sensitive fields."""
    properties = schema.get("properties", {})
    sensitive_fields = []

    # Identify sensitive fields
    for field_name in properties.keys():
        if any(keyword in field_name.lower()
               for keyword in ["password", "token", "secret", "card", "ssn"]):
            sensitive_fields.append(field_name)

    # Add security metadata if sensitive fields found
    if sensitive_fields:
        schema["x-sensitive-fields"] = sensitive_fields
        schema["x-requires-encryption"] = True
        schema["x-pii"] = True

    return schema
```

### Recursive Field Processing

Process all fields recursively:

```python
@manager.schema_transformer("Document", "1.0.0")
def add_field_metadata(schema: JsonSchema) -> JsonSchema:
    """Recursively add metadata to all fields."""

    def process_schema(s: dict, path: str = "") -> None:
        """Recursively process schema at all levels."""
        if "properties" in s:
            for field_name, field_schema in s["properties"].items():
                field_path = f"{path}.{field_name}" if path else field_name

                # Add field path
                field_schema["x-path"] = field_path

                # Add field type category
                field_type = field_schema.get("type")
                if field_type in ["string", "integer", "number", "boolean"]:
                    field_schema["x-category"] = "primitive"
                elif field_type == "array":
                    field_schema["x-category"] = "collection"
                elif field_type == "object":
                    field_schema["x-category"] = "nested"

                # Recursively process nested objects
                if field_type == "object":
                    process_schema(field_schema, field_path)

        # Process definitions
        if "$defs" in s:
            for def_name, def_schema in s["$defs"].items():
                process_schema(def_schema, f"$defs.{def_name}")

    process_schema(schema)
    return schema
```

### Schema Validation

Validate and fix schema issues:

```python
@manager.schema_transformer("Config", "1.0.0")
def validate_and_fix_schema(schema: JsonSchema) -> JsonSchema:
    """Ensure schema has required fields and fix common issues."""

    # Ensure title exists
    if "title" not in schema:
        schema["title"] = "Config"

    # Ensure description exists
    if "description" not in schema:
        schema["description"] = f"Schema for {schema['title']}"

    # Fix missing descriptions in properties
    if "properties" in schema:
        for field_name, field_schema in schema["properties"].items():
            if "description" not in field_schema:
                field_schema["description"] = (
                    field_name.replace("_", " ").title()
                )

    # Ensure required fields is a list
    if "required" in schema and not isinstance(schema["required"], list):
        schema["required"] = list(schema["required"])

    return schema
```

### Merging with External Metadata

Combine schema with external metadata:

```python
# External metadata (could come from config file)
FIELD_METADATA = {
    "User": {
        "1.0.0": {
            "name": {
                "ui_widget": "text_input",
                "placeholder": "Enter your full name",
                "help_text": "First and last name"
            },
            "email": {
                "ui_widget": "email_input",
                "placeholder": "user@example.com",
                "help_text": "We'll never share your email"
            }
        }
    }
}


@manager.schema_transformer("User", "1.0.0")
def add_ui_metadata(schema: JsonSchema) -> JsonSchema:
    """Merge external UI metadata into schema."""
    metadata = FIELD_METADATA.get("User", {}).get("1.0.0", {})
    properties = schema.get("properties", {})

    for field_name, field_schema in properties.items():
        if field_name in metadata:
            field_meta = metadata[field_name]
            field_schema["x-ui"] = {
                "widget": field_meta.get("ui_widget", "text_input"),
                "placeholder": field_meta.get("placeholder", ""),
                "helpText": field_meta.get("help_text", "")
            }

    return schema
```

### OpenAPI Enhancement

Add OpenAPI-specific fields:

```python
@manager.schema_transformer("User", "1.0.0")
def add_openapi_metadata(schema: JsonSchema) -> JsonSchema:
    """Add OpenAPI-specific metadata."""
    # Add tags
    schema["x-tags"] = ["users", "authentication"]

    # Add response codes
    schema["x-responses"] = {
        "200": "Successful operation",
        "400": "Invalid input",
        "404": "User not found"
    }

    # Add security requirements
    schema["x-security"] = [{"bearerAuth": []}]

    # Add operation IDs
    schema["x-operation-id"] = "getUser"

    return schema
```

## Reusable Transformers

Create generic transformers that work with any model:

```python
def add_timestamp_metadata(
    timestamp_field: str = "x-generated-at"
) -> Callable[[JsonSchema], JsonSchema]:
    """Factory function for timestamp transformers."""
    from datetime import datetime

    def transformer(schema: JsonSchema) -> JsonSchema:
        schema[timestamp_field] = datetime.utcnow().isoformat()
        return schema

    return transformer


def add_version_metadata(version: str) -> Callable[[JsonSchema], JsonSchema]:
    """Factory function for version transformers."""
    def transformer(schema: JsonSchema) -> JsonSchema:
        schema["x-version"] = version
        schema["x-api-version"] = f"v{version.split('.')[0]}"
        return schema

    return transformer


def add_deprecation_notice(
    deprecated_in: str, removed_in: str, replacement: str
) -> Callable[[JsonSchema], JsonSchema]:
    """Factory function for deprecation transformers."""
    def transformer(schema: JsonSchema) -> JsonSchema:
        schema["deprecated"] = True
        schema["x-deprecation"] = {
            "deprecated_in": deprecated_in,
            "removed_in": removed_in,
            "replacement": replacement,
            "message": f"Deprecated in {deprecated_in}. Use {replacement} instead."
        }
        return schema

    return transformer


# Use reusable transformers
@manager.schema_transformer("User", "1.0.0")
def _(schema: JsonSchema) -> JsonSchema:
    return add_timestamp_metadata()(schema)


@manager.schema_transformer("User", "1.0.0")
def _(schema: JsonSchema) -> JsonSchema:
    return add_version_metadata("1.0.0")(schema)


# Or apply directly
transformer = add_deprecation_notice("2.0.0", "3.0.0", "UserV2")
manager.schema_transformer("OldUser", "1.0.0")(transformer)
```

## Transformer Composition

Compose multiple transformers:

```python
from typing import Callable


def compose_transformers(
    *transformers: Callable[[JsonSchema], JsonSchema]
) -> Callable[[JsonSchema], JsonSchema]:
    """Compose multiple transformers into one."""
    def composed(schema: JsonSchema) -> JsonSchema:
        result = schema
        for transformer in transformers:
            result = transformer(result)
        return result
    return composed


# Individual transformers
def add_examples(schema: JsonSchema) -> JsonSchema:
    schema["examples"] = [{"name": "Alice"}]
    return schema


def add_metadata(schema: JsonSchema) -> JsonSchema:
    schema["x-version"] = "1.0.0"
    return schema


def add_security(schema: JsonSchema) -> JsonSchema:
    schema["x-requires-auth"] = True
    return schema


# Compose them
combined = compose_transformers(
    add_examples,
    add_metadata,
    add_security
)


# Register composed transformer
@manager.schema_transformer("User", "1.0.0")
def _(schema: JsonSchema) -> JsonSchema:
    return combined(schema)
```

## Conditional Application

Apply transformers only in certain conditions:

```python
import os


def when_environment(env: str) -> Callable[[JsonSchema], JsonSchema]:
    """Decorator to apply transformer only in specific environment."""
    def decorator(func) -> Callable[[JsonSchema], JsonSchema]:
        def wrapper(schema: JsonSchema) -> JsonSchema:
            current_env = os.getenv("ENVIRONMENT", "production")
            if current_env == env:
                return func(schema)
            return schema
        return wrapper
    return decorator


def when_field_exists(field_name: str) -> Callable[[JsonSchema], JsonSchema]:
    """Decorator to apply transformer only if field exists."""
    def decorator(func) -> Callable[[JsonSchema], JsonSchema]:
        def wrapper(schema: JsonSchema) -> JsonSchema:
            if field_name in schema.get("properties", {}):
                return func(schema)
            return schema
        return wrapper
    return decorator


# Use conditional decorators
@manager.schema_transformer("User", "1.0.0")
@when_environment("development")
def add_debug_info(schema: JsonSchema) -> JsonSchema:
    """Only adds debug info in development."""
    schema["x-debug"] = True
    schema["x-dev-notes"] = "This is a development build"
    return schema


@manager.schema_transformer("User", "1.0.0")
@when_field_exists("email")
def add_email_validation(schema: JsonSchema) -> JsonSchema:
    """Only adds validation if email field exists."""
    schema["properties"]["email"]["format"] = "email"
    return schema
```

## Testing Transformers

Test transformers in isolation:

```python
def test_add_examples_transformer() -> None:
    """Test that examples are added correctly."""
    # Create a simple schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        }
    }

    # Apply transformer
    result = add_examples(schema)

    # Verify examples added
    assert "examples" in result
    assert len(result["examples"]) > 0
    assert "name" in result["examples"][0]


def test_transformer_doesnt_mutate_input() -> None:
    """Test that transformer doesn't mutate input schema."""
    original = {
        "type": "object",
        "properties": {"name": {"type": "string"}}
    }

    # Make a copy to compare
    import copy
    original_copy = copy.deepcopy(original)

    # Apply transformer
    result = add_examples(original)

    # Original should be unchanged
    assert original == original_copy

    # Result should be different
    assert result != original
    assert "examples" in result


def test_transformer_with_manager() -> None:
    """Test transformer integration with ModelManager."""
    manager = ModelManager()

    @manager.model("Test", "1.0.0")
    class TestV1(BaseModel):
        value: str

    @manager.schema_transformer("Test", "1.0.0")
    def add_test_metadata(schema: JsonSchema) -> JsonSchema:
        schema["x-test"] = True
        return schema

    schema = manager.get_schema("Test", "1.0.0")

    assert "x-test" in schema
    assert schema["x-test"] is True
```

## Transformers vs Custom Generators

### When to Use Transformers

✅ **Use transformers when:**

- Adding metadata or examples
- Model-specific customizations
- Simple field modifications
- You don't need to change core generation logic
- You want composable, reusable functions
- Changes are independent of Pydantic's generation

**Example scenarios:**

- Adding x-* extension fields
- Adding examples
- Enhancing descriptions
- Adding UI hints
- Environment-specific modifications

### When to Use Custom Generators

✅ **Use custom generators when:**

- Need to change how Pydantic generates schemas
- Global changes affecting all schemas
- Deep integration with schema generation
- Need access to Pydantic's internal schema structure
- Changing core schema structure or dialect

**Example scenarios:**

- Using different JSON Schema dialect
- Changing how $defs are structured
- Global metadata for all schemas
- Deep format changes
- Integration with schema generation process

### Combining Both

Use generators for global changes, transformers for model-specific:

```python
from pydantic.json_schema import GenerateJsonSchema


class GlobalGenerator(GenerateJsonSchema):
    """Global changes for all schemas."""
    def generate(
        self, schema: Mapping[str, Any], mode: JsonSchemaMode = "validation"
    ) -> JsonSchema:
        json_schema = super().generate(schema, mode=mode)
        json_schema["$schema"] = self.schema_dialect
        json_schema["x-generated-by"] = "pyrmute"
        return json_schema

manager = ModelManager(
    default_schema_config=SchemaConfig(schema_generator=GlobalGenerator)
)


# Model-specific transformers
@manager.schema_transformer("User", "1.0.0")
def add_user_metadata(schema: JsonSchema) -> JsonSchema:
    """User-specific metadata."""
    schema["x-requires-auth"] = True
    return schema


@manager.schema_transformer("Product", "1.0.0")
def add_product_metadata(schema: JsonSchema) -> JsonSchema:
    """Product-specific metadata."""
    schema["x-cacheable"] = True
    return schema
```

## Common Patterns Library

### Add Field Constraints

```python
def add_string_constraints(
    min_length: int = 1,
    max_length: int = 1000,
    pattern: str | None = None
) -> Callable[[JsonSchema], JsonSchema]:
    """Add constraints to all string fields."""
    def transformer(schema: JsonSchema) -> JsonSchema:
        if "properties" in schema:
            for field_schema in schema["properties"].values():
                if field_schema.get("type") == "string":
                    if "minLength" not in field_schema:
                        field_schema["minLength"] = min_length
                    if "maxLength" not in field_schema:
                        field_schema["maxLength"] = max_length
                    if pattern and "pattern" not in field_schema:
                        field_schema["pattern"] = pattern
        return schema
    return transformer
```

### Add Documentation Links

```python
def add_documentation_links(base_url: str) -> Callable[[JsonSchema], JsonSchema]:
    """Add documentation links to schemas."""
    def transformer(schema: JsonSchema) -> JsonSchema:
        title = schema.get("title", "Unknown")
        schema["x-docs"] = f"{base_url}/models/{title.lower()}"

        if "properties" in schema:
            for field_name in schema["properties"]:
                schema["properties"][field_name]["x-docs"] = (
                    f"{base_url}/models/{title.lower()}#{field_name}"
                )

        return schema
    return transformer
```

### Add Localization Support

```python
def add_i18n_support(
    translations: dict[str, Any]
) -> Callable[[JsonSchema], JsonSchema]:
    """Add internationalization metadata."""
    def transformer(schema: JsonSchema) -> JsonSchema:
        model_name = schema.get("title", "")

        if model_name in translations:
            schema["x-i18n"] = translations[model_name]

        if "properties" in schema:
            for field_name, field_schema in schema["properties"].items():
                if field_name in translations.get(model_name, {}):
                    field_schema["x-i18n"] = translations[model_name][field_name]

        return schema
    return transformer

# Usage
translations = {
    "User": {
        "en": {"title": "User", "description": "User account"},
        "es": {"title": "Usuario", "description": "Cuenta de usuario"},
        "name": {
            "en": {"label": "Name", "placeholder": "Enter name"},
            "es": {"label": "Nombre", "placeholder": "Ingrese nombre"}
        }
    }
}

@manager.schema_transformer("User", "1.0.0")
def _(schema: JsonSchema) -> JsonSchema:
    return add_i18n_support(translations)(schema)
```

## Best Practices

1. **Don't mutate input** - Create modified copies, don't alter input schema
2. **Keep transformers simple** - One responsibility per transformer
3. **Make them reusable** - Use factory functions for flexibility
4. **Test independently** - Each transformer should be testable in isolation
5. **Document what changes** - Clear docstrings explaining modifications
6. **Consider order** - Transformers run in registration order
7. **Use for model-specific changes** - Global changes belong in custom generators

## Common Pitfalls

### Mutating Input Schema

```python
# ❌ BAD - Mutates input
def bad_transformer(schema: JsonSchema) -> JsonSchema:
    schema["x-custom"] = "value"  # Mutates input!
    return schema


# ✅ GOOD - Creates new dict
def good_transformer(schema: JsonSchema) -> JsonSchema:
    return {**schema, "x-custom": "value"}


# ✅ ALSO GOOD - Modifies safely
def also_good_transformer(schema: JsonSchema) -> JsonSchema:
    schema = schema.copy()  # or import copy; copy.deepcopy(schema)
    schema["x-custom"] = "value"
    return schema
```

### Assuming Schema Structure

```python
# ❌ BAD - Assumes properties exists
def bad_transformer(schema: JsonSchema) -> JsonSchema:
    schema["properties"]["name"]["x-label"] = "Name"  # KeyError if missing!
    return schema


# ✅ GOOD - Checks first
def good_transformer(schema: JsonSchema) -> JsonSchema:
    if "properties" in schema and "name" in schema["properties"]:
        schema["properties"]["name"]["x-label"] = "Name"
    return schema
```

### Overwriting Existing Fields

```python
# ❌ BAD - Overwrites examples
def bad_transformer(schema: JsonSchema) -> JsonSchema:
    schema["examples"] = [{"new": "example"}]  # Loses existing!
    return schema


# ✅ GOOD - Preserves existing
def good_transformer(schema: JsonSchema) -> JsonSchema:
    existing_examples = schema.get("examples", [])
    new_example = {"new": "example"}
    schema["examples"] = existing_examples + [new_example]
    return schema
```

## Next Steps

Now that you understand schema transformers:

**Related topics:**

- [Custom Schema Generators](custom-generators.md) - For deeper customization
- [Schema Generation](../user-guide/schema-generation.md) - Using transformers

**API Reference:**

- [SchemaConfig API](../reference/schema-config.md) - Complete SchemaConfig
    details
