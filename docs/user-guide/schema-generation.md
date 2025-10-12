# Schema Generation

pyrmute can generate JSON schemas for all your model versions. This guide
covers schema generation, customization, export options, and integration with
OpenAPI and other tools.

## Basic Schema Generation

Generate a JSON schema for any registered model:

```python
from pydantic import BaseModel, Field
from pyrmute import ModelManager

manager = ModelManager()


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    """User model version 1.0.0."""
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    age: int = Field(ge=0, le=150, description="User's age in years")


# Generate schema
schema = manager.get_schema("User", "1.0.0")
print(schema)
```

Output:
```json
{
  "title": "UserV1",
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "User's full name"
    },
    "email": {
      "type": "string",
      "description": "User's email address"
    },
    "age": {
      "type": "integer",
      "minimum": 0,
      "maximum": 150,
      "description": "User's age in years"
    }
  },
  "required": ["name", "email", "age"]
}
```

## Schema Modes

Pydantic supports two schema generation modes:

### Validation Mode (Default)

Generates schema for validating input data:

```python
schema = manager.get_schema("User", "1.0.0", mode="validation")
```

Use for:

- API request validation
- User input validation
- Data import validation

### Serialization Mode

Generates schema for serializing output data:

```python
schema = manager.get_schema("User", "1.0.0", mode="serialization")
```

Use for:

- API response documentation
- Data export formats
- Output specifications

**Key differences:**

- Serialization mode uses `serialization_alias` instead of `alias`
- May include computed fields
- Can have different required fields

Example:

```python
from pydantic import computed_field


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    first_name: str
    last_name: str

    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


# Validation schema - no full_name
validation_schema = manager.get_schema("User", "1.0.0", mode="validation")

# Serialization schema - includes full_name
serialization_schema = manager.get_schema("User", "1.0.0", mode="serialization")
```

## Exporting Schemas

### Export All Versions

Export schemas for all registered models:

```python
manager.dump_schemas("schemas/", indent=2)
```

Creates files like:
```
schemas/
├── User_v1.0.0.json
├── User_v2.0.0.json
├── Order_v1.0.0.json
└── Product_v1.0.0.json
```

### Export with Configuration

Apply custom configuration to all exports:

```python
from pyrmute import SchemaConfig

config = SchemaConfig(
    mode="serialization",
    by_alias=True
)

manager.dump_schemas("schemas/", config=config, indent=2)
```

### Separate Definition Files

Create separate files for nested models with `$ref` references:

```python
@manager.model("Address", "1.0.0", enable_ref=True)
class AddressV1(BaseModel):
    street: str
    city: str


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1


# Export with separate definitions
manager.dump_schemas(
    "schemas/",
    separate_definitions=True,
    ref_template="https://api.example.com/schemas/{model}_v{version}.json"
)
```

Creates:
```
schemas/
├── User_v1.0.0.json      # Contains $ref to Address
└── Address_v1.0.0.json   # Separate address schema
```

User schema:
```json
{
  "title": "UserV1",
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "address": {
      "$ref": "https://api.example.com/schemas/Address_v1.0.0.json"
    }
  }
}
```

## Custom Schema Generation

### Using SchemaConfig

Configure schema generation at the manager level:

```python
from pyrmute import SchemaConfig

config = SchemaConfig(
    mode="validation",
    by_alias=True,
    ref_template="#/$defs/{model}"
)

manager = ModelManager(default_schema_config=config)

# All schemas use this configuration
schema = manager.get_schema("User", "1.0.0")
```

### Per-Call Overrides

Override configuration for specific schema generation:

```python
# Manager has default config
manager = ModelManager(
    default_schema_config=SchemaConfig(mode="validation")
)

# Override for this call
schema = manager.get_schema(
    "User",
    "1.0.0",
    mode="serialization",  # Override mode
    by_alias=False         # Override by_alias
)
```

### Custom Schema Generators

Create custom schema generators for advanced control:

```python
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode
from typing import Any


class CustomSchemaGenerator(GenerateJsonSchema):
    """Custom schema generator with company metadata."""

    def generate(
        self,
        schema: dict[str, Any],
        mode: JsonSchemaMode = "validation"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode=mode)

        # Add custom metadata
        json_schema["x-company"] = "Acme Corp"
        json_schema["x-generated-by"] = "pyrmute"
        json_schema["$schema"] = self.schema_dialect

        return json_schema


# Use custom generator
config = SchemaConfig(schema_generator=CustomSchemaGenerator)
manager = ModelManager(default_schema_config=config)

schema = manager.get_schema("User", "1.0.0")
# Includes x-company and x-generated-by fields
```

### Schema Transformers

Apply post-processing to schemas without custom generators:

```python
from pyrmute import JsonSchema


@manager.schema_transformer("User", "1.0.0")
def add_examples(schema: JsonSchema) -> JsonSchema:
    """Add example data to schema."""
    schema["examples"] = [
        {"name": "Alice Smith", "email": "alice@example.com", "age": 30},
        {"name": "Bob Jones", "email": "bob@example.com", "age": 25}
    ]
    return schema


@manager.schema_transformer("User", "1.0.0")
def add_metadata(schema: JsonSchema) -> JsonSchema:
    """Add custom metadata."""
    schema["x-version"] = "1.0.0"
    schema["x-deprecated"] = False
    return schema


# Both transformers are applied
schema = manager.get_schema("User", "1.0.0")
# Includes examples and metadata
```

**Key points:**

- Transformers run after schema generation
- Multiple transformers can be registered per model
- They run in registration order
- Simpler than custom generators for basic customization

## Working with Nested Models

### Inline Definitions (Default)

Nested models are inlined in the schema:

```python
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str
    city: str


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1


schema = manager.get_schema("User", "1.0.0")
```

Output:
```json
{
  "title": "UserV1",
  "properties": {
    "name": {"type": "string"},
    "address": {
      "title": "AddressV1",
      "type": "object",
      "properties": {
        "street": {"type": "string"},
        "city": {"type": "string"}
      }
    }
  }
}
```

### Using $ref with Definitions

Enable `$ref` for reusable models:

```python
@manager.model("Address", "1.0.0", enable_ref=True)
class AddressV1(BaseModel):
    street: str
    city: str


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1


schema = manager.get_schema("User", "1.0.0")
```

Output:
```json
{
  "title": "UserV1",
  "properties": {
    "name": {"type": "string"},
    "address": {"$ref": "#/$defs/AddressV1"}
  },
  "$defs": {
    "AddressV1": {
      "title": "AddressV1",
      "type": "object",
      "properties": {
        "street": {"type": "string"},
        "city": {"type": "string"}
      }
    }
  }
}
```

## OpenAPI Integration

Generate OpenAPI-compatible schemas:

```python
from typing import List


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    """User account information."""
    id: int = Field(description="Unique user identifier")
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")


@manager.model("UserList", "1.0.0")
class UserListV1(BaseModel):
    """List of users."""
    users: List[UserV1]
    total: int = Field(description="Total number of users")

# Generate schemas for OpenAPI
user_schema = manager.get_schema("User", "1.0.0", mode="serialization")
user_list_schema = manager.get_schema("UserList", "1.0.0", mode="serialization")

# Use in OpenAPI spec
openapi_spec = {
    "openapi": "3.0.0",
    "info": {"title": "My API", "version": "1.0.0"},
    "paths": {
        "/users/{user_id}": {
            "get": {
                "responses": {
                    "200": {
                        "description": "User details",
                        "content": {
                            "application/json": {
                                "schema": user_schema
                            }
                        }
                    }
                }
            }
        },
        "/users": {
            "get": {
                "responses": {
                    "200": {
                        "description": "List of users",
                        "content": {
                            "application/json": {
                                "schema": user_list_schema
                            }
                        }
                    }
                }
            }
        }
    },
    "components": {
        "schemas": {
            "User": user_schema,
            "UserList": user_list_schema
        }
    }
}
```

### Versioned OpenAPI Endpoints

Create versioned API documentation:

```python
def generate_openapi_spec(version: str) -> JsonSchema:
    """Generate OpenAPI spec for a specific API version."""
    user_schema = manager.get_schema("User", version, mode="serialization")

    return {
        "openapi": "3.0.0",
        "info": {
            "title": "My API",
            "version": version
        },
        "paths": {
            "/users/{user_id}": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": user_schema
                                }
                            }
                        }
                    }
                }
            }
        }
    }

# Generate for each version
openapi_v1 = generate_openapi_spec("1.0.0")
openapi_v2 = generate_openapi_spec("2.0.0")
```

## Schema Validation

Use generated schemas to validate data:

```python
import jsonschema

schema = manager.get_schema("User", "1.0.0")

# Valid data
valid_data = {"name": "Alice", "email": "alice@example.com", "age": 30}
jsonschema.validate(valid_data, schema)  # Passes

# Invalid data
invalid_data = {"name": "Bob", "age": "thirty"}  # Missing email, wrong type
try:
    jsonschema.validate(invalid_data, schema)
except jsonschema.ValidationError as e:
    print(f"Validation error: {e.message}")
```

## Common Patterns

### Multiple Schema Formats

Generate schemas in different formats:

```python
# Validation schemas for API requests
manager.dump_schemas(
    "schemas/validation/",
    config=SchemaConfig(mode="validation"),
    indent=2
)

# Serialization schemas for API responses
manager.dump_schemas(
    "schemas/serialization/",
    config=SchemaConfig(mode="serialization"),
    indent=2
)
```

### Schema Documentation

Generate human-readable documentation:

```python
from pydantic import Field


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    """User account for the application.

    Users can authenticate and access protected resources.
    """
    name: str = Field(
        description="Full name of the user",
        examples=["Alice Smith", "Bob Jones"]
    )
    email: str = Field(
        description="Email address for login and notifications",
        pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )
    age: int = Field(
        ge=18,
        le=150,
        description="User's age in years (must be 18 or older)"
    )


# Schema includes all documentation
schema = manager.get_schema("User", "1.0.0")
```

### Schema Registry

Build a schema registry for all versions:

```python
def build_schema_registry() -> dict[str, Any]:
    """Build a registry of all schemas."""
    registry = {}

    for model_name in manager.list_models():
        registry[model_name] = {}

        for version in manager.list_versions(model_name):
            schema = manager.get_schema(model_name, version)
            registry[model_name][str(version)] = schema

    return registry


# Generate registry
registry = build_schema_registry()

# Access any schema
user_v1_schema = registry["User"]["1.0.0"]
user_v2_schema = registry["User"]["2.0.0"]
```

### Schema Diff

Compare schemas across versions:

```python
def compare_schemas(
    model_name: str, from_version: str, to_version: str
) -> None:
    """Compare schemas between versions."""
    from_schema = manager.get_schema(model_name, from_version)
    to_schema = manager.get_schema(model_name, to_version)

    # Get property changes
    from_props = set(from_schema.get("properties", {}).keys())
    to_props = set(to_schema.get("properties", {}).keys())

    added = to_props - from_props
    removed = from_props - to_props

    print(f"Schema changes from {from_version} to {to_version}:")
    print(f"  Added properties: {added}")
    print(f"  Removed properties: {removed}")

    # Or use ModelDiff for more detailed comparison
    diff = manager.diff(model_name, from_version, to_version)
    print(diff.to_markdown())


compare_schemas("User", "1.0.0", "2.0.0")
```

### Conditional Schema Features

Add conditional features to schemas:

```python
@manager.schema_transformer("User", "1.0.0")
def add_conditional_features(schema: JsonSchema) -> JsonSchema:
    """Add conditional validation rules."""

    # Add if-then-else logic
    schema["if"] = {
        "properties": {"age": {"minimum": 18}}
    }
    schema["then"] = {
        "properties": {
            "consent": {"const": True}
        },
        "required": ["consent"]
    }

    return schema
```

## Advanced Customization

### Custom JSON Schema Dialect

Specify a custom JSON Schema dialect:

```python
from pydantic.json_schema import GenerateJsonSchema
from pyrmute import JsonSchemaMode, JsonSchema

class Draft2020SchemaGenerator(GenerateJsonSchema):
    schema_dialect = "https://json-schema.org/draft/2020-12/schema"

    def generate(
        self, schema: Mapping[str, Any], mode: JsonSchemaMode = "validation"
    ) -> JsonSchema:
        json_schema = super().generate(schema, mode=mode)
        json_schema["$schema"] = self.schema_dialect
        return json_schema


config = SchemaConfig(schema_generator=Draft2020SchemaGenerator)
manager = ModelManager(default_schema_config=config)
```

### Adding Schema Extensions

Add custom extensions for specific tools:

```python
@manager.schema_transformer("User", "1.0.0")
def add_swagger_extensions(schema: JsonSchema) -> JsonSchema:
    """Add Swagger/OpenAPI extensions."""
    schema["x-swagger-router-model"] = "User"
    schema["x-tags"] = ["users"]

    # Add discriminator for polymorphism
    if "type" in schema.get("properties", {}):
        schema["discriminator"] = {
            "propertyName": "type",
            "mapping": {
                "admin": "#/components/schemas/AdminUser",
                "regular": "#/components/schemas/RegularUser"
            }
        }

    return schema
```

### Schema Introspection

Get information about nested models:

```python
# Get all nested models used by a model
nested_models = manager.get_nested_models("User", "1.0.0")

for nested in nested_models:
    print(f"  - {nested.name} v{nested.version}")


# Use this to build dependency graphs
def build_dependency_graph() -> dict[str, Any]:
    """Build a dependency graph of models."""
    graph = {}

    for model_name in manager.list_models():
        for version in manager.list_versions(model_name):
            key = f"{model_name}@{version}"
            nested = manager.get_nested_models(model_name, version)
            graph[key] = [f"{n.name}@{n.version}" for n in nested]

    return graph
```

## Best Practices

1. **Use transformers for simple customizations** - Easier than custom generators
2. **Enable `$ref` for shared models** - Reduces duplication
3. **Generate both validation and serialization schemas** - Different use cases
4. **Include examples and descriptions** - Better API documentation
5. **Version your schemas** - Keep schemas for all model versions
6. **Test schema validity** - Validate schemas against JSON Schema spec
7. **Cache generated schemas** - Avoid regenerating repeatedly

## Troubleshooting

### Schema Missing Fields

If fields are missing from generated schemas:

```python
# Check field visibility
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    _internal: str  # Private field (excluded from schema)

    model_config = {"exclude": ["_internal"]}
```

### $ref Not Working

Ensure model has `enable_ref=True`:

```python
# Wrong - ref won't work
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str


# Right - enables $ref
@manager.model("Address", "1.0.0", enable_ref=True)
class AddressV1(BaseModel):
    street: str
```

### Custom Generator Not Applied

Check that config is passed correctly:

```python
# Set at manager level
manager = ModelManager(
    default_schema_config=SchemaConfig(
        schema_generator=CustomGenerator
    )
)

# Or per-call
schema = manager.get_schema(
    "User",
    "1.0.0",
    config=SchemaConfig(schema_generator=CustomGenerator)
)
```

## Next Steps

Now that you understand schema generation:

**Advanced customization:**

- [Custom Generators](../advanced/custom-generators.md) - Deep dive into
    GenerateJsonSchema
- [Schema Transformers](../advanced/schema-transformers.md) - Advanced
  transformer patterns

**Related topics:**

- [Nested Models](nested-models.md) - How nested models appear in schemas
- [Discriminated Unions](../advanced/discriminated-unions.md) - Schema
    generation for polymorphic types
- [Registering Models](registering-models.md) - The `enable_ref` parameter

**API Reference:**

- [`SchemaConfig` API](../reference/schema-config.md) - Complete
    `SchemaConfig` details
- [`ModelManager` API](../reference/model-manager.md) - Complete
    `ModelManager` details
- [`ModelDiff` API](../reference/model-diff.md) - Complete `ModelDiff`
    details
