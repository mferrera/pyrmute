# CLI User Guide

The pyrmute CLI provides command-line tools for validating data, migrating
between versions, comparing schemas, and managing your model configurations.
This guide covers installation, configuration, and all available commands.

## Installation

Install pyrmute with CLI support:

```bash
pip install pyrmute[cli]
```

Verify installation:

```bash
pyrmute --help
```

## Quick Start

### 1. Initialize a Project

Create a new pyrmute project with example configuration:

```bash
pyrmute init
```

This creates:
- `models.py` - Example model definitions
- `pyproject.toml` or `pyrmute.toml` - Configuration file

### 2. View Registered Models

See all models and versions:

```bash
pyrmute info
```

Output:
```
Manager: default

Registered Models:

  User
    • v1.0.0
    • v2.0.0

Total: 1 models
```

### 3. Validate Data

Validate a JSON file against a schema version:

**data.json**

```json
{"name": "pyr", "email": "pyr@example.com"}
```

```bash
pyrmute validate -d data.json -s User -v 1.0.0
# ✓ Valid against User v1.0.0
```

## Configuration

### Configuration Files

pyrmute looks for configuration in this order:

1. Explicit config file (via `--config` flag)
2. `pyproject.toml` with `[tool.pyrmute]` section
3. `pyrmute.toml` in current directory
4. Auto-location of `__pyrmute_manager__` in Python files

### Basic Configuration

**pyproject.toml:**

```toml
[tool.pyrmute]
manager = "models"
```

**pyrmute.toml:**

```toml
[pyrmute]
manager = "models"
```

This tells pyrmute to import the `manager` attribute from the `models` module.

### Custom Manager Names

Use a specific attribute name:

```toml
[tool.pyrmute]
manager = "models:custom_manager"
```

This imports `custom_manager` from the `models` module.

### Multiple Managers

Configure multiple managers for different purposes:

```toml
[tool.pyrmute.managers]
default = "models"
api_v1 = "api.v1.models"
api_v2 = "api.v2.models"
```

Use with the `--manager` flag:

```bash
pyrmute info --manager api_v1
pyrmute validate -d data.json -s User -v 1.0.0 --manager api_v2
```

### Factory Functions

Use factory functions with initialization arguments:

```toml
[tool.pyrmute]
manager = "models:create_manager"
init_args = ["production"]

[tool.pyrmute.init_kwargs]
debug = false
cache_enabled = true
```

In `models.py`:

```python
def create_manager(
    env: str, debug: bool = False, cache_enabled: bool = True
) -> ModelManager:
    """Factory function for creating manager."""
    mgr = ModelManager()
    # Do stuff with arguments
    return mgr

__pyrmute_manager__ = create_manager
```

### Auto-Discovery

If no configuration file exists, pyrmute searches for:

- `models.py` with `__pyrmute_manager__` in current directory
- `*_models.py` files with `__pyrmute_manager__`
- `*/models.py` files with `__pyrmute_manager__`

## Commands

### `pyrmute init`

Initialize a new pyrmute project.

**Basic usage:**

```bash
pyrmute init
```

**Options:**

- `--project-dir PATH` - Directory to initialize (default: current)
- `--pyproject` - Use pyproject.toml instead of pyrmute.toml
- `--multiple` - Configure multiple managers

**Examples:**

Create single manager project:

```bash
pyrmute init
```

Create with pyproject.toml:

```bash
pyrmute init --pyproject
```

Create multi-manager project:

```bash
pyrmute init --multiple
```

Create in specific directory:

```bash
pyrmute init --project-dir ./my-project
```

### `pyrmute info`

Show information about registered models.

**Basic usage:**

```bash
pyrmute info [MANAGER]
```

**Arguments:**

- `MANAGER` - Manager name (default: "default")

**Options:**

- `--config PATH` - Path to config file

**Examples:**

Show default manager:

```bash
pyrmute info
```

Show specific manager:

```bash
pyrmute info api_v1
```

Use custom config:

```bash
pyrmute info --config ./custom.toml
```

**Output:**

```
Manager: default

Registered Models:

  User
    • v1.0.0
    • v2.0.0
    • v3.0.0

  Order
    • v1.0.0

Total: 2 models
```

### `pyrmute managers`

List all available managers from configuration.

**Basic usage:**

```bash
pyrmute managers
```

**Options:**

- `--config PATH` - Path to config file

**Output:**

```
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Name    ┃ Module         ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ default │ models         │
│ api_v1  │ api.v1.models  │
│ api_v2  │ api.v2.models  │
└─────────┴────────────────┘

Use with: pyrmute validate --manager <name> ...
```

### `pyrmute validate`

Validate JSON data against a schema version.

**Basic usage:**

```bash
pyrmute validate -d DATA -s SCHEMA -v VERSION
```

**Required options:**

- `-d, --data PATH` - Path to JSON data file
- `-s, --schema NAME` - Schema name
- `-v, --version VERSION` - Schema version

**Optional flags:**

- `-m, --manager NAME` - Manager name (default: "default")
- `-c, --config PATH` - Path to config file

**Examples:**

Validate user data:

```bash
pyrmute validate -d user.json -s User -v 1.0.0
```

Validate with specific manager:

```bash
pyrmute validate -d order.json -s Order -v 2.0.0 --manager api_v1
```

Use custom config:

```bash
pyrmute validate -d data.json -s Product -v 1.0.0 --config ./prod.toml
```

**Success output:**

```
✓ Valid against User v1.0.0
```

**Failure output:**

```
✗ Validation failed

Validation errors:
  • email: Field required
  • age: Input should be a valid integer
```

### `pyrmute migrate`

Migrate data from one schema version to another.

**Basic usage:**

```bash
pyrmute migrate -d DATA -s SCHEMA -f FROM_VERSION -t TO_VERSION
```

**Required options:**

- `-d, --data PATH` - Path to JSON data file
- `-s, --schema NAME` - Schema name
- `-f, --from VERSION` - Source version
- `-t, --to VERSION` - Target version

**Optional flags:**

- `-o, --output PATH` - Output file (default: stdout)
- `-m, --manager NAME` - Manager name (default: "default")
- `-c, --config PATH` - Path to config file

**Examples:**

Migrate to stdout:

```bash
pyrmute migrate -d user_v1.json -s User -f 1.0.0 -t 2.0.0
```

Migrate to file:

```bash
pyrmute migrate -d user_v1.json -s User -f 1.0.0 -t 2.0.0 -o user_v2.json
```

Multi-version migration:

```bash
pyrmute migrate -d user_v1.json -s User -f 1.0.0 -t 3.0.0
```

**Success output (to file):**

```
✓ Migrated User v1.0.0 → v2.0.0
Output written to: user_v2.json
```

**Success output (to stdout):**

```json
{
  "name": "Alice Smith",
  "email": "alice@example.com",
  "age": null
}
```

**Failure output:**

```
Error: Migration validation failed:
  • age: Input should be a valid integer
```

### `pyrmute diff`

Show differences between schema versions.

**Basic usage:**

```bash
pyrmute diff -s SCHEMA -f FROM_VERSION -t TO_VERSION
```

**Required options:**

- `-s, --schema NAME` - Schema name
- `-f, --from VERSION` - Source version
- `-t, --to VERSION` - Target version

**Optional flags:**

- `--format FORMAT` - Output format: "markdown" or "json" (default: "markdown")
- `-m, --manager NAME` - Manager name (default: "default")
- `-c, --config PATH` - Path to config file

**Examples:**

Show diff in markdown:

```bash
pyrmute diff -s User -f 1.0.0 -t 2.0.0
```

Show diff in JSON:

```bash
pyrmute diff -s User -f 1.0.0 -t 2.0.0 --format json
```

Compare with specific manager:

```bash
pyrmute diff -s Order -f 1.0.0 -t 2.0.0 --manager api_v1
```

**Markdown output:**

```markdown
# User: 1.0.0 → 2.0.0

## Added Fields

- `age: int | None` (optional)
- `phone: str` (required)

## Removed Fields

- `username`

## Modified Fields

- `email` - now optional

## Breaking Changes

- ⚠️  New required field 'phone' will fail for existing data without defaults
- ⚠️  Field 'email' changed from optional to required
- ⚠️  Removed fields 'username' will be lost during migration
```

**JSON output:**

```json
{
  "model_name": "User",
  "from_version": "1.0.0",
  "to_version": "2.0.0",
  "added_fields": ["age", "phone"],
  "removed_fields": ["username"],
  "modified_fields": {
    "email": {
      "required_changed": {
        "from": true,
        "to": false
      }
    }
  },
  "added_field_info": {
    "age": {
      "type": "int | None",
      "required": false,
      "default": null
    },
    "phone": {
      "type": "str",
      "required": true,
      "default": null
    }
  },
  "unchanged_fields": ["name"]
}
```

### `pyrmute export`

Export schemas to various formats.

**Basic usage:**

```bash
pyrmute export -f FORMAT -o OUTPUT_DIR
```

**Required options:**

- `-f, --format FORMAT` - Export format
- `-o, --output PATH` - Output directory

**Supported formats:**

- `json-schema` - JSON Schema format
- `avro` - Apache Avro schemas
- `protobuf` - Protocol Buffer schemas
- `typescript` - TypeScript interfaces and Zod schemas

**Optional flags:**

- `-m, --manager NAME` - Manager name (default: "default")
- `-c, --config PATH` - Path to config file
- `--organization STYLE` - Directory organization (TypeScript only): "flat",
      "major_version", "model"
- `--barrel-exports/--no-barrel-exports` - Generate index.ts files (TypeScript
      only, default: enabled)

**Examples:**

Export JSON schemas:

```bash
pyrmute export -f json-schema -o ./schemas
```

Export Avro schemas:

```bash
pyrmute export -f avro -o ./avro
```

Export ProtoBuf schemas:

```bash
pyrmute export -f protobuf -o ./protos
```

Export TypeScript schemas (flat organization):

```bash
pyrmute export -f typescript -o ./types
```

Export TypeScript with major_version organization (recommended):

```bash
pyrmute export -f typescript -o ./types --organization major_version
```

Export TypeScript organized by model:

```bash
pyrmute export -f typescript -o ./types --organization model
```

Export TypeScript without barrel exports:

```bash
pyrmute export -f typescript -o ./types --organization major_version --no-barrel-exports
```

**Success output:**

```
✓ Exported TypeScript schemas (major_version) with barrel exports to ./types/
```

**Directory structure (flat):**

```
types/
├── User.v1.0.0.ts
├── User.v2.0.0.ts
├── Order.v1.0.0.ts
└── Product.v1.0.0.ts
```

**Directory structure (major_version):**

```
types/
├── v1/
│   ├── User.v1.0.0.ts
│   ├── User.v1.1.0.ts
│   ├── Order.v1.0.0.ts
│   ├── Product.v1.0.0.ts
│   └── index.ts
├── v2/
│   ├── User.v2.0.0.ts
│   ├── Order.v2.0.0.ts
│   └── index.ts
└── index.ts
```

**Directory structure (model):**

```
types/
├── User/
│   ├── 1.0.0.ts
│   ├── 1.1.0.ts
│   ├── 2.0.0.ts
│   └── index.ts
├── Order/
│   ├── 1.0.0.ts
│   ├── 2.0.0.ts
│   └── index.ts
├── Product/
│   ├── 1.0.0.ts
│   └── index.ts
└── index.ts
```

**Note:** The `--organization` and `--barrel-exports` flags only apply to
TypeScript exports. They are ignored for other formats.

## Common Workflows

### Development Workflow

**1. Create new model version:**

```python
# models.py
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    email: str
    age: int | None = None  # New field

@manager.migration("User", "1.0.0", "2.0.0")
def add_age_field(data):
    return {**data, "age": None}
```

**2. Check what changed:**

```bash
pyrmute diff -s User -f 1.0.0 -t 2.0.0
```

**3. Test migration:**

```bash
pyrmute migrate -d test_data.json -s User -f 1.0.0 -t 2.0.0
```

**4. Validate migrated data:**

```bash
pyrmute migrate -d test_data.json -s User -f 1.0.0 -t 2.0.0 -o migrated.json
pyrmute validate -d migrated.json -s User -v 2.0.0
```

### CI/CD Integration

**Validate all test data:**

```bash
#!/bin/bash
for file in test_data/*.json; do
  pyrmute validate -d "$file" -s User -v 2.0.0 || exit 1
done
```

**Export schemas for documentation:**

```bash
# In CI pipeline
pyrmute export -f json-schema -o docs/schemas
pyrmute export -f typescript -o frontend/types
```

**Check for breaking changes:**

```bash
# Compare versions and fail if breaking changes detected
pyrmute diff -s User -f 1.0.0 -t 2.0.0 --format json > diff.json

# Parse diff.json and check for breaking changes
python scripts/check_breaking_changes.py diff.json
```

### Data Migration Pipeline

**Migrate multiple files:**

```bash
#!/bin/bash
for file in data/v1/*.json; do
  filename=$(basename "$file")
  pyrmute migrate \
    -d "$file" \
    -s User \
    -f 1.0.0 \
    -t 2.0.0 \
    -o "data/v2/$filename"
done
```

**Batch validation:**

```bash
#!/bin/bash
# Validate all migrated files
for file in data/v2/*.json; do
  echo "Validating $file..."
  pyrmute validate -d "$file" -s User -v 2.0.0
done
```

### Multi-Environment Setup

**Development environment:**

```toml
# dev.toml
[pyrmute]
manager = "models:create_manager"
init_args = ["development"]

[pyrmute.init_kwargs]
debug = true
```

**Production environment:**

```toml
# prod.toml
[pyrmute]
manager = "models:create_manager"
init_args = ["production"]

[pyrmute.init_kwargs]
debug = false
```

**Usage:**

```bash
# Development
pyrmute validate -d data.json -s User -v 2.0.0 --config dev.toml

# Production
pyrmute validate -d data.json -s User -v 2.0.0 --config prod.toml
```

### Generating Type Stubs

```bash
# Generate stubs for all models
pyrmute stubs -o src/myapp --package myapp

# Generate for specific models
pyrmute stubs -o src/myapp --package myapp --models "User:1.0.0,User:2.0.0"

# Organize by module
pyrmute create-module-map -o modules.json
# Edit modules.json to organize models
pyrmute stubs -o src/myapp --by-module --module-map modules.json
```

## Error Handling

### Configuration Errors

**Missing configuration:**

```
Error: No pyrmute configuration found. Create a pyproject.toml or
pyrmute.toml, or define __pyrmute_manager__ in models.py
```

**Solution:** Run `pyrmute init` or create a configuration file.

**Invalid module path:**

```
Error: Cannot import module 'models': No module named 'models'
```

**Solution:** Ensure the module path is correct and the file exists.

**Manager not found:**

```
Error: Manager 'api_v1' not found. Available managers: default
```

**Solution:** Check manager name spelling or run `pyrmute managers` to list available managers.

### Validation Errors

**Schema not found:**

```
Error: Model 'User' version '3.0.0' not found
```

**Solution:** Check available versions with `pyrmute info`.

**Invalid JSON:**

```
Error: Invalid JSON in data.json: Expecting property name enclosed in
double quotes: line 3 column 5 (char 25)
```

**Solution:** Validate JSON syntax with a JSON linter.

**Validation failed:**

```
✗ Validation failed

Validation errors:
  • email: Field required
  • age: Input should be a valid integer
```

**Solution:** Fix data to match schema requirements.

### Migration Errors

**No migration path:**

```
Error: No migration path found from User v1.0.0 to v3.0.0
```

**Solution:** Define migration functions for intermediate versions or create a
direct migration.

**Migration validation failed:**

```
Error: Migration validation failed:
  • age: Field required
```

**Solution:** Update migration function to provide required fields.

## Best Practices

### 1. Use Configuration Files

Always use configuration files rather than relying on auto-discovery:

```toml
# pyproject.toml - explicit and version-controlled
[tool.pyrmute]
manager = "models"
```

### 2. Validate Before Migrating

Always validate data before attempting migration:

```bash
# Check current version is valid
pyrmute validate -d data.json -s User -v 1.0.0

# Then migrate
pyrmute migrate -d data.json -s User -f 1.0.0 -t 2.0.0 -o migrated.json

# Validate migrated data
pyrmute validate -d migrated.json -s User -v 2.0.0
```

### 3. Review Diffs Before Deploying

Check schema differences before deploying new versions:

```bash
pyrmute diff -s User -f 1.0.0 -t 2.0.0
```

Pay attention to breaking changes warnings.

### 4. Use Meaningful Manager Names

For multi-manager setups, use descriptive names:

```toml
[tool.pyrmute.managers]
public_api = "api.public.models"
internal_api = "api.internal.models"
admin_api = "api.admin.models"
```

### 5. Export Schemas Regularly

Export schemas for documentation and external tools:

```bash
# Export during build process
pyrmute export -f json-schema -o docs/api/schemas
pyrmute export -f typescript -o frontend/src/types
```

### 6. Version Control Configuration

Always commit configuration files:

```bash
git add pyproject.toml
git add models.py
git commit -m "Add pyrmute configuration"
```

### 7. Test Migrations Locally

Test migrations with sample data before production:

```bash
# Create test data
cat > test_user.json << EOF
{
  "name": "Test User",
  "email": "test@example.com"
}
EOF

# Test migration
pyrmute migrate -d test_user.json -s User -f 1.0.0 -t 2.0.0
```

## Scripting and Automation

### Bash Scripts

**Validate all files in directory:**

```bash
#!/bin/bash
set -e

SCHEMA="User"
VERSION="2.0.0"

for file in data/*.json; do
  echo "Validating $file..."
  pyrmute validate -d "$file" -s "$SCHEMA" -v "$VERSION"
done

echo "All files validated successfully!"
```

**Migrate with backup:**

```bash
#!/bin/bash
set -e

INPUT="$1"
SCHEMA="$2"
FROM="$3"
TO="$4"

# Create backup
cp "$INPUT" "${INPUT}.backup"

# Migrate
pyrmute migrate \
  -d "$INPUT" \
  -s "$SCHEMA" \
  -f "$FROM" \
  -t "$TO" \
  -o "$INPUT.migrated"

# Validate migrated data
if pyrmute validate -d "$INPUT.migrated" -s "$SCHEMA" -v "$TO"; then
  mv "$INPUT.migrated" "$INPUT"
  echo "Migration successful!"
else
  echo "Migration validation failed, restoring backup"
  mv "${INPUT}.backup" "$INPUT"
  exit 1
fi
```

### Python Scripts

**Batch processing with error handling:**

```python
#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def migrate_file(input_file: Path, schema: str, from_ver: str, to_ver: str) -> bool:
    """Migrate a single file."""
    output_file = input_file.with_suffix('.migrated.json')

    try:
        result = subprocess.run(
            [
                'pyrmute', 'migrate',
                '-d', str(input_file),
                '-s', schema,
                '-f', from_ver,
                '-t', to_ver,
                '-o', str(output_file)
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Migrated {input_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to migrate {input_file}")
        print(e.stderr)
        return False

def main() -> None:
    data_dir = Path('data/v1')
    schema = 'User'
    from_version = '1.0.0'
    to_version = '2.0.0'

    files = list(data_dir.glob('*.json'))
    success_count = 0

    for file in files:
        if migrate_file(file, schema, from_version, to_version):
            success_count += 1

    print(f"\nMigrated {success_count}/{len(files)} files successfully")
    sys.exit(0 if success_count == len(files) else 1)

if __name__ == '__main__':
    main()
```

### Makefile Integration

```makefile
.PHONY: validate migrate export clean

# Validate all test data
validate:
	@echo "Validating test data..."
	@for file in tests/data/*.json; do \
		pyrmute validate -d $$file -s User -v 2.0.0 || exit 1; \
	done

# Migrate data from v1 to v2
migrate:
	@echo "Migrating data..."
	@mkdir -p data/v2
	@for file in data/v1/*.json; do \
		filename=$$(basename $$file); \
		pyrmute migrate \
			-d $$file \
			-s User \
			-f 1.0.0 \
			-t 2.0.0 \
			-o data/v2/$$filename; \
	done

# Export schemas for documentation
export:
	@echo "Exporting schemas..."
	@mkdir -p docs/schemas
	@pyrmute export -f json-schema -o docs/schemas

# Clean generated files
clean:
	@rm -rf data/v2 docs/schemas
```

## Troubleshooting

### Command Not Found

**Error:**

```
pyrmute: command not found
```

**Solution:**

```bash
# Ensure pyrmute is installed
pip install pyrmute[cli]

# Check installation
pip show pyrmute

# Add to PATH if needed
export PATH="$HOME/.local/bin:$PATH"
```

### Import Errors

**Error:**

```
Error: Cannot import module 'models'
```

**Solution:**

- Ensure you're in the correct directory
- Check that `models.py` exists
- Verify PYTHONPATH includes current directory

### Module Cache Issues

If you're getting stale data in tests:

```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

### Permission Errors

**Error:**

```
Error: Permission denied: output.json
```

**Solution:**

```bash
# Check file permissions
ls -l output.json

# Fix permissions
chmod 644 output.json

# Or use different output location
pyrmute migrate -d data.json -s User -f 1.0.0 -t 2.0.0 -o /tmp/output.json
```

## Next Steps

Now that you understand the CLI:

**User Guides:**

- [Registering Models](registering-models.md) - Define your models
- [Writing Migrations](writing-migrations.md) - Write migration functions
- [Schema Generation](schema-generation.md) - Schema generation functionality

**Advanced Topics:**

- [Custom Generators](../advanced/custom-generators.md) - Use custom schema
    generators
- [Migration Hooks](../advanced/migration-hooks.md) - Add hooks to migrations
- [Schema Transformers](../advanced/schema-transformers.md) - Transform
    schemas at generation

**API Reference:**

- [`ModelManager` API](../reference/model-manager.md) - Programmatic API
- [`ModelDiff` API](../reference/model-diff.md) - Programmatic API
