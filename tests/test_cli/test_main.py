"""Tests for pyrmute CLI."""

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest
from pytest import MonkeyPatch
from typer.testing import CliRunner

from pyrmute.cli.main import app

runner = CliRunner()


@pytest.fixture
def test_project(tmp_path: Path, monkeypatch: MonkeyPatch, worker_id: str) -> Path:
    """Create a test project with models."""
    models_file = tmp_path / f"models_{worker_id}.py"
    models_file.write_text("""
from pydantic import BaseModel
from pyrmute import ModelManager

manager = ModelManager()

@manager.composite("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str
    city: str

@manager.composite("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1

@manager.composite("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    email: str
    address: AddressV1
""")

    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(f"""
[pyrmute]
manager = "models_{worker_id}:manager"
""")
    monkeypatch.chdir(tmp_path)

    return tmp_path


def test_validate_success(cli_project: Path, sample_data: Path) -> None:
    """Test successful validation."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "validate",
                "--data",
                str(sample_data),
                "--schema",
                "User",
                "--version",
                "1.0.0",
            ],
        )

    assert result.exit_code == 0
    assert "✓" in result.stdout
    assert "Valid against User v1.0.0" in result.stdout


def test_validate_failure(cli_project: Path, tmp_path: Path) -> None:
    """Test validation failure with detailed errors."""
    invalid_data = tmp_path / "invalid.json"
    invalid_data.write_text(json.dumps({"wrong_field": "value"}))

    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "validate",
                "--data",
                str(invalid_data),
                "--schema",
                "User",
                "--version",
                "1.0.0",
            ],
        )

    assert result.exit_code == 1
    assert "✗" in result.stdout
    assert "Validation failed" in result.stdout


def test_validate_file_not_found(cli_project: Path) -> None:
    """Test validation with missing data file."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "validate",
                "--data",
                "nonexistent.json",
                "--schema",
                "User",
                "--version",
                "1.0.0",
            ],
        )

    assert result.exit_code == 1
    assert "File not found" in result.stdout


def test_validate_invalid_json(cli_project: Path, invalid_json_data: Path) -> None:
    """Test validation with invalid JSON."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "validate",
                "--data",
                str(invalid_json_data),
                "--schema",
                "User",
                "--version",
                "1.0.0",
            ],
        )

    assert result.exit_code == 1
    assert "Invalid JSON" in result.stdout


def test_validate_with_custom_manager(tmp_path: Path) -> None:
    """Test validation with custom manager name."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute.managers]
        default = "models"
        api = "api_models"
    """)
    )

    api_models = tmp_path / "api_models.py"
    api_models.write_text(
        dedent("""
        from pydantic import BaseModel
        from pyrmute import ModelManager

        manager = ModelManager()

        @manager.model("Product", "1.0.0")
        class ProductV1(BaseModel):
            name: str

        __pyrmute_manager__ = manager
    """)
    )

    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps({"name": "Widget"}))

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(
            app,
            [
                "validate",
                "--manager",
                "api",
                "--data",
                str(data_file),
                "--schema",
                "Product",
                "--version",
                "1.0.0",
            ],
        )

    assert result.exit_code == 0
    assert "Using manager: api" in result.stdout


def test_validate_with_explicit_config(tmp_path: Path, sample_data: Path) -> None:
    """Test validation with explicit config file."""
    config_file = tmp_path / "custom.toml"
    config_file.write_text(
        dedent("""
        [pyrmute]
        manager = "models"
    """)
    )

    models_file = tmp_path / "models.py"
    models_file.write_text(
        dedent("""
        from pydantic import BaseModel
        from pyrmute import ModelManager

        manager = ModelManager()

        @manager.model("User", "1.0.0")
        class UserV1(BaseModel):
            name: str

        __pyrmute_manager__ = manager
    """)
    )

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(
            app,
            [
                "validate",
                "--config",
                str(config_file),
                "--data",
                str(sample_data),
                "--schema",
                "User",
                "--version",
                "1.0.0",
            ],
        )

    assert result.exit_code == 0


def test_migrate_success_to_stdout(cli_project: Path, sample_data: Path) -> None:
    """Test successful migration to stdout."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "migrate",
                "--data",
                str(sample_data),
                "--schema",
                "User",
                "--from",
                "1.0.0",
                "--to",
                "2.0.0",
            ],
        )

    assert result.exit_code == 0
    output_data = json.loads(result.stdout)
    assert output_data["name"] == "Alice"
    assert output_data["email"] == "unknown@example.com"


def test_migrate_success_to_file(
    cli_project: Path, sample_data: Path, tmp_path: Path
) -> None:
    """Test successful migration to output file."""
    output_file = tmp_path / "output.json"

    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "migrate",
                "--data",
                str(sample_data),
                "--schema",
                "User",
                "--from",
                "1.0.0",
                "--to",
                "2.0.0",
                "--output",
                str(output_file),
            ],
        )

    assert result.exit_code == 0
    assert output_file.exists()
    assert "Migrated User v1.0.0 → v2.0.0" in result.stdout

    with open(output_file) as f:
        output_data = json.load(f)
    assert output_data["email"] == "unknown@example.com"


def test_migrate_file_not_found(cli_project: Path) -> None:
    """Test migration with missing data file."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "migrate",
                "--data",
                "nonexistent.json",
                "--schema",
                "User",
                "--from",
                "1.0.0",
                "--to",
                "2.0.0",
            ],
        )

    assert result.exit_code == 1
    assert "File not found" in result.stdout


def test_migrate_invalid_json(cli_project: Path, invalid_json_data: Path) -> None:
    """Test migration with invalid JSON."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "migrate",
                "--data",
                str(invalid_json_data),
                "--schema",
                "User",
                "--from",
                "1.0.0",
                "--to",
                "2.0.0",
            ],
        )

    assert result.exit_code == 1
    assert "Invalid JSON" in result.stdout


def test_migrate_with_custom_manager(tmp_path: Path) -> None:
    """Test migration with custom manager name."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute.managers]
        default = "models"
        versioned = "versioned_models"
    """)
    )

    models_file = tmp_path / "versioned_models.py"
    models_file.write_text(
        dedent("""
        from pydantic import BaseModel
        from pyrmute import ModelManager

        manager = ModelManager()

        @manager.model("Item", "1.0.0")
        class ItemV1(BaseModel):
            name: str

        @manager.model("Item", "2.0.0")
        class ItemV2(BaseModel):
            name: str
            description: str

        @manager.migration("Item", "1.0.0", "2.0.0")
        def add_description(data):
            return {**data, "description": "No description"}

        __pyrmute_manager__ = manager
    """)
    )

    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps({"name": "Widget"}))

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(
            app,
            [
                "migrate",
                "--manager",
                "versioned",
                "--data",
                str(data_file),
                "--schema",
                "Item",
                "--from",
                "1.0.0",
                "--to",
                "2.0.0",
            ],
        )

    assert result.exit_code == 0
    output_data = json.loads(result.stdout)
    assert output_data["description"] == "No description"


def test_managers_list_single_manager(cli_project: Path) -> None:
    """Test listing single manager."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(app, ["managers"])

    assert result.exit_code == 0
    assert "Available Managers" in result.stdout
    assert "default" in result.stdout
    assert "models" in result.stdout


def test_managers_list_multiple_managers(tmp_path: Path) -> None:
    """Test listing multiple managers."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute.managers]
        default = "models"
        api_v1 = "api.v1.models"
        api_v2 = "api.v2.models"
    """)
    )

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["managers"])

    assert result.exit_code == 0
    assert "default" in result.stdout
    assert "api_v1" in result.stdout
    assert "api_v2" in result.stdout


def test_managers_list_no_managers(tmp_path: Path) -> None:
    """Test listing when no managers configured."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["managers"])

    assert result.exit_code == 0
    assert "No managers configured" in result.stdout
    assert "pyproject.toml" in result.stdout


def test_managers_list_with_explicit_config(tmp_path: Path) -> None:
    """Test listing with explicit config file."""
    config_file = tmp_path / "custom.toml"
    config_file.write_text(
        dedent("""
        [pyrmute]
        manager = "models"
    """)
    )

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["managers", "--config", str(config_file)])

    assert result.exit_code == 0
    assert "default" in result.stdout


def test_info_default_manager(cli_project: Path) -> None:
    """Test showing info for default manager."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert "Manager: default" in result.stdout
    assert "Registered Models:" in result.stdout
    assert "User" in result.stdout
    assert "v1.0.0" in result.stdout
    assert "v2.0.0" in result.stdout


def test_info_named_manager(tmp_path: Path) -> None:
    """Test showing info for named manager."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute.managers]
        default = "models"
        api = "api_models"
    """)
    )

    api_models = tmp_path / "api_models.py"
    api_models.write_text(
        dedent("""
        from pydantic import BaseModel
        from pyrmute import ModelManager

        manager = ModelManager()

        @manager.model("Product", "1.0.0")
        class ProductV1(BaseModel):
            name: str

        @manager.model("Product", "2.0.0")
        class ProductV2(BaseModel):
            name: str
            price: float

        __pyrmute_manager__ = manager
    """)
    )

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["info", "api"])

    assert result.exit_code == 0
    assert "Manager: api" in result.stdout
    assert "Product" in result.stdout


def test_info_no_models(tmp_path: Path) -> None:
    """Test info when no models registered."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute]
        manager = "models"
    """)
    )

    models_file = tmp_path / "models.py"
    models_file.write_text(
        dedent("""
        from pyrmute import ModelManager
        manager = ModelManager()
        __pyrmute_manager__ = manager
    """)
    )

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert "No models registered" in result.stdout


def test_info_nonexistent_manager(cli_project: Path) -> None:
    """Test info for nonexistent manager."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(app, ["info", "nonexistent"])

    assert result.exit_code == 1
    assert "Error:" in result.stdout


def test_diff_markdown(cli_project: Path) -> None:
    """Test diff output in markdown format."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "diff",
                "--schema",
                "User",
                "--from",
                "1.0.0",
                "--to",
                "2.0.0",
                "--format",
                "markdown",
            ],
        )

    assert result.exit_code == 0
    assert "User" in result.stdout or "email" in result.stdout


def test_diff_json(cli_project: Path) -> None:
    """Test diff output in JSON format."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "diff",
                "--schema",
                "User",
                "--from",
                "1.0.0",
                "--to",
                "2.0.0",
                "--format",
                "json",
            ],
        )

    assert result.exit_code == 0
    json.loads(result.stdout)


def test_diff_invalid_format(cli_project: Path) -> None:
    """Test diff with invalid format."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "diff",
                "--schema",
                "User",
                "--from",
                "1.0.0",
                "--to",
                "2.0.0",
                "--format",
                "xml",
            ],
        )

    assert result.exit_code == 1
    assert "Unknown format" in result.stdout


def test_diff_with_custom_manager(tmp_path: Path) -> None:
    """Test diff with custom manager."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute.managers]
        versioned = "models"
    """)
    )

    models_file = tmp_path / "models.py"
    models_file.write_text(
        dedent("""
        from pydantic import BaseModel
        from pyrmute import ModelManager

        manager = ModelManager()

        @manager.model("Item", "1.0.0")
        class ItemV1(BaseModel):
            name: str

        @manager.model("Item", "2.0.0")
        class ItemV2(BaseModel):
            name: str
            price: float

        __pyrmute_manager__ = manager
    """)
    )

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(
            app,
            [
                "diff",
                "--manager",
                "versioned",
                "--schema",
                "Item",
                "--from",
                "1.0.0",
                "--to",
                "2.0.0",
            ],
        )

    assert result.exit_code == 0


def test_export_json_schema(cli_project: Path, tmp_path: Path) -> None:
    """Test exporting JSON schemas."""
    output_dir = tmp_path / "schemas"

    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "export",
                "--format",
                "json-schema",
                "--output",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0
    assert "Exported JSON Schema schemas" in result.stdout
    assert output_dir.exists()


def test_export_invalid_format(cli_project: Path, tmp_path: Path) -> None:
    """Test export with invalid format."""
    output_dir = tmp_path / "schemas"

    with patch("pyrmute.cli.main.Path.cwd", return_value=cli_project):
        result = runner.invoke(
            app,
            [
                "export",
                "--format",
                "invalid-format",
                "--output",
                str(output_dir),
            ],
        )

    assert result.exit_code == 1
    assert "Unknown format" in result.stdout


def test_export_with_custom_manager(tmp_path: Path) -> None:
    """Test export with custom manager."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute.managers]
        api = "models"
    """)
    )

    models_file = tmp_path / "models.py"
    models_file.write_text(
        dedent("""
        from pydantic import BaseModel
        from pyrmute import ModelManager

        manager = ModelManager()

        @manager.model("Product", "1.0.0")
        class ProductV1(BaseModel):
            name: str

        __pyrmute_manager__ = manager
    """)
    )

    output_dir = tmp_path / "schemas"

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(
            app,
            [
                "export",
                "--manager",
                "api",
                "--format",
                "json-schema",
                "--output",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0


def test_init_basic(tmp_path: Path) -> None:
    """Test basic project initialization."""
    project_dir = tmp_path / "new_project"

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["init", str(project_dir)])

    assert result.exit_code == 0
    assert "Project initialized!" in result.stdout
    assert (project_dir / "models.py").exists()
    assert (project_dir / "pyrmute.toml").exists()


def test_init_with_pyproject(tmp_path: Path) -> None:
    """Test initialization with pyproject.toml."""
    project_dir = tmp_path / "new_project"

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["init", str(project_dir), "--pyproject"])

    assert result.exit_code == 0
    assert (project_dir / "models.py").exists()
    assert (project_dir / "pyproject.toml").exists()


def test_init_multiple_managers(tmp_path: Path) -> None:
    """Test initialization with multiple managers."""
    project_dir = tmp_path / "new_project"

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["init", str(project_dir), "--multiple"])

    assert result.exit_code == 0
    assert "Multiple managers configured" in result.stdout


def test_init_existing_models(tmp_path: Path) -> None:
    """Test initialization skips existing models.py."""
    project_dir = tmp_path / "existing_project"
    project_dir.mkdir()

    models_file = project_dir / "models.py"
    models_file.write_text("# Existing content\n")

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["init", str(project_dir)])

    assert result.exit_code == 0
    assert "Skipping" in result.stdout
    assert models_file.read_text() == "# Existing content\n"


def test_init_current_directory(tmp_path: Path) -> None:
    """Test initialization in current directory."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert (tmp_path / "models.py").exists()


def test_init_permission_error(tmp_path: Path) -> None:
    """Test initialization handles permission errors."""
    project_dir = tmp_path / "restricted"
    project_dir.mkdir()
    project_dir.chmod(0o444)

    try:
        with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["init", str(project_dir)])
        assert result.exit_code == 1
        assert "Error:" in result.stdout
    finally:
        project_dir.chmod(0o755)


def test_json_file_loading(tmp_path: Path) -> None:
    """Test JSON file loading through validate command."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute]
        manager = "models"
    """)
    )

    models_file = tmp_path / "models.py"
    models_file.write_text(
        dedent("""
        from pydantic import BaseModel
        from pyrmute import ModelManager

        manager = ModelManager()

        @manager.model("Test", "1.0.0")
        class TestV1(BaseModel):
            value: int

        __pyrmute_manager__ = manager
    """)
    )

    valid_data = tmp_path / "valid.json"
    valid_data.write_text('{"value": 42}')

    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(
            app,
            [
                "validate",
                "--data",
                str(valid_data),
                "--schema",
                "Test",
                "--version",
                "1.0.0",
            ],
        )

    assert result.exit_code == 0


def test_error_formatting(tmp_path: Path) -> None:
    """Test error message formatting."""
    with patch("pyrmute.cli.main.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["info"])

    assert result.exit_code == 1
    assert "Error:" in result.stdout


def test_stubs_basic(test_project: Path) -> None:
    """Test basic stub generation."""
    output_dir = test_project / "stubs"

    result = runner.invoke(
        app,
        [
            "stubs",
            "-o",
            str(output_dir),
            "--package",
            "testapp",
            "--config",
            str(test_project / "pyrmute.toml"),
        ],
    )

    assert result.exit_code == 0
    assert "Generated type stubs" in result.stdout
    assert (output_dir / "__init__.pyi").exists()
    assert (output_dir / "py.typed").exists()

    content = (output_dir / "__init__.pyi").read_text()
    assert "class AddressV1_0_0(BaseModel):" in content
    assert "class UserV1_0_0(BaseModel):" in content
    assert "class UserV2_0_0(BaseModel):" in content


def test_stubs_specific_models(test_project: Path) -> None:
    """Test stub generation for specific models."""
    output_dir = test_project / "stubs"

    result = runner.invoke(
        app,
        [
            "stubs",
            "-o",
            str(output_dir),
            "--package",
            "testapp",
            "--models",
            "User:1.0.0,User:2.0.0",
            "--config",
            str(test_project / "pyrmute.toml"),
        ],
    )

    assert result.exit_code == 0

    content = (output_dir / "__init__.pyi").read_text()
    assert "class UserV1_0_0(BaseModel):" in content
    assert "class UserV2_0_0(BaseModel):" in content
    assert "class AddressV1_0_0(BaseModel):" in content


def test_stubs_no_nested(test_project: Path) -> None:
    """Test stub generation without nested dependencies."""
    output_dir = test_project / "stubs"

    result = runner.invoke(
        app,
        [
            "stubs",
            "-o",
            str(output_dir),
            "--package",
            "testapp",
            "--models",
            "User:1.0.0",
            "--no-nested",
            "--config",
            str(test_project / "pyrmute.toml"),
        ],
    )

    assert result.exit_code == 0

    content = (output_dir / "__init__.pyi").read_text()
    assert "class UserV1_0_0(BaseModel):" in content
    assert "class AddressV1_0_0(BaseModel):" not in content


def test_stubs_by_module(test_project: Path) -> None:
    """Test stub generation organized by module."""
    output_dir = test_project / "stubs"

    module_map = test_project / "modules.json"
    module_map.write_text(
        json.dumps(
            {
                "user": [["User", "1.0.0"], ["User", "2.0.0"]],
                "address": [["Address", "1.0.0"]],
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "stubs",
            "-o",
            str(output_dir),
            "--package",
            "testapp",
            "--by-module",
            "--module-map",
            str(module_map),
            "--config",
            str(test_project / "pyrmute.toml"),
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "user.pyi").exists()
    assert (output_dir / "address.pyi").exists()
    assert (output_dir / "__init__.pyi").exists()

    init_content = (output_dir / "__init__.pyi").read_text()
    assert "from .user import *" in init_content
    assert "from .address import *" in init_content


def test_create_module_map(test_project: Path) -> None:
    """Test module map creation."""
    output_file = test_project / "modules.json"

    result = runner.invoke(
        app,
        [
            "create-module-map",
            "-o",
            str(output_file),
            "--config",
            str(test_project / "pyrmute.toml"),
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        module_map = json.load(f)

    assert "address" in module_map
    assert "user" in module_map
    assert ["Address", "1.0.0"] in module_map["address"]
    assert ["User", "1.0.0"] in module_map["user"]
    assert ["User", "2.0.0"] in module_map["user"]


def test_stubs_invalid_model_spec(test_project: Path) -> None:
    """Test error handling for invalid model specification."""
    output_dir = test_project / "stubs"

    result = runner.invoke(
        app,
        [
            "stubs",
            "-o",
            str(output_dir),
            "--package",
            "testapp",
            "--models",
            "InvalidFormat",  # Missing version
            "--config",
            str(test_project / "pyrmute.toml"),
        ],
    )

    assert result.exit_code == 1
    assert "Invalid model spec" in result.stdout


def test_stubs_by_module_without_map(test_project: Path) -> None:
    """Test error when using --by-module without --module-map."""
    output_dir = test_project / "stubs"

    result = runner.invoke(
        app,
        [
            "stubs",
            "-o",
            str(output_dir),
            "--package",
            "testapp",
            "--by-module",
            "--config",
            str(test_project / "pyrmute.toml"),
        ],
    )

    assert result.exit_code == 1
    assert "--module-map is required" in result.stdout


def test_stubs_infers_package_name(test_project: Path) -> None:
    """Test that package name is inferred from output directory."""
    output_dir = test_project / "mypackage"

    result = runner.invoke(
        app,
        [
            "stubs",
            "-o",
            str(output_dir),
            "--config",
            str(test_project / "pyrmute.toml"),
        ],
    )

    assert result.exit_code == 0
    content = (output_dir / "__init__.pyi").read_text()
    assert "Type stubs for mypackage" in content
