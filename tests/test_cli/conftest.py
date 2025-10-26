"""Fixtures for the CLI."""

import json
import sys
from collections.abc import Generator
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture(autouse=True)
def cleanup_modules() -> Generator[None]:
    """Clean up sys.modules after each test.

    If new imported '*models.py' are added to tests, they must be added here.
    """
    yield
    modules_to_remove = [
        key for key in sys.modules if key.startswith(("api", "models", "test_"))
    ]
    for module in modules_to_remove:
        sys.modules.pop(module, None)


@pytest.fixture
def cli_project(tmp_path: Path) -> Path:
    """Create a temporary project with models and config."""
    models_file = tmp_path / "models.py"
    models_file.write_text(
        dedent("""
        from pydantic import BaseModel
        from pyrmute import ModelManager

        manager = ModelManager()

        @manager.model("User", "1.0.0")
        class UserV1(BaseModel):
            name: str

        @manager.model("User", "2.0.0")
        class UserV2(BaseModel):
            name: str
            email: str

        @manager.migration("User", "1.0.0", "2.0.0")
        def add_email(data):
            return {**data, "email": "unknown@example.com"}

        __pyrmute_manager__ = manager
    """)
    )

    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute]
        manager = "models"
    """)
    )

    return tmp_path


@pytest.fixture
def sample_data(tmp_path: Path) -> Path:
    """Create sample JSON data file."""
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps({"name": "Alice"}))
    return data_file


@pytest.fixture
def invalid_json_data(tmp_path: Path) -> Path:
    """Create invalid JSON data file."""
    data_file = tmp_path / "invalid.json"
    data_file.write_text("not valid json {")
    return data_file
