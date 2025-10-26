"""Helper functions for the CLI."""

import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def load_json_file(path: Path) -> Any:
    """Load JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {path}", e.doc, e.pos) from e


def write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Write JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    except PermissionError as e:
        raise PermissionError(f"Cannot write to {path}: Permission denied") from e
    except OSError as e:
        raise OSError(f"Failed to write {path}: {e}") from e


def print_success(message: str, manager: str = "default") -> None:
    """Print success message."""
    console.print(f"[green]✓[/green] {message}")
    if manager != "default":
        console.print(f"[dim]Using manager: {manager}[/dim]")


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[red]Error:[/red] {message}")


def print_config_help() -> None:
    """Print configuration help."""
    console.print("Configuration can be added to:")
    console.print("\n1. [bold]pyproject.toml[/bold]:")
    console.print("   [tool.pyrmute]")
    console.print('   manager = "models"\n')

    console.print("2. [bold]pyrmute.toml[/bold]:")
    console.print("   [pyrmute]")
    console.print('   manager = "models"\n')

    console.print("3. [bold]Auto-discovery[/bold]:")
    console.print("   Define __pyrmute_manager__ in models.py")

    console.print("\n[bold]Advanced: Factory Functions[/bold]")
    console.print("   # In models.py:")
    console.print("   def create_manager():")
    console.print("       return ModelManager(...)")
    console.print("   __pyrmute_manager__ = create_manager")

    console.print("\n[bold]Advanced: Initialization Args[/bold]")
    console.print("   [tool.pyrmute]")
    console.print('   manager = "models:create_manager"')
    console.print('   init_args = ["arg1", "arg2"]')
    console.print("   [tool.pyrmute.init_kwargs]")
    console.print("   debug = true")


def create_example_models_file(path: Path) -> None:
    """Create example models.py file."""
    if path.exists():
        console.print(f"[yellow]Skipping:[/yellow] {path} already exists")
        return

    try:
        path.write_text('''"""Versioned models managed by pyrmute."""

from pydantic import BaseModel
from pyrmute import ModelManager

# Create manager instance
manager = ModelManager()


# Version 1: Initial user model
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    """User model v1.0.0"""
    name: str
    email: str


# Version 2: Add age field
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    """User model v2.0.0"""
    name: str
    email: str
    age: int | None = None


# Define migration
@manager.migration("User", "1.0.0", "2.0.0")
def add_age_field(data):
    """Add optional age field."""
    return {**data, "age": None}


# Export manager for CLI discovery
__pyrmute_manager__ = manager


# Alternative: Factory function pattern
# def create_manager():
#     """Factory function for creating manager with custom setup."""
#     mgr = ModelManager()
#     # Register models...
#     return mgr
#
# __pyrmute_manager__ = create_manager
''')
        console.print(f"[green]✓[/green] Created {path}")
    except OSError as e:
        console.print(f"[red]Failed to create {path}:[/red] {e}")
        raise


def create_single_manager_config(project_dir: Path, use_pyproject: bool) -> None:
    """Create single manager configuration."""
    try:
        if use_pyproject:
            config_file = project_dir / "pyproject.toml"
            config_content = """[tool.pyrmute]
manager = "models"
"""
            if config_file.exists():
                console.print(f"\n[yellow]Add this to {config_file}:[/yellow]")
                console.print(config_content)
            else:
                config_file.write_text(f"""[project]
name = "{project_dir.name}"
version = "0.1.0"

{config_content}""")
                console.print(f"[green]✓[/green] Created {config_file}")
        else:
            config_file = project_dir / "pyrmute.toml"
            if not config_file.exists():
                config_file.write_text("""[pyrmute]
manager = "models"

# Optional: Use factory function
# manager = "models:create_manager"

# Optional: Pass initialization arguments
# init_args = []
# [pyrmute.init_kwargs]
# debug = false
""")
                console.print(f"[green]✓[/green] Created {config_file}")
    except OSError as e:
        console.print(f"[red]Failed to create config:[/red] {e}")
        raise


def create_multi_manager_config(project_dir: Path, use_pyproject: bool) -> None:
    """Create multi-manager configuration."""
    try:
        if use_pyproject:
            config_content = """[tool.pyrmute.managers]
default = "models"
api_v1 = "api.v1.models"
api_v2 = "api.v2.models"

# Optional: Configure specific manager with init args
# [tool.pyrmute.managers.api_v1]
# manager = "api.v1.models:create_manager"
# init_args = ["production"]
"""
        else:
            config_content = """[pyrmute.managers]
default = "models"
api_v1 = "api.v1.models"
api_v2 = "api.v2.models"

# Optional: Configure specific manager with init args
# [pyrmute.managers.api_v1]
# manager = "api.v1.models:create_manager"
# init_args = ["production"]
"""

        config_file = project_dir / (
            "pyproject.toml" if use_pyproject else "pyrmute.toml"
        )

        if config_file.exists() and use_pyproject:
            console.print(f"\n[yellow]Add this to {config_file}:[/yellow]")
            console.print(config_content)
        else:
            if use_pyproject:
                config_file.write_text(f"""[project]
name = "{project_dir.name}"
version = "0.1.0"

{config_content}""")
            else:
                config_file.write_text(config_content)
            console.print(f"[green]✓[/green] Created {config_file}")
    except OSError as e:
        console.print(f"[red]Failed to create config:[/red] {e}")
        raise


def print_next_steps(multiple: bool) -> None:
    """Print next steps after initialization."""
    if multiple:
        console.print("\n[bold]Multiple managers configured:[/bold]")
        console.print("  • default (models)")
        console.print("  • api_v1 (api.v1.models)")
        console.print("  • api_v2 (api.v2.models)")
        console.print("\n[bold]Commands:[/bold]")
        console.print("  pyrmute managers              - List all managers")
        console.print("  pyrmute info api_v1           - Show manager details")
        console.print("  pyrmute validate -M api_v1 ...  - Use specific manager")
    else:
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. Edit models.py to add your models")
        console.print("  2. Run: pyrmute info")
        console.print("  3. Run: pyrmute validate -d data.json -s User -v 1.0.0")
