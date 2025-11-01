"""Command-line interface for pyrmute."""

import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from ._helpers import (
    create_example_models_file,
    create_multi_manager_config,
    create_single_manager_config,
    load_json_file,
    print_config_help,
    print_error,
    print_next_steps,
    print_success,
    write_json_file,
)
from .config import (
    ConfigError,
    list_available_managers,
    load_manager,
)

app = typer.Typer(help="Schema evolution and migrations for Pydantic models")
console = Console()

ManagerOption = Annotated[
    str,
    typer.Option(
        ...,
        "--manager",
        "-m",
        help="Manager name (for multiple managers)",
    ),
]

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to config file (pyproject.toml or pyrmute.toml)",
    ),
]


@app.command()
def validate(
    data: Annotated[
        Path, typer.Option(..., "--data", "-d", help="Path to data file (JSON)")
    ],
    schema: Annotated[str, typer.Option(..., "--schema", "-s", help="Schema name")],
    version: Annotated[
        str, typer.Option(..., "--version", "-v", help="Schema version")
    ],
    manager: ManagerOption = "default",
    config: ConfigOption = None,
) -> None:
    """Validate data against a schema version."""
    try:
        mgr = load_manager(manager, config)
        data_dict = load_json_file(data)

        is_valid = mgr.validate_data(data_dict, schema, version)

        if is_valid:
            print_success(f"Valid against {schema} v{version}", manager)
            raise typer.Exit(0)

        console.print("[red]✗[/red] Validation failed")

        try:
            mgr.get(schema, version).model_validate(data_dict)
        except ValidationError as e:
            console.print("\n[red]Validation errors:[/red]")
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                console.print(f"  • {field}: {error['msg']}")
        except Exception as e:
            console.print(f"\n{e}")

        raise typer.Exit(1)

    except ConfigError as e:
        print_error(str(e))
        raise typer.Exit(1) from e
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        raise typer.Exit(1) from e
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in {data}: {e}")
        raise typer.Exit(1) from e
    except typer.Exit:
        raise
    except Exception as e:
        print(e)
        print_error(f"Validation error: {e}")
        raise typer.Exit(1) from e


@app.command()
def migrate(  # noqa: PLR0913
    data: Annotated[
        Path, typer.Option(..., "--data", "-d", help="Path to data file (JSON)")
    ],
    schema: Annotated[str, typer.Option(..., "--schema", "-s", help="Schema name")],
    from_version: Annotated[
        str, typer.Option(..., "--from", "-f", help="Source version")
    ],
    to_version: Annotated[str, typer.Option(..., "--to", "-t", help="Target version")],
    output: Annotated[
        Path | None,
        typer.Option(..., "--output", "-o", help="Output file (default: stdout)"),
    ] = None,
    manager: ManagerOption = "default",
    config: ConfigOption = None,
) -> None:
    """Migrate data from one schema version to another."""
    try:
        mgr = load_manager(manager, config)
        data_dict = load_json_file(data)

        migrated = mgr.migrate(data_dict, schema, from_version, to_version)

        if output:
            write_json_file(output, migrated.model_dump())
            print_success(f"Migrated {schema} v{from_version} → v{to_version}", manager)
            console.print(f"[dim]Output written to: {output}[/dim]")
        else:
            console.print(json.dumps(migrated.model_dump(), indent=2))

        raise typer.Exit(0)

    except ConfigError as e:
        print_error(str(e))
        raise typer.Exit(1) from e
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        raise typer.Exit(1) from e
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in {data}: {e}")
        raise typer.Exit(1) from e
    except ValidationError as e:
        print_error("Migration validation failed:")
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            console.print(f"  • {field}: {error['msg']}")
        raise typer.Exit(1) from e
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Migration error: {e}")
        raise typer.Exit(1) from e


@app.command()
def managers(
    config: ConfigOption = None,
) -> None:
    """List all available managers from configuration."""
    try:
        available = list_available_managers(config)

        if not available:
            console.print("[yellow]No managers configured[/yellow]\n")
            print_config_help()
            return

        table = Table(title="Available Managers")
        table.add_column("Name", style="cyan")
        table.add_column("Module", style="green")

        for name, module in sorted(available.items()):
            table.add_row(name, module)

        console.print(table)
        console.print("\n[dim]Use with: pyrmute validate --manager <name> ...[/dim]")

    except ConfigError as e:
        print_error(str(e))
        raise typer.Exit(1) from e


@app.command()
def info(
    manager: Annotated[str, typer.Argument(..., help="Manager name")] = "default",
    config: ConfigOption = None,
) -> None:
    """Show information about a specific manager."""
    try:
        mgr = load_manager(manager, config)

        console.print(f"[bold]Manager: {manager}[/bold]\n")

        models = mgr.list_models()

        if not models:
            console.print("[yellow]No models registered[/yellow]")
            return

        console.print("[bold]Registered Models:[/bold]\n")

        for model_name in sorted(models):
            versions = mgr.list_versions(model_name)
            console.print(f"  [bold]{model_name}[/bold]")
            for ver in versions:
                console.print(f"    • v{ver}")

        console.print(f"\n[dim]Total: {len(models)} models[/dim]")

    except ConfigError as e:
        print_error(str(e))
        raise typer.Exit(1) from e
    except typer.Exit:
        raise
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(1) from e


@app.command()
def diff(  # noqa: PLR0913
    schema: Annotated[str, typer.Option(..., "--schema", "-s", help="Schema name")],
    from_version: Annotated[
        str, typer.Option(..., "--from", "-f", help="Source version")
    ],
    to_version: Annotated[str, typer.Option(..., "--to", "-t", help="Target version")],
    format: Annotated[
        str, typer.Option(..., "--format", help="Output format (markdown, json)")
    ] = "markdown",
    manager: ManagerOption = "default",
    config: ConfigOption = None,
) -> None:
    """Show differences between schema versions."""
    try:
        mgr = load_manager(manager, config)
        diff_result = mgr.diff(schema, from_version, to_version)

        if format == "markdown":
            console.print(diff_result.to_markdown())
        elif format == "json":
            console.print(json.dumps(diff_result.to_dict(), indent=2))
        else:
            print_error(f"Unknown format: {format}")
            raise typer.Exit(1)

    except ConfigError as e:
        print_error(str(e))
        raise typer.Exit(1) from e
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Diff error: {e}")
        raise typer.Exit(1) from e


@app.command()
def export(  # noqa: PLR0913
    format: Annotated[
        str,
        typer.Option(..., "--format", "-f", help="Export format"),
    ],
    output: Annotated[
        Path, typer.Option(..., "--output", "-o", help="Output directory")
    ],
    organization: Annotated[
        Literal["flat", "major_version", "model"],
        typer.Option(
            ...,
            "--organization",
            help="Directory organization. Only applies to TypeScript exports.",
        ),
    ] = "flat",
    barrel_exports: Annotated[
        bool,
        typer.Option(
            ...,
            "--barrel-exports/--no-barrel-exports",
            help=(
                "Generate index.ts barrel exports. Only applies to TypeScript exports "
                "with non-flat organization."
            ),
        ),
    ] = True,
    manager: ManagerOption = "default",
    config: ConfigOption = None,
) -> None:
    """Export schemas to specified format.

    Examples:
        # Export TypeScript schemas with flat organization (default)
        pyrmute export -f typescript -o ./types

        # Export TypeScript organized by major version with barrel exports
        pyrmute export -f typescript -o ./types --organization major_version

        # Export TypeScript organized by model
        pyrmute export -f typescript -o ./types --organization model

        # Export without barrel exports
        pyrmute export -f typescript -o ./types --organization major_version --no-barrel-exports

        # Export other formats (organization options ignored)
        pyrmute export -f avro -o ./avro
        pyrmute export -f protobuf -o ./proto
        pyrmute export -f json-schema -o ./schemas
    """  # noqa: E501
    format_map = {
        "avro": ("dump_avro_schemas", "Avro"),
        "protobuf": ("dump_proto_schemas", "Protocol Buffer"),
        "typescript": ("dump_typescript_schemas", "TypeScript"),
        "json-schema": ("dump_schemas", "JSON Schema"),
    }

    if format not in format_map:
        print_error(f"Unknown format: {format}")
        console.print(f"Supported formats: {', '.join(format_map.keys())}")
        raise typer.Exit(1)

    try:
        mgr = load_manager(manager, config)
        output.mkdir(parents=True, exist_ok=True)

        method_name, display_name = format_map[format]
        method = getattr(mgr, method_name)

        if format == "typescript":
            method(
                output, organization=organization, include_barrel_exports=barrel_exports
            )

            org_info = ""
            if organization != "flat":
                org_info = f" ({organization})"
                if barrel_exports:
                    org_info += " with barrel exports"

            print_success(
                f"Exported {display_name} schemas{org_info} to {output}/", manager
            )
        else:
            if organization != "flat":
                console.print(
                    "[yellow]Note: --organization is only supported for TypeScript "
                    "exports[/yellow]"
                )
            if not barrel_exports:
                console.print(
                    "[yellow]Note: --barrel-exports is only supported for TypeScript "
                    "exports[/yellow]"
                )

            method(output)
            print_success(f"Exported {display_name} schemas to {output}/", manager)

    except ConfigError as e:
        print_error(str(e))
        raise typer.Exit(1) from e
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Export error: {e}")
        raise typer.Exit(1) from e


@app.command()
def init(
    project_dir: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Project directory",
            default_factory=lambda: Path.cwd(),
        ),
    ],
    use_pyproject: Annotated[
        bool,
        typer.Option(
            ...,
            "--pyproject",
            help="Use pyproject.toml instead of pyrmute.toml",
        ),
    ] = False,
    multiple: Annotated[
        bool,
        typer.Option(
            ...,
            "--multiple",
            help="Create config for multiple managers",
        ),
    ] = False,
) -> None:
    """Initialize a pyrmute project with example configuration."""
    try:
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        create_example_models_file(project_dir / "models.py")

        if multiple:
            create_multi_manager_config(project_dir, use_pyproject)
        else:
            create_single_manager_config(project_dir, use_pyproject)

        console.print("\n[green]✓ Project initialized![/green]")
        print_next_steps(multiple)

    except PermissionError as e:
        print_error(f"Permission denied: {e}")
        raise typer.Exit(1) from e
    except OSError as e:
        print_error(f"Failed to create project: {e}")
        raise typer.Exit(1) from e
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Initialization error: {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
