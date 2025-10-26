"""Configuration locatey and loading for pyrmute CLI."""

import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from pyrmute import ModelManager


class ConfigError(Exception):
    """Configuration loading error."""


class ManagerConfig(BaseModel):
    """Configuration for a single manager."""

    name: str
    module_path: str
    attribute: str = Field(default="manager")
    init_args: list[Any] = Field(default_factory=list)
    init_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("init_args")
    @classmethod
    def validate_init_args(cls, v: list[Any]) -> list[Any]:
        """Ensure init_args contains only basic types."""
        for i, arg in enumerate(v):
            if not isinstance(arg, (str, int, float, bool, type(None))):
                raise ValueError(
                    f"init_args[{i}] must be a basic type (str, int, float, "
                    f"bool, None), got {type(arg).__name__}"
                )
        return v

    @field_validator("init_kwargs")
    @classmethod
    def validate_init_kwargs(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure init_kwargs values are basic types."""
        for key, value in v.items():
            if not isinstance(value, (str, int, float, bool, type(None), list, dict)):
                raise ValueError(
                    f"init_kwargs['{key}'] must be a basic type, got "
                    f"{type(value).__name__}"
                )
        return v


def _format_validation_error(error: ValidationError, context: str = "") -> str:
    """Format a Pydantic ValidationError into a user-friendly message."""
    errors = error.errors()

    if len(errors) == 1:
        err = errors[0]
        field = ".".join(str(loc) for loc in err["loc"])
        msg = err["msg"]
        return f"{context}Invalid value for '{field}': {msg}"

    messages = []
    for err in errors:
        field = ".".join(str(loc) for loc in err["loc"])
        msg = err["msg"]
        messages.append(f"  - {field}: {msg}")

    return f"{context}Multiple validation errors:\n" + "\n".join(messages)


def locate_config(
    config_file: Path | None = None,
    start_dir: Path | None = None,
) -> dict[str, ManagerConfig]:
    """Locate pyrmute configuration in the following order.

    1. Explicit config file (if provided)
    2. pyproject.toml [tool.pyrmute]
    3. pyrmute.toml
    4. Auto-locate __pyrmute_manager__ in Python files

    Args:
        config_file: Explicit config file path
        start_dir: Directory to start searching from (defaults to cwd)

    Returns:
        Dict mapping manager names to ManagerConfig instances

    Raises:
        ConfigError: If configuration is invalid or not found
    """
    if start_dir is None:
        start_dir = Path.cwd()

    # 1. Explicit config file
    if config_file:
        if not config_file.exists():
            raise ConfigError(f"Config file not found: {config_file}")
        return _load_toml_config(config_file)

    # 2. Check pyproject.toml
    pyproject = start_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            config = _load_toml_config(pyproject)
            if config:
                return config
        except ConfigError:
            pass  # Fall through to next option

    # 3. Check pyrmute.toml
    pyrmute_toml = start_dir / "pyrmute.toml"
    if pyrmute_toml.exists():
        config = _load_toml_config(pyrmute_toml)
        if config:
            return config

    # 4. Auto-locate
    return _auto_locate_managers(start_dir)


def _load_toml_config(config_path: Path) -> dict[str, ManagerConfig]:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    # Handle pyproject.toml vs pyrmute.toml
    if config_path.name == "pyproject.toml":
        if "tool" not in data or "pyrmute" not in data["tool"]:
            raise ConfigError("No [tool.pyrmute] section found in pyproject.toml")
        config_data = data["tool"]["pyrmute"]
    else:
        config_data = data.get("pyrmute", data)

    managers = {}

    # Single manager format: [tool.pyrmute] or [pyrmute]
    # manager = "path.to.module"
    # or
    # manager = "path.to.module:manager_name"
    if "manager" in config_data:
        manager_spec = config_data["manager"]
        module_path, attribute = _parse_manager_spec(manager_spec)

        try:
            managers["default"] = ManagerConfig(
                name="default",
                module_path=module_path,
                attribute=attribute,
                init_args=config_data.get("init_args", []),
                init_kwargs=config_data.get("init_kwargs", {}),
            )
        except ValidationError as e:
            error_msg = _format_validation_error(
                e, context=f"In {config_path}, manager configuration: "
            )
            raise ConfigError(error_msg) from e

        return managers

    # Multiple managers format: [tool.pyrmute.managers] or [managers]
    # [tool.pyrmute.managers.default]
    # manager = "models"
    # or just:
    # managers.default = "models"
    # or:
    # managers.default = "models:custom_manager"
    if "managers" in config_data:
        managers_config = config_data["managers"]

        for name, spec in managers_config.items():
            # Handle both table and inline formats
            if isinstance(spec, dict):
                if "manager" not in spec:
                    raise ConfigError(
                        f"Manager '{name}' missing 'manager' key in configuration"
                    )
                manager_spec = spec["manager"]
                init_args = spec.get("init_args", [])
                init_kwargs = spec.get("init_kwargs", {})
            else:
                manager_spec = spec
                init_args = []
                init_kwargs = {}

            module_path, attribute = _parse_manager_spec(manager_spec)

            try:
                managers[name] = ManagerConfig(
                    name=name,
                    module_path=module_path,
                    attribute=attribute,
                    init_args=init_args,
                    init_kwargs=init_kwargs,
                )
            except ValidationError as e:
                error_msg = _format_validation_error(
                    e, context=f"In {config_path}, manager '{name}': "
                )
                raise ConfigError(error_msg) from e

        return managers

    raise ConfigError(
        f"No manager configuration found in {config_path}. "
        "Expected 'manager' or 'managers' key."
    )


def _resolve_manager(obj: Any, config: ManagerConfig) -> ModelManager:
    """Resolve an object to a ModelManager, handling various patterns."""
    if isinstance(obj, ModelManager):
        return obj

    if callable(obj):
        try:
            if config.init_args or config.init_kwargs:
                result = obj(*config.init_args, **config.init_kwargs)
            else:
                result = obj()

            return _resolve_manager(result, config)
        except Exception as e:
            raise ConfigError(f"Failed to initialize manager from callable: {e}") from e

    raise ConfigError(
        f"Object is neither a ModelManager instance nor callable. "
        f"Found {type(obj).__name__}"
    )


def _parse_manager_spec(spec: str) -> tuple[str, str]:
    """Parse a manager specification string.

    Examples:
        "models" -> ("models", "manager")
        "models:mgr" -> ("models", "mgr")
        "api.v1.models" -> ("api.v1.models", "manager")
        "api.v1.models:custom" -> ("api.v1.models", "custom")
    """
    if ":" in spec:
        module_path, attribute = spec.split(":", 1)
        return module_path.strip(), attribute.strip()
    return spec.strip(), "manager"


def _auto_locate_managers(start_dir: Path) -> dict[str, ManagerConfig]:
    """Auto-locate managers by searching for __pyrmute_manager__ in Python files.

    Searches for:
    - models.py in current directory
    - Any *_models.py files
    """
    candidates = []

    models_file = start_dir / "models.py"
    if models_file.exists():
        candidates.append(models_file)

    for pattern in ["*_models.py", "*/models.py"]:
        candidates.extend(start_dir.glob(pattern))

    for candidate in candidates:
        try:
            module_path = _get_module_path(candidate, start_dir)
            if _has_pyrmute_manager(candidate):
                name = "default" if candidate.name == "models.py" else candidate.stem
                try:
                    return {
                        name: ManagerConfig(
                            name=name, module_path=module_path, attribute="manager"
                        )
                    }
                except ValidationError as e:
                    error_msg = _format_validation_error(
                        e, context=f"Auto-detected configuration for {candidate}: "
                    )
                    raise ConfigError(error_msg) from e
        except ConfigError:
            raise
        except Exception:
            continue

    raise ConfigError(
        "No pyrmute configuration found. Create a pyproject.toml or pyrmute.toml, "
        "or define __pyrmute_manager__ in models.py"
    )


def _get_module_path(file_path: Path, base_dir: Path) -> str:
    """Convert file path to module path."""
    relative = file_path.relative_to(base_dir)
    return str(relative.with_suffix("")).replace("/", ".")


def _has_pyrmute_manager(file_path: Path) -> bool:
    """Check if a Python file contains __pyrmute_manager__."""
    try:
        content = file_path.read_text()
        return "__pyrmute_manager__" in content
    except Exception:
        return False


def load_manager(
    manager_name: str = "default",
    config_file: Path | None = None,
    start_dir: Path | None = None,
) -> ModelManager:
    """Load a ModelManager instance from configuration.

    Args:
        manager_name: Name of the manager to load
        config_file: Optional explicit config file
        start_dir: Directory to start searching from

    Returns:
        ModelManager instance

    Raises:
        ConfigError: If manager cannot be loaded
    """
    configs = locate_config(config_file, start_dir)

    if manager_name not in configs:
        available = ", ".join(configs.keys())
        raise ConfigError(
            f"Manager '{manager_name}' not found. Available managers: {available}"
        )

    config = configs[manager_name]
    return _import_manager(config)


def _import_module(cwd: Path, module_path: str) -> ModuleType:
    """Imports a module where a manager may or may not be located."""
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        module_file = cwd / module_path.replace(".", "/")
        if not module_file.suffix:
            module_file = module_file.with_suffix(".py")

        if not module_file.exists():
            raise ConfigError(f"Cannot import module '{module_path}': {e}") from e

        spec = importlib.util.spec_from_file_location(module_path, module_file)
        if spec is None or spec.loader is None:
            raise ConfigError(f"Cannot load module from {module_file}") from e

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = module
        spec.loader.exec_module(module)
        return module


def _import_manager(config: ManagerConfig) -> ModelManager:
    """Import and return a ModelManager from a module."""
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    module = _import_module(cwd, config.module_path)

    possible_attrs = [
        config.attribute,
        "__pyrmute_manager__",
        "manager",
    ]

    manager = None
    found_attr = None

    for attr_name in possible_attrs:
        manager = getattr(module, attr_name, None)
        if manager is not None:
            found_attr = attr_name
            break

    if manager is None:
        raise ConfigError(
            f"Module '{config.module_path}' does not have a manager attribute. "
            f"Tried: {', '.join(possible_attrs)}"
        )

    try:
        return _resolve_manager(manager, config)
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(
            f"Unexpected error resolving manager from '{found_attr}': {e}"
        ) from e


def list_available_managers(
    config_file: Path | None = None,
    start_dir: Path | None = None,
) -> dict[str, str]:
    """List all available managers from configuration.

    Returns:
        Dict mapping manager names to their module paths
    """
    try:
        configs = locate_config(config_file, start_dir)
        return {name: cfg.module_path for name, cfg in configs.items()}
    except ConfigError:
        return {}
