"""Tests for pyrmute configuration loading."""

import sys
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from pyrmute import ModelManager
from pyrmute.cli.config import (
    ConfigError,
    ManagerConfig,
    _auto_locate_managers,
    _format_validation_error,
    _get_module_path,
    _has_pyrmute_manager,
    _import_manager,
    _import_module,
    _load_toml_config,
    _parse_manager_spec,
    _resolve_manager,
    list_available_managers,
    load_manager,
    locate_config,
)

# ruff: noqa: PLR2004


def test_basic_config() -> None:
    """Test basic ManagerConfig creation."""
    config = ManagerConfig(
        name="default",
        module_path="models",
        attribute="manager",
    )
    assert config.name == "default"
    assert config.module_path == "models"
    assert config.attribute == "manager"
    assert config.init_args == []
    assert config.init_kwargs == {}


def test_config_with_defaults() -> None:
    """Test ManagerConfig with default values."""
    config = ManagerConfig(name="test", module_path="models")
    assert config.attribute == "manager"
    assert config.init_args == []
    assert config.init_kwargs == {}


def test_config_with_init_args() -> None:
    """Test ManagerConfig with init args."""
    config = ManagerConfig(
        name="test",
        module_path="models",
        init_args=["arg1", 123, True],
    )
    assert config.init_args == ["arg1", 123, True]


def test_config_with_init_kwargs() -> None:
    """Test ManagerConfig with init kwargs."""
    config = ManagerConfig(
        name="test",
        module_path="models",
        init_kwargs={"debug": True, "port": 8080},
    )
    assert config.init_kwargs == {"debug": True, "port": 8080}


def test_validation_invalid_init_args_type() -> None:
    """Test validation fails for invalid init_args types."""
    with pytest.raises(ValidationError) as exc_info:
        ManagerConfig(
            name="test",
            module_path="models",
            init_args=[object()],  # Invalid type
        )
    assert "init_args[0]" in str(exc_info.value)


def test_validation_invalid_init_kwargs_type() -> None:
    """Test validation fails for invalid init_kwargs types."""
    with pytest.raises(ValidationError) as exc_info:
        ManagerConfig(
            name="test",
            module_path="models",
            init_kwargs={"key": object()},
        )
    assert "init_kwargs['key']" in str(exc_info.value)


def test_validation_valid_basic_types() -> None:
    """Test validation accepts all valid basic types."""
    config = ManagerConfig(
        name="test",
        module_path="models",
        init_args=["string", 42, 3.14, True, None],
        init_kwargs={
            "str": "value",
            "int": 123,
            "float": 1.5,
            "bool": False,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        },
    )
    assert len(config.init_args) == 5
    assert len(config.init_kwargs) == 7


def test_simple_module() -> None:
    """Test parse simple module name."""
    module, attr = _parse_manager_spec("models")
    assert module == "models"
    assert attr == "manager"


def test_module_with_attribute() -> None:
    """Test parse module with custom attribute."""
    module, attr = _parse_manager_spec("models:custom_mgr")
    assert module == "models"
    assert attr == "custom_mgr"


def test_dotted_module() -> None:
    """Test parse dotted module path."""
    module, attr = _parse_manager_spec("api.v1.models")
    assert module == "api.v1.models"
    assert attr == "manager"


def test_dotted_with_attribute() -> None:
    """Test parse dotted module with attribute."""
    module, attr = _parse_manager_spec("api.v1.models:custom")
    assert module == "api.v1.models"
    assert attr == "custom"


def test_with_spaces() -> None:
    """Test parse spec with surrounding spaces."""
    module, attr = _parse_manager_spec("  models : mgr  ")
    assert module == "models"
    assert attr == "mgr"


def test_format_single_error() -> None:
    """Test formatting single validation error."""
    try:
        ManagerConfig(name="", module_path="models")
    except ValidationError as e:
        msg = _format_validation_error(e, "Test: ")
        assert "Test:" in msg
        assert "name" in msg.lower()


def test_format_multiple_errors() -> None:
    """Test formatting multiple validation errors."""
    try:
        ManagerConfig(name=1, module_path="", init_args=[object()])  # type: ignore[arg-type]
    except ValidationError as e:
        msg = _format_validation_error(e)
        assert "Multiple validation errors:" in msg
        assert "  -" in msg


def test_format_with_context() -> None:
    """Test error formatting with context prefix."""
    try:
        ManagerConfig(name="", module_path="models")
    except ValidationError as e:
        msg = _format_validation_error(e, context="In pyproject.toml: ")
        assert msg.startswith("In pyproject.toml:")


def test_resolve_direct_instance(manager: ModelManager) -> None:
    """Test resolving a direct ModelManager instance."""
    config = ManagerConfig(name="test", module_path="models")
    result = _resolve_manager(manager, config)
    assert result is manager
    assert isinstance(result, ModelManager)


def test_resolve_factory_function(manager: ModelManager) -> None:
    """Test resolving a factory function."""
    config = ManagerConfig(name="test", module_path="models")

    def factory() -> ModelManager:
        return manager

    result = _resolve_manager(factory, config)
    assert result is manager


def test_resolve_with_init_args() -> None:
    """Test resolving with initialization arguments."""
    config = ManagerConfig(
        name="test",
        module_path="models",
        init_args=["arg1", 42],
    )

    def factory(arg1: str, arg2: int) -> ModelManager:
        assert arg1 == "arg1"
        assert arg2 == 42
        return ModelManager()

    result = _resolve_manager(factory, config)
    assert isinstance(result, ModelManager)


def test_resolve_with_init_kwargs() -> None:
    """Test resolving with initialization keyword arguments."""
    config = ManagerConfig(
        name="test",
        module_path="models",
        init_kwargs={"debug": True, "port": 8080},
    )

    def factory(debug: bool = False, port: int = 3000) -> ModelManager:
        assert debug is True
        assert port == 8080
        return ModelManager()

    result = _resolve_manager(factory, config)
    assert isinstance(result, ModelManager)


def test_resolve_recursive_callable(manager: ModelManager) -> None:
    """Test resolving recursively nested callables."""
    config = ManagerConfig(name="test", module_path="models")

    def outer_factory() -> Callable[[], ModelManager]:
        def inner_factory() -> ModelManager:
            return manager

        return inner_factory

    result = _resolve_manager(outer_factory, config)
    assert result is manager


def test_resolve_invalid_type() -> None:
    """Test resolving fails for invalid types."""
    config = ManagerConfig(name="test", module_path="models")
    with pytest.raises(ConfigError) as exc_info:
        _resolve_manager("not a manager", config)
    assert "neither a ModelManager instance nor callable" in str(exc_info.value)


def test_resolve_factory_exception() -> None:
    """Test resolving handles factory exceptions."""
    config = ManagerConfig(name="test", module_path="models")

    def failing_factory() -> ModelManager:
        raise ValueError("Factory failed")

    with pytest.raises(ConfigError) as exc_info:
        _resolve_manager(failing_factory, config)
    assert "Failed to initialize manager from callable" in str(exc_info.value)


def test_get_simple_file(tmp_path: Path) -> None:
    """Test converting simple file path to module path."""
    file_path = tmp_path / "models.py"
    result = _get_module_path(file_path, tmp_path)
    assert result == "models"


def test_get_nested_file(tmp_path: Path) -> None:
    """Test converting nested file path to module path."""
    file_path = tmp_path / "api" / "v1" / "models.py"
    result = _get_module_path(file_path, tmp_path)
    assert result == "api.v1.models"


def test_get_deep_nesting(tmp_path: Path) -> None:
    """Test converting deeply nested file path."""
    file_path = tmp_path / "src" / "api" / "v1" / "models.py"
    result = _get_module_path(file_path, tmp_path)
    assert result == "src.api.v1.models"


def test_has_file_with_marker(tmp_path: Path) -> None:
    """Test detecting file with __pyrmute_manager__."""
    file_path = tmp_path / "models.py"
    file_path.write_text("__pyrmute_manager__ = manager\n")
    assert _has_pyrmute_manager(file_path) is True


def test_has_file_without_marker(tmp_path: Path) -> None:
    """Test detecting file without __pyrmute_manager__."""
    file_path = tmp_path / "models.py"
    file_path.write_text("manager = ModelManager()\n")
    assert _has_pyrmute_manager(file_path) is False


def test_has_nonexistent_file(tmp_path: Path) -> None:
    """Test handling nonexistent file."""
    file_path = tmp_path / "nonexistent.py"
    assert _has_pyrmute_manager(file_path) is False


def test_load_pyproject_single_manager(tmp_path: Path) -> None:
    """Test loading single manager from pyproject.toml."""
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text(
        dedent("""
        [tool.pyrmute]
        manager = "models"
    """)
    )

    configs = _load_toml_config(config_file)
    assert "default" in configs
    assert configs["default"].module_path == "models"
    assert configs["default"].attribute == "manager"


def test_load_pyproject_single_manager_with_attribute(tmp_path: Path) -> None:
    """Test loading single manager with custom attribute."""
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text(
        dedent("""
        [tool.pyrmute]
        manager = "models:custom_mgr"
    """)
    )

    configs = _load_toml_config(config_file)
    assert configs["default"].attribute == "custom_mgr"


def test_load_pyproject_with_init_args(tmp_path: Path) -> None:
    """Test loading manager config with init args."""
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text(
        dedent("""
        [tool.pyrmute]
        manager = "models:create_manager"
        init_args = ["production", 8080]
    """)
    )

    configs = _load_toml_config(config_file)
    assert configs["default"].init_args == ["production", 8080]


def test_load_pyproject_with_init_kwargs(tmp_path: Path) -> None:
    """Test loading manager config with init kwargs."""
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text(
        dedent("""
        [tool.pyrmute]
        manager = "models"
        [tool.pyrmute.init_kwargs]
        debug = true
        port = 3000
    """)
    )

    configs = _load_toml_config(config_file)
    assert configs["default"].init_kwargs == {"debug": True, "port": 3000}


def test_load_pyproject_multiple_managers(tmp_path: Path) -> None:
    """Test loading multiple managers from pyproject.toml."""
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text(
        dedent("""
        [tool.pyrmute.managers]
        default = "models"
        api_v1 = "api.v1.models"
        api_v2 = "api.v2.models:custom"
    """)
    )

    configs = _load_toml_config(config_file)
    assert len(configs) == 3
    assert "default" in configs
    assert "api_v1" in configs
    assert "api_v2" in configs
    assert configs["api_v2"].attribute == "custom"


def test_load_pyrmute_toml(tmp_path: Path) -> None:
    """Test loading from pyrmute.toml."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute]
        manager = "models"
    """)
    )

    configs = _load_toml_config(config_file)
    assert "default" in configs
    assert configs["default"].module_path == "models"


def test_load_pyproject_missing_section(tmp_path: Path) -> None:
    """Test error when pyproject.toml missing [tool.pyrmute]."""
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text(
        dedent("""
        [project]
        name = "test"
    """)
    )

    with pytest.raises(ConfigError) as exc_info:
        _load_toml_config(config_file)
    assert "No [tool.pyrmute] section found" in str(exc_info.value)


def test_load_no_manager_key(tmp_path: Path) -> None:
    """Test error when no manager configuration found."""
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text(
        dedent("""
        [tool.pyrmute]
        other_key = "value"
    """)
    )

    with pytest.raises(ConfigError) as exc_info:
        _load_toml_config(config_file)
    assert "No manager configuration found" in str(exc_info.value)


def test_load_invalid_manager_config(tmp_path: Path) -> None:
    """Test validation error for invalid config."""
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text(
        dedent("""
        [tool.pyrmute]
        manager = "models"
        init_args = "not a list"
    """)
    )

    with pytest.raises(ConfigError) as exc_info:
        _load_toml_config(config_file)
    assert "Invalid value for" in str(exc_info.value)


def test_load_multiple_managers_table_format(tmp_path: Path) -> None:
    """Test loading multiple managers with table format."""
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text(
        dedent("""
        [tool.pyrmute.managers.default]
        manager = "models"

        [tool.pyrmute.managers.api]
        manager = "api.models:create_manager"
        init_args = ["prod"]
    """)
    )

    configs = _load_toml_config(config_file)
    assert len(configs) == 2
    assert configs["api"].init_args == ["prod"]


def test_auto_locate_models_py(tmp_path: Path) -> None:
    """Test auto-locating models.py with marker."""
    models_file = tmp_path / "models.py"
    models_file.write_text("__pyrmute_manager__ = manager\n")

    configs = _auto_locate_managers(tmp_path)
    assert "default" in configs
    assert configs["default"].module_path == "models"


def test_auto_locate_custom_models_file(tmp_path: Path) -> None:
    """Test auto-locating custom *_models.py file."""
    models_file = tmp_path / "api_models.py"
    models_file.write_text("__pyrmute_manager__ = manager\n")

    configs = _auto_locate_managers(tmp_path)
    assert "api_models" in configs


def test_auto_no_manager_found(tmp_path: Path) -> None:
    """Test error when no manager can be auto-located."""
    with pytest.raises(ConfigError) as exc_info:
        _auto_locate_managers(tmp_path)
    assert "No pyrmute configuration found" in str(exc_info.value)


def test_auto_models_without_marker(tmp_path: Path) -> None:
    """Test that models.py without marker is not found."""
    models_file = tmp_path / "models.py"
    models_file.write_text("manager = ModelManager()\n")

    with pytest.raises(ConfigError):
        _auto_locate_managers(tmp_path)


def test_locate_explicit_config_file(tmp_path: Path) -> None:
    """Test loading explicit config file."""
    config_file = tmp_path / "custom.toml"
    config_file.write_text(
        dedent("""
        [pyrmute]
        manager = "models"
    """)
    )

    configs = locate_config(config_file=config_file)
    assert "default" in configs


def test_locate_explicit_config_not_found(tmp_path: Path) -> None:
    """Test error when explicit config file doesn't exist."""
    config_file = tmp_path / "nonexistent.toml"
    with pytest.raises(ConfigError) as exc_info:
        locate_config(config_file=config_file)
    assert "Config file not found" in str(exc_info.value)


def test_locate_pyproject_toml_priority(tmp_path: Path) -> None:
    """Test pyproject.toml is checked before pyrmute.toml."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        dedent("""
        [tool.pyrmute]
        manager = "from_pyproject"
    """)
    )

    pyrmute_toml = tmp_path / "pyrmute.toml"
    pyrmute_toml.write_text(
        dedent("""
        [pyrmute]
        manager = "from_pyrmute"
    """)
    )

    configs = locate_config(start_dir=tmp_path)
    assert configs["default"].module_path == "from_pyproject"


def test_locate_pyrmute_toml_fallback(tmp_path: Path) -> None:
    """Test pyrmute.toml is used when pyproject.toml absent."""
    pyrmute_toml = tmp_path / "pyrmute.toml"
    pyrmute_toml.write_text(
        dedent("""
        [pyrmute]
        manager = "models"
    """)
    )

    configs = locate_config(start_dir=tmp_path)
    assert configs["default"].module_path == "models"


def test_locate_auto_locate_fallback(tmp_path: Path) -> None:
    """Test auto-locate when no config files exist."""
    models_file = tmp_path / "models.py"
    models_file.write_text("__pyrmute_manager__ = manager\n")

    configs = locate_config(start_dir=tmp_path)
    assert "default" in configs


def test_import_standard_module(tmp_path: Path) -> None:
    """Test importing a standard module."""
    models_file = tmp_path / "test_models.py"
    models_file.write_text("TEST_VALUE = 42\n")

    with patch.object(sys, "path", [str(tmp_path), *sys.path]):
        module = _import_module(tmp_path, "test_models")
        assert hasattr(module, "TEST_VALUE")
        assert module.TEST_VALUE == 42


def test_import_nonexistent_module(tmp_path: Path) -> None:
    """Test error importing nonexistent module."""
    with pytest.raises(ConfigError) as exc_info:
        _import_module(tmp_path, "nonexistent_module")
    assert "Cannot import module" in str(exc_info.value)


def test_import_direct_manager(tmp_path: Path, manager: ModelManager) -> None:
    """Test importing direct manager instance."""
    models_file = tmp_path / "test_models.py"
    models_file.write_text(
        dedent("""
        from pyrmute import ModelManager
        manager = ModelManager()
    """)
    )

    config = ManagerConfig(name="test", module_path="test_models")

    with patch.object(sys, "path", [str(tmp_path), *sys.path]):
        result = _import_manager(config)
        assert isinstance(result, ModelManager)


def test_import_factory_function(tmp_path: Path) -> None:
    """Test importing manager from factory function."""
    models_file = tmp_path / "test_models.py"
    models_file.write_text(
        dedent("""
        from pyrmute import ModelManager

        def create_manager():
            return ModelManager()

        manager = create_manager
    """)
    )

    config = ManagerConfig(name="test", module_path="test_models")

    with patch.object(sys, "path", [str(tmp_path), *sys.path]):
        result = _import_manager(config)
        assert isinstance(result, ModelManager)


def test_import_with_pyrmute_marker(tmp_path: Path) -> None:
    """Test importing using __pyrmute_manager__ marker."""
    models_file = tmp_path / "test_models.py"
    models_file.write_text(
        dedent("""
        from pyrmute import ModelManager

        mgr = ModelManager()
        __pyrmute_manager__ = mgr
    """)
    )

    config = ManagerConfig(name="test", module_path="test_models")

    with patch.object(sys, "path", [str(tmp_path), *sys.path]):
        result = _import_manager(config)
        assert isinstance(result, ModelManager)


def test_import_missing_attribute(tmp_path: Path) -> None:
    """Test error when manager attribute not found."""
    models_file = tmp_path / "test_models.py"
    models_file.write_text("# No manager here\n")

    config = ManagerConfig(name="test", module_path="test_models")

    with patch.object(sys, "path", [str(tmp_path), *sys.path]):
        with pytest.raises(ConfigError) as exc_info:
            _import_manager(config)
        assert "does not have a manager attribute" in str(exc_info.value)


def test_import_wrong_type(tmp_path: Path) -> None:
    """Test error when attribute is not a ModelManager."""
    models_file = tmp_path / "test_models.py"
    models_file.write_text("manager = 'not a manager'\n")

    config = ManagerConfig(name="test", module_path="test_models")

    with patch.object(sys, "path", [str(tmp_path), *sys.path]):
        with pytest.raises(ConfigError) as exc_info:
            _import_manager(config)
        assert "neither a ModelManager instance nor callable" in str(exc_info.value)


def test_load_default_manager(tmp_path: Path) -> None:
    """Test loading the default manager."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute]
        manager = "test_models"
    """)
    )

    models_file = tmp_path / "test_models.py"
    models_file.write_text(
        dedent("""
        from pyrmute import ModelManager
        manager = ModelManager()
    """)
    )

    with patch.object(sys, "path", [str(tmp_path), *sys.path]):
        result = load_manager(config_file=config_file, start_dir=tmp_path)
        assert isinstance(result, ModelManager)


def test_load_named_manager(tmp_path: Path) -> None:
    """Test loading a named manager."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute.managers]
        default = "test_models"
        api = "api_models"
    """)
    )

    for name in ["test_models", "api_models"]:
        models_file = tmp_path / f"{name}.py"
        models_file.write_text(
            dedent("""
            from pyrmute import ModelManager
            manager = ModelManager()
        """)
        )

    with patch.object(sys, "path", [str(tmp_path), *sys.path]):
        result = load_manager("api", config_file=config_file, start_dir=tmp_path)
        assert isinstance(result, ModelManager)


def test_load_nonexistent_manager(tmp_path: Path) -> None:
    """Test error loading nonexistent manager."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute]
        manager = "test_models"
    """)
    )

    models_file = tmp_path / "test_models.py"
    models_file.write_text(
        dedent("""
        from pyrmute import ModelManager
        manager = ModelManager()
    """)
    )

    with patch.object(sys, "path", [str(tmp_path), *sys.path]):
        with pytest.raises(ConfigError) as exc_info:
            load_manager("nonexistent", config_file=config_file, start_dir=tmp_path)
        assert "Manager 'nonexistent' not found" in str(exc_info.value)


def test_list_single_manager(tmp_path: Path) -> None:
    """Test listing single manager."""
    config_file = tmp_path / "pyrmute.toml"
    config_file.write_text(
        dedent("""
        [pyrmute]
        manager = "models"
    """)
    )

    managers = list_available_managers(config_file=config_file)
    assert managers == {"default": "models"}


def test_list_multiple_managers(tmp_path: Path) -> None:
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

    managers = list_available_managers(config_file=config_file)
    assert len(managers) == 3
    assert managers["default"] == "models"
    assert managers["api_v1"] == "api.v1.models"
    assert managers["api_v2"] == "api.v2.models"


def test_list_no_config(tmp_path: Path) -> None:
    """Test listing returns empty dict when no config."""
    managers = list_available_managers(start_dir=tmp_path)
    assert managers == {}
