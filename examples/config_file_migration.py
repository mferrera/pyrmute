"""Configuration file migration example for CLI/desktop applications.

Scenario: Your application evolves over time, adding features and changing configuration
structure. Users upgrade your app but have old config files. You need to:
1. Detect the config version
2. Migrate seamlessly to current format
3. Preserve user settings
4. Save upgraded config for future runs

- Users don't coordinate upgrades (unlike APIs)
- Config files persist across versions
- Breaking changes need smooth transitions
- No downtime or deployment coordination needed

Example: CLI tool that evolves from:
- v1.0: Basic settings with simple flags
- v2.0: Structured logging and API configuration
- v3.0: Plugin system with advanced features
"""

import json
import shutil
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from pyrmute import ModelData, ModelManager

manager = ModelManager()


class LogLevel(StrEnum):
    """Log level options."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Theme(StrEnum):
    """UI theme options."""

    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


# v1.0.0 - Initial release (2022)
@manager.model("AppConfig", "1.0.0")
class AppConfigV1(BaseModel):
    """Original config format - simple boolean flags."""

    api_key: str
    debug: bool = False
    verbose: bool = False
    output_dir: str = "./output"


# v2.0.0 - Major refactor (2023)
@manager.model("LogConfig", "2.0.0", enable_ref=True)
class LogConfigV2(BaseModel):
    """Structured logging configuration."""

    level: LogLevel = LogLevel.INFO
    file: str | None = None
    format: str = "%(asctime)s - %(levelname)s - %(message)s"


@manager.model("APIConfig", "2.0.0", enable_ref=True)
class APIConfigV2(BaseModel):
    """API configuration."""

    key: str
    endpoint: str = "https://api.example.com/v1"
    timeout: int = 30
    retries: int = 3


@manager.model("AppConfig", "2.0.0")
class AppConfigV2(BaseModel):
    """Refactored config with structured sections."""

    api: APIConfigV2
    logging: LogConfigV2
    output_dir: str = "./output"
    cache_enabled: bool = True


# v3.0.0 - Plugin system (2024)
@manager.model("PluginConfig", "3.0.0", enable_ref=True)
class PluginConfigV3(BaseModel):
    """Plugin configuration."""

    name: str
    enabled: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)


@manager.model("UIConfig", "3.0.0", enable_ref=True)
class UIConfigV3(BaseModel):
    """User interface preferences."""

    theme: Theme = Theme.AUTO
    notifications: bool = True
    auto_update: bool = True


@manager.model("AppConfig", "3.0.0")
class AppConfigV3(BaseModel):
    """Current config with plugins and UI settings."""

    api: APIConfigV2
    logging: LogConfigV2
    output_dir: str = "./output"
    cache_enabled: bool = True
    cache_ttl: int = 3600  # New field
    plugins: list[PluginConfigV3] = Field(default_factory=list)
    ui: UIConfigV3 = Field(default_factory=UIConfigV3)
    last_updated: datetime = Field(default_factory=datetime.now)


@manager.migration("AppConfig", "1.0.0", "2.0.0")
def migrate_v1_to_v2(data: ModelData) -> ModelData:
    """Migrate from simple flags to structured config."""
    if data.get("debug"):
        log_level = "DEBUG"
    elif data.get("verbose"):
        log_level = "INFO"
    else:
        log_level = "WARNING"

    return {
        "api": {
            "key": data["api_key"],
            "endpoint": "https://api.example.com/v1",
            "timeout": 30,
            "retries": 3,
        },
        "logging": {
            "level": log_level,
            "file": None,
            "format": "%(asctime)s - %(levelname)s - %(message)s",
        },
        "output_dir": data.get("output_dir", "./output"),
        "cache_enabled": True,
    }


@manager.migration("AppConfig", "2.0.0", "3.0.0")
def migrate_v2_to_v3(data: ModelData) -> ModelData:
    """Add plugin system and UI preferences."""
    return {
        **data,
        "cache_ttl": 3600,
        "plugins": [],
        "ui": {
            "theme": "auto",
            "notifications": True,
            "auto_update": True,
        },
        "last_updated": datetime.now().isoformat(),
    }


class ConfigManager:
    """Manages configuration file loading, migration, and saving."""

    def __init__(self, config_path: Path | str) -> None:
        """Initializes the config manager."""
        self.config_path = Path(config_path)
        self.current_version = "3.0.0"

    def load(self) -> AppConfigV3:
        """Load config file, auto-migrating if needed."""
        if not self.config_path.exists():
            print(f"No config found at {self.config_path}")
            print("Creating default configuration...")
            return self._create_default_config()

        with open(self.config_path) as f:
            data = json.load(f)

        stored_version = data.pop("_version", "1.0.0")
        print(f"Found config version: {stored_version}")

        if stored_version == self.current_version:
            print("Config is up to date!")
            return AppConfigV3.model_validate(data)

        print(f"Migrating config: {stored_version} → {self.current_version}")

        try:
            migrated_config = cast(
                "AppConfigV3",
                manager.migrate(
                    data,
                    "AppConfig",
                    from_version=stored_version,
                    to_version=self.current_version,
                ),
            )

            print("✓ Migration successful!")

            self.save(migrated_config, backup=True)

            return migrated_config

        except Exception as e:
            print(f"✗ Migration failed: {e}")
            print("Using default config instead...")
            return self._create_default_config()

    def save(self, config: AppConfigV3, backup: bool = False) -> None:
        """Save config to disk with version metadata."""
        if backup and self.config_path.exists():
            backup_path = self.config_path.with_suffix(
                f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            self.config_path.rename(backup_path)
            print(f"Created backup: {backup_path}")

        data = config.model_dump(mode="json")
        data["_version"] = self.current_version

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"✓ Saved config to {self.config_path}")

    def _create_default_config(self) -> AppConfigV3:
        """Create a default configuration."""
        config = AppConfigV3(
            api=APIConfigV2(
                key="YOUR_API_KEY_HERE",
                endpoint="https://api.example.com/v1",
            ),
            logging=LogConfigV2(level=LogLevel.INFO),
            output_dir="./output",
        )

        self.save(config)
        return config

    def validate_config(self, config: AppConfigV3) -> list[str]:
        """Validate config and return any warnings/issues."""
        issues = []

        if config.api.key == "YOUR_API_KEY_HERE":
            issues.append("⚠️  API key not set - please update your config")

        if config.cache_ttl < 60:  # noqa: PLR2004
            issues.append("⚠️  Cache TTL very low - may impact performance")

        if config.logging.level == LogLevel.DEBUG:
            issues.append(
                "ℹ️  Debug logging enabled - may produce verbose output"  # noqa: RUF001
            )

        return issues


def create_sample_configs(base_dir: Path) -> None:
    """Create sample config files at different versions."""
    base_dir.mkdir(exist_ok=True)

    # v1.0.0 config (2022)
    v1_config = {
        "_version": "1.0.0",
        "api_key": "sk_test_abc123",
        "debug": True,
        "verbose": False,
        "output_dir": "./data",
    }

    with open(base_dir / "config_v1.json", "w") as f:
        json.dump(v1_config, f, indent=2)

    # v2.0.0 config (2023)
    v2_config = {
        "_version": "2.0.0",
        "api": {
            "key": "sk_test_xyz789",
            "endpoint": "https://api.example.com/v1",
            "timeout": 30,
            "retries": 3,
        },
        "logging": {
            "level": "INFO",
            "file": "/var/log/myapp.log",
            "format": "%(asctime)s - %(levelname)s - %(message)s",
        },
        "output_dir": "./output",
        "cache_enabled": True,
    }

    with open(base_dir / "config_v2.json", "w") as f:
        json.dump(v2_config, f, indent=2)

    # v3.0.0 config (2024)
    v3_config = {
        "_version": "3.0.0",
        "api": {
            "key": "sk_prod_def456",
            "endpoint": "https://api.example.com/v2",
            "timeout": 60,
            "retries": 5,
        },
        "logging": {
            "level": "WARNING",
            "file": None,
            "format": "%(asctime)s - %(levelname)s - %(message)s",
        },
        "output_dir": "./output",
        "cache_enabled": True,
        "cache_ttl": 7200,
        "plugins": [
            {"name": "export-csv", "enabled": True, "settings": {"delimiter": ","}},
            {"name": "notifications", "enabled": False, "settings": {}},
        ],
        "ui": {
            "theme": "dark",
            "notifications": True,
            "auto_update": False,
        },
        "last_updated": datetime.now().isoformat(),
    }

    with open(base_dir / "config_v3.json", "w") as f:
        json.dump(v3_config, f, indent=2)

    print(f"✓ Created sample configs in {base_dir}/")


def simulate_cli_usage() -> None:  # noqa: PLR0915
    """Simulate a CLI tool using config migration."""
    print("=" * 80)
    print("CLI TOOL - Configuration Migration Example")
    print("=" * 80)
    print("\nSimulating a CLI tool that evolves over time...")
    print("Users upgrade the app but have old config files.\n")

    samples_dir = Path("./sample_configs")
    create_sample_configs(samples_dir)

    print("\n" + "=" * 80)
    print("SCENARIO 1: User with v1.0.0 config")
    print("=" * 80)
    config_mgr_v1 = ConfigManager(samples_dir / "config_v1.json")
    config_v1 = config_mgr_v1.load()

    print("\nCurrent configuration:")
    print(f"  API Key: {config_v1.api.key}")
    print(f"  Log Level: {config_v1.logging.level}")
    print(f"  Theme: {config_v1.ui.theme}")
    print(f"  Plugins: {len(config_v1.plugins)}")

    issues = config_mgr_v1.validate_config(config_v1)
    if issues:
        print("\nValidation results:")
        for issue in issues:
            print(f"  {issue}")

    print("\n" + "=" * 80)
    print("SCENARIO 2: User with v2.0.0 config")
    print("=" * 80)
    config_mgr_v2 = ConfigManager(samples_dir / "config_v2.json")
    config_v2 = config_mgr_v2.load()

    print("\nCurrent configuration:")
    print(f"  API Endpoint: {config_v2.api.endpoint}")
    print(f"  Log File: {config_v2.logging.file or 'None'}")
    print(f"  Cache TTL: {config_v2.cache_ttl}s")

    print("\n" + "=" * 80)
    print("SCENARIO 3: User already on v3.0.0")
    print("=" * 80)
    config_mgr_v3 = ConfigManager(samples_dir / "config_v3.json")
    config_v3 = config_mgr_v3.load()

    print("\nPlugins configured:")
    for plugin in config_v3.plugins:
        status = "✓" if plugin.enabled else "✗"
        print(f"  {status} {plugin.name}")

    print("\n" + "=" * 80)
    print("SCENARIO 4: New user (no config)")
    print("=" * 80)
    config_mgr_new = ConfigManager(samples_dir / "config_new.json")
    config_mgr_new.load()

    print("\nDefault configuration created!")

    print("\n" + "=" * 80)
    print("SCHEMA EVOLUTION")
    print("=" * 80)
    diff = manager.diff("AppConfig", "1.0.0", "3.0.0")
    print("\nChanges from v1.0.0 to v3.0.0:")
    print(diff.to_markdown())

    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)

    shutil.rmtree(samples_dir)
    print(f"✓ Removed {samples_dir}/")


def show_best_practices() -> None:
    """Show best practices for config file management."""
    print("\n" + "=" * 80)
    print("BEST PRACTICES")
    print("=" * 80)

    practices = [
        "Always include '_version' field in your config files",
        "Create backups before migrating (see ConfigManager.save)",
        "Validate configs after migration to catch issues early",
        "Provide helpful error messages if migration fails",
        "Use default_factory for new fields to ensure backwards compatibility",
        "Test migrations with real user configs before releasing",
        "Document breaking changes in your CHANGELOG",
        "Consider using backward_compatible for non-breaking additions",
        "Store config in standard locations (~/.config, AppData, etc.)",
        "Log migration events for debugging user issues",
    ]

    for i, practice in enumerate(practices, 1):
        print(f"{i:2}. {practice}")


if __name__ == "__main__":
    simulate_cli_usage()
    show_best_practices()

    print("\n" + "=" * 80)
    print("This pattern is perfect for:")
    print("  ✓ CLI tools (dotfiles, rc files)")
    print("  ✓ Desktop applications (settings, preferences)")
    print("  ✓ IDE plugins (workspace configs)")
    print("  ✓ Dev tools (build configs, linter settings)")
    print("  ✓ Game save files")
    print("  ✓ Local databases (SQLite with JSON fields)")
    print("=" * 80)
