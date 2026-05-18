"""Configuration loader for YAML config files with env-var interpolation."""

import os
import re
from pathlib import Path
from typing import Any

import yaml


def _interpolate_env(value: Any) -> Any:
    """Recursively replace ${VAR} placeholders with environment variable values."""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return pattern.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(item) for item in value]
    return value


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load a YAML config file and interpolate ${ENV_VAR} placeholders.

    Args:
        path: Path to the YAML config file. Relative paths are resolved
              from the project root (directory containing this file's package).

    Returns:
        Parsed config as a nested dict with env vars substituted.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        # Resolve relative to project root (two levels up from src/utils/)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as fh:
        raw = yaml.safe_load(fh)

    return _interpolate_env(raw or {})
