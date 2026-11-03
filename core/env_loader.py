"""
Environment variable loader.

Loads variables from a .env file in the project root if it exists.
Variables already set in the environment (e.g., from .bashrc) take precedence
and are NOT overwritten.

This is used by the Claude Code discovery pipeline to ensure required env vars
(CLAUDE_CODE_USE_BEDROCK, AWS_REGION, etc.) are available even when the
launcher is invoked outside an interactive shell.
"""

import os
from pathlib import Path


def load_env(env_path: str | None = None, override: bool = False) -> dict[str, str]:
    """
    Load environment variables from a .env file.

    Supports simple KEY=VALUE lines, quoted values, and # comments.
    Lines starting with 'export ' are handled (the prefix is stripped).

    Args:
        env_path: Path to .env file. If None, searches for .env in the
                  directory of this file's project root (two levels up from
                  core/env_loader.py, i.e., the DataExplorer directory).
        override: If True, overwrite existing env vars. Default False
                  (existing vars from .bashrc take precedence).

    Returns:
        Dict of variables that were loaded (for logging/debugging).
    """
    if env_path is None:
        # Default: look for .env in project root
        project_root = Path(__file__).resolve().parent.parent
        env_path = str(project_root / ".env")

    if not os.path.isfile(env_path):
        return {}

    loaded = {}
    with open(env_path) as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Strip optional 'export ' prefix
            if line.startswith("export "):
                line = line[len("export "):]

            # Split on first '='
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Remove surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]

            if not key:
                continue

            # Only set if not already in environment (unless override=True)
            if override or key not in os.environ:
                os.environ[key] = value
                loaded[key] = value

    if loaded:
        print(f"Loaded {len(loaded)} env var(s) from {env_path}")

    return loaded
