"""Core configuration and path-safety helpers shared by every module.

The public names are re-exported here so callers can use the shorter
``code_helper.core`` import without knowing which module they live in.
"""

from code_helper.core.config import (
    ConfigError,
    Settings,
    get_chat_model,
    get_settings,
    load_environment,
    require_nvidia_credentials,
    require_pinecone_credentials,
)
from code_helper.core.paths import PathSecurityError, project_root, resolve_in_project

__all__ = [
    "ConfigError",
    "PathSecurityError",
    "Settings",
    "get_chat_model",
    "get_settings",
    "load_environment",
    "project_root",
    "require_nvidia_credentials",
    "require_pinecone_credentials",
    "resolve_in_project",
]
