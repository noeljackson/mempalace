"""
palace.py — Shared palace operations.

Consolidates storage access patterns used by both miners and the MCP server.
Routes through the pluggable storage backend (Chroma by default, Postgres
when configured) so downstream code sees a single collection interface.
"""

import os

from .storage import open_collection

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".mempalace",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    ".eggs",
    "htmlcov",
    "target",
}


def get_collection(palace_path: str, collection_name: str = "mempalace_drawers"):
    """Get or create the palace collection via the configured storage backend."""
    # For local Chroma backends, tighten directory permissions up-front.
    # The storage layer is tolerant of a missing/unwritable path for remote
    # backends (e.g. Postgres), so only apply this when the path looks local.
    if palace_path and not palace_path.lower().startswith(("postgres://", "postgresql://")):
        try:
            os.makedirs(palace_path, exist_ok=True)
            os.chmod(palace_path, 0o700)
        except (OSError, NotImplementedError):
            pass
    return open_collection(
        palace_path=palace_path,
        collection_name=collection_name,
        create=True,
    )


def file_already_mined(collection, source_file: str, check_mtime: bool = False) -> bool:
    """Check if a file has already been filed in the palace.

    When check_mtime=True (used by project miner), returns False if the file
    has been modified since it was last mined, so it gets re-mined.
    When check_mtime=False (used by convo miner), just checks existence.
    """
    try:
        get_kwargs = {"where": {"source_file": source_file}, "limit": 1}
        if check_mtime:
            get_kwargs["include"] = ["metadatas"]
        results = collection.get(**get_kwargs)
        if not results.get("ids"):
            return False
        if check_mtime:
            stored_meta = (results.get("metadatas") or [{}])[0] or {}
            stored_mtime = stored_meta.get("source_mtime")
            if stored_mtime is None:
                return False
            current_mtime = os.path.getmtime(source_file)
            return float(stored_mtime) == current_mtime
        return True
    except Exception:
        return False
