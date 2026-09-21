"""Resolve manifest task_file entries against a LIBERO dataset root.

Sampling manifests may store either absolute ``task_file`` paths, as written
by the manifest builders on the machine where they were generated, or paths
relative to a LIBERO HDF5 root, as shipped in this repository. Relative
entries are resolved against ``--libero-root`` when a tool provides it, or
against the ``LIBERO_ROOT`` environment variable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def default_libero_root() -> str | None:
    return os.environ.get("LIBERO_ROOT") or None


def resolve_task_file(value: Any, libero_root: str | os.PathLike[str] | None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    root = str(libero_root) if libero_root else default_libero_root()
    if not root:
        raise RuntimeError(
            f"Manifest row has a relative task_file ({path}); pass --libero-root "
            "or set LIBERO_ROOT to the LIBERO HDF5 root directory"
        )
    return Path(root) / path


def resolve_task_file_rows(
    rows: Sequence[Mapping[str, Any]], libero_root: str | os.PathLike[str] | None
) -> list[dict[str, Any]]:
    return [
        dict(row, task_file=str(resolve_task_file(row["task_file"], libero_root)))
        for row in rows
    ]
