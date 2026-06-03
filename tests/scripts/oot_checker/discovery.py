"""
Auto-discovers reference Python test files from config-declared paths.

Each OOT config declares:
    path: ${TORCH_DEVICE_ROOT}/tests/inductor/test_inductor_ops.py

The env-var prefix is unknown at check time, so we extract the basename
(e.g. "test_inductor_ops.py") and search for it under a given root directory.
"""

from __future__ import annotations

from pathlib import Path


def find_test_files(
    basenames: list[str],
    search_root: Path,
) -> dict[str, Path]:
    """
    Search for each basename under search_root and return a mapping of
    basename ---> resolved Path for every file that is found.

    Parameters
    ----------
    basenames : list[str]
        Test file basenames extracted from config paths
        (e.g. ["test_inductor_ops.py", "test_ops.py"]).
    search_root : Path
        Directory to search recursively (e.g. the repo root or tests/).

    Returns
    -------
    dict[str, Path]
        Maps basename --> first matching Path found under search_root.
        Basenames with no match are omitted.
    """
    found: dict[str, Path] = {}
    for basename in basenames:
        matches = list(search_root.rglob(basename))
        if matches:
            # Prefer matches inside a "tests/" subtree if there are multiple
            preferred = [m for m in matches if "tests" in m.parts]
            found[basename] = preferred[0] if preferred else matches[0]
    return found
