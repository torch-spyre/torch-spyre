# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Package version.

``__version__`` is a PEP 440 local version (``0.0.1+g<short-sha>``) whenever the
commit can be determined.  This project has no tagged releases, so without the
local segment every build of every commit would report the same ``0.0.1`` --
useless for anything keyed on the version, such as diagnostics provenance or a
compile cache that must invalidate when the code changes.

Two independent mechanisms produce the suffix:

* **Wheel builds.**  ``setup.py``'s ``BuildPyWithVersion`` rewrites this file
  inside ``build_lib``, baking a literal ``__version__ = "0.0.1+gabc1234"`` into
  the wheel.  Nothing is resolved at import time there -- an installed wheel has
  no ``.git`` beside it.
* **Source checkouts** (``pip install -e .``, ``uv sync``, running from the repo).
  ``build_py`` never runs for an editable install, so a baked literal would go
  stale on the next commit.  The block at the bottom detects a checkout and
  resolves the commit live at import time.

The first statement binding ``__version__`` MUST stay a plain string literal.
``pyproject.toml`` resolves the project version via
``[tool.setuptools.dynamic] version = {attr = "torch_spyre.version.__version__"}``,
and setuptools reads that with ``ast.literal_eval`` over the module's top-level
assignments (``setuptools.config.expand.StaticModule``) without importing it.  A
computed expression there -- including a bare name such as ``_BASE_VERSION`` --
raises ``ValueError`` and breaks every build.  ``_BASE_VERSION`` is therefore
derived *from* ``__version__`` rather than the reverse, and the override below is
nested inside an ``if`` so the static reader (which walks only top-level
statements, and takes the first binding) never sees it.
``tests/test_version.py::test_version_module_is_statically_readable`` locks this in.

A dirty working tree is reported as its ``HEAD`` commit with no marker, so
uncommitted edits are NOT reflected in the version.
"""

import os
import subprocess
from pathlib import Path


__all__ = [
    "__version__",
]

# Keep this a bare string literal -- see the module docstring.
__version__ = "0.0.1"

# Version without any local segment, i.e. what a tagged release would carry.
# Derived from __version__, never the reverse: a bare name on the right-hand side
# of the __version__ assignment is not ast.literal_eval-able and would break
# pyproject.toml's attr: resolution.
_BASE_VERSION = __version__

# Escape hatch for reproducible builds and for bisecting version-keyed problems.
_NO_GIT = os.environ.get("TORCH_SPYRE_VERSION_NO_GIT") == "1"

# A `.git` directory at the parent of the package directory is the
# source-checkout signal; an installed wheel has none.
#
# `__file__` is looked up defensively rather than referenced directly. This module
# is also read with `exec(f.read(), ns)` by setup.py's get_torch_spyre_version(),
# and a bare `__file__` raises NameError in an exec() namespace that lacks it --
# which would break the build. setup.py seeds `__file__` for exactly that reason,
# so the cwd fallback below is only a safety net for other exec()-style readers.
_MODULE_PATH = globals().get("__file__")
if _MODULE_PATH is not None:
    _REPO_ROOT = Path(_MODULE_PATH).resolve().parent.parent
else:
    _REPO_ROOT = Path.cwd()


def _git_short_sha(repo_root: Path) -> str | None:
    """Return the short ``HEAD`` sha for ``repo_root``, or ``None``.

    Swallows every failure: a missing or broken ``git``, a shallow or corrupt
    repository, a repository with no commits, a hung filesystem.  The version
    string is never important enough to raise from a module import.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    # Only accept something that keeps the result a valid PEP 440 local segment.
    # git yields lowercase hex; anything else means we misread the output.
    if not sha or not sha.isalnum():
        return None
    return sha


# Live resolution, source checkouts only.  `rev-parse --short HEAD` is used
# rather than `git describe --tags` on purpose: this repo has zero tags and CI
# checks out shallow without fetch-tags, so `describe` fails outright.
#
# The `.is_dir()` test comes last so an installed wheel spawns no subprocess at
# all.  The `"+" not in __version__` test makes this block idempotent and leaves
# an already-stamped wheel file inert, so a wheel unpacked next to an unrelated
# checkout cannot silently claim that repository's commit.
if not _NO_GIT and "+" not in __version__ and (_REPO_ROOT / ".git").is_dir():
    _sha = _git_short_sha(_REPO_ROOT)
    if _sha is not None:
        __version__ = f"{_BASE_VERSION}+g{_sha}"
    del _sha
