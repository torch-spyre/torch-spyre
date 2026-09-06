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
commit can be determined: this project has no tags, so a plain ``0.0.1`` would be
identical for every build. Wheels get the suffix baked in by ``setup.py``'s
``BuildPyWithVersion``; source checkouts, where ``build_py`` never runs, resolve
it live at import (see the block at the bottom of this file).

.. warning::
   The first statement binding ``__version__`` MUST stay a plain string literal.
   ``pyproject.toml`` resolves the project version through an ``attr:`` pointer to
   it, which setuptools reads with ``ast.literal_eval`` over top-level statements
   without importing the module -- so a computed expression, even a bare name,
   breaks every build. Hence ``_BASE_VERSION`` derives *from* ``__version__``, and
   the override is nested inside an ``if``. Locked in by
   ``tests/test_version.py::test_version_module_is_statically_readable``.

A dirty working tree reports its ``HEAD`` commit with no marker, so uncommitted
edits are not reflected in the version.
"""

import os
import subprocess
from pathlib import Path


__all__ = [
    "__version__",
]

# Keep this a bare string literal -- see the module docstring.
__version__ = "0.0.1"

# What a tagged release would carry, i.e. no local segment. Derived from
# __version__, never the reverse -- see the docstring warning.
_BASE_VERSION = __version__

# Escape hatch for reproducible builds and for bisecting version-keyed problems.
_NO_GIT = os.environ.get("TORCH_SPYRE_VERSION_NO_GIT") == "1"

# A `.git` directory beside the package directory is the source-checkout signal;
# an installed wheel has none.
#
# `__file__` is looked up defensively because setup.py also reads this module with
# `exec(f.read(), ns)`, and a bare `__file__` would raise NameError in a namespace
# that lacks it. setup.py seeds it, so the cwd fallback is only a safety net.
_MODULE_PATH = globals().get("__file__")
if _MODULE_PATH is not None:
    _REPO_ROOT = Path(_MODULE_PATH).resolve().parent.parent
else:
    _REPO_ROOT = Path.cwd()


def _git_short_sha(repo_root: Path) -> str | None:
    """Return the short ``HEAD`` sha for ``repo_root``, or ``None``.

    Swallows every failure -- missing ``git``, corrupt or empty repository, hung
    filesystem: the version string is not worth raising from a module import.
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
    # Keep the result a valid PEP 440 local segment; git yields lowercase hex, so
    # anything else means we misread the output.
    if not sha or not sha.isalnum():
        return None
    return sha


# Live resolution, source checkouts only. `rev-parse` rather than
# `git describe --tags` because this repo has zero tags and CI checks out shallow
# without fetch-tags, so `describe` fails outright.
#
# `.is_dir()` comes last so an installed wheel spawns no subprocess at all. The
# `"+" not in __version__` test makes the block idempotent and leaves an
# already-stamped wheel inert, so a wheel unpacked next to an unrelated checkout
# cannot claim that repository's commit.
if not _NO_GIT and "+" not in __version__ and (_REPO_ROOT / ".git").is_dir():
    _sha = _git_short_sha(_REPO_ROOT)
    if _sha is not None:
        __version__ = f"{_BASE_VERSION}+g{_sha}"
    del _sha
