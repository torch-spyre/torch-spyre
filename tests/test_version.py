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

"""Tests for torch_spyre.version.

``__version__`` is environment-dependent -- a source checkout yields
``0.0.1+g<sha>``, a wheel built without git yields a plain ``0.0.1`` -- so these
assert invariants and shape, never a hardcoded value.

``torch_spyre.version`` is imported directly rather than via ``import
torch_spyre`` so these run without a built ``_C`` extension.
"""

import ast
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    """Locate the repo root without assuming this file's path depth.

    The CI harness runs pytest from its own working directory with the test file
    passed by basename.
    """
    env_root = os.environ.get("TORCH_DEVICE_ROOT")
    if env_root and (Path(env_root) / "torch_spyre" / "version.py").is_file():
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "torch_spyre" / "version.py").is_file():
            return candidate

    raise RuntimeError(
        "cannot locate the torch-spyre repo root: no torch_spyre/version.py found "
        f"above {here} and TORCH_DEVICE_ROOT={env_root!r} does not contain it"
    )


def _load_version_module():
    """Import torch_spyre.version without importing the torch_spyre package."""
    module_path = _repo_root() / "torch_spyre" / "version.py"
    spec = importlib.util.spec_from_file_location("torch_spyre_version", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def version_mod():
    """torch_spyre.version, imported once per module.

    A fixture rather than a module-level call: an import failure here surfaces as
    an error on the individual tests instead of breaking collection of the whole
    file with a bare traceback.
    """
    return _load_version_module()


def _in_source_checkout() -> bool:
    """True when the live git lookup is expected to have fired.

    Evaluated at decoration time by ``skipif``, so it cannot use the fixture and
    resolves the repo root itself.
    """
    return (
        (_repo_root() / ".git").is_dir()
        and shutil.which("git") is not None
        and os.environ.get("TORCH_SPYRE_VERSION_NO_GIT") != "1"
    )


def test_version_is_valid_pep440(version_mod):
    """__version__ must parse as a PEP 440 version, local segment included."""
    packaging_version = pytest.importorskip("packaging.version")
    packaging_version.Version(version_mod.__version__)


def test_base_version_is_the_public_prefix(version_mod):
    """__version__ is either the bare base version or base + a local segment."""
    base = version_mod._BASE_VERSION
    assert version_mod.__version__ == base or version_mod.__version__.startswith(
        base + "+"
    )


def test_local_segment_shape_when_present(version_mod):
    """A local segment must be 'g' followed by a plausible short sha."""
    if "+" not in version_mod.__version__:
        pytest.skip("no local segment (wheel built without git metadata)")
    local = version_mod.__version__.split("+", 1)[1]
    assert local.startswith("g"), f"local segment should start with 'g': {local!r}"
    sha = local[1:]
    assert sha.isalnum(), f"sha should be alphanumeric: {sha!r}"
    assert len(sha) >= 4, f"sha suspiciously short: {sha!r}"


def test_version_module_is_statically_readable(version_mod):
    """The first top-level __version__ binding must be an ast.literal_eval-able str.

    pyproject.toml resolves the project version with
    ``version = {attr = "torch_spyre.version.__version__"}``, which setuptools
    reads via ``ast.literal_eval`` over top-level assignments WITHOUT importing
    the module (setuptools.config.expand.StaticModule). A computed first binding
    -- including a bare name such as ``_BASE_VERSION`` -- breaks every build.

    This replicates that reader so the constraint cannot regress silently.
    """
    source = Path(version_mod.__file__).read_bytes()
    module = ast.parse(source)

    values = [
        statement.value
        for statement in module.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name) and target.id == "__version__"
    ]
    assert values, "no top-level __version__ assignment found"

    # literal_eval raises ValueError on a non-literal; that failure IS the point.
    statically_read = ast.literal_eval(values[0])
    assert isinstance(statically_read, str)
    assert statically_read == version_mod._BASE_VERSION


@pytest.mark.skipif(
    not _in_source_checkout(), reason="not a git source checkout with git available"
)
def test_checkout_version_matches_head(version_mod):
    """In a checkout the local segment must be the real HEAD short sha."""
    sha = subprocess.run(
        ["git", "-C", str(version_mod._REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout.strip()
    expected = f"{version_mod._BASE_VERSION}+g{sha}"
    assert version_mod.__version__ == expected


def test_git_short_sha_returns_none_for_non_repo(version_mod, tmp_path):
    """A path that is not a git repository must yield None, not raise."""
    # A nonexistent path, not tmp_path itself: `git -C` walks UP looking for a
    # repository, and the temp dir could sit under one on some machines.
    assert version_mod._git_short_sha(tmp_path / "definitely-not-a-repo") is None


def test_no_git_env_var_suppresses_lookup(monkeypatch):
    """TORCH_SPYRE_VERSION_NO_GIT=1 must yield the bare base version."""
    monkeypatch.setenv("TORCH_SPYRE_VERSION_NO_GIT", "1")
    reloaded = _load_version_module()
    assert reloaded.__version__ == reloaded._BASE_VERSION
    assert "+" not in reloaded.__version__


def test_module_exports_only_version(version_mod):
    """__all__ stays limited to __version__; helpers are private."""
    assert version_mod.__all__ == ["__version__"]
