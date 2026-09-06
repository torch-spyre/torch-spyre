# Copyright 2026 The Torch-Spyre Authors.
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
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _find_version_file() -> Path:
    """Locate the version.py that is actually in effect, without importing it.

    Two layouts must both work:

    * **Source checkout** -- ``torch_spyre/version.py`` sits next to ``tests/``.
    * **CI wheel flow** -- ``.github/actions/install-prebuilt-torch-spyre`` installs
      the prebuilt wheel and then deletes the checked-out ``torch_spyre/`` package
      so it cannot shadow it, leaving ``tests/`` with no sibling source package.
      The installed wheel is authoritative there, and is the copy whose stamped
      version we actually want to assert on.

    The installed distribution is therefore consulted first, via its recorded file
    list -- ``import torch_spyre`` is avoided throughout because importing the
    package triggers the PyTorch backend autoload (and needs a compiled ``_C``).
    """
    try:
        dist = importlib.metadata.distribution("torch_spyre")
    except importlib.metadata.PackageNotFoundError:
        dist = None

    if dist is not None:
        for recorded in dist.files or ():
            if recorded.parts[-2:] == ("torch_spyre", "version.py"):
                located = Path(str(dist.locate_file(recorded))).resolve()
                if located.is_file():
                    return located

    # Source checkout: walk up for the sentinel rather than assuming this file's
    # depth, since the CI harness may run pytest from a different directory.
    for candidate in Path(__file__).resolve().parents:
        source_copy = candidate / "torch_spyre" / "version.py"
        if source_copy.is_file():
            return source_copy

    raise RuntimeError(
        "cannot locate torch_spyre/version.py: it is neither recorded in an "
        "installed torch_spyre distribution nor present in a parent directory of "
        f"{Path(__file__).resolve()}"
    )


def _read_static_version() -> str:
    """The ``__version__`` literal as it sits on disk, before any git override.

    Read statically rather than imported, so it reflects what the file ships with
    -- a wheel stamped at build time carries a local segment here, a source
    checkout carries the bare base version.
    """
    module = ast.parse(_find_version_file().read_bytes())
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id == "__version__":
                literal = ast.literal_eval(statement.value)
                assert isinstance(literal, str)
                return literal
    raise AssertionError("no top-level __version__ assignment found")


def _load_version_module():
    """Import torch_spyre.version without importing the torch_spyre package."""
    module_path = _find_version_file()
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

    Only a source checkout has a ``.git`` beside the package; the CI wheel flow
    installs into site-packages, where the lookup never fires. Evaluated at
    decoration time by ``skipif``, so it cannot use the fixture -- the directory
    tested here mirrors version.py's own ``_REPO_ROOT`` (its ``parent.parent``).
    """
    return (
        (_find_version_file().parent.parent / ".git").is_dir()
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
    """TORCH_SPYRE_VERSION_NO_GIT=1 must suppress the live git lookup.

    Only the lookup is suppressed, never an already-baked local segment: the
    guarded block is deliberately inert when ``__version__`` already carries a
    ``+`` so an unpacked wheel cannot be re-stamped from an unrelated checkout.
    So a stamped copy (the CI wheel flow) legitimately keeps its segment, and
    only an unstamped source copy collapses to the bare base version.
    """
    monkeypatch.setenv("TORCH_SPYRE_VERSION_NO_GIT", "1")
    baked = "+" in _read_static_version()

    reloaded = _load_version_module()

    if baked:
        assert reloaded.__version__ == _read_static_version()
    else:
        assert reloaded.__version__ == reloaded._BASE_VERSION
        assert "+" not in reloaded.__version__


def test_module_exports_only_version(version_mod):
    """__all__ stays limited to __version__; helpers are private."""
    assert version_mod.__all__ == ["__version__"]
