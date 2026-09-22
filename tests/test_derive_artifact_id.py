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

"""The runner-side derivation: .github/scripts/derive_artifact_id.py.

Loaded by path, like the other .github/scripts tests. No ClickHouse and no network -- the
identity functions it calls are stdlib-only, which is why the action needs no venv.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / ".github" / "scripts" / "derive_artifact_id.py"
LIB = REPO / "extensions" / "clickhouse-ingest"

# The id _package-image stamps into spyre-backend-dev/amd64, verified against prod.
BASE = "2b397099-6200-52fb-98c4-b603961a0582"


@pytest.fixture(scope="module")
def derive_mod():
    spec = importlib.util.spec_from_file_location("derive_artifact_id", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def base_file(tmp_path):
    # Upper-case and padded: the builder's spelling must not change the identity.
    f = tmp_path / "spyre_artifact_id.txt"
    f.write_text(f"  {BASE.upper()} \n")
    return str(f)


def test_a_leg_that_installed_something_chains_onto_the_base(derive_mod, base_file):
    record, aid, base = derive_mod.derive(
        "torch-spyre", "amd64", "torch-spyre@07379f50 lxml", base_file, str(LIB)
    )
    assert base == BASE
    assert aid != BASE
    # The record carries the HASH INPUTS, because a uuid5 cannot be read back. Given order
    # is preserved for readability; only the hash sorts (see installed_digest).
    assert record == f"{aid}|{BASE}|torch-spyre@07379f50,lxml"


def test_install_order_does_not_change_the_identity(derive_mod, base_file):
    a = derive_mod.derive("torch-spyre", "amd64", "b a", base_file, str(LIB))[1]
    b = derive_mod.derive("torch-spyre", "amd64", "a,b", base_file, str(LIB))[1]
    assert a == b


def test_an_unchanged_image_uses_its_own_id_verbatim(derive_mod, base_file):
    # The prebaked path: a derived id there would invent an artifact that never existed.
    record, aid, base = derive_mod.derive(
        "torch-spyre", "amd64", "", base_file, str(LIB)
    )
    assert aid == base == BASE
    assert record == f"{BASE}|{BASE}|"


def test_an_unstamped_image_derives_nothing(derive_mod, tmp_path):
    # Pre-#1782, or a standalone build with no orchestrator node.
    missing = str(tmp_path / "absent.txt")
    assert derive_mod.derive("torch-spyre", "amd64", "x", missing, str(LIB)) == (
        "",
        "",
        "",
    )
    empty = tmp_path / "empty.txt"
    empty.write_text("\n")
    assert derive_mod.derive("torch-spyre", "amd64", "x", str(empty), str(LIB)) == (
        "",
        "",
        "",
    )


def test_a_blank_component_refuses_rather_than_colliding(derive_mod, base_file):
    # A blank hash input mints a uuid every incomplete artifact would share.
    record, aid, base = derive_mod.derive("", "amd64", "x", base_file, str(LIB))
    assert (record, aid) == ("", "")
    assert base == BASE  # still reported, so the log can say what was read


def test_main_writes_the_record_and_both_step_outputs(
    derive_mod, base_file, tmp_path, monkeypatch
):
    out, gho = tmp_path / "artifact-id.txt", tmp_path / "gho.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gho))
    rc = derive_mod.main(
        [
            "--component",
            "torch-spyre",
            "--arch",
            "x86_64",
            "--installed",
            "lxml",
            "--base-id-file",
            base_file,
            "--library-dir",
            str(LIB),
            "--output-file",
            str(out),
        ]
    )
    assert rc == 0
    record = out.read_text()
    assert record.count("|") == 2 and record.startswith(record.split("|")[0])
    emitted = dict(line.split("=", 1) for line in gho.read_text().splitlines())
    assert emitted["artifact_id"] == record.split("|")[0]
    assert emitted["base_artifact_id"] == BASE


def test_main_writes_no_record_file_when_nothing_is_derivable(
    derive_mod, tmp_path, monkeypatch
):
    out, gho = tmp_path / "artifact-id.txt", tmp_path / "gho.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gho))
    rc = derive_mod.main(
        [
            "--component",
            "torch-spyre",
            "--arch",
            "x86_64",
            "--installed",
            "lxml",
            "--base-id-file",
            str(tmp_path / "absent.txt"),
            "--library-dir",
            str(LIB),
            "--output-file",
            str(out),
        ]
    )
    # Exit 0 and no file: the upload step is gated on the output, so nothing is uploaded.
    assert rc == 0 and not out.exists()
    assert gho.read_text().splitlines() == ["artifact_id=", "base_artifact_id="]


def test_a_broken_library_never_fails_the_build(
    derive_mod, base_file, tmp_path, monkeypatch
):
    # Telemetry must not redden a test run, so an unimportable library exits 0 with no id.
    gho = tmp_path / "gho.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gho))
    rc = derive_mod.main(
        [
            "--component",
            "torch-spyre",
            "--arch",
            "x86_64",
            "--base-id-file",
            base_file,
            "--library-dir",
            str(tmp_path / "nowhere"),
        ]
    )
    assert rc == 0
    assert gho.read_text().splitlines() == ["artifact_id=", "base_artifact_id="]


def test_the_script_uses_the_shared_library_not_a_local_copy(derive_mod):
    # The point of extensions/clickhouse-ingest is that ONE definition runs everywhere.
    # Asserted on the FILE, not object identity: the script binds the module under a private
    # package name (see below), so it is the same source loaded twice, not a copy.
    _, _, derive_id = derive_mod._identity(str(LIB))
    assert derive_id.__module__.endswith(".identity")
    assert sys.modules[derive_id.__module__].__file__ == str(
        LIB / "spyre_clickhouse_ingest" / "identity.py"
    )


def test_the_package_init_is_never_run(derive_mod):
    # It imports .client -> clickhouse_connect, a driver this script neither has nor needs.
    # Importing under a PRIVATE name also leaves the real package free for a caller in the
    # same process -- tests/test_ingest_xml.py imports it in this very pytest session.
    derive_mod._identity(str(LIB))
    assert derive_mod._PKG in sys.modules
    assert sys.modules[derive_mod._PKG].__name__ != "spyre_clickhouse_ingest"


def test_it_derives_with_no_clickhouse_driver_installed(tmp_path):
    # The regression this guards: importing the package __init__ pulled in clickhouse_connect,
    # so on a runner without it the script exited 0 having derived NOTHING -- silently, which
    # is the exact failure mode this whole chain exists to remove.
    base = tmp_path / "spyre_artifact_id.txt"
    base.write_text(BASE)
    blocker = (
        "import sys\n"
        "class B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'clickhouse_connect':\n"
        "            raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, B())\n"
        f"sys.argv = ['x', '--component', 'torch-spyre', '--arch', 'x86_64',\n"
        f"            '--installed', 'lxml', '--base-id-file', {str(base)!r},\n"
        f"            '--library-dir', {str(LIB)!r}]\n"
        f"exec(open({str(SCRIPT)!r}).read())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", blocker], capture_output=True, text=True, check=True
    )
    assert "artifact_id: " in out.stdout
    assert "<none" not in out.stdout, out.stderr


def test_a_second_call_honours_a_different_library_dir(derive_mod, tmp_path):
    # The private package name is rebound per call; caching it would serve the first dir
    # forever, which is how a bogus --library-dir came back as a working import.
    assert derive_mod._identity(str(LIB))[0].endswith("spyre_artifact_id.txt")
    with pytest.raises(ModuleNotFoundError):
        derive_mod._identity(str(tmp_path / "nowhere"))
    assert derive_mod._identity(str(LIB))[0].endswith("spyre_artifact_id.txt")
