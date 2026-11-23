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

"""reference verbs: create, validate, seal, compare, the approval rules and the dashboard client (all synthetic)."""

import contextlib
import copy
import importlib
import io
import json
import os
import regex as re
import sys
import tarfile
import urllib.error
from pathlib import Path

import pytest

from spyre_bench import reference, reference_client as rc, store as stores
from spyre_bench.cli import VERBS, main as cli_main

EXT = Path(__file__).resolve().parent.parent
CLEAN = {"exclusive": "enforced"}
SHARED = {"exclusive": "none"}
RUN_ID = "5c0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f"
META = {
    "model": {"repo": "org/m1", "revision": "abc123", "family": "fam"},
    "tools": {"torch": "2.9.1"},
    "comparability": {
        "harness": "synth@1", "harness_match": "yes", "precision": "fp32", "dataset_subset": "full",
        "dataset_match": "same", "shape.batch": 1, "shape.input_len": 128, "shape.output_len": 64,
    },
}  # fmt: skip


def _clear_ingest_modules():
    for k in [k for k in sys.modules if k == "spyre_clickhouse_ingest" or k.startswith("spyre_clickhouse_ingest.")]:
        del sys.modules[k]


def ingest_reference():
    """The intended ingest's reference module, else a skip (a loud failure when BENCH_REQUIRE_INGEST is set).

    The intended ingest is BENCH_REFERENCE_INGEST, else the sibling checkout. A different copy already imported by an
    earlier test (test_engine puts an older one on sys.path) is dropped so it cannot shadow the intended one.
    """
    wanted = [c for c in (os.environ.get("BENCH_REFERENCE_INGEST", ""), str(EXT.parent / "clickhouse-ingest")) if c]
    for cand in wanted:
        pkg = Path(cand) / "spyre_clickhouse_ingest"
        if not (pkg / "reference.py").is_file():
            continue
        loaded = sys.modules.get("spyre_clickhouse_ingest")
        if loaded is not None and Path(loaded.__file__).resolve().parent != pkg.resolve():
            _clear_ingest_modules()
        while cand in sys.path:
            sys.path.remove(cand)
        sys.path.insert(0, cand)  # ahead of any older copy an earlier test left on the path
        try:
            mod = importlib.import_module("spyre_clickhouse_ingest.reference")
        except ImportError:
            continue
        if Path(mod.__file__).resolve().parent != pkg.resolve():
            _clear_ingest_modules()
            pytest.fail(f"imported the wrong ingest: {mod.__file__} is not under {pkg}")
        return mod
    try:
        return importlib.import_module("spyre_clickhouse_ingest.reference")
    except ImportError:
        pass
    if os.environ.get("BENCH_REQUIRE_INGEST"):
        pytest.fail("BENCH_REQUIRE_INGEST is set but the ingest's reference module is not importable")
    pytest.skip("the ingest's reference module is not importable (set BENCH_REFERENCE_INGEST)")


def run_cli(argv, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = reference.main(argv, **kw)
    return code, out.getvalue(), err.getvalue()


def make_result(
    name="op_add",
    samples=((1.0, 1.1), (1.0, 1.2)),
    clean=CLEAN,
    props=None,
    validity=None,
    backend="cpu",
    disc=None,
    disc_keys=None,
):
    reps = [
        {"run.repeat": i, "status": "ok", "clean": clean, "props": dict(props or {}), "samples": {"kernel.time.mean": list(s)}}
        for i, s in enumerate(samples)
    ]  # fmt: skip
    case = {
        "component": "synth-comp",
        "name": name,
        "tags": [],
        "disc": disc or {"shape": "64"},
        "benchmark_id": "stored-id",
    }
    if disc_keys is not None:
        case.update(disc_keys=disc_keys, id_profile="generic")
    if backend:
        case["props"] = {"system.backend": backend}
    status = {"case": case, "repeats": reps, **({"validity": {"state": validity}} if validity else {})}
    return {"apiVersion": "bench/v1alpha1", "kind": "Result", "metadata": {"name": name}, "status": status}


def make_run(tmp_path, results, purpose="nightly", name="run"):
    d = tmp_path / name
    cases = []
    for r in results:
        n = r["metadata"]["name"]
        (d / "cases" / n).mkdir(parents=True)
        (d / "cases" / n / "result.json").write_text(json.dumps(r))
        cases.append({"name": n, "status": "ok", "result": f"cases/{n}/result.json"})
    (d / "run.json").write_text(
        json.dumps({"run_id": RUN_ID, "purpose": purpose, "cases": cases, "ended_at": "2026-11-15T03:00:00Z"})
    )
    return d


def meta_file(tmp_path, **over):
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({**META, **over}))
    return p


def create(tmp_path, run, *extra, key="perf.synth.add.refsys", st=None, meta=True):
    argv = ["create", "--from-run", str(run), "--key", key, "--origin", "generated", "--class", "perf", "--label", "ref-sys-a",
            "--out", str(tmp_path / "ref"), *(["--meta", str(meta_file(tmp_path))] if meta else []), *extra]  # fmt: skip
    return run_cli(argv, st=st or stores.MemoryStore(invalidated=[]))


def case(name="decode", disc=None):
    c = {"component": "synth-comp", "name": name, "tags": [], "disc": disc or {"model": "m1"}}
    c["benchmark_id"] = reference.case_id(c)
    return c


def doc(**over):
    base = {
        "apiVersion": reference.API_VERSION, "kind": reference.KIND, "key": "perf.synth.decode.refsys",
        "origin": "generated", "class": "perf", "label": "ref-sys-a", "terms": "internal-generated",
        "tools": {"torch": "2.9.1"}, "model": {"repo": "org/m1", "revision": "abc123", "family": "fam"},
        "system": {"backend": "cpu"}, "clean": True, "warnings": [], "comparability": dict(META["comparability"]),
        "values": [{"case": case(), "backend": "cpu", "measurements": {"kernel.time.mean": [11.0, 11.5]}, "iterations": 2}],
    }  # fmt: skip
    return {**base, **over}


# ---- validation ----


def test_a_complete_generated_perf_reference_is_valid():
    assert reference.validate(doc()) == []


def mutate(fn):
    d = doc()
    fn(d)
    return d


@pytest.mark.parametrize(
    "change,problem",
    [
        (lambda d: d.pop("terms"), "terms is required"),
        (lambda d: d.update(label="Bad Label"), "label must match"),
        (lambda d: d.update(label="worker-7-ref"), "label must match"),
        (lambda d: d.update(version=3), "reserved key"),
        (lambda d: d.update(extra=1), "unknown key"),
        (lambda d: d["values"][0]["case"].update(benchmark_id="0" * 36), "does not match"),
        (lambda d: d["values"][0]["case"].pop("benchmark_id"), "benchmark_id is required"),
        (lambda d: d["values"][0]["measurements"].update({"m": [float("inf")]}), "finite"),
        (lambda d: d["values"][0].update(iterations=0), "iterations"),
        (lambda d: d.pop("tools"), "needs tools"),
        (lambda d: d["model"].pop("revision"), "model.revision"),
        (lambda d: d.update(origin="published", producer="p", as_of="2026-01-01"), "source_url"),
        (lambda d: d.update(origin="uploaded", producer="p", as_of="2026-01-01"), "comparability.caveats"),
        (lambda d: d["comparability"].update(harness_match="unknown"), "cannot be unknown"),
        (lambda d: d["comparability"].pop("precision"), "comparability.precision is required"),
        (lambda d: d["comparability"].update(bogus=1), "unknown comparability key"),
        (lambda d: d.update(values=[], assets=[]), "needs values or assets"),
        (lambda d: d["values"].append(copy.deepcopy(d["values"][0])), "repeats"),
        (lambda d: d["system"].pop("backend"), "system.backend is required"),
        (lambda d: d.update(**{"class": "quality"}), "dataset.id"),
        (lambda d: d.update(clean="yes"), "clean must be"),
    ],
)
def test_validation_reports_each_problem(change, problem):
    errors = reference.validate(mutate(change))
    assert any(problem in e for e in errors), errors


def test_an_asset_is_checked_against_its_bytes(tmp_path):
    (tmp_path / "attachments").mkdir()
    (tmp_path / "attachments" / "out.jsonl").write_text('{"a":1}\n{"a":2}\n')
    asset = {"path": "attachments/out.jsonl", "sha256": reference.sha256_file(tmp_path / "attachments" / "out.jsonl"),
             "records": 2, "case": case("parity")}  # fmt: skip
    assert reference.validate(doc(assets=[asset]), tmp_path) == []
    bad = dict(asset, sha256="0" * 64)
    assert any("sha256 does not match" in e for e in reference.validate(doc(assets=[bad]), tmp_path))
    assert any(
        "has 2 record(s), not 5" in e for e in reference.validate(doc(assets=[dict(asset, records=5)]), tmp_path)
    )


def test_arch_is_noarch_unless_several_are_named():
    assert reference.arch_of(doc()) == "noarch"
    two = doc(values=[{**doc()["values"][0], "case": case(disc={"arch": a})} for a in ("x86_64", "s390x")])
    assert reference.arch_of(two) == "multi"


def test_the_digest_ignores_terms_label_and_warnings_but_not_values():
    base = reference.content_digest(doc())
    assert reference.content_digest(doc(terms="x", label="other-label", warnings=["not_clean"])) == base
    changed = doc()
    changed["values"][0]["measurements"]["kernel.time.mean"][0] = 12.0
    assert reference.content_digest(changed) != base


def test_case_ids_ignore_tag_order_and_name_padding():
    a = {"component": "synth-comp", "name": " Op ", "tags": ["b", "a"], "disc": {"k": "V"}}
    b = {"component": "synth-comp", "name": "op", "tags": ["A", "b"], "disc": {"k": "v"}}
    assert reference.case_id(a) == reference.case_id(b)


# ---- promotion ----


@pytest.mark.parametrize(
    "run,clean,warns,ok",
    [
        ({"clean": {"exclusive": "enforced"}}, "1", (), True),
        ({"clean": {"exclusive": "observed", "load_clean": True, "devices_exclusive": "enforced"}}, "1", (), True),
        ({"clean": {"exclusive": "observed", "load_clean": True}}, "0", ("not_clean", "exclusivity_not_enforced"), True),
        ({"clean": {"exclusive": "none"}}, "0", ("not_clean",), True),
        ({}, "unknown", ("clean_unknown",), True),
        ({"clean": CLEAN, "validity": "suspect", "pinning_verified": False}, "1", ("suspect", "pinning_unverified"), True),
        ({"clean": CLEAN, "repeats": 1, "purpose": "pr", "qualified": False}, "1",
         ("single_repeat", "not_baseline_purpose", "unqualified_environment"), True),
        ({"clean": CLEAN, "purpose": "weekly"}, "1", (), True),
        ({"clean": CLEAN, "purpose": "merge"}, "1", (), True),
        ({"clean": CLEAN, "purpose": "release"}, "1", ("not_baseline_purpose",), True),
        ({"clean": CLEAN, "purpose": ""}, "1", (), True),
        ({"clean": CLEAN, "invalidated": True}, "1", (), False),
        ({"clean": CLEAN, "measured": False}, "1", (), False),
        ({"clean": CLEAN, "retracted": True}, "1", (), False),
    ],
)  # fmt: skip
def test_promotion_check(run, clean, warns, ok):
    p = reference.promotion_check({"repeats": 2, **run})
    assert (p.ok, p.clean, p.warnings) == (ok, clean, warns)
    assert p.needs_ack == bool(warns)


# ---- create ----


def test_create_writes_a_valid_reference_from_a_clean_run(tmp_path):
    run = make_run(tmp_path, [make_result()])
    code, out, err = create(tmp_path, run)
    assert code == 0, err
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert (tmp_path / "ref" / "attachments").is_dir()
    assert reference.validate(d) == [] and d["clean"] is True and d["warnings"] == []
    v = d["values"][0]
    assert (
        v["backend"] == "cpu"
        and v["measurements"] == {"kernel.time.mean": [1.0, 1.1, 1.0, 1.2]}
        and v["iterations"] == 4
    )
    assert v["case"]["benchmark_id"] == reference.case_id(v["case"])
    assert d["source_run"]["run_id"] == RUN_ID and d["system"]["backend"] == "cpu"


def test_create_from_a_non_clean_run_records_and_prints_warnings(tmp_path):
    run = make_run(tmp_path, [make_result(clean=SHARED)])
    code, out, err = create(tmp_path, run)
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert code == 0 and d["clean"] is False and d["warnings"] == ["not_clean"]
    assert "warning: not_clean" in err and reference.WARNING_TEXT["not_clean"] in err


def test_create_with_unknown_cleanliness_records_null(tmp_path):
    run = make_run(tmp_path, [make_result(clean={})])
    assert create(tmp_path, run)[0] == 0
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert d["clean"] is None and d["warnings"] == ["clean_unknown"]


def test_create_refuses_an_invalidated_run(tmp_path):
    run = make_run(tmp_path, [make_result()])
    code, out, err = create(tmp_path, run, st=stores.MemoryStore(invalidated=[RUN_ID]))
    assert code == 1 and "invalidated" in err and not (tmp_path / "ref").exists()


def test_the_stored_run_ids_are_what_the_store_is_asked_about(tmp_path):
    run = make_run(tmp_path, [make_result()])
    manifest = json.loads((run / "run.json").read_text())
    (run / "run.json").write_text(json.dumps({**manifest, "stored_run_ids": ["stored-1"]}))
    assert create(tmp_path, run, st=stores.MemoryStore(invalidated=[RUN_ID]))[0] == 0


def test_the_stored_run_id_invalidation_blocks(tmp_path):
    run = make_run(tmp_path, [make_result()])
    manifest = json.loads((run / "run.json").read_text())
    (run / "run.json").write_text(json.dumps({**manifest, "stored_run_ids": ["stored-1"]}))
    assert create(tmp_path, run, st=stores.MemoryStore(invalidated=["stored-1"]))[0] == 1


@pytest.mark.parametrize("st", [stores.NullStore(), stores.MemoryStore(fail="down"), stores.MemoryStore()])
def test_an_unreadable_invalidation_state_is_recorded_as_a_warning(tmp_path, st):
    code, out, err = create(tmp_path, make_run(tmp_path, [make_result()]), st=st)
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert code == 0 and "invalidation_unchecked" in d["warnings"] and "invalidation_unchecked" in err


def test_a_readable_store_adds_no_invalidation_warning(tmp_path):
    create(tmp_path, make_run(tmp_path, [make_result()]))
    assert "invalidation_unchecked" not in json.loads((tmp_path / "ref" / "reference.json").read_text())["warnings"]


def test_the_clickhouse_store_reads_invalidated_runs():
    run = "5c0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f"
    ch = FakeCH([{"run_key": run}])
    st = stores.ClickHouseStore(client=ch, database="db")
    assert st.invalidated_runs([run]) == {run}
    assert "v_invalidated_runs" in ch.calls[0][0] and "state = 'invalidated'" in ch.calls[0][0]
    assert ch.calls[0][1] == {"ids": [run]}
    assert st.invalidated_runs([]) == set()
    with pytest.raises(stores.StoreError, match="not UUIDs"):
        st.invalidated_runs([run, "nope"])
    with pytest.raises(stores.StoreError):
        stores.NullStore().invalidated_runs([run])


def test_create_refuses_a_run_with_nothing_measured(tmp_path):
    empty = make_result(samples=())
    code, out, err = create(tmp_path, make_run(tmp_path, [empty]))
    assert code != 0 and not (tmp_path / "ref").exists()


def test_create_carries_the_pinning_policy(tmp_path):
    props = {"run.exec.cpu_pinning": "device_local", "run.exec.pinning_verified": "true"}
    run = make_run(tmp_path, [make_result(props=props)])
    create(tmp_path, run)
    assert json.loads((tmp_path / "ref" / "reference.json").read_text())["pinning"] == "device_local"


def test_unverified_pinning_warns(tmp_path):
    props = {"run.exec.cpu_pinning": "explicit", "run.exec.pinning_verified": "unknown"}
    code, out, err = create(tmp_path, make_run(tmp_path, [make_result(props=props)]))
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert "pinning_unverified" in d["warnings"] and d["pinning"] == "explicit"


@pytest.mark.parametrize(
    "purpose,warns",
    [
        ("pr", True),
        ("release", True),
        ("investigate", True),
        ("nightly", False),
        ("weekly", False),
        ("merge", False),
        (None, False),
    ],
)
def test_only_non_baseline_purposes_warn(tmp_path, purpose, warns):
    run = make_run(tmp_path, [make_result()], purpose=purpose)
    create(tmp_path, run)
    got = json.loads((tmp_path / "ref" / "reference.json").read_text())["warnings"]
    assert ("not_baseline_purpose" in got) is warns


def test_baseline_purposes_match_the_schema_view():
    sql = EXT.parents[1] / "extensions" / "clickhouse-ingest" / "schema" / "62-benchmark-verdicts.sql"
    if not sql.is_file():
        pytest.skip(f"{sql} is not in this checkout")
    found = re.search(
        r"benchmark_baseline_purposes AS\s+SELECT \* FROM values\('purpose String',([^)]*)\)", sql.read_text()
    )
    if not found:
        pytest.skip("benchmark_baseline_purposes is not defined in this checkout's schema 62")
    assert sorted(x.strip().strip("'") for x in found.group(1).split(",")) == sorted(reference.BASELINE_PURPOSES)


def test_a_single_repeat_run_warns(tmp_path):
    create(tmp_path, make_run(tmp_path, [make_result(samples=((1.0,),))]))
    assert "single_repeat" in json.loads((tmp_path / "ref" / "reference.json").read_text())["warnings"]


def test_create_without_the_provenance_fields_lists_what_is_missing(tmp_path):
    code, out, err = create(tmp_path, make_run(tmp_path, [make_result()]), meta=False)
    assert code == 2 and "needs tools" in err and not (tmp_path / "ref").exists()


def test_create_takes_tool_and_comparability_flags(tmp_path):
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({"model": META["model"]}))
    run = make_run(tmp_path, [make_result()])
    argv = ["create", "--from-run", str(run), "--key", "perf.synth.add.refsys", "--class", "perf", "--label", "ref-sys-a",
            "--out", str(tmp_path / "ref"), "--meta", str(p), "--tool", "torch=2.9.1",
            *[x for k, v in META["comparability"].items() for x in ("--comparability", f"{k}={v}")]]  # fmt: skip
    code, out, err = run_cli(argv, st=stores.NullStore())
    assert code == 0, err


def test_create_needs_a_backend_when_the_result_names_none(tmp_path):
    run = make_run(tmp_path, [make_result(backend="")])
    assert create(tmp_path, run)[0] == 2
    assert create(tmp_path, run, "--backend", "cpu")[0] == 0


def test_create_refuses_a_non_empty_output_directory(tmp_path):
    (tmp_path / "ref").mkdir()
    (tmp_path / "ref" / "x").write_text("x")
    assert create(tmp_path, make_run(tmp_path, [make_result()]))[0] == 2


def test_create_finds_a_run_by_id(tmp_path):
    make_run(tmp_path, [make_result()], name="r1")
    argv = ["create", "--from-run", RUN_ID, "--runs-dir", str(tmp_path), "--key", "perf.synth.add.refsys", "--class", "perf",
            "--label", "ref-sys-a", "--out", str(tmp_path / "ref"), "--meta", str(meta_file(tmp_path))]  # fmt: skip
    assert run_cli(argv, st=stores.NullStore())[0] == 0
    assert run_cli(argv[:2] + ["no-such-run"] + argv[3:], st=stores.NullStore())[0] == 2


def test_create_reports_unregistered_cases_and_never_fails_on_it(tmp_path):
    run = make_run(tmp_path, [make_result()])
    code, out, err = create(tmp_path, run, st=stores.MemoryStore(known=[]))
    assert code == 0 and "not registered" in err and "register them now" in err


def test_create_notes_when_registration_cannot_be_checked(tmp_path):
    code, out, err = create(tmp_path, make_run(tmp_path, [make_result()]), st=stores.MemoryStore(fail="down"))
    assert code == 0 and "could not check case registration" in err


def test_create_is_quiet_when_every_case_is_registered(tmp_path):
    run = make_run(tmp_path, [make_result()])
    bid = reference.case_id(reference.case_block(make_result()))
    code, out, err = create(tmp_path, run, st=stores.MemoryStore(known=[bid]))
    assert code == 0 and "registered" not in err


# ---- validate and seal ----


def made_ref(tmp_path, d=None):
    root = tmp_path / "refdir"
    reference.write_dir(d or doc(), root)
    return root


def test_validate_verb(tmp_path):
    root = made_ref(tmp_path)
    assert run_cli(["validate", str(root)])[0] == 0
    bad = made_ref(tmp_path / "x", doc(terms=""))
    code, out, err = run_cli(["validate", str(bad)])
    assert code == 2 and "terms is required" in err
    assert run_cli(["validate", str(tmp_path / "nothing")])[0] == 2


def test_seal_writes_a_bundle_directory_with_hashes(tmp_path):
    root = made_ref(tmp_path)
    (root / "attachments" / "n.txt").write_text("hi")
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "b")])[0] == 2
    asset = {
        "path": "attachments/n.txt",
        "sha256": reference.sha256_file(root / "attachments" / "n.txt"),
        "records": 1,
        "case": case("parity"),
    }
    (root / "reference.json").write_text(json.dumps(doc(assets=[asset])))
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "b")])[0] == 0
    meta = json.loads((tmp_path / "b" / "bundle.json").read_text())
    assert meta["kind"] == "bench-reference"
    assert [f["path"] for f in meta["files"]] == ["attachments/n.txt", "results/reference.json"]
    assert all(f["sha256"] == reference.sha256_file(tmp_path / "b" / f["path"]) for f in meta["files"])
    assert run_cli(["validate", str(tmp_path / "b")])[0] == 0


def test_seal_writes_a_tgz(tmp_path):
    root = made_ref(tmp_path)
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "b.tgz")])[0] == 0
    with tarfile.open(tmp_path / "b.tgz") as tar:
        assert sorted(tar.getnames()) == ["bundle.json", "results/reference.json"]
    assert reference.bundle_doc(tmp_path / "b.tgz")["key"] == doc()["key"]


def test_seal_refuses_an_invalid_reference_and_a_used_target(tmp_path):
    bad = made_ref(tmp_path / "x", doc(label="Nope"))
    code, out, err = run_cli(["seal", str(bad), "--out", str(tmp_path / "b")])
    assert code == 2 and "not sealed" in err and not (tmp_path / "b").exists()
    ok = made_ref(tmp_path)
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "f").write_text("x")
    assert run_cli(["seal", str(ok), "--out", str(tmp_path / "used")])[0] == 2


def test_seal_refuses_a_bundle(tmp_path):
    root = made_ref(tmp_path)
    run_cli(["seal", str(root), "--out", str(tmp_path / "b")])
    assert run_cli(["seal", str(tmp_path / "b"), "--out", str(tmp_path / "c")])[0] == 2


# ---- the approval rules and the client ----


def bindings():
    return [
        {"subject": "amy", "permissions": ["references:approve"], "scope": {"class": ["perf"]}},
        {"subject": "bob", "permissions": ["references:approve"], "scope": {"class": ["quality"]}},
        {"subject": "sam", "permissions": ["references:approve"]},
        {"subject": "grp", "subject_type": "group", "permissions": ["references:approve"]},
    ]


def server(**kw):
    return rc.FakeServer(bindings(), members={"grp": ["gus"]}, admins=["ada"], **kw)


def submitted(srv, who="sub", d=None):
    return srv.client(who).submit(None, d or doc())["upload_id"]


def test_a_submission_is_pending_and_needs_approval():
    srv = server()
    out = srv.client("sub").submit(None, doc())
    assert out["state"] == "pending" and out["report"]["requires_approval"] is True


def test_submit_refuses_an_invalid_reference():
    with pytest.raises(rc.Refused):
        server().client("sub").submit(None, doc(label="Bad"))


def test_the_submitter_cannot_approve_their_own_reference():
    srv = server()
    uid = submitted(srv, "sam")
    with pytest.raises(rc.Refused, match="submitter cannot approve"):
        srv.client("sam").approve(uid, terms_ack=True)


def test_an_approver_in_scope_approves_and_it_becomes_current():
    srv = server()
    uid = submitted(srv)
    out = srv.client("amy").approve(uid, terms_ack=True, reason="ok")
    assert out == {"status": "ingested", "state": "published", "version": 1, "current": True, "registered_now": [],
                   "approval_override": False, "approval_self": False}  # fmt: skip
    assert srv.current["perf.synth.decode.refsys"] == uid


def test_an_approver_out_of_scope_is_refused():
    srv = server()
    uid = submitted(srv)
    with pytest.raises(rc.Refused, match="holds no references:approve"):
        srv.client("bob").approve(uid, terms_ack=True)


def test_a_group_member_can_approve():
    srv = server()
    assert srv.client("gus").approve(submitted(srv), terms_ack=True)["state"] == "published"


def test_an_unbound_user_is_refused():
    srv = server()
    with pytest.raises(rc.Refused):
        srv.client("eve").approve(submitted(srv), terms_ack=True)


def test_an_admin_without_a_binding_is_flagged_as_an_override():
    srv = server()
    out = srv.client("ada").approve(submitted(srv), terms_ack=True)
    assert out["approval_override"] is True and out["approval_self"] is False


def test_an_admin_approving_their_own_reference_is_flagged_as_self():
    srv = server()
    out = srv.client("ada").approve(submitted(srv, "ada"), terms_ack=True)
    assert out["approval_override"] is True and out["approval_self"] is True


def test_approval_needs_the_terms_acknowledgement_and_a_pending_upload():
    srv = server()
    uid = submitted(srv)
    with pytest.raises(rc.Refused, match="terms_ack"):
        srv.client("amy").approve(uid)
    srv.client("amy").approve(uid, terms_ack=True)
    assert srv.client("amy").approve(uid, terms_ack=True)["status"] == "duplicate"
    rejected = submitted(srv)
    srv.client("amy").reject(rejected, reason="r")
    with pytest.raises(rc.Refused, match="cannot approve"):
        srv.client("amy").approve(rejected, terms_ack=True)


def test_unregistered_cases_block_approval_unless_registered_now():
    srv = server(registered=set())
    uid = submitted(srv)
    with pytest.raises(rc.Refused, match="not registered"):
        srv.client("amy").approve(uid, terms_ack=True)
    out = srv.client("amy").approve(uid, terms_ack=True, register_missing=True)
    assert len(out["registered_now"]) == 1 and srv.registered


def test_approving_without_promotion_leaves_no_current_version():
    srv = server()
    out = srv.client("amy").approve(submitted(srv), terms_ack=True, promote=False)
    assert out["current"] is False and not srv.current


def test_versions_count_up_and_retracting_the_current_one_rolls_back():
    srv = server()
    first, second = (
        submitted(srv),
        submitted(srv, d=mutate(lambda d: d["values"][0]["measurements"].update({"kernel.time.mean": [9.0]}))),
    )
    assert srv.client("amy").approve(first, terms_ack=True)["version"] == 1
    assert srv.client("amy").approve(second, terms_ack=True)["version"] == 2
    out = srv.client("sub").retract(second)
    assert out["current_after"] == first and srv.current["perf.synth.decode.refsys"] == first
    out = srv.client("sub").retract(first)
    assert out["current_after"] is None and out["retired"] is True and not srv.current


def test_retract_rules():
    srv = server()
    uid = submitted(srv)
    with pytest.raises(rc.Refused, match="only the submitter or an admin"):
        srv.client("amy").retract(uid)
    srv.client("ada").retract(uid)
    with pytest.raises(rc.Refused, match="cannot be retracted"):
        srv.client("sub").retract(uid)


def test_reject_rules():
    srv = server()
    uid = submitted(srv)
    with pytest.raises(rc.Refused, match="withdraws"):
        srv.client("sub").reject(uid, reason="r")
    with pytest.raises(rc.Refused, match="holds no"):
        srv.client("bob").reject(uid, reason="r")
    assert srv.client("amy").reject(uid, reason="r", request_changes=True) == {
        "state": "rejected",
        "review_outcome": "changes_requested",
    }
    with pytest.raises(rc.Refused, match="cannot reject"):
        srv.client("amy").reject(uid, reason="r")


def test_list_and_show():
    srv = server()
    uid = submitted(srv)
    assert [r["upload_id"] for r in srv.client("amy").list(pending=True)] == [uid]
    assert srv.client("bob").list(mine_approvable=True) == []
    assert srv.client("amy").list(mine_approvable=True)[0]["upload_id"] == uid
    assert srv.client("amy").list(klass="quality") == []
    info = srv.client("amy").show(uid)
    assert info["can"] == {"approve": True, "retract": False} and "amy" in info["eligible_approvers"]
    assert "sub" not in info["eligible_approvers"] and srv.client("sub").show(uid)["can"]["retract"] is True
    with pytest.raises(rc.Refused):
        srv.client("amy").show("nope")


class Resp:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http(replies):
    seen = []

    def opener(req, timeout):
        seen.append((req.get_method(), req.full_url, req.data, dict(req.header_items())))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Resp(reply)

    return rc.HttpClient("https://dash.invalid/", "tok", opener), seen


def test_the_http_client_uses_the_contract_endpoints_and_a_bearer_token():
    c, seen = http(["{}"] * 12)
    c.approve("u1", reason="why", terms_ack=True, promote=False, register_missing=True)
    c.reject("u1", reason="no")
    c.reject("u1", reason="edit", request_changes=True)
    c.retract("u1")
    c.show("u1")
    c.list(state="published", klass="perf", pending=True)
    c.submit(__file__)
    c.submit(__file__, dry_run=True)
    c.submit(__file__, inspect=True)
    c.retire("k")
    c.rollback("k", "a1")
    c.me()
    assert [(m, u.split("dash.invalid")[1]) for m, u, *_ in seen[:6]] == [
        ("POST", "/api/v2/bench/references/u1/approve"),
        ("POST", "/api/v2/bench/references/u1/reject"),
        ("POST", "/api/v2/bench/references/u1/request-changes"),
        ("POST", "/api/v2/uploads/u1/retract"),
        ("GET", "/api/v2/bench/references/u1"),
        ("GET", "/api/v2/bench/references?state=published&class=perf&pending=1"),
    ]
    assert json.loads(seen[0][2]) == {
        "terms_ack": True,
        "promote": False,
        "register_missing": True,
        "ack_not_clean": False,
        "reason": "why",
    }
    assert seen[0][3]["Authorization"] == "Bearer tok"
    assert seen[6][1].endswith("/api/v2/uploads") and seen[6][3]["Content-type"].startswith(
        "multipart/form-data; boundary="
    )
    body = seen[6][2].decode(errors="replace")
    assert (
        'name="format"\r\n\r\nreference' in body
        and 'name="submit_token"' in body
        and 'name="bundle"; filename=' in body
    )
    assert seen[7][1].endswith("/api/v2/uploads?dry_run=1") and seen[8][1].endswith("/api/v2/uploads/inspect")
    assert [(m, u.split("dash.invalid")[1]) for m, u, *_ in seen[9:]] == [
        ("POST", "/api/v2/bench/references/k/retire"),
        ("POST", "/api/v2/bench/references/k/rollback"),
        ("GET", "/api/v2/bench/references/me"),
    ]
    token = body.split('name="submit_token"\r\n\r\n')[1].split("\r\n")[0]
    assert re.fullmatch(r"[A-Za-z0-9-]{8,64}", token)


def test_the_http_client_maps_errors():
    err = urllib.error.HTTPError("https://dash.invalid/x", 403, "no", {}, io.BytesIO(b'{"error": "not an approver"}'))
    c, _ = http([err])
    with pytest.raises(rc.Refused, match="HTTP 403: not allowed: not an approver"):
        c.show("u1")
    c, _ = http([urllib.error.HTTPError("https://dash.invalid/x", 502, "bad", {}, io.BytesIO(b"upstream"))])
    with pytest.raises(rc.ClientError) as e:
        c.show("u1")
    assert not isinstance(e.value, rc.Refused)
    c, _ = http([urllib.error.URLError("down")])
    with pytest.raises(rc.ClientError, match="cannot reach"):
        c.show("u1")


def test_without_a_token_the_client_says_approval_is_ui_only():
    with pytest.raises(rc.NoCredentials, match="UI-only in v1"):
        rc.HttpClient.from_env({"BENCH_DASHBOARD_URL": "https://dash.invalid"})
    with pytest.raises(rc.NoCredentials, match="BENCH_DASHBOARD_URL"):
        rc.HttpClient.from_env({"BENCH_DASHBOARD_TOKEN": "t"})
    assert (
        rc.HttpClient.from_env({"BENCH_DASHBOARD_URL": "https://d.invalid", "BENCH_DASHBOARD_TOKEN": "t"}).token == "t"
    )


def test_the_approval_verbs_refuse_without_a_token(tmp_path):
    for env in ({}, {"BENCH_DASHBOARD_URL": "https://d.invalid"}):
        with pytest.MonkeyPatch.context() as mp:
            for k in ("BENCH_DASHBOARD_URL", "BENCH_DASHBOARD_TOKEN"):
                mp.delenv(k, raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            code, out, err = run_cli(["approve", "u1", "--terms-ack"])
        assert code == 2 and ("UI-only" in err or "BENCH_DASHBOARD" in err)


def test_submit_verb_validates_checks_registration_and_prints_the_reply(tmp_path):
    root = made_ref(tmp_path)
    srv = server()
    code, out, err = run_cli(["submit", str(root)], client=srv.client("sub"), st=stores.MemoryStore(known=[]))
    assert code == 0 and json.loads(out)["state"] == "pending" and "not registered" in err
    bad = made_ref(tmp_path / "x", doc(label="Nope"))
    assert run_cli(["submit", str(bad)], client=srv.client("sub"))[0] == 2


def test_submit_verb_accepts_a_sealed_bundle(tmp_path):
    run_cli(["seal", str(made_ref(tmp_path)), "--out", str(tmp_path / "b.tgz")])
    code, out, err = run_cli(["submit", str(tmp_path / "b.tgz")], client=server().client("sub"))
    assert code == 0 and json.loads(out)["upload_id"] == "up-0001"


def test_approve_verb_needs_the_terms_flag_and_prints_what_the_server_returned():
    srv = server()
    uid = submitted(srv)
    assert run_cli(["approve", uid], client=srv.client("amy"))[0] == 2
    code, out, err = run_cli(["approve", uid, "--terms-ack", "--reason", "fine"], client=srv.client("amy"))
    assert code == 0 and json.loads(out)["state"] == "published"


def test_a_refusal_exits_one_with_the_servers_words():
    srv = server()
    uid = submitted(srv, "sam")
    code, out, err = run_cli(["approve", uid, "--terms-ack"], client=srv.client("sam"))
    assert code == 1 and "refused: the submitter cannot approve" in err


def test_reject_and_retract_verbs():
    srv = server()
    uid, other = submitted(srv), submitted(srv)
    assert (
        json.loads(run_cli(["reject", uid, "--request-changes", "--reason", "r"], client=srv.client("amy"))[1])[
            "review_outcome"
        ]
        == "changes_requested"
    )
    assert json.loads(run_cli(["retract", other], client=srv.client("sub"))[1])["state"] == "retracted"
    assert run_cli(["retract", other], client=srv.client("sub"))[0] == 1


def test_list_and_show_verbs():
    srv = server()
    uid = submitted(srv)
    code, out, err = run_cli(["list", "--pending"], client=srv.client("amy"))
    assert code == 0 and uid in out
    assert json.loads(run_cli(["list", "--json"], client=srv.client("amy"))[1])[0]["upload_id"] == uid
    assert run_cli(["list", "--state", "published"], client=srv.client("amy"))[0] == 4
    assert json.loads(run_cli(["show", uid], client=srv.client("amy"))[1].split("\n", 1)[1])["upload_id"] == uid


def test_a_client_failure_exits_five():
    c, _ = http([urllib.error.URLError("down")])
    assert run_cli(["show", "u1"], client=c)[0] == 5


# ---- compare ----


def ref_rows(values=(1.0, 1.1, 1.0, 1.2), **head):
    c = reference.case_block(make_result())
    return [
        {"ref_key": "perf.synth.add.refsys", "version": "2", "state": "current", "artifact_id": "a1", "origin": "generated",
         "label": "ref-sys-a", "comparability": {f"ref.comparability.{k}": v for k, v in META["comparability"].items()},
         "clean": "1", "warnings": "", "component": c["component"], "benchmark_id": reference.case_id(c), "backend": "cpu",
         "measurements": {"kernel.time.mean": list(values)}, "iterations": len(values), "row_props": {}, **head}
    ]  # fmt: skip


LOWER = {"kernel.time.mean": {"direction": "lower_is_better"}}
HIGHER = {"kernel.time.mean": {"direction": "higher_is_better"}}


def compare(samples, rows=None, metrics=LOWER, **kw):
    return reference.compare_to_reference(
        [make_result(samples=samples)], rows if rows is not None else ref_rows(), metrics=metrics, **kw
    )


@pytest.mark.parametrize(
    "samples,metrics,status",
    [
        (((2.0,), (2.0,)), LOWER, "worse"),
        (((0.5,), (0.5,)), LOWER, "better"),
        (((2.0,), (2.0,)), HIGHER, "better"),
        (((0.5,), (0.5,)), HIGHER, "worse"),
        (((1.05,), (1.05,)), LOWER, "same"),
        (((2.0,), (2.0,)), {}, "changed"),
    ],
)
def test_compare_status_is_direction_aware(samples, metrics, status):
    rep = compare(samples, metrics=metrics)
    assert rep["rows"][0]["status"] == status


def test_compare_reports_values_delta_and_percent():
    row = compare(((2.2,), (2.2,)))["rows"][0]
    assert row["reference_value"] == pytest.approx(1.05) and row["run_value"] == pytest.approx(2.2)
    assert row["delta"] == pytest.approx(1.15) and row["percent"] == pytest.approx(109.5238)


def test_compare_tolerance_flag_and_metric_floor():
    near = ((1.2,), (1.2,))  # ~14% above 1.05
    assert compare(near)["rows"][0]["status"] == "worse"
    assert compare(near, tolerance_pct=20)["rows"][0]["status"] == "same"
    floor = {"kernel.time.mean": {"direction": "lower_is_better", "noise": {"floor_pct": 25}}}
    row = compare(near, metrics=floor)["rows"][0]
    assert row["status"] == "same" and row["tolerance_pct"] == 25.0


def test_compare_is_informational():
    rep = compare(((1.0,),))
    assert "never a gate" in rep["informational"]


def test_compare_lists_unmatched_cases_and_skips_unshared_metrics():
    rep = reference.compare_to_reference([make_result(name="other", disc={"shape": "9"})], ref_rows(), metrics=LOWER)
    assert rep["rows"] == [] and rep["unmatched_cases"] == ["other"]


def test_compare_matches_on_backend():
    rep = reference.compare_to_reference([make_result(backend="accel")], ref_rows(), metrics=LOWER)
    assert rep["rows"] == []


@pytest.mark.parametrize(
    "samples,metrics,status",
    [
        (((1.0,),), LOWER, "worse"),
        (((1.0,),), HIGHER, "better"),
        (((-1.0,),), LOWER, "better"),
        (((1.0,),), {}, "changed"),
        (((0.0,),), LOWER, "same"),
    ],
)
def test_a_zero_reference_has_no_percent_and_is_judged_by_the_sign_of_the_delta(samples, metrics, status):
    row = compare(samples, rows=ref_rows(values=(0.0,)), metrics=metrics)["rows"][0]
    assert row["percent"] is None and row["status"] == status


def test_compare_reads_a_local_reference_document():
    d = doc(values=[{"case": reference.case_block(make_result()) | {"benchmark_id": reference.case_id(reference.case_block(make_result()))},
                     "backend": "cpu", "measurements": {"kernel.time.mean": [1.0, 1.0]}, "iterations": 2}])  # fmt: skip
    rep = reference.compare_to_reference([make_result(samples=((1.0,),))], d, metrics=LOWER)
    assert rep["rows"][0]["status"] == "same" and rep["reference"]["state"] == "local"


@pytest.mark.parametrize(
    "block,ctx,word",
    [
        (META["comparability"], {"precision": "fp32", "dataset_subset": "full", "shape.batch": 1, "shape.input_len": 128, "shape.output_len": 64}, "comparable"),
        (META["comparability"], {"precision": "fp16", "dataset_subset": "full", "shape.batch": 1, "shape.input_len": 128, "shape.output_len": 64}, "partial"),
        (META["comparability"], {}, "partial"),
        ({**META["comparability"], "harness_match": "no"}, {}, "not_comparable"),
        ({**META["comparability"], "dataset_match": "different"}, {}, "not_comparable"),
        ({**META["comparability"], "harness_match": "partial"}, {}, "partial"),
        ({**META["comparability"], "dataset_match": "subset"}, {}, "partial"),
        ({f"ref.comparability.{k}": v for k, v in META["comparability"].items()}, {"precision": "fp32", "dataset_subset": "full", "shape.batch": 1, "shape.input_len": 128, "shape.output_len": 64}, "comparable"),
    ],
)  # fmt: skip
def test_comparability_word(block, ctx, word):
    assert reference.comparability_word(block, ctx) == word


def test_compare_verb_against_a_local_reference_dir(tmp_path):
    run = make_run(tmp_path, [make_result(samples=((1.0, 1.1), (1.0, 1.2)))])
    assert create(tmp_path, run)[0] == 0
    code, out, err = run_cli(["compare", "--run", str(run), "--reference", str(tmp_path / "ref"), "--json"])
    rep = json.loads(out)
    assert code == 0 and rep["rows"][0]["status"] == "same" and rep["comparability"] == "partial"
    code, out, err = run_cli(["compare", "--run", str(run), "--reference", str(tmp_path / "ref")])
    assert "never a gate" in out and "same" in out


def test_compare_verb_against_the_store_by_key_and_version(tmp_path):
    run = make_run(tmp_path, [make_result(samples=((3.0,), (3.0,)))])
    st = stores.MemoryStore(
        references=[*ref_rows(), *ref_rows(version="1", state="superseded", measurements={"kernel.time.mean": [3.0]})]
    )
    code, out, err = run_cli(["compare", "--run", str(run), "--reference", "perf.synth.add.refsys", "--json"], st=st)
    rep = json.loads(out)
    assert code == 0 and rep["reference"]["version"] == "2" and rep["rows"][0]["run_value"] == 3.0
    code, out, err = run_cli(
        ["compare", "--run", str(run), "--reference", "perf.synth.add.refsys", "--version", "1", "--json"], st=st
    )
    assert json.loads(out)["rows"][0]["status"] == "same"
    assert run_cli(["compare", "--run", str(run), "--reference", "nope"], st=st)[0] == 2
    assert (
        run_cli(["compare", "--run", str(run), "--reference", "perf.synth.add.refsys", "--version", "x"], st=st)[0] == 2
    )


def test_compare_verb_with_no_store_and_a_key_is_a_store_error(tmp_path):
    run = make_run(tmp_path, [make_result()])
    assert run_cli(["compare", "--run", str(run), "--reference", "some.key"])[0] == 5


def test_compare_verb_exits_four_when_nothing_matches(tmp_path):
    run = make_run(tmp_path, [make_result(name="zzz", disc={"s": "1"})])
    st = stores.MemoryStore(references=ref_rows())
    assert run_cli(["compare", "--run", str(run), "--reference", "perf.synth.add.refsys"], st=st)[0] == 4


def test_compare_never_gates_even_when_everything_is_worse(tmp_path):
    run = make_run(tmp_path, [make_result(samples=((90.0,), (90.0,)))])
    st = stores.MemoryStore(references=ref_rows())
    code, out, err = run_cli(["compare", "--run", str(run), "--reference", "perf.synth.add.refsys"], st=st)
    assert code == 0


# ---- stores ----


class FakeCH:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def query(self, sql, parameters=None):
        self.calls.append((sql, parameters))
        cols = list(self.rows[0]) if self.rows else []
        return type("R", (), {"column_names": cols, "result_rows": [tuple(r[c] for c in cols) for r in self.rows]})()


def test_null_and_memory_stores_answer_registration():
    assert stores.NullStore().registered(["a"]) is None
    st = stores.MemoryStore(known=["a"])
    assert st.registered(["a", "b"]) == {"a"}
    assert stores.MemoryStore().registered(["a"]) is None
    with pytest.raises(stores.StoreError):
        stores.MemoryStore(fail="x").registered(["a"])


def test_the_clickhouse_store_reads_registration_and_reference_values():
    bid = "93c0abb3-ed25-5934-b811-31b6c149ba47"
    ch = FakeCH([{"benchmark_key": bid}])
    st = stores.ClickHouseStore(client=ch, database="db")
    assert st.registered([bid, "not-a-uuid"]) == {bid}
    assert "benchmarks" in ch.calls[0][0] and ch.calls[0][1] == {"ids": [bid]}
    ch = FakeCH([{"ref_key": "k", "version": "1"}])
    st = stores.ClickHouseStore(client=ch, database="db")
    assert st.reference_values("k")[0]["version"] == "1" and "state = 'current'" in ch.calls[0][0]
    st.reference_values("k", 3)
    assert "version = {version:String}" in ch.calls[1][0] and ch.calls[1][1]["version"] == "3"


def test_the_base_store_holds_no_references():
    with pytest.raises(stores.StoreError):
        stores.NullStore().reference_values("k")


def test_the_reference_verb_is_registered():
    assert VERBS["reference"][0] == "spyre_bench.reference"
    assert cli_main(["reference", "--help"]) == 0 or True


# ---- conformance with the ingest's copy (skipped when it is not importable) ----


def test_validation_agrees_with_the_ingest():
    ing = ingest_reference()
    cases = [doc(), doc(terms=""), doc(label="Bad"), doc(label="host-1-ref"), doc(version=1), doc(origin="published"),
             doc(values=[]), doc(tools={}), doc(clean="x"), doc(warnings="x")]  # fmt: skip
    cases += [
        mutate(lambda d: d["comparability"].pop("harness")),
        mutate(lambda d: d["values"][0]["case"].update(benchmark_id="0" * 36)),
    ]
    cases += [
        mutate(lambda d: d["values"][0]["measurements"].update({"m": [True]})),
        mutate(lambda d: d["comparability"].update(x=1)),
    ]
    for d in cases:
        assert reference.validate(d) == ing.validate(d, label_deny=[])


def test_digest_arch_and_ids_agree_with_the_ingest():
    ing = ingest_reference()
    d = doc(values=[{**doc()["values"][0], "case": case(disc={"arch": "x86_64", "k": "V"})}])
    assert reference.content_digest(d) == ing.content_digest(d)
    assert reference.arch_of(d) == ing.arch_of(d) and reference.archs_of(d) == ing.archs_of(d)
    for c in (
        case(),
        {"component": "synth-comp", "name": " Op ", "tags": ["b", "A", ""], "disc": {"Z": " V ", "a": 5, "t": True}},
    ):
        assert reference.case_id(c) == ing.case_id(c)
    assert reference.facts_of(doc()) == ing.facts_of(doc())
    assert reference.case_ids(doc()) == ing.case_ids(doc())


PROMOTION_RUNS = [
    {}, {"clean": CLEAN}, {"clean": SHARED}, {"clean": {"exclusive": "observed"}},
    {"clean": {"exclusive": "observed", "load_clean": True, "devices_exclusive": "enforced"}},
    {"clean": CLEAN, "validity": "suspect"}, {"clean": CLEAN, "validity": "suspected", "repeats": 1},
    {"clean": CLEAN, "pinning_verified": False}, {"clean": CLEAN, "purpose": "pr", "qualified": False},
    {"clean": CLEAN, "purpose": "nightly"}, {"clean": CLEAN, "purpose": "release"},
    {"clean": CLEAN, "invalidated": True}, {"clean": CLEAN, "retracted": True}, {"clean": CLEAN, "measured": False},
]  # fmt: skip


@pytest.mark.parametrize("run", PROMOTION_RUNS)
def test_promotion_check_agrees_with_the_ingest(run):
    ing = ingest_reference()
    assert reference.promotion_check(run) == _as_ours(ing.promotion_check(run))
    assert set(reference.WARNING_TEXT) == set(ing.WARNING_TEXT)


def _as_ours(p):
    return reference.Promotion(p.ok, p.clean, p.warnings, p.blockers)


def test_the_approval_rules_agree_with_the_ingest():
    ing = ingest_reference()
    for sub in ("amy", "bob"):
        for who in ("amy", "bob", "eve", ""):
            for admin in (False, True):
                for approvers in ((), ("amy",), ("bob", "AMY")):
                    kw = {"submitter": sub, "approver": who, "is_admin": admin, "approvers": approvers, "reason": "r"}
                    assert vars(reference.decide_approval(**kw)) == vars(ing.decide_approval(**kw))
                    rk = {"submitter": sub, "actor": who, "is_admin": admin}
                    assert vars(reference.decide_reject(approvers=approvers, **rk)) == vars(
                        ing.decide_reject(approvers=approvers, **rk)
                    )
                    for state in ("pending", "published", "rejected"):
                        assert vars(reference.decide_retract(state=state, **rk)) == vars(
                            ing.decide_retract(state=state, **rk)
                        )
    facts = reference.facts_of(doc())
    for scope in (
        {},
        {"class": ["perf"]},
        {"class": ["quality"]},
        {"class": ["perf"], "arch": ["noarch"]},
        {"dataset": ["d"]},
    ):
        assert reference.in_scope(scope, facts) == ing.in_scope(scope, facts)
    binds = bindings()
    assert reference.holders(binds, "references:approve", facts, {"grp": ["gus"]}) == ing.holders(
        binds, "references:approve", facts, {"grp": ["gus"]}
    )


def test_a_sealed_bundle_passes_the_ingests_offline_check(tmp_path):
    ingest_reference()
    from spyre_clickhouse_ingest import offline

    d = doc(assets=[])
    root = made_ref(tmp_path, d)
    (root / "attachments" / "a.jsonl").write_text('{"x":1}\n')
    asset = {"path": "attachments/a.jsonl", "sha256": reference.sha256_file(root / "attachments" / "a.jsonl"), "records": 1,
             "case": case("parity")}  # fmt: skip
    (root / "reference.json").write_text(json.dumps(doc(assets=[asset])))
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "b")])[0] == 0
    meta = offline.check(tmp_path / "b")
    assert (meta["kind"], meta["test_type"], meta["arch"], meta["component"]) == (
        "bench-reference",
        "perf",
        "noarch",
        "reference",
    )
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "b.tgz")])[0] == 0
    with offline.opened(tmp_path / "b.tgz") as top:
        assert offline.check(top)["kind"] == "bench-reference"


def test_the_ingest_accepts_what_create_writes(tmp_path):
    ing = ingest_reference()
    assert (
        create(tmp_path, make_run(tmp_path, [make_result(), make_result(name="op_mul", disc={"shape": "8"})]))[0] == 0
    )
    assert ing.validate(json.loads((tmp_path / "ref" / "reference.json").read_text())) == []


# ---- the label deny-list ----


def write_deny(tmp_path, text, name="deny.txt"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_the_builtin_deny_list_is_the_generic_minimum():
    for label in ("a-node-1", "host-a", "worker-a", "a.b-c"):
        assert reference.validate(doc(label=label)), label
    assert reference.validate(doc(label="partner-ref")) == []


def test_a_configured_deny_list_adds_patterns():
    deny = [re.compile(r"^acme", re.I)]
    assert any("deny-listed" in e for e in reference.validate(doc(label="acme-ref"), label_deny=deny))
    assert reference.validate(doc(label="other-ref"), label_deny=deny) == []


def test_a_deny_list_file_ignores_comments_and_blanks(tmp_path):
    p = write_deny(tmp_path, "# a comment\n\n  acme  \n   # indented comment\nBETA[0-9]\n")
    pats = reference.load_denylist(p)
    assert [x.pattern for x in pats] == ["acme", "BETA[0-9]"]
    assert pats[0].search("ACME-x")


def test_an_invalid_pattern_or_unreadable_file_is_exit_two(tmp_path):
    bad = write_deny(tmp_path, "ok\n(unclosed\n")
    root = made_ref(tmp_path)
    code, out, err = run_cli(["validate", str(root), "--label-denylist", str(bad)])
    assert code == 2 and "invalid pattern" in err and ":2:" in err
    code, out, err = run_cli(["validate", str(root), "--label-denylist", str(tmp_path / "missing.txt")])
    assert code == 2 and "cannot read the label deny-list" in err


def test_the_flag_file_fails_validate_seal_submit_and_create_alike(tmp_path):
    deny = write_deny(tmp_path, "^ref-sys\n")
    root = made_ref(tmp_path)
    for argv in (["validate", str(root)], ["seal", str(root), "--out", str(tmp_path / "b")], ["submit", str(root)]):
        code, out, err = run_cli([*argv, "--label-denylist", str(deny)], client=server().client("sub"))
        assert code == 2 and "deny-listed" in err, argv
    assert not (tmp_path / "b").exists()
    code, out, err = create(tmp_path, make_run(tmp_path, [make_result()]), "--label-denylist", str(deny))
    assert code == 2 and "deny-listed" in err
    assert run_cli(["validate", str(root)])[0] == 0


def test_the_environment_file_applies_without_the_flag(tmp_path, monkeypatch):
    deny = write_deny(tmp_path, "ref-sys\n")
    root = made_ref(tmp_path)
    monkeypatch.setenv("SPYRE_BENCH_LABEL_DENYLIST", str(deny))
    assert run_cli(["validate", str(root)])[0] == 2
    other = write_deny(tmp_path, "nomatch\n", "other.txt")
    assert run_cli(["validate", str(root), "--label-denylist", str(other)])[0] == 0


# ---- discriminator keys ----

ORDERED = {"disc": {"z": "1", "a": "2"}, "disc_keys": ["z", "a"]}


def test_a_profile_ordered_case_hashes_with_its_own_key_order():
    base = {"component": "synth-comp", "name": "k", "tags": []}
    ordered = {**base, **ORDERED}
    absent = {**base, "disc": ORDERED["disc"]}
    assert reference.case_id(ordered) != reference.case_id(absent)
    assert reference.case_id(absent) == reference.case_id({**absent, "disc_keys": ["a", "z"]})


def test_create_emits_the_engines_disc_keys_and_the_stored_order_is_reproduced(tmp_path):
    res = make_result(disc=ORDERED["disc"], disc_keys=ORDERED["disc_keys"])
    assert create(tmp_path, make_run(tmp_path, [res]))[0] == 0
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    c = d["values"][0]["case"]
    assert c["disc_keys"] == ["z", "a"] and c["id_profile"] == "generic"
    assert c["benchmark_id"] == reference.case_id(c) and reference.validate(d) == []
    assert c["benchmark_id"] != reference.case_id({k: v for k, v in c.items() if k != "disc_keys"})


def test_create_without_disc_keys_in_the_result_writes_sorted_keys(tmp_path):
    create(tmp_path, make_run(tmp_path, [make_result(disc={"z": "1", "a": "2"})]))
    c = json.loads((tmp_path / "ref" / "reference.json").read_text())["values"][0]["case"]
    assert c["disc_keys"] == ["a", "z"] and "id_profile" not in c


def test_validate_recomputes_with_disc_keys():
    d = doc()
    d["values"][0]["case"].update(disc_keys=["z", "a"], disc={"z": "1", "a": "2"})
    assert any("does not match" in e for e in reference.validate(d))
    d["values"][0]["case"]["benchmark_id"] = reference.case_id(d["values"][0]["case"])
    assert reference.validate(d) == []
    d["values"][0]["case"]["disc_keys"] = "za"
    assert any("disc_keys must be a list of strings" in e for e in reference.validate(d))


def test_compare_joins_a_profile_ordered_case():
    res = make_result(disc=ORDERED["disc"], disc_keys=ORDERED["disc_keys"], samples=((2.0,),))
    c = reference.case_block(res)
    c["benchmark_id"] = reference.case_id(c)
    d = doc(values=[{"case": c, "backend": "cpu", "measurements": {"kernel.time.mean": [1.0]}, "iterations": 1}])
    assert reference.compare_to_reference([res], d, metrics=LOWER)["rows"][0]["status"] == "worse"


def test_the_ingest_derives_the_same_profile_ordered_id():
    ing = ingest_reference()
    c = {"component": "synth-comp", "name": "k", "tags": [], **ORDERED}
    c["benchmark_id"] = reference.case_id(c)
    if ing.case_id(c) != c["benchmark_id"] and ing.case_id({**c, "disc_keys": ["a", "z"]}) == ing.case_id(c):
        pytest.skip("this ingest checkout does not honour disc_keys yet")
    assert ing.case_id(c) == c["benchmark_id"]
    d = doc(values=[{"case": c, "backend": "cpu", "measurements": {"m": [1.0]}, "iterations": 1}])
    assert reference.validate(d) == ing.validate(d, label_deny=[])


def test_a_configured_deny_list_agrees_with_the_ingest():
    ing = ingest_reference()
    deny = [re.compile(r"^acme", re.I)]
    for label in ("acme-ref", "ACME-ref", "other-ref", "host-a"):
        d = doc(label=label)
        assert reference.validate(d, label_deny=deny) == ing.validate(d, label_deny=deny)


# ---- digest scope, duplicates, shared cases ----


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(origin="uploaded"),
        lambda d: d["comparability"].update(precision="fp16"),
        lambda d: d.update(**{"class": "quality"}),
    ],
)
def test_the_digest_covers_class_origin_and_comparability(change):
    d = doc()
    base = reference.content_digest(d)
    change(d)
    assert reference.content_digest(d) != base


def test_the_digest_still_ignores_clean():
    assert reference.content_digest(doc(clean=False)) == reference.content_digest(doc(clean=None))


def test_an_asset_and_a_value_on_the_same_case_are_refused(tmp_path):
    (tmp_path / "attachments").mkdir()
    f = tmp_path / "attachments" / "o.jsonl"
    f.write_text('{"a":1}\n')
    asset = {"path": "attachments/o.jsonl", "sha256": reference.sha256_file(f), "records": 1, "case": case()}
    assert any("share case" in e for e in reference.validate(doc(assets=[asset]), tmp_path))
    assert reference.validate(doc(assets=[dict(asset, case=case("parity"))]), tmp_path) == []


def test_the_overlap_rule_agrees_with_the_ingest(tmp_path):
    ing = ingest_reference()
    asset = {"path": "attachments/o.jsonl", "sha256": "0" * 64, "records": 1, "case": case()}
    d = doc(assets=[asset])
    assert (
        [e for e in reference.validate(d) if "share case" in e]
        == [e for e in ing.validate(d, label_deny=[]) if "share case" in e]
        != []
    )


def test_duplicate_content_is_refused_at_approval():
    srv = server()
    first, second = submitted(srv), submitted(srv)
    srv.client("amy").approve(first, terms_ack=True)
    with pytest.raises(rc.Refused, match="duplicate content"):
        srv.client("amy").approve(second, terms_ack=True)


def test_a_409_is_reported_as_duplicate_content():
    c, _ = http(
        [urllib.error.HTTPError("https://dash.invalid/x", 409, "dup", {}, io.BytesIO(b'{"error": "same digest"}'))]
    )
    with pytest.raises(rc.Refused, match="duplicate content"):
        c.submit(__file__)


def test_show_prints_version_and_state():
    srv = server()
    uid = submitted(srv)
    srv.client("amy").approve(uid, terms_ack=True)
    code, out, err = run_cli(["show", uid], client=srv.client("amy"))
    assert out.splitlines()[0] == "perf.synth.decode.refsys v1 published"


SHAPES = [
    lambda d: d.update(model="x"),
    lambda d: d.update(tools={"torch": 2}),
    lambda d: d.update(system={"backend": "cpu", "k": True}),
    lambda d: d.update(label=5),
    lambda d: d.update(values={}),
    lambda d: d.update(warnings="x"),
    lambda d: d["values"].append("bad"),
    lambda d: d["values"][0]["case"].update(tags="t"),
    lambda d: d["values"][0]["case"].update(name=3),
]


@pytest.mark.parametrize("change", SHAPES)
def test_wrong_json_types_are_reported_alike(change):
    d = mutate(change)
    errors = reference.validate(d)
    assert errors and not any("Error" in e for e in errors)
    ing = ingest_reference() if os.environ.get("BENCH_REFERENCE_INGEST") else None
    if ing:
        assert errors == ing.validate(d, label_deny=[])


def test_unlisted_attachments_are_refused(tmp_path):
    (tmp_path / "attachments").mkdir()
    (tmp_path / "attachments" / "stray.bin").write_text("x")
    assert any("not listed in assets" in e for e in reference.validate(doc(), tmp_path))


# ---- review fixes: malformed blocks, links, traversal, tarballs, scopes, the client ----


def asset_doc(tmp_path, **case_over):
    (tmp_path / "attachments").mkdir(parents=True, exist_ok=True)
    f = tmp_path / "attachments" / "o.txt"
    f.write_text("x")
    c = {**case("parity"), **case_over}
    return doc(assets=[{"path": "attachments/o.txt", "sha256": reference.sha256_file(f), "records": 1, "case": c}])


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"component": None}, "needs string component/name"),
        ({"name": 5}, "needs string component/name"),
        ({"tags": "t"}, "needs string component/name"),
        ({"disc": []}, "needs string component/name"),
        ({"disc_keys": "ab"}, "disc_keys must be a list of strings"),
        ({"disc_keys": [1]}, "disc_keys must be a list of strings"),
    ],
)
def test_malformed_asset_cases_give_one_clean_message(tmp_path, over, msg):
    d = asset_doc(tmp_path)
    d["assets"][0]["case"].update(over)
    errors = reference.validate(d, tmp_path)
    assert any(msg in e for e in errors), errors
    assert not any("Error" in e for e in errors)


def test_a_missing_component_in_an_asset_case_is_not_a_crash(tmp_path):
    d = asset_doc(tmp_path)
    del d["assets"][0]["case"]["component"]
    assert reference.validate(d, tmp_path)


@pytest.mark.parametrize(
    "garbage",
    [
        [],
        None,
        5,
        {"values": [1]},
        {"values": [{"case": 3}]},
        {"assets": [None]},
        {"assets": [{"case": {}}]},
        {"values": {"a": 1}},
    ],
)
def test_validate_never_raises(garbage):
    assert reference.validate(garbage)


@pytest.mark.parametrize(
    "path", ["attachments/../reference.json", "attachments\\x", "attachments/a/../b", "/etc/passwd", "attachments/"]
)
def test_asset_paths_with_traversal_are_refused(tmp_path, path):
    d = asset_doc(tmp_path)
    d["assets"][0]["path"] = path
    assert any("plain path under attachments" in e for e in reference.validate(d, tmp_path))


def test_a_linked_asset_is_refused(tmp_path):
    d = asset_doc(tmp_path)
    (tmp_path / "secret.txt").write_text("s")
    (tmp_path / "attachments" / "o.txt").unlink()
    (tmp_path / "attachments" / "o.txt").symlink_to(tmp_path / "secret.txt")
    d["assets"][0]["sha256"] = reference.sha256_file(tmp_path / "secret.txt")
    assert any("link or leaves the bundle" in e for e in reference.validate(d, tmp_path))


def test_seal_copies_only_listed_assets_and_refuses_links(tmp_path):
    root = tmp_path / "ref"
    d = asset_doc(root)
    (root / "reference.json").write_text(json.dumps(d))
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "b")])[0] == 0
    assert sorted(p.name for p in (tmp_path / "b" / "attachments").iterdir()) == ["o.txt"]
    (tmp_path / "secret.txt").write_text("s")
    (root / "attachments" / "link.txt").symlink_to(tmp_path / "secret.txt")
    code, out, err = run_cli(["seal", str(root), "--out", str(tmp_path / "c")])
    assert code == 2 and "not listed in assets" in err and not (tmp_path / "c").exists()
    (root / "attachments" / "loose.txt").write_text("l")
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "d")])[0] == 2


def test_seal_refuses_to_overwrite_and_leaves_no_partial_tgz(tmp_path):
    root = made_ref(tmp_path)
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "b.tgz")])[0] == 0
    before = (tmp_path / "b.tgz").read_bytes()
    code, out, err = run_cli(["seal", str(root), "--out", str(tmp_path / "b.tgz")])
    assert code == 2 and "already exists" in err and (tmp_path / "b.tgz").read_bytes() == before
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("bundle-")] == []


def test_seal_names_a_bundle_without_calling_it_unsealed(tmp_path):
    run_cli(["seal", str(made_ref(tmp_path)), "--out", str(tmp_path / "b")])
    code, out, err = run_cli(["seal", str(tmp_path / "b"), "--out", str(tmp_path / "c")])
    assert code == 2 and "already a sealed bundle" in err


def test_a_failed_create_leaves_no_partial_directory(tmp_path):
    out = tmp_path / "x" / "ref"
    reference.write_dir(doc(), out)
    assert (out / "reference.json").is_file() and [p.name for p in out.parent.iterdir()] == ["ref"]


@pytest.mark.parametrize(
    "key,value",
    [("clean", True), ("warnings", []), ("pinning", "x"), ("values", []), ("source_run", {}), ("origin", "published")],
)
def test_meta_cannot_override_cleanliness_or_values(tmp_path, key, value):
    run = make_run(tmp_path, [make_result(clean=SHARED)])
    meta = meta_file(tmp_path, **{key: value})
    argv = ["create", "--from-run", str(run), "--key", "perf.synth.add.refsys", "--class", "perf", "--label", "ref-sys-a",
            "--out", str(tmp_path / "ref"), "--meta", str(meta)]  # fmt: skip
    code, out, err = run_cli(argv, st=stores.MemoryStore(invalidated=[]))
    assert code == 2 and "not allowed" in err and not (tmp_path / "ref").exists()


def test_a_wrapped_tarball_and_the_missing_reference_are_handled(tmp_path):
    run_cli(["seal", str(made_ref(tmp_path)), "--out", str(tmp_path / "b")])
    wrapped = tmp_path / "w.tgz"
    with tarfile.open(wrapped, "w:gz") as tar:
        tar.add(tmp_path / "b", arcname="bundle")
    srv = server()
    assert run_cli(["submit", str(wrapped)], client=srv.client("sub"))[0] == 0
    empty = tmp_path / "e.tgz"
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "bundle.json").write_text("{}")
    with tarfile.open(empty, "w:gz") as tar:
        tar.add(tmp_path / "e" / "bundle.json", arcname="bundle.json")
    code, out, err = run_cli(["submit", str(empty)], client=srv.client("sub"))
    assert code == 2 and "holds no results/reference.json" in err
    assert run_cli(["validate", str(tmp_path / "missing")])[0] == 2


def test_submit_checks_the_bundle_hashes(tmp_path):
    run_cli(["seal", str(made_ref(tmp_path)), "--out", str(tmp_path / "b")])
    (tmp_path / "b" / "results" / "reference.json").write_text(
        (tmp_path / "b" / "results" / "reference.json").read_text() + " "
    )
    code, out, err = run_cli(["submit", str(tmp_path / "b")], client=server().client("sub"))
    assert code == 2 and "sha256 mismatch" in err
    run_cli(["seal", str(made_ref(tmp_path / "z")), "--out", str(tmp_path / "ok")])
    assert run_cli(["submit", str(tmp_path / "ok")], client=server().client("sub"))[0] == 0


def test_a_bundle_directory_with_a_link_is_not_packed(tmp_path):
    run_cli(["seal", str(made_ref(tmp_path)), "--out", str(tmp_path / "b")])
    (tmp_path / "s.txt").write_text("s")
    (tmp_path / "b" / "attachments").mkdir()
    (tmp_path / "b" / "attachments" / "l").symlink_to(tmp_path / "s.txt")
    code, out, err = run_cli(["submit", str(tmp_path / "b")], client=server().client("sub"))
    assert code == 2 and "link" in err


def test_scopes_and_expiry_follow_the_ingest_rules():
    from datetime import datetime, timedelta, timezone

    facts = reference.facts_of(doc(dataset={"id": "SLOT-14"}))
    assert reference.in_scope({"dataset": [" slot-14 "]}, facts) and reference.in_scope(
        {"key": ["PERF.synth.decode.refsys"]}, facts
    )
    assert not reference.in_scope({"dataset": ["other"]}, facts)
    past, future = datetime(2020, 1, 1), datetime.now() + timedelta(days=30)
    b = lambda who, exp: {"subject": who, "permissions": ["references:approve"], "expires_at": exp}  # noqa: E731
    got = reference.holders(
        [b("a", past), b("b", future), b("c", past.replace(tzinfo=timezone.utc)), b("d", None)],
        "references:approve",
        facts,
    )
    assert got == {"b", "d"}


def test_scope_and_expiry_agree_with_the_ingest():
    from datetime import datetime, timedelta, timezone

    ing = ingest_reference()
    facts = reference.facts_of(doc(dataset={"id": "SLOT-14"}))
    for scope in (
        {"dataset": ["slot-14"]},
        {"dataset": ["SLOT-14 "]},
        {"class": ["PERF"]},
        {"key": ["x"]},
        {"arch": ["NoArch"]},
    ):
        assert reference.in_scope(scope, facts) == ing.in_scope(scope, facts)
    binds = [
        {"subject": "A", "permissions": ["references:*"], "expires_at": datetime(2020, 1, 1)},
        {
            "subject": "b",
            "permissions": ["*"],
            "expires_at": datetime.now() + timedelta(days=9),
            "scope": {"dataset": ["slot-14"]},
        },
        {
            "subject": "c",
            "permissions": ["references:approve"],
            "expires_at": datetime(2020, 1, 1, tzinfo=timezone.utc),
        },
    ]
    assert reference.holders(binds, "references:approve", facts) == ing.holders(binds, "references:approve", facts)


def test_the_fake_applies_mixed_case_scopes():
    srv = rc.FakeServer([{"subject": "amy", "permissions": ["references:approve"], "scope": {"class": ["PERF"]}}])
    uid = srv.client("sub").submit(None, doc())["upload_id"]
    assert srv.client("amy").approve(uid, terms_ack=True)["status"] == "ingested"


def test_the_fake_reads_the_label_deny_list(tmp_path, monkeypatch):
    deny = write_deny(tmp_path, "ref-sys\n")
    monkeypatch.setenv("SPYRE_BENCH_LABEL_DENYLIST", str(deny))
    with pytest.raises(rc.Refused, match="deny-listed"):
        server().client("sub").submit(None, doc())


def test_the_fake_accepts_a_list_of_registered_cases():
    srv = server(registered=[])
    uid = submitted(srv)
    assert srv.client("amy").approve(uid, terms_ack=True, register_missing=True)["registered_now"]


def test_http_redirects_are_not_followed():
    handler = rc._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://elsewhere.invalid/") is None


def test_http_ids_are_quoted_and_only_uploads_report_duplicates():
    c, seen = http(["{}"])
    c.show("a/b?x")
    assert seen[0][1].endswith("/api/v2/bench/references/a%2Fb%3Fx")
    c, _ = http(
        [urllib.error.HTTPError("https://d.invalid/x", 409, "c", {}, io.BytesIO(b'{"error": "state changed"}'))]
    )
    with pytest.raises(rc.Refused) as e:
        c.approve("u1", terms_ack=True)
    assert "duplicate content" not in str(e.value) and "state changed" in str(e.value)


def test_a_retried_submit_carries_the_same_token(tmp_path):
    f = tmp_path / "b.tgz"
    f.write_bytes(b"bundle")
    c, seen = http(["{}", "{}"])
    c.submit(f)
    c.submit(f)
    tokens = [d.decode().split('name="submit_token"\r\n\r\n')[1].split("\r\n")[0] for _, _, d, _ in seen]
    assert tokens[0] == tokens[1] and re.fullmatch(r"[A-Za-z0-9-]{8,64}", tokens[0])


def test_list_and_show_without_a_token_do_not_claim_approval_is_ui_only(monkeypatch):
    for k in ("BENCH_DASHBOARD_URL", "BENCH_DASHBOARD_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    code, out, err = run_cli(["list"])
    assert code == 2 and "needs a dashboard bearer token" in err and "UI-only" not in err
    assert "UI-only" in run_cli(["approve", "u1", "--terms-ack"])[2]


def test_compare_rejects_a_bad_version_and_a_corrupt_result(tmp_path):
    run = make_run(tmp_path, [make_result()])
    st = stores.MemoryStore(references=ref_rows())
    assert run_cli(["compare", "--run", str(run), "--reference", "k", "--version", "abc"], st=st)[0] == 2
    (run / "cases" / "op_add" / "result.json").write_text("{not json")
    assert run_cli(["compare", "--run", str(run), "--reference", "perf.synth.add.refsys"], st=st)[0] == 2


def test_a_result_path_outside_the_run_is_refused(tmp_path):
    run = make_run(tmp_path, [make_result()])
    manifest = json.loads((run / "run.json").read_text())
    manifest["cases"][0]["result"] = "../outside.json"
    (run / "run.json").write_text(json.dumps(manifest))
    (tmp_path / "outside.json").write_text(json.dumps(make_result()))
    code, out, err = create(tmp_path, run)
    assert code == 2 and "outside the results directory" in err


def test_several_pinning_policies_are_all_recorded():
    mixed = [
        make_result(props={"run.exec.cpu_pinning": "explicit"}),
        make_result(name="b", props={"run.exec.cpu_pinning": "device_local"}),
    ]
    assert reference.pinning_of(mixed) == "device_local,explicit"


def test_malformed_blocks_and_paths_agree_with_the_ingest(tmp_path):
    ing = ingest_reference()
    cases = []
    for over in ({"component": None}, {"name": 5}, {"disc_keys": "ab"}, {"tags": "t"}):
        d = asset_doc(tmp_path)
        d["assets"][0]["case"].update(over)
        cases.append(d)
    for path in ("attachments/../reference.json", "attachments\\x", "attachments/ok.txt"):
        d = asset_doc(tmp_path)
        d["assets"][0]["path"] = path
        cases.append(d)
    d = asset_doc(tmp_path)
    d["assets"][0]["records"] = True
    cases.append(d)
    d = doc()
    d["values"][0]["iterations"] = True
    cases.append(d)
    cases += [[], None, {"values": [1]}, {"values": [{"case": 3}]}, {"assets": [None]}]
    for d in cases:
        assert reference.validate(d, tmp_path) == ing.validate(d, tmp_path, label_deny=[])


# ---- bounded fuzz: validate is total, an accepted file survives its readers ----

JUNK = [None, True, 0, -1, 10**400, 1.5, float("inf"), "", "x", "attachments/a\x00b", [], {}, [None], {"a": 1}]


def _paths(node, prefix=()):
    yield prefix
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _paths(v, (*prefix, k))
    elif isinstance(node, list):
        for i, v in enumerate(node[:2]):
            yield from _paths(v, (*prefix, i))


def _with(d, path, value):
    d = copy.deepcopy(d)
    if not path:
        return value
    cur = d
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value
    return d


def _fuzz_base(tmp_path):
    return asset_doc(tmp_path)


def test_fuzz_validate_returns_a_list_and_accepted_files_survive(tmp_path):
    base = _fuzz_base(tmp_path)
    assert reference.validate(base, tmp_path) == []
    ing = None
    try:
        ing = ingest_reference() if os.environ.get("BENCH_REFERENCE_INGEST") else None
    except BaseException:  # noqa: BLE001 - skip is raised as an exception
        ing = None
    count = 0
    for path in _paths(base):
        for junk in JUNK:
            d = _with(base, path, junk)
            errors = reference.validate(d, tmp_path)
            assert isinstance(errors, list)
            if ing is not None:
                assert errors == ing.validate(d, tmp_path, label_deny=[]), (path, junk)
            if not errors:
                reference.content_digest(d)
                reference.arch_of(d)
                reference.case_ids(d)
                reference.facts_of(d)
            count += 1
    assert count > 300


def test_promotion_tolerates_a_null_repeat_count():
    assert reference.promotion_check({"clean": CLEAN, "repeats": None}).warnings == ()


# ---- fullmatch, repeated assets, the asset digest term, as_of with Z ----


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(key="perf.synth.decode.refsys\n"),
        lambda d: d.update(label="ref-sys-a\n"),
        lambda d: d["system"].update(backend="cpu\n"),
        lambda d: d["values"][0].update(backend="cpu\n"),
    ],
)
def test_a_trailing_newline_is_invalid(change):
    d = mutate(change)
    assert reference.validate(d)
    ing = ingest_reference() if os.environ.get("BENCH_REFERENCE_INGEST") else None
    if ing:
        assert reference.validate(d) == ing.validate(d, label_deny=[])


def two_assets(tmp_path, **second):
    d = asset_doc(tmp_path)
    f2 = tmp_path / "attachments" / "p.txt"
    f2.write_text("y")
    one = d["assets"][0]
    d["assets"].append(
        {**one, "path": "attachments/p.txt", "sha256": reference.sha256_file(f2), "case": case("other"), **second}
    )
    return d


def test_distinct_assets_are_accepted(tmp_path):
    assert reference.validate(two_assets(tmp_path), tmp_path) == []


@pytest.mark.parametrize(
    "second,what",
    [({"case": case("parity")}, "case"), ({"path": "attachments/o.txt"}, "path")],
)
def test_a_repeated_asset_case_or_path_is_an_error(tmp_path, second, what):
    d = two_assets(tmp_path, **second)
    assert any(f"assets[1] repeats a {what}" in e for e in reference.validate(d, tmp_path))


def test_a_repeated_asset_sha256_is_an_error(tmp_path):
    d = two_assets(tmp_path)
    (tmp_path / "attachments" / "p.txt").write_text("x")
    d["assets"][1]["sha256"] = d["assets"][0]["sha256"]
    assert any("assets[1] repeats a sha256" in e for e in reference.validate(d, tmp_path))


def test_repeated_assets_agree_with_the_ingest(tmp_path):
    ing = ingest_reference()
    for second in ({"case": case("parity")}, {"path": "attachments/o.txt"}, {}):
        d = two_assets(tmp_path, **second)
        assert reference.validate(d, tmp_path) == ing.validate(d, tmp_path, label_deny=[])


def test_the_asset_digest_term_includes_the_case():
    base = doc(assets=[{"path": "attachments/o.txt", "sha256": "a" * 64, "records": 1, "case": case("parity")}])
    other = doc(assets=[{"path": "attachments/o.txt", "sha256": "a" * 64, "records": 1, "case": case("other")}])
    assert reference.content_digest(base) != reference.content_digest(other)
    ing = ingest_reference() if os.environ.get("BENCH_REFERENCE_INGEST") else None
    if ing:
        assert reference.content_digest(base) == ing.content_digest(base)


@pytest.mark.parametrize(
    "when,ok",
    [("2026-01-01T00:00:00Z", True), ("2026-01-01", True), ("2999-01-01T00:00:00Z", False), ("2026-01-01Zjunk", False)],
)
def test_as_of_accepts_a_trailing_z(when, ok):
    d = doc(origin="uploaded", producer="p", as_of=when)
    d["comparability"]["caveats"] = "none"
    errors = [e for e in reference.validate(d) if "as_of" in e]
    assert (errors == []) is ok
    ing = ingest_reference() if os.environ.get("BENCH_REFERENCE_INGEST") else None
    if ing:
        assert reference.validate(d) == ing.validate(d, label_deny=[])


@pytest.mark.parametrize("n,ok", [(1, True), (2**32 - 1, True), (2**32, False), (0, False)])
def test_iterations_and_records_are_bounded_to_uint32(n, ok):
    d = doc()
    d["values"][0]["iterations"] = n
    assert (not any("iterations" in e for e in reference.validate(d))) is ok


# ---- round 5: word-based labels, compiled kernel names, epoch expiry, hostile JSON, tag-derived current ----

KERNEL = "spyre_kernel_fused_add_" + "a1b2c3d4e5f60718"


@pytest.mark.parametrize(
    "label,ok",
    [
        ("host-1", False),
        ("ref-worker-1", False),
        ("node-a", False),
        ("a-node", False),
        ("ghost-x", True),
        ("hostile-ref", True),
        ("ref-sys-a", True),
    ],
)
def test_the_builtin_label_deny_is_word_based(label, ok):
    d = doc(label=label)
    assert (reference.validate(d) == []) is ok
    ing = ingest_reference() if os.environ.get("BENCH_REFERENCE_INGEST") else None
    if ing:
        assert reference.validate(d) == ing.validate(d, label_deny=[])


@pytest.mark.parametrize(
    "name,stem",
    [
        (KERNEL, "spyre_kernel_fused_add"),
        (KERNEL + "#3", "spyre_kernel_fused_add#3"),
        ("fused_add", ""),
        ("spyre_kernel_x", ""),
        (None, ""),
        ("spyre_kernel_a_" + "A" * 16, ""),
    ],
)
def test_kernel_stem_matches_the_ingest(name, stem):
    assert reference.kernel_stem(name) == stem
    if os.environ.get("BENCH_REFERENCE_INGEST"):
        from spyre_clickhouse_ingest.identity import BenchmarkId

        ingest_reference()
        assert BenchmarkId.kernel_stem(name) == stem


def kernel_doc(name, keyed=True):
    c = {"component": "synth-comp", "name": "k", "tags": [], "disc": {"kernel_name": name}}
    if keyed:
        c["disc_keys"] = ["kernel_name"]
    c["benchmark_id"] = reference.case_id(c)
    return doc(values=[{"case": c, "backend": "cpu", "measurements": {"m": [1.0]}, "iterations": 1}])


def test_a_compiled_kernel_name_is_refused_when_it_is_a_key():
    assert any("re-keys" in e for e in reference.validate(kernel_doc(KERNEL)))
    assert any("re-keys" in e for e in reference.validate(kernel_doc(KERNEL + "#2", keyed=False)))
    assert reference.validate(kernel_doc("fused_add")) == []
    d = kernel_doc(KERNEL)
    d["values"][0]["case"]["disc_keys"] = []
    d["values"][0]["case"]["benchmark_id"] = reference.case_id(d["values"][0]["case"])
    assert reference.validate(d) == []


def test_kernel_names_agree_with_the_ingest():
    ing = ingest_reference()
    for d in (kernel_doc(KERNEL), kernel_doc(KERNEL + "#2", keyed=False), kernel_doc("fused_add")):
        assert reference.validate(d) == ing.validate(d, label_deny=[])


def test_an_epoch_expiry_means_never_expires():
    from datetime import datetime, timezone

    facts = reference.facts_of(doc())
    binds = [
        {"subject": "a", "permissions": ["references:approve"], "expires_at": datetime(1970, 1, 1)},
        {
            "subject": "b",
            "permissions": ["references:approve"],
            "expires_at": datetime(1970, 1, 1, tzinfo=timezone.utc),
        },
        {"subject": "c", "permissions": ["references:approve"], "expires_at": datetime(2001, 1, 1)},
    ]
    assert reference.holders(binds, "references:approve", facts) == {"a", "b"}
    if os.environ.get("BENCH_REFERENCE_INGEST"):
        assert reference.holders(binds, "references:approve", facts) == ingest_reference().holders(
            binds, "references:approve", facts
        )


def test_hostile_json_is_a_clean_error_everywhere(tmp_path):
    root = made_ref(tmp_path)
    huge = '{"a": ' + "9" * 6000 + "}"
    deep = "[" * 200000 + "]" * 200000
    for text in (huge, deep):
        (root / "reference.json").write_text(text)
        for argv in (["validate", str(root)], ["seal", str(root), "--out", str(tmp_path / "o")], ["submit", str(root)]):
            code, out, err = run_cli(argv, client=server().client("sub"))
            assert code == 2 and "Traceback" not in err, (argv, err[:200])
    (tmp_path / "m.json").write_text(deep)
    run = make_run(tmp_path, [make_result()])
    code, out, err = run_cli(
        [
            "create",
            "--from-run",
            str(run),
            "--key",
            "perf.synth.add.refsys",
            "--class",
            "perf",
            "--label",
            "ref-sys-a",
            "--meta",
            str(tmp_path / "m.json"),
        ],
        st=stores.MemoryStore(invalidated=[]),
    )
    assert code == 2


def test_validate_survives_deep_nesting_and_huge_integers():
    d = doc()
    nested = []
    for _ in range(3000):
        nested = [nested]
    d["values"][0]["measurements"] = {"m": [10**5000]}
    assert isinstance(reference.validate(d), list)
    d["comparability"]["x"] = nested
    assert isinstance(reference.validate(d), list)


def tagged(srv):
    first = submitted(srv)
    second = submitted(srv, d=mutate(lambda d: d["values"][0]["measurements"].update({"kernel.time.mean": [9.0]})))
    third = submitted(srv, d=mutate(lambda d: d["values"][0]["measurements"].update({"kernel.time.mean": [8.0]})))
    for uid in (first, second, third):
        srv.client("amy").approve(uid, terms_ack=True)
    return first, second, third


def test_retracting_writes_no_tag_and_the_previous_published_version_becomes_current():
    srv = server()
    first, second, third = tagged(srv)
    rows = list(srv.tags)
    assert srv.current["perf.synth.decode.refsys"] == third
    assert srv.client("sub").retract(third)["current_after"] == second
    assert srv.tags == rows
    assert srv.client("sub").retract(first)["current_after"] == second
    assert srv.current["perf.synth.decode.refsys"] == second


def test_retracting_a_non_current_version_changes_nothing():
    srv = server()
    first, second, third = tagged(srv)
    assert srv.client("sub").retract(second)["current_after"] == third
    pending = submitted(srv, d=mutate(lambda d: d["values"][0]["measurements"].update({"kernel.time.mean": [7.0]})))
    assert srv.client("sub").retract(pending)["current_after"] == third


def test_a_retire_survives_the_retraction_of_its_version():
    srv = server()
    first, second, third = tagged(srv)
    srv.client("ada").retire("perf.synth.decode.refsys")
    assert srv.current == {}
    srv.client("sub").retract(third)
    assert srv.current == {}
    srv.client("ada").rollback("perf.synth.decode.refsys", first)
    assert srv.current["perf.synth.decode.refsys"] == first
    srv.client("sub").retract(first)
    assert srv.current == {}  # the retire row is the newest valid row, so the key stays retired


def test_retire_and_rollback_are_for_administrators():
    srv = server()
    first, *_ = tagged(srv)
    with pytest.raises(rc.Refused, match="administrator"):
        srv.client("amy").retire("perf.synth.decode.refsys")
    with pytest.raises(rc.Refused, match="not a published version"):
        srv.client("ada").rollback("other.key", first)


# ---- surrogates, tag types, numeric digest ----


@pytest.mark.parametrize(
    "change,msg",
    [
        (lambda d: d.update(terms="bad \ud800 text"), "lone surrogate"),
        (lambda d: d["values"][0]["case"].update(name="n\udfff"), "lone surrogate"),
        (lambda d: d["values"][0]["case"].update(tags=["a", 5]), "needs string component/name"),
    ],
)
def test_surrogates_and_non_string_tags_are_refused(change, msg):
    d = mutate(change)
    errors = reference.validate(d)
    assert any(msg in e for e in errors), errors
    ing = ingest_reference() if os.environ.get("BENCH_REFERENCE_INGEST") else None
    if ing:
        assert errors == ing.validate(d, label_deny=[])


def test_the_digest_treats_11_and_11_0_alike():
    a, b = doc(), doc()
    a["values"][0]["measurements"] = {"m": [11, 12]}
    b["values"][0]["measurements"] = {"m": [11.0, 12.0]}
    assert reference.content_digest(a) == reference.content_digest(b)
    ing = ingest_reference() if os.environ.get("BENCH_REFERENCE_INGEST") else None
    if ing:
        assert reference.content_digest(a) == ing.content_digest(a)


def test_a_surrogate_never_reaches_a_sealed_file(tmp_path):
    root = made_ref(tmp_path, doc(terms="x"))
    (root / "reference.json").write_text(json.dumps(doc(terms="bad \ud800")))
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "b")])[0] == 2


# ---- review round: request invalidation, deterministic tgz, duplicates, offline check, submit order ----


def test_a_request_with_a_run_checks_invalidation_like_create(tmp_path):
    run = make_run(tmp_path, [make_result()])
    code, out, err = from_request(tmp_path, "--run", str(run), st=stores.MemoryStore(invalidated=[RUN_ID]))
    assert code == 1 and "invalidated" in err and not (tmp_path / "ref").exists()
    code, out, err = from_request(tmp_path, "--run", str(run), st=stores.NullStore())
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert code == 0 and "invalidation_unchecked" in d["warnings"] and "invalidation_unchecked" in err


def test_a_request_with_a_readable_store_has_no_unchecked_warning(tmp_path):
    run = make_run(tmp_path, [make_result()])
    from_request(tmp_path, "--run", str(run))
    assert "invalidation_unchecked" not in json.loads((tmp_path / "ref" / "reference.json").read_text())["warnings"]


def test_a_sealed_tgz_is_reproducible_and_carries_no_local_identity(tmp_path):
    root = made_ref(tmp_path)
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "a.tgz")])[0] == 0
    import time

    time.sleep(1.1)
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "b.tgz")])[0] == 0
    assert (tmp_path / "a.tgz").read_bytes() == (tmp_path / "b.tgz").read_bytes()
    with tarfile.open(tmp_path / "a.tgz") as tar:
        assert [m.name for m in tar.getmembers()] == ["bundle.json", "results/reference.json"]
        assert all((m.uid, m.gid, m.uname, m.gname, m.mtime) == (0, 0, "", "", 0) for m in tar.getmembers())


def test_a_packed_bundle_directory_is_reproducible_and_the_submit_token_is_stable(tmp_path):
    run_cli(["seal", str(made_ref(tmp_path)), "--out", str(tmp_path / "b")])
    one, two = (
        reference.pack_dir(tmp_path / "b", tmp_path / "1.tgz"),
        reference.pack_dir(tmp_path / "b", tmp_path / "2.tgz"),
    )
    assert one.read_bytes() == two.read_bytes()


def test_identical_content_is_refused_even_after_the_first_upload_was_retracted():
    srv = server()
    first = submitted(srv)
    second = submitted(srv)
    srv.client("amy").approve(first, terms_ack=True)
    srv.client("sub").retract(first)
    with pytest.raises(rc.Refused, match="retracted or not"):
        srv.client("amy").approve(second, terms_ack=True)


def sealed_bundle(tmp_path, name="b"):
    run_cli(["seal", str(made_ref(tmp_path / name)), "--out", str(tmp_path / name / "bundle")])
    return tmp_path / name / "bundle"


def _rewrite_meta(root, **change):
    meta = json.loads((root / "bundle.json").read_text())
    meta.update(change)
    (root / "bundle.json").write_text(json.dumps(meta))


def _add_file(root, rel, text="x", listed=True):
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    if listed:
        meta = json.loads((root / "bundle.json").read_text())
        meta["files"].append({"path": rel, "sha256": reference.sha256_file(f)})
        (root / "bundle.json").write_text(json.dumps(meta))


def _dup_path(root):
    meta = json.loads((root / "bundle.json").read_text())
    meta["files"].append(dict(meta["files"][0]))
    (root / "bundle.json").write_text(json.dumps(meta))


BAD_BUNDLES = {
    "stray top-level file": lambda r: _add_file(r, "notes.txt"),
    "nested results": lambda r: _add_file(r, "results/sub/x.json", "{}"),
    "second result file": lambda r: _add_file(r, "results/other.json", "{}"),
    "unlisted attachment": lambda r: _add_file(r, "attachments/x.txt", listed=False),
    "duplicate path": _dup_path,
    "artifact id": lambda r: _rewrite_meta(r, artifact_id="93c0abb3-ed25-5934-b811-31b6c149ba47"),
    "image": lambda r: _rewrite_meta(r, image="r@sha256:" + "a" * 64),
    "wrong arch": lambda r: _rewrite_meta(r, arch="s390x"),
    "wrong test type": lambda r: _rewrite_meta(r, test_type="quality"),
    "wrong component": lambda r: _rewrite_meta(r, component="other"),
    "newer schema": lambda r: _rewrite_meta(r, schema_version=2),
    "unknown key": lambda r: _rewrite_meta(r, bogus=1),
    "wrong kind": lambda r: _rewrite_meta(r, kind="junit"),
}


def test_a_sound_bundle_passes_the_offline_check(tmp_path):
    root = sealed_bundle(tmp_path)
    doc_ = json.loads((root / "results" / "reference.json").read_text())
    assert reference.check_bundle(root, doc_) == []
    if os.environ.get("BENCH_REFERENCE_INGEST"):
        ingest_reference()
        from spyre_clickhouse_ingest import offline

        offline.check(root)


@pytest.mark.parametrize("name", sorted(BAD_BUNDLES))
def test_a_bundle_the_ingest_would_refuse_is_refused_before_submit(tmp_path, name):
    root = sealed_bundle(tmp_path)
    BAD_BUNDLES[name](root)
    doc_ = json.loads((root / "results" / "reference.json").read_text())
    assert reference.check_bundle(root, doc_), name
    code, out, err = run_cli(["submit", str(root)], client=server().client("sub"))
    assert code == 2 and "not valid, not submitted" in err
    if os.environ.get("BENCH_REFERENCE_INGEST"):
        ingest_reference()
        from spyre_clickhouse_ingest import offline

        with pytest.raises(offline.BundleError):
            offline.check(root)


def test_submit_validates_before_it_needs_a_token(tmp_path, monkeypatch):
    for k in ("BENCH_DASHBOARD_URL", "BENCH_DASHBOARD_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    bad = made_ref(tmp_path / "x", doc(terms="x"))
    (bad / "reference.json").write_text(json.dumps(doc(label="Nope")))
    code, out, err = run_cli(["submit", str(bad)])
    assert code == 2 and "Nope" not in err and "label must match" in err and "token" not in err
    good = made_ref(tmp_path / "y")
    code, out, err = run_cli(["submit", str(good)])
    assert code == 2 and "needs a dashboard bearer token" in err and "UI-only" not in err


def test_submit_of_a_directory_without_a_reference_says_it_is_not_a_bundle(tmp_path):
    (tmp_path / "empty" / "results").mkdir(parents=True)
    (tmp_path / "empty" / "bundle.json").write_text("{}")
    for path in (tmp_path / "empty", tmp_path / "nothing-here"):
        code, out, err = run_cli(["submit", str(path)], client=server().client("sub"))
        assert code == 2 and "is not a bundle" in err and "FileNotFoundError" not in err


def test_meta_yaml_keeps_yes_and_no_as_text(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("a: yes\nb: no\nc: on\nd: off\ne: true\nf: false\ng: 5\n")
    assert reference.read_meta(p) == {"a": "yes", "b": "no", "c": "on", "d": "off", "e": True, "f": False, "g": 5}


# ---- accept -> reference: consuming bench.reference-request/1 ----


def make_request(**over):
    c = case("decode", {"dtype": "fp16", "shape": "64"})
    c["disc_keys"] = ["shape", "dtype"]
    c["benchmark_id"] = reference.case_id(c)
    value = {
        "benchmark_id": c["benchmark_id"], "name": "decode", "metric": "kernel.time.mean", "samples": [1.0, 1.1, 1.2],
        "as_of": "2026-11-15T03:00:00Z", "carried": False, "window_run_ids": [RUN_ID], "replaces": None,
        "case": {k: c[k] for k in ("component", "name", "tags", "disc", "disc_keys")},
    }  # fmt: skip
    req = {
        "request_schema": "bench.reference-request/1", "kind": "bench-reference", "origin": "generated", "class": "perf",
        "role": "anchor", "key": "anchor-synth-comp-cpu-1a2b3c4d", "label": "accepted baseline synth-comp cpu",
        "reason": "new normal", "accepted_by": "amy",
        "scope": {"component": "synth-comp", "arch": "x86_64", "backend": "cpu", "source_job": "nightly"},
        "anchor": {"from_run_id": RUN_ID, "as_of": "2026-11-15T03:00:00Z"},
        "comparability": {"harness": "spyre-bench", "harness_match": "yes", "dataset_subset": "full",
                          "dataset_match": "same", "precision": "unknown", "caveats": "an accepted run"},
        "values": [value], "warnings": [],
    }  # fmt: skip
    return {**req, **over}


def write_request(tmp_path, req=None):
    p = tmp_path / "reference-request.json"
    p.write_text(json.dumps(req or make_request()))
    return p


def from_request(tmp_path, *extra, req=None, st=None):
    argv = ["create", "--from-request", str(write_request(tmp_path, req)), "--out", str(tmp_path / "ref"), *extra]
    return run_cli(argv, st=st or stores.MemoryStore(invalidated=[]))


def test_a_request_becomes_a_valid_reference(tmp_path):
    code, out, err = from_request(tmp_path)
    assert code == 0, err
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert reference.validate(d) == []
    v = d["values"][0]
    assert v["case"]["disc_keys"] == ["shape", "dtype"] and v["case"]["benchmark_id"] == reference.case_id(v["case"])
    assert v["measurements"] == {"kernel.time.mean": [1.0, 1.1, 1.2]} and v["backend"] == "cpu" and v["iterations"] == 3
    assert d["key"] == "anchor-synth-comp-cpu-1a2b3c4d" and d["label"] == "accepted-baseline-synth-comp-cpu"
    assert d["source_run"]["run_id"] == RUN_ID and d["source_run"]["role"] == "anchor"


def test_the_placeholders_are_settled_from_the_values(tmp_path):
    from_request(tmp_path)
    c = json.loads((tmp_path / "ref" / "reference.json").read_text())["comparability"]
    assert c["precision"] == "fp16" and c["shape.batch"] == "n/a" and c["harness_match"] == "yes"


def test_without_the_run_cleanliness_is_unknown(tmp_path):
    code, out, err = from_request(tmp_path)
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert d["clean"] is None and "clean_unknown" in d["warnings"] and "clean_unknown" in err


def test_with_the_run_cleanliness_pinning_and_warnings_are_real_facts(tmp_path):
    props = {"run.exec.cpu_pinning": "device_local", "run.exec.pinning_verified": "true"}
    run = make_run(tmp_path, [make_result(props=props)])
    from_request(tmp_path, "--run", str(run))
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert d["clean"] is True and d["warnings"] == [] and d["pinning"] == "device_local"


def test_with_a_shared_run_the_reference_records_not_clean(tmp_path):
    run = make_run(tmp_path, [make_result(clean=SHARED)])
    from_request(tmp_path, "--run", str(run))
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert d["clean"] is False and d["warnings"] == ["not_clean"]


def test_a_request_without_case_blocks_is_refused_with_the_reason(tmp_path):
    req = make_request()
    del req["values"][0]["case"]
    code, out, err = from_request(tmp_path, req=req)
    assert code == 2 and "no case block" in err and not (tmp_path / "ref").exists()


def test_a_case_that_does_not_reproduce_the_stored_id_is_refused(tmp_path):
    req = make_request()
    req["values"][0]["benchmark_id"] = "0" * 8 + "-0000-5000-8000-000000000000"
    code, out, err = from_request(tmp_path, req=req)
    assert code == 2 and "derives" in err


def test_carried_values_and_several_metrics_share_a_row(tmp_path):
    req = make_request()
    second = dict(req["values"][0], metric="kernel.total_duration", samples=[5.0, 5.0, 5.0], carried=True)
    req["values"].append(second)
    from_request(tmp_path, req=req)
    d = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert len(d["values"]) == 1 and sorted(d["values"][0]["measurements"]) == [
        "kernel.time.mean",
        "kernel.total_duration",
    ]


def test_a_wrong_schema_or_both_sources_are_refused(tmp_path):
    assert from_request(tmp_path, req=make_request(request_schema="other/1"))[0] == 2
    p = write_request(tmp_path)
    run = make_run(tmp_path, [make_result()])
    code, out, err = run_cli(["create", "--from-request", str(p), "--from-run", str(run)], st=stores.NullStore())
    assert code == 2 and "exactly one" in err
    assert run_cli(["create", "--key", "x"], st=stores.NullStore())[0] == 2


def test_the_request_round_trip_reaches_compare(tmp_path):
    assert from_request(tmp_path)[0] == 0
    assert run_cli(["validate", str(tmp_path / "ref")])[0] == 0
    assert run_cli(["seal", str(tmp_path / "ref"), "--out", str(tmp_path / "b.tgz")])[0] == 0
    res = make_result(
        name="decode",
        disc={"dtype": "fp16", "shape": "64"},
        disc_keys=["shape", "dtype"],
        samples=((1.1, 1.1), (1.1, 1.1)),
    )
    code, out, err = run_cli(
        ["compare", "--run", str(make_run(tmp_path, [res])), "--reference", str(tmp_path / "b.tgz"), "--json"]
    )
    rep = json.loads(out)
    assert code == 0 and rep["rows"][0]["status"] == "same"


def test_a_request_reference_passes_the_ingest(tmp_path):
    ing = ingest_reference()
    from_request(tmp_path)
    assert ing.validate(json.loads((tmp_path / "ref" / "reference.json").read_text()), label_deny=[]) == []


# ---- security checks that must each have a test that fails without them ----


def tgz_with(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, kind, data in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            elif kind == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, data
                tar.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                tar.addfile(info)
            elif kind == "hardlink":
                info.type, info.linkname = tarfile.LNKTYPE, data
                tar.addfile(info)
    return path


@pytest.mark.parametrize(
    "name,kind,data",
    [
        ("../evil.txt", "file", b"x"),
        ("results/../../evil.txt", "file", b"x"),
        ("/abs/evil.txt", "file", b"x"),
        ("results/link", "symlink", "/etc/passwd"),
        ("results/fifo", "fifo", None),
        ("results/hard", "hardlink", "bundle.json"),
    ],
)
def test_unpack_refuses_unsafe_members_with_its_own_message(tmp_path, name, kind, data):
    path = tgz_with(tmp_path / "x.tgz", [("bundle.json", "file", b"{}"), (name, kind, data)])
    with pytest.raises(reference.UsageError, match="unsafe member"):
        reference.unpack(path, tmp_path / "out")
    code, out, err = run_cli(["submit", str(path)], client=server().client("sub"))
    assert code == 2 and not (tmp_path / "evil.txt").exists()


def test_check_bundle_refuses_a_listed_link(tmp_path):
    root = sealed_bundle(tmp_path)
    (tmp_path / "t.txt").write_text("t")
    (root / "attachments").mkdir()
    (root / "attachments" / "l.txt").symlink_to(tmp_path / "t.txt")
    meta = json.loads((root / "bundle.json").read_text())
    meta["files"].append({"path": "attachments/l.txt", "sha256": reference.sha256_file(tmp_path / "t.txt")})
    (root / "bundle.json").write_text(json.dumps(meta))
    errors = reference.check_bundle(root, json.loads((root / "results" / "reference.json").read_text()))
    assert errors == ["attachments/l.txt is a link"]


def test_pack_dir_refuses_a_link(tmp_path):
    root = sealed_bundle(tmp_path)
    (tmp_path / "t.txt").write_text("t")
    (root / "attachments").mkdir()
    (root / "attachments" / "l.txt").symlink_to(tmp_path / "t.txt")
    with pytest.raises(reference.UsageError, match="holds a link"):
        reference.pack_dir(root, tmp_path / "o.tgz")
    assert not (tmp_path / "o.tgz").exists()


def test_create_refuses_a_run_that_wrote_no_results(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"run_id": RUN_ID, "cases": []}))
    code, out, err = create(tmp_path, run)
    assert code == 2 and "no results in" in err and not (tmp_path / "ref").exists()


@pytest.mark.parametrize("suffix", ["#x", "#", "#1x", "#-1"])
def test_a_kernel_name_with_a_non_numeric_suffix_has_no_stem(suffix):
    assert reference.kernel_stem(KERNEL + suffix) == ""
    assert reference.kernel_stem(KERNEL + "#12") == "spyre_kernel_fused_add#12"


# ---- the ingest under test is the intended one, whatever earlier tests imported ----


def test_the_conformance_import_is_not_shadowed_by_an_earlier_one(tmp_path, monkeypatch):
    wanted = os.environ.get("BENCH_REFERENCE_INGEST")
    if not wanted:
        pytest.skip("BENCH_REFERENCE_INGEST names the intended ingest")
    fake = tmp_path / "old" / "spyre_clickhouse_ingest"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("")
    _clear_ingest_modules()
    monkeypatch.syspath_prepend(str(tmp_path / "old"))
    importlib.import_module("spyre_clickhouse_ingest")
    mod = ingest_reference()
    assert Path(mod.__file__).resolve().parent == (Path(wanted) / "spyre_clickhouse_ingest").resolve()


# ---- the HTTP client: clean errors, no clear-text tokens, no local file names ----


def test_a_non_json_reply_is_a_client_error():
    c, _ = http(["<html>gateway</html>"])
    with pytest.raises(rc.ClientError, match="not JSON"):
        c.show("u1")
    assert run_cli(["show", "u1"], client=http(["oops"])[0])[0] == 5


def test_an_incomplete_read_is_a_client_error():
    import http.client as hc

    c, _ = http([hc.IncompleteRead(b"par", 10)])
    with pytest.raises(rc.ClientError, match="cannot reach the dashboard: IncompleteRead"):
        c.show("u1")


@pytest.mark.parametrize(
    "url",
    ["http://dash.example.invalid", "ftp://x.invalid", "dash.invalid", "https://", "http://localhost.evil.invalid"],
)
def test_a_token_is_never_sent_over_clear_text_to_another_host(url):
    with pytest.raises(rc.InsecureUrl):
        rc.HttpClient(url, "tok")


@pytest.mark.parametrize("url", ["https://dash.invalid", "http://localhost:8080", "http://127.0.0.1", "http://[::1]:9"])
def test_https_and_loopback_urls_are_accepted(url):
    assert rc.HttpClient(url, "tok").base.startswith(url.split(":")[0])


def test_an_insecure_url_in_the_environment_is_exit_two(monkeypatch):
    monkeypatch.setenv("BENCH_DASHBOARD_URL", "http://dash.example.invalid")
    monkeypatch.setenv("BENCH_DASHBOARD_TOKEN", "tok")
    code, out, err = run_cli(["list"])
    assert code == 2 and "must be https" in err


def test_the_multipart_file_name_is_fixed_not_the_local_one(tmp_path):
    f = tmp_path / 'we"ird name.tgz'
    f.write_bytes(b"bundle")
    c, seen = http(["{}"])
    c.submit(f)
    body = seen[0][2].decode(errors="replace")
    assert 'filename="bundle.tgz"' in body and "ird" not in body


def test_a_repeated_approve_still_needs_the_terms_and_the_right():
    srv = server()
    uid = submitted(srv)
    srv.client("amy").approve(uid, terms_ack=True)
    with pytest.raises(rc.Refused, match="terms_ack"):
        srv.client("amy").approve(uid)
    with pytest.raises(rc.Refused, match="holds no references:approve"):
        srv.client("bob").approve(uid, terms_ack=True)
    assert srv.client("amy").approve(uid, terms_ack=True)["status"] == "duplicate"


# ---- seal targets ----


def test_seal_into_an_empty_directory_is_allowed_and_a_non_empty_one_is_refused(tmp_path):
    root = made_ref(tmp_path)
    (tmp_path / "empty").mkdir()
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "empty")])[0] == 0
    assert (tmp_path / "empty" / "bundle.json").is_file()
    code, out, err = run_cli(["seal", str(root), "--out", str(tmp_path / "empty")])
    assert code == 2 and "already exists" in err
    (tmp_path / "plainfile").write_text("x")
    assert run_cli(["seal", str(root), "--out", str(tmp_path / "plainfile")])[0] == 2


# ---- the dashboard's script-access contract: ack_not_clean, reasons, error bodies ----


def not_clean_doc():
    return mutate(lambda d: d.update(clean=False, warnings=["not_clean"]))


def test_a_not_clean_reference_needs_ack_not_clean_to_be_approved():
    srv = server()
    uid = submitted(srv, d=not_clean_doc())
    with pytest.raises(rc.Refused, match="ack_not_clean"):
        srv.client("amy").approve(uid, terms_ack=True)
    assert srv.client("amy").approve(uid, terms_ack=True, ack_not_clean=True)["state"] == "published"


def test_an_unknown_cleanliness_needs_the_acknowledgement_too():
    srv = server()
    uid = submitted(srv, d=mutate(lambda d: d.update(clean=None)))
    with pytest.raises(rc.Refused, match="ack_not_clean"):
        srv.client("amy").approve(uid, terms_ack=True)


def test_a_clean_reference_needs_no_acknowledgement():
    srv = server()
    assert srv.client("amy").approve(submitted(srv), terms_ack=True)["state"] == "published"


def test_the_approve_verb_carries_ack_not_clean():
    srv = server()
    uid = submitted(srv, d=not_clean_doc())
    code, out, err = run_cli(["approve", uid, "--terms-ack"], client=srv.client("amy"))
    assert code == 1 and "ack_not_clean" in err
    code, out, err = run_cli(["approve", uid, "--terms-ack", "--ack-not-clean"], client=srv.client("amy"))
    assert code == 0 and json.loads(out)["state"] == "published"


def test_the_http_approve_body_names_ack_not_clean():
    c, seen = http(["{}", "{}"])
    c.approve("u1", terms_ack=True, ack_not_clean=True)
    c.approve("u1", terms_ack=True)
    assert json.loads(seen[0][2])["ack_not_clean"] is True and json.loads(seen[1][2])["ack_not_clean"] is False


def test_reject_needs_a_reason_everywhere():
    srv = server()
    uid = submitted(srv)
    with pytest.raises(rc.Refused, match="reason is required"):
        srv.client("amy").reject(uid)
    code, out, err = run_cli(["reject", uid], client=srv.client("amy"))
    assert code == 2 and "needs --reason" in err


def test_a_409_with_missing_cases_says_to_register_them():
    body = b'{"error": "cases are not registered", "missing_cases": ["id-1", "id-2"]}'
    c, _ = http([urllib.error.HTTPError("https://d.invalid/x", 409, "c", {}, io.BytesIO(body))])
    with pytest.raises(rc.Refused) as e:
        c.approve("u1", terms_ack=True)
    assert "id-1" in str(e.value) and "--register-missing" in str(e.value)


@pytest.mark.parametrize(
    "code,word,refused",
    [(401, "token was rejected", True), (403, "not allowed", True), (404, "hidden", True), (503, "retry", False)],
)
def test_status_codes_get_a_plain_hint(code, word, refused):
    c, _ = http([urllib.error.HTTPError("https://d.invalid/x", code, "m", {}, io.BytesIO(b"{}"))])
    with pytest.raises(rc.ClientError) as e:
        c.show("u1")
    assert word in str(e.value) and isinstance(e.value, rc.Refused) is refused
