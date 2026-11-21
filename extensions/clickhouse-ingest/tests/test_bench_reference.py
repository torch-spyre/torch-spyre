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

"""bench-reference: the file format, the approval rules, the bundle kind and the writer at approval."""

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from spyre_clickhouse_ingest import bundle, offline, reference
from spyre_clickhouse_ingest.identity import BenchmarkId
from spyre_clickhouse_ingest.writer import DryRunClient

CASE = {
    "component": "synth-comp",
    "name": "decode/latency",
    "tags": [],
    "disc": {"model": "m1", "tp": "1"},
}
CASE["benchmark_id"] = reference.case_id(CASE)
PARITY = {
    "component": "synth-comp",
    "name": "parity/outputs",
    "tags": [],
    "disc": {"model": "m1"},
}
PARITY["benchmark_id"] = reference.case_id(PARITY)


def doc(**over):
    base = {
        "apiVersion": reference.API_VERSION, "kind": reference.KIND, "key": "perf.synth.decode.refsys.latency",
        "origin": "generated", "class": "perf", "label": "ref-sys-a", "terms": "internal-generated",
        "tools": {"torch": "2.9.1"}, "model": {"repo": "org/m1", "revision": "abc123"},
        "system": {"backend": "cpu", "precision": "fp32"}, "clean": True, "warnings": [],
        "comparability": {"harness": "synth@1", "harness_match": "yes", "precision": "fp32", "dataset_subset": "full",
                          "dataset_match": "same", "shape.batch": 1, "shape.input_len": 128, "shape.output_len": 64},
        "values": [{"case": CASE, "backend": "cpu", "measurements": {"llm.itl.median": [11.0, 11.5]}, "iterations": 2}],
    }  # fmt: skip
    return {**base, **over}


def doc_v(value):
    d = doc()
    d["values"][0]["measurements"]["llm.itl.median"] = [value, value + 0.5]
    return d


def quality_doc(**over):
    d = doc(
        **{
            "class": "quality",
            "dataset": {"id": "ds1", "subset": "full"},
            "key": "quality.synth.slot-12.refsys.acc",
        }
    )
    d["comparability"] = {"harness": "synth@1", "harness_match": "yes", "precision": "fp32", "dataset_subset": "full",
                          "dataset_match": "same", "fewshot": 5, "chat_template": "applied"}  # fmt: skip
    return {**d, **over}


# ---- validation ----


def test_a_complete_generated_perf_reference_is_valid():
    assert reference.validate(doc()) == []


@pytest.mark.parametrize(
    "over,problem",
    [
        ({"origin": "scraped"}, "origin must be"),
        ({"label": "worker-31"}, "label must match"),
        ({"label": "Ab"}, "label must match"),
        ({"terms": " "}, "terms is required"),
        ({"tools": {}}, "needs tools"),
        ({"model": {"repo": "x"}}, "model.revision"),
        ({"system": {}}, "system.backend"),
        ({"clean": "yes"}, "clean must be"),
        ({"version": 3}, "reserved key"),
        ({"approved_by": "x"}, "reserved key"),
        ({"bogus": 1}, "unknown key"),
        ({"values": []}, "needs values or assets"),
    ],
)
def test_a_bad_reference_is_refused(over, problem):
    assert any(problem in e for e in reference.validate(doc(**over))), (
        reference.validate(doc(**over))
    )


def test_values_are_checked_row_by_row():
    bad = copy.deepcopy(doc())
    bad["values"][0]["measurements"] = {"m": [float("nan")]}
    bad["values"][0]["iterations"] = 0
    bad["values"][0]["case"]["benchmark_id"] = "00000000-0000-5000-8000-000000000000"
    errors = " | ".join(reference.validate(bad))
    assert (
        "finite" in errors
        and "iterations (n) must be 1.." in errors
        and "does not match" in errors
    )
    twice = doc(values=doc()["values"] * 2)
    assert any("repeats" in e for e in reference.validate(twice))


def test_comparability_is_required_by_origin_and_class():
    d = doc()
    del d["comparability"]["precision"]
    assert any("comparability.precision" in e for e in reference.validate(d))
    d = doc(origin="uploaded", producer="partner-team", as_of="2026-01-01")
    errors = reference.validate(d)
    assert any("caveats" in e for e in errors)
    d["comparability"]["caveats"] = "none"
    d["comparability"]["harness_match"] = "unknown"
    assert reference.validate(d) == []
    published = doc(
        origin="published",
        producer="vendor",
        as_of="2026-01-01",
        source_url="https://example.invalid/card",
    )
    published["comparability"]["caveats"] = "none"
    assert any("non-empty caveats" in e for e in reference.validate(published))
    gen = doc()
    gen["comparability"]["harness_match"] = "unknown"
    assert any("cannot be unknown" in e for e in reference.validate(gen))
    assert any(
        "fewshot" in e
        for e in reference.validate(quality_doc(comparability=doc()["comparability"]))
    )


def test_a_future_as_of_and_a_missing_url_are_refused():
    future = (datetime.now(timezone.utc) + timedelta(days=3)).date().isoformat()
    d = doc(origin="published", producer="v", as_of=future)
    d["comparability"]["caveats"] = "differs"
    errors = " | ".join(reference.validate(d))
    assert "future" in errors and "source_url" in errors


def test_the_digest_is_the_meaning_not_the_wording():
    a, b = (
        doc(),
        doc(
            label="other-label", terms="different", warnings=["not_clean"], clean=False
        ),
    )
    assert reference.content_digest(a) == reference.content_digest(b)
    c = doc()
    c["values"][0]["measurements"]["llm.itl.median"] = [12.0]
    assert reference.content_digest(a) != reference.content_digest(c)
    ident = reference.identity(a)
    assert (ident.component, ident.arch, ident.kind, ident.id12) == (
        "reference",
        "noarch",
        "generic",
        reference.content_digest(a)[:12],
    )


def test_arch_is_noarch_unless_several_named_archs():
    assert reference.arch_of(doc()) == "noarch"
    d = doc()
    one = copy.deepcopy(d["values"][0])
    one["case"] = {**CASE, "disc": {**CASE["disc"], "arch": "s390x"}}
    one["case"]["benchmark_id"] = reference.case_id(one["case"])
    two = copy.deepcopy(one)
    two["case"] = {**CASE, "disc": {**CASE["disc"], "arch": "ppc64le"}}
    two["case"]["benchmark_id"] = reference.case_id(two["case"])
    assert reference.arch_of(doc(values=[one])) == "noarch"
    assert reference.arch_of(doc(values=[one, two])) == "multi"


def test_value_entries_hash_as_the_case_itself():
    (e,) = reference.value_entries(doc())
    assert (
        str(
            BenchmarkId.derive(
                e["component"], e["name"], e["tags"], e["disc"], e["disc_keys"]
            )
        )
        == CASE["benchmark_id"]
    )


def test_an_asset_must_match_its_bytes_and_count(tmp_path):
    (tmp_path / "attachments").mkdir()
    body = b'{"a":1}\n{"a":2}\n'
    (tmp_path / "attachments" / "out.jsonl").write_bytes(body)
    import hashlib

    asset = {
        "path": "attachments/out.jsonl",
        "sha256": hashlib.sha256(body).hexdigest(),
        "records": 2,
        "case": PARITY,
    }
    d = doc(values=[], assets=[asset])
    assert reference.validate(d, tmp_path) == []
    (e,) = reference.value_entries(d)
    assert (
        e["measurements"] == {"ref.records": [2.0]}
        and e["run_props"]["ref.asset_only"] == "1"
    )
    assert any(
        "2 record" in x or "3 record" in x
        for x in reference.validate(
            doc(values=[], assets=[{**asset, "records": 3}]), tmp_path
        )
    )
    assert any(
        "sha256" in x
        for x in reference.validate(
            doc(values=[], assets=[{**asset, "sha256": "0" * 64}]), tmp_path
        )
    )


# ---- lifecycle ----


def test_lifecycle_transitions():
    assert reference.next_state(None, "submit") == "pending"
    assert reference.next_state("pending", "approve") == "published"
    assert reference.next_state("published", "retract") == "retracted"
    assert reference.next_state("pending", "request_changes") == "rejected"
    for state, action in (
        ("published", "approve"),
        ("retracted", "retract"),
        ("rejected", "approve"),
        ("pending", "bogus"),
    ):
        with pytest.raises(ValueError):
            reference.next_state(state, action)


# ---- approval rules ----


def binding(subject, scope=None, **kw):
    return {
        "subject": subject,
        "subject_type": "user",
        "permissions": ["references:approve"],
        "scope": scope or {},
        **kw,
    }


def test_scope_fails_closed_and_ands_dimensions():
    facts = reference.facts_of(doc())
    assert reference.in_scope({}, facts)
    assert reference.in_scope({"class": ["perf"], "component": ["synth-comp"]}, facts)
    assert not reference.in_scope({"class": ["quality"]}, facts)
    assert not reference.in_scope(
        {"model_family": ["granite"]}, facts
    )  # no fact: fails closed
    assert not reference.in_scope({"component": ["other"]}, facts)


def test_the_approver_set_skips_the_submitter_expired_inactive_and_out_of_scope():
    facts = reference.facts_of(doc())
    now = datetime.now(timezone.utc)
    bindings = [
        binding("Amy@x.com"), binding("bob@x.com", {"class": ["quality"]}), binding("cy@x.com", expires_at=now - timedelta(days=1)),
        binding("di@x.com", active=False), binding("eve@x.com", {"component": ["synth-comp"]}),
        {"subject": "team-synth", "subject_type": "group", "permissions": ["references:approve"], "scope": {"class": ["perf"]}},
    ]  # fmt: skip
    members = {"team-synth": {"Fay@x.com", "amy@x.com"}}
    assert reference.eligible_approvers(bindings, facts, "amy@x.com", members, now) == {
        "eve@x.com",
        "fay@x.com",
    }
    assert reference.holders(bindings, "references:approve", facts, members, now) >= {
        "amy@x.com",
        "eve@x.com",
        "fay@x.com",
    }


def test_a_non_admin_never_approves_their_own_and_needs_the_binding():
    ok = reference.decide_approval(
        submitter="amy", approver="Bob", is_admin=False, approvers={"bob"}, reason=""
    )
    assert ok.allowed and not ok.override and ok.props["ref.approved_by"] == "bob"
    own = reference.decide_approval(
        submitter="amy", approver="amy", is_admin=False, approvers={"amy", "bob"}
    )
    assert not own.allowed and "own" in own.error
    unbound = reference.decide_approval(
        submitter="amy", approver="cy", is_admin=False, approvers={"bob"}
    )
    assert not unbound.allowed and "binding" in unbound.error


def test_an_admin_may_approve_anything_and_the_override_is_flagged():
    own = reference.decide_approval(
        submitter="amy", approver="amy", is_admin=True, approvers=set()
    )
    assert own.allowed and own.override and own.self_approved
    assert (
        own.props["ref.approval_override"] == "1"
        and own.props["ref.approval_self"] == "1"
    )
    other = reference.decide_approval(
        submitter="amy", approver="zed", is_admin=True, approvers={"bob"}
    )
    assert other.allowed and other.override and not other.self_approved
    bound_admin = reference.decide_approval(
        submitter="amy", approver="bob", is_admin=True, approvers={"bob"}
    )
    assert (
        bound_admin.allowed
        and not bound_admin.override
        and "ref.approval_override" in bound_admin.props
    )
    assert bound_admin.props["ref.approval_override"] == ""


def test_every_origin_waits_for_approval():
    assert all(reference.requires_approval(o) for o in reference.ORIGINS)


def test_retract_and_reject_rules():
    assert reference.decide_retract(
        submitter="amy", actor="amy", is_admin=False, state="pending"
    ).allowed
    assert reference.decide_retract(
        submitter="amy", actor="amy", is_admin=False, state="published"
    ).allowed
    assert not reference.decide_retract(
        submitter="amy", actor="bob", is_admin=False, state="published"
    ).allowed
    assert reference.decide_retract(
        submitter="amy", actor="root", is_admin=True, state="published"
    ).allowed
    assert not reference.decide_retract(
        submitter="amy", actor="amy", is_admin=True, state="retracted"
    ).allowed
    assert not reference.decide_reject(
        submitter="amy", actor="amy", is_admin=False, approvers={"amy"}
    ).allowed
    assert reference.decide_reject(
        submitter="amy", actor="bob", is_admin=False, approvers={"bob"}
    ).allowed
    assert not reference.decide_reject(
        submitter="amy", actor="cy", is_admin=False, approvers={"bob"}
    ).allowed


# ---- promoting a run ----


@pytest.mark.parametrize(
    "run,clean,warns,ok",
    [
        ({"clean": {"exclusive": "enforced"}, "repeats": 3}, "1", (), True),
        ({"clean": {"exclusive": "observed", "load_clean": True, "devices_exclusive": "enforced"}, "repeats": 3}, "1", (), True),
        ({"clean": {"exclusive": "observed", "load_clean": True}, "repeats": 3}, "0", ("not_clean", "exclusivity_not_enforced"), True),
        ({"clean": {"exclusive": "none"}, "repeats": 3}, "0", ("not_clean",), True),
        ({"clean": {}, "repeats": 3}, "unknown", ("clean_unknown",), True),
        ({"clean": {"exclusive": "enforced"}, "repeats": 1, "validity": "suspect", "pinning_verified": False}, "1",
         ("suspect", "pinning_unverified", "single_repeat"), True),
        ({"clean": {"exclusive": "enforced"}, "invalidated": True}, "1", (), False),
        ({"clean": {"exclusive": "enforced"}, "measured": False}, "1", (), False),
    ],
)  # fmt: skip
def test_a_non_clean_run_may_become_a_reference_with_warnings(run, clean, warns, ok):
    p = reference.promotion_check({"repeats": 2, **run})
    assert (p.ok, p.clean, p.warnings) == (ok, clean, warns) or (
        not ok and p.clean == clean and not p.ok
    )
    assert p.needs_ack == bool(warns)


# ---- the bundle kind ----


def seal(tmp_path, d=None, name="b", **meta):
    src = tmp_path / (name + "-src")
    (src / "attachments").mkdir(parents=True)
    (src / "reference.json").write_text(json.dumps(d or doc()))
    root = tmp_path / name
    (root / "results").mkdir(parents=True)
    (root / "results" / "reference.json").write_text(json.dumps(d or doc()))
    files = [
        {"path": p, "sha256": offline.sha256(root / p)}
        for p in offline.bundle_files(root)
    ]
    (root / "bundle.json").write_text(
        json.dumps({"kind": "bench-reference", "files": files, **meta})
    )
    return root


def test_a_reference_bundle_names_no_artifact_and_derives_arch_and_tier(tmp_path):
    meta = offline.check(seal(tmp_path))
    assert (meta["kind"], meta["test_type"], meta["arch"], meta["component"]) == (
        "bench-reference",
        "perf",
        "noarch",
        "reference",
    )
    assert (
        offline.check(seal(tmp_path, quality_doc(), name="q"))["test_type"] == "quality"
    )


@pytest.mark.parametrize(
    "meta,problem",
    [
        ({"artifact_id": "93c0abb3-ed25-5934-b811-31b6c149ba47"}, "names no artifact"),
        ({"arch": "s390x"}, "arch must be noarch"),
        ({"test_type": "quality"}, "differs from the reference's class"),
        ({"component": "spyre-inference"}, "component reference"),
    ],
)
def test_a_reference_bundle_cannot_claim_an_artifact_arch_or_tier(
    tmp_path, meta, problem
):
    with pytest.raises(offline.BundleError, match=problem):
        offline.check(seal(tmp_path, **meta))


def test_noarch_is_for_references_only(tmp_path):
    root = tmp_path / "j"
    (root / "results").mkdir(parents=True)
    (root / "results" / "a.xml").write_text(
        "<testsuite><testcase name='t'/></testsuite>"
    )
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "artifact_id": "93c0abb3-ed25-5934-b811-31b6c149ba47",
                "test_type": "unit",
                "arch": "noarch",
            }
        )
    )
    with pytest.raises(offline.BundleError, match="bench-reference"):
        offline.check(root)


def test_a_reference_is_held_in_references_not_the_relay_inbox(tmp_path):
    root = seal(tmp_path)
    meta = offline.check(root)
    assert "/references/" in offline.upload_path(
        meta, offline.digest(root)
    ) and "/inbox/" not in offline.upload_path(meta, "x")


def test_sealing_a_reference_dir_round_trips(tmp_path):
    src = tmp_path / "ref"
    (src / "attachments").mkdir(parents=True)
    (src / "reference.json").write_text(json.dumps(doc()))
    args = offline.parser().parse_args(
        ["--offline", "--reference-dir", str(src), "--out", str(tmp_path / "out")]
    )
    out = offline.write(args)
    meta = offline.check(out)
    assert (
        meta["kind"] == "bench-reference"
        and "arch" in meta
        and meta["arch"] == "noarch"
    )
    assert json.loads((out / "bundle.json").read_text()).get("arch") is None


# ---- the writer, against a recording client ----


class Reads(DryRunClient):
    """An in-memory stand-in for the tables the writer reads and writes; `fail` makes the next insert(s) into
    a table raise, `versions` pre-seeds completed versions of the key."""

    def __init__(self, known=(), versions=0, fail=None):
        super().__init__()
        self.known, self.versions, self.fail = set(known), versions, dict(fail or {})

    def insert(self, table, rows, column_names=None, database=None, **kw):
        if self.fail.get(table, 0) > 0:
            self.fail[table] -= 1
            raise RuntimeError(f"boom {table}")
        super().insert(table, rows, column_names=column_names, database=database, **kw)
        if table == "artifacts" and getattr(self, "race", None):
            self.rows["artifacts"].append(self.race())
            self.race = None

    def _completed(self):
        done = {r["artifact_id"] for r in self.rows.get("artifact_results", [])}
        return [a for a in self.rows.get("artifacts", []) if a["artifact_id"] in done]

    def query(self, sql, *a, **kw):
        from spyre_clickhouse_ingest.writer import _Rows

        p = kw.get("parameters") or {}
        arts, res = (
            self.rows.get("artifacts", []),
            self.rows.get("artifact_results", []),
        )
        if "FROM d.benchmarks WHERE benchmark_id IN" in sql:
            return _Rows([(i,) for i in p["ids"] if i in self.known])

        def merged(aid):
            out, rows = {}, [x for x in arts if x["artifact_id"] == aid]
            for x in rows:
                out.update(x["props"])
            return out if rows else None

        def latest():  # artifact id -> (current version, when it took it)
            res = {}
            for i, x in enumerate(arts):
                if x["props"]["ref.key"] and "ref.version" in x["props"]:
                    prev = res.get(x["artifact_id"])
                    took = (
                        x.get("_ts", i)
                        if not prev or prev[0] != x["props"]["ref.version"]
                        else prev[1]
                    )
                    res[x["artifact_id"]] = (x["props"]["ref.version"], took)
            return res

        if (
            "FROM d.v_artifacts WHERE artifact_id" in sql
            and "props['ref.upload_id']" in sql
        ):
            pr = merged(p["a"])
            if pr is None:
                return _Rows([])
            return _Rows(
                [(pr["ref.version"], pr["ref.upload_id"], pr["ref.submitted_by"], pr)]
            )
        if "SELECT props['ref'] FROM d.v_artifacts" in sql:
            pr = merged(p["a"])
            return _Rows([(pr.get("ref", ""),)] if pr else [])
        if "maxIf(audit_timestamp" in sql:
            held = sorted(
                (took, a)
                for a, (v, took) in latest().items()
                if v == p["v"]
                and next(x for x in arts if x["artifact_id"] == a)["props"]["ref.key"]
                == p["k"]
            )
            return _Rows([(a,) for _, a in held])
        if "max(toUInt32OrZero(v))" in sql:
            mine = [
                int(v)
                for a, (v, _) in latest().items()
                if merged(a)["ref.key"] == p["k"]
            ]
            return _Rows([(max([self.versions, *mine]),)])
        if "SELECT max(audit_timestamp) FROM d.artifacts" in sql:
            return _Rows([(None,)])
        if "SELECT ref FROM d.artifact_refs" in sql:
            hit = [
                r
                for r in self.rows.get("artifact_refs", [])
                if r["artifact_id"] == p["a"]
            ]
            return _Rows([(hit[0]["ref"],)] if hit else [])
        if "argMax(tuple(t.tag, t.artifact_id)" in sql:
            tags = [
                t
                for t in self.rows.get("artifact_tags", [])
                if t["props"].get("ref.key") == p["k"]
            ]
            return _Rows(
                [(tags[-1]["tag"], tags[-1]["artifact_id"])] if tags else [("", "")]
            )
        if "SELECT props['ref.version'] FROM d.v_artifacts" in sql:
            pr = merged(p["a"])
            return _Rows([(pr["ref.version"],)] if pr else [])
        if "FROM d.artifact_tags" in sql:
            tags = [
                t
                for t in self.rows.get("artifact_tags", [])
                if t["props"].get("ref.key") == p["k"]
            ]
            return _Rows([(tags[-1]["tag"], tags[-1]["artifact_id"])] if tags else [])
        if "count() FROM d.artifact_refs" in sql:
            return _Rows(
                [
                    (
                        sum(
                            r["artifact_id"] == p["artifact_id"]
                            and r["ref"] == p["ref"]
                            for r in self.rows.get("artifact_refs", [])
                        ),
                    )
                ]
            )
        if "count() FROM d.artifacts" in sql:
            return _Rows([(sum(x["artifact_id"] == p["artifact_id"] for x in arts),)])
        if "count() FROM d.artifact_results" in sql:
            return _Rows(
                [
                    (
                        sum(
                            r["artifact_id"] == p["artifact_id"]
                            and r["run_id"] == p["run_id"]
                            for r in res
                        ),
                    )
                ]
            )
        if "count() FROM d.benchmark_runs" in sql:
            runs = self.rows.get("benchmark_runs", [])
            return _Rows(
                [
                    (
                        sum(
                            r["run_id"] == p["run_id"]
                            and r["component"] == p["component"]
                            for r in runs
                        ),
                    )
                ]
            )
        return super().query(sql, *a, **kw)


UP = "01a13151-2096-79bf-bd9c-311b8c8bc729"
APPROVAL = {"ref.approved_by": "bob", "ref.approval_reason": "checked"}


def ctx(**over):
    return reference.WriteContext(**{"submitter": "amy", "upload_id": UP, "approval": APPROVAL, "bundle_sha256": "ab" * 32,
                                     "bundle_url": "https://art.invalid/references/x.tgz", **over})  # fmt: skip


def test_an_unregistered_case_is_refused_until_registered_now():
    c = Reads()
    with pytest.raises(reference.ReferenceError, match="not registered"):
        reference.write(c, "d", doc(), ctx())
    assert not c.rows
    out = reference.write(c, "d", doc(), ctx(register_missing=True))
    assert (
        out["registered_now"] == [CASE["benchmark_id"]] and out["status"] == "ingested"
    )


def test_approval_writes_the_artifact_rows_verdict_and_current_tag():
    c = Reads(known=[CASE["benchmark_id"]])
    out = reference.write(c, "d", doc(clean=False, warnings=["not_clean"]), ctx())
    assert (out["version"], out["arch"], out["current"]) == (1, "noarch", True)
    (art,) = c.rows["artifacts"]
    assert (art["component"], art["kind"], art["arch"], art["origin"]) == (
        "reference",
        "generic",
        "noarch",
        "built",
    )
    p = art["props"]
    assert (p["ref.key"], p["ref.version"], p["ref.submitted_by"], p["ref.approved_by"], p["ref.clean"], p["ref.warnings"], p["ref.visibility"]) == (
        "perf.synth.decode.refsys.latency", "1", "amy", "bob", "0", "not_clean", "internal")  # fmt: skip
    assert p["ref.comparability.harness_match"] == "yes" and p[
        "ref.digest"
    ] == reference.content_digest(doc())
    (run,) = c.rows["benchmark_runs"]
    rp = run["props"]
    assert (
        rp["run.ref_only"],
        rp["run.trust"],
        rp["run.record_type"],
        rp["upload_id"],
        rp["ref.key"],
    ) == ("1", "external", "reference", UP, "perf.synth.decode.refsys.latency")
    assert (
        run["run_id"] == out["run_id"] == reference.run_id_for(doc(), "amy", UP)
        and run["backend"] == "cpu"
    )
    (v,) = c.rows["artifact_results"]
    assert (v["test_type"], v["state"], v["arch"], v["result_kind"]) == (
        "perf",
        "passed",
        "noarch",
        "performance",
    )
    assert (
        v["props"]["bundle_kind"] == "bench-reference" and v["run_id"] == out["run_id"]
    )
    (tag,) = c.rows["artifact_tags"]
    assert (
        tag["tag"],
        tag["tag_family"],
        tag["artifact_id"],
        tag["props"]["ref.key"],
    ) == ("ref-current", "reference", art["artifact_id"], p["ref.key"])
    assert c.rows["artifact_refs"][0]["ref"] == "https://art.invalid/references/x.tgz"


def test_approval_without_promotion_writes_no_tag_and_the_version_counts_up():
    c = Reads(known=[CASE["benchmark_id"]], versions=2)
    out = reference.write(c, "d", doc(), ctx(promote=False))
    assert (
        out["version"] == 3
        and "artifact_tags" not in c.rows
        and out["current"] is False
    )


def test_a_dry_run_validates_and_writes_nothing():
    c = Reads(known=[CASE["benchmark_id"]])
    out = reference.write(c, "d", doc(), ctx(), dry_run=True)
    assert out["status"] == "would-ingest" and not c.rows and out["unregistered"] == []
    out = reference.write(Reads(), "d", doc(), ctx(), dry_run=True)
    assert out["unregistered"] == [CASE["benchmark_id"]] and out["registered_now"] == []
    out = reference.write(Reads(), "d", doc(), ctx(register_missing=True), dry_run=True)
    assert out["registered_now"] == [CASE["benchmark_id"]] and out["unregistered"] == []
    with pytest.raises(reference.ReferenceError, match="origin must be"):
        reference.write(c, "d", doc(origin="x"), ctx(), dry_run=True)


def test_an_uploaded_reference_is_upstream_origin():
    d = doc(origin="uploaded", producer="partner-team", as_of="2026-01-01")
    d["comparability"]["caveats"] = "none"
    c = Reads(known=[CASE["benchmark_id"]])
    reference.write(c, "d", d, ctx())
    assert c.rows["artifacts"][0]["origin"] == "upstream"


def _args(props, dry_run=False):
    from types import SimpleNamespace

    return SimpleNamespace(
        dry_run=dry_run, bundle_prop=props, uploader="", bundle_url=""
    )


def test_the_bundle_writer_backstops_self_approval_and_trusts_no_submitter_from_the_bundle(
    tmp_path,
):
    root = seal(tmp_path, uploader="mallory")
    meta = offline.check(root)
    refused = {
        "self": ([("approved_by", "amy"), ("upload_id", UP), ("submitted_by", "amy")], "cannot approve their own"),
        "none": ([("upload_id", UP)], "written at approval"),
        "no submitter (the bundle's uploader is not one)": ([("approved_by", "bob"), ("upload_id", UP)], "written at approval"),
        "bad time": ([("approved_by", "bob"), ("upload_id", UP), ("submitted_by", "amy"), ("approved_at", "yesterday")], "ISO-8601"),
    }  # fmt: skip
    for props, problem in refused.values():
        with pytest.raises(offline.BundleError, match=problem):
            bundle._ingest_reference(root, meta, _args(props))


def test_a_flagged_self_approval_always_records_both_flags(tmp_path, monkeypatch):
    from spyre_clickhouse_ingest import client

    captured = {}
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "d")
    monkeypatch.setattr(
        client.ClickHouse,
        "connect",
        classmethod(lambda cls, **k: Reads(known=[CASE["benchmark_id"]])),
    )
    monkeypatch.setattr(
        reference,
        "write",
        lambda c, db, d, ctx, dry: captured.update(ctx=ctx) or {"status": "ingested"},
    )
    root = seal(tmp_path)
    props = [
        ("approved_by", "Amy"),
        ("upload_id", UP),
        ("submitted_by", "amy"),
        ("approval_override", "1"),
    ]
    bundle._ingest_reference(root, offline.check(root), _args(props))
    a = captured["ctx"].approval
    assert (
        a["ref.approved_by"],
        a["ref.approval_override"],
        a["ref.approval_self"],
    ) == ("amy", "1", "1")
    bundle._ingest_reference(
        root,
        offline.check(root),
        _args([("approved_by", "bob"), ("upload_id", UP), ("submitted_by", "amy")]),
    )
    a = captured["ctx"].approval
    assert (a["ref.approval_override"], a["ref.approval_self"]) == ("", "")


def test_wait_is_refused_for_a_reference_before_anything_is_put(tmp_path):
    from types import SimpleNamespace

    root = seal(tmp_path)
    args = SimpleNamespace(
        base_url="https://art.invalid",
        repo="r",
        token="t",
        root="/x/{arch}",
        wait=5,
        as_folder=False,
        inbox="inbox",
    )
    with pytest.raises(offline.BundleError, match="held for approval"):
        offline.upload(root, args)


# ---- a failed write resumes; a completed one is a duplicate ----


def _tables(c):
    return {t: len(r) for t, r in c.rows.items()}


@pytest.mark.parametrize(
    "table", ["benchmark_runs", "artifact_tags", "artifact_results"]
)
def test_a_write_that_fails_part_way_resumes_with_the_same_run_and_version(table):
    c = Reads(known=[CASE["benchmark_id"]], fail={table: 1})
    with pytest.raises(reference.ReferenceError) as err:
        reference.write(c, "d", doc(), ctx())
    assert err.value.retry
    assert not any(r["artifact_id"] for r in c.rows.get("artifact_results", []))
    out = reference.write(c, "d", doc(), ctx())
    assert out["status"] == "resumed" and out["version"] == 1
    tables = _tables(c)
    tags_after_resume = tables.pop("artifact_tags")
    assert tags_after_resume in (
        1,
        2,
    )  # a resume that failed after its tag writes it again
    assert tables == {
        "artifacts": 1,
        "artifact_refs": 1,
        "benchmark_runs": 1,
        "artifact_results": 1,
    }
    assert (
        c.rows["benchmark_runs"][0]["run_id"]
        == c.rows["artifact_results"][0]["run_id"]
        == out["run_id"]
    )
    again = reference.write(c, "d", doc(), ctx())
    assert (
        again["status"] == "duplicate"
        and again["run_id"] == out["run_id"]
        and again["version"] == 1
    )
    assert (
        _tables(c)["artifact_tags"] == tags_after_resume
        and _tables(c)["artifacts"] == 1
    )


def test_a_completed_retry_never_moves_the_slot_back():
    c = Reads(known=[CASE["benchmark_id"]])
    reference.write(c, "d", doc_v(11.0), ctx())
    reference.write(
        c, "d", doc_v(12.0), ctx(upload_id="01a13151-2096-79bf-bd9c-311b8c8bc702")
    )
    tags = len(c.rows["artifact_tags"])
    assert reference.write(c, "d", doc_v(11.0), ctx())["status"] == "duplicate"
    assert len(c.rows["artifact_tags"]) == tags


def test_the_same_content_under_another_upload_is_refused():
    c = Reads(known=[CASE["benchmark_id"]])
    reference.write(c, "d", doc(), ctx())
    with pytest.raises(reference.ReferenceError, match="already version 1") as err:
        reference.write(
            c,
            "d",
            doc(label="other-label"),
            ctx(upload_id="01a13151-2096-79bf-bd9c-311b8c8bc702"),
        )
    assert not err.value.retry


def test_an_incomplete_attempt_keeps_its_number_and_its_retry_completes_it():
    c = Reads(known=[CASE["benchmark_id"]], fail={"benchmark_runs": 1})
    a_upload, b_upload = UP, "01a13151-2096-79bf-bd9c-311b8c8bc702"
    with pytest.raises(reference.ReferenceError):
        reference.write(c, "d", doc_v(11.0), ctx())
    b = reference.write(c, "d", doc_v(12.0), ctx(upload_id=b_upload))
    assert b["version"] == 2
    a = reference.write(c, "d", doc_v(11.0), ctx(upload_id=a_upload))
    assert (a["status"], a["version"]) == ("resumed", 1)
    versions = sorted(r["props"]["ref.version"] for r in c.rows["artifacts"])
    assert versions == ["1", "2"] and len(c.rows["artifact_results"]) == 2


def test_a_version_already_held_moves_the_claim_to_the_next_free_number(monkeypatch):
    c = Reads(known=[CASE["benchmark_id"]])
    reference.write(c, "d", doc_v(11.0), ctx())
    monkeypatch.setattr(
        reference,
        "next_version",
        lambda *a: 1 if not getattr(c, "second", False) else 2,
    )
    c.second = False
    calls = []
    real = reference._holders_of_version

    def holders(*a):
        calls.append(1)
        c.second = True
        return real(*a)

    monkeypatch.setattr(reference, "_holders_of_version", holders)
    out = reference.write(
        c, "d", doc_v(12.0), ctx(upload_id="01a13151-2096-79bf-bd9c-311b8c8bc702")
    )
    assert out["version"] == 2
    assert sorted(r["props"]["ref.version"] for r in c.rows["artifacts"]) == ["1", "2"]


def test_a_resumed_write_keeps_the_first_attempts_approver():
    c = Reads(known=[CASE["benchmark_id"]], fail={"artifact_results": 1})
    with pytest.raises(reference.ReferenceError):
        reference.write(
            c,
            "d",
            doc(),
            ctx(approval={"ref.approved_by": "bob", "ref.approval_reason": "first"}),
        )
    reference.write(
        c,
        "d",
        doc(),
        ctx(approval={"ref.approved_by": "eve", "ref.approval_reason": "retry"}),
    )
    assert c.rows["artifacts"][0]["props"]["ref.approved_by"] == "bob"
    assert {t["props"]["promoted_by"] for t in c.rows["artifact_tags"]} == {"bob"}


def test_the_digest_covers_class_origin_and_comparability_and_run_ids_ignore_login_case():
    a = reference.content_digest(doc())
    changed = doc()
    changed["comparability"]["precision"] = "bf16"
    assert reference.content_digest(changed) != a
    assert reference.content_digest(doc(origin="uploaded")) != a
    assert reference.run_id_for(doc(), "Amy", UP) == reference.run_id_for(
        doc(), "amy", UP
    )


def test_an_asset_row_cannot_merge_with_a_value_row():
    asset = {
        "path": "attachments/o.jsonl",
        "sha256": "0" * 64,
        "records": 2,
        "case": CASE,
    }
    errors = reference.validate(doc(assets=[asset]))
    assert any("share case" in e for e in errors)


def test_scope_is_case_insensitive_and_expiry_tolerates_naive_times():
    assert reference.in_scope({"component": ["Synth-Comp"]}, reference.facts_of(doc()))
    naive_future = datetime.now() + timedelta(days=1)  # noqa: DTZ005 -- clickhouse-connect returns naive DateTime
    bs = [
        binding("amy@x.com", expires_at=naive_future),
        binding("bob@x.com", expires_at=datetime.now() - timedelta(days=1)),
    ]  # noqa: DTZ005
    assert reference.holders(bs, "references:approve", reference.facts_of(doc())) == {
        "amy@x.com"
    }


def test_authorize_approval_is_the_one_call_a_route_makes():
    bs = [binding("bob@x.com", {"class": ["perf"]})]
    ok = reference.authorize_approval(
        doc(), submitter="amy@x.com", approver="Bob@x.com", is_admin=False, bindings=bs
    )
    assert ok.allowed and not ok.override
    assert not reference.authorize_approval(
        doc(), submitter="amy@x.com", approver="cy@x.com", is_admin=False, bindings=bs
    ).allowed
    assert reference.authorize_approval(
        doc(), submitter="amy@x.com", approver="amy@x.com", is_admin=True, bindings=bs
    ).self_approved


def test_the_event_row_matches_the_result_uploads_ddl():
    from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR

    ev = reference.upload_event(upload_id=UP, state="pending", actor="Amy", artifact_id=reference.identity(doc()).artifact_id,
                                run_id=reference.run_id_for(doc(), "amy", UP), doc=doc())  # fmt: skip
    assert (
        ev["format"] == "reference" and ev["actor"] == "amy" and ev["arch"] == "noarch"
    )
    sql = (SCHEMA_DIR / "49-result-uploads.sql").read_text()
    assert "'reference'" in sql and "'bench'" in sql


# ---- deny-list, case keys, baseline purposes ----


def test_the_label_denylist_is_data_from_a_file(tmp_path, monkeypatch):
    assert any("label must match" in e for e in reference.validate(doc(label="host-7")))
    assert reference.validate(doc(label="ref-sys-a")) == []
    lst = tmp_path / "deny.txt"
    lst.write_text("# extra patterns\n\n^acme-\nsecret\n")
    assert [p.pattern for p in reference.label_denylist(str(lst))] == [
        "^acme-",
        "secret",
    ]
    assert reference.validate(
        doc(label="acme-box"), label_deny=reference.label_denylist(str(lst))
    )
    monkeypatch.setenv(reference.ENV_LABEL_DENYLIST, str(lst))
    assert any("deny-listed" in e for e in reference.validate(doc(label="acme-box")))
    assert any(
        "deny-listed" in e for e in reference.validate(doc(label="my-secret-ref"))
    )
    assert reference.validate(doc(label="ref-sys-a")) == []
    monkeypatch.delenv(reference.ENV_LABEL_DENYLIST)
    assert reference.validate(doc(label="acme-box")) == []


def test_a_bad_denylist_pattern_is_an_error(tmp_path):
    lst = tmp_path / "deny.txt"
    lst.write_text("(unclosed\n")
    with pytest.raises(ValueError, match="bad pattern"):
        reference.label_denylist(str(lst))


def test_disc_keys_make_a_profile_ordered_id_reproducible():
    case = {
        "component": "synth-comp",
        "name": "serve/x",
        "tags": [],
        "disc": {"tp": "4", "in": "128"},
    }
    sorted_id = reference.case_id(case)
    ordered = {**case, "disc_keys": ["tp", "in"], "id_profile": "vllm"}
    ordered_id = reference.case_id(ordered)
    assert (
        ordered_id != sorted_id
        and reference.case_id({**case, "disc_keys": ["in", "tp"]}) == sorted_id
    )
    ordered["benchmark_id"] = ordered_id
    d = doc(
        values=[
            {
                "case": dict(ordered),
                "backend": "cpu",
                "measurements": {"m": [1.0]},
                "iterations": 1,
            }
        ]
    )
    assert reference.validate(d) == []
    d["values"][0]["case"]["benchmark_id"] = sorted_id
    assert any("does not match" in e for e in reference.validate(d))
    d["values"][0]["case"].update(benchmark_id=ordered_id, disc_keys="tp")
    assert any("disc_keys must be a list" in e for e in reference.validate(d))
    (e,) = reference.value_entries(doc(values=[{"case": {**ordered, "benchmark_id": ordered_id}, "backend": "cpu",
                                                "measurements": {"m": [1.0]}, "iterations": 1}]))  # fmt: skip
    assert e["disc_keys"] == ("tp", "in")
    assert (
        str(
            BenchmarkId.derive(
                e["component"], e["name"], e["tags"], e["disc"], e["disc_keys"]
            )
        )
        == ordered_id
    )


def test_baseline_purposes_are_the_view_rows_and_anything_else_warns():
    from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR

    sql = (SCHEMA_DIR / "62-benchmark-verdicts.sql").read_text()
    body = sql.split("CREATE VIEW IF NOT EXISTS benchmark_baseline_purposes AS")[
        1
    ].split(";")[0]
    assert (
        tuple(__import__("re").findall(r"'([a-z-]+)'", body.split("values(")[1]))
        == reference.BASELINE_PURPOSES
    )
    base = {"clean": {"exclusive": "enforced"}, "repeats": 3}
    for purpose in (None, "", "nightly", "Weekly", "merge"):
        assert (
            "not_baseline_purpose"
            not in reference.promotion_check({**base, "purpose": purpose}).warnings
        )
    for purpose in (
        "pr",
        "pr-check",
        "smoke",
        "adhoc",
        "dev",
        "release",
        "investigate",
    ):
        assert (
            "not_baseline_purpose"
            in reference.promotion_check({**base, "purpose": purpose}).warnings
        )


# ---- malformed files, asset paths, scope key ----


@pytest.mark.parametrize(
    "over,problem",
    [
        ({"model": "x"}, "model must be an object"),
        ({"dataset": "x"}, "dataset must be an object"),
        ({"system": "cpu"}, "system must be an object"),
        ({"tools": ["torch"]}, "tools must be an object"),
        ({"tools": {"torch": 2}}, "version strings"),
        ({"source_run": "r"}, "source_run must be an object"),
        ({"terms": 5}, "terms must be a string"),
        ({"warnings": "not_clean"}, "warnings must be a list"),
        ({"values": "x"}, "values must be a list"),
        ({"values": ["x"]}, "needs a case object"),
        ({"comparability": []}, "comparability must be an object"),
    ],
)
def test_wrong_json_types_are_validation_errors_not_crashes(over, problem):
    assert any(problem in e for e in reference.validate(doc(**over))), (
        reference.validate(doc(**over))
    )


def test_a_case_with_wrong_types_and_a_bool_count_are_refused():
    bad = doc()
    bad["values"][0]["case"] = {**CASE, "tags": "x", "disc": []}
    assert any("tags list" in e for e in reference.validate(bad))
    bad = doc()
    bad["values"][0]["iterations"] = True
    assert any("iterations" in e for e in reference.validate(bad))


def test_a_write_of_a_malformed_file_is_a_refusal_not_a_retry():
    c = Reads(known=[CASE["benchmark_id"]])
    with pytest.raises(
        reference.ReferenceError, match="tools must be an object"
    ) as err:
        reference.write(c, "d", doc(tools=["a"]), ctx())
    assert not err.value.retry


def test_check_reference_reports_a_malformed_file_as_a_bundle_error(tmp_path):
    root = seal(tmp_path, doc(model="x"))
    with pytest.raises(offline.BundleError, match="model must be an object"):
        offline.check(root)


@pytest.mark.parametrize(
    "path",
    [
        "attachments/../../x.jsonl",
        "attachments/a/../../x",
        "attachments\\x",
        "x/attachments/a",
    ],
)
def test_an_asset_path_cannot_leave_attachments(tmp_path, path):
    asset = {"path": path, "sha256": "0" * 64, "records": 1, "case": PARITY}
    assert any(
        "plain path under attachments" in e
        for e in reference.validate(doc(values=[], assets=[asset]), tmp_path)
    )


def test_a_linked_or_unlisted_attachment_is_refused(tmp_path):
    import hashlib

    (tmp_path / "attachments").mkdir()
    body = b'{"a":1}\n'
    (tmp_path / "attachments" / "out.jsonl").write_bytes(body)
    (tmp_path / "attachments" / "stray.txt").write_text("x")
    asset = {
        "path": "attachments/out.jsonl",
        "sha256": hashlib.sha256(body).hexdigest(),
        "records": 1,
        "case": PARITY,
    }
    assert any(
        "not listed in assets" in e
        for e in reference.validate(doc(values=[], assets=[asset]), tmp_path)
    )
    (tmp_path / "attachments" / "stray.txt").unlink()
    assert reference.validate(doc(values=[], assets=[asset]), tmp_path) == []
    (tmp_path / "attachments" / "out.jsonl").unlink()
    outside = tmp_path.parent / "outside.jsonl"
    outside.write_bytes(body)
    (tmp_path / "attachments" / "out.jsonl").symlink_to(outside)
    assert any(
        "link or leaves" in e
        for e in reference.validate(doc(values=[], assets=[asset]), tmp_path)
    )


def test_an_approver_scope_can_be_limited_to_a_key():
    facts = reference.facts_of(doc())
    assert reference.in_scope({"key": [doc()["key"]]}, facts)
    assert not reference.in_scope({"key": ["other.key.here"]}, facts)
    assert "key" in reference.SCOPE_DIMENSIONS


def test_a_concurrent_approval_that_inserts_first_wins_the_version():
    c = Reads(known=[CASE["benchmark_id"]])
    mine = reference.identity(doc_v(12.0)).artifact_id

    def competitor():
        return {"artifact_id": "99999999-0000-5000-8000-000000000001", "_ts": -1,
                "props": {"ref.key": doc()["key"], "ref.version": "1", "ref.upload_id": "x", "ref.submitted_by": "zed"}}  # fmt: skip

    c.race = competitor
    out = reference.write(c, "d", doc_v(12.0), ctx())
    assert (
        out["version"] == 2
    )  # the loser renumbered itself with a later row and completed
    rows = [a for a in c.rows["artifacts"] if a["artifact_id"] == mine]
    assert [r["props"]["ref.version"] for r in rows] == ["1", "2"]
    assert {r["props"]["ref.version"] for r in c.rows["benchmark_runs"]} == {"2"}


def test_a_resumed_older_write_does_not_take_the_slot_back():
    c = Reads(known=[CASE["benchmark_id"]], fail={"artifact_results": 1})
    with pytest.raises(reference.ReferenceError):
        reference.write(
            c, "d", doc_v(11.0), ctx()
        )  # A: tagged, then failed before its verdict
    reference.write(
        c, "d", doc_v(12.0), ctx(upload_id="01a13151-2096-79bf-bd9c-311b8c8bc702")
    )  # B: v2, current
    tags = len(c.rows["artifact_tags"])
    out = reference.write(c, "d", doc_v(11.0), ctx())
    assert out["status"] == "resumed" and out["version"] == 1
    assert len(c.rows["artifact_tags"]) == tags
    assert (
        c.rows["artifact_tags"][-1]["artifact_id"]
        == reference.identity(doc_v(12.0)).artifact_id
    )


def test_a_retry_with_another_bundle_url_keeps_the_first_address():
    c = Reads(known=[CASE["benchmark_id"]], fail={"artifact_results": 1})
    with pytest.raises(reference.ReferenceError):
        reference.write(
            c, "d", doc(), ctx(bundle_url="https://art.invalid/references/one.tgz")
        )
    reference.write(
        c, "d", doc(), ctx(bundle_url="https://art.invalid/references/two.tgz")
    )
    assert [r["ref"] for r in c.rows["artifact_refs"]] == [
        "https://art.invalid/references/one.tgz"
    ]


def test_upload_id_must_be_a_uuid_and_a_null_repeat_count_is_tolerated(tmp_path):
    root = seal(tmp_path)
    props = [
        ("approved_by", "bob"),
        ("upload_id", "not-a-uuid"),
        ("submitted_by", "amy"),
    ]
    with pytest.raises(offline.BundleError, match="not a UUID"):
        bundle._ingest_reference(root, offline.check(root), _args(props))
    assert reference.promotion_check(
        {"clean": {"exclusive": "enforced"}, "repeats": None}
    ).ok


def _fuzz_cases():
    """Every JSON path of a valid file set to each junk value, plus a variant that carries an asset."""
    junk = [
        None,
        0,
        "",
        "x",
        [],
        {},
        [[]],
        {"a": []},
        True,
        [None],
        [[1]],
        10**400,
        -(10**400),
        1e308,
        {"a": {"b": []}},
        "\u0000",
        ["a"],
    ]
    for base in (
        doc(),
        doc(
            values=[],
            assets=[
                {
                    "path": "attachments/o.jsonl",
                    "sha256": "0" * 64,
                    "records": 1,
                    "case": PARITY,
                }
            ],
        ),
    ):
        paths = []

        def walk(o, p=()):
            paths.append(p)
            for k, v in (
                o.items()
                if isinstance(o, dict)
                else enumerate(o)
                if isinstance(o, list)
                else ()
            ):
                walk(v, p + (k,))

        walk(base)
        for p in paths:
            for j in junk:
                d = copy.deepcopy(base)
                if p:
                    o = d
                    for k in p[:-1]:
                        o = o[k]
                    o[p[-1]] = j
                else:
                    d = j
                yield d


def test_validate_is_total_whatever_the_json_holds(tmp_path):
    (tmp_path / "attachments").mkdir()
    n = 0
    for d in _fuzz_cases():
        n += 1
        errors = reference.validate(d, tmp_path)
        assert isinstance(errors, list)
        if not errors:  # an accepted file must survive every later stage
            reference.content_digest(d)
            reference.facts_of(d)
            reference.value_entries(d)
            reference.arch_of(d)
            reference.case_ids(d)
    assert n > 1000


def test_a_trailing_newline_is_not_a_slug_or_a_label():
    assert any(
        "label must match" in e for e in reference.validate(doc(label="ref-sys-a\n"))
    )
    assert any(
        "key must be a slug" in e
        for e in reference.validate(doc(key="perf.synth.decode.refsys.latency\n"))
    )
    assert any(
        "closed backend" in e
        for e in reference.validate(doc(system={"backend": "cpu\n"}))
    )


def test_repeated_asset_cases_paths_or_bytes_are_refused_and_the_asset_case_is_in_the_digest(
    tmp_path,
):
    a1 = {
        "path": "attachments/a.jsonl",
        "sha256": "1" * 64,
        "records": 1,
        "case": PARITY,
    }
    a2 = {
        "path": "attachments/b.jsonl",
        "sha256": "2" * 64,
        "records": 1,
        "case": PARITY,
    }
    assert any(
        "repeats a case" in e
        for e in reference.validate(doc(values=[], assets=[a1, a2]))
    )
    other = {**PARITY, "name": "parity/other"}
    other["benchmark_id"] = reference.case_id(other)
    a3 = {**a2, "case": other, "path": "attachments/a.jsonl"}
    assert any(
        "repeats a path" in e
        for e in reference.validate(doc(values=[], assets=[a1, a3]))
    )
    a4 = {**a2, "case": other, "sha256": "1" * 64}
    assert any(
        "repeats a sha256" in e
        for e in reference.validate(doc(values=[], assets=[a1, a4]))
    )
    same_bytes_other_case = {**a1, "case": other}
    assert reference.content_digest(
        doc(values=[], assets=[a1])
    ) != reference.content_digest(doc(values=[], assets=[same_bytes_other_case]))


def test_as_of_accepts_a_z_suffix():
    d = doc(origin="uploaded", producer="partner-team", as_of="2026-01-01T00:00:00Z")
    d["comparability"]["caveats"] = "none"
    assert reference.validate(d) == []


def test_current_reports_the_real_outcome_of_a_suppressed_resume():
    c = Reads(known=[CASE["benchmark_id"]], fail={"artifact_results": 1})
    with pytest.raises(reference.ReferenceError):
        reference.write(c, "d", doc_v(11.0), ctx())
    reference.write(
        c, "d", doc_v(12.0), ctx(upload_id="01a13151-2096-79bf-bd9c-311b8c8bc702")
    )
    assert reference.write(c, "d", doc_v(11.0), ctx())["current"] is False
    assert (
        reference.write(Reads(known=[CASE["benchmark_id"]]), "d", doc(), ctx())[
            "current"
        ]
        is True
    )


def test_a_failed_address_insert_is_repaired_from_the_artifact_row():
    c = Reads(known=[CASE["benchmark_id"]], fail={"artifact_refs": 1})
    with pytest.raises(reference.ReferenceError):
        reference.write(
            c, "d", doc(), ctx(bundle_url="https://art.invalid/references/one.tgz")
        )
    reference.write(
        c, "d", doc(), ctx(bundle_url="https://art.invalid/references/two.tgz")
    )
    assert [r["ref"] for r in c.rows["artifact_refs"]] == [
        "https://art.invalid/references/one.tgz"
    ]


def test_counts_are_bounded_by_the_destination_column():
    top = reference.UINT32_MAX
    d = doc()
    d["values"][0]["iterations"] = top
    assert reference.validate(d) == []
    for bad in (top + 1, 2**40, 0, -1):
        d["values"][0]["iterations"] = bad
        assert any("iterations (n) must be" in e for e in reference.validate(d))
    asset = {
        "path": "attachments/o.jsonl",
        "sha256": "0" * 64,
        "records": top + 1,
        "case": PARITY,
    }
    assert any(
        "records must be" in e
        for e in reference.validate(doc(values=[], assets=[asset]))
    )


def test_the_bound_is_the_column_the_rows_land_in():
    from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR

    sql = (SCHEMA_DIR / "30-benchmarks.sql").read_text()
    assert "iterations   UInt32" in sql


def test_a_compiled_kernel_name_is_refused_because_the_writer_re_keys_it():
    case = {
        "component": "synth-comp",
        "name": "kernel/x",
        "tags": [],
        "disc": {"kernel_name": "spyre_kernel_fused_add_ab3de5gh7jk9mn1p"},
        "disc_keys": ["kernel_name"],
    }
    case["benchmark_id"] = reference.case_id(case)
    d = doc(
        values=[
            {
                "case": case,
                "backend": "cpu",
                "measurements": {"m": [1.0]},
                "iterations": 1,
            }
        ]
    )
    assert any("re-keys" in e for e in reference.validate(d))
    plain = {**case, "disc": {"kernel_name": "fused_add"}}
    plain["benchmark_id"] = reference.case_id(plain)
    assert (
        reference.validate(
            doc(
                values=[
                    {
                        "case": plain,
                        "backend": "cpu",
                        "measurements": {"m": [1.0]},
                        "iterations": 1,
                    }
                ]
            )
        )
        == []
    )


def test_an_epoch_expiry_means_never_and_ghost_is_not_a_host():
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert reference.holders(
        [binding("amy@x.com", expires_at=epoch)],
        "references:approve",
        reference.facts_of(doc()),
    ) == {"amy@x.com"}
    assert reference.validate(doc(label="ghost-x")) == []
    assert any("label must match" in e for e in reference.validate(doc(label="host-1")))
    assert any(
        "label must match" in e for e in reference.validate(doc(label="ref-worker-1"))
    )


def test_a_deep_or_huge_json_is_a_bundle_error_not_a_traceback(tmp_path):
    root = tmp_path / "b"
    (root / "results").mkdir(parents=True)
    (root / "results" / "reference.json").write_text("[" * 100000 + "]" * 100000)
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "kind": "bench-reference",
                "files": [
                    {
                        "path": "results/reference.json",
                        "sha256": offline.sha256(root / "results" / "reference.json"),
                    }
                ],
            }
        )
    )
    with pytest.raises(offline.BundleError):
        offline.check(root)
    (root / "results" / "reference.json").write_text('{"x": ' + "9" * 6000 + "}")
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "kind": "bench-reference",
                "files": [
                    {
                        "path": "results/reference.json",
                        "sha256": offline.sha256(root / "results" / "reference.json"),
                    }
                ],
            }
        )
    )
    with pytest.raises(offline.BundleError):
        offline.check(root)


def test_append_to_run_is_refused_for_a_reference(tmp_path):
    root = seal(tmp_path)
    args = _args([("approved_by", "bob"), ("upload_id", UP), ("submitted_by", "amy")])
    args.append_to_run = True
    with pytest.raises(offline.BundleError, match="append-to-run"):
        bundle._ingest(root, args)


def test_a_resume_keeps_the_wording_of_the_first_attempt_on_the_value_rows():
    c = Reads(known=[CASE["benchmark_id"]], fail={"artifact_results": 1})
    with pytest.raises(reference.ReferenceError):
        reference.write(
            c, "d", doc(label="first-label", warnings=["not_clean"], clean=False), ctx()
        )
    reference.write(c, "d", doc(label="second-label", warnings=[], clean=True), ctx())
    assert c.rows["artifacts"][0]["props"]["ref.label"] == "first-label"
    assert {r["props"]["ref.label"] for r in c.rows["benchmark_runs"]} == {
        "first-label"
    }
    assert {r["props"]["ref.warnings"] for r in c.rows["benchmark_runs"]} == {
        "not_clean"
    }


def test_promote_is_sticky_across_a_resume():
    c = Reads(known=[CASE["benchmark_id"]], fail={"artifact_results": 1})
    with pytest.raises(reference.ReferenceError):
        reference.write(c, "d", doc(), ctx(promote=False))
    out = reference.write(
        c, "d", doc(), ctx(promote=True)
    )  # the retry's value must not change what attempt 1 meant
    assert out["current"] is False and "artifact_tags" not in c.rows
    c2 = Reads(known=[CASE["benchmark_id"]], fail={"artifact_results": 1})
    with pytest.raises(reference.ReferenceError):
        reference.write(c2, "d", doc(), ctx(promote=True))
    assert reference.write(c2, "d", doc(), ctx(promote=False))["current"] is True


def test_a_lone_surrogate_and_non_string_tags_are_refused():
    for field in ("terms", "producer"):
        assert any(
            "surrogate" in e for e in reference.validate(doc(**{field: "x\ud800"}))
        ), field
    bad = doc()
    bad["values"][0]["case"] = {**CASE, "tags": [1, 2]}
    assert any("tags list" in e for e in reference.validate(bad))


def test_the_digest_ignores_1_versus_1_dot_0():
    a, b = doc(), doc()
    a["values"][0]["measurements"]["llm.itl.median"] = [11, 12]
    b["values"][0]["measurements"]["llm.itl.median"] = [11.0, 12.0]
    assert reference.content_digest(a) == reference.content_digest(b)
