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

"""`results --from-bundle <dir|tgz>`: record an offline results bundle in spyre_v2.

The bundle (offline.py) goes through the path a connected run takes -- `results` for JUnit,
vllm.py for vLLM bench JSON -- so its rows link exactly as that run's do. It writes no
artifact: the artifact must already be recorded, found by artifact_id, else by image digest.
What a minimal bundle.json leaves out is derived here: arch and component from the artifact,
the run key as manual:<uploader>:<bundle sha256>, so a re-upload stays one run.
Exit codes: 0 ingested or already ingested, 1 failed (retry), 2 rejected, 3 incomplete.
"""

import contextlib
import getpass
import json
import os
import sys
from pathlib import Path

from . import vllm
from .identity import DerivedId, RunId
from .offline import (
    FAILED,
    MANUAL_PREFIX,
    STATUS,
    BundleError,
    check,
    digest,
    inbox_keys,
    is_manual,
    opened,
    run_key,
    specs,
)

SOURCE = "bundle"


def run_id(meta: dict) -> str:
    """The run_id the bundle's rows land under: a Jenkins key hashes as a connected run's."""
    key = meta["run_key"]
    source = SOURCE if is_manual(key) else "jenkins"
    return RunId.derive(source, key, meta["arch"], meta["test_type"])


def resolve_artifact(client, db: str, meta: dict):
    """(Resolution, the field that named it): the id first, else the image, lookup only.

    Each named field must resolve to the same artifact; one that names none (an unknown id)
    is passed over when another resolves.
    """
    from .resolver import resolve

    found, missed = {}, []
    for field, sp in specs(meta):
        try:
            r = resolve(sp, meta.get("arch", ""), client=client, db=db, lookup="only")
        except ValueError as err:
            missed.append(f"{field} {sp!r}: {err}")
            continue
        if r is None:
            missed.append(f"{field} {sp!r}: not recorded in {db}")
        else:
            found.setdefault(r.artifact_id, (r, field))
    if len(found) > 1:
        names = {aid: field for aid, (_, field) in found.items()}
        raise BundleError(
            f"the bundle's artifact fields name different artifacts: {names}"
        )
    if not found:
        raise BundleError("artifact not recorded: " + "; ".join(missed))
    return next(iter(found.values()))


def existing_run(client, db: str, rid: str, component: str) -> tuple:
    """(bundle digests of the run's recorded verdicts, the source files of its cases)."""
    verdicts = client.query(
        f"SELECT props['bundle_sha256'] FROM {db}.artifact_results "
        "WHERE run_id = {rid:UUID} AND state != 'running'",
        parameters={"rid": rid},
    ).result_rows
    files = client.query(
        f"SELECT DISTINCT props['source_file'] FROM {db}.test_case_runs "
        "WHERE component = {c:String} AND run_id = {rid:UUID}",
        parameters={"rid": rid, "c": component},
    ).result_rows
    return {r[0] for r in verdicts}, {r[0] for r in files}


def verdict_recorded(client, db: str, aid: str, rid: str, test_type: str) -> bool:
    return (
        client.query(
            f"SELECT count() FROM {db}.artifact_results WHERE artifact_id = {{aid:UUID}} "
            "AND run_id = {rid:UUID} AND test_type = {t:String} AND state != 'running'",
            parameters={"aid": aid, "rid": rid, "t": test_type},
        ).result_rows[0][0]
        > 0
    )


def results_argv(meta: dict, root: Path, aid: str, rid: str, props: dict, strict: bool):
    """The `results` argv recording this bundle: the artifact by id, lookup only."""
    key = meta["run_key"]
    argv = [
        "--schema", "v2", "--xml-dir", str(root / "results"),
        "--component", meta["component"], "--arch", meta["arch"],
        "--trigger-type", meta["test_type"], "--run-id", rid,
        "--artifact", f"id:{aid}", "--lookup", "only", "--registry", "off",
        "--run-url", meta.get("run_url") or props.get("bundle_url", ""),
    ]  # fmt: skip
    if meta.get("started_at"):
        argv += ["--triggered-at", meta["started_at"]]
    if not is_manual(key):
        argv += ["--jenkins-run-key", key]
    for flag, value in (
        ("--workflow", meta.get("workflow")),
        ("--tag-family", meta.get("tag_family")),
    ):
        if value:
            argv += [flag, value]
    for tag in meta.get("tags", []):
        argv += ["--tag", tag]
    for k, v in props.items():
        if v:
            argv += ["--result-prop", f"{k}={v}"]
    return argv + (["--strict"] if strict else [])


def vllm_rows(root: Path, meta: dict) -> list:
    """The bundle's flat vLLM rows, exactly as the live ingest extracts them."""
    perf = meta.get("perf", {})
    with contextlib.redirect_stdout(sys.stderr):
        return vllm.extract_rows(
            str(root / "results"), perf.get("head_branch", ""), perf.get("head_sha", ""),
            "", "0", meta.get("workflow") or "vLLM Benchmark", 0, arch=meta["arch"],
            model=perf.get("model", ""),
        )  # fmt: skip


def ingest(args) -> int:
    try:
        with opened(Path(args.from_bundle)) as root:
            report = _ingest(root, args)
    except BundleError as err:
        print(json.dumps({"status": STATUS[err.code], "reason": str(err)}))
        return err.code
    print(json.dumps(report, sort_keys=True))
    return 0


def _ingest(root: Path, args) -> dict:
    meta = check(root)
    key = run_key(meta)
    if not is_manual(key) and args.trusted_job_prefix:
        if not any(key.startswith(p) for p in args.trusted_job_prefix):
            raise BundleError(
                f"run key {key!r} is not from a trusted job ({args.trusted_job_prefix}); "
                f"leave run_key out, or use {MANUAL_PREFIX}<who>:<token>"
            )
    if args.expect_key and args.expect_key not in inbox_keys(meta):
        raise BundleError(
            f"its folder {args.expect_key!r} is not its artifact_id or sha256-<image digest> "
            f"({sorted(inbox_keys(meta))})"
        )
    from .client import ClickHouse, ClickHouseEnv

    db = ClickHouseEnv.target_database()
    if not db:
        raise BundleError("no database: set CLICKHOUSE_DB_V2", FAILED)
    try:
        client = ClickHouse.connect(database=db)
        if args.dry_run:
            client.set_client_setting("readonly", "2")
    except Exception as err:  # noqa: BLE001 -- unreachable is a retry, not a verdict on the bundle
        raise BundleError(f"spyre_v2 unreachable: {err}", FAILED) from None
    r, via = resolve_artifact(client, db, meta)
    sha = digest(root)
    uploader = args.uploader or meta.get("uploader") or getpass.getuser()
    meta.setdefault("arch", DerivedId.arch(r.arch))
    meta.setdefault("component", r.component)
    meta["run_key"] = key or f"{MANUAL_PREFIX}{uploader}:{sha}"
    if not (meta["arch"] and meta["component"]):
        raise BundleError(
            f"artifact {r.artifact_id} has no arch/component to default to; set them"
        )
    aid, rid = r.artifact_id, run_id(meta)
    report = {"run_key": meta["run_key"], "run_id": rid, "artifact_id": aid, "artifact_from": via,
              "test_type": meta["test_type"], "bundle_sha256": sha}  # fmt: skip
    seen, files = existing_run(client, db, rid, meta["component"])
    if seen == {sha}:
        return {**report, "status": "duplicate"}
    if seen:
        raise BundleError(
            f"run {meta['run_key']!r} ({rid}) is already recorded by another run or bundle"
        )
    names = {Path(p).name for p in (root / "results").iterdir()}
    if files - names:
        raise BundleError(f"run {rid} already has cases from {sorted(files - names)}")
    if args.dry_run:
        return {**report, "status": "would-ingest"}
    props = {"source": SOURCE, "uploader": uploader, "bundle_sha256": sha,
             "bundle_url": args.bundle_url}  # fmt: skip
    if meta.get("kind") == "vllm":
        report["benchmark_runs"] = write_vllm(client, db, root, meta, aid, rid, props)
    else:
        write_junit(db, root, meta, aid, rid, props, args.strict)
    if not verdict_recorded(client, db, aid, rid, meta["test_type"]):
        raise BundleError(
            f"no {meta['test_type']} verdict landed for {aid} under {rid}", FAILED
        )
    return {**report, "status": "ingested"}


def write_junit(db, root, meta, aid, rid, props, strict) -> None:
    from .results import main as results

    os.environ["CLICKHOUSE_DB_V2"] = db
    # stdout carries only this command's one-line JSON report.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            results(results_argv(meta, root, aid, rid, props, strict))
    except SystemExit as exit_:
        if exit_.code not in (0, None):
            raise BundleError(f"results exited {exit_.code}", FAILED) from None


def write_vllm(client, db, root, meta, aid, rid, props) -> int:
    """The live vLLM leg's v2 rows: benchmarks + benchmark_runs, then its performance verdict."""
    from .client import tables_present
    from .schema import ARTIFACT_RESULTS, BENCHMARK_RUNS, BENCHMARKS
    from .writer import insert_artifact_result

    if not tables_present(
        client, db, tables=(BENCHMARKS, BENCHMARK_RUNS, ARTIFACT_RESULTS)
    ):
        raise BundleError(f"{db} lacks the benchmark tables", FAILED)
    rows = vllm_rows(root, meta)
    if not rows:
        raise BundleError("no vLLM metric extracted from results/")
    try:
        n = vllm.write_benchmarks(client, db, rows, rid)
        insert_artifact_result(
            client, db, artifact_id=aid, run_id=rid, test_type=meta["test_type"],
            state=meta.get("perf", {}).get("state", "passed"), arch=meta["arch"],
            result_kind="performance", duration_s=vllm.duration_s(rows),
            props={**{k: v for k, v in props.items() if v},
                   "run_url": meta.get("run_url") or props.get("bundle_url", "")},
        )  # fmt: skip
    except Exception as err:  # noqa: BLE001 -- a write error is a retry; the writes dedup
        raise BundleError(f"vLLM write failed: {err!r}", FAILED) from None
    return n
