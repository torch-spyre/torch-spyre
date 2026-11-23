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

"""Read-only stores behind plan (reuse lookup), run (rows landed) and verdict: none, ClickHouse, in-memory.

The engine never writes rows; the ingest does. A store that cannot answer raises StoreError, which callers report
(exit 5) rather than treat as "nothing stored".
"""

import os
import uuid
from datetime import datetime, timezone

import regex as re

from spyre_bench import engine

LABEL_PREFIX = "env.label."
ID_BATCH = 1000
VS_CANDIDATES = 8
DB_NAME = re.compile(r"^[A-Za-z0-9_]{1,64}$")
STATUS_OF_VIEW = {"unchanged": "pass"}  # the SQL view's word for the engine's `pass`
MISSING_PHRASES = ("does not exist", "doesn't exist")
INSTALL_HINT = "install clickhouse-connect (pip install 'spyre-bench[clickhouse]')"


class StoreError(RuntimeError):
    """The store is unreachable, misconfigured or refused the query."""


def missing_object(error, name):
    """True when `error` says that the object `name` itself is absent (not merely that its SQL mentions it).

    ClickHouse echoes the whole query, so the name alone proves nothing: it must sit in front of "does not exist" or
    inside an "Unknown table ... 'name'" identifier.
    """
    text = str(error)
    quoted = re.search(rf"Unknown (?:table|expression)[^'\n]*'[^']*\b{re.escape(name)}\b[^']*'", text)
    plain = re.search(rf"\b{re.escape(name)}\W{{0,3}}\s+(?:does not|doesn't) exist", text)
    return bool(quoted or plain)


class Store:
    """Interface. Subclasses answer from their own backend; the base answers "nothing stored"."""

    name = "none"

    def stored_results(self, benchmark_ids, subject_id):
        """Rows for engine.reuse_lookup: benchmark_id, subject_id, env_id, labels, validity, n, ts, run_id."""
        return []

    def invalidations(self, run_ids):
        """Active invalidation events for these runs (none are readable yet)."""
        return []

    def invalidated_runs(self, run_ids):
        """The subset of run_ids with an active run-scope invalidation; raises StoreError when this store cannot tell."""
        raise StoreError(f"store {self.name!r} cannot report invalidations")

    def resolve_vs(self, kind, value, product, purposes=(), arch=""):
        """Candidate artifacts for a `vs=` target, newest first: artifact_id, component, arch, ts, source, ref_key, ref_state."""
        return []

    def target_results(self, kind, value, target, benchmark_ids):
        """Stored results of the resolved target, shaped like stored_results rows (plus purpose, trust)."""
        return self.stored_results(benchmark_ids, target["artifact_id"])

    def check(self):
        """Raise StoreError if the store cannot possibly work (missing settings); never connects."""

    def landed(self, run_id):
        """The benchmark_ids that have measurement rows for a run."""
        raise StoreError(f"store {self.name!r} cannot report landed rows")

    def verdict_rows(self, run_id):
        """Raw verdict rows (dicts) of one run."""
        raise StoreError(f"store {self.name!r} holds no verdicts")

    def latest_run(self, artifact_id):
        """The run id of the artifact's newest run that has measurement rows, or None."""
        raise StoreError(f"store {self.name!r} holds no runs")

    def anchored_rows(self, run_id):
        """Anchored-verdict rows of one run, or None when the store has no anchored view."""
        return None

    def series_runs(self, run_id):
        """Per series and metric of one run: samples and baseline eligibility (what `accept` takes values from)."""
        raise StoreError(f"store {self.name!r} holds no runs")

    def anchor_rows(self, component, arch, backend, source_job):
        """The current anchor values of one scope (carried into a new anchor version for series not re-accepted)."""
        return []

    def case_rows(self, benchmark_ids):
        """Registered cases by id: dicts benchmark_key, component, name, tags, props (what a request's case block is rebuilt from)."""
        return []

    def series_window(self, row, n):
        """The series' newest `n` baseline-eligible runs up to and including `row`'s run: dicts run_key, run_ts, samples."""
        raise StoreError(f"store {self.name!r} holds no runs")

    def client(self):
        """A query client on the store's database, for readers that ship their own SQL (invalidate)."""
        raise StoreError(f"store {self.name!r} has no query client")

    def registered(self, benchmark_ids):
        """The subset of benchmark_ids already registered, or None when this store cannot tell."""
        return None

    def reference_values(self, key, version="current"):
        """Value rows (dicts) of one reference version from v_bench_reference_values; version is "current" or an int."""
        raise StoreError(f"store {self.name!r} holds no references")


class NullStore(Store):
    pass


class MemoryStore(Store):
    """In-memory store for tests and demos: lists of dicts shaped like the rows each method returns."""

    name = "memory"

    def __init__(
        self,
        results=(),
        landed=None,
        verdicts=None,
        runs=None,
        fail=None,
        anchored=None,
        series=None,
        anchors=None,
        windows=None,
        cases=None,
        known=None,
        references=(),
        invalidated=None,
        targets=(),
        events=(),
    ):
        self.cases = cases if cases is not None else []
        self.windows = windows if windows is not None else []
        self.anchored = anchored
        self.series = series if series is not None else {}
        self.anchors = anchors if anchors is not None else []
        self.results = list(results)
        self.known = known
        self.invalidated = invalidated
        self.references = list(references)
        self.targets = list(targets)
        self.events = list(events)
        self.landed_ids = landed if landed is not None else {}
        self.verdicts = verdicts if verdicts is not None else {}
        self.runs = runs if runs is not None else {}
        self.fail = fail
        self.calls = []

    def _check(self, what):
        self.calls.append(what)
        if self.fail:
            raise StoreError(self.fail)

    def stored_results(self, benchmark_ids, subject_id):
        self._check("stored_results")
        return [r for r in self.results if r["benchmark_id"] in set(benchmark_ids) and r["subject_id"] == subject_id]

    def invalidations(self, run_ids):
        return [e for e in self.events if e.get("run_id") in set(run_ids)]

    def resolve_vs(self, kind, value, product, purposes=(), arch=""):
        self._check("resolve_vs")
        want = {"build": "build", "artifact": "artifact_id", "tag": "tags", "ref": "ref_key"}.get(kind)
        out = []
        for t in self.targets:
            if kind == "main":
                hit = t.get("source") in ("nightly", "main")
            elif kind == "tag":
                hit = value in t.get("tags", ())
            else:
                hit = t.get(want) == value
            if hit:
                out.append({"source": kind, "ref_key": "", "ref_state": "", **t})
        return sorted(out, key=lambda t: str(t.get("ts")), reverse=True)

    def target_results(self, kind, value, target, benchmark_ids):
        self._check("target_results")
        ids = set(benchmark_ids)
        return [r for r in self.results if r["benchmark_id"] in ids and r["subject_id"] == target["artifact_id"]]

    def landed(self, run_id):
        self._check("landed")
        got = self.landed_ids.get(run_id, set())
        return set(got() if callable(got) else got)

    def verdict_rows(self, run_id):
        self._check("verdict_rows")
        return list(self.verdicts.get(run_id, []))

    def latest_run(self, artifact_id):
        self._check("latest_run")
        return self.runs.get(artifact_id)

    def anchored_rows(self, run_id):
        self._check("anchored_rows")
        return None if self.anchored is None else list(self.anchored.get(run_id, []))

    def series_runs(self, run_id):
        self._check("series_runs")
        return list(self.series.get(run_id, []))

    def series_window(self, row, n):
        self._check("series_window")
        got = [
            w
            for w in self.windows
            if w["series"] == (row["benchmark_key"], row["metric"], row["backend"], row["source_job"])
        ]
        return sorted(got, key=lambda w: w["run_ts"])[-n:]

    def case_rows(self, benchmark_ids):
        self._check("case_rows")
        return [c for c in self.cases if c["benchmark_key"] in set(benchmark_ids)]

    def anchor_rows(self, component, arch, backend, source_job):
        self._check("anchor_rows")
        return [
            r for r in self.anchors
            if (r["component"], r["arch"], r["backend"], r["source_job"]) == (component, arch, backend, source_job)
        ]  # fmt: skip

    def invalidated_runs(self, run_ids):
        self._check("invalidated_runs")
        if self.invalidated is None:
            return super().invalidated_runs(run_ids)
        return set(run_ids) & set(self.invalidated)

    def registered(self, benchmark_ids):
        self._check("registered")
        return None if self.known is None else set(benchmark_ids) & set(self.known)

    def reference_values(self, key, version="current"):
        self._check("reference_values")
        rows = [r for r in self.references if r["ref_key"] == key]
        want = {"current": lambda r: r.get("state") == "current"}.get(
            version, lambda r: str(r["version"]) == str(version)
        )
        return [r for r in rows if want(r)]


def connection_settings(env=None):
    """ClickHouse connection settings from CLICKHOUSE_* (database also BENCH_CLICKHOUSE_DB)."""
    env = os.environ if env is None else env

    def get(name, default=""):
        return (env.get(name) or "").strip() or default

    host = get("CLICKHOUSE_HOST")
    if not host:
        raise StoreError("CLICKHOUSE_HOST is unset; the clickhouse store needs CLICKHOUSE_HOST (and CLICKHOUSE_PASS)")
    port = get("CLICKHOUSE_PORT", "443")
    if not port.isdigit():
        raise StoreError(f"CLICKHOUSE_PORT is not a number: {port!r}")
    db = get("BENCH_CLICKHOUSE_DB") or get("CLICKHOUSE_DB_V2") or get("CLICKHOUSE_DB")
    if not db or not DB_NAME.match(db):
        raise StoreError(
            "database name unset or invalid: set BENCH_CLICKHOUSE_DB (or CLICKHOUSE_DB_V2) to [A-Za-z0-9_]+"
        )
    return {
        "host": host,
        "port": int(port),
        "username": get("CLICKHOUSE_USER", "default"),
        "password": get("CLICKHOUSE_PASS"),
        "secure": get("CLICKHOUSE_SECURE", "1") not in ("0", "false", "no"),
        "database": db,
    }


def validity_word(value):
    """A stored validity in the engine's vocabulary; anything else (absent, misspelt) is invalid."""
    return value if value in engine.VALIDITY_ORDER else "invalid"


def _uuid_or_none(value):
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None


class ClickHouseStore(Store):
    """Reads the v2 tables and views over clickhouse_connect. `client` is injectable for tests."""

    name = "clickhouse"

    def __init__(self, client=None, database=None, env=None):
        self._client = client
        self._env = env
        self._db = database

    def check(self):
        """Raise StoreError now if the connection settings are missing, without connecting."""
        if self._client is None:
            connection_settings(self._env)

    def _connection(self):
        if self._client is None:
            settings = connection_settings(self._env)
            try:
                import clickhouse_connect
            except ImportError as e:
                raise StoreError(f"clickhouse_connect is not importable ({e}); {INSTALL_HINT}") from e
            self._db = settings.pop("database")
            try:
                self._client = clickhouse_connect.get_client(**settings, database=self._db)
            except Exception as e:  # noqa: BLE001 - any connect failure must surface as a store error
                raise StoreError(f"cannot connect to ClickHouse: {type(e).__name__}: {e}") from e
        if self._db is None:
            self._db = connection_settings(self._env)["database"]
        return self._client

    def client(self):
        return self._connection()

    def _rows(self, sql, parameters):
        client = self._connection()
        sql = sql.replace("{db}", f"`{self._db}`")
        try:
            result = client.query(sql, parameters=parameters)
        except Exception as e:  # noqa: BLE001 - any query failure must surface as a store error
            raise StoreError(f"ClickHouse query failed: {type(e).__name__}: {e}") from e
        return [dict(zip(result.column_names, row)) for row in result.result_rows]

    def stored_results(self, benchmark_ids, subject_id):
        subject = _uuid_or_none(subject_id)
        ids = sorted({i for i in (_uuid_or_none(b) for b in benchmark_ids) if i})
        if not subject or not ids:
            return []
        out = []
        for start in range(0, len(ids), ID_BATCH):  # one Array(UUID) parameter must stay under the HTTP limit
            out += self._stored_batch(subject, ids[start : start + ID_BATCH])
        return out

    def _stored_batch(self, subject, ids):
        rows = self._rows(
            "SELECT toString(run_id) AS run_key, toString(benchmark_id) AS benchmark_key, max(run_ts) AS stored_at, "
            "groupArray(run_props) AS props_all, "
            "max(arrayMax(arrayMap(v -> length(v), mapValues(samples)))) AS n "
            "FROM {db}.v_benchmark_results_enriched "
            "WHERE artifact_id = {subject:UUID} AND benchmark_id IN {ids:Array(UUID)} "
            "GROUP BY run_id, benchmark_id",
            {"subject": subject, "ids": ids},
        )
        out = []
        for r in rows:
            props_all = [p or {} for p in (r["props_all"] or [{}])]
            props = props_all[0]
            ts = r["stored_at"]
            if not isinstance(ts, datetime):
                continue  # no usable timestamp: unknown, so never reused
            env_ids = {p.get("env.id") for p in props_all}
            validity = min(
                (validity_word(p.get("run.validity")) for p in props_all), key=engine.VALIDITY_ORDER.index
            )  # one weak row weakens the whole (run, benchmark)
            out.append(
                {
                    "run_id": r["run_key"],
                    "benchmark_id": r["benchmark_key"],
                    "purpose": props.get("run.purpose", ""),
                    "trust": "external" if any(p.get("run.trust") == "external" for p in props_all) else "internal",
                    "subject_id": subject,
                    "env_id": (env_ids.pop() or None) if len(env_ids) == 1 else None,
                    "labels": {k: v for k, v in props.items() if k.startswith(LABEL_PREFIX)},
                    "validity": validity,
                    "n": int(r["n"] or 0),
                    "ts": ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
                }
            )
        return out

    def resolve_vs(self, kind, value, product, purposes=(), arch=""):
        n = {"n": VS_CANDIDATES, "arch": arch}
        if kind == "artifact":
            sql = (
                "SELECT toString(artifact_id) AS artifact_id, component, arch, ts, 'artifact' AS source, '' AS ref_key, "
                "'' AS ref_state FROM {db}.v_artifacts WHERE artifact_id = {v:UUID}"
            )
            params = {"v": _uuid_or_none(value) or str(uuid.UUID(int=0))}
        elif kind == "tag":
            sql = (
                "SELECT toString(artifact_id) AS artifact_id, component, arch, resolved_ts AS ts, 'tag' AS source, "
                "'' AS ref_key, '' AS ref_state FROM {db}.v_tag_resolution WHERE tag = {v:String} AND component = {c:String} "
                "ORDER BY resolved_ts DESC LIMIT {n:UInt32}"
            )
            params = {"v": value, "c": product, **n}
        elif kind == "build":
            sql = (
                "SELECT toString(a.artifact_id) AS artifact_id, a.component AS component, a.arch AS arch, a.ts AS ts, "
                "'build' AS source, '' AS ref_key, '' AS ref_state FROM {db}.v_artifacts AS a "
                "WHERE a.component = {c:String} AND if(a.arch IN ('amd64', 'x86', 'x86-64'), 'x86_64', a.arch) = {arch:String} "
                "AND a.artifact_id IN (SELECT artifact_id FROM {db}.v_artifact_results_published WHERE props['orch_run_key'] = {v:String}) "
                "ORDER BY a.ts DESC LIMIT {n:UInt32}"
            )
            params = {"v": value, "c": product, **n}
        elif kind == "ref":
            sql = (
                "SELECT toString(v.artifact_id) AS artifact_id, 'reference' AS component, v.arch AS arch, "
                "v.registered_at AS ts, 'reference' AS source, v.ref_key AS ref_key, v.state AS ref_state, x.product AS product "
                "FROM {db}.v_bench_references AS v LEFT JOIN "
                "(SELECT artifact_id, any(component) AS product FROM {db}.v_bench_reference_values GROUP BY artifact_id) AS x "
                "ON x.artifact_id = v.artifact_id WHERE v.ref_key = {v:String} "
                "ORDER BY v.registered_at DESC LIMIT {n:UInt32}"
            )
            params = {"v": value, **n}
        else:
            sql = (
                "SELECT toString(e.artifact_id) AS artifact_id, any(a.component) AS component, any(a.arch) AS arch, "
                "max(e.run_ts) AS ts, 'main' AS source, '' AS ref_key, '' AS ref_state "
                "FROM {db}.v_benchmark_results_enriched AS e INNER JOIN {db}.v_artifacts AS a ON a.artifact_id = e.artifact_id "
                "WHERE a.component = {c:String} AND if(a.arch IN ('amd64', 'x86', 'x86-64'), 'x86_64', a.arch) = {arch:String} "
                "AND e.run_props['run.purpose'] IN {p:Array(String)} "
                "AND e.run_props['run.validity'] = 'valid' AND e.run_props['run.trust'] != 'external' "
                "GROUP BY e.artifact_id ORDER BY ts DESC LIMIT {n:UInt32}"
            )
            params = {"c": product, "p": list(purposes) or ["nightly"], **n}
        return self._rows(sql, params)

    def target_results(self, kind, value, target, benchmark_ids):
        if kind != "ref":
            return self.stored_results(benchmark_ids, target["artifact_id"])
        ids = sorted({i for i in (_uuid_or_none(b) for b in benchmark_ids) if i})
        if not ids:
            return []
        rows = self._rows(
            "SELECT toString(artifact_id) AS run_key, toString(benchmark_id) AS benchmark_key, "
            "count() AS n "
            "FROM {db}.v_bench_reference_values WHERE ref_key = {k:String} AND state = 'current' "
            "AND benchmark_id IN {ids:Array(UUID)} GROUP BY artifact_id, benchmark_id",
            {"k": value, "ids": ids},
        )
        return [
            {
                "run_id": r["run_key"],
                "benchmark_id": r["benchmark_key"],
                "subject_id": target["artifact_id"],
                "env_id": None,
                "labels": {},
                "validity": "valid",
                "trust": "internal",
                "purpose": "reference",
                "reference": True,
                "n": int(r["n"] or 0),
                "ts": target.get("ts") if isinstance(target.get("ts"), datetime) else datetime.now(timezone.utc),
            }
            for r in rows
        ]

    def landed(self, run_id):
        run = _uuid_or_none(run_id)
        if not run:
            raise StoreError(f"run id {run_id!r} is not a UUID")
        rows = self._rows(
            "SELECT DISTINCT toString(benchmark_id) AS benchmark_key FROM {db}.benchmark_runs WHERE run_id = {run:UUID}",
            {"run": run},
        )
        return {r["benchmark_key"] for r in rows}

    def latest_run(self, artifact_id):
        artifact = _uuid_or_none(artifact_id)
        if not artifact:
            raise StoreError(f"artifact id {artifact_id!r} is not a UUID")
        rows = self._rows(
            "SELECT toString(run_id) AS run_key FROM {db}.v_benchmark_metric_verdicts "
            "WHERE run_id IN (SELECT run_id FROM {db}.v_benchmark_results_enriched WHERE artifact_id = {artifact:UUID}) "
            "GROUP BY run_id ORDER BY max(ts) DESC LIMIT 1",
            {"artifact": artifact},
        )
        return rows[0]["run_key"] if rows else None

    def invalidated_runs(self, run_ids):
        ok = {u for u in (_uuid_or_none(r) for r in run_ids) if u}
        bad = sorted(str(r) for r in run_ids if _uuid_or_none(r) is None)
        if bad:
            raise StoreError(f"run id(s) {bad} are not UUIDs, so their invalidation cannot be checked")
        if not ok:
            return set()
        rows = self._rows(
            "SELECT DISTINCT toString(run_id) AS run_key FROM {db}.v_invalidated_runs "
            "WHERE state = 'invalidated' AND run_id IN {ids:Array(UUID)}",
            {"ids": sorted(ok)},
        )
        return {r["run_key"] for r in rows}

    def registered(self, benchmark_ids):
        ids = sorted({i for i in (_uuid_or_none(b) for b in benchmark_ids) if i})
        found = set()
        for start in range(0, len(ids), ID_BATCH):
            rows = self._rows(
                "SELECT DISTINCT toString(benchmark_id) AS benchmark_key FROM {db}.benchmarks "
                "WHERE benchmark_id IN {ids:Array(UUID)}",
                {"ids": ids[start : start + ID_BATCH]},
            )
            found |= {r["benchmark_key"] for r in rows}
        return found

    def reference_values(self, key, version="current"):
        where = "state = 'current'" if version == "current" else "version = {version:String}"
        return self._rows(
            "SELECT ref_key, version, state, toString(artifact_id) AS artifact_id, origin, label, comparability, "
            "clean, warnings, component, toString(benchmark_id) AS benchmark_id, backend, measurements, iterations, "
            "row_props FROM {db}.v_bench_reference_values WHERE ref_key = {key:String} AND " + where,
            {"key": key, "version": str(version)},
        )

    def verdict_rows(self, run_id):
        run = _uuid_or_none(run_id)
        if not run:
            raise StoreError(f"run id {run_id!r} is not a UUID")
        return self._rows(
            "SELECT toString(run_id) AS run_key, toString(benchmark_id) AS benchmark_key, name, component, backend, "
            "arch, source_job, metric, tier, status, reason, baseline_kind, n_baseline, n_current, baseline_value, "
            "current_value, delta_pct, worse_pct, floor_pct "
            "FROM {db}.v_benchmark_metric_verdicts WHERE run_id = {run:UUID} ORDER BY name, metric",
            {"run": run},
        )

    def anchored_rows(self, run_id):
        run = _uuid_or_none(run_id)
        if not run:
            raise StoreError(f"run id {run_id!r} is not a UUID")
        try:
            return self._rows(
                "SELECT toString(run_id) AS run_key, toString(benchmark_id) AS benchmark_key, name, backend, arch, source_job, "
                "metric, status, reason, version, anchor_value, current_value, delta_pct, worse_pct, floor_pct, since_ts, n_exceeding_runs, "
                "days_exceeding "
                "FROM {db}.v_benchmark_anchored_verdicts WHERE run_id = {run:UUID} ORDER BY name, metric",
                {"run": run},
            )
        except StoreError as e:
            if missing_object(e, "v_benchmark_anchored_verdicts"):
                return None  # a database without schema 88: no anchored verdicts, not an error
            raise

    def series_runs(self, run_id):
        run = _uuid_or_none(run_id)
        if not run:
            raise StoreError(f"run id {run_id!r} is not a UUID")
        return self._rows(
            "SELECT toString(run_id) AS run_key, toString(benchmark_id) AS benchmark_key, name, component, backend, arch, "
            "source_job, metric, cur AS samples, eligible, off_purpose, suspect, run_ts "
            "FROM {db}.v_benchmark_series_runs WHERE run_id = {run:UUID} ORDER BY name, metric",
            {"run": run},
        )

    def series_window(self, row, n):
        bench, run = _uuid_or_none(row["benchmark_key"]), _uuid_or_none(row.get("run_key"))
        if not bench or not run:
            raise StoreError(f"benchmark or run id of {row.get('name')!r} is not a UUID")
        rows = self._rows(
            "SELECT toString(run_id) AS run_key, run_ts, cur AS samples FROM {db}.v_benchmark_series_runs "
            "WHERE benchmark_id = {b:UUID} AND component = {c:String} AND arch = {a:String} AND backend = {be:String} "
            "AND source_job = {j:String} AND metric = {m:String} AND eligible AND (run_ts, run_id) <= "
            "(SELECT (min(run_ts), run_id) FROM {db}.v_benchmark_series_runs WHERE run_id = {run:UUID} "
            "AND benchmark_id = {b:UUID} AND metric = {m:String} AND backend = {be:String} AND source_job = {j:String} "
            "AND arch = {a:String} GROUP BY run_id) "
            "ORDER BY run_ts DESC, run_id DESC LIMIT {n:UInt32}",
            {"b": bench, "run": run, "c": row["component"], "a": row["arch"], "be": row["backend"], "j": row["source_job"],
             "m": row["metric"], "n": n},
        )  # fmt: skip
        return sorted(rows, key=lambda r: (r["run_ts"], r["run_key"]))

    def anchor_rows(self, component, arch, backend, source_job):
        return self._rows(
            "SELECT toString(benchmark_id) AS benchmark_key, component, arch, backend, source_job, metric, "
            "anchor_samples AS samples, anchor_from, version "
            "FROM {db}.v_benchmark_anchor WHERE component = {c:String} AND arch = {a:String} "
            "AND backend = {b:String} AND source_job = {j:String} ORDER BY benchmark_key, metric",
            {"c": component, "a": arch, "b": backend, "j": source_job},
        )

    def case_rows(self, benchmark_ids):
        ids = sorted({i for i in (_uuid_or_none(b) for b in benchmark_ids) if i})
        out = []
        for start in range(0, len(ids), ID_BATCH):
            out += self._rows(
                "SELECT toString(benchmark_id) AS benchmark_key, component, name, tags, props "
                "FROM {db}.benchmarks WHERE benchmark_id IN {ids:Array(UUID)} ORDER BY benchmark_id",
                {"ids": ids[start : start + ID_BATCH]},
            )
        return out


def open_store(name, client=None):
    """Store by name: none | clickhouse."""
    if name in (None, "", "none"):
        return NullStore()
    if name == "clickhouse":
        return ClickHouseStore(client=client)
    raise StoreError(f"unknown store {name!r} (none | clickhouse)")
