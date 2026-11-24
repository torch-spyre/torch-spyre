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

"""Deterministic synthetic Perf Lab data (synthetic names only) for tests, sandboxes and the UI prototype.

`build(seed)` returns {table: [row dicts]} in JSONEachRow shape; `load(tables, sink)` inserts them in dependency
order through any `sink(table, rows)`. Everything is a pure function of the seed, so ids and values are stable.
Shapes: 3 archs, 2 components, 40 benchmarks, 60 nightly runs per series, a regression at run 45 on two
benchmarks, a subject change at run 30, a pinned/unpinned A/B, a TP4 multi-card run (4 ranks x 64 steps, spans),
invalidations, a reference target, and environment fingerprints and placements for every run."""

import json
import random
import uuid
from datetime import datetime, timedelta, timezone

NS = uuid.UUID("5e1f0000-0000-5000-8000-000000000f1e")
BASE = datetime(
    2026, 9, 16, 2, 0, tzinfo=timezone.utc
)  # night 1; night 60 is 2026-11-14
NIGHTS = 60
REGRESSION_NIGHT = 45
SUBJECT_NIGHT = 30
FW_NIGHT = 38
ARCHS = ("x86_64", "ppc64le", "s390x")
ARCH_FACTOR = {"x86_64": 1.0, "ppc64le": 1.15, "s390x": 1.4}
Z = "00000000-0000-0000-0000-000000000000"
TS = "synth-torch-spyre"
SI = "synth-inference"

# (name, config) of 28 torch-spyre benchmarks; the first is regressed at night 45
TS_BENCHES = [("synth_matmul_bf16_1024", "m1024")] + [
    (f"synth_{op}_{sz}", sz)
    for op in (
        "add",
        "layernorm",
        "softmax",
        "gelu",
        "conv2d",
        "attention",
        "embedding",
    )
    for sz in ("s", "m", "l", "xl")
][:27]
# (name, metrics): 12 spyre-inference benchmarks; the first is regressed at night 45
SI_BENCHES = [
    ("synth_serve_tp4_c8", ("mean_tpot_ms", "mean_ttft_ms", "output_throughput")),
    ("synth_serve_tp1_c1", ("mean_tpot_ms", "mean_ttft_ms", "output_throughput")),
    ("synth_serve_tp1_c16", ("mean_tpot_ms", "mean_ttft_ms", "output_throughput")),
    ("synth_serve_tp2_c4", ("mean_tpot_ms", "mean_ttft_ms", "output_throughput")),
    ("synth_serve_tp2_c8", ("mean_tpot_ms", "mean_ttft_ms", "output_throughput")),
    ("synth_serve_tp4_c1", ("mean_tpot_ms", "mean_ttft_ms", "output_throughput")),
    ("synth_serve_tp4_c16", ("mean_tpot_ms", "mean_ttft_ms", "output_throughput")),
    ("synth_embed_len512", ("mean_ttft_ms", "request_throughput")),
    ("synth_embed_len2048", ("mean_ttft_ms", "request_throughput")),
    ("synth_startup_cold", ("total_duration_ms",)),
    ("synth_lat_tp4_out256", ("latency_ms",)),
    ("synth_lat_tp4_out1", ("latency_ms",)),
]
TP4_CASE = "synth_serve_tp4_c8"
AB_BENCH, AB_BENCH_1 = "synth_lat_tp4_out256", "synth_lat_tp4_out1"
TP4_TRACE = "synth_tp4_decode_trace"
SPAN_PATTERNS = (
    ("host", "synth.host.launch"),
    ("host", "synth.host.sync"),
    ("device", "synth.dev.attn_matmul"),
    ("device", "synth.dev.mlp_matmul"),
    ("collective", "synth.coll.allreduce_attn"),
    ("collective", "synth.coll.allreduce_mlp"),
)
SPAN_STEPS = (0, 1, 2, 3, 30, 31, 32, 63)
LAYERS = 40
RANKS = 4
STEPS = 64
SLOW_RANK = 2
NODE_CLASSES = {
    "synth-xeon-2s": {
        "host.sockets": "2",
        "host.numa_nodes": "4",
        "host.cores_per_socket": "32",
        "host.smt": "on",
    },
    "synth-xeon-1s": {
        "host.sockets": "1",
        "host.numa_nodes": "2",
        "host.cores_per_socket": "48",
        "host.smt": "off",
    },
    "synth-power-2s": {
        "host.sockets": "2",
        "host.numa_nodes": "2",
        "host.cores_per_socket": "20",
        "host.smt": "on",
    },
    "synth-z-1s": {
        "host.sockets": "1",
        "host.numa_nodes": "1",
        "host.cores_per_socket": "16",
        "host.smt": "on",
    },
}
NODES = {  # node -> (node class, arch)
    "node-x86-01": ("synth-xeon-2s", "x86_64"),
    "node-x86-02": ("synth-xeon-2s", "x86_64"),
    "node-x86-11": ("synth-xeon-1s", "x86_64"),
    "node-p-01": ("synth-power-2s", "ppc64le"),
    "node-z-01": ("synth-z-1s", "s390x"),
}


def uid(*parts):
    return str(uuid.uuid5(NS, "|".join(str(p) for p in parts)))


def fmt(dt, ms=False):
    return dt.strftime("%Y-%m-%d %H:%M:%S") + (
        f".{dt.microsecond // 1000:03d}" if ms else ""
    )


def night(i):
    return BASE + timedelta(days=i - 1)


def fingerprint_of(node_class, fw="fw-1.0", pinning="none"):
    fp = {
        **NODE_CLASSES[node_class],
        "host.kernel": "synth-6.1",
        "host.governor": "performance",
        "host.cpu_pinning": pinning,
        "device.firmware": fw,
        "device.pcie_gen": "5",
        "device.numa_node": "0",
    }
    labels = {"node_class": node_class, "cpu_pinning": pinning}
    return uid("fp", sorted(fp.items())), fp, labels


class Gen:
    def __init__(self, seed=20261115):
        self.rng = random.Random(seed)
        self.t = {k: [] for k in (
            "artifacts", "artifact_tags", "artifact_results", "artifact_refs", "result_uploads", "benchmarks",
            "benchmark_runs", "environments", "node_classes", "environment_fingerprints", "environment_nodes",
            "environment_qualifications", "run_placements", "result_invalidations")}  # fmt: skip
        self.benches = {}  # name -> (benchmark_id, component)
        self.envs = {}
        self.artifacts = {}

    def add(self, table, **row):
        self.t[table].append(row)

    # ---- dimensions
    def bench(self, component, name, **props):
        bid = uid("bench", component, name, sorted(props.items()))
        if name in self.benches:
            return bid
        self.benches[name] = (bid, component)
        self.add("benchmarks", ts=fmt(BASE - timedelta(days=30)), benchmark_id=bid, component=component, name=name,
                 tags=["synthetic"], props=props)  # fmt: skip
        return bid

    def env(self, node_class, fw="fw-1.0", pinning="none"):
        env_id, fp, labels = fingerprint_of(node_class, fw, pinning)
        if env_id not in self.envs:
            self.envs[env_id] = (node_class, fw, pinning)
            self.add("environment_fingerprints", ts=fmt(BASE - timedelta(days=30), True), env_id=env_id,
                     fingerprint_version=1, labels=labels, fingerprint=fp, members=[], devices="[]")  # fmt: skip
        return env_id

    def artifact(self, component, arch, name, build, kind="image", **props):
        aid = uid("artifact", component, arch, name, build)
        if aid not in self.artifacts:
            self.artifacts[aid] = (component, arch, name, build)
            self.add("artifacts", ts=fmt(BASE - timedelta(days=31)), artifact_id=aid, component=component, arch=arch,
                     kind=kind, artifact_name=name, origin="built", identity_deps=[f"synth-toolchain@{build[:8]}"],
                     props={"id12": aid.replace("-", "")[:12], "build": build, **props})  # fmt: skip
        return aid

    # ---- measurements
    def samples(self, base, cv, n):
        return [round(base * max(0.2, 1 + self.rng.gauss(0, cv)), 4) for _ in range(n)]

    def run_row(
        self,
        run_id,
        bid,
        component,
        ts,
        arch,
        repeat,
        metrics,
        props,
        backend="spyre",
        iterations=5,
    ):
        self.add("benchmark_runs", ts=fmt(ts + timedelta(hours=1)), run_id=run_id, benchmark_id=bid, component=component,
                 backend=backend, measurements=metrics, iterations=iterations,
                 props={"arch": arch, "run.repeat": str(repeat), "run_utc": fmt(ts), **props})  # fmt: skip

    def placement(self, run_id, repeat, ts, env_id, node, arch, clean):
        dev = f"SYN-{node.upper()}-{repeat % 4}"
        devs = [
            {
                "node": node,
                "serial": dev,
                "pci": f"0000:{repeat % 4:02x}:00.0",
                "firmware": "fw",
                "link": "pcie",
            }
        ]
        self.add("run_placements", ts=fmt(ts + timedelta(hours=1), True), run_id=run_id, repeat=repeat,
                 started_at=fmt(ts + timedelta(minutes=5 * repeat), True), env_id=env_id,
                 placement_id=uid("placement", node, dev), nodes=[node], devices=devs, links=[], clean=clean)  # fmt: skip

    # ---- scenarios
    def nightly(self):
        specs = []  # (component, arch, benches)
        for arch in ARCHS:
            specs.append(
                (
                    TS,
                    arch,
                    [
                        (n, ("total_duration_ms", "kernel_mean_ms"))
                        for n, _ in TS_BENCHES
                    ],
                )
            )
        specs.append(
            (
                SI,
                "x86_64",
                [(n, m) for n, m in SI_BENCHES if not n.startswith("synth_lat")],
            )
        )
        nodes = {
            "x86_64": ["node-x86-01", "node-x86-02"],
            "ppc64le": ["node-p-01"],
            "s390x": ["node-z-01"],
        }
        base = {}
        for comp, _, benches in specs:
            for n, metrics in benches:
                for m in metrics:
                    base.setdefault(
                        (n, m),
                        self.rng.uniform(2, 80)
                        if "throughput" not in m
                        else self.rng.uniform(300, 900),
                    )
        cv = {n: self.rng.uniform(0.004, 0.02) for _, _, bs in specs for n, _ in bs}
        for comp, arch, benches in specs:
            ids = {n: self.bench(comp, n, record_type="op" if comp == TS else "serving", config_name=n.rsplit("_", 1)[-1])
                   for n, _ in benches}  # fmt: skip
            for i in range(1, NIGHTS + 1):
                ts = night(i)
                run_id = uid("run", comp, arch, i)
                node = nodes[arch][i % len(nodes[arch])]
                nc = NODES[node][0]
                fw = "fw-1.1" if (arch == "x86_64" and i >= FW_NIGHT) else "fw-1.0"
                env_id = self.env(nc, fw)
                build = (
                    "build-b"
                    if (arch == "x86_64" and i >= SUBJECT_NIGHT)
                    else "build-a"
                )
                aid = self.artifact(comp, arch, f"synth-{comp}-devel", build)
                self.add("artifact_results", ts=fmt(ts + timedelta(hours=2)), artifact_id=aid, run_id=run_id,
                         result_kind="performance", test_type="perf", state="passed", arch=arch, duration_s=900.0,
                         props={"run_url": f"https://ci.example.invalid/synth/{comp}/{arch}/{i}"})  # fmt: skip
                self.add("artifact_tags", ts=fmt(ts + timedelta(hours=2)), tag=f"nightly-{ts:%Y-%m-%d}",
                         tag_family="nightly", artifact_id=aid, refs=[], published_refs=[], props={})  # fmt: skip
                if i == NIGHTS:
                    self.add("artifact_tags", ts=fmt(ts + timedelta(hours=2)), tag="nightly", tag_family="nightly",
                             artifact_id=aid, refs=[], published_refs=[], props={})  # fmt: skip
                clean = {
                    "exclusive": "observed" if i % 9 else "none",
                    "co_tenants": "unknown",
                    "idle_cpu_before": "97.4",
                }
                rprops = {
                    "env.id": env_id, "env.name": f"synth-env-{nc}", "run.tier": "nightly", "run.suite.name": "synth-core" if comp == TS else "synth-serve",
                    "run.purpose": "nightly", "run.trust": "ci", "run.invoker": "gha", "run.trigger": "schedule",
                    "run.validity": "valid", "run.profile_level": "off",
                    "run.clean.exclusive": clean["exclusive"], "run.exec.cpu_pinning": "none",
                    "run.exec.pinning_verified": "unknown",
                    "source_file": f"synth-nightly-perf-{arch}/{comp}/results.json",
                }  # fmt: skip
                for repeat in (1, 2, 3):
                    self.placement(run_id, repeat, ts, env_id, node, arch, clean)
                    rep_shift = self.rng.gauss(0, 0.004)
                    for n, metrics in benches:
                        met = {}
                        for m in metrics:
                            v = base[(n, m)] * ARCH_FACTOR[arch] * (1 + rep_shift)
                            if (
                                arch == "x86_64"
                                and i >= REGRESSION_NIGHT
                                and n in ("synth_matmul_bf16_1024", TP4_CASE)
                            ):
                                if m in (
                                    "total_duration_ms",
                                    "kernel_mean_ms",
                                    "mean_tpot_ms",
                                    "mean_ttft_ms",
                                ):
                                    v *= 1.12
                                if m == "output_throughput":
                                    v *= 0.90
                            if (
                                arch == "x86_64"
                                and i >= SUBJECT_NIGHT
                                and n in ("synth_gelu_m", "synth_gelu_l", "synth_add_s")
                            ):
                                v *= 0.94
                            met[m] = self.samples(v, cv[n], 5)
                        self.run_row(
                            run_id, ids[n], comp, ts, arch, repeat, met, rprops
                        )
                if i in (20,) and arch == "x86_64" and comp == TS:
                    self.invalidate_run(
                        run_id,
                        ts,
                        "run",
                        "bad_card",
                        "card thermal throttling reported by the node",
                    )
                if i == 52 and arch == "x86_64" and comp == TS:
                    self.invalidate_run(
                        run_id,
                        ts,
                        "run",
                        "noisy_neighbour",
                        "unexpected co-tenant load",
                        state="suspect",
                    )
        # one node-scope event touches the ppc64le night 10 run through its placement
        self.t["result_invalidations"].append(self._inv(
            uid("inv", "node"), night(10) + timedelta(hours=20), "node", None, node="node-p-01",
            window_from=fmt(night(10) - timedelta(hours=1), True), window_to=fmt(night(10) + timedelta(minutes=7), True),
            reason_code="governor_powersave", reason="node ran with the powersave governor", state="invalidated"))  # fmt: skip

    def _inv(
        self,
        inv_id,
        ts,
        kind,
        run_id,
        state="invalidated",
        reason_code="",
        reason="",
        **kw,
    ):
        row = {"ts": fmt(ts, True), "invalidation_id": inv_id, "scope_kind": kind, "run_id": run_id, "state": state,
               "reason_code": reason_code, "reason": reason, "actor": "synth-owner@example.invalid"}  # fmt: skip
        row.update(kw)
        return row

    def invalidate_run(self, run_id, ts, kind, code, reason, state="invalidated"):
        self.t["result_invalidations"].append(
            self._inv(
                uid("inv", run_id),
                ts + timedelta(days=1),
                kind,
                run_id,
                state,
                code,
                reason,
            )
        )

    def ab(self):
        """Pinned/unpinned A/B on out256 and out1: 2 node classes x 2 policies x 5 runs."""
        ids = {
            n: self.bench(
                SI, n, record_type="serving", config_name=n.rsplit("_", 1)[-1]
            )
            for n in (AB_BENCH, AB_BENCH_1)
        }
        aid = self.artifact(SI, "x86_64", f"synth-{SI}-devel", "build-b")
        k = 0
        for nc, node, effect in (
            ("synth-xeon-2s", "node-x86-01", 0.82),
            ("synth-xeon-1s", "node-x86-11", 1.0),
        ):
            for pinning in ("none", "device_local"):
                env_id = self.env(nc, "fw-1.1", pinning)
                for j in range(5):
                    k += 1
                    ts = night(51) + timedelta(hours=6 + k)
                    run_id = uid("run", "ab", nc, pinning, j)
                    self.add("artifact_results", ts=fmt(ts + timedelta(hours=2)), artifact_id=aid, run_id=run_id,
                             result_kind="performance", test_type="perf", state="passed", arch="x86_64", duration_s=300.0, props={})  # fmt: skip
                    clean = {"exclusive": "enforced", "co_tenants": "0/0"}
                    rprops = {
                        "env.id": env_id, "env.name": f"synth-env-{nc}", "run.tier": "experiment", "run.suite.name": "synth-pinning-ab",
                        "run.purpose": "experiment", "run.trust": "lab", "run.invoker": "shell", "run.validity": "valid",
                        "run.clean.exclusive": "enforced", "run.exec.cpu_pinning": pinning,
                        "run.exec.cpu_set_source": "device_numa" if pinning != "none" else "none",
                        "run.exec.pinning_verified": "true" if pinning != "none" else "unknown",
                        "source_file": "synth-ab-perf-x86_64/synth/results.json",
                    }  # fmt: skip
                    f = effect if pinning == "device_local" else 1.0
                    for repeat in (1, 2, 3):
                        self.placement(
                            run_id, repeat, ts, env_id, node, "x86_64", clean
                        )
                        for n, lat in ((AB_BENCH, 2600.0), (AB_BENCH_1, 31.0)):
                            self.run_row(run_id, ids[n], SI, ts, "x86_64", repeat,
                                         {"latency_ms": self.samples(lat * f, 0.008, 5)}, rprops)  # fmt: skip

    def tp4(self):
        """One TP4 run: step values per rank (64 steps) and spans (40 layers x 6 patterns over SPAN_STEPS)."""
        ts = night(58)
        run_id = uid("run", "tp4")
        env_id = self.env("synth-xeon-2s", "fw-1.1")
        aid = self.artifact(SI, "x86_64", f"synth-{SI}-devel", "build-b")
        self.add("artifact_results", ts=fmt(ts + timedelta(hours=2)), artifact_id=aid, run_id=run_id,
                 result_kind="performance", test_type="perf", state="passed", arch="x86_64", duration_s=600.0, props={})  # fmt: skip
        case = self.bench(
            SI,
            "synth_serve_tp4_trace",
            record_type="serving",
            config_name="trace",
            world_size="4",
        )
        trace = self.bench(
            SI, TP4_TRACE, record_type="step_trace", world_size="4", case_id=case
        )
        span_ids = {p: self.bench(SI, f"synth_span_{p[1]}", record_type="span", lane=p[0], kernel_name=p[1], case_id=case)
                    for p in SPAN_PATTERNS}  # fmt: skip
        common = {
            "env.id": env_id, "env.name": "synth-env-synth-xeon-2s", "run.tier": "experiment", "run.suite.name": "synth-multicard", "run.purpose": "experiment",
            "run.trust": "lab", "run.validity": "valid", "run.world_size": "4", "run.group_id": uid("group", "tp4")[:8],
            "run.repeat": "1", "run.prompt": "", "run.clean.exclusive": "observed", "run.reporter_version": "layout-2",
            "run.reporter_version_source": "inferred", "run.trace_uri": "https://assets.example.invalid/synth/tp4.json",
            "source_file": "synth-multicard-perf-x86_64/synth/trace.json",
            "role.total": "step_ms", "role.wait": "comms_wait_ms", "role.compute": "device_compute_ms",
            "role.busy": "device_busy_ms", "agg.step_ms": "max", "agg.comms_wait_ms": "mean",
            "agg.device_compute_ms": "mean", "agg.device_busy_ms": "mean",
        }  # fmt: skip
        clean = {"exclusive": "observed"}
        self.placement(run_id, 1, ts, env_id, "node-x86-01", "x86_64", clean)
        self.run_row(run_id, case, SI, ts, "x86_64", 1,
                     {"mean_tpot_ms": self.samples(61.0, 0.01, 5), "mean_ttft_ms": self.samples(310.0, 0.01, 5)}, common)  # fmt: skip
        rng = self.rng
        totals = {}
        for r in range(RANKS):
            tot, wait, comp, busy = [], [], [], []
            for s in range(STEPS):
                base = (
                    155.0
                    + (8.0 if r == SLOW_RANK and rng.random() < 0.8 else 0.0)
                    + rng.gauss(0, 1.2)
                )
                if s == 0:
                    base = 310.0 + (30 if r == SLOW_RANK else 0)
                if s == 30:
                    base += 60
                if r == 1 and s in (12, 47):
                    base += 40  # host stall outliers on one card
                tot.append(round(base, 3))
                c = 43.5 + (1.2 if r == SLOW_RANK else 0) + rng.gauss(0, 0.3)
                comp.append(round(c, 3))
                busy.append(round(c + 17 + rng.gauss(0, 0.3), 3))
                wait.append(
                    round(
                        2.0 + rng.random()
                        if r == SLOW_RANK
                        else 20.0 + rng.gauss(0, 2) + (60 if s == 30 else 0),
                        3,
                    )
                )
            totals[r] = tot
            m = {
                "step_index": [float(s) for s in range(STEPS)],
                "step_kind": [0.0] + [1.0] * (STEPS - 1),
                "cold": [1.0] + [0.0] * (STEPS - 1),
                "step_ms": tot, "comms_wait_ms": wait, "device_compute_ms": comp, "device_busy_ms": busy,
                "summary.itl.step_ms": [round(sum(tot[1:]) / (STEPS - 1), 3)],
            }  # fmt: skip
            self.run_row(
                run_id,
                trace,
                SI,
                ts,
                "x86_64",
                1,
                m,
                {**common, "run.rank": str(r)},
                iterations=STEPS,
            )
        for p in SPAN_PATTERNS:
            for r in range(RANKS):
                cols = {
                    k: []
                    for k in (
                        "step",
                        "seq",
                        "parent_seq",
                        "depth",
                        "layer",
                        "start_ns",
                        "dur_ns",
                    )
                }
                for s in SPAN_STEPS:
                    t0 = 0
                    for layer in range(LAYERS):
                        slot = [q[1] for q in SPAN_PATTERNS].index(p[1])
                        dur = {
                            "host": 40_000,
                            "device": 700_000,
                            "collective": 150_000,
                        }[p[0]]
                        if p[0] == "device" and r == SLOW_RANK:
                            dur = int(dur * 1.08)
                        if p[0] == "collective":
                            dur = (
                                60_000
                                if r == SLOW_RANK
                                else 150_000 + 5_000 * (layer % 3)
                            )
                            if s == 30:
                                dur *= 4 if r != SLOW_RANK else 1
                        dur = int(dur * (1 + rng.gauss(0, 0.03)))
                        for k, v in zip(
                            cols,
                            (
                                s,
                                layer * 6 + slot,
                                -1,
                                0,
                                layer,
                                t0 + slot * 900_000,
                                dur,
                            ),
                        ):
                            cols[k].append(float(v))
                        t0 += 6 * 900_000
                self.run_row(run_id, span_ids[p], SI, ts, "x86_64", 1, cols, {**common, "run.rank": str(r)},
                             iterations=len(cols["step"]))  # fmt: skip

    def references(self):
        """One approved reference version and the value rows (comparison targets, never baselines)."""
        aid = self.artifact(
            "reference",
            "x86_64",
            "perf.synth-lm.decode.ref-system.latency",
            "v1",
            kind="generic",
        )
        ts = night(40)
        rp = {"ref.key": "perf.synth-lm.decode.ref-system.latency", "ref.version": "1", "ref.origin": "uploaded",
              "ref.class": "perf", "ref.label": "Synthetic reference system", "ref.approved_by": "synth-approver",
              "ref.approved_at": fmt(ts), "ref.submitted_by": "synth-submitter", "ref.as_of": "2026-10-01",
              "ref.comparability.workload": "same", "ref.comparability.precision": "same",
              "ref.comparability.hardware": "different", "ref.terms": "internal-synthetic",
              "ref.visibility": "internal"}  # fmt: skip
        self.artifacts[aid] = ("reference", "x86_64", rp["ref.key"], "v1")
        for row in reversed(self.t["artifacts"]):
            if row["artifact_id"] == aid:
                row["props"].update(rp)
                break
        run_id = uid("run", "reference")
        upload = uid("upload", "reference")
        self.add("artifact_tags", ts=fmt(ts), tag="ref-current", tag_family="reference", artifact_id=aid, refs=[],
                 published_refs=[], props={"promoted_by": "synth-approver", "ref.key": rp["ref.key"], "ref.version": "1"})  # fmt: skip
        self.add("artifact_results", ts=fmt(ts), artifact_id=aid, run_id=run_id, result_kind="performance",
                 test_type="perf", state="passed", arch="x86_64", duration_s=0.0, props={"upload_id": upload})  # fmt: skip
        self.add("result_uploads", ts=fmt(ts, True), upload_id=upload, state="published", actor="synth-approver",
                 artifact_id=aid, run_id=run_id, format="benchmarks", result_kind="performance", test_type="perf",
                 arch="x86_64", filename="reference.json", size_bytes=1, content_sha256="0" * 64, bundle_url="", props={})  # fmt: skip
        for n, lat in ((AB_BENCH, 2300.0), (AB_BENCH_1, 29.0)):
            self.run_row(run_id, self.benches[n][0], SI, ts, "x86_64", 1, {"latency_ms": [lat] * 3},
                         {**rp, "run.ref_only": "1", "run.record_type": "reference", "run.trust": "external",
                          "upload_id": upload, "source_file": "synth-reference/ref.json"}, backend="ref:synthetic")  # fmt: skip

    def environments(self):
        t0 = BASE - timedelta(days=30)
        for nc, hp in NODE_CLASSES.items():
            self.add("node_classes", ts=fmt(t0, True), name=nc, spec_digest=uid("nc", nc)[:12], host_profile=hp,
                     qualification={"expiry_days": "30", "cv_limit_pct": "1.5", "probe": "synth-canary"})  # fmt: skip
        for env_id, (nc, fw, pin) in self.envs.items():
            node_names = [n for n, (c, _) in NODES.items() if c == nc]
            for node in node_names:
                first = (
                    night(1) if fw == "fw-1.0" else night(FW_NIGHT) - timedelta(hours=1)
                )
                if fw == "fw-1.1" and nc != "synth-xeon-2s":
                    first = night(1)
                self.add("environment_nodes", ts=fmt(first, True), node=node, env_id=env_id, node_class=nc, valid_from=fmt(first, True))  # fmt: skip
                q_at = first + timedelta(minutes=30)
                expired = nc == "synth-xeon-1s"
                self.add("environment_qualifications", ts=fmt(q_at, True), node=node, env_id=env_id, probe="synth-canary",
                         outcome="fail" if (nc == "synth-z-1s") else "pass", cv_pct=0.8 if nc != "synth-z-1s" else 2.4,
                         limit_pct=1.5, repeats=5, checks={"governor": "ok"}, evidence_run_id=uid("canary", node, env_id),
                         qualified_at=fmt(q_at, True),
                         expires_at=fmt(q_at + timedelta(days=30 if not expired else 20), True))  # fmt: skip
        for nc in NODE_CLASSES:
            self.add("environments", ts=fmt(t0, True), name=f"synth-env-{nc}", spec_digest=uid("env", nc)[:12],
                     labels={"node_class": nc}, devices="[]", host="{}", policy={"validity_floor": "valid"})  # fmt: skip


def build(seed=20261115):
    g = Gen(seed)
    g.nightly()
    g.ab()
    g.tp4()
    g.references()
    g.environments()
    return g.t


ORDER = (
    "benchmarks", "artifacts", "artifact_tags", "artifact_refs", "artifact_results", "result_uploads",
    "environments", "node_classes", "environment_fingerprints", "environment_nodes", "environment_qualifications",
    "run_placements", "result_invalidations", "benchmark_runs",
)  # fmt: skip


def load(tables, sink, batch=5000):
    for name in ORDER:
        rows = tables.get(name, [])
        for i in range(0, len(rows), batch):
            sink(name, rows[i : i + batch])


SCHEMA_FILES = (
    "10-functional-tests", "20-artifacts", "30-benchmarks", "49-result-uploads", "50-artifact-views", "60-benchmark-views",
    "62-benchmark-verdicts", "81-bench-environments", "82-run-placements", "83-result-invalidations", "85-benchmark-verdict-views", "86-bench-references", "87-bench-trace-views", "88-bench-anchored-verdicts", "90-bench-span-values",
)  # fmt: skip


def schema_statements():
    """The DDL of SCHEMA_FILES in order, refreshable MVs last (62's MV reads the verdict views of 85)."""
    from .apply_schema import SCHEMA_DIR, SchemaApplier

    stmts = [
        st
        for n in SCHEMA_FILES
        for st in SchemaApplier.statements((SCHEMA_DIR / f"{n}.sql").read_text())
    ]
    return [x for x in stmts if "REFRESH EVERY" not in x] + [
        x for x in stmts if "REFRESH EVERY" in x
    ]


# What the refreshable MV in 62 appends, run once over the whole fixture (a sandbox waits up to 30 minutes for the refresh).
VERDICT_FILL_SQL = """INSERT INTO benchmark_metric_verdicts
   (run_id, benchmark_id, component, backend, arch, source_job, ts, metric, tier, baseline_kind, baseline_run_ids, n_baseline,
    n_current, baseline_value, current_value, delta_pct, worse_pct, score, floor_pct, status, reason, policy_version)
   SELECT run_id, benchmark_id, component, backend, arch, source_job, ts, metric, tier, baseline_kind, baseline_run_ids,
          n_baseline, n_current, baseline_value, current_value, delta_pct, worse_pct, score, floor_pct, status, reason, 'fixture'
   FROM v_benchmark_metric_verdicts"""  # fmt: skip


def jsonl(rows):
    return "\n".join(json.dumps(r, default=str) for r in rows)


def stats(tables):
    runs = {r["run_id"] for r in tables["benchmark_runs"]}
    return {
        "tables": {k: len(v) for k, v in tables.items()},
        "runs": len(runs),
        "benchmarks": len({b["benchmark_id"] for b in tables["benchmarks"]}),
    }


def ui_calls(api, run, ids):
    """(file name, response) for each endpoint at the states the UI prototype draws: ids is bench(name) / rid(i) plus the dates."""
    matmul = ids["bench"](TS_BENCHES[0][0])
    tp4, ab = ids["tp4"], ids["ab_run"]
    x = lambda comp, arch, i: uid("run", comp, arch, i)  # noqa: E731
    regress_at = night(REGRESSION_NIGHT + 2) + timedelta(hours=4)
    out = {
        "overview": api.overview(run, ids["now"]),
        "overview_regression": api.overview(run, as_of=regress_at),
        "overview_old_invalidated": api.overview(run, as_of=night(22)),
        "suites_x86_torch_spyre": api.suites(run, ids["now"], arch="x86_64", component=TS, limit=100),
        "suites_regression": api.suites(run, as_of=regress_at, arch="x86_64", limit=100),
        "series_matmul": api.series(run, matmul, ids["now"], metric="total_duration_ms", arch="x86_64", days=100),
        "series_serve_tp4": api.series(run, ids["bench"](TP4_CASE), ids["now"], metric="mean_tpot_ms", arch="x86_64", days=100),
        "series_ab": api.series(run, ids["bench"](AB_BENCH), ids["now"], days=100, reference_key=ids["ref_key"]),
        "run_nightly_46": api.run_detail(run, x(TS, "x86_64", 46), ids["now"], limit=50),
        "run_invalidated_20": api.run_detail(run, x(TS, "x86_64", 20), ids["now"], limit=5),
        "run_partial_ppc_10": api.run_detail(run, x(TS, "ppc64le", 10), ids["now"], limit=5),
        "run_tp4": api.run_detail(run, tp4, ids["now"]),
        "environment_nightly_46": api.environment(run, x(TS, "x86_64", 46), ids["now"]),
        "environment_unqualified_z5": api.environment(run, x(TS, "s390x", 5), ids["now"]),
        "environment_ab_pinned": api.environment(run, ab, ids["now"]),
        "steps_tp4": api.steps(run, tp4, ids["now"]),
        "steps_none": api.steps(run, x(TS, "x86_64", 46), ids["now"]),
        "spans_tp4_summary": api.spans(run, tp4, ids["now"]),
        "spans_tp4_timeline_r2_s3": api.spans(run, tp4, ids["now"], view="timeline", rank=2, step=3),
        "spans_tp4_kernels_collective": api.spans(run, tp4, ids["now"], view="kernels", lane="collective"),
        "compare_runs_44_46": api.compare(run, ids["now"], a_run=x(TS, "x86_64", 44), b_run=x(TS, "x86_64", 46)),
        "compare_runs_across_subject_29_30": api.compare(run, ids["now"], a_run=x(TS, "x86_64", 29), b_run=x(TS, "x86_64", 30)),
        "compare_derive_decode": api.compare(run, ids["now"], kind="derive", run_ids=ids["ab_runs"][:10], n_benchmark=ids["bench"](AB_BENCH),
                                             one_benchmark=ids["bench"](AB_BENCH_1), n_tokens=256),
        "compare_effect_pinning": api.compare(run, ids["now"], kind="effect", benchmark_id=ids["bench"](AB_BENCH), metric="latency_ms", arch="x86_64"),
        "compare_reference": api.compare(run, ids["now"], kind="reference", run_id=ab, key=ids["ref_key"], metric="latency_ms"),
        "references": api.references(run, ids["now"], state="all"),
        "reference_detail": api.references(run, ids["now"], key=ids["ref_key"]),
    }  # fmt: skip
    return out


def main(argv=None):
    import argparse
    import os
    from pathlib import Path

    from . import perf_api

    ap = argparse.ArgumentParser(
        description="Fill a throwaway/sandbox ClickHouse database with the synthetic Perf Lab data, and optionally write UI fixtures."
    )
    ap.add_argument(
        "--clickhouse", required=True, help="host:port of the HTTP interface"
    )
    ap.add_argument(
        "--database", required=True, help="must start with sandbox_ or perf_fixture"
    )
    ap.add_argument("--user", default="default")
    ap.add_argument("--password", default=os.environ.get("CLICKHOUSE_PASSWORD", ""))
    ap.add_argument("--secure", action="store_true")
    ap.add_argument(
        "--create-database",
        action="store_true",
        help="CREATE DATABASE IF NOT EXISTS (not for sandbox logins)",
    )
    ap.add_argument(
        "--apply-schema",
        action="store_true",
        help="apply the schema files the data layer reads (CREATE ... IF NOT EXISTS)",
    )
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument(
        "--ui-fixtures",
        type=Path,
        help="write each API response as JSON into this directory",
    )
    a = ap.parse_args(argv)
    if not a.database.startswith(("sandbox_", "perf_fixture")):
        raise SystemExit(
            "refusing: --database must start with sandbox_ or perf_fixture (never a shared database)"
        )
    import clickhouse_connect

    host, port = a.clickhouse.rsplit(":", 1)
    kw = {
        "host": host,
        "port": int(port),
        "username": a.user,
        "password": a.password,
        "secure": a.secure,
    }
    if a.create_database:
        admin = clickhouse_connect.get_client(**kw)
        admin.command(f"CREATE DATABASE IF NOT EXISTS {a.database}")
        admin.close()
    client = clickhouse_connect.get_client(database=a.database, **kw)
    if a.apply_schema:
        for stmt in schema_statements():
            client.command(stmt)
    tables = build(a.seed)

    def sink(table, rows):
        client.raw_insert(table, insert_block=jsonl(rows).encode(), fmt="JSONEachRow")

    load(tables, sink)
    client.command(VERDICT_FILL_SQL)
    print(json.dumps(stats(tables)))
    if a.ui_fixtures:
        run = perf_api.clickhouse_runner(client)
        bench = {
            r["name"]: str(r["benchmark_id"])
            for r in client.query(
                "SELECT name, benchmark_id FROM benchmarks LIMIT 1000"
            ).named_results()
        }
        ab_runs = [
            str(r["r"])
            for r in client.query(
                "SELECT DISTINCT run_id AS r FROM benchmark_runs WHERE props['run.suite.name'] = 'synth-pinning-ab' ORDER BY r LIMIT 50"
            ).named_results()
        ]
        ids = {"now": night(NIGHTS) + timedelta(hours=4), "bench": bench.__getitem__, "tp4": uid("run", "tp4"), "ab_run": ab_runs[0],
               "ab_runs": ab_runs, "ref_key": "perf.synth-lm.decode.ref-system.latency"}  # fmt: skip
        a.ui_fixtures.mkdir(parents=True, exist_ok=True)
        for name, body in ui_calls(perf_api, run, ids).items():
            (a.ui_fixtures / f"{name}.json").write_text(
                json.dumps(body, indent=1, sort_keys=True, default=str)
            )
        index = {"generated_for": "now=" + ids["now"].isoformat(), "benchmarks": bench, "tp4_run": ids["tp4"], "ab_runs": ab_runs,
                 "reference_key": ids["ref_key"], "nightly_runs": {f"{c}/{ar}/{i}": uid("run", c, ar, i) for c, ar in ((TS, "x86_64"), (TS, "ppc64le"), (TS, "s390x"), (SI, "x86_64")) for i in (1, 20, 30, 45, 46, 60)}}  # fmt: skip
        (a.ui_fixtures / "index.json").write_text(
            json.dumps(index, indent=1, sort_keys=True)
        )
        print(
            f"wrote {len(list(a.ui_fixtures.glob('*.json')))} files to {a.ui_fixtures}"
        )


__all__ = [
    "build",
    "load",
    "stats",
    "jsonl",
    "uid",
    "night",
    "SCHEMA_FILES",
    "VERDICT_FILL_SQL",
    "ui_calls",
    "main",
]

if __name__ == "__main__":
    main()
