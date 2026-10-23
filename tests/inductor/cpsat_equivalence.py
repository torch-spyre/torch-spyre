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

"""Equivalence harness for refactors of the CP-SAT layout solver.

Records, for every ``SpyreOptimizationGraph.solve``, the model proto and outcome of
each ladder stage plus the decisions read off the solved graph, then diffs
two recordings. A refactor that claims "identical CP model" must produce an
empty diff.

Solves run deterministically: one worker, ``random_seed=0`` and no wall-clock
limit, only a ``max_deterministic_time`` budget (``CPSAT_EQUIV_DTIME``,
default 10), so a FEASIBLE incumbent is reproducible and compared exactly.

Three ways to record:

* inside pytest, with this module loaded as a plugin (``CPSAT_EQUIV_GROUPS=file``
  also logs, per coarse-tiling pass, the solve's joined runs against
  ``derive_tiling_groups``)::

      CPSAT_EQUIV_DIR=out PYTHONPATH=tests/inductor \\
          python3 -m pytest -p cpsat_equivalence tests/inductor/test_x.py

* a device-free replay of the synthetic graphs, the JSON captures and any
  solver-input pickles (``probes/cpsat-graph/deenum`` captures)::

      python3 tests/inductor/cpsat_equivalence.py record out [caps_dir ...]

* then compare::

      python3 tests/inductor/cpsat_equivalence.py diff base_out change_out
"""

from __future__ import annotations

import copy
import dataclasses
import difflib
import gzip
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import regex as re

_DTIME = float(os.environ.get("CPSAT_EQUIV_DTIME", "10"))


class _Recorder:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.case = "standalone"
        self.index = 0
        self.stages: Optional[list[dict]] = None

    def begin_case(self, case: str) -> None:
        self.case = case
        self.index = 0

    def _key(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.case)[-180:]
        return f"{safe}#{self.index}"

    def stage(self, solver, model, status: int, objective: bool) -> None:
        from ortools.sat.python import cp_model

        if self.stages is None:
            return
        text = str(model.proto)
        digest = hashlib.sha256(text.encode()).hexdigest()
        n = len(self.stages)
        path = self.out / f"{self._key()}.stage{n}.proto.gz"
        with gzip.open(path, "wt") as f:
            f.write(text)
        provenance = getattr(model, "_spyre_provenance", None)
        if provenance is not None:
            with gzip.open(
                self.out / f"{self._key()}.stage{n}.prov.json.gz", "wt"
            ) as f:
                json.dump([p.as_dict() for p in provenance], f)
        feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        self.stages.append(
            {
                "proto_sha": digest,
                "status": solver.StatusName(status),
                "objective": solver.ObjectiveValue() if feasible else None,
                "objective_used": objective,
            }
        )

    def finish_graph(self, graph, strategy, error: Optional[str]) -> None:
        """``finish`` for a solve of a graph built from the IR: the same
        decisions, read off the solved nodes and candidate copies."""
        unit = graph.alignment
        decisions = {}
        items = [*graph.get_nodes(), *strategy.transforms]
        for item in items:
            chosen = {}
            for edge in getattr(item, "inputs", ()):
                t = edge.transform
                if t is not None and t.offset is not None:
                    c = edge.relayout
                    chosen[edge.src.name] = [
                        c.group,
                        c.source_division,
                        c.consumer_division,
                        t.offset * unit,
                    ]
            decisions[item.name] = {
                "address": None if item.offset is None else item.offset * unit,
                "spill_reason": item.spill_reason,
                "chosen_division": (
                    0 if item in strategy.transforms else strategy.solved.get(item)
                ),
                "chosen_relayouts": dict(sorted(chosen.items())),
            }
        self._write(decisions, error)

    def _write(self, decisions: dict, error: Optional[str]) -> None:
        record = {
            "case": self.case,
            "stages": self.stages,
            "decisions": decisions,
            "error": error,
        }
        with open(self.out / f"{self._key()}.json", "w") as f:
            json.dump(record, f, indent=1, sort_keys=True, default=str)
        self.stages = None
        self.index += 1


_RECORDER: Optional[_Recorder] = None


def _record_stage(solver, model, *, objective: bool = False) -> int:
    """``SolveSettings.record``: solve one ladder stage deterministically and
    record it."""
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 1e9
    solver.parameters.max_deterministic_time = _DTIME
    status = solver.Solve(model)
    if _RECORDER is not None:
        _RECORDER.stage(solver, model, status, objective)
    return status


def install(out: str | Path) -> _Recorder:
    """Patch ``SpyreOptimizationGraph.solve`` so every solve is recorded under
    ``out``, each ladder stage through ``SolveSettings.record``."""
    global _RECORDER
    from torch_spyre._inductor.scratchpad.buffer_graph import SpyreOptimizationGraph

    _RECORDER = _Recorder(Path(out))
    if getattr(SpyreOptimizationGraph, "_cpsat_equiv_installed", False):
        return _RECORDER
    orig_solve = SpyreOptimizationGraph.solve

    def solve(self, model=None, cost_expr=None, strategy=None):
        from torch_spyre._inductor.scratchpad.lazy_strategy import LazyStrategy

        if strategy is None:
            strategy = LazyStrategy()
        self.settings = dataclasses.replace(self.settings, record=_record_stage)
        rec = _RECORDER
        if rec is None or rec.stages is not None:
            return orig_solve(self, model, cost_expr, strategy)
        rec.stages = []
        try:
            plan = orig_solve(self, model, cost_expr, strategy)
        except Exception as exc:
            rec.finish_graph(self, strategy, f"{type(exc).__name__}: {exc}")
            raise
        rec.finish_graph(self, strategy, None)
        return plan

    SpyreOptimizationGraph.solve = solve
    SpyreOptimizationGraph._cpsat_equiv_installed = True
    return _RECORDER


# ---- loop groups: the solve's joined runs vs derive_tiling_groups ----------

_CASE = "standalone"


def _joined_runs(graph) -> list[list[str]]:
    """The tiled runs the solve's decisions join: consecutive tileable ops on
    consecutive ticks running the same non-empty nest."""
    runs: list[list[str]] = []
    prev = None
    node = graph.first
    while node is not None:
        tiling = node.tiling
        nest = tiling.level_counts if tiling is not None else ()
        if nest:
            if (
                prev is not None
                and prev.produced_at + 1 == node.produced_at
                and prev.tiling.level_counts == nest
            ):
                runs[-1].append(node.name)
            else:
                runs.append([node.name])
            prev = node
        else:
            prev = None
        node = node.next.after if node.next else None
    return runs


def install_group_check(out: str | Path) -> None:
    """Log, per coarse-tiling pass, whether the last solve's joined runs and
    ``derive_tiling_groups`` agree (one JSON line per pass to ``out``)."""
    from torch_spyre._inductor.scratchpad import buffer_graph, coarse_tiling

    last: dict[str, Any] = {}
    orig_solve = buffer_graph.SpyreOptimizationGraph.solve
    orig_apply = coarse_tiling.CoarseTilingPass.apply_pass

    def solve(self, *args, **kwargs):
        plan = orig_solve(self, *args, **kwargs)
        last["runs"] = _joined_runs(self)
        return plan

    def apply_pass(self, graph):
        runs = last.pop("runs", None)
        if runs is not None:
            groups = [
                [op.get_name() for op in ops]
                for ops, _ in coarse_tiling.derive_tiling_groups(graph, self._choices)
            ]
            run_of = {name: i for i, run in enumerate(runs) for name in run}
            refines = all(
                len({run_of.get(name) for name in group}) == 1
                and None not in {run_of.get(name) for name in group}
                for group in groups
            )
            with open(out, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "case": _CASE,
                            "equal": sorted(map(sorted, runs))
                            == sorted(map(sorted, groups)),
                            "groups_refine_runs": refines,
                            "runs": runs,
                            "groups": groups,
                        }
                    )
                    + "\n"
                )
        return orig_apply(self, graph)

    buffer_graph.SpyreOptimizationGraph.solve = solve
    coarse_tiling.CoarseTilingPass.apply_pass = apply_pass


# ---- pytest plugin ---------------------------------------------------------


def pytest_configure(config) -> None:
    out = os.environ.get("CPSAT_EQUIV_DIR")
    if out:
        install(out)
    groups = os.environ.get("CPSAT_EQUIV_GROUPS")
    if groups:
        install_group_check(groups)


def pytest_runtest_setup(item) -> None:
    global _CASE
    _CASE = item.nodeid
    if _RECORDER is not None:
        _RECORDER.begin_case(item.nodeid)


# ---- device-free replay ----------------------------------------------------


def _solve(case: str, buffers, limit: int, alignment: int, cost_expr) -> None:
    """Solve the graph of ``buffers``; a buffer with no candidate divisions is
    committed, so its node is placement-only."""
    from torch_spyre._inductor.scratchpad.buffer_graph import SolveSettings
    from buffer_spec import build_graph
    from torch_spyre._inductor.scratchpad.lazy_strategy import LazyStrategy

    assert _RECORDER is not None
    _RECORDER.begin_case(case)
    buffers = copy.deepcopy(buffers)
    graph = build_graph(
        buffers,
        alignment,
        limit // alignment,
        SolveSettings(time_limit_seconds=0, record=_record_stage),
        {b.name: b.residency_reason for b in buffers if b.residency_reason},
    )
    strategy = LazyStrategy()
    try:
        graph.solve(cost_expr=cost_expr, strategy=strategy)
    except Exception as exc:  # recorded by the patch; keep replaying
        print(f"  {case}: {type(exc).__name__}: {exc}", flush=True)


def _replay(out: str, caps: list[str]) -> None:
    import torch  # noqa: F401  (before torch_spyre)
    from torch_spyre._inductor import config

    sys.path.insert(0, str(Path(__file__).parent))
    from cooptimization_capture_loader import LARGE_CAPTURE_PATH, load_captures
    from synthetic_cooptimization_graphs import synthetic_graphs

    install(out)
    limit = 2 * 1024 * 1024
    groups: dict[str, Any] = {"synthetic": synthetic_graphs()}
    groups["captures"] = load_captures()
    groups["captures_large"] = load_captures(LARGE_CAPTURE_PATH)
    for group, cases in groups.items():
        for name, graphs in cases.items():
            for i, g in enumerate(graphs):
                print(f"{group}/{name}[{i}]", flush=True)
                _solve(f"{group}/{name}[{i}]", g.buffers, limit, 128, None)
                # Placement-only: the same buffers with nothing to choose.
                flat = copy.deepcopy(g.buffers)
                for b in flat:
                    b.core_divisions = []
                _solve(f"{group}/{name}[{i}]/placement", flat, limit, 128, None)
    for caps_dir in caps:
        for path in sorted(Path(caps_dir).glob("*.pkl")):
            case = f"{Path(caps_dir).name}/{path.stem}"
            print(case, flush=True)
            try:
                with open(path, "rb") as f:
                    cap = pickle.load(f)
            except (AttributeError, TypeError) as exc:
                # Pickled with a class this tree no longer has.
                print(f"  {case}: not loadable here: {exc}", flush=True)
                continue
            with config.patch(
                sencores=cap["sencores"], auto_coarse_tiling=cap["auto_coarse_tiling"]
            ):
                _solve(
                    case,
                    cap["buffers"],
                    cap["limit"],
                    cap["alignment"],
                    cap["cost_expr"],
                )


# ---- diff ------------------------------------------------------------------


def _proto_diff(base: Path, change: Path, key: str, n: int, lines: int = 40) -> str:
    def read(root):
        p = root / f"{key}.stage{n}.proto.gz"
        if not p.exists():
            return []
        with gzip.open(p, "rt") as f:
            return f.read().splitlines()

    out = list(
        difflib.unified_diff(
            read(base), read(change), "base", "change", n=2, lineterm=""
        )
    )
    return "\n".join(out[:lines])


def _error(record: dict) -> str:
    """The recorded error without its wall-clock time."""
    text = re.sub(r"after \d+\.\d+s", "after Ns", record["error"] or "")
    return re.sub(r"0x[0-9a-f]+", "0x", text)[:200]


def diff(base_dir: str, change_dir: str, verbose: bool = True) -> int:
    base, change = Path(base_dir), Path(change_dir)
    keys = {p.name[: -len(".json")] for p in base.glob("*.json")}
    keys |= {p.name[: -len(".json")] for p in change.glob("*.json")}
    bad = 0
    for key in sorted(keys):
        b, c = base / f"{key}.json", change / f"{key}.json"
        if not b.exists() or not c.exists():
            print(f"MISSING {'base' if not b.exists() else 'change'}: {key}")
            bad += 1
            continue
        rb, rc = json.loads(b.read_text()), json.loads(c.read_text())
        problems = []
        sb, sc = rb["stages"] or [], rc["stages"] or []
        if len(sb) != len(sc):
            problems.append(f"stage count {len(sb)} != {len(sc)}")
        for n, (x, y) in enumerate(zip(sb, sc)):
            if x["proto_sha"] != y["proto_sha"]:
                problems.append(f"stage {n} proto differs")
                if verbose and len(problems) == 1:
                    problems.append(_proto_diff(base, change, key, n))
            for field in ("status", "objective", "objective_used"):
                if x[field] != y[field]:
                    problems.append(f"stage {n} {field}: {x[field]} != {y[field]}")
        if rb["decisions"] != rc["decisions"]:
            names = [
                k
                for k in sorted(set(rb["decisions"]) | set(rc["decisions"]))
                if rb["decisions"].get(k) != rc["decisions"].get(k)
            ]
            problems.append(f"decisions differ on {names[:8]}")
        if _error(rb) != _error(rc):
            problems.append(f"error: {rb['error']!r} != {rc['error']!r}")
        if problems:
            bad += 1
            print(f"DIFF {key}")
            for p in problems:
                print("   ", p.replace("\n", "\n    "))
    print(f"{len(keys)} solves, {bad} differ")
    return bad


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "record":
        _replay(rest[0], rest[1:])
    elif cmd == "diff":
        sys.exit(1 if diff(rest[0], rest[1]) else 0)
    else:
        raise SystemExit(f"unknown command {cmd}")
