import numpy as np
import math
import time
import z3  # new dependency!

from dataclasses import dataclass, field, replace
from typing import Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import patches


@dataclass
class CoreDivision:
    output_splits: dict[int, int] = field(default_factory=dict)
    reduction_splits: dict[int, int] = field(default_factory=dict)

    @property
    def cores_used(self) -> int:
        return math.prod(self.output_splits.values()) * math.prod(
            self.reduction_splits.values()
        )

    @property
    def is_clean(self) -> bool:
        return not self.reduction_splits

    @property
    def output_partition(self) -> int:
        return math.prod(self.output_splits.values())

    def signature_key(self):
        return tuple(sorted(self.output_splits.items())) if self.is_clean else None

    @property
    def label(self) -> str:
        out = ",".join(f"s{s}/{f}" for s, f in sorted(self.output_splits.items()))
        red = ",".join(f"~s{s}/{f}" for s, f in sorted(self.reduction_splits.items()))
        return " ".join(p for p in (out, red) if p) or "whole"


@dataclass
class LifetimeBoundBuffer:
    """
    Defines the data fields required for a plan solver.
    """

    name: str
    size: int
    start_time: int
    end_time: int
    address: Optional[int] = None
    in_place: list[str] = field(default_factory=list)
    heuristic: Optional[float] = None
    core_divisions: list[CoreDivision] = field(default_factory=list)
    parent_proj: dict[str, list[CoreDivision]] = field(default_factory=dict)
    chosen_division: Optional[int] = None


@dataclass
class Box:
    """A merge group's memory layout spanning [start_time, end_time)."""

    start_time: int
    end_time: int
    goff: z3.ArithRef
    gsize: z3.ArithRef
    var: z3.BoolRef


def plot_layout(capacity: int, buffers: list[LifetimeBoundBuffer]):
    assert np.all([b.start_time is not None for b in buffers]), (
        "Start time must be defined"
    )
    assert np.all([b.end_time is not None for b in buffers]), "End time must be defined"

    def _fp(b):
        # The reserved per-core footprint, rounded up exactly as the solver
        # does (ceil(size / output_partition)) -- so a buffer that fills to 12
        # is drawn to 12, not to a fractional 11.75.
        if not b.core_divisions or b.chosen_division is None:
            return b.size
        return int(
            np.ceil(b.size / b.core_divisions[b.chosen_division].output_partition)
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(
        y=capacity,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"SPM Ceiling ({capacity})",
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="SPM Floor")

    max_time = max([b.end_time for b in buffers if b.end_time is not None])
    ax.set_xlim(0, max_time + 1)
    ax.set_ylim(-capacity, capacity * 2)
    ax.set_xlabel("Time (Logical Steps)")
    ax.set_ylabel("Memory Address (Bytes)")
    ax.set_title("Calculated Memory Layout")

    rects = {}
    texts = {}

    # Initialize the visual blocks
    for b in buffers:
        rect = patches.Rectangle(
            (b.start_time, 0),
            b.end_time - b.start_time,
            _fp(b),
            linewidth=1.5,
            edgecolor="black",
            facecolor="skyblue",
            alpha=0.8,
        )
        ax.add_patch(rect)
        rects[b.name] = rect

        txt = ax.text(
            0,
            0,
            f"Buffer {b.name}",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
        )
        texts[b.name] = txt

    step_text = ax.text(
        0.02, 0.90, "", transform=ax.transAxes, fontsize=12, weight="bold"
    )
    fig.canvas.draw()
    fig.canvas.flush_events()

    msg = "FINISHED! Showing Final Configuration"
    for b in buffers:
        if b.address is not None:
            rects[b.name].set_visible(True)
            texts[b.name].set_visible(True)
            rects[b.name].set_y(b.address)
            texts[b.name].set_position(
                (
                    b.start_time + (b.end_time - b.start_time) / 2,
                    b.address + _fp(b) / 2,
                )
            )
        else:
            rects[b.name].set_visible(False)
            texts[b.name].set_visible(False)

    # Add a warning box for spilled buffers
    spilled_buffers = [b.name + f" : {_fp(b)}" for b in buffers if b.address is None]
    if spilled_buffers:
        msg += f"\nSpilled to DRAM: {', '.join(spilled_buffers)}"
    else:
        msg += "\n No evictions required"

    step_text.set_text(msg)
    plt.show()


@dataclass
class _InPlaceCandidate:
    src: str
    dst: str


class Z3MemoryPlanSolver:
    """
    Solving is a single-phase satisfiability search -- no optimization. We know
    a lower bound on spills a priori (the forced set: oversized buffers and
    those with no local consumer), so we ask the Solver for a plan that spills
    nothing beyond it and relax the spill budget upward only if that's
    infeasible. Solver.check() returns at the first feasible model, and seeding
    at the forced count means the first check is usually already optimal -- no
    optimality proof, no time-budget burn. Divisions are chosen for feasibility
    (matching + fit) only; this does not optimize peak or core utilization.

    A deterministic bottom-justify post-step (bottom_justify=True) then slides
    every placement unit down to the lowest free address, removing the float
    gaps the search leaves behind without raising the peak.
    """

    def __init__(
        self,
        buffer_size: int,
        alignment: int = 1,
        time_limit_seconds: float = 10.0,
        max_path_length: int = 8,
        bottom_justify: bool = True,
    ) -> None:
        self._buffer_size = buffer_size // alignment
        self._alignment = alignment
        self._time_limit_seconds = time_limit_seconds
        self._max_path_length = max_path_length
        self._bottom_justify = bottom_justify

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        candidates = [
            _InPlaceCandidate(src=parent, dst=b.name)
            for b in buffers
            for parent in b.in_place
        ]

        # Work on copies so we never mutate the caller's buffers -- mutating in
        # place means a second plan_layout call on the same list re-applies the
        # +1 / divide and gets garbage. end_time is bumped to an exclusive bound
        # and size is converted to alignment units for the solver only.
        working = [
            replace(
                b,
                end_time=b.end_time + 1,
                size=int(np.ceil(b.size / self._alignment)),
            )
            for b in buffers
        ]

        offsets, spilled, chosen_div = self._run(working, candidates)
        offsets = {k: v * self._alignment for k, v in offsets.items()}

        return [
            replace(
                b,
                address=None if b.name in spilled else offsets.get(b.name),
                chosen_division=chosen_div.get(b.name, b.chosen_division),
            )
            for b in buffers
        ]

    def _run(
        self,
        tensors: list[LifetimeBoundBuffer],
        candidates: list[_InPlaceCandidate],
    ) -> tuple[dict[str, int], set[str], dict[str, int]]:
        by_name = {t.name: t for t in tensors}
        self._check_candidates(by_name, candidates)

        opt = z3.Solver()
        buffer_vars = self._add_buffer_vars(opt, tensors)
        groups = self._add_merge_groups(opt, tensors, candidates, buffer_vars)
        forced = self._add_core_division(opt, tensors, buffer_vars)
        model = self._search(opt, tensors, buffer_vars, forced)
        return self._extract(model, tensors, buffer_vars, groups)

    @staticmethod
    def _enumerate_dag_paths(
        candidates: list[_InPlaceCandidate], max_path_length: int = 8
    ) -> list[list[str]]:
        """Enumerate all simple paths of length >= 2 in the candidate DAG.

        max_path_length bounds the number of tensors per path to prevent
        combinatorial explosion on large graphs.
        """
        succ: dict[str, list[str]] = {}
        nodes: set[str] = set()
        for c in candidates:
            succ.setdefault(c.src, []).append(c.dst)
            nodes.add(c.src)
            nodes.add(c.dst)

        paths: list[list[str]] = []

        def dfs(path: list[str]) -> None:
            if len(path) >= 2:
                paths.append(list(path))
            if len(path) >= max_path_length:
                return
            for nxt in succ.get(path[-1], []):
                if nxt not in path:
                    path.append(nxt)
                    dfs(path)
                    path.pop()

        for n in nodes:
            dfs([n])

        return paths

    @staticmethod
    def _check_candidates(
        by_name: dict[str, LifetimeBoundBuffer],
        candidates: list[_InPlaceCandidate],
    ) -> None:
        for c in candidates:
            if c.src not in by_name or c.dst not in by_name:
                raise ValueError(f"Candidate {c} references unknown tensor")

    @staticmethod
    def _z3_max(xs: list[z3.ExprRef]) -> z3.ArithRef:
        """max(xs) as a Z3 expression (nested If). Z3 has no built-in Max over
        a list; xs may mix Python ints (non-div buffers) and Z3 vars."""
        m = xs[0]
        for x in xs[1:]:
            m = z3.If(x > m, x, m)
        return m

    @staticmethod
    def _apply_no_overlap_constraint(opt: z3.Solver, boxes: list[Box]) -> None:
        def time_overlap(a: Box, b: Box) -> bool:
            return a.start_time < b.end_time and b.start_time < a.end_time

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                # if buffers do not overlap in time then they
                # are unrelated and have no relative constraints
                if not time_overlap(a, b):
                    continue
                # if a and b are active, then max of a must be less than
                # or equal to min or vice versa.
                opt.add(
                    z3.Implies(
                        z3.And(a.var, b.var),
                        z3.Or(
                            a.goff + a.gsize <= b.goff,
                            b.goff + b.gsize <= a.goff,
                        ),
                    )
                )

    @staticmethod
    def _create_box(
        tensors: list[LifetimeBoundBuffer],
        goff: z3.ArithRef,
        gsize: z3.ArithRef,
        var: z3.BoolRef,
    ) -> Box:
        return Box(
            start_time=min(t.start_time for t in tensors),
            end_time=max(t.end_time for t in tensors),
            goff=goff,
            gsize=gsize,
            var=var,
        )

    def _add_buffer_vars(
        self, opt: z3.Solver, tensors: list[LifetimeBoundBuffer]
    ) -> dict[str, dict[str, z3.ExprRef]]:
        """Allocate each per-tensor variable together with the constraint that
        defines it. Returns buffer_vars: name -> {offset, in_buffer, eff_size,
        and div_var for all buffers}."""
        M = self._buffer_size
        buffer_vars: dict[str, dict[str, z3.ExprRef]] = {}
        for t in tensors:
            n = t.name
            entry: dict[str, z3.ExprRef] = {}
            buffer_vars[n] = entry

            entry["in_buffer"] = z3.Bool(f"in_buf_{n}")  # is buffer in lx?
            offset = z3.Int(f"off_{n}")
            opt.add(offset >= 0, offset <= M)
            entry["offset"] = offset  # where is the buffer in lx?

            if not t.core_divisions:
                # wrap as a Z3 value so every buffer_vars entry is an ExprRef
                entry["eff_size"] = z3.IntVal(t.size)
                continue

            dv = z3.Int(f"div_{n}")
            opt.add(dv >= 0, dv <= len(t.core_divisions) - 1)
            entry["div_var"] = dv  # what core division index are we using?

            per_core = [
                int(np.ceil(t.size / cd.output_partition)) for cd in t.core_divisions
            ]
            sv = z3.Int(f"size_{n}")
            opt.add(
                z3.Or(
                    [z3.And(dv == i, sv == per_core[i]) for i in range(len(per_core))]
                )
            )  # connect the effective size to the chosen index
            entry["eff_size"] = (
                sv  # what is the effective size of the buffer given the current cd?
            )
        return buffer_vars

    def _add_merge_groups(
        self,
        opt: z3.Solver,
        tensors: list[LifetimeBoundBuffer],
        candidates: list[_InPlaceCandidate],
        buffer_vars: dict[str, dict[str, z3.ExprRef]],
    ) -> list[tuple[list[str], z3.BoolRef]]:
        """Placement groups (singletons + in-place paths) with their shared-
        offset/peak constraints, residency, and pairwise no-overlap. Returns the
        groups list (needed by the bottom-justify step)."""
        M = self._buffer_size
        peak = z3.Int("peak")  # what is the peak memory useage?
        opt.add(peak >= 0, peak <= M)  # Constrain it to valid range.

        # determine all the valid merge groups within the compute graph.
        paths = self._enumerate_dag_paths(
            candidates, max_path_length=self._max_path_length
        )

        # Each singleton tensor and each in-place path is a candidate placement
        # group; give each one an activation var keyed by its member names.
        group_members = [[t.name] for t in tensors] + paths

        # track if the current merge group is active.
        groups: list[tuple[list[str], z3.BoolRef]] = [
            (names, z3.Bool(f"group_{'_'.join(names)}")) for names in group_members
        ]

        by_name = {t.name: t for t in tensors}
        groups_containing = defaultdict(list)
        boxes = []
        for g in groups:
            names, var = g
            ts = [by_name[n] for n in names]

            # track the offset of the merge group.
            goff = z3.Int(f"goff_{'_'.join(names)}")

            # constrain the merge group offset.
            opt.add(goff >= 0, goff <= M)

            # Merge-group size = the largest member footprint. gsize is fully
            # determined by members, so it's a max *expression* (nested If),
            # not a decision variable -- no extra var or constraints needed.
            members = [buffer_vars[n]["eff_size"] for n in names]
            gsize = self._z3_max(members)

            for n in names:
                groups_containing[n].append(g)
                # if the current merge group 'var' is active
                # the offset of buffers contained in that merge
                # group must match the merge group offset.
                opt.add(z3.Implies(var, buffer_vars[n]["offset"] == goff))

            # if the current merge group 'var' is active
            # the offset + group size must the less than
            # the peak usage.
            opt.add(z3.Implies(var, goff + gsize <= peak))

            # create the memory layout for each merge group spanning
            # from min t to max t within group with captured variables
            boxes.append(self._create_box(ts, goff, gsize, var))

        # If a buffer is active it must be only contained by one merge group
        # the merge group can be a singleton group or a larger group
        # but a buffer must strictly belong to one active group.
        for n, gs in groups_containing.items():
            opt.add(z3.Sum([var for _, var in gs]) == buffer_vars[n]["in_buffer"])

        self._apply_no_overlap_constraint(opt, boxes)
        return groups

    def _normalize(self, cd):
        key = cd.signature_key()
        if key is None:
            self.dirty[0] -= 1
            return self.dirty[0]
        return self.sig_ids.setdefault(key, len(self.sig_ids))

    def _get_children(self, tensors):
        children_of = defaultdict(list)
        for t in tensors:
            for parent, _ in t.parent_proj.items():
                children_of[parent].append(
                    (t.name, [self._normalize(cd) for cd in t.core_divisions])
                )
        return children_of

    def _trim_oversized_tensors(self, opt, tensors, children_of, buffer_vars):
        forced = set()
        for t in tensors:
            min_size = (
                min(
                    int(np.ceil(t.size / cd.output_partition))
                    for cd in t.core_divisions
                )
                if t.core_divisions
                else t.size
            )
            no_consumer = bool(t.core_divisions) and not children_of.get(t.name)
            if min_size > self._buffer_size or no_consumer:
                forced.add(t.name)
                opt.add(z3.Not(buffer_vars[t.name]["in_buffer"]))
        return forced

    def _implicate_core_division(self, opt, tensors, children_of, buffer_vars):
        for t in tensors:
            kids = children_of.get(t.name, [])
            if not kids:
                continue
            own_ids = [self._normalize(cd) for cd in t.core_divisions]
            child_matches = []
            for child, child_ids in kids:
                pairs = [
                    z3.And(
                        buffer_vars[t.name]["div_var"] == i,
                        buffer_vars[child]["div_var"] == j,
                    )
                    for i in range(len(own_ids))
                    for j in range(len(child_ids))
                    if own_ids[i] == child_ids[j]
                ]
                child_matches.append(z3.Or(pairs) if pairs else z3.BoolVal(False))
            opt.add(z3.Implies(buffer_vars[t.name]["in_buffer"], z3.Or(child_matches)))

    def _add_core_division(
        self,
        opt: z3.Solver,
        tensors: list[LifetimeBoundBuffer],
        buffer_vars: dict[str, dict[str, z3.ExprRef]],
    ) -> set[str]:
        """children_of relation, forced spills, and core-division matching.
        Returns the forced-spill set (the search's lower-bound seed)."""
        self.sig_ids = {}
        self.dirty = [0]

        children_of = self._get_children(tensors)
        forced = self._trim_oversized_tensors(opt, tensors, children_of, buffer_vars)
        self._implicate_core_division(opt, tensors, children_of, buffer_vars)
        return forced

    def _search(
        self,
        opt: z3.Solver,
        tensors: list[LifetimeBoundBuffer],
        buffer_vars: dict[str, dict[str, z3.ExprRef]],
        forced: set[str],
    ) -> z3.ModelRef:
        """Spill-budget satisfiability search: ask for a plan that spills
        nothing beyond `forced`, relaxing the budget upward only if infeasible.
        Returns the first feasible model -- seeded at len(forced), so the first
        check is usually already optimal (no optimality proof, no budget burn)."""
        if self._time_limit_seconds:
            opt.set("timeout", int(self._time_limit_seconds * 1000))

        spill_count = z3.Sum(
            [z3.If(buffer_vars[t.name]["in_buffer"], 0, 1) for t in tensors]
        )

        n_tensors = len(tensors)
        lo = len(forced)
        iterations = []  # (budget, status, seconds)
        model = None
        t_start = time.perf_counter()
        for budget in range(lo, n_tensors + 1):
            opt.push()
            opt.add(spill_count <= budget)
            t0 = time.perf_counter()
            status = opt.check()
            iterations.append((budget, status, time.perf_counter() - t0))
            if status == z3.sat:
                model = opt.model()
                opt.pop()
                break
            opt.pop()
        total = time.perf_counter() - t_start

        ######################################
        # Debug: search timing / iterations
        ######################################
        won_budget = iterations[-1][0] if model is not None else None
        lines = [
            "[Z3 spill-budget search]",
            f"  tensors            : {n_tensors}",
            f"  forced spills      : {lo} ({', '.join(sorted(forced)) or 'none'})",
            f"  budget seed / max  : {lo} / {n_tensors}",
            f"  time limit / check : {self._time_limit_seconds}s",
            f"  iterations (relax) : {len(iterations)} ({max(0, len(iterations) - 1)})",
        ]
        for i, (budget, status, secs) in enumerate(iterations):
            lines.append(
                f"    [{i}] spill<={budget:<4d} {str(status):>7s} {secs * 1e3:9.2f} ms"
            )
        outcome = (
            f"SAT @ spill<={won_budget}" if model is not None else "NO FEASIBLE PLAN"
        )
        lines.append(f"  result             : {outcome}")
        lines.append(f"  total search time  : {total * 1e3:.2f} ms")
        print("\n".join(lines))

        if model is None:
            raise RuntimeError("Z3 memory planner found no feasible plan")
        return model

    def _extract(
        self,
        model: z3.ModelRef,
        tensors: list[LifetimeBoundBuffer],
        buffer_vars: dict[str, dict[str, z3.ExprRef]],
        groups: list[tuple[list[str], z3.BoolRef]],
    ) -> tuple[dict[str, int], set[str], dict[str, int]]:
        """Read the model into (offsets, spilled, chosen_div). When
        bottom_justify is set, slide each placement unit down to the lowest free
        address (preserving in-place merges, never raising the peak)."""

        def bval(b):
            return z3.is_true(model.eval(b, model_completion=True))

        def ival(x):
            return model.eval(x, model_completion=True).as_long()

        by_name = {t.name: t for t in tensors}
        spilled = {
            t.name for t in tensors if not bval(buffer_vars[t.name]["in_buffer"])
        }
        chosen_div = {
            n: ival(v["div_var"]) for n, v in buffer_vars.items() if "div_var" in v
        }

        def footprint(t):
            if t.core_divisions:
                return int(
                    np.ceil(
                        t.size / t.core_divisions[chosen_div[t.name]].output_partition
                    )
                )
            return t.size

        if not self._bottom_justify:
            return (
                {
                    t.name: ival(buffer_vars[t.name]["offset"])
                    for t in tensors
                    if bval(buffer_vars[t.name]["in_buffer"])
                },
                spilled,
                chosen_div,
            )

        # An active group is a placement unit (its members share one base, so
        # in-place merges are preserved as a single block).
        units = [
            {
                "members": names,
                "fp": max(footprint(by_name[n]) for n in names),
                "s": min(by_name[n].start_time for n in names),
                "e": max(by_name[n].end_time for n in names),
                "base0": ival(buffer_vars[names[0]]["offset"]),
            }
            for names, var in groups
            if bval(var)
        ]
        return self._justify(units), spilled, chosen_div

    @staticmethod
    def _justify(units: list[dict[str, Any]]) -> dict[str, int]:
        """Slide each placement unit down to the lowest free address. A unit is
        {members, fp, s (start), e (end), base0 (current base)}. Processing in
        current-base order and giving each the lowest non-conflicting slot
        preserves the relative stacking, so the peak never increases -- it only
        squeezes out the float gaps the feasibility search leaves. Returns a
        name -> address map."""
        placed = []
        offsets = {}
        for u in sorted(units, key=lambda u: (u["base0"], u["s"])):
            # lowest base whose [base, base+fp) clears every already-placed unit
            # that overlaps this one in time
            obstacles = sorted(
                (p["base"], p["base"] + p["fp"])
                for p in placed
                if u["s"] < p["e"] and p["s"] < u["e"]
            )
            base = 0
            for lo, hi in obstacles:
                if base + u["fp"] <= lo:
                    break  # fits in the gap below this obstacle
                if base < hi:
                    base = hi  # otherwise bump above it
            u["base"] = base
            placed.append(u)
            for n in u["members"]:
                offsets[n] = base
        return offsets


# Active planner backend. Swap to OrToolsMemoryPlanSolver to use CP-SAT.
SOLVER = Z3MemoryPlanSolver


def solve(
    buffers: list[LifetimeBoundBuffer], buffer_size: int
) -> dict[str, LifetimeBoundBuffer]:
    solver = SOLVER(buffer_size=buffer_size)
    return {b.name: b for b in solver.plan_layout(buffers)}


def placement_example():
    def divs():
        return [
            CoreDivision(output_splits={256: 4}),
            CoreDivision(output_splits={1: 4}),
        ]

    buffers = [
        LifetimeBoundBuffer(
            name="A",
            size=60,
            start_time=0,
            end_time=2,
            heuristic=None,
            address=0,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="B",
            size=30,
            start_time=1,
            end_time=4,
            heuristic=None,
            address=60,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="C",
            size=30,
            start_time=2,
            end_time=13,
            heuristic=None,
            address=90,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="D",
            size=30,
            start_time=3,
            end_time=4,
            heuristic=None,
            address=30,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="E",
            size=30,
            start_time=4,
            end_time=5,
            heuristic=None,
            address=0,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="F",
            size=60,
            start_time=5,
            end_time=6,
            heuristic=None,
            address=30,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="G",
            size=30,
            start_time=6,
            end_time=15,
            heuristic=None,
            address=None,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="H",
            size=30,
            start_time=7,
            end_time=8,
            heuristic=None,
            address=0,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="I",
            size=30,
            start_time=8,
            end_time=9,
            heuristic=None,
            address=30,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="J",
            size=15,
            start_time=9,
            end_time=16,
            heuristic=None,
            address=75,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="K",
            size=15,
            start_time=10,
            end_time=12,
            heuristic=None,
            address=0,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="L",
            size=15,
            start_time=11,
            end_time=12,
            heuristic=None,
            address=15,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="M",
            size=15,
            start_time=12,
            end_time=13,
            heuristic=None,
            address=30,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="N",
            size=45,
            start_time=13,
            end_time=15,
            heuristic=None,
            address=45,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="O",
            size=45,
            start_time=14,
            end_time=15,
            heuristic=None,
            address=0,
            in_place=[],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="P",
            size=45,
            start_time=15,
            end_time=16,
            heuristic=None,
            address=90,
            in_place=["N"],
            core_divisions=divs(),
        ),
        LifetimeBoundBuffer(
            name="Q",
            size=75,
            start_time=16,
            end_time=17,
            heuristic=None,
            address=0,
            in_place=[],
            core_divisions=divs(),
        ),
    ]

    by_name = {b.name: b for b in buffers}
    order = sorted(by_name)
    for parent, child in zip(order, order[1:]):
        by_name[child].parent_proj[parent] = divs()

    plot_layout(40, [v for v in solve(buffers, 40).values()])


if __name__ == "__main__":
    placement_example()
