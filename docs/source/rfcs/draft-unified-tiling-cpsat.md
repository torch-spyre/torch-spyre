# RFC (draft) — Coarse Tiling Optimization

| Field | Value |
|---|---|
| Status | Draft — pending issue number |
| Area | Compiler |
| Roadmap | Compiler Optimization Roadmap — collateral document 1, Phase 1 |
| Target | `torch_spyre/_inductor` — `wsr`, `scratchpad`, and `padding.py` |
| Depends on | Compiler Optimization Roadmap (parent, `draft-compiler-optimization.md`); Optimization and Hinting Strategy (collateral document 0); RFC 1358 (Coarse Tiling); RFC 0047 (Tensors with Device-Specific Layouts) |

> **Draft note.** RFC sources live in
> [`torch-spyre/rfcs`](https://github.com/torch-spyre/rfcs) as
> `NNNN-PascalCase/NNNN-PascalCaseRFC.md`, where `NNNN` is the GitHub issue
> number. This file is the working draft; on filing the issue it moves to that
> repository and gains a row plus a summary paragraph in
> `docs/source/rfcs/index.md`.

## Summary

This is **collateral document 1** of the
[Compiler Optimization Roadmap](draft-compiler-optimization.md), covering the
roadmap's **Phase 1 — coarse tiling for sizing and temporal op splitting,
jointly with core division**. It instantiates the roadmap's shared model
(M1–M9) and hint spine (H1–H5) for the tiling axis; per the roadmap's rule that
a phase document may not restate the model, it cites those requirements by
number rather than re-deriving them, and specifies only what the tiling axis
adds.

Concretely, it adds **tiling** — a *temporal split*, sequential loop iterations
on one core — as a third axis of the single CP-SAT model (M1) that already
chooses **core division** — a *spatial split*, parallel across cores — and LX
residency together, so all three partitioning decisions are made against one
injected objective (M2). Tiling and division collapse into a per-operation
**config** enumerated and tabled with the rest of the model (M4), feasible on
the *combination* rather than per subsystem (M5). The solver enumerates every
valid output-range tiling option per operation and lets tiling *groups* emerge
from minimizing that objective — a group is a maximal run of operations with no
loop-nest break — rather than being pre-computed from hint scopes. A tiling or
division hint is a **pin** in the roadmap's pinned/optimized sense (M3): it
collapses its axis at enumeration time (H2) and the solver optimizes the rest of
the graph around it.
Reduction-axis tiling enters the optimized space **single-level only** — one
reduction dim split once, pinned as a singleton group (R4.6); the nested
output+reduction shapes remain a known wrong-numerics bug this RFC does not fix
and the enumerator never emits (R1.8, R9). Padding, which the roadmap folds into
this phase because it is the first
that needs it for sizing (consequence 3), enters as a derived scalar on
candidate rows and — only if measurement justifies it — as extra rows, adding no
new decision axis (R10).

## Relationship to the roadmap

The roadmap states the shared model once; this document satisfies the parts the
tiling axis touches, cites them by number, and defers the rest to the phases and
tracks that own them.

**Satisfied here (Phase 1).**

| Roadmap requirement | How this document satisfies it |
|---|---|
| M1 — one CP-SAT solve, one objective | Tiling joins the existing joint division + residency model as a third axis (§1–§3); no separate tiling stage (*Alternatives*). |
| M2 — injected objective | This phase **delivers** the objective namespace and sympy→CP-SAT lowering later phases consume (§4, R3); today's two-phase lexicographic solve collapses into a single minimized cost. |
| M3 — pinned / optimized modes | A tiling or division hint pins its axis (R5); the (untiled, work-division-seed) config is the baseline candidate M3's invariant requires (R2.4). |
| M4 — enumerate-and-table | The (tiling, division) `PartitionConfig` cross product, with derived scalars bound by `AddElement` (§1, R2). |
| M5 — feasibility on the combination | Span and divisibility are checked on the config *pair*, discharging the `core_split_estimate = 1` guess (R2.3) — roadmap consequence 1. |
| M6 — predicates reused, never reimplemented | Every legality predicate is reused from `wsr`/`pass_utils`/`work_division` (R1.4, R4.7). |
| M7 — pure prediction, then apply; offline fidelity | A pure predictor maps configs to a buffer set with no IR mutation (R7); predicted values are recorded (H5) for the roadmap's *offline* fidelity check, not an in-compiler verify pass. |
| M8 — addresses from the final solve | The placement re-solve over the real post-tiling buffers is authoritative for addresses (§5, R7.3). |
| M9 — failure never discards a pin | Only an `INFEASIBLE` model — or a timeout with no feasible incumbent — raises `SolveError`; the pipeline then reverts to the retained span-overflow tiler plus greedy placement, over the already-pinned candidate sets (R8.3). A timeout that *has* an incumbent applies it. |
| H1–H5 — hint spine | Tiling and pad keys register under H1, lower to pins under H2, are validated by H3, shrink the model under H4, and explain themselves under H5 (R5, R10); collateral document 0 owns the registry itself. |
| C2 — determinism, tractability, fallback | Determinism, warm-start, and the fallback path (R8). |

**Owed to later phases** (roadmap Phase 1, *"Owed to later phases"*): the M4
config encoding, the M2 objective namespace, the M7 predictor, and **tiling-aware
per-core views** evaluated against a *predicted* post-tiling frame (R2.6) — the
last of which Phase 3 cannot start without.

**Deferred to other phases / tracks, not solved here.**

| Concern | Owner |
|---|---|
| Time-varying addresses, defragmentation | Phase 2 / collateral document 2 |
| Relayout *pricing*, restickify placement and sharing, padding as layout legalization | Phase 3 / collateral document 3 (R6, R10) |
| Op re-ordering as a decision | Phase 4 / collateral document 4 (R4.4) |
| Enumeration scaling (signature dedup, per-dimension channeling, lazy enumeration) | Phase 5 / collateral document 5 (§6) |
| Cost-model calibration in microseconds | Track C1 (R9) |

## Motivation

The roadmap's *Motivation* enumerates seven consequences of decisions taken at
different points of `CustomPreSchedulingPasses` by cost models that cannot see
each other. This document closes the two that are the tiling axis's —
**consequence 1** (the span budget is spent twice) and **consequence 2** (tiling
buys LX residency only by accident) — and the sizing half of **consequence 3**
(padding cannot influence the layouts it must satisfy; the legalization half is
Phase 3). The rest of this section is the axis-specific evidence for those,
stated in code terms rather than re-deriving the roadmap.

The three code sites this axis touches, each blind to the others, are the
tiling-axis slice of the roadmap's decision table:

| Decision | Where | Cost model | Blind to |
|---|---|---|---|
| Coarse tiling (loop nest / WSR) | `wsr/coarse_tile.py`, applied at `passes.py:430` (hints, pre-stickify) and `passes.py:448` (span overflow, post-stickify) | `_combo_cost` = `(total tiles, #tiled dims, max split, combo)`, first feasible wins | core division, LX |
| Core division | `work_division.py`, two pass-list entries at `passes.py:451-452` expanding to three passes (`_distribute_work` is `@_runs(cost_model_matmul_division, work_distribution)`, `:382`) | `_matmul_split_cost` (µs) for matmuls; priority heuristic otherwise | LX, tiling |
| LX residency + placement | `scratchpad/ilp_solver_ortools.py`, `passes.py:455` | CP-SAT, two-phase lexicographic: `spill_cost()` then `sum(cores)` | tiling |

**Over-tiling (consequence 1).** `plan_span_overflow_tile`
(`wsr/span_overflow_hint_analysis.py:1479`) is handed `core_split_estimate = 1`,
hardcoded at both `ChunkingInfo` construction sites (`:666`, `:827`), and carries
the TODO at `:1504-1506`: *"make a common planner for Work Division and Working
Set Reduction together, so this pass can get a proper `core_split_estimate`
instead of the hardcoded 1."* It chooses a tiling as if the op ran on one core; work
division then splits the same dimensions again. The `MAX_SPAN_BYTES` constraint
is satisfied twice over and the emitted tile counts are larger than necessary,
costing loop overhead and, where tiling forces boundary copies, HBM traffic.

**Tiling never buys LX residency (consequence 2).** Shrinking a chain's working
set so it fits the 2 MB LX is the purpose of working-set reduction, yet the
tiling planner cannot see LX occupancy and the LX solver cannot choose a tiling.
`docs/source/compiler/scratchpad_planning.md:578` lists "**No coarse-tiling
integration** when that pass also drives split decisions" among the remaining
gaps under "Co-optimization is still limited" (`:555`). It is logged there against
`StrategyBCoOptimizingAllocator` (`:557`), but the limitation is shared verbatim
by the CP-SAT `CoOptimizingAllocator` (`:534`) that this RFC extends.

The channel through which tiling would buy residency is specific, and naming it
sharpens what the joint model is for. Residency accrues to a cut-free run's
**interior** — the per-tile scratch that never materializes and the read-side
tile copies, both tile-sized and fixed-address, hence LX-eligible. It never
accrues to anything crossing a group boundary, which is HBM-only by construction
(*Background*). So the decision the objective must make is not "tile more" but
"where to place cuts such that the buffers left in the interior are the ones
worth pinning" — a question neither of today's two planners can even pose.

The joint pattern is already proven for two of the three axes.
`CoOptimizingAllocator` (`scratchpad/allocator.py:1476`) hands enumerated
`CoreDivision` candidates plus producer/consumer slicing-match tables to
`CpSatLayoutSolver`, which picks division and placement in one model. This RFC
adds tiling as a third axis of that same model.

Making the objective caller-supplied is not a motivation this document argues for
on its own — it is roadmap requirement **M2**, and Phase 1 is the phase that
*delivers* the injectable objective namespace later phases consume (§4). One
enabling fact is worth recording because it is what makes the delivery cheap:
recent work on the `tighten-spill-cost` branch encapsulated the per-buffer spill
term into `_LifetimeBufferWithCpVars.spill_cost()`, which now returns a
`cp_model.LinearExpr` already gated on `1 - in_buffer`. That makes today's
hardcoded objective a single overridable hook — the natural point at which to
hang M2's injection.

## Background: what exists today

### Coarse tiling

`coarse_tile(graph, groups, group_idx_offset=0)` (`wsr/coarse_tile.py:1072`)
annotates `groups` as a bare `list[tuple]`; the documented contract is
`(ops, levels)` where `levels` is `[(hint_id, count), ...]` outermost-first. It
runs in two phases:

- `plan_coarse_tile_groups(operations, groups)` (`:186`) — **zero mutation**, and
  it raises `Unsupported` before any transformation rather than half-applying.
  Produces `{id(op): CoarseTileInfo}` using `_planned_tile_extents_per_level`
  (`:309`), which reads pre-mutation `op.data.ranges` / `reduction_ranges`, and
  `_tiled_dims_for_dep` (`:421`), which filters those extents by
  `dep.index.free_symbols`.
- `_apply_plan` (`:967`) — real IR mutation: `_divide_ranges` (`:817`),
  `_divide_reduction_ranges` (`:924`), layout resize via `_resize_device_layout`
  (`_inductor/ir.py:112`), then buffer propagation. That propagation inserts
  *tile-sized* copies into separately allocated full-size buffers —
  `_allocate_full_buffer` (`:1519`) then `_insert_copy_op` (`:1657`) on the write
  side, `_insert_read_copy_ops` (`:1907`) on the read side — plus reduction
  accumulation (`_insert_combine_op` (`:2307`), `_insert_reduction_copy_op`
  (`:2404`)).

Groups are produced by two mutually exclusive sources:
`hints_to_coarse_tile_groups` (`wsr/coarse_tile_hints.py:269`), which collects
contiguous runs of ops sharing a `frozenset` of hint IDs; and
`span_overflow_groups` (`wsr/coarse_tile_span_overflow.py:209`), which groups on
`_auto_span_plan_signature` (`:48`), lets a consumer *adopt* a run's split via
`can_conform_pointwise_tile` (`:448`), lets a `Reduction` join an open run via
`_reduction_shares_group_tiled_dim` (`:68`), and returns a
`(groups, dim_hint_assignments)` pair without applying the hints. The latter is
the existing "common tiling across ops" primitive, and the direct ancestor of
what this RFC generalizes.

`validate_coarse_tile_groups` (`:112`) forbids a hint scope spanning two groups;
`_validate_contiguous` (`:785`) requires a group to be a contiguous slice of
`graph.operations`.

### What a group boundary materializes

Buffer propagation is **per buffer at the boundary**, not per group.
`_propagate_tiled_op` (`:1322`) asks `_find_outside_consumers` whether anything
reads the buffer from a different outer `loop_group_id`, or whether it is a graph
output, and branches:

- **Neither** — the buffer is per-tile scratch reused every iteration. Its
  `output_tiled_dims` is set to `[]` and **no full-size buffer is allocated at
  all**. Values interior to a cut-free run never round-trip.
- **Either** — `_allocate_full_buffer` (`:1519`) splices a `SpyreEmptyFallback`
  into `graph.operations` *before the first op of the loop group*,
  `_insert_copy_op` (`:1657`) appends a copy op carrying
  `MutationLayoutSHOULDREMOVE(full_buf)` after the tiled op, and outside
  consumers are patched to read the full buffer. Symmetrically on the read side,
  `_full_buffer_read_deps` (`:1425`) / `_insert_read_copy_ops` (`:1907`) insert a
  tile-sized copy before a consumer that reads a buffer produced outside its own
  group.

The LX consequences are asymmetric, and they are what give a cut its price. A
tiled → tiled cut expands one producer → consumer edge into a four-node chain,
and each node lands differently:

```text
  ╭─ group A loop body ─────────╮                ╭─ group B loop body ────╮
  │ tile scratch ──▶ copy op    ├──▶ full_buf ──▶┤ read copy ──▶ consumer │
  │ Eligible         HBM only   │    HBM only    │ Eligible               │
  │ tile alloc       no storage │    full alloc  │ tile alloc             │
  ╰─────────────────────────────╯                ╰────────────────────────╯
```

`full_buf` is drawn outside both boxes because it is in neither loop body:
`_allocate_full_buffer` splices it in *before the first op of the loop group*,
and only the copy op — which does run in group A's body — touches it on the
write side. Everything inside a box executes once per tile; `full_buf` is
allocated once and is the sole node that outlives an iteration.

Note the write side is **two** nodes and the read side is **one**. The
write-side copy owns no storage — it aliases into a separately allocated
`full_buf` — whereas the read-side copy allocates the tile buffer it writes.
The two write-side rows below are therefore one relationship read from both
ends: the copy op points at `full_buf`; `full_buf` never points back.

| Node | Where | LX status | Why |
|---|---|---|---|
| Interior per-tile scratch | inside a group, no cut | **Eligible** | its own write address is fixed. `_is_tiled_advancing` (`scratchpad/utils.py:218`) keys on `output_tiled_dims` — *does this op's own write advance* — not `loop_tiled_dims`, which only says the op sits in a tiled loop. Sitting in the loop is not itself disqualifying |
| Write-side copy op (`coarse_tile_copy_*`) | producer, in the loop body | **HBM only** | it owns no storage: its layout **is** `MutationLayoutSHOULDREMOVE(full_buf)`, so its stores land in `full_buf`. `_op_output_good_for_lx_reuse` rejects that layout outright (`allocator.py:218`) — there is no allocation to place |
| Boundary `full_buf` | spans the cut | **HBM only** | stamped `"op not allowed"` (`allocator.py:308`): its producer is a `SpyreEmptyFallback`, i.e. an `ExternKernel`, and `_op_output_good_for_lx_reuse` requires a `ComputedBuffer`. Two later gates would reject it anyway, but never run — it is the mutation *target*, so its name is in `mutated_buffers` → `"mutation target"` (`:315`); and in the tiled → tiled case group B's read of it advances → `"tiled (advancing)"` via `_is_read_advancing_anywhere` (`:324`) |
| Read-side tile copy | consumer, in the loop body | **Eligible** | unlike the write-side copy this one *owns* its buffer — a physically smaller allocation with fresh contiguous tile-local strides, not an aliased view of `full_buf`. Its own write is fixed |

The interior scratch survives for a reason worth stating explicitly, because it
is not automatic: the write-side copy op derives its read and write tiled-dim
decisions *separately* (`_fixed_level_extents` vs. real per-level extents). Its
read of the scratch is deliberately non-advancing — the scratch is reused in
place — so `_is_read_advancing_anywhere`, which walks a buffer's *readers*, does
not flag it. Had the copy's read advanced, the scratch would be HBM too and
tiling would buy no residency at all.

Two facts follow that the design sections depend on. First, **a boundary buffer
never occupies LX**, so a cut has no LX-occupancy term to price (§3) — though a
cut is not otherwise free on the read side: where the consumer is tiled, the
read copy's advancing read evicts an untiled producer from LX candidacy (the
row-3 discussion below). Second, **tiling buys residency through the
interior**: what becomes pinnable is the loop-internal scratch and the read-side
tile copies, both tile-sized and fixed-address. `_is_tiled_advancing`'s docstring
states the rule directly — "a loop-internal buffer (e.g. drained by a copy op
every iteration) can be tiled yet have its own write pinned at a fixed address;
such a buffer is LX-eligible."

Both the full buffer and the copy ops are **inserted into `graph.operations`**,
not merely allocated. Liveness is index-based (`calculate_liveness`), so each
insertion shifts lifetime ticks for everything downstream of it (R7.1).

#### Not every cut materializes a buffer

`_apply_plan`'s propagation loop (`:1158-1164`) iterates
`for group_ops, _ in groups` — it visits **only ops inside a group**. An untiled
producer is therefore never rewritten: it gets no `full_buf`, acquires no
`MutationLayoutSHOULDREMOVE`, and keeps whatever LX eligibility it had. Only the
tiled consumer's own `_full_buffer_read_deps` / `_insert_read_copy_ops` fires.
What a cut costs therefore depends on which side carries a tile:

| Cut | Write side | Read side | Producer LX |
|---|---|---|---|
| tiled → tiled | `full_buf` + mutation copy op | tile-sized read copy | HBM only |
| tiled → untiled | `full_buf` + mutation copy op | consumer reads `full_buf` directly | HBM only |
| untiled → tiled | **nothing** — producer never visited | tile-sized read copy | **evicted** — advancing read (below) |
| untiled → untiled | nothing | nothing | unchanged |

Only the first two rows allocate. The third is the most common in practice — a
forced cut whose *producer* side is untileable (§2) is a row 3 or row 4, since
an untileable op is by definition never in a group. The converse is row 2: a
tiled chain feeding an untileable consumer still materializes `full_buf` plus
the mutation copy.

Row 3 allocates nothing on the write side, but it is not free. An untiled →
tiled cut requires a copy of a tile into the narrowed-span op: the read copy
reads the *full* producer buffer, advancing one tile per iteration
(`_insert_read_copy_ops` builds real per-level read extents,
`coarse_tile.py:2169-2213`, and says so at `:2131-2137`). That advancing read
is exactly what `_is_read_advancing_anywhere` walks, so the allocator stamps
the producer `"tiled (advancing)"` (`allocator.py:324`) — the same rejection
the first table applies to `full_buf`. The producer keeps its layout and is
never rewritten, but it loses LX candidacy. The exception is a read invariant
along every tiled dim of the consumer (`_tiled_dims_for_dep` filters it to
nothing): such a read stays fixed and the producer's eligibility survives.
Row 3's price is therefore one tile-sized copy plus, in the advancing case,
the producer's residency.

### Reduction-axis tiling

Coarse tiling can divide a *reduction* range, not only an output range.
`_divide_reduction_ranges` (`wsr/coarse_tile.py:924`) shrinks `K` to `K/T`,
leaving the output ranges untouched — the reduction axis is tiled in the
iteration space, and no buffer changes shape. Each iteration therefore writes a
full-shaped **partial** result, and `_propagate_tiled_reduction_op` (`:2501`)
folds those partials: a full-size HBM accumulator (`_allocate_full_buffer`,
`:1519`) seeded with `_reduction_identity_value` (`:764`), then a per-iteration
`_insert_combine_op` (`:2307`) mutating it in place through the reduction's
monoid operator, plus — in the nested case — a second, tile-sized accumulator
drained outward by `_insert_reduction_copy_op` (`:2404`).

It is gated by `enable_reduction_tiling` (`config.py:82`, default on) and is
reachable only through `spyre_hint`; the automatic planner never emits it
(`SpanOverflowTileLevel.is_reduction` is hardcoded `False`,
`wsr/span_overflow_hint_analysis.py:85-99`, because reduction-range tiling
"would require partial-result accumulation").

Four properties of that machinery bear directly on this RFC, and together they
are why it enters scope **single-level only and as a pinned singleton** (R1.8,
R4.6) rather than fully or as a fusible axis:

- **The accumulator is loop-carried.** Tile `t` reads what tile `t-1` wrote, so
  the op cannot share a loop nest with peers that tile at the same level.
  `_plan_is_loop_invariant_at_reduction_levels` (`:559`) admits only peers that
  are loop-invariant at every level some group member tiles a reduction dim at
  (`_group_reduction_tiled_levels_in_group`, `:485`), and `_seed_buffer_for_carry`
  (`:575`) rejects carry-propagating recurrences outright.
- **That invariant is not pairwise**, so §3's cut model cannot express it
  (R4.6).
- **It is not a pure working-set reduction.** The input span shrinks by the tile
  count, but an extra output-shaped HBM buffer appears and is
  read-modify-written once per tile — a trade whose sign depends on the
  `K`-to-output ratio, not a strict win.
- **`_validate_reduction_tiling` (`:1233`) over-approves.** Its docstring lists
  nested "outer output dims + innermost reduction dim (e.g. outer M + inner K
  for mm)" as supported, but every e2e test of that shape is either
  `correctness=False` ("nested tiling + reduction correctness bug") or
  `@pytest.mark.skip` ("inconsistent loop_count across reduction fill/combine
  nodes") in `tests/inductor/test_coarse_tile_e2e.py`. Only *single-level*
  reduction-axis tiling is numerically validated today. A predicate that admits
  known-wrong plans cannot serve as an enumerator's feasibility gate (R1.4).

### The CP-SAT solver

`CpSatLayoutSolver` (`scratchpad/ilp_solver_ortools.py:321`) works in alignment
units, wrapping each buffer in `_LifetimeBufferWithCpVars` (`:149`) or
`_CoreDivisionBufferWithCpVars` (`:244`). The joint wrapper creates:

```python
self.division = m.new_int_var(0, len(b.core_divisions) - 1, f"div_{b.name}")
self.eff_size = m.new_int_var(0, max(per_core), f"eff_size_{b.name}")
self.cores    = m.new_int_var(0, max(cores_used), f"occ_{b.name}")
m.add_element(self.division, per_core,   self.eff_size)
m.add_element(self.division, cores_used, self.cores)
```

Constraints: residency gated on pairwise division compatibility with every
consumer — `constrain_residency` (`:282`) loops the consumers, reads their pairs
from the precomputed `cd_parent_matches` via `match_pairs()` (`:279`), and
delegates each edge to the generic `_gate_divisions` helper (`:132`) with
`in_buffer` as the enforce literal; in-place reuse as shared-offset relaxation
(`_add_inplace_relaxation`, `:520`); and, called from that relaxation, global 2D
no-overlap over optional rectangles
`[start_time, end_time) × [offset, offset + eff_size)` with presence literal
`in_buffer` (`_add_no_overlap_2d`, `:568`).

Objective (`_run`, `:446`) is two-phase lexicographic: minimize
`sum(spill_cost)`, lock it with a rounded inequality
(`model.add(sum(hbm_terms) <= round(solver.ObjectiveValue()))`, `:479`), then
maximize `sum(cores)`. The second lexicographic phase is skipped entirely when no
buffer has a division to choose, which is the placement-only path. (This
"two-phase" is the objective's lexicographic structure, distinct from the
roadmap's Phase N; §4 / M2 collapse it to one minimized cost.)

### Why the three cannot simply be reordered

The tiling decision needs exact padded byte sizes and stick counts, so it wants
to run *after* stickification. The LX solver needs the buffer set that tiling
produces, so it wants to run *after* tiling. Work division needs stick counts
and is constrained by the same span limit as tiling. Any purely sequential
ordering makes one of the three guess about another — which is precisely the
`core_split_estimate = 1` guess that exists today. Folding them into M1's single
solve is what removes the guess; that solve sits post-stickification (§5).

## Proposed design

Sections 1–3 instantiate the roadmap's shared-model requirements for the tiling
axis: the config cross product is the M4 enumerate-and-table encoding (§1), its
joint span/divisibility check is M5 feasibility-on-the-combination (§1–§2), and
the per-adjacent-pair loop-nest-break booleans are the Phase 1 variables from
which grouping falls out of the objective (§2–§3). Section 4 delivers the M2
objective namespace, §5 places the solve in the pass list under M7/M8, and §6
covers tractability under C2 and defers enumeration scaling to Phase 5.

### 1. Config-as-unit (M4 encoding, M5 feasibility)

Collapse tiling (temporal split) and division (spatial split) into a single
per-operation **config** index. Precomputing the feasible cross product absorbs
every nonlinearity — divisibility, stick alignment, `MAX_SPAN_BYTES`, core
budget — into a table lookup, which is exactly how `CoreDivision` is already
consumed via `AddElement`. This is M4's enumerate-and-table rule applied to a
two-axis candidate; M5 is what makes the pair, not each axis alone, the unit of
feasibility (R2.3).

Divisibility in particular is enforced **at enumeration**, not by a runtime
constraint: `enumerate_tile_options` draws split counts only from
`divisors(basis)` (R2.2), so every tile size in the table already divides its
dimension evenly. An `AddModuloEquality` alternative — let the solver pick any
tile size, then constrain `size mod tile == 0` — is deliberately **not** used:
modular constraints are the nonlinear encoding M4 exists to avoid, they break the
linear-binding rule (R3.7), and they rediscover at solve time what enumeration
already knows. Enumeration's cost is bounded by the R1.7 caps; the
modular-constraint cost is not.

`CoreDivision` (`scratchpad/plan_solver.py:94`) is generalized rather than
forked. It carries just two stored fields — `output_splits` and
`reduction_splits`, both the coeff-keyed encoding from
`pass_utils.splits_by_index_coeff` — with `cores_used`, `output_partition`,
`is_clean`, and `signature_key()` all derived, so wrapping it costs nothing:

```python
@dataclass(frozen=True)
class TileOption:
    """One candidate coarse tiling of a single op, in that op's own frame."""
    dims: tuple[tuple[int, int], ...]     # (host_dim, split_count), outermost-first
    dedup_key: Hashable                   # R2.4 dedup only — not a compatibility key

    @property
    def tile_count(self) -> int: ...      # product of split counts


@dataclass
class PartitionConfig:
    """One jointly-feasible (tiling, core division) pair for an op."""
    division: CoreDivision
    tile: TileOption | None = None        # None == untiled (today's behaviour)
    per_core_bytes: int = 0               # precomputed
    cores_used: int = 0
    tile_count: int = 1
```

Two things about `TileOption` are load-bearing and easy to get wrong.

**`host_dim` is op-local; a level id is not.** `host_dim` indexes
`op_out_coords` (`pass_utils.py:363`) — the same frame
`SpanOverflowTileLevel.selected_host_dim` already uses, and the frame
`_candidate_host_dims`, `_split_candidates_for_host_dim`,
`can_conform_pointwise_tile` and `_dims_to_hints` all speak. It deliberately does
**not** carry a level or hint id. `coarse_tile`'s `levels` are keyed by
`hint_id`, which is a *group-scoped* identity: each member resolves a hint_id to
its own dimension through `op.dim_hints` → `loop_var` →
`_loop_var_to_ranges_pos` (`coarse_tile.py:729`). Under §3 a group does not exist
until the solve finishes, so a hint id cannot be assigned to a per-op option
enumerated before it. `span_overflow_groups` already sequences this correctly —
op-local signatures during grouping, then `next_hint_id` allocated only when a
group closes (`coarse_tile_span_overflow.py:303-310`, `:543-544`), then
`_dims_to_hints` (`:152`) per member, each op resolving its own `loop_var` from
its own output coordinates. The solver mints ids the same way, post-solve (R4.5).

**Compatibility is a relation, not a key.** `dedup_key` serves R2.4 and nothing
else. Two ops in one group routinely tile *different* host dims — the producer
its V output dim, the consumer the corresponding N dim — so equality of any
per-op key is the wrong test; `span_overflow_groups` matches split counts and
then verifies loop-variable correspondence through the dep. And
`can_conform_pointwise_tile` (`span_overflow_hint_analysis.py:1421`) is
asymmetric and non-equality: it asks whether one op can *adopt* another's
`split_by_host_dim`, checking divisibility, stick boundaries, and *sufficiency*
(the adopted split must fully cover the adopting op's own span pressure). §2's
`(tile_src, tile_dst, cut)` triple table holds a pairwise predicate natively,
so this costs nothing to express — see R4.7.

`CoreDivisionBuffer.core_divisions` becomes `configs: list[PartitionConfig]`,
`chosen_division` becomes `chosen_config`, and `cd_parent_matches` becomes
`config_matches`. Today's behaviour is exactly the `tile=None` slice of the new
space, so parity is directly checkable.

The first two renames are mechanical; **`config_matches` is not** (R2.6).
`_cd_parent_matches` (`allocator.py:1973`) does not compare coeff-keyed
signatures — it compares *physical per-core views* built from the producer's
`write_dep.index` and the consumer's `read_dep.index` through
`_prepare_per_core_view` / `_per_core_view_on_buf`, deliberately, because (per
its own docstring) that is "correct across reductions/reshapes, where a
coeff-keyed signature would conflate axes." Those indices and device layouts are
precisely what `_divide_ranges` and `_resize_device_layout` rewrite. So a
match entry is a property of the *config* pair, not the division pair, and it
cannot be read off the IR the solver sees.

Note also what the cross product is not: a filter. Divisions come from
`enumerate_work_division_candidates`, whose per-dim factors are
`divisors(basis)` over the op's *iteration space* — which tiling has already
divided. Tiling both **loosens** the span guard (`get_per_core_span` is
evaluated on the tiled space, so smaller tiles admit divisions that were
infeasible before — the R2.3 direction) and **tightens** divisibility
(`divisors(M/T)` is smaller than `divisors(M)`, and `adjust_it_space_for_sticks`
shifts the stick basis too). **Each tiling option therefore has its own division
candidate set, enumerated against its own iteration space.**

### 2. Model variables

The two-axis config index is the per-operation candidate the roadmap's Phase 1
"Variables" calls for; this section adds its second half — **one boolean per
adjacent pair of operations deciding whether a loop nest breaks there** — and the
per-buffer machinery that prices what a break materializes.

This is an **extension**, not a rewrite: the CP-SAT solver and allocator
interface gain config-based solving, but every existing path is preserved and the
untiled / work-division-seed config is always in the candidate set (R1.3, R2.4).
With the tiling gate off (R8.1) the model reproduces today's division and
residency solve, so parity is directly checkable and regressions are structurally
excluded.

The per-buffer vars live on a new wrapper, `_TilingBufferWithCpVars`, extending
`_CoreDivisionBufferWithCpVars` — the same subclass-and-override step that class
made over `_LifetimeBufferWithCpVars`. This is deliberate, not incidental: the
base wrapper exists so "one object flows through the solve instead of a buffer
list shadowed by a parallel `name -> {var}` dict" (`ilp_solver_ortools.py:150`),
and the tiling vars are exactly the kind of per-buffer state that would otherwise
accrete into such a dict. New vars are created in `__post_init__` after
`super().__post_init__()`, and the constraints tying them together are added by
the hooks, as today.

The candidate space is **two-level**, mirroring how §1 enumerates it — tiling
first, then that tiling's own division set:

- `tile[b] ∈ [0, |T_b|)` — which tile option. Slot 0 is reserved for the **unity**
  option (no dimension split), which every op has.
- `div[b] ∈ [0, |D_b(tile)|)` — which core division, *within* the chosen tiling.
- `config[b]` — the flat pair index, tied by a single
  `AddAllowedAssignments([tile[b], div[b], config[b]], <valid triples>)`. The
  division sets are ragged (a tiling both loosens the span guard and tightens
  divisibility, §1), so an allowed-assignment table is what expresses that
  exactly; a rectangular `tile × D_max` encoding would admit pairs that do not
  exist.
- `eff_size[b]`, `cores[b]`, `tile_count[b]` — each via
  `AddElement(config, <table>, var)`, generalizing the two existing
  `add_element` calls.
- `in_buffer[b]`, `offset[b]`, `top[b]`, `merge_vars[b][parent]` — unchanged.
- `cut_parents[b]`/`cut_children[b]`, `escapes[b]`, `boundary_op[b]`,
  `full_size[b]`, `boundary_view[b]` — new; §2 defines them.

Keeping tiling as its own level is what makes `tiled[b]` free:

```text
tiled[b]  ⟺  tile[b] != 0
```

No 0/1 table and no `AddElement` — being tiled is *not selecting the unity
option*, read straight off the variable. A flat `config[b]` would have had to
recover the same fact through a lookup, having just discarded it.

It also shrinks the cut tables. Whether two ops can share a loop nest is a
property of their **tilings**, not their divisions — `can_conform_pointwise_tile`
takes `(op, split_by_host_dim, sencores)`, where `sencores` is the machine's core
count, not either op's chosen division (R4.7). So the per-edge table is keyed on
`(tile_src, tile_dst, cut)`, `|T_src| × |T_dst|` rows rather than
`|T_src|·|D_src| × |T_dst|·|D_dst|`. Division compatibility stays where it
already lives, on the slicing gate (`config_matches`), which is conditional under
`in_buffer` rather than unconditional like cut.

`cut[i]` is graph-level in *indexing* but is **not** a single shared graph
variable — it is realized as the per-buffer cut bools tied by equality that §2
details below (`cut_parents`/`cut_children`, reconciled by `_add_cut_equalities`).
What stays graph-level is the indexing: `cut[i]` runs over *program-order
adjacency*, not dataflow, so the two ops it joins need not be related to any one
buffer — and its feasible values come from the `(tile_src, tile_dst, cut)` triple
for that op pair. A multi-output op compounds this: one edge, several buffers. So
it has no single per-buffer home, which is why it lives as equated per-buffer
claims rather than one variable.

What each wrapper holds is a **neighbour view**, split by direction:
`cut_parents[b]` and `cut_children[b]`, built once in `__post_init__`. Edge `i`
joins ops `i` and `i+1`; for each edge `b` spans, the wrapper mints one bool
and indexes it twice — under the edge's upstream op (its *parent*, `i`) in
`cut_parents[b]`, and under its downstream op (its *child*, `i+1`) in
`cut_children[b]`. The split is what makes each claim's direction explicit: an
op name alone does not identify an edge end — an op interior to `b`'s span is
the child of the edge entering it and the parent of the edge leaving it — and
R4.7's admitting predicate is directional, so which end a claim refers to must
be carried by the dict identity, not by convention. Both dicts mirror
`merge_vars` (neighbour name → var) and `cd_parent_matches` (neighbour name →
table), so the boundary machinery indexes the way the rest of the wrapper
already does.

Each wrapper mints its **own** bools, exactly as `merge_vars` does — the wrapper
stays self-contained and depends on nothing outside itself. Where two buffers
span the same edge, their bools are tied by an **equality**, so the duplicates
are duplicates in name only.

Building the dicts costs nothing. The wrapper is already a `LifetimeBoundBuffer`,
whose `uses` is the sorted list of op indices at which the buffer is accessed,
with `start_time`/`end_time` derived from it (`plan_solver.py:56`) and already
serving as the time axis of `_add_no_overlap_2d`. The edges spanned by `b` are
`[start_time, end_time - 1)` — producer up to last use, excluding the edge
after the last use, which no consumer crosses (`end_time` is `uses[-1] + 1`).

**This structure depends on the op order being fixed** (R4.4). Because the
solver never reorders, program-order adjacency is static and fully known when the
wrappers are constructed, so the topology can be baked in up front. Were
reordering a solver decision, adjacency would itself be variable and the claim
dicts could not be built up front at all.

Buffers spanning a shared edge are tied by one equality per claim, posted by an
`_add_cut_equalities(model, tensors)` sweep in `_run`, alongside the existing
`_add_inplace_relaxation` / `_add_core_division` steps. No new type: the
per-edge tables and pins ride on the buffer the same way `core_divisions` and
`cd_parent_matches` already do, each wrapper installs its own
`AddAllowedAssignments` and pins from them, and the sweep only reconciles
claims. CP-SAT presolve substitutes equality-linked bools away, so the solved
model is the size it would have been with one shared var.

Tying by equality rather than by sharing one variable is what makes the
invariant **checkable**: the sweep sees every claim on every edge, so "all
claimants agree" is an assertion it can make. With claims direction-indexed,
orientation is assertable too: every claim knows whether its key op is the
edge's parent or its child, so the sweep can require all claimants of an edge
to agree on which op is which — the same producer-then-consumer orientation
R4.7's predicate is evaluated in and the `(tile_src, tile_dst, cut)` table is
built in. A claim keyed the wrong way round surfaces as a reconciliation
failure instead of silently tying the wrong pair. A design where wrappers alias a
single var has nothing to assert — the object is either the right one or it
silently is not.

Reading the plan back works the same way. R4.5 needs `groups` in `coarse_tile`'s
`(ops, levels)` shape, which means turning solved cuts into contiguous runs; the
equalities guarantee every claimant of an edge reports the same value, so
`_extract` can walk the wrappers in op order and cut where any claim is 1.

**Every per-config scalar reaches the model the same way.** Any function
`f(PartitionConfig) -> int` can be precomputed into a per-buffer table and bound
to a symbol by one more `AddElement`. That is the only extension mechanism the
model needs, and it is what the enumeration cost buys: a nonlinear property of a
tiling becomes a table entry rather than a constraint. It also fixes the limit —
a *scalar* derived from the chosen tile is available to the objective, but the
tile **shape** is not a decision variable. An objective term sensitive to *which*
dimension was split must fold that into a precomputed per-config number (§4).

New, at graph level:

- **`cut[i]`** — one boolean per adjacent pair in `graph.operations`.
  `cut[i] == 0` means the two ops share a loop nest.

`cut[i]` is *determined*, not merely constrained: a precomputed table of
`(tile_src, tile_dst, cut)` triples installed with `AddAllowedAssignments`
admits `cut == 0` only on tiling-compatible config pairs, so an incompatible
pair forces a cut. This is the same shape the per-edge helper `_gate_divisions`
(`:132`) already uses for divisions, widened by one column.

`cut[i]` is indexed over *program-order adjacency* in `graph.operations`, because
that is what a loop nest is. It is distinct from `config_matches` (§1, today's
`cd_parent_matches`), which stays a plain pair set rather than a triple table
because `constrain_residency` applies it *conditionally*, under `in_buffer`,
whereas cut holds unconditionally.

Stickification/relayout optimization — choosing configs to *avoid* a restickify,
which would need a second per-edge `relayout[e]` variable over *dataflow*
producer→consumer edges — is **the roadmap's Phase 3, not this phase** (R6, R9).

Crucially, `cut[i]` is indexed over **all** adjacent pairs in `graph.operations`,
not just tileable ones, and is **pinned to 1** at any boundary where either side
cannot be tiled — or was tiled on a reduction axis, hint- or solver-chosen
(R4.6). That
is what makes §3's contiguity guarantee structural rather than aspirational: an
untileable op can never end up inside a cut-free run.

The reduction-axis case is pinned for a sharper reason than untileability. The
rule governing such a group quantifies over the whole run
(`_plan_is_loop_invariant_at_reduction_levels`), and pairwise compatibility
provably does not compose into it, so no widening of the triple table would make
it expressible. R4.6 carries the counterexample; the single-level reduction
options R1.8 now admits are pinned as singletons the same way, so the case is
handled identically whether the reduction tiling is hint- or solver-chosen.

#### A cut adds a buffer, it does not evict its tiled producer

A cut does not evict its tiled producer. Both branches of `_propagate_tiled_op` end
with `output_tiled_dims = []`, and the second says so outright
(`coarse_tile.py:1367`): *"The tiled op's own buffer is always loop-internal
scratch here: it is fully drained by the copy op inserted above before the next
iteration overwrites it."* So `b` keeps its tile-sized layout, stays out of
`mutated_buffers` (`full_buf` is the mutation target, not `b`), is not
tiled-advancing, and is not read-advancing — the copy op's read of it is fixed.
`b` is LX-eligible either way. What a cut changes is **how many tile-sized
eligible buffers exist**:

| Solver's choice | Eligible allocations | Which |
|---|---|---|
| no cut | **1** | `b`, read in-group |
| a cut | **2** | `b` (still scratch) **+** the read copy in the consuming group |
| `in_buffer[b] == 0` | **0** | — |

The second buffer is the point. `b` is never rewritten into `full_buf`;
`_allocate_full_buffer` mints a *separate* HBM buffer, `_insert_copy_op` drains
`b` into it, and `_insert_read_copy_ops` allocates a fresh tile-sized buffer in
the consuming group. Only that last one is new to the packing model.

In the applied IR the two no longer meet: outside consumers are patched to
read `full_buf`, so `b`'s applied lifetime ends at its last in-group reader —
the write-side copy op inserted immediately after the tiled op, or a later
in-group consumer — while the read copy lives in the consuming group. The
**model** cannot claim that ordering: it runs pre-mutation (§5), so `b`'s
interval is derived from pre-mutation `uses`, whose last entry is still the
outside consumer — on the model's time axis the two rectangles overlap at the
consumer's tick. This document keeps that overlap as deliberate conservatism: `b`
retains its full pre-mutation extent whatever the cut variables say, the read
copy is an additional optional rectangle, and no stacking is assumed. The
error direction is safe — a cut's LX footprint is over-, never under-stated —
and it biases the solver toward fewer cuts, never toward a wrong address; the
placement re-solve prices the real, applied intervals. Expressing the trade
exactly (a cut-conditional interval end, e.g. a complementary
optional-rectangle pair sharing one offset var) is deferred until this
pessimism is shown to matter.

So the model needs no residency constraint tying `in_buffer[b]` to `cut[i]`. What
it does need is for the second rectangle's **existence** to be conditional, since
it exists only when the cut does:

- **`tiled[b]`** — bool, `tile[b] != 0` (§ *Model variables*): the chosen tiling
  is not the unity option. Gates what materializes on each side of a cut — a
  tiled producer yields a `full_buf` and a copy op, an untiled one yields neither
  (`_apply_plan`'s propagation loop visits only ops *inside* a group).
- **`cut_parents[b]` / `cut_children[b]`** — the direction-indexed claim dicts
  over the edges `b` spans (producer to last use): one minted bool per spanned
  edge, keyed by the edge's upstream (parent) op in the first and by its
  downstream (child) op in the second. Buffers sharing an edge are tied by
  equality, not by aliasing one var.
- **`escapes[b]`** — bool. Does any consumer land outside `b`'s own run? Cut-free
  runs are contiguous, so that is exactly "some spanned edge is cut" — an OR
  over the spanned-edge bools (either dict enumerates them exactly once), with
  no per-consumer bookkeeping.
- **`boundary_op[b]`** — bool. Does a `full_buf` get allocated for `b`? An
  existence flag, not a residency gate: the buffer it names is real IR that
  `coarse_tile` inserts, which shifts every downstream liveness tick (R7.1).

```text
escapes[b]      ⟺  OR(cut_children[b].values())  ∨  b is a graph output
boundary_op[b]  ⟺  tiled[b] ∧ escapes[b]
```

The graph-output disjunct is not redundant with the existing
`"graph output (no clone)"` gate. Under `clone_at_graph_boundaries()` a graph
output *may* reside (`allocator.py:334-342`), but `_find_outside_consumers`
treats a graph output as escaping, so a **tiled** graph output still materializes
a `full_buf` even with no consumer outside its run.

#### Read copies are conditional rectangles

`_insert_read_copy_ops` allocates one tile-sized buffer per full-extent input a
tiled op reads, deduped by buffer name. These are ordinary allocations with their
own tile-local strides — LX-eligible, and exactly the buffers tiling exists to
make resident. They enter `_add_no_overlap_2d` as real rectangles.

But they **do not exist until the solver decides they do**, and in two different
ways, per `_full_buffer_read_deps`:

- reading a cross-group producer — exists only if a cut separates them;
  conditional on `escapes[producer]`;
- reading a graph input, constant, or untiled producer — exists whenever the
  consumer is tiled; conditional on `tiled[consumer]`.

Neither is unconditional, so neither rectangle's presence is `in_buffer` alone.
Treating them as always-present over-reserves LX in the cut-free case; omitting
them under-reserves, with a known direction — the model would **undervalue
tiling**, since the omitted buffers are precisely the ones tiling makes pinnable
(R7.1). Both rectangles are therefore optional, with the literal above as the
presence condition and `in_buffer` gating residency on top of it.

The second bullet carries an eviction side too. When the full-extent input is
an untiled `ComputedBuffer` producer and the consumer's dep advances along a
tiled dim, the realized read copy evicts that producer (*Background*, row 3).
The model must say so: the read copy's existence literal enforces
`in_buffer[producer] == 0` on such edges. Advancement is a property of the
*(edge, consumer tile option)* pair, not of the edge alone — a dep touching
host dim `N` is invariant under an option that tiles only `M` and advancing
under one that tiles `N`, and mixed cases are common (broadcast inputs above
all). The implication is therefore keyed per pair: precomputed at
table-construction time by evaluating `_tiled_dims_for_dep` under each
option's per-level extents, and enforced only under the consumer tile options
whose tiled dims intersect the dep's free symbols. Options that leave the dep
invariant omit it. This is a narrower rule than the general eviction
constraint R4.8 forbids — it fires only on row-3 edges, under only the tile
options that realize them, from the mechanism the allocator will actually
apply.

Who creates what is split cleanly between solve and apply. The wrappers and
tables for these predicted rectangles are built by model construction from the
enumerated configs — pure prediction, no IR. The buffers themselves are
created only by the apply step (§5), which reads the solver's declared per-op
config and cut assignment and runs `coarse_tile` as today. Predicted
rectangles take **interstitial time coordinates**: the tick axis is scaled by
a small constant so inserted-op positions land between the integer ticks of
real ops — the write copy just after its producer, the read copy just before
its consumer — and a predicted insertion never renumbers a real buffer's
lifetime. R7.5's index-shifting concern thereby applies to realized IR only,
where `calculate_liveness` recomputes ticks before the placement re-solve;
R7.2's fidelity check compares predicted against realized lifetimes after
normalizing both to rank order — as equality for buffers no cut touches, and
as containment (predicted ⊇ realized) for a cut producer, whose model
rectangle deliberately keeps the pre-mutation extent while its applied
lifetime ends at its last in-group reader.

#### Boundary shape and core division

`boundary_op[b]` says a full buffer appears; its shape and slicing are
deterministic functions of the producer's chosen config, since
`_allocate_full_buffer` derives the full buffer's device layout by scaling the
per-tile one up with `_resize_device_layout`. Both are table entries, bound
exactly as `eff_size[b]` and `cores[b]` are:

- **`full_size[b]`** — `AddElement(config[b], <full-extent footprint>, ...)`.
  This is *not* `eff_size[b] × tile_count[b]`: the full buffer is stickified at
  the full host extent, so per-tile padding to stick boundaries does not survive
  the scale-up. At `eps = 64`, a stick dim of 320 split 4 ways is `4 × ceil(80/64)
  = 512` elements per-tile but `ceil(320/64) = 320` as one buffer.
- **`boundary_view[b]`** — the full buffer's per-core view, from the same table
  mechanism, so agreement against the consumer's read is expressible. Without it
  the model cannot distinguish two orthogonal producer/consumer divisions that
  happen to share a slice count from a genuine match — the conflation that has
  already produced one wrong-output bug in the co-optimizing path (R2.6).

**Pricing is track C1's, not this document's.** What a cut *costs* — the HBM
traffic through `full_buf`, the loop overhead, how the cut's two rectangles
(sequential only in applied IR, §2) trade against the one they replace — is
calibrated by the roadmap's cost-model track C1, not fixed here. This section
fixes only what exists and what is eligible; the objective consumes
`boundary_op[b]`, `full_size[b]` and `boundary_view[b]` but does not define what
they are worth.

### 3. Tiling groups fall out of the cut variables

**A tiling group is a maximal cut-free run of operations.** This realizes the
roadmap's Phase 1 promise that "grouping then falls out of the objective — a
tiling group is a maximal run with no break — rather than being precomputed by a
grouping heuristic that a wrong guess would make unrecoverable." It is the
central simplification, and it buys three things at once:
1. `_validate_contiguous` is satisfied *structurally*. Because `cut[i]` ranges
   over adjacent pairs of `graph.operations` and is pinned to 1 wherever either
   side is untileable (§2), a maximal cut-free run is by construction a
   contiguous slice of `graph.operations` — exactly what `_validate_contiguous`
   (`:785`) checks against the `op_to_position` map `coarse_tile` builds.
   Contiguity never has to be expressed as a constraint. An untileable op
   between two ops that would otherwise merge simply forces a cut; hoisting such
   interlopers out of the way stays `reorder_unhinted_interlopers`'s job (R4.4),
   not the solver's.
2. Grouping is driven purely by the objective, as required: the solver merges
   two ops into a group exactly when agreeing on a tiling scores better than
   paying for the boundary copies a cut would materialize.
3. A cut is never priced *directly*. It is priced through the consequences it
   materializes, and this RFC's job is to make those consequences **visible** to
   the objective rather than to price them: `boundary_op[b]` says a `full_buf`
   exists, `full_size[b]` and `boundary_view[b]` say at what shape and slicing,
   and the read copies enter the packing model as optional rectangles under the
   same literals (§2). There is deliberately no `n_groups` penalty term — the
   performance profile is read off the real outcomes (tile shape, LX pinning
   status), not off a proxy for them. Whether a cut still warrants an explicit
   signal of its own — a loop-overhead term beyond the consequences it
   materializes — is an open point for track C1.

   What those consequences are *worth* — HBM traffic through `full_buf`, loop
   overhead, and how a cut's two tile-sized rectangles (sequential in applied
   IR, conservatively concurrent in the model — §2) trade against the one they
   replace — is the roadmap's cost-model **track C1**, calibrated there rather
   than here. Note only that the consequences are producer-dependent, so
   any price must be too: a cut between two tiled configs materializes a
   full-size HBM buffer plus a mutation copy op, while a cut whose producer is
   *untiled* materializes only a read copy in the consumer — though it can
   still cost that producer its LX candidacy (*Background*, row 3). A uniform
   per-cut constant would overprice every untiled → tiled edge, which is most
   forced cuts.

`validate_coarse_tile_groups`'s invariant (a hint scope must not split across
groups) becomes a constraint: `cut[i] == 0` is forced for every `i` interior to
a hint scope.

### 4. The M2 objective namespace (delivered by this phase)

Requirement M2 — that the objective be injected rather than hardcoded — is a
shared-model requirement, but Phase 1 is where the namespace and lowering are
*built*, because it is the first phase whose axis contributes terms and the
roadmap lists "the M2 objective namespace" among what Phase 1 owes later phases.
This section specifies that deliverable; it does not re-argue M2's rationale (see
the roadmap). A new module `torch_spyre/_inductor/scratchpad/cost_expr.py`
provides:

- **A symbol namespace** the solver binds to model variables. Per-buffer:
  `size`, `read_count`, `in_lx`, `spilled`, `cores`, `tile_count`,
  `is_intermediate`. Aggregator: `SumOverBuffers`. Globals:
  `total_hbm_bytes`, `peak_lx_bytes`, `idle_cores`.
  `peak_lx_bytes` is defined as the packing high-water mark — one
  `AddMaxEquality` over the `top[b]` vars `_add_no_overlap_2d` already creates
  (`:568`) — *not* a time-indexed occupancy sum, which would cost a constraint
  per timestep and is why the naive reading of "peak" is rejected.
  `tile_count` and `in_lx` are the primitives from which a performance profile
  is derived; there is no group-count symbol.

  `SumOverEdges` is **reserved, not provided**. With relayout deferred (R6) no
  edge-indexed term survives, so shipping the aggregator with nothing to
  aggregate would be dead API. The name is held for R6.3's
  `relayout_bytes = SumOverEdges(relayout[e] * bytes[e])`, and `relayout_bytes`
  is likewise absent for now.

  Anything beyond this list must arrive as a per-config scalar, precomputed into
  the §2 table and bound by one more `AddElement`. The tile *shape* is
  deliberately not a symbol (R3.7).
- **A lowering** `lower(expr, bindings) -> cp_model.LinearExpr` over an
  explicitly bounded sympy subset (R3.3). Anything outside that subset raises
  `CostExpressionError` naming the offending node. Silently approximating an
  objective is worse than a compile error.
- **A default objective** built from today's terms:
  `SumOverBuffers(spill_cost) - SumOverBuffers(cores)`.

The objective is a **single expression minimized in one phase** — the model
computes one total cost and minimizes it (`Minimize(expr)`), with no
lexicographic sequence and no per-phase locking. This is the collapse M2
mandates ("one expression in one unit, not a ranking"), not a decision this
document makes on its own; `CostSpec` therefore wraps a single sympy expression,
and a bare expression is the normal form. The hard guarantee that parallelism can
never buy a spill is no longer structural but a matter of relative weight — the
default weights the spill term to dominate the core term so the practical outcome
tracks today's spill-first intent (R3.2, R3.5).

**The predictor is load-bearing for what this objective *means*, not only for how
accurate it is.** §3 prices a cut partly through residency the run's interior
loses, and no symbol here expresses that directly — there is no "this buffer
would have been scratch under a different cut assignment" term. The pricing works
because the *predicted buffer set itself* varies with the cut assignment: a
different assignment yields a different set of buffers to sum `spill_cost` over.
So R7.1's predictor is not merely an accuracy input to the objective, it is part
of the objective's definition. A predictor that omits the buffers tiling creates
(R7.5) does not make the objective slightly wrong — it makes it price a different
question.

### 5. Pipeline placement — decide once, apply in dependency order

This section places the solve in the pass list under the roadmap's M7 (pure
prediction, then apply) and M8 (addresses from the final solve). The solve needs
device layouts, so it sits post-stickification. Manual hints keep applying
pre-stickification exactly as today, preserving the rationale spelled out at
`passes.py:419-425` (running before stickification means `_divide_ranges` never
calls `_resize_device_layout`, which is what dissolved the
`insert_restickify`→hint cross-phase contract, issue #3135). The solver sees
hint-tiled ops as **pinned single-config buffers** — a pin in M3's sense, lowered
at enumeration time under H2 (R5) — and optimizes the rest of the graph around
them.

```text
  propagate_named_dims, validate_named_dims          # 426-427
  assign_dim_hints                                   # 428
  _maybe_reorder_unhinted_interlopers                # 429
  _maybe_coarse_tile_hints                           # 430 — unchanged, hints
                                                     #       stay authoritative
  # --- Tensor Layout (Stickification), passes.py:433-441 ---
  split_multi_ops                                    # 433
  propagate_spyre_tensor_layouts                     # 434
  validate_ops, optimize_restickify_locations        # 435-436
  finalize_layouts                                   # 437
  insert_restickify                                  # 438
  enforce_indirect_access_layout,
  insert_post_mutation_restickify, insert_bmm_padding # 439-441
  dedup_and_promote_constants                        # 443
+ unified_partition_solve                            # NEW: one CP-SAT model
+     -> per-op config, cuts, residency intent
+ coarse_tile(graph, groups, group_idx_offset)       # apply chosen groups, at the
                                                     # 448 slot span-overflow holds
  span_reduction, _distribute_work                   # 451-452 — commit divisions
  _maybe_scratchpad_planning                         # 455 — placement-only
                                                     # re-solve, divisions fixed,
                                                     # warm-started from intent
```

`_maybe_coarse_tile_span_overflow` is subsumed **on the success path**:
span-forced tiling becomes a feasibility constraint on configs rather than a
separate pass. The pass itself is retained — skipped when the solve succeeds,
run verbatim when it raises (the R8.3 failure path) — so a graph whose
feasibility requires tiling still compiles when the joint solve fails.

The final placement solve remains **authoritative for addresses** — this is M8.
The joint model decides on a *predicted* buffer set; addresses are then computed
over the real post-tiling buffers.

**The solver emits only legal, systematically-applicable plans; an illegal
emission is a failure, not a fallback case.** This splits the seam between the
predicted model and the applied IR cleanly in two. A *sizing or lifetime*
misprediction is legal — the plan still applies — and merely degrades to a
spill, never a wrong address, repriced by the placement re-solve. But a plan
`coarse_tile` cannot apply, or one whose residency was gated on a per-core view
the applied IR does not honour, is **illegal**: the model must make it
infeasible **by construction** — every admitting predicate (R4.7 cut
compatibility, R2.6 view agreement) must be *sufficient* for applicability, not
merely necessary — and the post-apply validation that checks this is an
assertion whose firing is a hard failure, never a silent degrade-to-greedy that
would mask the modelling bug. This is distinct from *no* solution (INFEASIBLE,
or a timeout with no incumbent), which is the legitimate R8.3 fallback to
greedy.

### 6. Tractability (track C2; enumeration scaling is Phase 5)

- **Whole-graph model.** The solve is a single CP-SAT instance over the entire
  graph — it is **not** decomposed at matmul or any other op boundary, and makes
  no op-specific segmentation assumption. Matmul operands come from HBM, so
  matmuls are untileable; §2's untileable-pinning already forces cuts on their
  edges as a consequence of that, not as a special case.

  Those forced cuts are also **cheaper than they look**. A cut whose untileable
  op is the producer is a row 3 or row 4 of the cut-cost table (*Background*):
  it materializes a read copy at most, never a full-size HBM buffer — though a
  row-3 edge still costs the tile copy and, when the read advances, the untiled
  producer's LX candidacy. A tiled chain feeding an untileable consumer is
  row 2 and does pay `full_buf` plus the mutation copy. The common reading —
  that *every* cut costs a full-size round-trip — would still make a
  whole-graph model look far more expensive than it is, and is the main reason
  segmentation at matmul boundaries is unnecessary rather than merely unwanted.
- **No per-op config cap.** Model size is controlled by external pruning of the
  enumerated config set rather than a fixed ceiling (see R2.4 and *Open
  questions*). The enumeration-scaling levers themselves — signature dedup,
  per-dimension channeling, lazy enumeration — are the roadmap's **Phase 5** and
  are deferred there; this phase ships the eager cross product, leaving those
  levers to Phase 5.
- **Enumeration cost, not just model size.** Two per-op costs now scale with the
  tiling-option count rather than being paid once:
  `enumerate_work_division_candidates` runs per option (R2.2), and
  `_views_for_divs`'s sympy-heavy prep is no longer candidate-invariant, so its
  cache key gains the tile (R2.6) and it is built once per `(op, dep, buf, tile)`
  rather than once per `(op, dep, buf)`. The prep was introduced precisely to keep
  view cost proportional to ops rather than candidates; tiling reinstates a factor
  of the option count.
- **Warm-start** via `AddHint` with the current heuristic's plan. The solve is
  then genuinely **anytime**: a timeout keeps the best incumbent found (R8.3),
  and because that incumbent is never worse than the warm-start plan under the
  injected objective, early stopping does not regress below the heuristic's own
  plan — the spill-dominant default (R3.5) keeps that tracking today's spill-first
  intent. The harder floor is the `SolveError` path — `INFEASIBLE` or no
  incumbent, caught at the new pass slot, then the existing tiling method with
  the greedy solver — which is M9 ("failure never discards a pin"): the fallback
  runs over the already-pinned candidate sets, so a hint is respected there too.
  Determinism and fallback follow track C2.

### 7. Padding (Phase 1, not an axis)

The roadmap folds padding into Phase 1 because this is the first phase that needs
it — for **sizing** (roadmap consequence 3). Padding gets no decision axis and no
index space of its own; it lands entirely inside the config tables §1 already
builds. Two cases, exactly as the roadmap frames them.

**The pad required to reach legality is *derived*, so it is a scalar on each
candidate row.** `compute_padding` (`padding.py:73`) rounds a dimension up to one
stick, and given the layout and the tiling there is no freedom — nothing to
decide. The only change is that a config's predicted buffer sizes are computed
from the padded `device_size`, not from the unpadded shape, so the sizing every
later phase is defined over is correct. This is precisely the case M4's "no new
index space" rule was written for: `f(config) -> int` (here, padded bytes) is a
table entry, not a variable.

**Padding *beyond* legality is the one part with a genuine choice, and it enters
as extra rows, not a new axis.** A discretionary pad can unlock a divisibility
that `valid_split` requires (`work_division.py:809-823`), turning an illegal core
split into a legal one — so it widens the feasible config set. Those extra
configs are additional rows in the same per-op table, ranked by the same
objective; still **no new index space**.

Whether those rows are worth enumerating is an empirical question this document
does not presume the answer to, because padding is not free today:
`lower_pad_sequence` (`pass_utils.py:1191`) emits a four-op sequence — allocate,
fill constant, fill the pad region, copy — per matmul, with no sharing between two
matmuls reading the same operand (`padding.py:183`); the y operand's buffer grows
and competes for LX; and `K → K_padded` widens the SDSC iteration space at codegen
(`_extend_matmul_k_to_padded`, `codegen/superdsc.py:870`). The gate is a
measurement — an unaligned-`K` matmul against the same model pre-padded by hand.
If the delta is noise, the derived pad stays purely in the apply step and **no
discretionary pad row is ever enumerated**; only if it is real do the extra rows
land.

Either way this phase lifts the fixed policies in `padding.py`, since the derived
amount cannot be expressed without them: pad operands other than y, dimensions
other than K, ends other than the right, and multiples other than one stick; and
share a padded buffer between matmuls reading the same operand rather than
emitting a pad sequence per matmul (`padding.py:183`). The **other** half of
consequence 3 — padding as a *layout legalization* tool, which would remove the
issue #1756 restriction at `propagate_layouts.py:271`, `:455`, and `:1078` — is
**not** this phase's: it lands in Phase 3 with the layout search that consumes it.

## Requirements

### R1 — Tiling enumeration

- **R1.1** Add `enumerate_tile_options(op, *, max_dims, max_splits_per_dim,
  max_options) -> list[TileOption]`, returning **all** feasible options within
  the R1.7 caps, in the deterministic feasibility-tiered order R1.2 defines and
  truncated to `max_options` by it, tiling output ranges or a single reduction
  level (R1.8). This is the behavioural change
  from `_search_min_cost_tile_plan`
  (`:1269`), which returns the *first* combo in `_combo_cost` order. Both of its
  failure modes are preserved: it *raises* `Unsupported` when no combo passes
  (`:1395`, `:1399`), and returns `None` only when there are no candidate host
  dims at all (`:1303`).
- **R1.2** `_combo_cost` is **dropped** — the injected objective (§4) is the only
  ranking, applied by the solver over the enumerated set, so the enumerator does
  not pre-rank options by a cost proxy. It instead emits every feasible option
  within the R1.7 caps in a **deterministic, feasibility-tiered** order, and
  truncation to `max_options` (when a cap binds) drops from that order's tail.
  The tiers, outermost first: (1) the **mandatory keeps** — the untiled option
  (R1.3) and the work-division-seed pair (R2.4) — never truncated; (2)
  span-pressure-relieving options (R1.9); (3) speculative residency-driven
  options. Within a tier the order is a canonical key over the option's
  `(host_dim, split_count)` tuples, which doubles as the deterministic tie-break
  R8.4 relies on. Feasibility priority, never cost, decides which options survive
  the cap; the objective decides quality among the survivors. The
  constraint-based encoding that lets larger problems avoid the cap altogether —
  per-dimension channeling — is Phase 5.
- **R1.3** The untiled option `TileOption(dims=())` is always present, so
  feasibility is never worse than today.
- **R1.4** Validity predicates must be **reused, not reimplemented**. From
  `wsr/span_overflow_hint_analysis.py`: `_within_stick_host_dim` (`:240`),
  `_post_tile_stick_alignment_error` (`:263`), `_candidate_host_dims` (`:911`),
  `_cap_split_candidates` (`:979`), `_input_stick_alignment_error` (`:1042`),
  `_split_candidates_for_host_dim` (`:1100`), `_iter_split_combos` (`:1195`),
  `_combined_tile_stick_alignment_error` (`:1215`),
  `_remaining_span_candidates_after_tile` (`:1236`),
  `_host_dim_has_legal_nontrivial_split` (`:936`, the R1.9 candidate source), and
  `can_conform_pointwise_tile` (`:1421`, the R4.7 adoption predicate). From
  `pass_utils.py`: `coeff_through_floor` (`:848`, sub-stick guard) and
  `op_out_coords` (`:363`, the frame `host_dim` indexes).

  `_remaining_span_candidates_after_tile` carries more weight than its position
  in that list suggests: it is the span-*sufficiency* check — does any overflow
  survive this tiling — and both public entry points compose it
  (`_search_min_cost_tile_plan` at `:1344`, `can_conform_pointwise_tile` at
  `:1471`). R2.3's joint per-core, per-tile span feasibility is that same
  question asked of a config pair, so it is the predicate to extend rather than
  restate.

  `_seed_buffer_for_carry` (`:575`) **is** reused — it rejects carry-propagating
  recurrences, which single-level reduction tiling (R1.8) must reject too.
  `_validate_reduction_tiling` (`coarse_tile.py:1233`) is deliberately **not**: it
  over-approves, admitting the nested known-wrong shapes (*Background*), so it
  cannot be the feasibility gate. What gates the reduction options R1.8 does emit
  is instead structural — a single reduction level, no nesting — backed by R1.6's
  apply-and-compare-to-CPU test. `_validate_reduction_tiling` continues to run
  inside `coarse_tile` on the hint path, unchanged.
- **R1.5** Derived quantities come from the existing zero-mutation planner —
  `_planned_tile_extents_per_level` (`:309`) for the extents themselves, and
  `_tiled_dims_for_dep` (`:421`) to filter them per dep. No new extent arithmetic
  is written.
- **R1.6** Every returned option must be applicable *and* numerically correct.
  Applicability alone is insufficient: a test that applies each option via
  `_apply_plan` and asserts no `Unsupported` is raised would pass on the known
  wrong-numerics nested reduction shapes (*Background*), because their failure
  mode is a silent wrong answer, not an exception. The enumerator test therefore
  applies each option **and** compares against CPU, in the manner of
  `run_coarse_tile_test(..., correctness=True)`.
- **R1.7** Existing caps are the defaults, and stay where they are defined today
  in `wsr/span_overflow_hint_analysis.py` rather than migrating to `config.py`:
  `_MAX_TILE_DIMS = 3`, `_MAX_TILE_COMBOS = 512`, `_MAX_SPLITS_PER_DIM = 16`
  (`:143-145`), `_MAX_AUTO_TILE_SPLIT_COUNT = 64` (`:149`).
- **R1.8** **Reduction-axis tiling is single-level only.** `enumerate_tile_options`
  may emit a `TileOption` that divides a `reduction_ranges` entry, but only as a
  **single level** — one reduction dim, split once, with no other level (output
  or reduction) in the same option. Nested output+reduction and multi-level
  reduction shapes are **never** emitted: they are exactly the shapes
  `test_coarse_tile_e2e.py` marks `correctness=False` ("nested tiling + reduction
  correctness bug") or `@pytest.mark.skip` ("inconsistent loop_count across
  reduction fill/combine nodes"), and fixing that numerics bug is **out of
  scope** — this RFC never emits the illegal option (§5). A single-level
  reduction-tiled op enters the model **pinned as a singleton** (R4.6), and its
  predicted footprint includes the accumulator/fill/combine buffers (R7.1).
  Hint-driven reduction-axis tiling is unaffected — it still applies
  pre-stickification through `_maybe_coarse_tile_hints` — and those ops enter the
  model pinned the same way (R4.6, R5.6).
- **R1.9** **Candidate dims are not span-pressure-only.** `_candidate_host_dims`
  (`:911`) takes `list[SpanOverflowCandidate]`, so it surfaces only dims already
  under span pressure — which is correct for the span-overflow planner and wrong
  here. An op with no span overflow yields no candidate dims, hitting R1.1's
  "returns `None` when there are no candidate host dims at all" path. An
  enumerator built strictly on R1.4's list would therefore return **only the
  untiled option for exactly the ops where tiling-to-buy-LX-residency matters**,
  silently nullifying the second defect in *Motivation*.

  Candidate dims are instead the union of the span-pressure dims and every host
  dim passing `_host_dim_has_legal_nontrivial_split` (`:936`) — an existing
  helper, already built on `_split_candidates_for_host_dim`. Ordering stays
  pressure-first (`_candidate_host_dims`'s own ordering, then the remainder) —
  the R1.2 tiers put span-pressure dims ahead of speculative ones — so when the
  `max_options` cap binds it discards the speculative options before the
  pressure-relieving ones.

### R2 — Config construction and joint feasibility

- **R2.1** `PartitionConfig` pairs a `CoreDivision` with a `TileOption | None`
  and precomputes `per_core_bytes`, `cores_used`, `tile_count`.
- **R2.2** Divisions continue to come from `enumerate_work_division_candidates`
  (`work_division.py:753`) unchanged, including all five of its guards: at most
  one reduction dim split (`:812`), no coordinate-masked dim split (`:819`),
  TOPK left unsplit (`:776`), the core budget `prod(splits) <= max_cores`
  (`:810`), and a per-core span within `MAX_SPAN_BYTES` on every tensor dep
  (`:814-818`). It is called **once per tiling option**, against that option's
  divided iteration space, because its per-dim factors are `divisors(basis)` over
  `iteration_space_from_op` and its span guard is evaluated on the same space
  (§1). Each tiling option therefore carries its own division candidate set;
  neither set is a subset of the untiled one.
- **R2.3** **Span feasibility is evaluated on the pair, not per subsystem.** A
  config is feasible iff its per-core, per-tile span is within `MAX_SPAN_BYTES`
  (`work_division.py:73`, `65535 * 4096` ≈ 256 MiB). This generalizes the
  per-core-only check R2.2 already performs at `:814-818`, and discharges the
  `core_split_estimate = 1` TODO.

  Stating the over-tiling defect in config terms: tiles are **sequential** loop
  iterations and cores are **parallel**, but both draw down the same divisibility
  budget on a dimension. `T=4` tiles × `C=32` cores over `M=512` and `T=1` × `C=32`
  cut the span by different factors but only the second spends the whole budget on
  parallelism. Today the two decisions each spend that budget as if alone, which
  is why `MAX_SPAN_BYTES` ends up satisfied twice over. A joint feasibility check
  is what lets the objective spend it once.
- **R2.4** Configs are deduped by signature; there is **no fixed per-op cap** —
  model size is controlled by external pruning of the enumerated set (*Open
  questions*). The pair (work-division seed, untiled) is always retained, so the
  model's feasible set always contains today's answer.
- **R2.5** An op with no feasible config raises `Unsupported` at the same
  pipeline point it does today.
- **R2.6** **`config_matches` needs tiling-aware per-core views.** This is the
  one part of the `CoreDivision` → `PartitionConfig` migration that is not a
  rename, and it is on the critical path: `config_matches` gates residency
  through `constrain_residency`, so it cannot be deferred alongside the relayout
  work that shares its machinery (R6.3).

  `_views_for_divs` (`allocator.py:2079`) caches the sympy-heavy prep under
  `(op name, dep, buf_name)` on the explicit assumption that it is
  candidate-invariant — true when a candidate is only a core division, false
  once a candidate also carries a tiling. Three consequences:

  - The prep cache key gains the tile: `(op name, dep, buf_name, tile)`.
    Divisions of one op under different tilings must not share a prep.
  - `_prepare_per_core_view` (`pass_utils.py:1467`) and `_per_core_view_on_buf`
    (`:1696`) must accept a *predicted* post-tiling frame — divided ranges and
    the resized device layout — rather than reading the op's current layout.
    The solve runs before `coarse_tile` applies (§5), so at match-construction
    time no tiled IR exists to read.
  - That predicted frame is the same artefact R7.1 produces. R2.6 and R7.1 share
    one predictor; a divergence between them is a wrong-residency bug, not a
    mispredicted size, so it does **not** enjoy R7.4's degrade-to-spill safety.

  The existing conservatism is retained: a candidate whose slicing is
  unrepresentable is excluded from matching and the producer falls back to HBM.
  A tiling whose predicted frame cannot be built is excluded the same way — never
  pin on a slicing that cannot be verified.

### R3 — Injected cost function (the M2 deliverable)

R3 specifies the M2 objective namespace this phase delivers. It is not a separate
design decision from M2 — the collapse to a single minimized cost (R3.2) is M2's,
the per-config `AddElement` binding (R3.7) is M4's, and the fact that a pinned
axis contributes no objective terms is H4's. What R3 adds is the concrete
signature, grammar, and scaling rules for the tiling axis.

- **R3.1** Signature change. Today (`scratchpad/plan_solver.py:261`, `:298`)
  neither ABC is keyword-only, and only one takes `log_lx_usage`:

  ```python
  # today
  def plan_layout(
      self, buffers: Sequence[LifetimeBoundBuffer], log_lx_usage: bool = False
  ) -> list[LifetimeBoundBuffer]: ...

  def plan_layout_and_core_divisions(
      self, buffers: Sequence[CoreDivisionBuffer]
  ) -> list[CoreDivisionBuffer]: ...
  ```

  ```python
  # proposed — `objective` added, trailing arguments made keyword-only
  def plan_layout(
      self,
      buffers: Sequence[LifetimeBoundBuffer],
      *,
      objective: CostSpec | sympy.Expr | None = None,
      log_lx_usage: bool = False,
  ) -> list[LifetimeBoundBuffer]: ...

  def plan_layout_and_core_divisions(
      self,
      buffers: Sequence[CoreDivisionBuffer],
      *,
      objective: CostSpec | sympy.Expr | None = None,
  ) -> list[CoreDivisionBuffer]: ...
  ```

  Introducing `*` is source-compatible: every in-tree caller already passes
  `log_lx_usage` by keyword (`allocator.py:184`, `:1348`). The four concrete
  overrides move in lockstep — `ilp_solver_ortools.py:348`, `greedy_solver.py:134`,
  `firstfit_bestfit_solver.py:186`, `simulated_annealing.py:122`.

- **R3.2** The objective is a **single total expression minimized in one phase**
  (`Minimize(expr)`). There is no lexicographic sequence and no per-phase
  locking: today's two-phase lexicographic solve (`ilp_solver_ortools.py:479`) is
  **replaced, not generalized**. The hard guarantee that parallelism can never
  buy a spill becomes a weighting choice (R3.5) — a term whose scale must
  dominate another is expressed by its coefficient, subject to the
  `COST_SCALE`/overflow rules in R3.4.
- **R3.3** Supported sympy subset, stated exhaustively. Products of two model
  variables are expensive in CP-SAT (they must be reified), so they are limited
  now and may relax later:
  - `Add`; `Mul` with at most one non-constant factor per term (otherwise
    reified with `AddMultiplicationEquality`); `Pow` with a small non-negative
    integer exponent (expanded); `Integer` / `Rational` / `Float` coefficients.
  - `Min` / `Max` → `AddMinEquality` / `AddMaxEquality` over reified int vars.
  - `Piecewise` whose conditions are boolean model vars, reified via
    `OnlyEnforceIf`.
  - **Rejected**, with `CostExpressionError` naming the node: transcendentals
    (`log`, `exp`, `sqrt`), division by a variable, unbound free symbols,
    symbolic shapes.
- **R3.4** Rational and float coefficients are scaled to integers by a
  documented `COST_SCALE` (lcm of denominators, capped). Raise if the scaled
  coefficients would risk int64 overflow rather than silently wrapping.
- **R3.5** `objective=None` selects the default single-phase objective built from
  today's terms (`SumOverBuffers(spill_cost) - SumOverBuffers(cores)`), with the
  spill term weighted to dominate. Because the solve is single-phase rather than
  two-phase lexicographic, **exact bit-identity with today's plans is not
  required**; the guarantee is spill-parity (no plan spills a buffer today's
  objective would have kept resident) with no core-count regression at equal
  spill.
- **R3.6** The four placement-only solvers (`greedy`, `firstfit`, `bestfit`,
  `simulated_annealing`; registry at `allocator.py:2108-2113`) accept the
  parameter for ABC conformance, ignore a non-`None` objective, and log a warning
  once. The ABC docstring states this explicitly — the contract must not imply
  support these solvers lack. Note `LAYOUT_SOLVER` has a fifth value, `cpsat`,
  which is handled ahead of that registry (`allocator.py:2159`) and is the one
  solver for which `objective` is honoured.
- **R3.7** **Symbol binding is bounded.** Every symbol resolves to either an
  existing model variable or a single `AddElement` lookup over a per-config table
  computed at enumeration time. No symbol may add constraints scaling with
  anything but the buffer count and the adjacent-pair count (the edge count
  rejoins this list when R6.3 lands `relayout[e]`) — which is why `peak_lx_bytes` is the
  packing high-water mark over the existing `top[b]` vars rather than a
  time-indexed occupancy sum, and why the tile *shape* is not a symbol. A term
  needing shape sensitivity precomputes a per-config scalar instead. Adding a
  symbol that violates this is a design error, not a performance trade-off.

### R4 — Tiling groups

- **R4.1** Groups are **not** pre-computed. `cut[i]` booleans over adjacent pairs
  of `graph.operations` define them, fixed by `AddAllowedAssignments` over the
  per-edge `(tile_src, tile_dst, cut)` triple table, which admits `cut == 0`
  only on tiling-compatible config pairs.
- **R4.2** `cut[i]` is pinned to 1 at every boundary where either side is
  untileable. Contiguity is therefore structural and `_validate_contiguous`
  (`coarse_tile.py:785`) passes by construction — a maximal cut-free run of a
  list whose untileable positions are all cut is a contiguous slice of that
  list. A test asserts this directly rather than relying on the argument.
- **R4.3** `cut[i] == 0` is forced for every `i` interior to a hint scope,
  preserving `validate_coarse_tile_groups`'s invariant.
- **R4.4** `reorder_unhinted_interlopers` continues to run as a pre-step. The
  solver does not reorder operations. **Graph reordering is out of scope — it is
  the roadmap's Phase 4.**

  Execution order is the topological order of `graph.operations`, established at
  lowering and guaranteed by `GraphLowering` (`passes.py:404`). It is *a*
  topological order, not the only valid one, so an unhinted op sitting between
  two ops the solver would like to fuse is an artifact of that linearization
  rather than a semantic constraint — which is why `reorder_unhinted_interlopers`
  can relocate such ops on a dataflow legality check alone.

  The solver nonetheless takes the order as given, and every interloper the
  hint-driven pre-step did not move becomes a **forced cut**. That is a
  plan-quality ceiling, not a correctness problem, and a bounded one: by the cut
  table most forced cuts are the untiled → tiled row, which allocates no
  `full_buf` — though when the consumer's read advances it still costs the
  untiled producer its LX candidacy (*Background*, row 3). The expensive
  tiled → tiled row requires an interloper between two ops the solver actively
  wanted to fuse.

  Lifting this is harder than re-running the existing pass, which is why the
  roadmap makes it a separate phase (Phase 4) rather than a near-term follow-on
  here. Reordering sits at pipeline position 5, deliberately
  **before** stickification; the joint solve runs late — post-stickification
  (§5), after layouts are committed and after `optimize_restickify_locations`
  has chosen restickify sites against the current order — with the
  placement-only re-solve last at `_maybe_scratchpad_planning`. A solve → relocate →
  re-solve loop would move ops past decisions already made on the assumption they
  would not move. Hoisting the tiling decision earlier instead conflicts with
  `_maybe_coarse_tile_span_overflow` being post-stickification precisely because
  it needs `device_layout` for span arithmetic.

  The fixed order is also what makes the cut claims static dicts (§2):
  program-order adjacency is static, so each buffer's neighbour topology — and
  each spanned edge's parent/child orientation — is fully known when its
  wrapper is constructed.
- **R4.5** The solver emits **two** artefacts, mirroring `span_overflow_groups`
  exactly, because `coarse_tile` alone is not enough:

  - `groups` in the documented `(ops, levels)` shape, passed with the existing
    `group_idx_offset` parameter so emitted `loop_group_id`s do not collide with
    those the hint pass already stamped; and
  - `dim_hint_assignments` — `(op, list[DimHint])` pairs built by
    `_dims_to_hints` (`coarse_tile_span_overflow.py:152`) from each op's
    `TileOption.dims` and the hint ids minted for its run.

  **Two namespaces have to stay disjoint, not one.** `group_idx_offset` handles
  `loop_group_id`; `hint_id` needs the same treatment separately, and does not
  get it for free. The span-overflow path mints from a reserved base —
  `_SPAN_OVERFLOW_HINT_ID = 10000` (`coarse_tile_span_overflow.py:45`),
  incremented one block per closed group — precisely so its ids cannot collide
  with the user `spyre_hint` ids `assign_dim_hints` stamps. The solver is a
  **third** source and needs its own reserved base above both. Without one,
  `validate_coarse_tile_groups` (`coarse_tile.py:112`) sees a single `hint_id`
  in two groups and raises — and it raises during apply, after the solve has
  succeeded, so the failure surfaces as an illegal emission (§5) rather than as
  anything the model could have ruled out.

  `op.dim_hints` is an **input** to `plan_coarse_tile_groups`, not an output of
  it: the hint lookups that build `hint_id_to_ranges_pos` read it. The
  span-overflow path assigns them explicitly before calling `coarse_tile`
  (`passes.py:357-365`, "a pure planning step: it decides each op's dim_hints but
  does not set them"). The apply step does the same, in the same order, and
  derives `group_idx_offset` from existing `loop_group_id[0]` values the same way.
  No new application path is introduced.
- **R4.6** `cut[i]` is pinned to 1 on **both** boundaries of any op tiled on a
  reduction axis — hint-driven or solver-chosen (R1.8) — exactly as for an
  untileable op (R4.2). Such an op is therefore always a singleton group and
  never shares a loop nest with a neighbour.

  This is not merely conservative — it is what keeps the pairwise cut table
  sound. The invariant governing a reduction-tiled group is
  `_plan_is_loop_invariant_at_reduction_levels` (`coarse_tile.py:559`): at every
  level where *some* member tiles a reduction dim, *every* other member must be
  loop-invariant at that level. That quantifies over the whole group, and
  adjacent-pair compatibility does not compose into it. Counterexample: `A` tiles
  a reduction dim at level `L`, `B` is loop-invariant at `L`, `C` tiles an output
  dim at `L`. Pair `(A,B)` is legal and pair `(B,C)` is legal — `B`'s invariance
  says nothing about `C` — yet the run `{A,B,C}` violates the invariant. Pairwise
  tables can express "these two agree"; they cannot express a predicate
  quantified over a run whose membership they are simultaneously deciding.
  Lifting R4.6 needs run-identity in the model (per-run, per-level literals), not
  a wider triple table.
- **R4.7** "Tiling-compatible" in R4.1 is a **pairwise predicate evaluated at
  table-construction time**, not equality of a per-op key. For an ordered adjacent
  pair, `cut == 0` is admitted iff:

  1. the consumer can adopt the producer's split —
     `can_conform_pointwise_tile(op, split_by_host_dim, config.sencores)`
     (`span_overflow_hint_analysis.py:1421`), which checks divisibility, stick
     boundaries, and sufficiency; **and**
  2. the loop variables correspond through the dep — the symbol tiling the
     consumer's dim must appear in the producer's tiled coordinate as seen
     through the read, the check `_reduction_shares_group_tiled_dim`
     (`coarse_tile_span_overflow.py:68`) performs. Matching split counts alone is
     necessary but not sufficient: two unrelated dims can split into the same
     count.

  Both are reused, not reimplemented (R1.4). Because
  `can_conform_pointwise_tile` is **directional**, the predicate is evaluated in
  program order — producer then consumer — matching the direction
  `span_overflow_groups` already conforms in. Any pair for which correspondence
  cannot be established fails closed to `cut == 1`, preserving the existing
  conservatism: an unverifiable pair is never fused into a possibly-desynchronized
  loop.
- **R4.8** What a cut materializes is a **model variable**, not a post-hoc
  property. Per buffer the model carries `cut_parents[b]`/`cut_children[b]`
  (the direction-indexed dicts over the cut vars on the edges `b` spans, §2),
  `tiled[b]`, `escapes[b]`, and
  `boundary_op[b] ⟺ tiled[b] ∧ escapes[b]` (§2). The solve precedes IR mutation
  (R7.1), so `full_buf` and the copy op do not exist yet and their own LX
  rejections are not available to the model; what it can see is the producer's
  own output and the buffers a cut would add.

  A cut does **not** evict its *tiled* producer. `_propagate_tiled_op` sets
  `output_tiled_dims = []` on both branches (`coarse_tile.py:1367`), so `b` stays
  loop-internal tile-sized scratch and remains LX-eligible whether or not it
  escapes. There is no `boundary_op[b] ⟹ in_buffer[b] == 0` constraint, and any
  model that adds one is wrong. The one eviction the model does carry is the
  row-3 rule (§2): a read copy whose advancing read crosses an untiled → tiled
  edge enforces `in_buffer[producer] == 0`, because the realized copy stamps
  that producer `"tiled (advancing)"` (*Background*).

  `tiled[b]` is `tile[b] != 0` — the candidate space is two-level (tiling, then
  that tiling's divisions), so being tiled is read off the variable rather than
  looked up. It is required in the conjunction, not optional: an untiled → tiled
  cut materializes no `full_buf` at all (R4.2's forced cuts are all of this
  form), so `escapes[b]` alone would claim a boundary buffer that never gets
  allocated.

  These vars live on a `_TilingBufferWithCpVars` wrapper extending
  `_CoreDivisionBufferWithCpVars`. They are **not** kept in a parallel
  `name -> {var}` dict — avoiding exactly that is why the wrapper hierarchy
  exists (`ilp_solver_ortools.py:150`).
- **R4.9** The read-side tile copies are **optional rectangles**, not
  unconditional ones. `_insert_read_copy_ops` creates a tile-sized LX-eligible
  buffer per full-extent input of a tiled op, and whether it exists is a solver
  decision: conditional on `escapes[producer]` for a cross-group read, on
  `tiled[consumer]` for a graph input, constant, or untiled producer. Presence in
  `_add_no_overlap_2d` is that literal; `in_buffer` gates residency on top of it.
  Modelling them as always-present over-reserves LX on cut-free runs; omitting
  them undervalues tiling, since they are precisely the buffers tiling makes
  pinnable.
- **R4.10** The boundary buffer's **shape and core division** are model variables
  too — `full_size[b]` and `boundary_view[b]`, bound by `AddElement` over
  per-config tables like `eff_size[b]`/`cores[b]`. `full_size[b]` is not
  `eff_size[b] × tile_count[b]`; the full buffer is stickified once at the full
  host extent, so per-tile stick padding does not survive the scale-up.
  `boundary_view[b]` exists so producer/consumer agreement at the boundary is
  checked on the physical per-core view (R2.6), not on a slice count two
  orthogonal divisions can share. Both are inputs the objective consumes; what
  they are **worth** is track C1's concern, not this document's.

### R5 — Hints (registered under the H-spine)

Tiling and division hints are **pins** in the roadmap's M3 sense, handled through
the H1–H5 spine that collateral document 0 owns. This RFC does not build the
registry or the validator — it **registers its keys and rules against them**. In
particular, a hinted axis contributes no variables and no search (H4): pinning is
an enumeration-time domain restriction (H2), not a post-solve override.

- **R5.1** The tiling keys (`tiles` / `slices` / `num_tiles_per_dim`) and the
  `work_div` key register under **H1** with their value schema and scope. Manual
  `spyre_hint` tiling applies pre-stickification exactly as today and remains
  authoritative; the affected op enters the model as a **pinned single-config
  buffer** — the H2 lowering of a hint to a pin, leaving exactly one option in the
  op's candidate list.
- **R5.2** The solver never re-tiles or un-tiles a hinted op — a direct
  consequence of the H2 pin, not a separate rule the solver enforces.
- **R5.3** Where no hint is present, the axis is *optimized* (M3): the solver
  tiles automatically.
- **R5.4** `SPYRE_INDUCTOR_IGNORE_HINTS=1` corresponds to the roadmap's
  `SPYRE_INDUCTOR_IGNORE_HINTS` behaviour — it drops the pins, handing those ops
  to the solver as ordinary un-hinted ops.
- **R5.5** *Deferred (a follow-on to this phase, not a roadmap phase).* Growing an
  existing hint group with solver-chosen neighbours requires either moving hint
  application post-stickification or emitting nested groups with a matching
  `loop_group_id` prefix. Out of scope here; recorded so the limitation is
  understood, not discovered.
- **R5.6** A hint that tiles a reduction axis still applies, and
  `enable_reduction_tiling` (`config.py:82`) keeps its current default and
  meaning. The affected op enters the model as a pinned single-config buffer
  (R5.1) with both its `cut[i]` boundaries pinned to 1 (R4.6). Setting
  `SPYRE_INDUCTOR_ENABLE_REDUCTION_TILING=0` makes such a hint raise
  `Unsupported`, unchanged by this RFC.
- **R5.7** Validation is **H3's**, not a new mechanism: a malformed key is named
  before enumeration (level 1), a hinted value that survives no config is named
  during enumeration (level 2), and a set of individually-realizable pins that is
  jointly infeasible across ops routes to the post-`INFEASIBLE` diagnostic (level
  3). The one axis-specific obligation is that the tiling predicates (R1.4) are
  the *same* ones the model constrains against, satisfying M6 for the fallback
  path. Every committed tiling/division decision records its source under **H5**
  (`decision_reason`).
- **R5.8** A global **untiled default** needs no per-op hinting: with
  `AUTO_COARSE_TILING` off (R8.1) every un-hinted op is pinned to the unity
  tiling (`tile=None`), so the tiling optimization can be turned off while manual
  tiling hints still apply and the solver still co-optimizes division and
  residency. This hints-honoured / rest-defaulted mode is distinct from
  `SPYRE_INDUCTOR_IGNORE_HINTS` (R5.4), which instead *drops* the pins.

### R6 — Stickification / relayout: Phase 3, not here

Choosing configs to *avoid* a restickify — a "stickification optimization" that
models each producer→consumer edge's relayout cost and lets the objective trade
it off — is the roadmap's **Phase 3** (collateral document 3), not this phase
(R9). The `relayout[e]` variable, its per-edge triple table, and the
`relayout_bytes` objective term are all deferred there. The exception, called out
in R6.3, is the tiling-aware per-core views this phase *must* deliver because
Phase 3 depends on them.

- **R6.1** The compiler keeps inserting restickifies wherever configs force one,
  exactly as today (`insert_restickify`, `passes.py:438`). The solver neither
  models nor minimizes that cost.
- **R6.2** *Implication — accepted pessimism.* Because relayout cost is invisible
  to the objective, the solver may pick a (tiling, division) config whose
  physical per-core view disagrees with a neighbour's and thereby force a
  restickify a relayout-aware model would have avoided. Those bytes are real but
  unpriced, so plans can be **pessimistic on relayout-driven HBM traffic**. This
  never makes a plan *infeasible* — a restickify can always be inserted — it only
  means the model cannot prefer the cheaper-to-stickify config. In the worst case
  the joint solve is no better than today on relayout, and possibly worse, since
  tiling introduces new per-core views that today's un-tiled graph never had.
- **R6.3** Lifting this is follow-on work, but only the *pricing* half is
  deferred. The two halves must not be bundled:

  - **Lands in this phase (R2.6)** — this is exactly the "tiling-aware per-core
    views" the roadmap lists under Phase 1's *"Owed to later phases"*:
    `_prepare_per_core_view` (`pass_utils.py:1467`) and `_per_core_view_on_buf`
    (`:1696`) evaluated against a predicted post-tiling frame. `config_matches`
    depends on this to gate residency, so it is not optional and cannot wait for
    Phase 3.
  - **Deferred to Phase 3:** `relayout[e]` as a *determined*, cost-only edge
    variable, precomputed once per edge at enumeration time from those same views,
    plus a `relayout_bytes = SumOverEdges(relayout[e] * bytes[e])` term in the
    objective namespace, inside R3.7's linear-binding rule.

  An earlier draft filed the view extension under the deferred half. That would
  have left the non-deferred `config_matches` path depending on a mechanism this
  phase never builds; Phase 3's work is then a variable and a cost term over views
  that already exist by then.

### R7 — Prediction fidelity and application (M7, M8)

R7 instantiates M7 (pure prediction, then apply) and M8 (addresses from the final
solve) for the tiling axis. One alignment point matters: per M7 the predictor's
fidelity is settled **offline against recorded plans**, not by an assertion pass
inside the compiler — "no phase owes a verify mode, and none adds one." So what
this phase owes is that the *inputs* to that offline check exist (R7.2), recorded
under H5; it does not add an in-compiler verify flag. This concerns **fidelity**
— the *accuracy* of predicted sizes, lifetimes, and views — not **legality**:
that an applied plan is well-formed is guaranteed by construction and backstopped
by §5's post-apply assertion, whose firing is a hard failure (§5). What M7 defers
offline is the accuracy check, never the legality guard.

- **R7.1** A pure predictor maps a candidate config set to the predicted buffer
  set (sizes, lifetimes, boundary copies) with **no IR mutation**. R1.8 bounds
  what it has to model: output-range tiling materializes only the boundary copies
  of `_allocate_full_buffer`/`_insert_copy_op`/`_insert_read_copy_ops`, and
  **single-level** reduction-axis tiling adds exactly the accumulator, identity
  fill, and combine op of `_propagate_tiled_reduction_op` — no nested second
  accumulator, since R1.8 emits no nested option. The predictor models those three
  for a solver-chosen reduction option, or the objective would read reduction
  tiling as free LX relief (*Background* property 3). Hint-driven reduction tiling
  applies pre-stickification (`passes.py:430`), so by the time the solve runs its
  buffers are already real IR the model simply sees.

  The predictor's second output is the **post-tiling frame** — divided ranges
  plus the resized device layout — that R2.6 evaluates per-core views against.
  One predictor serves both; they must not drift apart.
- **R7.2** Each decision **records** the predicted values it was scored against —
  per-buffer size and lifetime, and the predicted per-core view (R2.6) — alongside
  the decision itself under H5. The roadmap's offline fidelity check reads those
  records and compares them to the realized post-tiling IR after the fact, under
  the rank-order normalization M7 mandates (R7.5); this phase adds no
  in-compiler verify pass. The **view** record is the load-bearing one: a
  mispredicted size degrades to a spill under R7.4, whereas a mispredicted view
  means residency was gated on a slicing agreement that does not hold — a
  wrong-data bug, and the highest-risk area in the design. The offline check
  therefore treats the view comparison as its primary assertion and gets its own
  recorded-plan fixtures.
- **R7.3** Application order is: decide → `coarse_tile` → commit divisions →
  placement-only re-solve. Addresses always come from the final solve over real
  buffers.
- **R7.4** The placement re-solve is warm-started from the joint solve's
  residency intent. A buffer that no longer fits degrades to a spill with a
  distinct per-buffer `residency_reason` (`plan_solver.py:68`) — which surfaces
  through the solver-level `spill_reasons` map (`plan_solver.py:219`) and the
  allocator's `reject_reasons` mirror (`allocator.py:137`) — so the mispredict is
  visible rather than silent.
- **R7.5** The predictor models **inserted operations and their positions**, not
  just the buffers those operations allocate. `coarse_tile` splices the full
  buffer in before the group's first op, the write copy after the tiled op, and
  read copies before their consumer (*Background*). Liveness is index-based
  (`calculate_liveness` over `graph.operations`), so in realized IR each
  insertion shifts lifetime ticks for everything downstream — a systematic
  offset, not noise. The model sidesteps the renumbering by placing predicted
  insertions at §2's interstitial coordinates on a scaled tick axis, so every
  real buffer's lifetime stays stable in the model; the realized offsets appear
  only after apply, are recomputed wholesale by `calculate_liveness`, and are
  compared against the prediction under rank-order normalization (R7.2).

  It must equally model the LX candidates tiling **creates**: the interior
  per-tile scratch (whose `output_tiled_dims` becomes `[]`, making it eligible)
  and the read-side tile copies (ordinary tile-sized allocations, also eligible).
  These do not exist until `coarse_tile` runs, so omitting them leaves the final
  placement re-solve holding buffers the joint objective never scored. The bias
  has a known direction — the model would **undervalue tiling**, since the
  buffers it fails to see are exactly the ones tiling exists to make pinnable —
  so this is not a wash that averages out across a graph.

### R8 — Robustness, gating, determinism (track C2; fallback is M9)

R8 is this axis's instance of track C2 (determinism, tractability, fallback) and
of M9 (failure never discards a pin). The fallback path (R8.3) is M9 case 1 — a
timed-out or unavailable solver degrades to the pinned candidate sets — and the
determinism rules (R8.4) are C2's `num_search_workers = 1` /
`random_seed = 0` regime. Per C2, a timeout is unattributable across axes, so the
fallback degrades *all* axes at once; this phase does not assume its own is the
one that keeps its solved value.

- **R8.1** New gate `UNIFIED_TILING` / `config.unified_tiling`, **default off**.
  The bare `UPPER_SNAKE` form matches the LX-planning family this gate composes
  with (`LX_PLANNING`, `CO_OPTIMIZING_LX_PLANNING`, `LAYOUT_SOLVER`;
  `config.py:22-25`, `:111`), while the diagnostic flags below keep the newer
  `SPYRE_`-prefixed style of `SPYRE_INDUCTOR_*` — the split is deliberate, not
  accidental. Requires `LAYOUT_SOLVER=cpsat` and `CO_OPTIMIZING_LX_PLANNING=1`,
  the latter itself default-off today (`config.py:23-25`); warn and no-op
  otherwise.

  A second gate `AUTO_COARSE_TILING` / `config.auto_coarse_tiling`, **default
  off**, governs whether the solver introduces tiling on **un-hinted** ops — the
  R1.9 residency-driven candidates and R5.3's optimized axis. With
  `UNIFIED_TILING` on but `AUTO_COARSE_TILING` off, the joint model still
  co-optimizes core division and residency and still honours tiling hints (R5.1),
  but every un-hinted op stays `tile=None`; this is the safe-rollout state, whose
  plans differ from today only in the M2 objective collapse, not in any new
  tiling. Turning it on admits the enumerator's non-span-pressure options (R1.9).
  Span-forced tiling a graph *requires* for feasibility is not gated by it — that
  path remains the retained span-overflow tiler (§5, R8.3).
- **R8.2** Warm-start the model via `AddHint` with the current heuristic's plan,
  so a timed-out solve keeps an incumbent no worse than today's answer rather
  than dropping to the fallback (R8.3).
- **R8.3** The solve has three outcomes, not two. **A feasible solution —
  `OPTIMAL`, or a `FEASIBLE` incumbent when the deterministic time budget (R8.4)
  is spent — is applied as-is.** The model only ever emits legal plans (§5), so
  an incumbent needs no further vetting; a timeout is a *quality* limit, not a
  failure. **Only `INFEASIBLE`, or a timeout with no incumbent, raises
  `SolveError`** (`plan_solver.py:27`), caught by a **new handler at the
  `unified_partition_solve` slot** — the existing try/except at
  `allocator.py:2211` wraps only `_maybe_scratchpad_planning` (pass 455) and
  never sees this pass. On that raise the joint plan is discarded whole and the
  pipeline reverts to the existing tiling method with the greedy solver:
  `_maybe_coarse_tile_span_overflow` runs exactly as today (retained, §5), the
  heuristic division passes at 451-452 proceed unchanged, and placement at
  pass 455 drops straight to placement-only greedy — `allocator.py:2211`'s
  fallback path, entered directly rather than after a second `SolveError`, since
  a solve that could not even find a feasible point makes another CP-SAT attempt
  a poor bet. The graph is unmutated until a plan is applied — the solve and its
  outcome check precede `coarse_tile` — so the raise starts the fallback from
  clean IR, and the span-forced tiler still rescues the graphs that need it. An
  *illegal* emission is a separate matter entirely: a hard failure, not a
  fallback (§5).
- **R8.4** Determinism means **a given model yields an identical plan across
  runs**, not that similar graphs yield similar plans. Keep
  `num_search_workers = 1` under `torch.are_deterministic_algorithms_enabled()`,
  `random_seed = 0`, and a fixed signature-defined enumeration order so
  equal-cost ties resolve the same way every run. **Plan stability under input
  perturbation is an explicit non-goal** — by construction a changed graph is not
  expected to produce a similar plan — so no symmetry-breaking beyond seed and
  enumeration order is added. One caveat: a solve that stops early and keeps a
  feasible solution (R8.3) is reproducible only under a **deterministic** stop
  criterion (`max_deterministic_time`, not wall-clock); otherwise the accepted
  solution varies across machines.
- **R8.5** `ortools` remains the optional extra named `cpsat`
  (`pyproject.toml:36-38`, `ortools>=9.0`); the import stays guarded
  (`ilp_solver_ortools.py:86-93`) and `_make_cpsat_solver` (`allocator.py:2116`)
  keeps translating a missing `ortools` into the greedy fallback. `sympy` needs no
  new dependency: it is not declared directly but arrives transitively via torch
  (`sympy>=1.13.3`), and Inductor already depends on it.

### R9 — Non-goals

Most of these are not "never" — they are *other phases and tracks* of the
roadmap. They are listed as non-goals of **this** document so the scope boundary
is explicit, with the owner named.

- **No ring transfers.** The `core_div_mismatch` hard wall stays. Dissolving it
  needs a data ring or reduce-sum ring emitted in the SuperDSC schedule, which
  is separate work outside the roadmap.
- **No cost-model calibration in microseconds.** This phase delivers the M2
  *mechanism* (§4); tuning the objective to measured µs is the roadmap's **track
  C1**, not this document.
- **No operation reordering** beyond the existing `reorder_unhinted_interlopers`.
  Order as a decision is the roadmap's **Phase 4** (R4.4).
- **No time-varying addresses / defragmentation.** LX relocation and compaction
  are the roadmap's **Phase 2**; this phase keeps today's single-address
  placement and co-optimizes residency against it.
- **No change** to `_matmul_split_cost` in `work_division.py`.
- **No stickification / relayout optimization.** The solver does not model or
  minimize restickify cost; that is the roadmap's **Phase 3** (R6). Configs are
  chosen blind to relayout, which can be pessimistic on relayout-driven HBM
  traffic — the tiling-aware per-core views Phase 3 needs are the one piece that
  lands here (R2.6).
- **No padding as layout legalization.** The sizing half of padding lands here
  (R10); using padding to remove the issue #1756 layout-search restriction is
  **Phase 3**.
- **No *nested* or *fused* reduction-axis tiling.** The solver may choose
  single-level reduction tiling (R1.8), but never a nested output+reduction or
  multi-level shape, and a reduction-tiled op is always a pinned singleton, never
  fused into a group (R4.6). Two boundaries stay out of scope here: the nested
  wrong-numerics bug this RFC does not fix (so those options are never emitted);
  and fusing a reduction-tiled op with neighbours, which the pairwise cut model
  cannot express (R4.6) and which needs per-run/per-level literals. The capability
  itself is untouched — `enable_reduction_tiling` keeps its default and hints keep
  working. Follow-on work is described under *Open questions*.

### R10 — Padding (not an axis, §7)

Padding lands in this phase for **sizing** (roadmap consequence 3) and adds no
decision axis. It is entirely per-config table content (M4).

- **R10.1** The legality pad is **derived**, not decided: `compute_padding`
  (`padding.py:73`) rounds a dimension to one stick, so given a config's layout
  and tiling there is exactly one value. It is a scalar on the candidate row, and
  a config's predicted buffer sizes are computed from the padded `device_size`.
  **No new index space** — the M4 `f(config) -> int` case.
- **R10.2** Padding *beyond* legality enters as **additional config rows**, never
  a new variable. A discretionary pad can satisfy a divisibility `valid_split`
  requires (`work_division.py:809-823`), turning an illegal core split into a
  legal one and so widening the feasible config set; the extra configs are ranked
  by the same objective.
- **R10.3** Whether R10.2's discretionary rows are enumerated at all is **gated on
  a measurement** — an unaligned-`K` matmul against the same model pre-padded by
  hand, accounting for `lower_pad_sequence`'s four-op cost (`pass_utils.py:1191`),
  the y-operand buffer growth, and the `K → K_padded` iteration-space widening
  (`_extend_matmul_k_to_padded`, `codegen/superdsc.py:870`). If the delta is
  noise, the derived pad (R10.1) stays purely in the apply step and no pad row is
  added.
- **R10.4** This phase lifts `padding.py`'s fixed policies, since the derived
  amount cannot be expressed without them: pad operands other than y, dimensions
  other than K, ends other than the right, and multiples other than one stick;
  and share one padded buffer between matmuls that read the same operand rather
  than emitting a pad sequence per matmul (`padding.py:183`).
- **R10.5** A pad-amount pin registers as a hint key under **H1**, lowers to a pin
  under **H2** (it collapses the pad scalar / selects among R10.2 rows), and is
  validated by **H3** exactly as the tiling keys are (R5.7). A pad pin that makes
  an op's config set empty is named at H3 level 2.
- **R10.6** Padding as a **layout-legalization** tool — removing the issue #1756
  restriction at `propagate_layouts.py:271`, `:455`, `:1078` — is **not** in this
  phase; it is Phase 3, with the layout search that consumes it (R9).

## Files

**New**

- `torch_spyre/_inductor/scratchpad/cost_expr.py` — the **M2 objective namespace**
  this phase delivers: symbol namespace, `CostSpec`, sympy→CP-SAT lowering,
  `CostExpressionError`.
- `torch_spyre/_inductor/wsr/enumerate_tilings.py` — `enumerate_tile_options`,
  built on the R1.4 predicates; output ranges plus single-level reduction (R1.8).

**Modified**

- `scratchpad/plan_solver.py` — `TileOption` (op-local `dims`, §1),
  `PartitionConfig`; `CoreDivision` retained as a config field; the R3.1
  signatures.
- `scratchpad/ilp_solver_ortools.py` — new `_TilingBufferWithCpVars` subclass of
  `_CoreDivisionBufferWithCpVars` (`:244`) carrying the two-level `tile`/`div`
  pair in place of `division`, plus the boundary vars (R4.8–R4.10); per-buffer
  direction-indexed cut-claim dicts (`cut_parents`/`cut_children`) reconciled
  by an `_add_cut_equalities` sweep in `_run`; read
  copies as optional rectangles in `_add_no_overlap_2d` (`:568`); relayout
  deferred (R6); single-phase objective driven by `CostSpec`; `_extract` writes
  `chosen_config` and reconstructs `groups` from the solved cuts.
- `scratchpad/allocator.py` — `_enumerate_core_divisions` (`:1558`) becomes config
  enumeration; `_cd_parent_matches` (`:1973`) becomes `_config_matches`, on
  tiling-aware views (R2.6, **not** a rename); `_views_for_divs` (`:2079`) takes
  the predicted frame and its `prep_cache` key gains the tile;
  `_commit_divisions` (`:1605`) also emits `groups` **and**
  `dim_hint_assignments` for `coarse_tile` (R4.5).
- `pass_utils.py` — `_prepare_per_core_view` (`:1467`) and `_per_core_view_on_buf`
  (`:1696`) accept a predicted post-tiling frame instead of reading the op's
  current ranges and device layout (R2.6); `lower_pad_sequence` (`:1191`) gains
  shared-operand emission so two matmuls reading the same operand share one padded
  buffer (R10.4).
- `padding.py` — `compute_padding` (`:73`) exposed as the per-config derived-pad
  scalar (R10.1); the fixed policies in `insert_bmm_padding` (`:163`) lifted so pad
  operand/dimension/end/multiple follow the chosen config (R10.4). The
  layout-legalization half (#1756) is **not** touched here — that is Phase 3
  (R10.6).
- `passes.py` — insert `unified_partition_solve` (with its R8.3 `SolveError`
  handler) and the apply step; skip `_maybe_coarse_tile_span_overflow` when
  the solve succeeds — retained verbatim as the R8.3 fallback tiler, not
  deleted.
- `config.py` — `unified_tiling`, `auto_coarse_tiling` (R8.1) (and
  `enable_discretionary_pad`, gated by the R10.3 measurement). No `verify_tile_prediction` flag: per M7 the
  fidelity check is offline, so the phase records predicted values (R7.2, under
  H5) rather than adding an in-compiler verify pass.
- `wsr/span_overflow_hint_analysis.py` — expose the predicates as reusable
  helpers; `_search_min_cost_tile_plan` becomes a thin ranked wrapper over the
  enumerator.

**Docs updated on landing**

`docs/source/compiler/work_division_planning.md`,
`docs/source/compiler/scratchpad_planning.md` (the "no coarse-tiling
integration" gap), `docs/source/compiler/coarse_tiling_loops.md`, and
`docs/source/rfcs/index.md` (row plus summary).

## Testing and verification

1. **Parity, gate off.** `tests/inductor/test_scratchpad_solver.py` (CP-SAT
   coverage lives in `JointDivisionSolverTests:776` and
   `TestCpSatPlacementOnly:1121`), `test_scratchpad_use.py`, `test_coarse_tiling.py`,
   `test_coarse_tile_e2e.py`, `test_span_overflow_hint_analysis.py` all pass
   unchanged. All five carry a CI config yaml under
   `tests/configs/torch_spyre_tests/inductor/` with
   `unlisted_test_mode: mandatory_success`, so a test added to any of them must
   be green to land — an expected failure satisfies that, an unexpected pass
   does not.
2. **Parity, gate on with tiling disabled.** Spill outcomes must match today's
   CP-SAT output and core counts must not regress at equal spill. Exact
   bit-identity is **not** required, because the objective is now single-phase
   (R3.2, R3.5); this is the regression guard for the `CoreDivision` →
   `PartitionConfig` migration.
3. **Cost lowering.** Unit tests for the R3.3 accept/reject table and R3.4
   scaling (single-phase `Minimize` of one total expression; no per-phase
   locking), with a `CostExpressionError` case per rejected construct. Plus R3.7:
   every namespace symbol adds at most one `AddElement` (or the single
   `AddMaxEquality` for `peak_lx_bytes`), and model size grows linearly in buffer
   count and adjacent-pair count as the graph scales. Also assert `SumOverEdges`
   and `relayout_bytes` are **absent** from the exported namespace (§4) — a
   reserved name that silently resolves would let an objective reference a term
   the model never constrains.
4. **Cut tables and structural contiguity.** The `cut[i]` triple table is total —
   every `(tile_src, tile_dst)` pair appears exactly once — and `cut[i]` is
   pinned to 1 at every untileable boundary and on **both** boundaries of every
   reduction-axis-tiled op, hint- or solver-chosen (R4.6), so such an op is always
   a singleton group. Then the property that
   §3 rests on: for any solution, every maximal cut-free run is a contiguous
   slice of `graph.operations` (R4.2), asserted directly rather than argued.

   For R4.7, assert the admitting predicate is evaluated **directionally**: a
   pair whose consumer can adopt the producer's split but not the reverse admits
   `cut == 0`, and a pair where loop-variable correspondence cannot be
   established fails closed to `cut == 1`. Assert the claim indexing carries the
   same orientation: for every wrapper, each spanned edge's `cut_parents` entry
   and `cut_children` entry resolve to the same bool after the equality sweep,
   and the edge's triple table takes `tile_src` from the edge's parent op and
   `tile_dst` from its child. For R4.5, assert the emitted
   `(groups, dim_hint_assignments)` pair round-trips — applying the hints then
   calling `coarse_tile` reproduces exactly the grouping the solver chose, with
   no `loop_group_id` collision against hint-pass groups.
5. **Enumerator completeness and scope.** Brute-force reference on small shapes;
   the enumerator's set must equal the reference's. Every reduction option is
   **single-level** — one reduction dim, no other level — and **no** nested
   output+reduction or multi-level reduction option is emitted (R1.8), asserted
   over `test_coarse_tile_e2e.py`'s Group 4 and Group 5 shapes, whose nested
   variants are the `correctness=False` / skipped ones. Every option applies
   **and** matches CPU numerically (R1.6), not merely applies without
   `Unsupported` — this is what keeps the admitted single-level reduction options
   legal.

   The R1.9 guard needs its own case, because it is the one that fails silently:
   for an op under **no** span pressure but with a legally splittable host dim,
   assert the enumerator returns more than the untiled option. Built on
   `_candidate_host_dims` alone this returns a single option and the solver simply
   never tiles that op — no error, no warning, and the LX-residency motivation
   quietly does nothing.
6. **Prediction fidelity (offline, M7).** The coarse-tiling and scratchpad suites
   emit the prediction records R7.2 requires (predicted per-buffer sizes,
   lifetimes, and per-core views, recorded under H5). An **offline** check —
   against recorded plans, not an in-compiler verify flag — compares each record
   to the realized post-tiling IR under the rank-order normalization of §2's
   interstitial coordinates. Per M7 no verify mode ships in the compiler; the test
   asserts the records exist and the offline comparison holds.
7. **Boundary buffer LX status.** For a graph tiled into two groups, assert the
   `full_buf` that `_allocate_full_buffer` produces has a non-`None`
   `residency_reason` (`"mutation target"` or `"tiled (advancing)"`), that the
   copy op's own output is likewise rejected, and — the positive half — that the
   interior per-tile scratch and the read-side tile copies *are* LX candidates.
   This is the table in *Background* asserted rather than argued. Assert too that
   the producer **keeps** a `None` `residency_reason` across a cut: both branches
   of `_propagate_tiled_op` set `output_tiled_dims = []`, so a cut must not evict
   it, and a regression here would resurrect the eviction constraint R4.8 rules
   out. Also assert the predicted lifetimes agree with the realized ones after
   `coarse_tile` has inserted its ops, under the rank-order normalization of
   §2's interstitial coordinates (R7.5) — equality for buffers no cut touches,
   containment (predicted ⊇ realized) for cut producers, whose model rectangles
   deliberately keep the pre-mutation extent (§2) — since insertion renumbers
   every downstream tick in the realized IR.

   Then the **model side**, which is the half that can go wrong silently
   (R4.8–R4.10). Assert the 1 / 2 / 0 rule directly: a cut-free run yields one
   LX-eligible tile-sized buffer for `b`, a cut yields two (`b` plus the
   consuming group's read copy), and `in_buffer[b] == 0` yields none. Assert the
   read copies are **optional** rectangles (R4.9) — that a cut-free solution
   reserves no space for a read copy that will not be created, and that a
   solution cutting an untiled producer still creates one in the consumer.
   Assert `full_size[b]` equals the realized `full_buf` footprint and
   `boundary_view[b]` the realized per-core view, both after `coarse_tile` runs;
   drift there feeds the cost model wrong numbers while the solve still reports
   optimal.

   Then the untiled→tiled case, which is the cut-cost table's most surprising
   row and the one most forced cuts land on: assert **no** `full_buf` is
   allocated and no `MutationLayoutSHOULDREMOVE` op appears. For the producer,
   assert both halves of the row-3 rule (§2): with an advancing consumer read it
   is stamped `"tiled (advancing)"` — evicted, and the model's
   `in_buffer[producer] == 0` implication agrees — while with a read invariant
   along every tiled dim it keeps a `None` `residency_reason` and stays a
   candidate. Left unasserted, this row is the one a future change to the
   propagation loop would silently break.
8. **Per-core view prediction (R2.6).** For every config of every op in the
   coarse-tiling suites, assert the predicted per-core view equals the one
   recomputed by `_prepare_per_core_view` after `coarse_tile` has actually run.
   Separately, assert `_views_for_divs`'s `prep_cache` never returns a prep built
   under a different tiling — the stale-prep failure is silent, and it produces a
   `config_matches` entry claiming two configs slice a buffer identically when
   they do not. Pair with a negative test: two configs of one op differing *only*
   in tiling must not share a cache entry.
9. **Over-tiling fix.** A case where span overflow forces tiling *and* work
   division splits the same dim: assert the joint model picks a strictly smaller
   tile count than the `core_split_estimate = 1` path.
10. **Determinism.** Identical plans across two runs under
    `torch.use_deterministic_algorithms(True)`.
11. **Fallback and incumbent** — the two non-optimal outcomes (R8.3). *Failure:*
    with the gate on and the solver forced to `INFEASIBLE` or an injected
    `SolveError` (an epsilon time limit no longer forces this — it now yields a
    feasible incumbent that is applied; a zero limit is skipped by the
    `if self._time_limit_seconds` guard, `ilp_solver_ortools.py:457`), a graph
    that requires span-forced tiling compiles through the retained span-overflow
    path and matches today's plan, and the failed solve leaves no trace in the IR
    (it precedes `coarse_tile`, R8.3) — M9 case 1: the pinned candidate sets
    survive, so a hint is still respected. *Incumbent:* with an epsilon
    deterministic budget on a solvable graph, assert the timed-out solve applies
    its feasible incumbent — the graph is tiled per that plan, not dropped to the
    greedy fallback — and that the incumbent is no worse than the warm-start plan
    under the objective.
12. **Padding (R10).** Assert the derived legality pad reaches buffer sizing —
    predicted sizes are computed from the padded `device_size`, not the unpadded
    shape (R10.1). Extend `tests/inductor/test_padding.py` with a **shared-operand**
    case: two matmuls reading the same operand emit one padded buffer, not two
    (R10.4). Report the R10.3 padding-cost measurement (unaligned-`K` matmul vs.
    hand-pre-padded) either way, and — only if it justifies discretionary pad —
    add a **chosen-pad** case where a pad row unlocks a core split that
    divisibility would otherwise block (R10.2).

## Alternatives considered

**A separate CP-SAT tiling stage ahead of layout planning.** Cleaner to land and
test, but it reproduces the current defect in a new place: a tiling chosen
without seeing LX occupancy or the core division still has to guess. Rejected in
favour of one joint model — which is the roadmap's M1, not a choice re-opened
here.

**Keeping the hardcoded objective and adding tiling terms to it.**
Requires editing the solver for every cost experiment, and the interesting
question — how to trade HBM traffic against parallelism against loop overhead —
is exactly the one that needs iteration. Rejected in favour of injection, which
is M2. (Today's objective is two-phase lexicographic; §4 replaces it with a
single-phase weighted one — the collapse M2 mandates, see R3.2.)

**Pre-computing tiling groups from producer/consumer connectivity, then having
CP-SAT pick one tiling per group.** A smaller model, but the grouping heuristic
becomes a second place where a wrong guess is unrecoverable, and grouping is
precisely what the objective should decide. Rejected in favour of cut variables.

**Modelling tiling and division as independent variables** rather than a
precomputed config cross product. Keeps the model smaller in variable count, but
reintroduces the products and divisibility conditions as nonlinear constraints.
The config encoding absorbs them into `AddElement` table lookups at the cost of
enumeration, which is bounded by the caps in R1.7.

## Resolved design decisions

These four were raised as open questions and have been resolved for phase 1;
each aligns with a roadmap requirement rather than standing alone:

- **Segmentation granularity — resolved: whole-graph.** The solve is a single
  CP-SAT instance over the entire graph, not decomposed at matmul or any other op
  boundary (§6) — this is M1 ("one CP-SAT solve"). No op-specific break is
  assumed; matmul cuts fall out of untileability, not a segmentation rule.
- **How cuts should be priced — resolved: loops are free.** A cut is priced only
  through the consequences it materializes; no `n_groups` or per-cut term (§3). A
  cut that materializes no boundary copy costs nothing. This is a working
  assumption for track C1 and can be revised with a loop-overhead term if a later
  cost model justifies it.
- **Config cap per op — resolved: no cap.** The model does not cap configs per op;
  model size is controlled by external pruning of the enumerated set (§6, R2.4),
  and shrinking the table is the roadmap's Phase 5.
- **Default objective — resolved: keep today's terms, single-phase.** The default
  stays today's spill and core terms, now combined into one single-phase weighted
  objective rather than the two-phase lexicographic solve (§4, R3.2, R3.5) — the
  M2 collapse.

## Open questions

Most of these are owned by a roadmap track or a later phase; they are listed here
because tiling is where the question first bites.

- **Objective tuning (track C1).** The single-phase default reproduces today's
  terms with the spill term weighted to dominate. What weighting, and what
  additional terms (tile count / loop overhead, `peak_lx_bytes`), should the
  default carry once the mechanism is trusted? Calibration is C1's, but the tiling
  terms are what it first has to weigh.
- **Discretionary padding (R10.3).** Does the padding-cost measurement justify
  enumerating pad-beyond-legality rows, or does the derived pad stay purely in the
  apply step? Open until the benchmark runs.
- **Stickification (Phase 3).** Relayout cost is unmodelled (R6), which can be
  pessimistic on relayout-driven HBM traffic. When is Phase 3's `relayout[e]` work
  worth landing, and does tiling make that pessimism large enough to reprioritize
  it?
- **Nested and fused reduction-axis tiling.** Single-level reduction tiling is in
  scope (R1.8); two extensions are not. *Nested* output+reduction needs the
  wrong-numerics bug fixed first so `_validate_reduction_tiling`'s stated contract
  matches reality — deliberately out of scope, as this RFC does not take on
  library fixes elsewhere. *Fused* reduction tiling — letting a reduction-tiled op
  share a loop nest rather than stay a singleton — is the modelling question:
  the group invariant needs per-run, per-level literals rather than pairwise cut
  tables (R4.6), a real increase in model size. Is either worth paying for?
