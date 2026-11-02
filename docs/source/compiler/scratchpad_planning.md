# Scratchpad (LX) optimization

Where LX scratchpad planning sits in torch-spyre today, and what we are
working on next.

:::{admonition} Status
:class: note

Scratchpad planning runs by default. The pass is gated by `lx_planning`,
which has defaulted to `1` since [#2459](https://github.com/torch-spyre/torch-spyre/pull/2459).
The OR-Tools CP-SAT solver (`config.layout_solver = "cpsat"`) is the
default. Greedy, first-fit, and best-fit are available as opt-ins;
`layout_solver` can also be set from the `LAYOUT_SOLVER` environment
variable.

Co-optimization with work distribution is on by default.
`config.co_optimizing_lx_planning` (`CO_OPTIMIZING_LX_PLANNING=0` to opt
out) enlarges each op's set of candidate splits (pointwise
dim-flips, the matmuls' tilings offered to neighbours, cross-matmul split
transfer, a shared batch-major `B/M` tiling for matmuls and reductions),
then searches the cross-product for the assignment that minimizes HBM
traffic. Every candidate, including work division's seed, must satisfy hard
work-division constraints; generated alternatives also pass stick validation.
An op with no legal candidate raises `Unsupported`.
:::

**Quick navigation:**

- [Hardware context](#hardware-context)
- [Why scratchpad planning matters](#why-scratchpad-planning-matters)
- [Assumptions](#assumptions)
- [Pipeline position](#pipeline-position)
- [Optimizations on softmax](#optimizations-on-softmax)
- [Implementation](#implementation)
- [Solvers](#solvers)
- [Co-optimization with work-distribution](#co-optimization-with-work-distribution)
- [LX context switching](#lx-context-switching)
- [Current limitations](#current-limitations)
- [Target patterns](#target-patterns)
- [Future work](#future-work)

## Hardware context

Each Spyre core has a 2 MB on-core scratchpad (LX) alongside shared HBM.
LX reads are much cheaper than HBM and have no cross-core contention, so
the planner aims to keep reused tensors on-core and let HBM traffic happen
only at the graph boundary.

:::{figure} ../_static/images/lx/memory-hierarchy.svg
:alt: Spyre memory hierarchy. Large slow HBM shared by 32 cores, each with a 2 MB LX scratchpad.
:width: 480px
:align: center

HBM is plentiful but slow and shared. LX is small but fast and core-local.
The compiler picks which buffers live where.
:::

| Parameter | Value | Config |
|---|---|---|
| Total LX per core | 2 MB | fixed |
| Program/debug reservation | 64 KB | fixed |
| Backend-reserved fraction | 20% | `DXP_LX_FRAC_AVAIL` |
| Usable LX per core | ~1.55 MB | `round_up_128(int(((2<<20) - (64<<10)) * (1 - frac_avail)))` |
| Alignment | 128-byte (stick) | implicit |
| Cores | 1 to 32 | `SENCORES` |
| Per-core HBM span limit | (256 MB) | hardware, separate from LX |
| Inter-core data ring | yes | not yet used by compiler |
| Inter-core reduce-sum ring | yes | not yet used by compiler |

## Why scratchpad planning matters

Spyre is often memory-bound: compute cores stall waiting on HBM. Every byte
the compiler can keep on LX between producer and consumer is a byte the
runtime never has to fetch.

Take a single-core softmax over a `(512, 1024)` fp16 tensor, 1 MB of input.
The lowered op sequence is `max → sub → exp → sum → div`. Total HBM traffic
depends on which intermediates land on LX:

| Stage | What changes | HBM read+write | Speedup vs baseline |
|---|---|---|---|
| 1. baseline (HBM only) | every intermediate goes through HBM | 8MN + 4N | 1.0x |
| 2. pin reduction outputs to LX | `max` and `sum` outputs stay on-core | 8MN | ~1.0x (reductions are tiny) |
| 3. + in-place ops on LX | `exp`, `sub` reuse their input's address | 3MN | ~2.7x |
| 4. + clone the input to LX | one pass over HBM, everything else stays | 2MN | ~4.0x |

The ideal memory time after stage 4 is roughly 25% of baseline. End-to-end
measurements on this softmax kernel show the median runtime drop from
32.5 µs to 23.7 µs, a 27% reduction. The gap between the ideal and the
measured result is fixed per-bundle overhead.

The four stages map onto code under `torch_spyre/_inductor/scratchpad/`:
LX-eligible op outputs (stage 2), in-place reuse (stage 3), and
input-boundary cloning (stage 4), which `ScratchpadAllocator` performs
inline via `_eligible_clone_inputs`, gated by `clone_at_graph_boundaries()`.

## Assumptions

### LX state survives kernel boundaries

The planner assumes LX state persists across SuperDSC bundle boundaries. It
operates on the flat operations list before fusion and has no awareness of
where bundle boundaries will fall, so allocation decisions can span
multiple bundles.

There is a correctness gap under VF multi-tenancy: the runtime may wipe LX
on context switch at any bundle boundary. Once SpyreCode with symbolic
addresses is available, fusion will not be limited by the number of
tensors used by the bundle, and bundle boundaries should only land at
FallbackKernels, which are visible to the planner.

### Working sets are already right-sized

Tile size selection (BLOCK_M, BLOCK_N, BLOCK_K, etc.) to fit operands
within ~1.6 MB is a pre-Inductor concern, the same class of problem GPU
autotuners solve. Spad opt begins after tiling. Given operations whose
working sets are feasible, the planner decides which buffers to pin to
LX, at what addresses, and for how long. Tiling determines whether data
*can* fit; spad opt determines whether it *does* fit.

### No eviction from LX

Buffers placed on LX stay until end-of-life. There is no mechanism to
move a buffer to HBM and reload it later. This is deliberate. Eviction
only wins when a buffer is read many times on LX, goes dormant, then is
read many times again, which is rare in practice. Pre-Inductor tiling
already keeps per-op working sets small. The remaining problem (which
buffers to keep on LX when accumulated live buffers exceed capacity) is
better solved by smarter placement and spill decisions at allocation
time than by runtime eviction with its graph mutation complexity and
extra HBM round-trips.

## Pipeline position

Scratchpad planning runs at the end of `CustomPreSchedulingPasses`,
after work division has stamped per-op core splits:

```
deadcode_elimination
propagate_named_dims                  # named-dimension metadata (pre-stickification)
assign_dim_hints
insert_bmm_padding                    # pad matmul y's K (pre-stickification)
split_multi_ops
propagate_spyre_tensor_layouts        # assign FixedTiledLayout
validate_ops
optimize_restickify_locations
reorder_nonstick_dims                 # reorder matmul non-stick dims for work division
finalize_layouts
insert_restickify
enforce_indirect_access_layout
reorder_nonstick_dims_mutation        # execute deferred reorders after insert_restickify
insert_post_mutation_restickify
insert_restickify_padding
dedup_and_promote_constants
_maybe_coarse_tile_span_overflow      # span-overflow coarse tiling (post-stickification)
span_reduction                        # work-division: enforce 256 MB span
cost_model_matmul_division            # work-division: matmul cost model
work_distribution                     # work-division: default distributor
_maybe_scratchpad_planning            # ← THIS PASS, gated by config.lx_planning
```

Two ordering constraints fix this slot:

- **Work division must run first.** Scratchpad planning reads the
  symbol-keyed `iteration_space_ownership` committed by work division to
  compute per-core buffer sizes. Work division also decides whether adjacent
  ops have compatible core splits.
  Incompatible splits trigger `core_div_mismatch` and disqualify shared
  buffers from LX (see [Current limitations](#current-limitations)).
- **Stickification must run first.** All buffers need `FixedTiledLayout`
  for device-memory size computation.

## Optimizations on softmax

The softmax example (`max → sub → exp → sum → div` over a `(512, 1024)`
input) is the easiest way to see what the planner does as each
optimization is added.

:::{figure} ../_static/images/lx/softmax-stages.svg
:alt: Four stages of LX optimization on softmax. Baseline, pin reductions, in-place, clone-input.
:width: 720px
:align: center

Each stage corresponds to one capability the planner gained. Boxes
coloured red touch HBM, green stays on LX, yellow is in-place reuse, and
blue is a clone inserted by the planner.
:::

**Stage 1, baseline.** No LX. Every op reads and writes HBM. For
`(M, N) = (512, 1024)` with a reduction along axis 0, total HBM I/O is
`8MN + 4N` bytes (eight full passes over the matrix plus four passes over
the reduction vector).

**Stage 2, pin reduction outputs to LX.** `max` and `sum` produce small
vectors (`1 × N`) that the next op reads immediately. Routing these
through LX instead of HBM costs almost no LX budget but eliminates
the `4N` term. On large `M × N` shapes this is a tiny relative win, but
it sets up the next two optimizations, which are large.

**Stage 3, in-place ops.** When a buffer is on LX and its last reader is
itself dying-after-this-op, the output of the next op can reuse the same
LX address. `exp` and `sub` are flagged as `torch.Tag.pointwise`
and therefore in-placeable. After stage 3, the only HBM access left is
the graph input and graph output, for `3MN` bytes total, a 62% reduction.

**Stage 4, clone the input to LX.** The graph input is read by several
ops. Without a clone each reader would re-fetch from HBM.
`ScratchpadAllocator._eligible_clone_inputs` detects multi-use inputs that
fit in LX and inserts a `clone` op at the front of the graph, gated by
`clone_at_graph_boundaries()`. The clone reads HBM once and
writes LX; every subsequent op reads from LX. After stage 4 total HBM is
`2MN`, the input read plus the output write, which is the theoretical
minimum for this graph.

Numbers from a 1000-iteration measurement (after 200 warm-up runs):

| Variant | dim | M×N | cores | LX | clone | in-place | median (µs) |
|---|---|---|---|---|---|---|---|
| baseline | 0 | 512×1024 | 1 | off | n/a | off | **32.51** |
| stage 2 | 0 | 512×1024 | 1 | on | n/a | off | 27.66 |
| stage 3 | 0 | 512×1024 | 1 | on | n/a | exp,sub | 23.93 |
| stage 4 | 0 | 512×1024 | 1 | on | yes | exp,sub | **23.67** |
| 4-core | 0 | 512×1024 | 4 | on | yes | exp,sub | 32.17 |

The 4-core run is *slower* on this small shape because communication and
work-distribution overhead dominate. Multi-core LX wins on larger
tensors, see below.

### Multi-core LX

A `(1024, 2048)` fp16 tensor is 4 MB, bigger than any single core's LX.
Splitting the rows over four cores gives each core a `(256, 2048)` slice
(~1 MB per core) that fits.

:::{figure} ../_static/images/lx/multicore-tiling.svg
:alt: Splitting a 4 MB tensor across four cores so each per-core slice fits in LX
:width: 580px
:align: center

For tensors larger than 2 MB, the same shape that overflows single-core
LX fits comfortably once it has been split across cores by work
distribution.
:::

Multi-core LX is not free. Adjacent ops can request different splits (one
sliced by rows, the next by columns), in which case the shared buffer is
stuck on HBM. That mismatch is what motivates co-optimization (below).

## Implementation

### Architecture

Scratchpad planning has three layers with separate concerns:

:::{figure} ../_static/images/lx/allocator-architecture.svg
:alt: ScratchpadAllocator delegates to a pluggable MemoryPlanSolver and runs optional graph passes around it
:width: 480px
:align: center

`ScratchpadAllocator` runs pre-passes (clone insertion), gathers
`LifetimeBoundBuffer`s, hands them to a pluggable solver, then writes the
chosen LX addresses onto buffer layouts. `CoOptimizingAllocator`
extends this flow with a split-search step before the solver runs.
:::

The relevant code lives under `torch_spyre/_inductor/scratchpad/`:

| File | Responsibility |
|---|---|
| `passes.py` | `ScratchpadOptimizationPass` ABC, `_NameSwapHandler` |
| `plan_solver.py` | `MemoryPlanSolver` ABC (declarative exclusion via `partition`/`excluded`), `LifetimeBoundBuffer` |
| `greedy_solver.py` | `GreedyLayoutSolver` |
| `firstfit_bestfit_solver.py` | `FirstFitLayoutSolver`, `BestFitLayoutSolver` |
| `ilp_solver_ortools.py` | `CpSatLayoutSolver` (OR-Tools CP-SAT) |
| `cpsat_expr_printer.py` | `cost_expr_to_cpsat`, lowering the sympy cost expression into the CP-SAT objective |
| `simulated_annealing.py` | `SimulatedAnnealingLayoutSolver` |
| `cooling_schedules.py` | cooling schedules for the annealing search |
| `torch_spyre/csrc/perm_layout_native.cpp` | `NativePermutationLayoutSolver`, the C++ permutation packer |
| `graph_editor.py` | `GraphEditor`, the clone/rewrite helper used by input- and output-boundary cloning |
| `allocator.py` | `ScratchpadAllocator`, `CoOptimizingAllocator`, and the single LX-eligibility predicate (`_residency_reasons`, one reason per buffer) |
| `utils.py` | liveness, mem usage, op-name/eligibility helpers |

### Entry point

```python
scratchpad_planning(graph, allocator=ScratchpadAllocator())
```

`ScratchpadAllocator` runs the following pipeline:

1. **Input-boundary cloning.** When `clone_at_graph_boundaries()` is set,
   `_eligible_clone_inputs` walks graph inputs and inserts a `clone` for any
   HBM input that is read more than once *and* fits on LX. The clone output
   becomes a fresh LX-eligible buffer.
2. **Buffer analysis.** `_generate_buffers` produces one
   `LifetimeBoundBuffer` per buffer, *including* the ones that may not
   reside. Nothing is filtered out; `ScratchpadAllocator._residency_reasons`
   (in `allocator.py`) decides eligibility and the verdict rides along as
   `residency_reason` (see
   [Declarative exclusion](#declarative-exclusion) below).
3. **Layout planning.** The solver partitions off every barred buffer
   (`MemoryPlanSolver.partition`), then assigns an `address` to each of the
   rest it can fit; whatever is left gets `address=None` and stays on HBM.
4. **Push allocation.** Successful placements are written to
   `layout.allocation["lx"] = addr` on each buffer's `FixedTiledLayout`.
5. **Post-passes.** Reserved for solver-driven graph mutations such as op
   re-ordering. Output-boundary cloning already runs as part of push
   allocation: `_push_allocation` calls
   `graph_editor.push_allocation_with_clone(..., input=False)` and
   `change_graph_output` to promote a producer to LX and clone the value
   back to a graph output.

### Matmul output staging before a collective

After the solver commits LX buffers and relayout copies, the co-optimizing
allocator can stage a fresh matmul output in a gap that is free for the whole
graph. Its sole reader must be a bare all-reduce. The allocator inserts an HBM
clone after the matmul and makes the collective read and mutate that clone.
This changes the matmul's output placement; it does not remove the HBM copy
required by the collective.

Staging uses only space left after other allocations. It cannot displace a
chosen relayout copy. Enabling another relayout can remove a free gap and change
which outputs are staged. This is a performance dependency, not a reason to
relax either path's safety checks.

Staging is a named post-solve exception to ordinary buffer residency. It requires
one fresh, untiled matmul output, one bare all-reduce reader, private single-writer
storage, complete reads and an accepted ownership view. Other opaque users,
aliases, unsupported back gaps and K-split writers are refused. The selected
collective uses the HBM clone. It does not read LX directly.

`test_real_gap_solver_keeps_stages_disjoint_or_declines` checks full placed spans,
including reduction splits and relayout copies. The captured-graph test
`test_stage_push_rewrites_a_captured_matmul_collective` checks the clone and rewire.

Set `SPYRE_LX_COLLECTIVE_OUTPUT_STAGING=0` to disable staging independently of
ownership certificates. The switch clears staging state before placement and
leaves solver allocations and ordinary output clones unchanged. Its value is
part of the compile cache key. `test_staging_switch_declines_before_placement`
checks this boundary.

### Declarative exclusion

The ordinary pre-solve verdict is computed in `ScratchpadAllocator._residency_reasons`
in `allocator.py`, and carried
to the solver as a single field, `LifetimeBoundBuffer.residency_reason`:
`None` means the buffer may be pinned, any string is the reason it may not.

**No buffer is ever dropped.** A barred buffer is still handed to the
solver so it keeps participating in slicing matching and in-place chains: a
forced-out consumer keeps its producers' residency viable instead of
orphaning them, and an in-place parent reference always resolves. Honouring
the verdict is therefore each solver's responsibility, via
`MemoryPlanSolver.excluded`, and every solver (greedy, first-fit, best-fit,
simulated annealing, CP-SAT) routes its exclusions through it.

**Where a check belongs.** Precomputable from the graph ⇒ it lives in
`_residency_reasons` as a reason string. Depends on the solver's free variables ⇒
it stays a constraint in the solver: today that is only CP-SAT's per-edge
slicing match over the division variables and its in-place merge gate.
Capacity is the exception that belongs to neither allocator: it is solver
state, so it lives on `MemoryPlanSolver.excluded` alongside the tag.

The checks, in evaluation order (the first failure is the reason reported):

<!-- BEGIN GENERATED BUFFER RESIDENCY RULES -->
| Rule / reason label | Why | Requires | Exceptions / certificate | Covering tests |
|---|---|---|---|---|
| `op not allowed` | Only supported computed outputs can be placed. Mutation layouts use external storage; the debug flag only bypasses the operation-name denylist. | supported producer | None | `TestLoopCarryLxEligibility`; `test_buffer_residency_adapter_rejection_strings` |
| `unsized (no device layout)` | Without a device layout, the allocator cannot measure the LX footprint. | computable footprint | None | `TestEmptyLxEligibility`; `test_buffer_residency_adapter_rejection_strings` |
| `empty tensor` | A zero-sized tensor needs no LX reservation. | nonempty output | None | `TestEmptyLxEligibility` |
| `mutation target` | Offset writes cannot use one fixed LX base. A counted-loop carry is accepted only when the solver also chooses its update division. | no mutation target | Counted-loop carry: `_is_loop_carry_storage` (joint division choice) | `TestLoopCarryLxEligibility` |
| `tiled (advancing)` | LX addresses cannot use affine.apply symbols. The write or read advances per tile; loop_info supplies the tile geometry. | fixed LX address | None | `TestLoopCarryLxEligibility`; `test_buffer_residency_adapter_rejection_strings` |
| `read by restickify` | A transposed or uncertified local read may cross cores. The joint solver checks each candidate division, so the committed-division barrier can reject a valid candidate (#4655). | fixed divisions agree or exact local-read proof | None | `test_buffer_residency_adapter_rejection_strings` |
| `extern kernel user or live across extern kernel` | Extern kernels read HBM. PR #3683 added this guard; enable_lx_context_switching uses LxContextSwitchingPass to dump and restore LX around the call. | extern boundary protected | None | `test_buffer_residency_adapter_rejection_strings` |
| `live across multi-output extern kernel` | Multi-output extern calls need HBM under either flag setting. LxContextSwitchingPass does not bracket them; the earlier extern guard handles the disabled case. | multi-output boundary protected | None | `test_buffer_residency_adapter_rejection_strings` |
| `index tensor or indirectly accessed` | Data-dependent indices and the values they address resolve through HBM. | direct addressing | None | `test_buffer_residency_adapter_rejection_strings` |
| `graph output is a ReinterpretView` | A boundary output clone cannot rewrap a ReinterpretView. | rewritable output clone | None | `test_buffer_residency_adapter_rejection_strings` |
| `graph output mutated after production` | An output clone placed after its producer would copy a stale value before a later mutation. Only a validated drain plan moves that clone after the counted loop. | output clone sees final value | Validated output drain: `validated_drain_plans` (clone follows counted loop) | `TestLoopCarryLxEligibility` |
| `partial/offset read` | Sliced or multi-offset LX reads can address the wrong data. A certified zero-offset prefix needs a served full-backing gather. | complete read | Zero-offset linear prefix: `prefix_lx_gather_source` (one reader and one writer, full-backing served copy, physical stride preserved, unsplit K, disjoint LX spans); Contained local read: `local_lx_subrange_source` (rollout enabled, compact materializer or certified cache scatter, storage and lifetime permission, exact local byte coverage) | `test_buffer_residency_adapter_rejection_strings`; `test_prefix_lx_contraction_requires_exact_backing_stride`; `test_prefix_linear_reads_a_separate_full_backing_lx_gather`; `test_prefix_linear_without_served_copy_spills_the_source`; `test_prefix_linear_scheduler_demotion_retains_valid_hbm_clone`; `test_prefix_lx_contraction_requires_exact_backing_stride[second_reader]`; `test_registry_exception_keeps_later_guards` |
| `core div mismatch: ...` | Fixed divisions must agree across users. The joint solver chooses compatible divisions instead of rejecting a provisional mismatch. | compatible fixed division | None | `test_buffer_residency_adapter_rejection_strings` |
| `graph output (no clone); no consumer reads it from LX` | Without a boundary clone, a graph output must land in HBM. Residency also saves no reads when the producer has no LX consumer. | untiled reuse or boundary clone | None | `TestLoopCarryLxEligibility`; `test_buffer_residency_adapter_rejection_strings` |
| `lx back gap` | A generic LX back gap can misaddress a read. The prefix exception needs a full-backing copy, an unsplit-K reader and a uniform per-core gap. | no unsupported LX gap | Zero-offset linear prefix: `prefix_lx_gather_source` (one reader and one writer, full-backing served copy, physical stride preserved, unsplit K, disjoint LX spans) | `test_prefix_linear_reads_a_separate_full_backing_lx_gather`; `test_buffer_residency_adapter_rejection_strings`; `test_prefix_lx_contraction_requires_exact_backing_stride`; `test_prefix_linear_without_served_copy_spills_the_source`; `test_prefix_linear_scheduler_demotion_retains_valid_hbm_clone`; `test_prefix_lx_contraction_requires_exact_backing_stride[second_reader]` |
<!-- END GENERATED BUFFER RESIDENCY RULES -->

This table comes from `BUFFER_RESIDENCY_RULES` in
`scratchpad/residency_rules.py`. Run `python tools/lx_planning/generate_residency_rules.py`
to update it. The allocator evaluates these rules in this order. An exception
clears only its own rule. Storage, lifetime, capacity and later rules still apply.
Reason labels group related diagnostics. A returned reason can include more
detail, such as the restickify proof failure or the mismatched core division.
The adapter tests check the returned strings; the table does not replace them.

Graph inputs use `_input_residency_reason`. They also require boundary cloning,
rewritable uses, complete reads and useful LX reuse.

A narrow exception permits an untiled matmul to read a zero-offset prefix made
of complete storage sticks. It requires one private, full-width producer and a
separate gather of its **complete backing buffer**, with the original physical
row stride preserved. The reader must not split K. Source and copy spans must
be disjoint, and both must fit in frontend LX. Only the outermost stick axis
may have an unused tail. Other offset, strided, partial, aliased or multi-reader
cases remain excluded. The fixed placement path must already have a copy plan;
the joint solver must guarantee the copy through its relayout decision.

This untiled exception does not require same-tile-group relayout search.

That last distinction is the only difference between the two allocators,
and it is a parameter (`division_is_fixed`) rather than a second predicate.

With `auto_coarse_tiling` on, the joint path does not bar an op whose menu
offers a tiling for `graph output (no clone)` or `no consumer reads it from
LX`: both hold only while the op is untiled. The buffer carries the reason as
`untiled_bar` instead, and the CP-SAT solve keeps it out of LX unless it tiles
the op.

### Prefix demotion contract

If preparation rejects a prefix gather, scheduling retries after clearing the
source and copy LX allocations, their physical LX views and their relayout
registry entry. The ordinary HBM clone remains. It has one writer and keeps the
full physical extent and stride. Its consumers keep valid reads from that clone.
Preparation discards the rejected kernels and builds kernels for the HBM path.
Unrelated LX allocations stay unchanged.

This is an amendment to D6-2 in the merge plan. It does not remove the clone or
restore the original graph topology. It is not plan-exact rollback. The focused
test checks finite and NaN unused tails against the existing exact reference.
A bitwise HBM-placement comparison requires a matched same-division control;
this test does not force an allocator result to create one.

### Certified exception contracts

The generated table names each exception, its certificate, its required
conditions and its tests. Add later ownership exceptions to the registry.
An ownership proof does not replace storage, lifetime or capacity checks.
The untiled prefix case does not require same-tile-group search.

The planner and final emitter share the `local_lx_subrange_geometry` gate.
Work division receives immutable facts through `local_lx_read_geometry`.
At emission, `local_lx_emission_contract` repeats the gate on the live graph.
Keep these paths on the same gate. A geometry certificate does not replace
storage, alias, lifetime or allocation checks.

The symbolic proof registry below is separate from storage permission. It
selects the existing containment proofs in order. The view-recovery caller
keeps its floor-only, single-driven-axis restriction. An emitted-access proof
alone does not authorize a recovery candidate.
`factor_slice_axis` is listed as "Factored ownership" in the transfer-certificate
table below. It is separate from `SLICE_OWNERSHIP_PROOFS`.

<!-- BEGIN GENERATED OWNERSHIP PROOFS -->
| Ownership proof | Certificate | Requires | Covering tests |
|---|---|---|---|
| Affine rectangle containment; emitted access | `project_affine_slice_axis` | one contained loop partition, complete read rectangle, unchanged physical coordinates | `test_q_rope_ownership_keeps_the_whole_pair_in_its_slice`; `test_q_projection_rope_handoff_preserves_pairs_and_other_consumer`; `test_ownership_callers_keep_their_admission_policy` |
| Floor-affine rectangle containment; floor-only recovery | `project_floor_affine_slice_axis` | one ascending affine floor, complete physical read rectangle, unchanged coordinates and caller admission | `test_floor_slice_projection_rejects_unproven_access`; `test_attention_output_repack_handoff`; `test_ownership_callers_keep_their_admission_policy` |
<!-- END GENERATED OWNERSHIP PROOFS -->

The following rows describe transfer checks and shortlist selection. They do
not change admission order or authorize storage reuse. A reserved shortlist
slot is a search heuristic, not an ownership proof or a performance guarantee.

<!-- BEGIN GENERATED TRANSFER CERTIFICATES -->
| Certificate | Function | Requires | Predicate tests | Integration tests |
|---|---|---|---|---|
| Full-replica gather | `torch_spyre._inductor.scratchpad.lx_relayout._movement_supported` | ownership flag ON; complete source partition; one destination slice; uniform fan-in/out; all source fragments; shuffle fan-in budget | `test_work_division_hint.py::test_full_replica_movement_tracks_ownership_flag`; `test_work_division_hint.py::test_movement_rule_reads_the_owner_maps` | `test_solver_relayout_e2e.py::test_attention_cache_transpose_handoff` |
| Full-replica shortlist reservation | `torch_spyre._inductor.scratchpad.allocator.CoOptimizingAllocator._cap_relayout_groups` | ownership flag ON; cheapest priced full replica; an existing unpinned slot; sibling pins, prices and cap preserved; solver may decline the copy | `test_solver_relayout_decision.py::test_relayout_full_replica_preserves_sibling_pins` | `test_solver_relayout_e2e.py::test_attention_cache_transpose_handoff` |
| Updated-input cache snapshot | `torch_spyre._inductor.scratchpad.lx_relayout._is_activation_source` | earlier direct cache mutation; immutable restickified snapshot; original cache load and dependencies preserved | `test_work_division_hint.py::test_lx_relayout_activation_policy_is_source_wide` | `test_solver_relayout_e2e.py::test_attention_cache_transpose_handoff` |
| Fewer-core complete partitions | `torch_spyre._inductor.scratchpad.lx_relayout.core_domain_rejection` | integral core ratio; complete disjoint partitions; larger domain uses every compute core; physical spans stay separate | `test_solver_relayout_candidates.py::test_consumer_rules_are_shared_with_the_committed_collector` | `test_solver_relayout_e2e.py::test_k_projection_rope_complete_contraction` |
| Persistent external cache scatter | `torch_spyre._inductor.scratchpad.lx_relayout._persistent_scatter_consumer` | fresh immutable matmul source; one external indirect overwrite; no source alias or cache read; mutation order and lifetime preserved | `test_solver_relayout_candidates.py::test_persistent_scatter_requires_independent_immutable_source` | `test_solver_relayout_e2e.py::test_v_projection_cache_handoff_preserves_persistent_alias` |
| Contained local LX read | `torch_spyre._inductor.scratchpad.lx_relayout.local_lx_emission_contract` | ownership flag ON; shared graph and compact-read geometry gate; same-core byte certificate; fresh final access; disjoint live allocations | `test_lx_read_contracts.py::test_stale_or_unsafe_final_facts_are_rejected`; `test_work_division_hint.py::test_local_read_rollout_declines_before_graph_access` | `test_solver_relayout_e2e.py::test_gate_up_full_parent_repartition` |
| Full-parent copy for local subranges | `torch_spyre._inductor.scratchpad.lx_relayout.materialize_lx_relayouts` | complete parent copied once per destination group; every partial reader re-certified against the destination view; positive disjoint LX spans | `test_subrange_transfer.py::test_gate_up_siblings_have_distinct_copy_identity`; `test_solver_relayout_decision.py::test_copy_is_sized_by_the_destination_span_not_the_source_share` | `test_solver_relayout_e2e.py::test_gate_up_full_parent_repartition` |
| Pointwise persistent-scatter materializer | `torch_spyre._inductor.scratchpad.lx_relayout._pointwise_scatter_materializer` | ownership flag ON; one certified compact parent subrange; complete scatter read; persistent-scatter storage, alias and ordering checks | `test_solver_relayout_candidates.py::test_pointwise_scatter_materializer_admission` | None |
| Direct cache-snapshot subrange read | `torch_spyre._inductor.scratchpad.lx_relayout._local_lx_scatter_geometry` | single-writer complete immutable snapshot; one direct partial read; other reads are scatter indices; standard 128-byte sticks; zero inner origin and exact rectangle size; external mutation target and final same-core ownership checks preserved | `test_solver_relayout_candidates.py::test_direct_cache_snapshot_scatter_geometry` | None |
| Factored ownership | `torch_spyre._inductor.slice_ownership.factor_slice_axis` | exact zero-origin dense product; independent digits; complete rectangular slices; no address rewrite or storage permission | `test_work_division_hint.py::test_factored_slice_rejects_inexact_products` | `test_solver_relayout_e2e.py::test_k_projection_rope_complete_contraction` |
<!-- END GENERATED TRANSFER CERTIFICATES -->

`SPYRE_LX_EXTENDED_OWNERSHIP=0` disables the additional ownership proofs while
retaining the earlier prefix-copy rules. Its default is ON. The flag does not
disable storage, alias, lifetime or capacity validation. Compare successive
revisions with this flag OFF in both; do not compare a later OFF result with
the preceding revision's ON result. Default ON is intentional: the contained
multi-loop affine path is enabled when this change lands.

### Per-core size and core-division mismatch

A buffer's LX footprint is its **per-core** size, not its total size: a
buffer split across `N` cores only needs `total / N` bytes on each core's
scratchpad. `get_ncores_for_buffers` (in `utils.py`) decides that `N` for
each buffer and is the gate for whether a buffer is even eligible.

- **Sizing is writer-authoritative.** The op that *writes* a buffer
  determines how the data is physically spread across cores, so the
  divisor is the writer's core count, not the maximum over all users. A
  reader on more cores only touches its own (smaller) slice of that
  residency. (Earlier code used `max()` over users, which under-sized a
  buffer whose writer ran on fewer cores than a consumer (e.g. a 1-core
  producer feeding a 32-core matmul) and wrongly pinned an over-large
  buffer to a single core's LX.) Graph inputs have no in-graph writer and
  fall back to the readers' (matching) count.
- **Mismatch detection compares per-core views.** Two ops agree on a
  buffer only if their `PerCoreView` (`_per_core_view_on_buf` in
  `pass_utils.py`), which device dim each core's slice occupies, and the
  core→slice mapping, matches. A genuine single-core "owns the whole
  buffer" access is encoded distinctly from a multi-core broadcast that
  also touches the whole buffer, so the two never compare equal by
  accident. A writer/reader core-count disagreement, a partial-sum
  (K-split-reduction) writer, or any unrepresentable geometry yields
  `core_div_mismatch` (`-1`) and disqualifies the buffer from LX.

:::{figure} ../_static/images/lx/core-div-mismatch-spill.svg
:alt: When writer and reader agree on the per-core split the buffer stays on LX at no off-chip cost; when they disagree it is written to HBM and read back, costing twice its size in off-chip traffic.
:width: 100%

A buffer disqualified by `core_div_mismatch` is written to HBM by its
producer and read back by its consumer. That boundary costs `2 x S`
off-chip bytes for a buffer of size `S`, where an on-LX buffer would have
cost none.
:::

### Codegen integration

Once `layout.allocation["lx"]` is set:

- `spyre_kernel.py` removes LX-allocated buffers from kernel args
  (core-local, no HBM backing needed).
- `codegen/compute_ops.py` writes `component_` as `"lx"`, `memOrg_` as
  LX only, and `startAddressCoreCorelet_` as the baked-in LX address
  (the same address per core on their respective scratchpads).

## Solvers

`config.layout_solver`
(`"greedy" | "firstfit" | "bestfit" | "cpsat" | "simulated_annealing"`)
picks the solver; it defaults from the `LAYOUT_SOLVER` environment
variable (falling back to `"cpsat"`).

### GreedyLayoutSolver

Walks transition points in chronological order. At each point it
deallocates expired buffers, then for each newly-live buffer:

1. If a declared in-place parent is alive at the previous time step and
   the child fits in the parent's slot, reuse the parent's address.
2. Otherwise find a free block. Try address 0, then above the high-water
   mark, then gaps between live allocations.

It is simple, easy to reason about, and in-place reuse is automatic.
Decisions are local, though. Placing buffer A at address 0 can block a
later large buffer C that would have benefited from a low address.

### FirstFitLayoutSolver and BestFitLayoutSolver

Both solvers see *all* buffers up front, sort them topologically with
ties broken by ascending lifetime, and place them shortest-life-first
into the free address space.

For each buffer, free gaps during its lifetime are computed by
subtracting the address intervals of every overlapping placed buffer.
In-place parent addresses are kept as candidate gaps so the child can
land on top of them.

The two solvers differ only in the gap-selection policy:

- `FirstFitLayoutSolver` picks the first gap large enough.
- `BestFitLayoutSolver` picks the gap that leaves the smallest remainder
  after placement.

Both naturally avoid the "buffer at address 0 blocks everything else"
failure mode of the greedy solver. They are not yet selected by default.
Once a deeptools dependency clears, first-fit is the expected default.

### CpSatLayoutSolver

`config.layout_solver = "cpsat"` selects an OR-Tools CP-SAT solver that
models placement as a global 2D no-overlap (each resident buffer is an
optional `[lifetime] × [address, address + size)` rectangle) and
minimizes total HBM transfer traffic, so a buffer that would be re-read by
*N* consumers costs `N × size` when spilled. In-place reuse is encoded by
shortening a parent's lifetime by the single handoff tick, letting the
in-place child legally share its slot.

It requires the optional `ortools` package
(`pip install torch-spyre[cpsat]`); when it is missing, the allocator logs
a warning and falls back to the greedy solver, so a `"cpsat"` request
without co-optimization always degrades to a correct plan. Without
co-optimization the CP-SAT solver only *places* buffers on each op's
pre-determined core division; with `co_optimizing_lx_planning` it is driven
by the joint `CoOptimizingAllocator` (below), which additionally chooses
each op's core division -- see [The co-optimizer](co_optimizer.md) for what
happens to that fallback when `ortools` is missing *and* co-optimization is
requested.

### SimulatedAnnealingLayoutSolver

`config.layout_solver = "simulated_annealing"` selects
`SimulatedAnnealingLayoutSolver`, which takes a first-fit, best-fit, or
greedy placement as the initial layout and then runs a simulated-annealing
search over buffer orderings to reduce fragmentation. Each step reinserts a
buffer and keeps or rejects the new ordering according to a cooling
schedule, so the search can escape the local minima that trap the
single-pass solvers. See
[Simulated Annealing Layout Planner](simulated_annealing_layout.md) for the
algorithm and the tunable schedule parameters.

Note this is placement-only. With `co_optimizing_lx_planning` the same config
value instead selects `SaCoOptimizingSolver`, a *different* class that anneals
the core divisions and the placement jointly. See
[Joint core-division + LX placement](sa_co_optimization.md).

## Co-optimization with work-distribution

Work division optimizes each op independently for parallelism. Adjacent
ops sharing a buffer can get different splits (different shapes mean
different optimal decompositions), which triggers `core_div_mismatch`
and disqualifies the shared buffer from LX even when it would have fit.
`CoOptimizingAllocator` (the default; gated by
`config.co_optimizing_lx_planning`, env var `CO_OPTIMIZING_LX_PLANNING`)
treats split choices and LX placement jointly instead of leaving this
interaction to two independent passes. See [The co-optimizer](co_optimizer.md)
for why this joint search exists, how its production solvers (CP-SAT and
simulated annealing), the pruned exhaustive-search fallback, and (for CP-SAT)
solver-driven coarse tiling each work, and which decisions -- scratchpad
placement, work division, and coarse tiling -- it covers today.

## LX context switching

LX data corruption (clobbering) can happen when two conditions hold together: (1) two
*separate* `torch.compile`s each plan their own LX addresses independently, with no shared view
of what the other has pinned; and (2) one runs nested inside the other, a `FallbackKernel`'s
eager body launching a second, separately-compiled Spyre program (via a nested `torch.compile`,
or any eager op compiled standalone through `ops/eager.py`), while the outer graph still needs
a buffer it already has LX-resident. The inner compile has no knowledge of that buffer and may
reuse its address for its own scratch:

```
op0 (write r -> LX)  ...  FallbackKernel (opaque)  ...  op1 (read r <- LX)
```

An earlier fix (PR3683) closed this by refusing LX residency outright to any buffer live
across such a call (`_extern_kernel_in_live_range` in `allocator.py`), correct, but every
access to that buffer then pays a full HBM round trip, not just the one crossing: cost scales
as `(1 + read_count)·size/BW`, growing with reuse.

`LxContextSwitchingPass`
(`torch_spyre/_inductor/scratchpad/lx_context_switching.py`) protects the residency directly
instead of giving it up. For each LX-resident buffer whose lifetime strictly straddles a risky
`FallbackKernel`, it inserts a **dump** clone (LX → HBM) immediately before the call and a
**restore** clone (HBM → LX, back to the buffer's exact original address) immediately after
it. The bracketed call itself is never modified; ordering is enforced purely through
`GraphLowering.additional_buffer_deps`/`additional_star_deps`, upstream's own mechanism for a
fake, non-lifetime-extending ordering dependency. Cost is a fixed `2·size/BW`, independent of
how many times the buffer is reused elsewhere.

Not every `FallbackKernel` needs bracketing. A cheap classification skips it entirely when the
op is a confirmed CPU-only fallback (`ops/fallbacks.py`'s shared, once-verified `_fallback`
body) or explicitly opted out via `mark_lx_safe(op)` for an op whose author has confirmed that
no intermediate buffers will ever write into LX. Otherwise, the buffer-lifetime check above
is the load-bearing gate. Classifying op behavior by namespace or registry proved unreliable
in general, since `ops/eager.py` compiles plenty of aten ops (`mm`, `add`, `softmax`,
`embedding`, …) standalone, and any of them can appear as the risky call.

Measured on an 8-layer, 512×512 fp16 repro (`read_count = 1` per bracketed buffer,
`tests/inductor/test_lx_context_switching.py`):

| Configuration | Correctness | `kernel_ms` | HBM crossings for `r` |
|---|---|---|---|
| No fix | ❌ (`diff > 0`) | 0.313 | 0 (fully LX, but corrupted) |
| Context switching (default) | ✅ | 0.386 to 0.388 | 2: fixed dump + restore |
| PR3683 guard alone | ✅ | 0.395 to 0.396 | 2: this buffer's own write + read, both via HBM |

These costs are equal in bytes moved at `read_count = 1`, yet the guard still measures
consistently slower: HBM reads/writes actually happen in small chunks, so an op processing data
directly from HBM must frequently reverse the traffic direction on the memory bus, lowering
effective bandwidth. When the data fits on LX, moving it there entirely before processing needs
no such direction-reversal. The guard pays that penalty on every touch of `r`; context
switching only pays it for the dump/restore pair.

Both mechanisms are gated by one flag, `config.enable_lx_context_switching`
(`ENABLE_LX_CONTEXT_SWITCHING`, default on): **on**, `_extern_kernel_in_live_range` is skipped
and `LxContextSwitchingPass` is registered as a `post_optimization_pass` instead; **off**, the
guard runs exactly as PR3683 shipped it and the new pass is never registered. It is a real
either/or, not a partial toggle, so old and new behavior stay directly comparable while the new
mechanism earns trust in production; removing the guard entirely is a follow-up once it does.

## Current limitations

### Greedy single-pass, no lookahead (default solver)

The greedy solver processes ops in topological order making irrevocable
placement decisions without considering future ops. First-fit and
best-fit mitigate this by sorting all buffers up front before placing.

### No defragmentation

`_find_free_block` can locate holes between allocations but cannot
compact the address space. Allocate/deallocate cycles fragment LX.

### Co-optimization is still limited

`CoOptimizingAllocator` implements the joint
work-division + LX planning idea. It searches pointwise dim-flips, the
matmuls' tilings offered to neighbours, cross-matmul split transfer, and a
shared batch-major `B/M` split for matmuls and reductions. It still never
flips a matmul's split onto a single dim (to protect compute
utilization). Remaining gaps:

- **The search is exhaustive and per-leaf cost is high.** It scores the
  full cross-product of candidates with no pruning; each leaf rebuilds the
  filtered op view. Adding the reduction `B/M` option makes `mha_4h`
  converge fully on `B/4·M/8` but pushes the search into the tens of
  seconds (the per-leaf graph-view rebuild dominates). Hoisting the
  split-invariant work out of the per-leaf path, or pruning the tree, is
  needed before this is on by default.
- **Not every producer reconciles.** A matmul input whose producer is a
  plain pointwise op (e.g. an attention `Q·scale` multiply) is not yet
  offered a split matching how the matmul reads it, so that producer can
  stay in HBM where the sendnn reference keeps it on LX.
- **No performance model.** The "honor compute-bound ops, search the
  rest" rule is still a heuristic; the trade between compute throughput
  and memory traffic is not scored.
- **Coarse-tiling integration**: landed for CP-SAT (`#4768`) and simulated annealing
  (`#4456`-`#4894`) — see [The co-optimizer](co_optimizer.md#b-lx-placement-work-division-and-coarse-tiling)
  for current status.

The factored-`B/M` and cross-matmul transfer code is marked TEMP/TODO:
the intent is for work division to assign consistent splits directly, at
which point these compensating options can be removed.

### No cross-core ring utilization

The hardware has a data ring (core-to-core LX reads/writes) and a
reduce-sum ring (cross-core sum reduction, useful for matmul K-splits).
The compiler does not yet generate code that uses either ring. The
`core_div_mismatch` hard wall exists because without ring transfers, a
buffer split N ways in one op cannot be read by M cores in the next
(with M ≠ N). Ring support could remove this wall by redistributing
data across cores without going through HBM (the ring is always faster
than HBM). Enabling it requires compiler and codegen support to emit
ring transfer instructions in the SuperDSC schedule.

## Target patterns

The test suite `test_scratchpad_patterns.py` encodes patterns the greedy
allocator cannot handle (`@expectedFailure`). Each documents a class of
problem to be solved:

| Pattern | Problem | What's needed |
|---|---|---|
| Simple fragmentation | Greedy places A at addr 0, blocking later large allocation C | Placement aware of future deallocations |
| Staircase (up/down) | Increasing or decreasing buffer sizes overflow LX under greedy append | Lookahead and placement-order optimization |
| GQ attention | Large/small buffer lifecycle alternation (Q_K, scores vs. max, denominators) | Size-aware packing exploiting lifecycle patterns |
| MoE MLP | Many buffers of varying sizes and lifetimes, shared hidden state | Stack-like placement with complex lifetime management |

Best-fit and first-fit pass several patterns the greedy solver fails.
The remaining `@expectedFailure` cases motivate the items in
[Future work](#future-work).

## Future work

The extensions below build on the current co-optimization flow through the
`MemoryPlanSolver` and `ScratchpadOptimizationPass` interfaces, so they can
be added without disturbing the rest of the planner.

### Richer co-optimization

Current state: pointwise dim-flips, matmul-tiling seeding, cross-matmul
split transfer, and a shared batch-major `B/M` split offered to matmuls
and softmax reductions. Planned extensions:

- **Make the search affordable.** Hoist the split-invariant graph-view /
  mem-usage build out of the per-leaf path (or prune the cross-product)
  so the exhaustive search does not run into tens of seconds once
  reductions also carry options.
- **Offer matmul-input producers a matching split.** A plain pointwise
  producer feeding a matmul should be able to adopt the split the matmul
  reads it with, so that producer pins to LX (closing the gap with the
  sendnn reference, which keeps both attention pre-multiplies on LX).
- **Replace the "honor compute-bound ops, search the rest" heuristic**
  with a performance model that scores compute throughput against memory
  traffic, so matmul (and other compute-bound) splits can be searched too
  when the trade actually pays off.
- **Remove the compensating options** once work division assigns
  consistent splits directly (the factored-`B/M` and cross-matmul
  transfer code is marked TEMP/TODO for exactly this).
- **Joint operation with the `coarse_tiling` pass**: landed for CP-SAT (`#4768`) and simulated
  annealing (`#4456`-`#4894`) — see
  [The co-optimizer](co_optimizer.md#b-lx-placement-work-division-and-coarse-tiling) for
  current status.

### Solver-driven graph mutations

`ScratchpadOptimizationPass` plug-ins run before or after the solver.
Candidates under evaluation:

- **Buffer evictions.** Move a buffer from LX to HBM and bring it back
  later. This is the counterpart of the "no eviction" assumption above
  and is only worthwhile when liveness shows it pays off.
- **Operation re-ordering.** Re-order independent ops to extend or
  shorten lifetimes for better packing.
- **Driving cloning from the solver.** Input-boundary cloning currently
  runs inline in `ScratchpadAllocator` with a heuristic. The longer-term
  plan is for the solver to decide which clones pay off based on the
  global layout.

### Cross-core ring transfers

Remove the `core_div_mismatch` hard wall by emitting data-ring or
reduce-sum-ring transfers in the SuperDSC schedule, so a buffer split N
ways in one op can feed a different M-way split in the next without
going through HBM. Requires compiler and codegen support.

### Non-terminal kernel hints

Extend the runtime to support a non-terminal kernel annotation. A
bundle marked non-terminal guarantees no context switch before the next
bundle, preserving LX state across the boundary. The compiler emits the
annotation based on cross-bundle LX liveness.

This buys real time on tightly coupled op sequences (for example,
softmax decomposed across bundles due to the 6-tensor limit). It needs
runtime scheduler support and compiler liveness tracking across bundle
boundaries.

## Testing

Three suites cover the planner:

- `tests/inductor/test_scratchpad_solver.py`: solver-level unit tests.
  Buffers are constructed directly as `LifetimeBoundBuffer` lists and
  fed to each solver.
- `tests/inductor/test_scratchpad_use.py`: end-to-end op-level checks
  that LX is actually used for representative graphs.
- `tests/inductor/test_scratchpad_patterns.py`: the `@expectedFailure`
  patterns above. Promoting one to passing is the typical signal that a
  new solver or pass is doing useful work.
- `tests/inductor/test_inductor_ops_lx_planning.py`: runs the full
  Inductor op suite under `LX_PLANNING=1` to catch regressions.

An auto-generated coverage suite expands op coverage beyond the
hand-written patterns above. It composes each supported op with simple
reduction or pointwise tails, so every supported op is exercised on
the planner without a hand-written test. The suite catches planning
bugs that the hand-written cases miss.

## Related documents

- [`work_division_planning.md`](work_division_planning.md) describes how
  work distribution decides per-op core splits that scratchpad planning
  then consumes.
- [`coarse_tiling_loops.md`](coarse_tiling_loops.md) describes coarse
  tiling, which reduces working sets so adjacent ops can fit on LX in
  the first place.
- [`hbm_pool_planning.md`](hbm_pool_planning.md) describes the
  complementary device-memory pass, which packs every intermediate that
  LX planning did not claim into a shared HBM segment.

### Physical geometry and feature extraction

`core_mapping.py` owns immutable `PerCoreView` values and physical owner
interpretation. Symbolic projections stay in `slice_ownership.py`. A caller
passes its rollout choice explicitly. Geometry cannot grant storage permission.
The old `pass_utils.PerCoreView` name remains an import alias.

The dense subrange certificate records physical domains, byte offsets, strides
and fragment coverage. It does not allocate storage, approve a reader or emit a
native shuffle. The planner must still check storage, writers, aliases,
lifetimes and supported emission.

`cost_features.py` owns feature extraction. Planning and reporting use the same
functions. `dump_cost_model.py` keeps diagnostic output and latest-report state.
This extraction does not change coefficients, units or physical core domains.

### Local-read permission and final facts

The planner checks graph structure, independent storage, writers, aliases and
lifetimes. It exports `LocalReadGeometry` for one residency edge. Work division
uses this immutable value to test physical ownership. Missing facts decline a
partial match. Geometry alone cannot authorize storage.

At the actual emitted read, the scheduler's provider repeats the planner's
live checks. It returns a frozen `LocalReadEmission` value. The emitter checks
that the final TensorArg still has the certified shape, format, owner map and
allocation. It checks addresses and overlap before rebasing the read. A missing
provider fails closed. A preparation retry creates a fresh provider and facts;
no local-read certificate is reused across demotion.

The earlier materialized-relayout registry still certifies copy identities.
This interface does not alter that contract or add a new attention operation.

### Matched OFF-plan comparison

Use `tools/lx_planning/run_off_plan_capture.py` on the same existing fixture at each adjacent
source pair, from the repository root in a clean developer checkout or Git
worktree with the native test dependencies. The driver reads that checkout's
HEAD and refuses tracked changes. It does not run from a source archive without
Git metadata. Set `SPYRE_LX_EXTENDED_OWNERSHIP=0` before either process starts.
Use fresh compile caches and the same upstream commit, inputs, hints, planner
seed, worker count and other compiler settings. The contract records the
verified input and hint hashes. Do not compare unmatched source configurations.

The read-only observer records successful final codegen. It keeps final graph
storage and physical layouts, dependencies, relayout registry, complete OpSpecs,
argument binding and HBM pool size. It does not force a layout or a division.
Unknown fields stop capture. Empty captures and cache hits cannot pass.
Buffer names and program order remain significant; differences require review.

Compare the two saved records with `tools/lx_planning/lx_semantic_plan.py`. Repeat the
adjacent OFF/OFF comparison through the final stack, then compare its OFF plan
with the matched #5160 OFF plan. Numerical checks remain separate. No recorded
native OFF/OFF result is included in this PR. Separate historical receipts cover
finite fixtures at pinned source revisions; they do not establish whole-model
equality or plan equality after later upstream changes. Host tests of the
recorder do not establish that native comparisons have passed.

The pre-flag references are #5160 at `525dfd8c` and its squash merge on main,
`10f40954`. A later main checkout is also accepted only when its recorded HEAD
matches the checkout, it descends from that squash merge, it belongs to a local
`main`, `origin/main` or `upstream/main` ref, and its config file has the same
verified hash. Fetch the baseline history and main ref before using a later
checkout. Only these verified sources may omit `lx_extended_ownership`; the
missing flag represents OFF. Other missing settings stop capture. Iteration-axis
order remains significant in the saved plan.

### Cumulative local-read rollout

`SPYRE_LX_EXTENDED_OWNERSHIP=0` also disables the optional local subrange
admissions. This includes same-core compact readers and full-parent copies
that serve them. The geometry helpers remain available for validation.
Existing prefix support remains active. Mandatory span, ownership, lifetime,
alias and final-emission checks stay active. The switch is part of the compile
cache key; reuse of an ON plan cannot establish an OFF-plan comparison.

The same switch disables the hinted compact-reader FX rewrite. Direct cache
subranges use the local-read admission gates. Their physical and final-access
validators remain active. No separate switch or relaxed cache guard is added.
