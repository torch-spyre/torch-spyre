# Multi-Core Work Division for Indirect Access

## Summary

Indirect access reads or writes rows of a table at positions chosen at runtime
by an index tensor:

* **gather** (`out = x[i]`) reads rows from a value table `x`;
* **scatter** (`out[i] = src`) writes rows into a destination table `out`.

This page describes running both across multiple Spyre cores **correctly** under
work division.

The single load-bearing rule, in both directions, is: **the work-division
planner must never split along one of the shared indirect table's data
dimensions.** That table — the value table for a gather, the destination table
for a scatter — is shared at a single base across cores: every core must be able
to address any row. Slicing one of its data dimensions (e.g. the hidden dim of
an embedding, the head dim of a KV cache) per core silently returns wrong
results.

The fix makes such ops correct. It does **not** parallelise the data dimensions
themselves; that is future work (see
[Limitations](#limitations-and-future-work)).

Scatter carries **one extra condition** gather does not: because the
data-dependent index sits on the *output*, parallelising the index-entry
dimension is only safe when no two cores target the same destination row. We
enable it for **overwrite** scatters and leave **accumulating** scatters serial.

## Background: how indirect access is compiled

### Gather — indirect on a load

A gather is lowered to a **Pointwise `identity` op** with three tensor
arguments:

| Arg | Role | Device coordinates |
|---|---|---|
| index | the positions to read, tagged `KERNEL_IDX` | regular, statically known |
| value | the table being read from (shared) | carries an `IndirectAccess(name)` node at the gathered dimension |
| output | the gathered result | regular, statically known |

The indirection lives in a **load**: `inner_fn` reads the value table at a
runtime-computed index.

### Scatter — indirect on a store

A scatter (`out[i] = src`) is lowered to a **`Scatter` IR node** whose
`inner_fn` reads the source **directly**; the data-dependent destination lives
in `Scatter.output_indexer`:

```python
def inner_fn(index):
    i0, i1, i2 = index
    tmp0 = ops.load(arg2_1, i2 + 512*i1 + 32768*i0)   # reads src directly
    return tmp0
# the indirection is in Scatter.output_indexer, keyed on the `indices` closure
```

Its three tensor arguments mirror the gather, with the roles of value and output
swapped:

| Arg | Role | Device coordinates |
|---|---|---|
| index | the positions to write, tagged `KERNEL_IDX` | regular, statically known |
| src (input) | the values being written | regular — per-core base advances with the split |
| output | the destination table (shared) | carries an `IndirectAccess(name)` node at the scattered dimension |

### The shared model

In both cases the `IndirectAccess(name)` node (see `op_spec.IndirectAccess`)
marks the shared table's **indirect axis** — the dimension whose row is selected
at runtime by the index. That axis has **no iteration-space symbol**; its address
is resolved on the device via `SEGMENT_OFFSETS`, with `maxDimSizes == 1` for that
dimension.

Work division splits the op's iteration space across cores. For a gather over
`x : [M, K, N]` with `i : [Q]`, the iteration space is `{d0 = Q, d1 = K,
d2 = N}` and the output is `[Q, K, N]`. The value's gather axis `M` is **not** an
iteration variable — the iteration dims are the index-entry dim (`Q`) and the
value's data dims (`K`, `N`). A scatter over `out : [M, K, N]` with `i : [Q]` has
the same iteration space `{d0 = Q, d1 = K, d2 = N}`; the destination's scatter
axis `M` is the indirect axis.

### The gather already parallelises along the index — by construction

`get_mem_deps_from_rw`
([pass_utils.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py))
filters indirect reads out of work division's inputs, so the planner only ever
sees the **index read** and the **output write** — both regular tensors. It
naturally distributes cores over their dimensions, which include the index-entry
dim. So index-driven parallelism works for a gather without special handling.

For a scatter : its index-entry dim is **absent** from the
destination's direct coordinates (the row is data-dependent), so it needs an
explicit nudge to become a split axis — see
[fix #2](#2-parallelise-over-the-index-entry-dim) below.

## The gap

The planner ranks dimensions by size and splits the largest. When a shared
table's data dim (`K` or `N`) is the largest — the common case for **wide rows**
(embedding `hidden`, expert weight matrices, attention `head_dim`) — it splits
that dimension. That is wrong, because the table is **shared**: every core must
address any row, so it cannot be sliced per core.

## The scatter-only correctness condition: index uniqueness

Splitting the index-entry dim `Q` hands each core a disjoint set of source rows /
index entries. For a gather the dimension being split is the *output*, so the
slices are trivially disjoint. For a scatter the destination rows are `idx[j]` —
*data-dependent*. Disjoint `Q`-ranges map to disjoint destination rows **if
`idx` is injective**. If two entries on different cores collide
(`idx[j1] == idx[j2]`), they race. Gather has no analogue — reads never conflict.

We therefore enable the index-entry split for **overwrite** scatters
(`Scatter.scatter_mode is None`, e.g. `index_put`): PyTorch already leaves
duplicate-index overwrites non-deterministic, so a per-core split stays within
that contract. **Accumulating** scatters (`atomic_add`, e.g. `scatter_add_`)
would need atomic writes to remain correct under duplicate indices, so they are
left for a single core.

## The proposed fix

Five changes, in order of importance.

### Where the rules live: the work-division constraint framework

The split rules are expressed as a single op constraint in the centralized
work-division constraint framework
([work_division_constraints.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/work_division_constraints.py)).
`indirect_access_constraints(ctx)` returns a `ConstraintResult` with two sets:

* **`forbidden`** — the shared-table data dims that must never be split (fix #1),
  combined with the partial-last-stick rule (fix #5). This is a **hard**
  constraint: unlike the framework's soft `blocked` (which span reduction may
  override to meet the memory-span limit), a `forbidden` dim is removed from the
  span-reduction candidate set too, so it is never split under any circumstance.
* **`force_output`** — a scatter's index-entry dim, promoted to output-split
  priority (fix #2).

`collect_work_division_constraints` merges this with the other op constraints
(coordinate-mask, conv-spatial, QFP8WT), and the split passes consume the merged
result: `span_reduction_pass` feeds `forbidden` into `must_split_vars`' candidate
filter, and `work_distribution_pass` passes `forbidden` / `force_output` into
`_default_split`. The computation itself lives in
[pass_utils.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py)
(`indirect_forbidden_split_syms`, `indirect_store_entry_syms`), so it is reusable
and has no dependency on `work_division.py`.

### 1. Forbid splitting the shared table's data dims — the correctness fix

Both directions are unified behind `_shared_indirect_coords`
([pass_utils.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py)),
which returns the `IndirectAccess`-aware device coordinates of every shared
indirect tensor of an op:

* gather — each indirect value *read*;
* scatter — the indirect destination *write*, applying `indirect_store_subs_from_op`
  so the row axis becomes an `IndirectAccess` and only its data dims remain as
  coordinate symbols.

`shared_indirect_data_syms` then takes the non-`IndirectAccess` coordinate
symbols of those coordinates — the same extraction for both directions. The split
passes consult it through `indirect_forbidden_split_syms`, which combines it with
the partial-last-stick rule ([fix #5](#5-stick-align-the-index-entry-split-partial-last-stick))
and surfaces the result as the `forbidden` set of `indirect_access_constraints`.
`span_reduction_pass` excludes those symbols from `must_split_vars`' candidate set
and `_default_split` removes them from the output and reduction priority lists, so
the core budget is distributed only over the index-entry dims — **divide by the
index, not the table.**

When the index dim cannot absorb all the cores, the op falls back to fewer cores
rather than splitting a data dim. **Correct-but-serial is the intended
trade-off; silent corruption is not.**

### 2. Parallelise over the index-entry dim

A gather's index-entry dim is already an output coordinate, so it is a split axis
by default. A scatter's index-entry dim is **absent** from the destination's
direct coordinates (the row is data-dependent), so `prioritize_dimensions`
classifies it as a *reduction* dim, which a non-reduction op never splits.
`indirect_store_entry_syms` returns those entry dims (the iteration symbols left
after removing the data dims); they become the `force_output` set of
`indirect_access_constraints`, and `_default_split`'s `force_output_syms`
promotes them to output priority so the distributor splits them.

The split round-trips through the existing coefficient encoding without new
machinery: a scatter's entry dim has coefficient 0 in the (indirect) write index,
so `splits_by_index_coeff` encodes it via the first **non-indirect** read — the
direct `src` load selected by `_first_non_indirect_read_index` — and
`apply_splits_from_index_coeff` decodes it the same way at codegen.

This is gated on the [uniqueness condition](#the-scatter-only-correctness-condition-index-uniqueness):
`indirect_store_entry_syms` returns dims only for overwrite scatters.

### 3. Shared-table span guard

The shared table's coordinates carry `IndirectAccess`, so it is visible to the
per-core span check with `get_per_core_span` treating an `IndirectAccess`
coordinate as contributing its full device extent (any core may touch any row)
and never splitting it. A gather's value table is pulled in as an extra TensorDep
(it is not in `args`), via `collect_indirect_value_tds`; a scatter's destination
is the output TensorDep, already covered.

### 4. Deterministic split round-trip

The split plan is encoded with the coefficients of the read/write index
expressions. An indirect read carries data-dependent symbols whose coefficients
are not a stable identity key, so the encode side (`apply_splits`) and both
decode sites (`work_distribution_pass`, `create_op_spec`) prefer the first
**non-indirect** read as the reduction-split reference index, via
`_first_non_indirect_read_index`. This same reference also carries a scatter's
entry-dim split (fix #2).

### 5. Stick-align the index-entry split (partial last stick)

Enabling the index-entry split (fixes #1–2) exposes a second hazard when the
entry count is **not a whole multiple of the index stick** (32 int32 entries per
128-byte stick). Work division splits the entry dim in whole sticks, so an
*even* per-core slice of a partial last stick — e.g. `Q = 40` = one full stick +
8 — hands the second core a slice that **straddles the index stick boundary**.
The backend cannot step a sticked dimension across a stick boundary *within* a core,
so the entries past the boundary are read from the wrong device addresses.

The fix pads the **gather output**'s entry-dim `device_size` up to the next stick
multiple (e.g. `40 → 64`), leaving the logical size unchanged.
`pass_utils.padded_entry_output_stl` owns this — the single coordinate search
(`_find_entry_output_dim`) plus the size math — and returns the grown device
layout (or `None` when there is nothing to pad); the pre-scheduling pass
`enforce_indirect_access_layout._pad_output_for_stick_aligned_split` just applies
it. The physical allocation grows, so the per-core base becomes stick-aligned —
matching the index tensor, whose device layout is already stick-padded — and the
D2H copy extracts the logical rows from the larger allocation (its
`physical_exceeds_logical` path). To keep the per-core base stride consistent,
`superdsc` grows the SDSC iteration to the padded size **before**
`_create_sdsc_tensors` computes strides; otherwise the base lands element-aligned
(mid-stick) and the split still miscompiles.

`indirect_forbidden_split_syms`
([pass_utils.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py))
enforces the invariant, **forbid unless provably padded**, as a single predicate:
it forbids a partial-last-stick entry dim — reading the index's *unpadded*
logical count (`d.ranges`), which the output padding never changes — **unless**
`is_output_stick_aligned_for_entry(op)` confirms the output extent is already a
whole multiple of the index `eps` (either naturally, or because the padding pass
grew it).

So a padded gather splits stick-aligned, while anything unpadded falls back to a
single core rather than miscompiling. A **scatter cannot be padded**: its
destination is written in place (a mutation layout) and its entry row is
data-dependent (an `IndirectAccess` coord), so `_find_entry_output_dim` finds no
paddable entry, `is_output_stick_aligned_for_entry` returns `False`, the
forbiddance is never lifted, and a partial-stick scatter stays on a single core —
correct-but-serial rather than miscompiling.

Stick-aligned counts (a multiple of 32) are unaffected: no padding, the
forbiddance never fires, and they split exactly as before.

## Detection: load side vs. store side

Both directions discover their indirect symbols before scheduling via
`indirect_access_subs_from_op`
([pass_utils.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py)),
which merges:

* `_build_indirect_load_subs` — re-executes `inner_fn` with `_IndirectIndexFinder`
  to learn which buffer's **load** produced each indirect index (gather);
* `_build_indirect_store_subs` — recovers the scatter index buffer from the
  `Scatter.output_indexer` closure (via `_find_scatter_index_buf_names`) and maps
  each indirect symbol in the **write** dep to it (scatter).

Both map an indirect symbol to `IndirectAccess(name)`, which `device_coordinates`
substitutes into the shared table's coordinates.

## Worked examples

### Gather

`out = x[i]` with `x : [128, 64, 512]`, `i : [256]`, `SENCORES=32`. Iteration
space `{d0 = Q = 256, d1 = K = 64, d2 = N = 512}`; the value's gather axis
`M = 128` is addressed by `IndirectAccess`. The index stickifies to `256 / 32 =
8` sticks on `d0`.

| | Without the fix | With the fix |
|---|---|---|
| Largest dim | `K = 64` | (forbidden) |
| Split | `K`, 32 cores | `d0` (index), 8 cores |
| Value tensor | per-core column base diverges | shared at base 0 |
| Result | wrong (every core reads column 0) | correct |

### Scatter

`out[i] = src` with `out, src : [5, 64, 512]`, `i : [5]` (a permutation),
`SENCORES=32`. Iteration space `{d0 = Q = 5, d1 = K = 64, d2 = N = 512}`; the
destination's scatter axis `M = 5` is addressed by `IndirectAccess`. The index
stickifies `d0` to `ceil(5 / 32) = 1` stick.

| | Without the fix | With the fix |
|---|---|---|
| Largest dim | `K = 64` | (forbidden) |
| Split | `K`, 32 cores | `d0` (index), 1 core (Q stickifies to 1) |
| Destination | per-core column base pinned, all cores write columns `[0,2)` | shared at base 0, row from `IndirectAccess` |
| Result | wrong / backend abort | correct (serial — index too small to split) |

In both cases parallelism is set by the index size in sticks:
`cores = core_split(ceil(Q / 32), SENCORES)` for a 1-D index — the largest
divisor of the index's stick count that fits the core budget (`Q = 256 → 8
sticks → 8`, `Q = 1024 → 32`; a partial count like `Q = 40` pads to `2` sticks →
`2` cores ([fix #5](#5-stick-align-the-index-entry-split-partial-last-stick)),
and a non-power-of-two `SENCORES = 6` rounds `8` sticks down to `4`). A spatial
(non-stick) index dimension splits directly. A small index (the scatter `Q = 5`
here) runs correct-but-serial.

## Limitations and future work

* **Parallelism is capped by the index size.** The current implementation
  parallelises only over the index dimension. Workloads with a small, fixed index
  but wide rows — MoE expert gathers (top-k routing), paged-attention KV reads
  (short block tables), wide embedding lookups at small batch — get few cores
  (often 1–2). They are **correct**, but not maximally parallel.

* **Accumulating scatters run serially.** `scatter_add_` / `index_put(...,
  accumulate=True)` are left on a single core because a multi-core split would
  need atomic accumulate to stay correct under duplicate indices. Enabling them
  requires backend atomic-add support.

* **Multi-index scatter is not divided.** When the destination index is built
  from more than one index tensor, `_build_indirect_store_subs` cannot
  unambiguously map each indirect symbol to its source buffer, so it returns no
  subs and the op stays unsplit (correct, serial).

## Implementation

| File | Change |
|---|---|
| `_inductor/pass_utils.py` | `_build_indirect_load_subs`, `_build_indirect_store_subs`, `_wrap_indirect_subs`, `indirect_access_subs_from_op` (merges both), `indirect_store_subs_from_op`, `_first_non_indirect_read_index`; the partial-stick entry accessors (fix #5): `_find_entry_output_dim` (the single coordinate search), `is_output_stick_aligned_for_entry` (the guard's alignment predicate), `padded_entry_output_stl` (returns the grown output layout); the split-rule computation: `_shared_indirect_coords` (gather reads + scatter destination), `shared_indirect_data_syms`, `_non_indirect_coord_syms`, `indirect_forbidden_split_syms` (shared-table dims + partial-stick rule), `indirect_store_entry_syms` |
| `_inductor/work_division_constraints.py` | `indirect_access_constraints` — registers the split rules as one op constraint; `ConstraintResult.forbidden` (hard, never-split) / `.force_output` fields, merged by `collect_work_division_constraints` and consumed by every split pass |
| `_inductor/work_division.py` | consumes `constraint_result.forbidden` / `.force_output` (feeds `forbidden` into `must_split_vars`; `forbidden_split_syms` + `force_output_syms` in `_default_split`); `collect_indirect_value_tds` + `IndirectAccess` span guard; `_resolve_layout` |
| `_inductor/enforce_indirect_access_layout.py` | `_pad_output_for_stick_aligned_split` — applies the grown output layout from `pass_utils.padded_entry_output_stl` to a partial-stick gather output (fix #5) |
| `_inductor/codegen/superdsc.py` | grow the SDSC index-entry iteration to the padded output size before `_create_sdsc_tensors` so the per-core base is stick-aligned (fix #5) |
| `_inductor/spyre_kernel.py` | non-indirect read index in `create_op_spec` |

## See also

* [Work Division Planning](work_division_planning.md) — the three-pass planner
  this builds on.
