# PR: Coarse-Tiling Loop IR for the Spyre Backend

## Summary

This PR implements the end-to-end infrastructure for **coarse-level tiling
loops** in the Spyre compilation pipeline.  Tiling is the key program
transformation for *working set reduction*: by splitting a large computation
into K time-domain chunks, most tensors can fit in scratchpad instead of HBM,
dramatically reducing off-chip traffic and enabling effective scratchpad
utilization.

The design is structured as three cooperating layers:

| Layer | Component | Role |
|---|---|---|
| 1 | `CustomPreSchedulingPasses` / `coarse_tile()` | Stamps loop metadata on `ir.Operation` objects and rewrites ranges |
| 2 | `build_loop_scheduler_nodes` / `CountedLoopSchedulerNode` | Groups stamped ops into a counted loop node visible to the scheduler |
| 3 | `LoopSpec` + codegen | Serializes the loop structure into `bundle.mlir` as `scf.for` loops with `affine.apply` address arithmetic |

---

## Small Example

Two chained pointwise ops (`add` then `mul`) over `[1024, 4096]` tensors,
both placed in a single tiling group with K=2 outer / M=4 inner:

```python
def f(a, b, c):
    with spyre_hint(tiles={"K": 2}):     # outer: 2 × 512 rows
        with spyre_hint(tiles={"M": 4}): # inner: 4 × 1024 columns
            y = a + b
            z = y * c
    return z
```

Both ops are stamped with `loop_group_id=(0,0)` / `loop_count=[2,4]` and
appear together in the innermost `LoopSpec` body.  Each tile is 512 × 1024
elements — the intermediate result `y` stays in scratchpad across both
dispatches within the same tile.  The generated MLIR:

```mlir
scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {   // K=2 outer
  scf.for %i_1 = %c0 to %loop_bound_1 step %c1 { // M=4 inner
    %addr_0 = affine.apply #map_0(%i_0, %i_1)[%sym_1]
    sdscbundle.sdsc_execute (%addr_0) { sdsc_filename="sdsc_0.json", ... }  // add
    %addr_1 = affine.apply #map_0(%i_0, %i_1)[%sym_2]
    sdscbundle.sdsc_execute (%addr_1) { sdsc_filename="sdsc_1.json", ... }  // mul
  }
}
```

where `#map_0 = affine_map<(d0, d1)[s0] -> (s0 + 4194304*d0 + 2048*d1)>`.

---

## Layer 1 — Pre-Scheduling IR Pass (`coarse_tile.py`)

**New file:** `torch_spyre/_inductor/coarse_tile.py`

- `coarse_tile(ops, groups, tiled_dims)` is the public API.  Each group is
  `(ops, loop_count)` or `(ops, loop_count, tiled_dims)`.
- Stamps three attributes on each `ir.Operation`:
  - `loop_group_id: tuple[int, ...]` — nesting path (e.g. `(0,)`, `(0, 0)`)
  - `loop_count: list[sympy.Expr]` — trip counts outermost-first
  - `loop_tiled_dims: list[list[int]]` — which tensor dims are tiled at each level
- Divides the tiled iteration-space range of each op by its loop trip count,
  so the body op sees the per-iteration (reduced) working set.
- Plugged into `CustomPreSchedulingPasses` via `config.coarse_tiling_groups_fn`.

**Config additions (`config.py`):**
- `coarse_tiling: bool` — master enable (env `COARSE_TILING=1`)
- `coarse_tiling_groups_fn: Optional[Callable]` — injected group-detection function
- `bundle_hbm_symbols: bool` — gate for symbol-based HBM addressing (see below)

---

## Layer 2 — Scheduler Pass (`scheduler.py`)

**New class: `CountedLoopSchedulerNode`**

A `GroupedSchedulerNode` subclass wrapping a list of inner `SchedulerNode`s
with a loop trip count.  `can_fuse` returns `False` to prevent Inductor's
fusion pass from merging across loop boundaries.

**New function: `build_loop_scheduler_nodes(nodes)`**

Post-fusion pass that scans `list[BaseSchedulerNode]` for runs sharing a
`loop_group_id` and wraps them in `CountedLoopSchedulerNode`.  Handles
arbitrary nesting depth: inner groups are recursively wrapped before the
outer group is created.

Registered in `CustomPostFusionPasses` and runs before `spyre_fuse_nodes`.

---

## Layer 3 — LoopSpec & Codegen

### `op_spec.py` — `LoopSpec` data structure

```python
@dataclasses.dataclass
class LoopSpec:
    count: sympy.Expr
    body: list[OpSpec | LoopSpec]
```

Each `OpSpec` gains a `tiled_symbols: list[Symbol]` field listing the
iteration-space symbols whose corresponding tensor dimension is tiled by the
enclosing loop.

### `spyre_kernel.py` — codegen integration

`SuperDSCScheduling.codegen_node()` now handles `CountedLoopSchedulerNode`:
drives `SpyreKernel` for the inner ops and wraps the resulting `OpSpec`s in a
`LoopSpec`.  `SpyreKernel` propagates `loop_tiled_dims` into `tiled_symbols`
on each `OpSpec`.

### `codegen/compute_ops.py` — symbol-based HBM addressing

`generate_sdsc()` gains a `use_symbols: bool` parameter.

- **`use_symbols=True`**: HBM tensor addresses are registered as negative
  symbol IDs in the SDSC JSON and emitted as `%sym_N` constants in
  `bundle.mlir`.  Per-op `affine_strides` dict drives `affine.apply`
  computation for tiled dimensions.
- **`use_symbols=False`** (default): concrete per-core addresses baked
  directly into the JSON — identical to main-branch behaviour.

### `codegen/bundle.py` — `bundle.mlir` generation

`generate_bundle()` now handles `LoopSpec` entries:

- **Pass 1** compiles all `OpSpec` leaves depth-first, collecting symbol
  values and affine strides.
- **Pass 2** emits `bundle.mlir`: module-level `#map_N` affine map
  definitions (deduplicated), loop bound constants, `%sym_N` constants,
  and nested `scf.for` loops with `affine.apply` per tiled tensor.
- **Guard**: if `use_symbols=False` and the spec tree contains tiled ops
  inside a `LoopSpec`, raises `RuntimeError` rather than silently
  miscompiling.

---

## Symbol Gating (`bundle_hbm_symbols`)

Backend compiler support for the `sdscbundle` symbol table is still under
development.  The symbol-indirection path is therefore **opt-in**:

```bash
BUNDLE_HBM_SYMBOLS=1  # or config.bundle_hbm_symbols = True
```

When the flag is False (default):
- HBM addresses are concrete integers in the SDSC JSON — main-branch behaviour.
- Tiled ops inside a `LoopSpec` raise `RuntimeError` (misconfiguration guard).

When the flag is True:
- `%sym_N` constants and `affine.apply` are emitted in `bundle.mlir`.
- Full coarse-tiling address arithmetic is active.

`COARSE_TILING=1` requires `BUNDLE_HBM_SYMBOLS=1`.

---

## Tests

Six new test files, all mock-based (no device required):

| File | What it tests |
|---|---|
| `test_loop_spec.py` | `LoopSpec` / `OpSpec` data structures and serialization |
| `test_coarse_tile_pass.py` | `coarse_tile()` IR pass: range rewriting, attribute stamping, nested groups |
| `test_counted_loop_node.py` | `CountedLoopSchedulerNode` and `build_loop_scheduler_nodes` |
| `test_sdsc_tiled_address.py` | `generate_sdsc` and `compile_op_spec` symbol/affine-stride paths |
| `test_bundle_loop.py` | `generate_bundle` MLIR output: loop structure, affine maps, symbol constants |
| `test_coarse_tile_e2e.py` | End-to-end: `coarse_tile()` → `OpSpec` / `LoopSpec` wiring through codegen |

CI config YAMLs and workflow matrix entries added for all six.

---

## Files Changed

| File | Change |
|---|---|
| `torch_spyre/_inductor/coarse_tile.py` | **New** — Layer 1 IR pass |
| `torch_spyre/_inductor/scheduler.py` | `CountedLoopSchedulerNode`, `build_loop_scheduler_nodes` |
| `torch_spyre/_inductor/op_spec.py` | `LoopSpec` dataclass; `tiled_symbols` field on `OpSpec` |
| `torch_spyre/_inductor/spyre_kernel.py` | `codegen_node` handles `CountedLoopSchedulerNode` |
| `torch_spyre/_inductor/passes.py` | Register `build_loop_scheduler_nodes` in post-fusion passes |
| `torch_spyre/_inductor/wrapper.py` | Pass `LoopSpec` entries through kernel wrapper |
| `torch_spyre/_inductor/config.py` | `coarse_tiling`, `coarse_tiling_groups_fn`, `bundle_hbm_symbols` |
| `torch_spyre/_inductor/codegen/compute_ops.py` | `use_symbols` parameter; concrete-address path |
| `torch_spyre/_inductor/codegen/superdsc.py` | Thread `use_symbols` through `compile_op_spec` |
| `torch_spyre/_inductor/codegen/bundle.py` | Full `LoopSpec`-aware MLIR generation |
| `torch_spyre/execution/async_compile.py` | Pass `bundle_hbm_symbols` config to `generate_bundle` |
| `docs/source/compiler/coarse_tiling_loops.md` | Design doc with motivating example |
| `tests/inductor/test_*.py` (×6) | New test suites |
| `tests/configs/torch_spyre_tests/inductor/*.yaml` (×6) | CI config files |
| `.github/workflows/torch_spyre_tests.yaml` | Workflow matrix entries |

---

## What's Not In This PR

- **`spyre_hint` user API** (PR #2226): the motivating example uses it as
  intended syntax, but the backing test uses the `coarse_tile()` API directly.
- **Group-detection heuristic**: `coarse_tiling_groups_fn` is injected by the
  caller; no automatic group detection is included here.
- **Backend symbol-table support**: the `bundle_hbm_symbols=True` path is
  complete on this side; the backend compiler work is tracked separately.
