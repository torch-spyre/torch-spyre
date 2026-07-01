# Remove `is_reduction` from Coarse-Tile Levels Triple — Design

**Date:** 2026-07-01

---

## Problem

The levels list produced by `_hints_levels` has shape
`list[tuple[int, sympy.Integer, bool]]` — `(hint_id, count, is_reduction)`.
The `is_reduction` boolean is derived from a single "winning" op's `DimHint` and
is used as a **group-wide flag** inside `_stamp_group` to decide whether to call
`_divide_ranges` or `_divide_reduction_ranges` for **every** op in the group.

This is wrong.  In a flash-attention-style group the same hint dimension (Lk)
is:

- An **output dimension** on pointwise ops like `mul`, `sub`, `exp`, `add`
  (shape `[B, H, Lq, Lk]`) — should call `_divide_ranges`.
- A **reduction dimension** on ops like `amax`, `sum`, `batched_matmul`
  (shape `[B, H, Lq]`) — should call `_divide_reduction_ranges`.

When the group-wide flag disagrees with an individual op's reality:

- `is_reduction_level=True` on a pointwise op → `_divide_ranges` is never
  called → the op's output range for Lk is not divided → the op iterates
  over the full Lk per tile.
- `is_reduction_level=False` on a Reduction op → `_divide_reduction_ranges`
  is never called → the op's reduction range for Lk is not divided → the
  reducer sweeps the full Lk per tile.

Both failure modes silently produce wrong codegen.

The root cause is that `is_reduction` is not a group-level property.
It is a per-op, per-dimension property that is already correctly captured in
each op's own `DimHint.is_reduction` field (set by `_assign_dim_hints_impl`
from each op's `reduction_named_dims`).

---

## Fix

### 1. Drop `is_reduction` from the levels tuple

Change the levels list shape from `list[tuple[int, sympy.Integer, bool]]` to
`list[tuple[int, sympy.Integer]]` — `(hint_id, count)`.

**Files and sites that produce levels:**

| Site | Change |
|------|--------|
| `_hints_levels` (coarse_tile.py:135) | Drop `h.is_reduction` from `levels.append(...)` |
| `span_overflow_groups` (coarse_tile.py:482–488) | Drop `is_reduction` from the tuple appended to `levels`; the `DimHint` constructed there keeps `is_reduction=is_reduction` (always `False` for span-overflow) because `DimHint.is_reduction` remains correct per-op ground truth |
| Module docstring (line 37) | Update triple description |
| `coarse_tile()` docstring (line 564) | Update triple description |
| `_stamp_group` docstring (line 1372) | Update triple description |
| `_hints_levels` docstring (line 107) | Update |

### 2. `_stamp_group`: per-op dispatch using the op's own lookup tables

The per-op lookup tables `hint_id_to_ranges_pos` and
`hint_id_to_reduction_ranges_pos` are already built correctly from each op's
own `DimHint.is_reduction`.  Replace the `is_reduction_level` branch with
unconditional lookups in both tables:

**Current loop (lines 1422–1442):**

```python
for hint_id, count, is_reduction_level in levels:
    if is_reduction_level:
        rpos = hint_id_to_reduction_ranges_pos.get(hint_id)
        op_tiled_dims.append([])
        op_tiled_reduction_dims.append([rpos] if rpos is not None else [])
        if isinstance(op.data, Reduction):
            _divide_reduction_ranges(op, count, [rpos] if rpos is not None else [])
    else:
        opos = hint_id_to_ranges_pos.get(hint_id)
        op_tiled_dims.append([opos] if opos is not None else [])
        op_tiled_reduction_dims.append([])
        _divide_ranges(op, count, [opos] if opos is not None else [])
```

**Replacement:**

```python
for hint_id, count in levels:
    opos = hint_id_to_ranges_pos.get(hint_id)
    rpos = hint_id_to_reduction_ranges_pos.get(hint_id)
    op_tiled_dims.append([opos] if opos is not None else [])
    op_tiled_reduction_dims.append([rpos] if rpos is not None else [])
    _divide_ranges(op, count, [opos] if opos is not None else [])
    if isinstance(op.data, Reduction):
        _divide_reduction_ranges(op, count, [rpos] if rpos is not None else [])
```

**Correctness by case:**

| Op type | Dim role for this hint | `opos` | `rpos` | Effect |
|---------|----------------------|--------|--------|--------|
| Pointwise | output dim | found | None | `_divide_ranges` only ✓ |
| Pointwise | broadcast (no matching dim) | None | None | no-op ✓ |
| Reduction | output dim | found | None | `_divide_ranges` only ✓ |
| Reduction | reduction dim | None | found | `_divide_reduction_ranges` only ✓ |
| Reduction | broadcast | None | None | no-op ✓ |

The mixed case (`opos` and `rpos` both non-None at the same level) is already
caught by `_validate_reduction_tiling` and raises `RuntimeError`.

`_divide_ranges` with `tiled_dims=[]` is a no-op (empty loop body).
`_divide_reduction_ranges` is gated by `isinstance(op.data, Reduction)` so it
is never called on a Pointwise op.

### 3. `DimHint.is_reduction` — unchanged

`DimHint.is_reduction` remains the per-op ground truth.  It is set correctly by
`_assign_dim_hints_impl` and is still used to build the per-op lookup tables in
`_stamp_group`.  The INFO-level logging in `hints_to_coarse_tile_groups`
(line 399) also reads it correctly — no change there.

---

## What does NOT change

- `CoarseTileInfo` fields (`loop_tiled_dims`, `loop_tiled_reduction_dims`) —
  unchanged; they are populated correctly by the new per-op dispatch.
- `_validate_reduction_tiling`, `_propagate_tiled_op`,
  `_compute_fill_loop_info` — all read from the stamped `CoarseTileInfo`, not
  from the levels triple; no change needed.
- `DimHint` dataclass — unchanged.
- `_hint_key` — unchanged.
- `hints_to_coarse_tile_groups` grouping logic — unchanged.

---

## Tests

Two new tests added to `TestCoarseTileSpyreHints` in
`tests/inductor/test_coarse_tile_e2e.py`:

### `test_hint_mixed_output_and_reduction_loopspec`

**Purpose:** Direct regression for Bug 2 — `_stamp_group` using the wrong
divide function because of the group-wide flag.

Setup: two ops share a group under `spyre_hint({"Lk": 2})`:
- Op1: pointwise `mul` with Lk as an output dim, shape `[H, Lq, Lk]`
- Op2: `sum` over Lk, shape `[H, Lq]` (Lk is a reduction dim)

Shapes: `H=8, Lq=64, Lk=128` (Lk/2 = 64 elements = 1 stick at fp16 — stick-aligned).

Assertions (source inspection, mocked kernel):
- `LoopSpec(` present
- `count=sympify('2')` present — the Lk loop level is stamped
- The generated source contains two separate OpSpec entries (one for the
  pointwise with `loop_tiled_dims` populated, one for the reduction with
  `loop_tiled_reduction_dims` populated) — verified by checking that
  `tiled_symbols` appears for the pointwise op and `tiled_reduction_symbols`
  (or absence of tiled output symbols) appears for the reduction op.

### `test_hint_flash_attention_two_loop_levels`

**Purpose:** Integration regression — the full flash-attention graph now has
both the H and Lk loop levels stamped, and `_stamp_group` correctly divides
each op's ranges using the per-op lookup.

Same graph as `test_hint_flash_attention_loopspec` (B=1, H=8, Lq=256, Lk=256,
D=64, hints `{B:1}/{H:4}/{Lk:2}`).

Assertions (source inspection, mocked kernel):
- `count=sympify('4')` present (H level)
- `count=sympify('2')` present (Lk level)
- At least 2 `LoopSpec(` occurrences OR the single LoopSpec has nested
  structure — confirms both levels survive into codegen.

---

## Scope

**Modified files:**
- `torch_spyre/_inductor/coarse_tile.py` — `_hints_levels`, `span_overflow_groups`,
  `_stamp_group`, and docstrings at lines 37, 564, 1372

**Modified test file:**
- `tests/inductor/test_coarse_tile_e2e.py` — two new test methods
