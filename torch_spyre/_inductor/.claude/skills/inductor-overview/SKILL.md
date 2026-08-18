---
name: inductor-overview
description: "Guidance specific to torch_spyre/_inductor: compilation pipeline internals, pass ordering, and conventions for code changes in this subtree. Use when working on files under torch_spyre/_inductor/."
---

# torch_spyre/_inductor Guidance

This skill is scoped to `torch_spyre/_inductor/` and is discovered
automatically by Claude Code for work under this subtree, independent of the
top-level `.claude/skills/` directory. It is owned and maintained by the
CODEOWNERS of this subtree.

For repo-wide context (Spyre hardware, device registration, general
compilation pipeline overview), see the top-level `project-overview` skill
first if you haven't already.

## Debugging a compilation

Inductor caches compiled artifacts under `/tmp/torchinductor_<user>/` in
two layers, and only one of them is cleared by `FxGraphCache.clear()`. If
a pass change doesn't seem to take effect — wrong `LoopSpec`/wrapper
structure persists — suspect the stale wrapper-`.py` cache layer before
your code. See
[`references/debugging-compiled-artifacts.md`](references/debugging-compiled-artifacts.md)
for the cache-layer breakdown and where to find generated SDSC/superdsc
JSON, MLIR bundles, and `output_code.py` for a given compilation.

## Layout and stride semantics

`stride_map`, `device_stride`, and `pytorch_stride` measure three
different things and are easy to conflate — and different ops in the same
kernel can legitimately commit to different device layouts for the same
logical dimension (not a bug). See
[`references/layout-and-stride-semantics.md`](references/layout-and-stride-semantics.md)
for worked before/after-restickify examples and the general formula for
computing `stride_map` from a device dim's coordinate expression.

## Terminology: `hbm_pool` is not scratchpad

`allocation={'hbm_pool': ...}` in generated `TensorArg`/`OpSpec` output
means a bulk-allocated **HBM** region (`memory_planning.py`'s
`INTERMEDIATES_SEGMENT`) for tensors used within a single kernel — it is
*not* scratchpad. Only `allocation={'lx': ...}` is actual on-chip
scratchpad memory. (`hbm_pool` was renamed from the older key name `pool`;
if you see `'pool'` in older docs or history, it refers to the same thing.)
Whether a buffer's address is pinned (no `affine.apply` needed) or needs
per-iteration `affine.apply` addressing is determined by
`loop_info.output_tiled_dims`/`tiled_dims_per_read` and
`TensorArg.device_tile_advance_expr` — not a standalone flag. (An older
`per_tile_fixed` boolean duplicated this decision and was removed; if you
see it mentioned in older docs or history, it's the same "pinned vs.
advancing" distinction now derived from `loop_info`/
`device_tile_advance_expr` directly.) `lx` buffers are typically pinned;
`hbm_pool` buffers are ordinary HBM addresses that usually still need
per-iteration addressing like any other HBM operand — but check
`loop_info`/`device_tile_advance_expr` for the specific buffer rather than
assuming from the allocation kind alone.

## `WrapperHandler` swaps must account for stride, not just name

CLAUDE.md's "wrap, never reconstruct" rule for `ComputedBuffer.inner_fn`
covers *why* to use a `WrapperHandler`. The failure mode to watch for once
you're using one: a plain `NameSwapHandler`-style rename forwards a
consumer's load index **unmodified**. That's correct when the old and new
buffers are addressing-equivalent, but silently computes wrong addresses
when they have different strides for the same dimension (e.g. redirecting
a consumer from a tile-local scratch buffer to a full-size buffer after
promoting the consumer's own iteration space). `_patch_consumers` in
`wsr/coarse_tile.py` hit exactly this bug historically and now carries the
fix as its documented pattern: it computes a stride-coefficient rewrite
via `_retile_load_index` (parameterized by a `_RetiledBufferInfo`) and
applies it through `_NameAndIndexSwapHandler` whenever the old and new
buffer strides differ, falling back to plain `NameSwapHandler` only when
they don't. Any new pass that swaps a buffer name/identity under a
consumer's `inner_fn` should check whether this applies — a bare rename
is only safe when the swap is addressing-equivalent.

## Test execution conventions

Never run Spyre tests in parallel (the device is exclusive to one
process), and don't run the full `tests/` suite locally as a pre-push
check — CI covers it. See
[`references/testing-conventions.md`](references/testing-conventions.md)
for the specific suites to run locally instead.

## Adding to this skill

Follow the top-level `CLAUDE.md` conventions for `SKILL.md` frontmatter:
a quoted single-line `description`, not a multi-line `>-` block scalar.
