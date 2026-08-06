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
Buffers in `lx` are `per_tile_fixed=True` (pinned address, no
`affine.apply` needed); `hbm_pool` buffers are ordinary HBM addresses that
still need per-iteration `affine.apply` addressing like any other HBM
operand.

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
map (`_stride_rewrite_map`) and applies it via
`_retile_load_index_from_strides`/`_NameAndIndexSwapHandler` whenever the
old and new buffer strides differ, falling back to plain `NameSwapHandler`
only when they don't. Any new pass that swaps a buffer name/identity under
a consumer's `inner_fn` should check whether this applies — a bare rename
is only safe when the swap is addressing-equivalent.

## Test execution conventions

Never run Spyre tests in parallel (the device is exclusive to one
process), and don't run the full `tests/` suite locally as a pre-push
check — CI covers it. See
[`references/testing-conventions.md`](references/testing-conventions.md)
for the specific suites to run locally instead.

## Adding to this skill

- Keep guidance specific to `torch_spyre/_inductor/` here. Repo-wide
  conventions (license headers, commit signing, `import regex`, line length)
  stay in the top-level `CLAUDE.md` — don't duplicate them.
- Follow the top-level `CLAUDE.md` conventions for `SKILL.md` frontmatter:
  a quoted single-line `description`, not a multi-line `>-` block scalar.
- Companion reference files for this skill (decision trees, checklists,
  templates) belong alongside this file, under
  `torch_spyre/_inductor/.claude/skills/inductor-overview/`.
