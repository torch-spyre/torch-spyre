# Documentation Tools

Scripts that regenerate the real, current compiler output quoted in
`docs/source/`. Compiler internals (IR field names, symbol naming,
generated OpSpec/bundle.mlir text) change frequently; these scripts exist
so that refreshing a doc's example means re-running a script and diffing,
not hand-editing stale snippets from memory.

## `capture_coarse_tile_ir.py`

Regenerates every IR/OpSpec/`bundle.mlir` snippet quoted in
[`docs/source/compiler/coarse_tiling_loops.md`](../source/compiler/coarse_tiling_loops.md)'s
"Small Example" section. It runs the same computation as the e2e test
`test_hint_nested_loop_with_scratchpad` (nested `spyre_hint` tiling over a
two-input add followed by a multiply) standalone via `torch.compile`, with
the pass-pipeline logger raised so the real, current IR at each stage is
printed instead of just the compiled result.

### Usage

```bash
# Always clear the Inductor cache first — see "Gotcha" below.
rm -rf /tmp/torchinductor_$USER

python3 docs/tools/capture_coarse_tile_ir.py > /tmp/coarse_tile_capture.txt 2>&1
```

This produces, in order:

1. `graph.operations` IR dump — the `AFTER PRE-SCHEDULING` snapshot from
   `CustomPreSchedulingPasses`, in the same format `format_operations()`
   produces (real `layout=`, `allocation=`, `op_it_space_splits=`,
   `dim_hints=`, `loop_info=CoarseTileInfo(...)`, and the literal
   `Pointwise`/`Reduction` `inner_fn` source).
2. The generated OpSpec/LoopSpec Python wrapper source (what
   `codegen_kernel()` emits) — `LoopSpec`/`OpSpec` objects with real
   `tiled_symbols`, `tiled_symbol_trip_counts`, `device_tile_advance_expr`,
   and `TensorArg` allocation info.
3. The generated `bundle.mlir` MLIR text.

Diff the relevant section of `/tmp/coarse_tile_capture.txt` against the doc
and update the doc's prose/snippets to match reality — do not hand-edit the
doc's existing snippets by inference; the compiler internals drift often
enough that guessing is how the doc goes stale in the first place.

Like `capture_for_each_tile_ir.py` below, this script mocks `subprocess.run`
with a local `_fake_backend_compiler` rather than invoking the real
`dbo-opt` — see that script's "Backend-compiler mock" section for why a bare
mock with no side effect is not sufficient.

### Capturing a single pass's output

To see the `CoarseTileInfo` state immediately after `coarse_tile()` stamps
`loop_info` — before `split_multi_ops`, stickification, work division, or
scratchpad planning touch anything — pass `--debug` together with
`SPYRE_LOG_PASSES` set to the pass name (or `all`/`1` for every pass):

```bash
rm -rf /tmp/torchinductor_$USER
SPYRE_LOG_PASSES=_maybe_coarse_tile_hints \
    python3 docs/tools/capture_coarse_tile_ir.py --debug \
    > /tmp/coarse_tile_debug_capture.txt 2>&1
```

Look for the `AFTER <pass_name>` line. This is the mechanism
`_should_log_pass` in `torch_spyre/_inductor/passes.py` gates on.

### Options

The example's shape is a CLI flag, not a hardcoded constant — useful if a
future doc revision wants a different tile count or tensor size:

| Flag | Default | Meaning |
|---|---|---|
| `--outer-tiles` | `2` | `num_tiles_per_dim` for the outer `spyre_hint` (dim A) |
| `--inner-tiles` | `4` | `num_tiles_per_dim` for the inner `spyre_hint` (dim B) |
| `--size-a` | `1024` | Size of dim A |
| `--size-b` | `4096` | Size of dim B |
| `--sencores` | `4` | `SENCORES` value (kept small so `bundle.mlir`'s per-core address expansion stays quotable in the doc — see the "Generated `bundle.mlir`" section's note on why `sencores=4` rather than the real default of 32) |
| `--debug` | off | Log at `DEBUG` instead of `INFO`; combine with `SPYRE_LOG_PASSES` for per-pass dumps |

If you change `--outer-tiles`/`--inner-tiles`/`--size-a`/`--size-b`/`--sencores`
from the defaults, every numeric value quoted throughout the doc's Small
Example section (tile shapes, byte strides, core offsets, loop bounds) will
no longer match — re-derive them all from the new capture, don't just patch
the one number you changed.

### Gotcha: stale Inductor cache silently suppresses the dump

Inductor's fxgraph cache lives at `/tmp/torchinductor_$USER`. If a prior run
already compiled the exact same graph with the exact same config, a second
run **skips recompilation entirely** — including the pass-pipeline logging
this script depends on — and you get a mostly-empty capture with no error.
Always `rm -rf /tmp/torchinductor_$USER` before a run whose *logging
output* matters, not just its return value — this is the same class of
gotcha as stale test failures from the fxgraph cache; it bites capture
scripts exactly like it bites test reruns.

## `capture_for_each_tile_ir.py`

The `for_each_tile` analog of `capture_coarse_tile_ir.py` above: regenerates
every IR/OpSpec/`bundle.mlir` snippet quoted in
[`docs/source/compiler/coarse_tiling_loops.md`](../source/compiler/coarse_tiling_loops.md)'s
"Small Example" section for the `for_each_tile`-driven version of the
example (`y = a + b; z = y * c`, tiled along dim 0 with `for_each_tile`
instead of nested `spyre_hint` scopes).

### Usage

```bash
rm -rf /tmp/torchinductor_$USER

python3 docs/tools/capture_for_each_tile_ir.py > /tmp/for_each_tile_capture.txt 2>&1
```

This produces the same three sections as `capture_coarse_tile_ir.py` — the
`graph.operations` IR dump, the generated OpSpec/LoopSpec Python wrapper
source, and the generated `bundle.mlir` — but for the single-level loop
`for_each_tile` stamps directly via `_stamp_direct_loop_info`, rather than
the two-level nest `spyre_hint` produces.

### Backend-compiler mock

This script mocks `subprocess.run` with `_fake_backend_compiler`, which
writes an empty `spyrecode.json` directly instead of invoking the real
`dbo-opt` binary — the same local reimplementation of the test suite's
`mock_backend_compiler()` fixture (`tests/inductor/utils_inductor.py`) that
`capture_coarse_tile_ir.py` uses, kept as a local copy here since
`docs/tools/` cannot import from `tests/`. Since PR #4708 made the
`spyreCodeDir/spyrecode.json` artifact check unconditional — a real backend
compiler can exit 0 without writing it, so `_run_backend_compiler` never
trusted the exit code alone — a bare `mock_patch("subprocess.run")` with no
side effect fails that check even though the mocked subprocess "succeeded."
`_fake_backend_compiler` writes the artifact itself so the capture can reach
`bundle.mlir` generation without needing a working `dbo-opt` binary at all.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--tile-size` | `128` | `for_each_tile`'s `tile_size` |
| `--size-a` | `1024` | Size of dim 0 |
| `--size-b` | `4096` | Size of dim 1 |
| `--sencores` | `4` | `SENCORES` value (kept small for the same reason as `capture_coarse_tile_ir.py` — see its Options section) |
| `--debug` | off | Log at `DEBUG` instead of `INFO`; combine with `SPYRE_LOG_PASSES` for per-pass dumps, e.g. `SPYRE_LOG_PASSES=splice_while_loops` to see `loop_info` immediately after `for_each_tile`'s own stamping pass |

As with `capture_coarse_tile_ir.py`, changing the shape flags means
re-deriving every quoted numeric value in the doc from a fresh capture,
not patching individual numbers by hand.
