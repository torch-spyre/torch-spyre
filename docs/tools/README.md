# Documentation Tools

Scripts that regenerate the real, current compiler output quoted in
`docs/source/`. Compiler internals (IR field names, symbol naming,
generated OpSpec/bundle.mlir text) change frequently; these scripts exist
so that refreshing a doc's example means re-running a script and diffing,
not hand-editing stale snippets from memory.

## `capture_for_each_tile_ir.py`

Regenerates
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

This produces three sections: the
`graph.operations` IR dump, the generated OpSpec/LoopSpec Python wrapper
source, and the generated `bundle.mlir` — for the single-level loop
`for_each_tile` stamps directly via `_stamp_direct_loop_info`.

### Backend-compiler mock

This script mocks `subprocess.run` with `_fake_backend_compiler`, which
writes an empty `spyrecode.json` directly instead of invoking the real
`dbo-opt` binary — a local reimplementation of the test suite's
`mock_backend_compiler()` fixture (`tests/inductor/utils_inductor.py`), kept as a local copy here since
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
| `--sencores` | `4` | `SENCORES` value (kept small so the dumped IR stays readable) |
| `--debug` | off | Log at `DEBUG` instead of `INFO`; combine with `SPYRE_LOG_PASSES` for per-pass dumps, e.g. `SPYRE_LOG_PASSES=splice_while_loops` to see `loop_info` immediately after `for_each_tile`'s own stamping pass |

Changing the shape flags means
re-deriving every quoted numeric value in the doc from a fresh capture,
not patching individual numbers by hand.
