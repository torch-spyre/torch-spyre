# Debugging compiled artifacts

## Two cache layers, and why clearing one isn't enough

Inductor's on-disk cache under `/tmp/torchinductor_<user>/` has two layers
that get conflated:

1. **`fxgraph/`** — the FX graph cache, keyed by pass UUID hashes. Cleared
   by `torch._inductor.codecache.FxGraphCache.clear()`.
2. **Compiled wrapper `.py` files**, stored in two-character subdirectories
   (e.g. `yl/cylvm3....py`). These are the actual generated Python wrapper
   modules. `FxGraphCache.clear()` does **not** clear this layer — if a
   matching wrapper file is still on disk, Inductor imports it directly and
   skips codegen entirely.

If you change a compiler pass and a test still shows the old `LoopSpec`
nesting, iteration space, or wrapper structure despite the change, suspect
a stale layer-2 wrapper file before suspecting your code. Symptoms look
like "my pass isn't running" even when it is.

**Force a true full recompile:**

```bash
rm -rf /tmp/torchinductor_<user>/
```

Clearing only `fxgraph/` (`rm -rf /tmp/torchinductor_<user>/fxgraph/`) is
not sufficient when the wrapper-file layer is what's stale.

`output_code.py` (see below) is ground truth for what the *current* source
actually produces — if it disagrees with what a test run just did, that's
the cache, not a bug in your pass.

## Where to find generated artifacts

Set `TORCH_COMPILE_DEBUG=1` to get a full dump per compilation under:

```
torch_compile_debug/run_<timestamp>-pid_<pid>/torchinductor/<graph_name>/output_code.py
```

`output_code.py` is the generated Python wrapper — the single most useful
artifact for confirming what codegen actually produced, independent of any
cache question.

The generated SuperDSC bundle (MLIR passed to the back-end compiler) lives
under the Inductor cache itself:

```
/tmp/torchinductor_<user>/inductor-spyre/sdsc_<op names>_<hash>/bundle.mlir
```

The directory name is derived from the fused op names in that kernel, so
it's greppable by op name when you're not sure which hash directory to
look in.

For a full worked example of what a coarse-tiled kernel's generated
`OpSpec` (Python wrapper source) and `bundle.mlir` actually look like side
by side, see the "Generated OpSpec" and "Generated `bundle.mlir`" sections
in
[`docs/source/compiler/coarse_tiling_loops.md`](../../../../../../docs/source/compiler/coarse_tiling_loops.md).

## Provenance tracking

`SpyreGraphTransformObserver` (`provenance.py`) tracks whether passes drop
buffer origin/provenance info across the pass pipeline. It's disabled by
default; enable with `INDUCTOR_PROVENANCE=1`, or it follows
`TORCH_COMPILE_DEBUG=1` when `INDUCTOR_PROVENANCE` is unset. Set
`INDUCTOR_PROVENANCE=0` to force it off explicitly. `INDUCTOR_PROVENANCE`
itself is parsed by upstream PyTorch Inductor's config layer, not by
torch-spyre — torch-spyre's `provenance.py` only reads the resulting
`trace.provenance_tracking_level` config value, so `grep INDUCTOR_PROVENANCE
torch_spyre/` turning up nothing is expected. See
[`docs/source/compiler/adding_operations.md`](../../../../../../docs/source/compiler/adding_operations.md)
for the full contract (`SOURCELESS_CREATION_PASSES`,
`INTENTIONAL_PROVENANCE_REMAP_PASSES`,
`INTENTIONAL_PROVENANCE_REMOVAL_PASSES`).
