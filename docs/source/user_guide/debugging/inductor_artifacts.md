# Inductor Debug Artifacts

**Stack:** torch-spyre (new, Inductor-based).

`TORCH_COMPILE_DEBUG=1` causes `torch.compile` to dump the intermediate
representation of each compiled function to disk. These artifacts are
the primary tool for answering *"what did the compiler actually do with
my model?"*.

:::{tip}
For a step-by-step debugging workflow that uses these artifacts (plus
the `sendnn` backend bisect), see [Debugging](index.md). This page is a
reference for the artifact layout itself.
:::

## Enabling artifact dumps

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCH_COMPILE_DEBUG=1 \
python my_script.py
```

`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` is important — without it,
Inductor will reuse cached compilation results and no new artifacts
are written.

Artifacts land under `/tmp/torchinductor_<user>/` or
`./torch_compile_debug/` depending on the PyTorch version.

## Directory layout

```
torch_compile_debug/
└── run_<timestamp>-pid_<pid>/
    ├── torchdynamo/
    │   └── debug.log
    └── torchinductor/
        ├── aot_model___0_debug.log
        └── model__0_inference_0.0/
            ├── fx_graph_readable.py                            ← ATen graph (human-readable)
            ├── fx_graph_runnable.py                            ← self-contained runnable graph
            ├── fx_graph_transformed.py                         ← FX graph after Inductor passes
            ├── inductor_provenance_tracking_node_mappings.json ← IR ↔ source mapping
            ├── ir_pre_fusion.txt                               ← LoopLevelIR before fusion
            ├── ir_post_fusion.txt                              ← LoopLevelIR after fusion
            ├── output_code.py                                  ← generated host code
            └── sdsc_<index>.json                               ← per-kernel specs fed to DeepTools backend
```

## What each layer tells you

### `fx_graph_readable.py`

The ATen graph after Dynamo capture. Reading it answers:

- Is the operation you expect actually present, or was it decomposed?
- Are input shapes and dtypes what you expect?
- Did any unwanted decomposition change semantics?

### `fx_graph_transformed.py`

The FX graph after Inductor's pre-grad and post-grad passes (padding
insertion, fusion hints, etc.). Diff this against
`fx_graph_readable.py` to see what the frontend passes changed.

### `ir_pre_fusion.txt` / `ir_post_fusion.txt`

LoopLevelIR — nested loops with buffer shapes and strides. Reading it
answers:

- Do loop ranges match the tensor sizes *including padding*?
- Did fusion happen where you expected?

Mismatches here typically indicate a bug in Inductor lowering or in
stickification.

### `sdsc_<index>.json`

The final specifications handed to the DeepTools back-end — one
`sdsc_<index>.json` per compiled kernel in the graph (e.g.,
`sdsc_0.json`, `sdsc_1.json`, …), indexed in lowering order. Each file
encodes:

- Op name (e.g., `clone`, `bmm`, `layernorm`)
- Input/output tensor layouts (`device_size`, `stride_map`,
  `device_dtype`)
- Work division (how cores split the op)
- Scratchpad allocations

Bugs that only show up in the final output frequently trace back to one
of these files — when a kernel produces the wrong numeric result, find
the corresponding `sdsc_<index>.json` (cross-reference `output_code.py`
to map kernel index → op) and inspect it first.

### `inductor_provenance_tracking_node_mappings.json`

When `INDUCTOR_PROVENANCE=1` is also set, this JSON records the
mapping between IR nodes and the original source ops. Combined with
[`tlparse`](https://github.com/pytorch/tlparse) this gives you an
HTML visualisation of how each source op flowed through the pipeline.

### `output_code.py`

The generated host code — what actually runs on CPU to launch the
compiled kernels. Look here when you suspect launch overhead or host-
side glue is on the critical path.

## Inductor provenance tracking

`INDUCTOR_PROVENANCE=1` + `TORCH_TRACE=<dir>` produces a trace log that
[`tlparse`](https://pypi.org/project/tlparse/) renders into a
three-stage HTML viewer showing how each source op is transformed
through Inductor.

```bash
pip install tlparse
```

```bash
TORCH_TRACE=~/my_trace_log_dir \
INDUCTOR_PROVENANCE=1 \
python my_script.py

tlparse log_file_name.log --inductor-provenance
```

Known limitation: the post-grad panel renders empty when the program
contains only a single operator, and link-highlighting may break in
that case. See the
[PyTorch provenance docs][pytorch-provenance] for the upstream
reference.

## Quick reference

```bash
# Dump everything for a minimal reproducer
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCH_SPYRE_DEBUG=1 \
TORCH_COMPILE_DEBUG=1 \
INDUCTOR_PROVENANCE=1 \
python my_reproducer.py

# Locate the artifacts
find . -name "sdsc_*.json" 2>/dev/null
find /tmp -name "fx_graph_readable.py" 2>/dev/null
```

## See also

- [Debugging](index.md) — step-by-step use of these artifacts during a
  bug investigation
- [PyTorch Inductor Provenance docs][pytorch-provenance]

[pytorch-provenance]: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_inductor_provenance.html
