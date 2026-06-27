# Source-to-Kernel Provenance Audit

Phase 1 tooling for [torch-spyre#2574](https://github.com/torch-spyre/torch-spyre/issues/2574).
It traces a `SimpleMLP` through the Spyre compilation pipeline and records, at
each stage, which provenance fields are present on the **actual compile-path
objects** — producing a measurement-only Markdown report.

## What it measures

Six stages, mapped to [issue #2574](https://github.com/torch-spyre/torch-spyre/issues/2574):

1. PyTorch model source
2. FX graph (pre-grad and post-grad)
3. Inductor passes
4. LoopLevelIR
5. OpSpec
6. SuperDSC JSON

For each stage it records which fields are present and their observed type:

- **FX node meta:** `stack_trace`, `nn_module_stack`, `source_fn_stack`,
  `original_aten`, `from_node`
- **IR / LoopLevelIR attributes:** `origins`, `origin_node`, `traceback`,
  `get_stack_traces()`

The report is **measurement-only** — counts and presence computed from the run,
no hand-written analysis. Interpretation is a separate deliverable.

## How it works

Everything is observed in-process during **one** `torch.compile`, via read-only
monkey-patches installed for the duration of the call. No `torch.export` (a
separate front-end, not on the compile path), no iteration-space heuristics, no
`/tmp` globbing. Each stage reads the real object:

| Stage | Hook (class-level, in `captures.py`) | Reads |
| --- | --- | --- |
| 2 pre-grad FX | `CustomPreGradPasses.__call__` | `node.meta` fields |
| 2 post-grad FX | `CustomPostPasses.__call__` | `node.meta` fields |
| 3 passes + 4 LoopLevelIR | `CustomPreSchedulingPasses.__call__` | `graph.operations[*]` origins/origin_node/traceback/get_stack_traces (value **and** `hasattr`), before & after the pass list |
| 5 OpSpec | `SpyreKernel.create_op_spec` | input `ComputedBuffer.origins`; `OpSpec` declared fields + per-instance provenance-field population on the returned `OpSpec` |
| 6 SuperDSC | `SuperDSCScheduling.define_kernel` + `async_compile.get_output_dir` | `kernel_name`, per-kernel origins, `get_kernel_metadata`, exact bundle dirs |

Stage 6's JSON is then read by `superdsc.py` from the **exact** per-kernel
output directories captured at compile time (keyed by `kernel_name`) — not a
guessed location.

Caches are force-disabled (`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` +
`force_disable_caches` + `torch._dynamo.reset()`) so scheduling and codegen
actually run; a cache hit would silently skip `create_op_spec`/`define_kernel`.

## Layout

```
provenance/
├── audit.py             # entry point: one compile → capture → bundle read → report
├── reference_mlp.py     # the model under audit (the subject, not tooling)
├── README.md
└── pipeline/
    ├── captures.py      # all six in-process stage hooks (one context manager)
    ├── superdsc.py      # Stage 6: read sdsc_*.json from captured exact dirs
    └── report.py        # measurement-only Markdown renderer
```

## Running

Requires a working Spyre device (first device access triggers init, ~60s).

```bash
cd docs/source/user_guide/examples/provenance
python audit.py                         # default outputs (below)
python audit.py --output /tmp/report.md --raw /tmp/raw.json
```

Outputs:

- **`provenance_audit.md`** — the measurement-only report (for [issue #2574](https://github.com/torch-spyre/torch-spyre/issues/2574)).
- **`provenance_capture_raw.json`** — the full captured structure (every stage,
  every field, plus a `_hooks` block recording whether each hook installed and
  fired). Useful for verifying a run and for the interpretation deliverable.

stdout prints a per-stage fired/count summary so a hook that fails to install on
a given torch build is immediately visible rather than silently missing.

## Reading the report

- **Counts table** — node/op/kernel/file counts and `OpSpec`'s declared fields.
- **Stage × Field matrix** — each column is measured **on the object that stage
  produces** (FX node → LoopLevelIR `ComputedBuffer` → the `OpSpec` dataclass →
  the emitted JSON). A leading **Layer** column groups fields by where they live:
  `FX` (FX-node meta) or `IR` (`ComputedBuffer` attribute). The two IR columns
  are the same LoopLevelIR: **LoopLevelIR (pre-pass)** entering the Spyre
  pre-scheduling passes and **LoopLevelIR (post-pass)** after they mutate it in
  place (e.g. inserting `restickify` buffers — which can change the op count and
  null `origin_node` on synthesized buffers; issue #2574's "Inductor passes" →
  "LoopLevelIR"). Every column tests **population**
  (the field exists *and* is non-empty; `0` counts, `None`/`[]`/`{}`/`""` do
  not). Symbols: `✅` present & non-empty on all · `◐ n/N` on some · `❌`
  reachable here but measured empty/absent (a genuine drop **or** an op with no
  source to carry, e.g. the unused `traceback` slot) · `➖` not applicable (no
  such slot, or — for FX-meta downstream — carried only indirectly via
  `origins`). The drop reads as the `✅`/`◐` → `❌` at `OpSpec`
  (the dataclass has no provenance field). The capture records `hasattr` per IR
  attribute, distinguishing an empty slot from a non-existent one.
- **Per-stage detail** — observed field `type` per node/op, source lines, the
  exact `kernel → ops → origins` mapping, and whether each emitted `sdsc_*.json`
  carries any provenance field.

## Verifying a run

In `provenance_capture_raw.json`, every entry under `_hooks` should read
`installed`, and each stage's `fired` should be `true`. An `install-failed` or
`error` there means a class/signature drifted on the pinned torch build and that
stage's row is missing data — fix the hook in `captures.py` before trusting the
report.
