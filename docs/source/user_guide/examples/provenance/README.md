# Source-to-Kernel Provenance Audit

Phase 1 tooling for [torch-spyre#2574](https://github.com/torch-spyre/torch-spyre/issues/2574).
It traces a `SimpleMLP` through the Spyre compilation pipeline and records, at
each stage, which provenance fields are present on the **actual compile-path
objects** — producing a measurement-only Markdown report.

## What it measures

Six stages, mapped to [issue #2574](https://github.com/torch-spyre/torch-spyre/issues/2574):

1. PyTorch model source
2. FX graph (pre-grad and post-grad)
3. LoopLevelIR (pre-pass)
4. LoopLevelIR (post-pass)
5. OpSpec
6. SuperDSC JSON

For each stage it records which fields are present and their observed type:

- **FX node meta:** `stack_trace`, `nn_module_stack`, `source_fn_stack`,
  `original_aten`, `from_node`
- **IR / LoopLevelIR attributes:** `origins`, `origin_node`, `traceback`,
  `get_stack_traces()`
- **OpSpec / SuperDSC (`Spyre`):** `debug_handle` — the source-to-kernel record
  (`id`, `source`, `aten_op`, `ir_chain`, `fused_from`) carried on `OpSpec` and
  serialized into each `sdsc_*.json`

The report is **measurement-only** — everything is computed from the run, with
no hand-written analysis or recommendations.

## How it works

Everything is observed in-process during **one** `torch.compile`, via read-only
monkey-patches installed for the duration of the call. From Stage 2 to 5, each stage reads the real object. Stage 6's JSON is then read by `superdsc.py` from the **exact** per-kernel
output directories captured at compile time (keyed by `kernel_name`).

| Stage | Hook (class-level, in `captures.py`) | Reads |
| --- | --- | --- |
| 2a FX Graph (pre-grad) | `CustomPreGradPasses.__call__` | `node.meta` fields + data-flow (`all_input_nodes`) |
| 2b FX Graph (post-grad)  | `CustomPostPasses.__call__` | `node.meta` fields + data-flow (`all_input_nodes`) |
| 3 & 4 LoopLevelIR (pre-pass & post-pass) | `CustomPreSchedulingPasses.__call__` | `graph.operations[*]` origins/origin_node/traceback/get_stack_traces (value **and** `hasattr`) + each op's full attribute inventory, before & after the pass list |
| 5 OpSpec | `SpyreKernel.create_op_spec` | input `ComputedBuffer.origins`; `OpSpec` declared fields + per-instance provenance-field population on the returned `OpSpec` |
| 6 SuperDSC | `SuperDSCScheduling.define_kernel` + `async_compile.get_output_dir` | `kernel_name`, per-kernel origins, `get_kernel_metadata`, exact bundle dirs |

Note that caches are force-disabled (`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` +
`force_disable_caches` + `torch._dynamo.reset()`) so scheduling and codegen
actually run; a cache hit would silently skip `create_op_spec`/`define_kernel`.

## Layout

```
provenance/
├── audit.py             # entry point: one compile → capture → bundle read → report
├── reference_mlp.py     # the model under audit (the subject, not tooling)
├── README.md
└── pipeline/
    ├── fields.py        # single source of truth for the provenance field-name lists
    ├── captures.py      # all six in-process stage hooks (one context manager)
    ├── superdsc.py      # Stage 6: read sdsc_*.json from captured exact dirs
    └── report.py        # measurement-only Markdown renderer
```

## Running

Requires a working Spyre device.

```bash
cd docs/source/user_guide/examples/provenance
python audit.py                         # default outputs (below)
python audit.py --output /tmp/report.md --raw /tmp/raw.json
```

Outputs:

- **`provenance_audit.md`** — the measurement-only report (for [issue #2574](https://github.com/torch-spyre/torch-spyre/issues/2574)). The committed copy is an **example snapshot**; its timestamp and `debug_handle` ids are specific to the run/machine, so regenerate with `audit.py` to reproduce for your own setup.
- **`provenance_capture_raw.json`** — the full captured structure (every stage,
  every field, plus a `_hooks` block recording whether each hook installed and
  fired). Useful for verifying a run.

stdout prints a per-stage fired/count summary so a hook that fails to install on
a given torch build is immediately visible rather than silently missing.

## Reading the report

- **Source → Kernel lineage** — a Mermaid graph of how each source line
  flows through the six stages: **Source → FX Graph pre-grad → FX Graph post-grad →
  LoopLevelIR pre-pass → LoopLevelIR post-pass → OpSpec → SuperDSC**. A fan-out
  is a decomposition (e.g. `linear` → `permute` + `mm` + `add`); a fan-in is a
  fusion (several ops → one kernel). An op with no source of its own attaches to
  every source-bearing producer whose buffer it consumes.
- **Stage × Field matrix** — each column is measured **on the object that stage
  produces** (FX node → LoopLevelIR `ComputedBuffer` → the `OpSpec` dataclass →
  the emitted JSON). A leading **Layer** column groups fields by where they live:
  `FX` (FX-node meta), `IR` (`ComputedBuffer` attribute), or `Spyre` (the
  Phase-2a `debug_handle`). The two IR columns are the same LoopLevelIR
  before/after the Spyre pre-scheduling passes (issue #2574's "Inductor passes"
  → "LoopLevelIR"). Every column tests **population** (the field exists *and* is
  non-empty; `0` counts, `None`/`[]`/`{}`/`""` do not). Symbols: `✓` present on
  all · `◐ n/N` on some · `✗` reachable but empty/absent (dropped) · `–` not
  created yet, or carried indirectly inside another field. The capture records
  `hasattr` per IR attribute, distinguishing an empty slot from a non-existent
  one.
- **Per-stage detail** — observed field `type` per node/op, source lines, the
  exact `kernel → ops → origins` mapping, and whether each emitted `sdsc_*.json`
  carries a `debug_handle`. Each stage also lists the full field inventory it
  observed (e.g. all `node.meta` keys, `OpSpec` declared fields) so a provenance
  field not yet tracked in a column is still surfaced.

## Verifying a run

In `provenance_capture_raw.json`, every entry under `_hooks` should read
`installed`, and each stage's `fired` should be `true`. An `install-failed` or
`error` there means a class/signature drifted on the pinned torch build and that
stage's row is missing data — fix the hook in `captures.py` before trusting the
report.
