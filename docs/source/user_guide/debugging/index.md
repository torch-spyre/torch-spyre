# Debugging

```{toctree}
:hidden:
:maxdepth: 2

inductor_artifacts
unified_logging_framework
op_spec_lab
```

**Scope:** correctness — *why is the result wrong?* For performance
questions (*why is it slow?*) see [Profiling](../profiling/index.md).

This guide describes a systematic approach to debugging incorrect or
unexpected behaviour in Torch-Spyre. The workflow applies whether you
are investigating a wrong numerical result, a compilation failure, or a
runtime error.

## Overview

Debugging a Torch-Spyre issue typically follows these layers, working
from the outside in:

1. **Isolate** — reduce the problem to a minimal, self-contained script
2. **Observe data transfers** — verify tensors arrive on device correctly
3. **Capture failure context** — when a compile or runtime error occurs,
   enable [FFDC](../profiling/ffdc.md) (`TORCH_SPYRE_FFDC=1`) and
   retrieve `torch.spyre.get_diagnostic_report()` after the failure while
   the report files are still available
4. **Inspect compiler artifacts** — trace the issue through the
   compilation pipeline (FX Graph → Loop IR → `sdsc_<index>.json`)
5. **Replay the OpSpec on its own** — capture the failing kernel as a
   standalone script and run just the back-end half of the compile
   ([The OpSpec Lab](op_spec_lab.md))

---

## Step 1 — Create a Minimal Reproducer

Before investigating, reduce the failing model or script to the smallest
possible program that still shows the wrong behaviour. This makes the
compiler artifacts much easier to read.

```python
import torch

# Minimal reproducer — replace with the failing op
x = torch.arange(65, dtype=torch.float16)
result = x.clone().to("cpu")
print(result)
```

---

## Step 2 — Enable Debug Environment Variables

The following environment variables control the level of diagnostic output:

| Variable | Effect |
|----------|--------|
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` | Forces full recompilation on every run; ensures you see fresh artifacts, not cached ones |
| `TORCH_SPYRE_DEBUG=1` | Logs all CPU↔Spyre data transfers, including tensor shapes, layouts, and raw values |
| `TORCH_COMPILE_DEBUG=1` | Writes intermediate compiler artifacts to a local directory for offline inspection |
| `TORCH_SPYRE_FFDC=1` | Captures an [FFDC](../profiling/ffdc.md) JSON report on frontend-compile / backend-compile / runtime / unimplemented failures. Separate from `USE_SPYRE_PROFILER` (profiler build flag); not set by default on pods. |
| `SPYRE_INDUCTOR_LOG=1` | *Deprecated.* Use `TORCH_LOGS="torch_spyre.inductor"` instead (INFO level) |
| `SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` | *Deprecated.* Use `TORCH_LOGS="+torch_spyre.inductor"` instead (DEBUG level) |
| `SPYRE_LOG_FILE=path/to/file.log` | Redirect Spyre Inductor log output to a file |
| `TORCH_SPYRE_DOWNCAST_WARN=0` | Suppress int64→int32 warnings |
| `TORCH_LOGS="+inductor"` | PyTorch provided tool to selectively enable Inductor or other parts of the `torch.compile` to the log |

### Programmatic Logging Control

For log levels not supported by `TORCH_LOGS` (WARNING, CRITICAL, DISABLED) or
for runtime control, use the programmatic API:

```python
from torch_spyre import logging_config

# Set specific log levels
logging_config.set_log_level('spyre.inductor', 'WARNING')
logging_config.set_log_level('spyre.runtime', 'CRITICAL')

# Per-pass DEBUG logging (for compiler pipeline debugging)
logging_config.set_log_level('spyre.inductor.passes', 'DEBUG')
logging_config.set_log_passes('all')  # or specific passes
```

See [Programmatic Configuration](../profiling/environment_variables.md#programmatic-configuration)
for complete API documentation.

Run your reproducer with FFDC plus the main debug flags enabled:

```bash
TORCH_SPYRE_FFDC=1 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCH_SPYRE_DEBUG=1 \
TORCH_COMPILE_DEBUG=1 \
python my_reproducer.py
```

`TORCH_COMPILE_DEBUG` writes artifacts to a subdirectory under
`/tmp/torchinductor_<user>/` (or `torch_compile_debug/` in the current
directory, depending on your PyTorch version).

---

## Step 3 — Capture FFDC Report

If the failure is a compile error, runtime launch error, or
unimplemented operation, FFDC can preserve the exception context and
nearby artifact paths before logs are cleaned up.

Enable capture before reproducing:

```bash
TORCH_SPYRE_FFDC=1 python my_reproducer.py
```

After the failure, retrieve the newest valid report:

```python
import torch
import torch_spyre

report = torch.spyre.get_diagnostic_report()
if report is not None:
    print(report["failure"]["category"])
    print(report["failure"]["file"], report["failure"]["lineno"])
    print(report["failure"]["message"])
    print(report["_report_path"])
```

See the [FFDC guide](../profiling/ffdc.md) for report locations,
artifact interpretation, and pod/CI workflow.

---

## Step 4 — Examine Compiler Artifacts

`TORCH_COMPILE_DEBUG` preserves one subdirectory per compiled function.
Inside you will find the intermediate representation at each stage of
the pipeline:

```
torch_compile_debug/
└── run_<timestamp>-pid_<pid>/
    ├── torchdynamo/
    │   └── debug.log
    └── torchinductor/
        ├── aot_model___0_debug.log
        └── model__0_inference_0.0/
            ├── fx_graph_readable.py                              ← traced FX Graph (ATen ops)
            ├── fx_graph_runnable.py                              ← self-contained runnable graph
            ├── fx_graph_transformed.py                           ← FX Graph after Inductor passes
            ├── inductor_provenance_tracking_node_mappings.json   ← IR-to-source mapping
            ├── ir_pre_fusion.txt                                 ← LoopLevelIR before kernel fusion
            ├── ir_post_fusion.txt                                ← LoopLevelIR after kernel fusion
            └── output_code.py                                    ← generated host code
```

### What to look for at each layer

**FX Graph** (`fx_graph_readable.py`)
Verify the traced operation matches what you expect. Check that the
operation is present, that input shapes are correct, and that no
unexpected decompositions have changed the semantics.

**LoopLevelIR** (`ir_pre_fusion.txt`, `ir_post_fusion.txt`)
Check that loop ranges and buffer shapes reflect the correct tensor
sizes including padding. Mismatches here indicate a problem in the
Inductor lowering or stickification pass.

**`sdsc_<index>.json`**
The final specifications fed to the DeepTools back-end compiler — one
file per compiled kernel (`sdsc_0.json`, `sdsc_1.json`, …), indexed in
lowering order. Each file encodes the op name, input/output tensor
layouts (`device_size`, `stride_map`, `device_dtype`), work division,
and scratchpad allocations. Bugs that appear only in the final output
often trace back to one of these files.

### Example: debugging an incorrect `clone` result

Consider a `float16` tensor of size `[65]`. The default Spyre layout
pads the stick dimension to 128 bytes (64 elements per stick), so the
tensor is laid out on device as shape `[2, 64]` — two sticks.

`TORCH_SPYRE_DEBUG=1` output confirming the CPU→Spyre transfer is
correct:

```
[TORCH_SPYRE_DEBUG] copy_to_device: shape=[65] dtype=float16
  device_layout: SpyreTensorLayout(device_size=[2, 64], stride_map=[64, 1], ...)
  transfer OK
```

Inspecting the `sdsc_<index>.json` for the clone kernel (locate it via
`output_code.py`, which maps each launched kernel to its index) then
revealed:

```
{
  "op": "clone",
  "input_layout": { "device_size": [2, 64], ... },
  "copy_range": [1, 64]   // ← only copying the first stick, should be [2, 64]
}
```

The bug: the codegen only emitted a copy for the first stick (`[1, 64]`)
instead of both sticks (`[2, 64]`). The second stick — which holds
element index 64 — was never written, leaving it at zero.
*(See [issue #524](https://github.com/torch-spyre/torch-spyre/issues/524)
for the full investigation.)*

---

## Checklist for Filing a Bug Report

When opening an issue, include:

- [ ] Minimal reproducer script
- [ ] Full error output or incorrect value observed
- [ ] PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- [ ] Torch-Spyre version or commit SHA
- [ ] Output of `TORCH_SPYRE_DEBUG=1` showing the data transfer log
- [ ] Relevant excerpts from `fx_graph_readable.py` and the affected
  `sdsc_<index>.json`
- [ ] FFDC report path / `failure.category` when `TORCH_SPYRE_FFDC=1` was set

---

## Quick Reference

```bash
# Full debug run
TORCH_SPYRE_FFDC=1 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCH_SPYRE_DEBUG=1 \
TORCH_COMPILE_DEBUG=1 \
python my_reproducer.py

# Find the generated artifacts
find . -name "sdsc_*.json" 2>/dev/null
find /tmp -name "fx_graph_readable.py" 2>/dev/null
```
