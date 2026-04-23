# Debugging Guide

This guide describes a systematic approach to debugging incorrect or
unexpected behaviour in Torch-Spyre. The workflow applies whether you
are investigating a wrong numerical result, a compilation failure, or a
runtime error.

## Overview

Debugging a Torch-Spyre issue typically follows these layers, working
from the outside in:

1. **Isolate** — reduce the problem to a minimal, self-contained script
2. **Observe data transfers** — verify tensors arrive on device correctly
3. **Inspect compiler artifacts** — trace the issue through the
   compilation pipeline (FX Graph → Loop IR → sdsc.json)
4. **Bisect frontend vs. backend** — use the `sendnn` backend to
   determine whether the bug is in Torch-Spyre's front-end or in the
   DeepTools back-end compiler

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
| `SPYRE_INDUCTOR_LOG=1` | Enable Spyre-specific Inductor logging |
| `SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` | Set Spyre Inductor log verbosity (DEBUG, INFO, WARNING, ERROR) |
| `SPYRE_LOG_FILE=path/to/file.log` | Redirect Spyre Inductor log output to a file |

Run your reproducer with all three enabled:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCH_SPYRE_DEBUG=1 \
TORCH_COMPILE_DEBUG=1 \
python my_reproducer.py
```

`TORCH_COMPILE_DEBUG` writes artifacts to a subdirectory under
`/tmp/torchinductor_<user>/` (or `torch_compile_debug/` in the current
directory, depending on your PyTorch version).

---

## Step 3 — Examine Compiler Artifacts

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

**sdsc.json**
This is the final specification fed to the DeepTools back-end compiler.
It encodes the op name, input/output tensor layouts (`device_size`,
`stride_map`, `device_dtype`), work division, and scratchpad allocations.
Bugs that appear only in the final output often trace back here.

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

Inspecting `sdsc.json` for the clone kernel then revealed:

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

## Step 4 — Bisect Frontend vs. Backend with `sendnn`

If the sdsc.json looks correct, the bug is likely in the DeepTools
back-end compiler. To confirm, re-run the same script using the
`sendnn` backend instead of `spyre`:

```python
import torch

def test(a, b):
    return torch.eq(a, b).to(dtype=torch.float16)

x = torch.tensor([-0.0, -0.0], dtype=torch.float16)
y = torch.tensor([0.0, 0.0], dtype=torch.float16)

# Test with sendnn backend
compiled = torch.compile(test, backend="sendnn")
result = compiled(x, y).to(dtype=torch.bool)
print(f"sendnn: {result}")

# Compare with spyre backend
compiled_spyre = torch.compile(test, backend="spyre")
result_spyre = compiled_spyre(x.to("spyre"), y.to("spyre")).to("cpu").to(dtype=torch.bool)
print(f"spyre:  {result_spyre}")
```

| Outcome | Interpretation |
|---------|----------------|
| `sendnn` wrong, `spyre` wrong | Bug is in the **back-end compiler** (DeepTools); file issue against the backend |
| `sendnn` correct, `spyre` wrong | Bug is in **Torch-Spyre's front-end** (codegen, stickification, or host code) |
| Both correct | The issue may be environment-specific or in data transfer |

*(See [issue #628](https://github.com/torch-spyre/torch-spyre/issues/628)
for an example where both backends returned the same incorrect result,
confirming a back-end compiler bug.)*

---

## Checklist for Filing a Bug Report

When opening an issue, include:

- [ ] Minimal reproducer script
- [ ] Full error output or incorrect value observed
- [ ] PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- [ ] Torch-Spyre version or commit SHA
- [ ] Output of `TORCH_SPYRE_DEBUG=1` showing the data transfer log
- [ ] Relevant excerpts from `fx_graph_readable.py` and `sdsc.json`
- [ ] Result of the `sendnn` comparison (frontend vs. backend bisect)

---

## Quick Reference

```bash
# Full debug run
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCH_SPYRE_DEBUG=1 \
TORCH_COMPILE_DEBUG=1 \
python my_reproducer.py

# Find the generated artifacts
find . -name "sdsc.json" 2>/dev/null
find /tmp -name "fx_graph_readable.py" 2>/dev/null
```
