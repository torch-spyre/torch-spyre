---
name: project-overview
description: "Overview of the torch-spyre repository, Spyre device architecture, compilation pipeline, and codebase structure. Use when asking about how torch-spyre works, the repo layout, Spyre hardware, the Inductor backend, or getting oriented in the codebase."
---

# torch-spyre Project Overview

You are working in the **torch-spyre** repository — an out-of-tree PyTorch
backend that registers the **IBM Spyre AI Accelerator** as a first-class
PyTorch device (`"spyre"`).

When the user asks about this project, use this document to provide accurate
context. Read relevant source files as needed to supplement.

---

## What is Spyre?

The **IBM Spyre Accelerator** is a high-performance AI accelerator (32 cores,
5nm, 128 GB LPDDR5/card, >300 TOPS at 75W) for IBM Z, LinuxONE, and Power
systems. Its fundamental memory primitive is a **stick** — a 128-byte-aligned
chunk. All Spyre memory and compute work on sticks, which drives the design of
torch-spyre.

## How torch-spyre Works

torch-spyre uses PyTorch's **PrivateUse1** mechanism to register `"spyre"` as
a device:

```python
torch.utils.rename_privateuse1_backend("spyre")
```

It supports two execution paths:

1. **Eager mode** — operator-by-operator dispatch via auto-generated Python
   wrappers that call SendNN
2. **Compiled mode** — `torch.compile()` via a custom Inductor backend that
   generates SuperDSC descriptors compiled by the `dxp_standalone` backend
   compiler

---

## Repository Structure

```
torch-spyre/
├── torch_spyre/                 # Main Python package
│   ├── __init__.py              # Device registration, lazy init, _autoload entry point
│   ├── _monkey_patch.py         # Patches torch.Tensor for Spyre awareness
│   ├── constants.py             # DEVICE_NAME = "spyre"
│   ├── csrc/                    # C++ sources (pybind11 → torch_spyre._C and ._hooks)
│   │   ├── module.cpp           # Main pybind11 module: startRuntime, launchKernel, encodeConstant
│   │   ├── spyre_tensor_impl.*  # SpyreTensorLayout, SpyreTensorImpl (tiled tensor layout)
│   │   ├── spyre_mem.*          # Device memory allocation (spyre_empty_strided, etc.)
│   │   ├── spyre_hooks.cpp      # PrivateUse1 hooks + SpyreGuardImpl (separate ._hooks module)
│   │   ├── spyre_views.*        # View/reshape layout computation
│   │   └── spyre_sendnn_utils.* # SendNN tensor descriptor utilities
│   ├── _inductor/               # Inductor compiler backend (see below)
│   ├── ops/                     # Ops team: eager, fallbacks, decompositions, lowering, customops
│   │   ├── eager.py             # Base ops file
│   │   ├── fallbacks.py         # CPU fallback operator registry
│   ├── execution/               # Runtime: async_compile, kernel_runner, OpSpec
│   ├── device/                  # Device interface and override utilities
│   └── memory/                  # Memory/DMA (placeholder for future work)
│
├── tests/                       # Test suite
│   ├── test_ops.py              # Eager operator tests
│   ├── test_modules.py          # nn.Module tests
│   ├── inductor/                # Compiled-mode tests
│   │   ├── test_inductor_ops.py # Parameterized compile-path op tests
│   │   ├── test_building_blocks.py # End-to-end block tests (MLP, attention)
│   │   └── utils_inductor.py    # Test utilities (compare_with_cpu, ParameterizedTestMeta)
│   └── tensor/                  # Layout unit tests
│
├── docs/                        # Architecture docs
│   ├── spyre.md, pytorch_on_spyre.md, compiler_architecture.md
│   ├── tensor_layouts.md        # Tiled tensor layout specification
│   ├── adding_operations.md     # How to add new ops
│   └── work_division_*.md       # Multi-core parallelism docs
│
├── examples/                    # Usage examples (softmax, gelu, mul, etc.)
├── setup.py                     # Build: C++ extension compilation
├── pyproject.toml               # PEP 517/518 metadata, deps (torch~=2.10.0)
└── tools/                       # Developer tooling (lint, format, mypy)
```

---

## Inductor Backend (`torch_spyre/_inductor/`)

The compiled-mode pipeline is the most complex part. Here is the flow:

### Compilation Pipeline

```
torch.compile(model)
  → Dynamo captures FX graph
  → Decompositions (decompositions.py)
  → Inductor LoopLevelIR lowering (lowering.py)
  → Scheduler passes (passes.py):
      1. Stickify pass (stickify.py) — propagate SpyreTensorLayout through graph
      2. Core division (core_division.py) — assign multi-core work splitting
      3. Scratchpad planning (scratchpad.py) — optional LX on-chip memory reuse
  → SpyreKernel codegen (spyre_kernel.py) — LoopLevelIR → KernelSpec
  → SuperDSC generation (codegen/superdsc.py) — KernelSpec → JSON descriptor
  → dxp_standalone (backend compiler) — JSON → g2.graph.cbor binary
  → SpyreSDSCKernelRunner — calls _C.launch_kernel() at runtime
```

### Key Files

| File | Role |
|---|---|
| `__init__.py` | Backend registration, `compile_fx` wrapping, `_light_autoload` |
| `patches.py` | `enable_spyre_context()` — configures Inductor for Spyre |
| `ir.py` | `FixedTiledLayout` (extends FixedLayout with device_layout), `SpyreReduction` |
| `lowering.py` | ATen → LoopLevelIR lowerings (mm, bmm, pointwise, reductions) |
| `decompositions.py` | FX decompositions (layer_norm → exx2+scale+norm, gt→ge*ne, etc.) |
| `customops.py` | `spyre::*` custom op definitions (compact, swap, slice, gelu, etc.) |
| `stickify.py` | Layout propagation pass — assigns FixedTiledLayout to each node |
| `core_division.py` | Multi-core work division planning (up to 32 cores) |
| `scratchpad.py` | LX scratchpad memory planning (2MB on-chip, enabled by LX_PLANNING=1) |
| `passes.py` | Orchestrates the three scheduler passes |
| `spyre_kernel.py` | `SpyreKernel` — translates LoopLevelIR to `KernelSpec` |
| `dsc.py` | `SuperDSCScheduling` — emits kernel definition code |
| `wrapper.py` | `SpyrePythonWrapperCodegen` — host-side wrapper code |
| `choices.py` | `SpyreHeuristics` — disables fusion/cooperative reductions |
| `constants.py` | Op name constants (BATCH_MATMUL_OP, TRANSPOSE_OP, etc.) |
| `errors.py` | Spyre-specific error classes |
| `temp_passes.py` | `relayout_linear_weights` — ensures weight contiguity for mm |
| `runtime/__init__.py` | `KernelSpec`, `TensorArg`, `ConstantArg` dataclasses |
| `runtime/async_compile.py` | `SpyreAsyncCompile` — SDSC→binary compilation |
| `runtime/kernel_runner.py` | `SpyreSDSCKernelRunner` — kernel execution via `_C.launch_kernel` |
| `codegen/superdsc.py` | `generate_sdsc()` — KernelSpec → SuperDSC JSON |
| `codegen/compute_ops.py` | Pointwise/reduction/matmul SDSC generation |
| `codegen/data_ops.py` | Transpose/clone/slice SDSC generation |

---

## Key Abstractions

### SpyreTensorLayout

The core layout abstraction (defined in C++, exposed via pybind11). Standard
PyTorch `size+stride` cannot represent tiled tensors.

- **`device_size: list[int]`** — padded, higher-dimensional on-device shape
  (always row-major)
- **`stride_map: list[int]`** — host stride for each device dim (-1 = synthetic
  or padded dimension with no host correspondence)
- **`device_dtype: DataFormats`** — on-device data format (SEN169_FP16,
  IEEE_FP32, etc.)

Example: `(5, 100, 150)` float16 →
`SpyreTensorLayout(device_size=[100, 3, 5, 64], stride_map=[150, 64, 15000, 1], device_dtype=SEN169_FP16)`
where 64 = elements per 128-byte stick for fp16.

### KernelSpec

Central compilation data structure describing a single Spyre operation:

- `op: str` — operation name
- `dimensions: list[int]` — iteration space
- `args: list[TensorArg | ConstantArg]` — inputs/outputs with layout info
- `scales: list[list[int]]` — dimension mapping (op dim → host tensor dim per
  arg)
- `op_info: dict` — op-specific metadata (constants, core_division, etc.)

### FixedTiledLayout

IR-level layout extending Inductor's `FixedLayout` with
`device_layout: SpyreTensorLayout` and optional `allocation` dict for scratchpad
info.

---

## C++ Modules

Two separate pybind11 modules:

1. **`torch_spyre._C`** (heavy): Runtime lifecycle, kernel launching, memory
   management, layout classes, type utilities
2. **`torch_spyre._hooks`** (lightweight): PrivateUse1 hooks registration +
   device guard (loaded early, no heavy init)

---

## Build System

`setup.py` does:

1. **C++ compilation**: Two `CppExtension` targets (`_C` and `_hooks`) linking
   against `sendnn`, `flex`
2. **Entry point**: `torch.backends` → `torch_spyre = torch_spyre:_autoload`

Key external deps: `torch~=2.10.0`, `sendnn`, `flex`, `dxp_standalone`

---

## Initialization Flow

1. PyTorch discovers `torch_spyre` via `torch.backends` entry point → calls
   `_autoload()`
2. Imports `._hooks` → registers PrivateUse1 hooks + device guard
3. Renames backend to `"spyre"`, registers device module
4. Imports eager ops, preloads Inductor decomposition overrides
5. Wraps `compile_fx` for transparent Spyre detection
6. **Lazy init**: Heavy runtime (`flex.CreateRuntimeInterface`,
   `_C.start_runtime`) only starts on first device access

---

## Testing

```bash
python3 -m pytest tests/                  # All tests
python3 -m pytest tests/test_ops.py       # Eager ops
python3 -m pytest tests/inductor/        # Compiled ops
python3 -m pytest tests/tensor/           # Layout tests
```

Key test patterns:

- `compare_with_cpu(fn, *inputs)` — runs fn on CPU vs Spyre via torch.compile,
  compares
- `ParameterizedTestMeta` — metaclass for generating parameterized test methods
  from PARAMS dicts
- Default tolerances: `rtol=atol=1e-1`, `dtype=float16`

---

## Important Environment Variables

| Variable | Purpose |
|---|---|
| `TORCH_SPYRE_DEBUG=1` | C++ debug logging + `-O0` build |
| `SENCORES` | Number of Spyre cores (1–32, default 32) |
| `LX_PLANNING=1` | Enable LX scratchpad planning |
| `TORCH_SPYRE_DOWNCAST_WARN=0` | Suppress float32→float16 warnings |
| `DT_DEEPRT_VERBOSE=-1` | Reduce runtime verbosity |

---

## Supported Operations (Compiled Path)

**Pointwise**: abs, add, clamp, eq, exp, ge, gelu, le, log, mul, ne, neg,
reciprocal, relu, rsqrt, sigmoid, softplus, sqrt, square, sub, tanh, to_dtype,
truediv, where

**Reductions**: mean, exx2 (fused mean/variance), layernormscale, layernormnorm

**Matrix ops**: mm (2D matmul), bmm (3D and 4D batch matmul)

**Data movement**: transpose, clone, swap, slice (sparse ops), squeeze/unsqueeze
(via views)

**Custom fused ops**: layer_norm (→ exx2 + layernormscale + layernormnorm),
gelu, softplus, clamp

**Modules**: nn.Linear
