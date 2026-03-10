# Accelerator-Specific Profiling: Beyond NVIDIA CUDA

> What information is captured differently for accelerators other than NVIDIA? What are the sources?

---

## 1. Supported Accelerators in PyTorch Profiler

The `ProfilerActivity` enum supports:

| Activity | Backend | Tracing Library | Source |
|---|---|---|---|
| `CPU` | All platforms | Kineto (built-in) | [pytorch/kineto](https://github.com/pytorch/kineto) |
| `CUDA` | NVIDIA GPUs | CUPTI | [CUPTI Docs](https://docs.nvidia.com/cupti/), [CUPTI SDK](https://developer.nvidia.com/cupti) |
| `XPU` | Intel GPUs (Arc, Data Center) | PTI (Profiling Tools Interfaces) | [intel/pti-gpu](https://github.com/intel/pti-gpu) |
| `HPU` | Habana Gaudi | SynapseAI Profiler | [HabanaAI/gaudi-pytorch-bridge](https://github.com/HabanaAI/gaudi-pytorch-bridge) |
| `MTIA` | Meta Training & Inference Accelerator | Internal Meta tooling | [PyTorch MTIA docs](https://docs.pytorch.org/docs/stable/mtia.html) |
| `PrivateUse1` | Custom backends (Ascend NPU, etc.) | Backend-specific | -- |

Additionally, **Google TPU** uses a completely separate profiling path via **XProf** (not integrated with `ProfilerActivity`).

---

## 2. AMD ROCm GPU Profiling

### Sources
- **roctracer**: [ROCm/roctracer](https://github.com/ROCm/roctracer) (deprecated, moved to rocm-systems)
- **rocprofiler**: [ROCm/rocprofiler](https://github.com/ROCm/rocprofiler)
- **rocprofiler-sdk**: [ROCm/rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) (deprecated, moved to rocm-systems)
- **ROCm Systems (unified repo)**: [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)
- **roctracer docs**: https://rocm.docs.amd.com/projects/roctracer/en/latest/index.html
- **Kineto ROCm integration**: [pytorch/kineto](https://github.com/pytorch/kineto) (contains `RoctracerActivity` and `RocprofActivityApi`)

### Tracing Backend
- Uses **roctracer** library (via Kineto `RoctracerActivity`)
- Newer path via **rocprofiler** (`RocprofActivityApi`)

### Per-Kernel Information Captured

| Field | Captured? | Notes |
|---|---|---|
| Kernel name | Yes | Demangled function name |
| Grid dimensions | Yes | gridX, gridY, gridZ |
| Block/workgroup dimensions | Yes | workgroupX, workgroupY, workgroupZ |
| Shared memory (LDS) | Yes | `groupSegmentSize` (combined, not split static/dynamic) |
| Registers per thread | **No** | Not available via roctracer |
| Blocks per CU | **No** | Not computed |
| Warps (wavefronts) per CU | **No** | Not computed |
| Estimated occupancy | **No** | Occupancy calculation is CUPTI-only in Kineto |
| Queued timestamp | **No** | Not captured |
| Context ID | **No** | Not captured |
| Stream ID | Yes | HIP stream |
| Correlation ID | Yes | Links to CPU-side HIP API calls |
| Start/end timestamps | Yes | Nanosecond precision (CLOCK_MONOTONIC) |

### Memory Operations
- Memory copies: source/destination pointers, size, copy kind (DtoH, HtoD, DtoD), bandwidth (GB/s)
- Memory allocations: pointer, size, correlation ID

### Runtime API Tracing
- HIP runtime calls are traced (equivalent to CUDA runtime calls)
- No driver-level API tracing

### Chrome Trace JSON Differences (vs CUDA)

```
CUDA kernel args:
  "registers per thread": 86, "shared memory": 32768,
  "blocks per SM": 0.025, "warps per SM": 0.1,
  "est. achieved occupancy %": 0, "grid": [...], "block": [...]

ROCm kernel args:
  "shared memory": X, "grid": [...], "block": [...],
  "kind": "Dispatch Kernel"
  // NO registers, NO occupancy, NO blocks/warps per CU
```

### Additional Profiling (Outside PyTorch Profiler)
The standalone `rocprof` CLI can capture hardware counters NOT available in PyTorch profiler:
- VGPR count, SGPR count
- LDS size, scratch memory
- Wavefronts launched
- VALUInsts, SALUInsts (instruction counts)
- FetchSize, WriteSize (memory bandwidth)
- L2CacheHit rate
- MemUnitStalled, MemUnitBusy

---

## 3. Google TPU Profiling (via PyTorch/XLA)

### Sources
- **XProf**: [openxla/xprof](https://github.com/openxla/xprof)
- **OpenXLA**: https://openxla.org/
- **PyTorch/XLA**: [pytorch/xla](https://github.com/pytorch/xla)
- **TPU profiling docs**: https://docs.cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm

### Profiling Path
- **Does NOT use `ProfilerActivity`** or Kineto
- Uses **XProf** (from OpenXLA project) -- completely separate profiling system
- Visualization via **TensorBoard** with `tensorboard-plugin-profile`

### How to Profile

```python
import torch_xla.debug.profiler as xp

# Start profiler server
server = xp.start_server(9012)

# Use trace context manager
with xp.Trace('model_inference'):
    output = model(input)
```

### Metrics Captured

| Category | Metrics |
|---|---|
| **Operation-level** | Duration per XLA HLO operation |
| **Step timing** | Step time breakdown (compute, communication, idle) |
| **Memory** | HBM (High Bandwidth Memory) usage over time |
| **Communication** | Inter-device communication patterns |
| **Host vs Device** | Attribution of time spent on host vs TPU |

### Analysis Tools
- **Overview page**: High-level performance summary
- **Trace viewer**: Timeline of operations across TPU cores
- **Memory profile viewer**: Memory consumption tracking
- **Graph viewer**: HLO (High-Level Operation) graph visualization
- **Pod viewer**: Multi-host TPU performance

### What is NOT Captured (vs CUDA)
- No per-kernel register usage (TPU has different compute model)
- No grid/block dimensions (TPU uses tiles, not thread blocks)
- No shared memory metrics (TPU uses HBM directly)
- No occupancy metric (TPU has different execution model -- systolic array/MXU)
- MXU utilization and per-core TFLOPS may require Google Cloud's internal tools

---

## 4. Intel XPU Profiling

### Sources
- **PTI-GPU**: [intel/pti-gpu](https://github.com/intel/pti-gpu)
- **Intel Extension for PyTorch (xpupti)**: https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.10+xpu/tutorials/features/profiler_kineto.html
- **Intel oneAPI optimization guide**: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/pti-gpu.html

### Tracing Backend
- Uses **PTI (Profiling Tools Interfaces)** via `xpupti` plugin in Kineto
- Integrates with Level Zero driver and SYCL runtime

### Activity Types Collected

| Activity Type | Description |
|---|---|
| `CONCURRENT_KERNEL` | Compute kernel executions |
| `GPU_MEMCPY` | Device memory copies |
| `GPU_MEMSET` | Memory fill operations |
| `XPU_RUNTIME` | SYCL runtime API calls |
| `XPU_DRIVER` | Level Zero driver API calls |

### Per-Kernel Information

| Field | Captured? | Notes |
|---|---|---|
| Kernel name | Yes | SYCL kernel name |
| Kernel ID | Yes | Unique kernel identifier |
| Start/end timestamps | Yes | |
| Device UUID | Yes | |
| SYCL queue ID | Yes | |
| Level Zero queue handle | Yes | |
| Context handle | Yes | |
| Append timestamp | Yes | When kernel was appended to queue |
| Submit timestamp | Yes | When kernel was submitted to hardware |
| Correlation ID | Yes | Links to CPU-side API calls |
| Registers per thread | **No** | Not captured |
| Shared memory | **No** | Not broken down |
| Grid/block dimensions | **No** | Not in current implementation |
| Occupancy | **No** | Not computed |

### Memory Operations
- Memory copies: bytes, bandwidth (GB/s), source/destination memory type, memcpy type
- Memory sets: bytes, bandwidth (GB/s)
- Memory operation IDs for tracking

### Profiler Overhead Tracking
- Intel XPU profiler tracks its own overhead: duration, occupancy %, count

### Note
Intel Extension for PyTorch (IPEX) had its own kineto profiler, deprecated since IPEX 2.5. Upstream PyTorch's profiler is now the recommended path.

---

## 5. Habana HPU (Gaudi) Profiling

### Sources
- **Gaudi PyTorch Bridge**: [HabanaAI/gaudi-pytorch-bridge](https://github.com/HabanaAI/gaudi-pytorch-bridge)
- **Gaudi Tutorials**: [HabanaAI/Gaudi-tutorials](https://github.com/HabanaAI/Gaudi-tutorials)
- **Habana profiling docs**: https://docs.habana.ai/en/latest/PyTorch/Reference/Python_Packages.html
- **Perfetto trace viewer for Gaudi**: https://perfetto.habana.ai

### Tracing Backend
- **Most minimal** integration of all backends
- Single activity type: `HPU_OP`

### What is Captured
- Operator name and timing
- Correlation with CPU-side PyTorch operators
- Basic device synchronization events

### What is NOT Captured via PyTorch Profiler
- No per-kernel hardware metrics
- No grid/block information (Gaudi has different compute architecture -- TPC)
- No memory bandwidth metrics
- No occupancy information

### External Profiling Tools
Habana provides richer profiling through:
- **SynapseAI Profiler**: Detailed hardware profiling
- **`hl-smi`**: Hardware monitoring (similar to `nvidia-smi`)
- These capture TPC utilization, HBM bandwidth, power consumption

---

## 6. Meta MTIA Profiling

### Sources
- **PyTorch MTIA docs**: https://docs.pytorch.org/docs/stable/mtia.html
- **PyTorch profiler docs (MTIA activity)**: https://docs.pytorch.org/docs/stable/profiler.html
- **Source code**: Integrated directly in [pytorch/pytorch](https://github.com/pytorch/pytorch) (no separate public repo)

### Activity Types

| Activity Type | Description |
|---|---|
| `MTIA_CCP_EVENTS` | Core compute processor events |
| `MTIA_RUNTIME` | MTIA runtime API calls |
| `MTIA_INSIGHT` | Performance analysis/insights |

### Notes
- MTIA is an internal Meta accelerator -- limited public documentation
- Second richest non-CUDA integration (after XPU) with 3 activity types
- The `MTIA_INSIGHT` type suggests built-in performance analysis recommendations
- MTIA v1 reports TFLOPS/W metrics but specifics of profiler capture are not publicly documented

---

## 7. PrivateUse1 / Custom Backend

### How It Works
Custom backends register via `c10::register_privateuse1_backend("backend_name")` and implement their own tracing.

### Activity Types Available

| Activity Type | Description |
|---|---|
| `CONCURRENT_KERNEL` | Compute kernel executions |
| `GPU_MEMCPY` | Device memory copies |
| `GPU_MEMSET` | Memory fill operations |
| `GPU_USER_ANNOTATION` | User annotations on device |
| `PRIVATEUSE1_RUNTIME` | Backend runtime API calls |
| `PRIVATEUSE1_DRIVER` | Backend driver API calls |

---

## 8. Cross-Accelerator Comparison

### Device Properties

| Property | CUDA | ROCm | XPU | TPU | HPU |
|---|---|---|---|---|---|
| Device name | Yes | Yes | Yes (UUID) | Via XProf | Yes |
| Compute capability | Yes (major.minor) | GFX version | No | N/A | No |
| Total memory | Yes | Yes | No | Via XProf | No |
| SM/CU/EU count | Yes (numSms) | Yes (numCUs) | No | N/A | No |
| Max threads per block | Yes | Yes | No | N/A | No |
| Warp/wavefront size | Yes (32) | Yes (64) | No | N/A | No |
| Registers per block | Yes | No | No | N/A | No |
| Shared mem per block | Yes | Yes (LDS) | No | N/A | No |

### Per-Kernel / Per-Operation Information

| Field | CUDA | ROCm | XPU | TPU | HPU |
|---|---|---|---|---|---|
| **Kernel/op name** | Yes | Yes | Yes | Yes (HLO op) | Yes |
| **Duration** | Yes (us) | Yes (us) | Yes | Yes | Yes |
| **Grid dimensions** | Yes [x,y,z] | Yes [x,y,z] | No | N/A (tiles) | No |
| **Block dimensions** | Yes [x,y,z] | Yes [x,y,z] | No | N/A | No |
| **Registers per thread** | Yes | No | No | N/A | No |
| **Shared memory** | Yes (static+dynamic) | Partial (LDS combined) | No | N/A | No |
| **Blocks per SM/CU** | Yes | No | No | N/A | No |
| **Warps per SM/CU** | Yes | No | No | N/A | No |
| **Est. occupancy %** | Yes | No | No | N/A | No |
| **Stream ID** | Yes | Yes | Yes (SYCL queue) | N/A | No |
| **Context ID** | Yes | No | Yes | N/A | No |
| **Correlation ID** | Yes | Yes | Yes | N/A | Yes |
| **Queued timestamp** | Yes | No | No | No | No |
| **Submitted timestamp** | No | No | Yes | No | No |
| **Device ID** | Yes | Yes | Yes (UUID) | Yes (core) | Yes |

### Memory Profiling

| Feature | CUDA | ROCm | XPU | TPU | HPU |
|---|---|---|---|---|---|
| **Allocation tracking** | Yes | Yes | Yes | Via XProf | Limited |
| **Total reserved** | Yes | Yes | No | No | No |
| **Total allocated** | Yes | Yes | No | No | No |
| **Per-event bytes delta** | Yes | Yes | Yes | No | No |
| **Memory address** | Yes | Yes | No | No | No |
| **Memory bandwidth (GB/s)** | Computed | Computed | Yes | Unknown | No |
| **Memory copy tracking** | Yes (HtoD, DtoH, DtoD) | Yes | Yes | No | No |
| **Memory set tracking** | Yes | Yes | Yes | No | No |

### Runtime & Driver API Tracing

| API Layer | CUDA | ROCm | XPU | TPU | HPU |
|---|---|---|---|---|---|
| **Runtime APIs** | `cuda_runtime` | HIP runtime | `XPU_RUNTIME` (SYCL) | N/A | No |
| **Driver APIs** | CUPTI driver | No | `XPU_DRIVER` (L0) | N/A | No |
| **API call names** | Yes | Yes | Yes | N/A | No |
| **Correlation to kernels** | Yes | Yes | Yes | N/A | No |

### Flow Events & Correlation

| Feature | CUDA | ROCm | XPU | TPU | HPU |
|---|---|---|---|---|---|
| **CPU-to-device flow arrows** | Yes (`ac2g`) | Yes | Yes | N/A | No |
| **Forward/backward markers** | Yes (`fwdbwd`) | Yes | Yes | N/A | No |
| **Multi-stream visualization** | Yes | Yes | Yes | N/A | No |

### Profiler Output Formats

| Format | CUDA | ROCm | XPU | TPU | HPU |
|---|---|---|---|---|---|
| `key_averages().table()` | Yes | Yes | Yes | No (use XProf) | Yes |
| `export_chrome_trace()` | Yes | Yes | Yes | No | Yes |
| `export_stacks()` | Yes | Yes | Yes | No | Yes |
| `export_memory_timeline()` | Yes | Yes | Yes | No | Limited |
| TensorBoard integration | Yes | Yes | Yes | Yes (primary) | Limited |

---

## 9. CPU-Side Operator Information (Common Across All Backends)

These fields are captured regardless of accelerator (they come from PyTorch's CPU-side tracing):

| Field | Description | Requires |
|---|---|---|
| Operator name | e.g., `aten::addmm`, `aten::linear` | Always |
| Duration (CPU) | Wall-clock time on CPU | Always |
| Process/thread ID | PID and TID | Always |
| External ID | Unique event identifier | Always |
| Sequence number | Autograd sequence | Always |
| **Input shapes** | Tensor dimensions | `record_shapes=True` |
| **Input types** | Data types (float, half, etc.) | `record_shapes=True` |
| **Input strides** | Memory layout strides | `record_shapes=True` |
| **Concrete inputs** | Scalar values | `record_shapes=True` |
| **Python stack trace** | Source file and line | `with_stack=True` |
| **FLOPS estimate** | Floating-point ops | `with_flops=True` |
| **Module hierarchy** | nn.Module name | `with_modules=True` |

---

## 10. Key Gaps by Accelerator

| Missing Feature | Impact |
|---|---|
| ROCm: No registers, no occupancy | Cannot optimize for register pressure or occupancy on AMD GPUs via PyTorch profiler alone |
| TPU: Separate profiling system | No unified profiling across CPU+TPU in a single trace |
| XPU: No kernel dimensions | Cannot analyze work distribution on Intel GPUs |
| HPU: Minimal integration | Must rely on SynapseAI for real hardware profiling |
| All non-CUDA: No occupancy | CUPTI's occupancy estimation is not replicated for other backends |

---

## 11. Unique Characteristics Per Accelerator

### CUDA (NVIDIA)
- **Richest profiling data** of any accelerator
- Unique: register usage, occupancy estimation, blocks/warps per SM
- CUPTI provides deep hardware visibility
- Supports CUDA Graphs profiling
- NCCL distributed communication profiling

### ROCm (AMD)
- Good kernel-level timing but **missing hardware utilization metrics**
- `rocprof` CLI provides additional hardware counters (VGPR, SGPR, cache hit rates) but these are NOT in PyTorch profiler
- Uses wavefronts (64 threads) instead of warps (32 threads)
- LDS (Local Data Share) reported as combined value, not split static/dynamic

### Intel XPU
- **Richest scheduling metadata**: append, submit, start, end timestamps
- No kernel geometry (grid/block) in current implementation
- Driver-level (Level Zero) API tracing available
- Built-in overhead tracking

### Google TPU
- **Completely separate profiling system** (XProf + TensorBoard)
- Different compute model: systolic array (MXU), not thread-based
- No grid/block concept -- uses tiles
- HBM profiling instead of shared/global memory
- Best for multi-host/pod profiling

### Habana HPU (Gaudi)
- **Most minimal PyTorch integration**
- Must use SynapseAI Profiler for hardware details
- TPC (Tensor Processing Core) architecture, different from GPU SMs

---

## 12. Summary: Information Richness

```
Information Richness (for hardware profiling):
  CUDA ████████████████████ (100%) -- Full kernel details + occupancy + registers
  ROCm █████████████        (65%)  -- Kernel timing + dimensions, no occupancy
  TPU  ████████████         (60%)  -- Separate system, good coverage but different model
  XPU  ██████████           (50%)  -- Timing + scheduling, no dimensions
  HPU  ████                 (20%)  -- Basic timing only

Information Richness (for CPU-side tracing):
  All  ████████████████████ (100%) -- Same CPU-side info regardless of device
```

### For Performance Optimization, You Need:

| Goal | CUDA | ROCm | XPU | TPU | HPU |
|---|---|---|---|---|---|
| Identify slow kernels | PyTorch Profiler | PyTorch Profiler | PyTorch Profiler | XProf | SynapseAI |
| Optimize occupancy | PyTorch Profiler | `rocprof` | External tools | N/A | SynapseAI |
| Memory optimization | PyTorch Profiler | PyTorch Profiler | PyTorch Profiler | XProf | SynapseAI |
| Distributed perf | PyTorch Profiler + NCCL | PyTorch Profiler + RCCL | PyTorch Profiler | XProf | SynapseAI |
| Kernel tuning | Nsight Compute | `rocprof` | VTune/Advisor | N/A | SynapseAI |
