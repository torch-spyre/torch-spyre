# RFCs

This section lists the Request For Comments (RFCs) that describe the design
decisions behind Torch-Spyre. RFCs are written before implementation and serve
as a record of why things are built the way they are.

The full RFC sources live in the
[`torch-spyre/rfcs`](https://github.com/torch-spyre/rfcs)
repository. To propose a new RFC, open an issue first, then
copy the
[template](https://github.com/torch-spyre/rfcs/tree/main/NNNN-template)
and submit a pull request.

## Index

| RFC | Title | Area |
|-----|-------|------|
| [0047](https://github.com/torch-spyre/rfcs/blob/main/0047-TiledTensors/0047-TiledTensorsRFC.md) | Tensors with Device-Specific Layouts | Tensor layouts |
| [0099](https://github.com/torch-spyre/rfcs/blob/main/0099-MultiDevice/0099-MultiDeviceRFC.md) | Multi-Spyre Device Support in PyTorch | Distributed |
| [0171](https://github.com/torch-spyre/rfcs/blob/main/0171-SpyreDevice/0171-SpyreDeviceRFC.md) | Spyre Device Construct in PyTorch | Device integration |
| [0186](https://github.com/torch-spyre/rfcs/blob/main/0186-TestFrameworks/0186-TestFrameworks.md) | Test Frameworks | Testing |
| [0601](https://github.com/torch-spyre/rfcs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md) | Spyre Profiling Toolkit | Profiling |
| [0682](https://github.com/torch-spyre/rfcs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md) | Kernel Tile Intermediate Representation | Compiler IR |
| [1069](https://github.com/torch-spyre/rfcs/blob/main/1069-SpyreTensorLayoutExtraction/1069-SpyreTensorLayoutExtraction.md) | SpyreTensorLayout Extraction via CPU Compilation | Tensor layouts |
| [1287](https://github.com/torch-spyre/rfcs/blob/main/1287-SpyreTestFramework/1287-SpyreTestFrameworkRFC.md) | Test Suite Configuration for Upstream PyTorch Tests on OOT Devices | Testing |
| [1358](https://github.com/torch-spyre/rfcs/blob/main/1358-CoarseTiling/1358-CoarseTiling.md) | Coarse Tiling | Compiler |
| [1632](https://github.com/torch-spyre/rfcs/blob/main/1632-ModelEnablement/1632-ModelEnablement.md) | Model Enablement Tracking | Model enablement |
| [1632-v2](https://github.com/torch-spyre/rfcs/blob/main/1632-ModelEnablement-v2/1632-ModelEnablement-v2.md) | Model Enablement v2 | Model enablement |
| [1633](https://github.com/torch-spyre/rfcs/blob/main/1633-E2EModelPerf/1633-E2EModelPerf.md) | End-to-End Model Performance Testing | Performance |
| [2676](https://github.com/torch-spyre/rfcs/blob/main/2676-SpyreMetricsApiExtension/2676-SpyreMetricsApiExtensionRFC.md) | Spyre Metrics API Extension Package | Profiling |
| [2696](https://github.com/torch-spyre/rfcs/blob/main/2696-AiuSmiExtension/2696-AiuSmiExtensionRFC.md) | aiu-smi Extension Package | Profiling |
| [2971](https://github.com/torch-spyre/rfcs/blob/main/2971-FP32ElementArrangement/2971-FP32ElementArrangementRFC.md) | FP32 Element Arrangement | Compiler |

## Summaries

### RFC 0047 — Tensors with Device-Specific Layouts

Defines the Spyre tiled tensor layout model: `device_size`, `stride_map`, and the
stick abstraction. Motivates why PyTorch's single-stride-per-dimension layout
cannot represent tiled tensors, and specifies the `SpyreTensorLayout` data
structure that maps between PyTorch coordinates and Spyre device memory.

See also: [Tensor Layouts](../user_guide/tensors_and_layouts.md)

### RFC 0099 — Multi-Spyre Device Support in PyTorch

Describes the additions required to bring a Spyre-based collective communication
library into the PyTorch ecosystem. Specifies a module that implements the
PyTorch distributed interfaces and registers as the default process group for
Spyre devices, backed by an external Spyre communication library.

### RFC 0171 — Spyre Device Construct in PyTorch

Describes how Spyre integrates as a first-class PyTorch device: registration
via `PrivateUse1`, dispatch keys, allocator, and the `torch.compile` Inductor
backend hook. Covers the design choices behind device naming and the extension
mechanism used to avoid upstream PyTorch changes.

See also: [Architecture Overview](../architecture/index.rst)

### RFC 0186 — Test Frameworks

Defines the testing frameworks and conventions used by torch-spyre, including
the compiled-path test infrastructure, the `ParameterizedTestMeta` metaclass,
and the `compare_with_cpu` utility for validating Spyre results against CPU
reference outputs.

### RFC 0601 — Spyre Profiling Toolkit

Proposes a set of profiling tools spanning the full stack — from PyTorch-level
execution traces to device-level hardware metrics. Covers PyTorch Profiler
integration via `REGISTER_PRIVATEUSE1_PROFILER`, dual-memory profiling (DDR
and scratchpad), AIU SMI for device monitoring, IR instrumentation-based
fine-grained profiling, and the Holistic Trace Analyser for Spyre.

See also: [Profiling](../user_guide/profiling/index.md)

### RFC 0682 — Kernel Tile Intermediate Representation (KTIR)

Defines the Kernel Tile IR — an MLIR-based data-parallel intermediate
representation that replaces SuperDSC bundles as the target for the
Torch-Spyre compiler back-end. KTIR expresses tile-level operations,
scratchpad allocation, and the load/store traffic between device
memory and scratchpad in a hardware-independent form that DeepTools
then lowers to device-specific code.

See also: [Compiler Backend](../compiler/backend.md)

### RFC 1069 — SpyreTensorLayout Extraction via CPU Compilation

Proposes capturing the `SpyreTensorLayout` for each operation in a PyTorch model
by compiling and running the model on the CPU. Using the CPU as a proxy for
Spyre execution extracts the exact layouts expected on Spyre without requiring
full operator support on the target backend.

See also: [Tensor Layouts](../user_guide/tensors_and_layouts.md)

### RFC 1287 — Test Suite Configuration for Upstream PyTorch Tests on OOT Devices

Defines a YAML-based configuration schema (driven by `PYTORCH_TEST_CONFIG`)
that lets out-of-tree backends like Spyre reuse PyTorch's upstream test
suite without drowning in noise. OOT teams declare supported ops, dtypes,
and devices, and can selectively skip or xfail upstream tests, override
tolerances, inject custom inputs, and tag variants.

### RFC 1358 — Coarse Tiling

Records the motivation and design rationale behind coarse-level tiling in
the compiler: take a sequence of operations that share an iteration-space
dimension, split that dimension into K (possibly symbolic) chunks, and emit
the body inside a counted outer loop. This time-domain tiling is the key
transformation for working-set reduction — it reshapes the computation so
most tensors fit in the LX scratchpad — and the RFC captures the constraints
that forced each choice in the loop IR design.

See also: [Coarse-Tiling Loop IR](../compiler/coarse_tiling_loops.md)

### RFC 1632 — Model Enablement Tracking

Describes how to systematically measure and track progress toward enabling
models on Spyre. Recommends using vLLM (rather than HuggingFace) model
definitions when discovering ops and modules, since vLLM definitions match
what actually ships in production. Proposes a dashboard with two metrics
per model — percentage of ops covered in `torch-spyre` and percentage of
modules covered in `vllm-spyre` — supplemented by hybrid end-to-end tests
where unenabled modules fall back to CPU.

### RFC 1632-v2 — Model Enablement v2

Supersedes the original tracking framework with a label-based, test-driven
approach built around the `hf-adapters` library, which enables models
through minimal runtime patches rather than custom forks. Defines five test
types — covering model loading, smoke testing, embedding comparison, and
token-level comparison — organized into a tiered CI/CD structure spanning
PR-level, daily, and weekly regression checks, so teams can reliably assess
which models are production-ready.

### RFC 1633 — End-to-End Model Performance Testing

Consolidates fragmented performance measurement (PELE, fmwork, OLMES,
BFCL, etc.) around vLLM as the backend so regressions, output mismatches,
and quality issues surface systematically. Covers three measurement
dimensions — correctness against HuggingFace references, benchmarking
(latency, throughput, TTFT, ITL), and quality evals (GSM8K, MMLU, and
use-case-specific benchmarks) — leaning on upstream tooling such as
`HfRunner`, `VLLMRunner`, `vllm bench`, and `lm-evaluation-harness`.

### RFC 2676 — Spyre Metrics API Extension Package

Proposes hosting `spyre-metrics-api` (Python import name `spyremetrics`) inside
the torch-spyre repository under a top-level `extensions/` directory, as an
independently packaged and versioned Python distribution. The package provides a
public Python API for reading and parsing IBM Spyre performance metric files
that contain hardware performance counters emitted by the Spyre runtime and
driver layer. It abstracts the binary format, supports both PF- and VF-based
metric formats, and exposes typed metric definitions with units and scaling.

### RFC 2696 — aiu-smi Extension Package

Proposes hosting `aiu-smi`, a performance monitoring tool for the IBM Spyre
accelerator, inside the torch-spyre repository under the `extensions/`
directory, as an independently packaged and versioned Python distribution.
`aiu-smi` periodically reads Spyre metric data and prints per-device telemetry
(power, temperature, busy percentage, PT-array utilization, memory bandwidth,
reserved, active, and peak memory, host CPU and memory, and process mapping) in
text or CSV form. It consumes the `spyre-metrics-api` extension.

### RFC 2971 — FP32 Element Arrangement

Specifies how widening a tensor's elements on device (16-bit `DL16` or `BF16` to
32-bit `FP32`) leaves them staggered rather than in standard stick order: all
values are correct, but within-stick position no longer matches logical order.
The Inductor backend tracks this as an Element Arrangement (EA) per layout and
gates op legality on it through `is_ea_compatible` and `validate_ops`. FP32 is
ephemeral, never stored, and only a transient widening.
