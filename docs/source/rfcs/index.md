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
| [0171](https://github.com/torch-spyre/rfcs/blob/main/0171-SpyreDevice/0171-SpyreDeviceRFC.md) | Spyre Device Construct in PyTorch | Device integration |
| [0186](https://github.com/torch-spyre/rfcs/blob/main/0186-TestFrameworks/0186-TestFrameworks.md) | Test Frameworks | Testing |
| [0601](https://github.com/torch-spyre/rfcs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md) | Spyre Profiling Toolkit | Profiling |
| [0682](https://github.com/torch-spyre/rfcs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md) | Kernel Tile Intermediate Representation | Compiler IR |
| [1287](https://github.com/torch-spyre/rfcs/blob/main/1287-SpyreTestFramework/1287-SpyreTestFrameworkRFC.md) | Test Suite Configuration for Upstream PyTorch Tests on OOT Devices | Testing |
| [1632](https://github.com/torch-spyre/rfcs/blob/main/1632-ModelEnablement/1632-ModelEnablement.md) | Model Enablement Tracking | Model enablement |
| [1633](https://github.com/torch-spyre/rfcs/blob/main/1633-E2EModelPerf/1633-E2EModelPerf.md) | End-to-End Model Performance Testing | Performance |

## Summaries

### RFC 0047 — Tensors with Device-Specific Layouts

Defines the Spyre tiled tensor layout model: `device_size`, `stride_map`, and the
stick abstraction. Motivates why PyTorch's single-stride-per-dimension layout
cannot represent tiled tensors, and specifies the `SpyreTensorLayout` data
structure that maps between PyTorch coordinates and Spyre device memory.

See also: [Tensor Layouts](../user_guide/tensors_and_layouts.md)

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

### RFC 1287 — Test Suite Configuration for Upstream PyTorch Tests on OOT Devices

Defines a YAML-based configuration schema (driven by `PYTORCH_TEST_CONFIG`)
that lets out-of-tree backends like Spyre reuse PyTorch's upstream test
suite without drowning in noise. OOT teams declare supported ops, dtypes,
and devices, and can selectively skip or xfail upstream tests, override
tolerances, inject custom inputs, and tag variants.

### RFC 1632 — Model Enablement Tracking

Describes how to systematically measure and track progress toward enabling
models on Spyre. Recommends using vLLM (rather than HuggingFace) model
definitions when discovering ops and modules, since vLLM definitions match
what actually ships in production. Proposes a dashboard with two metrics
per model — percentage of ops covered in `torch-spyre` and percentage of
modules covered in `vllm-spyre` — supplemented by hybrid end-to-end tests
where unenabled modules fall back to CPU.

### RFC 1633 — End-to-End Model Performance Testing

Consolidates fragmented performance measurement (PELE, fmwork, OLMES,
BFCL, etc.) around vLLM as the backend so regressions, output mismatches,
and quality issues surface systematically. Covers three measurement
dimensions — correctness against HuggingFace references, benchmarking
(latency, throughput, TTFT, ITL), and quality evals (GSM8K, MMLU, and
use-case-specific benchmarks) — leaning on upstream tooling such as
`HfRunner`, `VLLMRunner`, `vllm bench`, and `lm-evaluation-harness`.
