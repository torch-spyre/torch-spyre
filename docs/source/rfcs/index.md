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

See also: [Profiling](../user_guide/profiling.md)

### RFC 0682 — Kernel Tile Intermediate Representation (KTIR)

Defines the Kernel Tile IR — an MLIR-based data-parallel intermediate
representation that replaces SuperDSC bundles as the target for the
Torch-Spyre compiler back-end. KTIR expresses tile-level operations,
scratchpad allocation, and the load/store traffic between device
memory and scratchpad in a hardware-independent form that DeepTools
then lowers to device-specific code.

See also: [Compiler Backend](../compiler/backend.md)
