# Glossary

This page is the lookup reference for terms used across Torch-Spyre
documentation. For the conceptual primer that introduces these terms in
context, see [Key concepts](key_concepts.md).

Other pages can reference any entry here with the MyST `term` role:
`` {term}`stickification` `` renders as a hyperlink to the definition
below.

:::{glossary}
BYTES_IN_STICK
  The 128-byte alignment constant used by the runtime, compiler, and
  tensor-layout code. One stick at fp16 holds 64 elements. The size
  matches the natural granularity of LPDDR5 ↔ LX scratchpad transfers
  on Spyre. See [Key concepts §4](key_concepts.md#4-sticks-and-tiled-tensors).

corelet
  One of two execution units inside a Spyre core. Each corelet contains
  an 8×8 systolic PE array (the {term}`PT` execution unit) and a 1D
  {term}`SFU` vector unit. Both corelets in a core share the same 2 MB
  {term}`LX scratchpad`.

dataflow
  An execution model in which operations fire as soon as their inputs
  are ready, rather than being driven by a program counter. Spyre
  executes a compile-time-scheduled dataflow graph, which is what gives
  it deterministic latency. See [Dataflow Accelerator Architecture](../architecture/dataflow_architecture.md).

DCI
  Data Conversion Information. The `DataConversionInfo` struct (built
  by `generate_dci()` in `spyre_mem.cpp`) that bundles loop ranges,
  host and device strides, and dtype info. The runtime feeds it to
  `copyAsync` to drive a host ↔ LPDDR5 DMA transfer.

decomposition
  An FX graph rewrite that turns one ATen op into a sequence of
  Spyre-native or custom ops. Example: `aten.addmm` decomposes into
  `matmul + scale + add`. Decompositions are how Torch-Spyre covers
  ATen ops that have no single hardware-level equivalent. See
  [Key concepts §6](key_concepts.md#6-graph-breaks).

Deeptools
  IBM's proprietary backend compiler that consumes the {term}`SuperDSC`
  JSON IR and emits a Spyre device binary. Torch-Spyre is the
  open-source frontend; Deeptools is the closed backend. See
  [Compiler architecture](../compiler/architecture.md).

DMA
  Direct Memory Access. On Spyre, the PCIe path that carries tensor
  data between host memory and the device's LPDDR5.

FixedTiledLayout
  A Torch-Spyre subclass of Inductor's `FixedLayout` that augments the
  PyTorch `(size, stride)` description with a {term}`SpyreTensorLayout`
  carrying tiled device-side shape, a host-to-device stride map, and
  the device dtype. This is the layout abstraction that makes tiled
  tensors representable inside Inductor. See
  [Tensor Layouts](../user_guide/tensors_and_layouts.md).

flex runtime
  The Spyre device runtime that the C++ `SpyreAllocator` wraps. It
  owns the underlying device memory and issues kernel launches without
  exposing raw pointers (an IBM Z security requirement).

fold
  An affine-transform parameterization in {term}`SuperDSC` (`alpha *
  index + beta`) that lets one JSON artifact describe the per-core
  behavior of all 32 cores compactly. Fold properties cover core,
  corelet, row, and time dimensions.

graph break
  An interruption inside a `torch.compile`-d region where Inductor
  cannot lower an op, so the partial result round-trips to the CPU,
  the unsupported op runs there, and the data comes back. A single
  graph break in the hot path can wipe out the performance gains from
  surrounding compiled code. See
  [Key concepts §6](key_concepts.md#6-graph-breaks).

HBM
  In SuperDSC field names (e.g. `memOrg_.hbm`), `hbm` is a legacy
  label for device memory in general. Spyre's actual device memory is
  {term}`LPDDR5`, not HBM.

KTIR
  KernelTile IR. The MLIR-based dialect designed as the successor to
  {term}`SuperDSC`. KTIR generalizes the SuperDSC concepts (compute
  tiles, scratchpad staging, compile-time core partitioning) into a
  community specification for any dataflow accelerator. See the
  [KTIR RFC](https://github.com/torch-spyre/rfcs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md).

LPDDR5
  Spyre's off-chip device memory. 128 GB on the PCIe card. Equivalent
  in role to a GPU's HBM, but with a different memory technology and a
  different cost/bandwidth profile. The legacy {term}`HBM` field name
  in SuperDSC refers to LPDDR5.

LX planning
  The compiler pass that decides which tensors live in the
  {term}`LX scratchpad` versus {term}`LPDDR5` at each point in the
  computation. See [Scratchpad Planning](../compiler/scratchpad_planning.md).

LX scratchpad
  The 2 MB SRAM scratchpad on each Spyre core. Compiler-managed —
  there is no hardware cache. Both corelets in a core share the same
  scratchpad. See [Key concepts §3](key_concepts.md#3-memory-hierarchy).

OpFunc
  A {term}`Deeptools` primitive that implements one hardware operation
  on Spyre. Native ATen ops map to single OpFuncs; custom ops lower to
  one or more OpFuncs. SuperDSC `computeOp_` entries reference OpFuncs
  by name.

PE array
  Processing Element array. An 8×8 systolic array of multiply-accumulate
  units inside each corelet, used for matrix-style compute through the
  {term}`PT` execution unit.

PrivateUse1
  PyTorch's official extension mechanism for out-of-tree backends.
  Torch-Spyre uses it to register `"spyre"` as a first-class device
  name without forking PyTorch. See
  [Runtime](../runtime/index.md).

PT
  The Processing Tensor execution unit on each corelet. Backed by the
  {term}`PE array`, it runs matrix-style compute (matmul, fused
  pointwise epilogues). The other unit on a corelet is the {term}`SFU`.

restickify
  An explicit re-tile op (`spyre::restickify`) inserted by the
  `insert_restickify` compiler pass when two adjacent operations
  disagree on tile structure. Preserves correctness when layout
  propagation cannot pick one consistent tiling for the producer and
  consumer. See [Inductor frontend](../compiler/inductor_frontend.md).

SDSC
  See {term}`SuperDSC`. The two terms are interchangeable in code and
  filenames (e.g. `sdsc_0.json`, `generate_sdsc()`).

SENCORES
  The number of Spyre cores the compiler targets. Default 32 (one full
  card). Lowering it via the `SENCORES` environment variable is
  primarily a debugging tool; it changes work-division decisions and
  can be useful for isolating per-core behavior.

SFP
  See {term}`SFU`. Used interchangeably in some code paths.

SFU
  Special Function Unit (sometimes Special Function Processor, SFP).
  The 1D vector unit on each corelet that handles non-linear
  activations such as GELU, softmax, and other element-wise functions
  the {term}`PT` unit does not implement.

span reduction
  The first of two work-division passes (`span_reduction()`). Analyzes
  the iteration space of each reduction and determines how its span
  can be split across cores. Followed by {term}`work distribution`.
  See [Work Division Planning](../compiler/work_division_planning.md).

SPMD
  Single Program, Multiple Data. Every core runs the same program on
  its own slice of the data, picked by core ID. Spyre's execution
  model is SPMD across the 32 cores.

SpyreTensorImpl
  The C++ subclass of `at::TensorImpl` that carries Spyre-specific
  layout metadata (a {term}`SpyreTensorLayout`) alongside the standard
  PyTorch tensor fields. Registered through the {term}`PrivateUse1`
  hook system.

SpyreTensorLayout
  The descriptor inside a {term}`SpyreTensorImpl` (and a
  {term}`FixedTiledLayout`) that carries the tiled device-side size,
  a stride map from host dimensions to device dimensions, and the
  device dtype.

stick
  A 128-byte aligned memory chunk on Spyre, equal to 64 fp16 elements.
  The unit of LPDDR5 ↔ LX transfer and the basic granularity of the
  tiled tensor layout. Defined by the {term}`BYTES_IN_STICK` constant.

stickification
  The transformation from a host-strided PyTorch layout to a tiled
  Spyre device layout. Run during the `propagate_spyre_tensor_layouts`
  pass on the LoopLevel IR. After this pass every `ComputedBuffer`
  carries a {term}`FixedTiledLayout`. See
  [Inductor frontend](../compiler/inductor_frontend.md).

SuperDSC
  Super Design Space Config. Torch-Spyre's current JSON-based IR. One
  artifact per scheduled kernel; encodes the per-core schedule, tensor
  descriptors, memory addresses, and the compute op. Cached through
  the standard `torch.compile` artifact system. The successor is
  {term}`KTIR`. See [Key concepts §7](key_concepts.md#7-compilation-pipeline).

tile
  A contiguous sub-tensor assigned to a single core. On Spyre, a tile
  is built from one or more {term}`stick`s.

work distribution
  The second of two work-division passes (`work_distribution()`).
  Assigns the spans identified by {term}`span reduction` to the 32
  cores. Enforces equal stick counts per core (no load imbalance) and
  per-core addressable device memory limits. See
  [Work Division Planning](../compiler/work_division_planning.md).

work slice
  The slice of the iteration space assigned to a single core by
  {term}`work distribution`. Encoded in SuperDSC as `coreIdToWkSlice_`
  and `numWkSlicesPerDim_`.
:::
