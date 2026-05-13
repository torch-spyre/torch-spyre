# How Torch-Spyre works: an out-of-tree PyTorch backend

> **Summary.** Torch-Spyre is an out-of-tree PyTorch backend for the IBM Spyre dataflow
> accelerator. It does not fork PyTorch. It uses the extension points that PyTorch already
> provides: PrivateUse1, the TorchInductor backend API, and custom tensor layouts. Torch-Spyre
> runs model inference on Spyre today through IBM's Foundation Model Stack (FMS). Granite 3.3 8B
> is one of the models we run in production. We are close to running an unmodified HuggingFace
> model end to end with just `model.to("spyre")` and `torch.compile`. This page describes how the
> integration was built.

This page is written for new team members, hardware collaborators, and anyone who wants to
understand how Torch-Spyre integrates with PyTorch as an out-of-tree backend. It walks through
the four main design challenges we hit and the PyTorch extension mechanism that addressed each
one. API names, op lists, and pass hooks reflect the state of the codebase at the time of
writing. For current state, see the rest of this documentation and the
[repo](https://github.com/torch-spyre/torch-spyre).

---

## A device with a different execution model from a GPU

The code below runs Granite 3.3 8B on Spyre today through IBM's Foundation Model Stack (FMS).

```python
import torch
from fms.models import get_model

model = get_model("granite", "3.3-8b-instruct", device_type="spyre")
compiled = torch.compile(model, backend="spyre")
output = compiled(input_ids.to("spyre"))
```

FMS wraps the model definition. Everything below that line is standard PyTorch: `.to("spyre")`,
`torch.compile`, and a device string. We are working toward the same snippet with a stock
HuggingFace `AutoModelForCausalLM` in place of `get_model`, with no modifications.

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-8b-instruct")
model = model.to("spyre")
compiled = torch.compile(model)
```

Closing the remaining gap (op coverage, dynamic shapes, KV-cache handling) is active work. The
rest of this post explains the extension points that got us this far.

The IBM Spyre Accelerator does not operate the way PyTorch assumes hardware operates. Spyre is a
32-core dataflow processor that delivers over 300 TOPS within a 75W PCIe card. It is designed for
enterprise AI inference on IBM Z and Power systems. Its design priorities are deterministic
latency, hardware-level security isolation, and power efficiency. Meeting those priorities
required an architecture that differs from a GPU in several ways.

![Four challenges and the PyTorch extension mechanisms that addressed each one](../_static/images/how-torch-spyre-works/fig0-journey-overview.svg)

*Figure 0: The four challenges we address in this post, with the PyTorch extensibility mechanism
that handled each one.*

> **From research to production.** Since this work began, Spyre has shipped in two production
> systems: as the AI accelerator in IBM z17 mainframes (which support up to 48 Spyre cards, each
> delivering 300+ TOPS) and in IBM Power11 servers via a 75W PCIe gen5 x16 card with 128 GB of
> LPDDR5 memory.
> The Torch-Spyre integration described in this post targets that PCIe card configuration.
> [[IBM Z press release]](https://newsroom.ibm.com/ai-on-z) · [[IBM Power11 press release]](https://newsroom.ibm.com/2025-07-08-ibm-power11-raises-the-bar-for-enterprise-it) · [[IBM Research blog]](https://research.ibm.com/blog/spyre-for-z)

![IBM Spyre Accelerator PCIe card](../_static/images/how-torch-spyre-works/spyre-pcie-card.png)

*The IBM Spyre™ Accelerator. It delivers over 300 TOPS within a 75W PCIe gen5 x16 form factor
and ships with 128 GB of LPDDR5. Photo: IBM.*

![Spyre Core Architecture: 2 Corelets with Shared LX Scratchpad](../_static/images/how-torch-spyre-works/fig-spyre-core-architecture.svg)

*Figure 1a: Each Spyre core contains two corelets (each with an 8×8 systolic Processing Element (PE) array and a 1D Special Function (SFU) vector unit) sharing a 2 MB LX scratchpad (SRAM). Cores communicate via a bi-directional ring interconnect at 128 B/cycle per direction. Architecture derived from the [RaPiD AI accelerator](https://doi.org/10.1109/ISCA52012.2021.00021) (Venkataramani et al., ISCA 2021).*

**Dataflow execution instead of SIMT.** On a GPU, thousands of threads execute the same
instruction on different data in lock-step. This is the Single Instruction, Multiple Threads
(SIMT) model. SIMT works well when arithmetic intensity is high and memory access patterns are
regular. The decode phase of LLM inference has the opposite characteristics. It is
memory-bandwidth bound, execution is sequential across layers, and attention memory access
patterns can be irregular. Spyre addresses this with a dataflow execution model. Operations form
a directed acyclic graph. Data flows through a chain of operators, and the hardware is configured
per operation at compile time. Because the execution schedule is fixed ahead of time, latency is
deterministic. There is no thread scheduling jitter and no dynamic dispatch overhead. There are
no warps, no thread blocks, and no shared memory in the CUDA sense.

**Tiled memory instead of strided memory.** The tiled layout was chosen to match the hardware.
The 128-byte aligned sticks of 64 fp16 elements (a constant we call `BYTES_IN_STICK=128`) match
the natural granularity of data transfers between DDR and scratchpad on the internal datapaths.
This lets the hardware load a full stick of contiguous elements in a single transfer, so each
memory access delivers as much useful data as possible. The tradeoff is that PyTorch's
`(size, stride)` model cannot express this layout. A 2D tensor stored as a tiled 3D structure has
no single integer stride for the second dimension. The stride jumps at every tile boundary.

**Explicit data movement instead of hardware caches.** Implicit caches introduce
non-determinism. Eviction policy, cache pressure from other workloads, and prefetcher heuristics
all affect latency in ways that the software stack cannot predict or control. Spyre's 2 MB
programmable scratchpad per core removes that uncertainty. The compiler decides exactly what
lives in Static Random-Access Memory (SRAM) at each point in the computation. Data moves from
Spyre's 128 GB LPDDR5 device memory into the 2 MB per-core SRAM scratchpad only when the compiler
has scheduled that transfer. This is what allows Spyre to guarantee latency bounds that GPU-class
hardware typically cannot. The cost is that every data transfer must be deliberate. The
Torch-Spyre backend is responsible for allocating tensors to the scratchpad. The backend compiler
(Deeptools) handles the actual scheduling of data movement between DDR and the scratchpad,
including double-buffering.

![Spyre memory hierarchy and data flow](../_static/images/how-torch-spyre-works/fig1-memory-hierarchy.svg)

*Figure 1: All data movement is planned by the compiler through explicit load and store
instructions. Data moves between 128 GB of LPDDR5 and a 2 MB per-core SRAM scratchpad. Spyre has
no hardware cache.*

![Execution latency, GPU compared to Spyre](../_static/images/how-torch-spyre-works/figA-latency-comparison.svg)

*Figure 1b (illustrative). GPU execution is subject to thread scheduling jitter, cache
evictions, and dynamic dispatch overhead. The compiler-planned dataflow on Spyre produces a flat,
deterministic latency profile for the same model. Latency values in the figure are illustrative.
Exact results depend on the model and batch configuration.*

---

## Challenge 1: making PyTorch recognize a new device

PyTorch reserves a mechanism called PrivateUse1 for out-of-tree backends. We based our
implementation on [OpenReg](https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension),
PyTorch's reference implementation for out-of-tree device registration. Two lines of Python rename
the device.

```python
torch.utils.rename_privateuse1_backend("spyre")
torch._register_device_module("spyre", make_spyre_module())
```

After that, `"spyre"` is a first-class device name. Those two calls register the dispatch tables,
the `.to("spyre")` method on tensors and modules, and device string parsing.

Renaming the device is only the entry point. We also had to implement the C++ infrastructure
that gives the device its actual behavior. That infrastructure includes a device guard that
tracks context through thread-local storage, a custom allocator that wraps the Spyre flex memory
system without exposing raw pointers (a security requirement on IBM Z), a custom `TensorImpl`
that carries tiled memory metadata alongside the standard PyTorch layout, a hooks interface that
reports device availability to the PyTorch runtime, and `SpyreStream`, which matches
`torch.cuda.Stream` semantics so that asynchronous execution works as users expect.

![Device registration architecture](../_static/images/how-torch-spyre-works/fig2-device-registration.svg)

*Figure 2: All Torch-Spyre components are out-of-tree and the PyTorch core is unmodified. The
blue path shows how `model.to("spyre")` flows through the PrivateUse1 hooks to the hardware.*

Initialization is lazy. Importing `torch_spyre` registers the device name and module, but the
C++ runtime does not start until the device is used for the first time.

PrivateUse1 covers most of what an out-of-tree backend needs. The dispatch infrastructure that
integrates GPUs with PyTorch is available to out-of-tree backends through the same pathway. The
basic extension points are well documented. `torch.accelerator` and OpenReg give a clear starting
point. The deeper extension points (scheduler passes and codegen hooks) are not well documented.
For those, we read the Ascend NPU and Intel GPU source code to understand the patterns.

---

## Challenge 2: teaching PyTorch a memory layout it had never seen

After the device was registered, PyTorch knew that Spyre existed. The harder problem was
tensors. Tensors on Spyre are not laid out the way PyTorch expects.

Consider a standard `(1024, 256)` fp16 tensor. In PyTorch this has strides `(256, 1)`. Element
`[i, j]` lives at offset `i × 256 + j`. This model works for row-major, column-major, and
transposed views. It works for almost everything in the GPU world.

On Spyre, the same tensor is physically stored as four tiles of 64-element sticks. Each stick is
128 bytes and holds 64 fp16 elements. The layout is organized as if the tensor had shape
`(4, 1024, 64)`. The element at position `[i, 63]` and the element at `[i, 64]` are not separated
by a stride of 1. They sit in different tiles with a jump between them. There is no single
integer stride that describes this layout, and the current PyTorch layout model cannot represent
it.

We needed a new abstraction. We built `FixedTiledLayout` as a subclass of the TorchInductor
`FixedLayout` class. `FixedTiledLayout` augments the standard host-side description with a
`SpyreTensorLayout` descriptor. The descriptor carries the tiled device-side size, a stride map
from host dimensions to device dimensions, and the device dtype. This is sufficient to describe
how data is stored on the device and how to move it between the host and device representations.

![Tensor layout transformation from host to device](../_static/images/how-torch-spyre-works/fig3-tensor-layout.svg)

*Figure 3: A (1024, 256) tensor becomes a (4, 1024, 64) tiled structure on Spyre. The stride
breaks at every tile boundary, so the layout cannot be expressed as a single integer stride.*

The harder question was where in the Inductor pipeline we should propagate these layouts. We
tried propagating through Dynamo using `FakeTensor`. That approach was too invasive and too
fragile to maintain out-of-tree. We then considered deferring propagation to the final codegen
stage, as Triton does. That would have been too late, because we need layout information for
memory planning and multi-core work division before code generation begins.

The right place turned out to be the Inductor scheduler phase, using custom pass hooks. We run
two kinds of passes. The first set runs over the FX graph before it is lowered to the Inductor
LoopLevel IR. The second set runs over the LoopLevel IR itself before codegen.

**Layout propagation** runs on the LoopLevel IR. A function called
`propagate_spyre_tensor_layouts()` traverses the scheduler graph and converts the standard
`FixedLayout` of each tensor into our `FixedTiledLayout`. The conversion rules depend on the
operator type. Pointwise ops inherit the layout of their inputs. Reductions, and matmul in
particular, need special handling because the stick dimension of the output is related to the
contracted dimension of the input. External kernels, which include custom ops and fallbacks,
receive a safe default tiled layout.

**Core work division** also runs on the LoopLevel IR, after layout propagation. Two passes
cooperate. `span_reduction()` identifies which iteration dimensions can be parallelized across
cores and how the span of each reduction partitions. `work_distribution()` then assigns those
spans to the 32 cores and embeds the plan into the IR for codegen to consume.

One caveat is worth mentioning. The scheduler pass hooks are underscore-prefixed private APIs.
They have been stable for us, but they can change between PyTorch releases without notice. We
plan for occasional breakage.

`FixedTiledLayout` is designed to be reusable beyond Spyre. Any accelerator with non-strided,
block-aligned memory, such as a systolic array chip, an FPGA design, or scratchpad-based
inference silicon, faces the same mismatch with the PyTorch stride model. We would like to see
this become a standard Inductor extension point.

---

## Challenge 3: extending TorchInductor for dataflow compilation

With a registered device and a way to express tiled tensors, the last challenge was code
generation. TorchInductor was designed with GPUs in mind. Its scheduler, codegen, and backend
assume thread blocks, dynamic dispatch, and Triton kernels. Spyre operates differently at every
level. Work is partitioned across cores at compile time instead of being dispatched by hardware
at runtime. The output is a tile-level IR instead of a GPU kernel. No one had extended Inductor
for this kind of architecture before.

The Inductor backend registration API is more composable than we expected. We provided our own
scheduling class, wrapper codegen, and device op overrides. All three are registered at import
time, and all three live out-of-tree. Those three registrations were enough to swap in a
completely different execution model without modifying the Inductor core. Our backend currently
produces SuperDSC, a JSON-based IR that describes tile-level compute graphs for the 32 cores of
Spyre. SuperDSC is an interim format. Its successor, KTIR, is discussed later in this page.

The most unusual piece in the pipeline is compile-time work division. On a GPU, the hardware
scheduler distributes thread blocks at runtime. On Spyre, the compiler makes that decision
statically. The compiler partitions each iteration dimension across 32 cores, distributes sticks
evenly across cores, respects per-core memory limits, and embeds the resulting plan directly into
the SuperDSC IR. The result is a self-contained compiled artifact that is cached through the
standard `torch.compile` caching system.

The full compilation path, from the `torch.compile` entry point through the Spyre-specific
passes to the device binary, is shown in Figure 4.

![Torch-Spyre compilation pipeline](../_static/images/how-torch-spyre-works/fig4-compilation-pipeline.svg)

*Figure 4: The Spyre-specific passes (orange) operate on two IR levels. The first set runs on
the FX Graph before Inductor lowering. The second set runs on the LoopLevel IR before codegen.
The gray boxes mark PyTorch-standard stages. The SuperDSC IR (blue) is the compiled artifact.
KTIR is the planned replacement.*

*The exact timing of these passes is still in flux. We keep finding cleaner extension points and
moving passes around. What does not change is what each pass does: layout propagation, work
division, and scratchpad planning.*

The compilation pipeline uses the standard PyTorch components without modification. These are
the FX graph capture, AOTAutograd, and the Inductor scheduler. The Spyre-specific passes were
created through the published extension APIs. We did not fork PyTorch. We do still monkey-patch
a few internal APIs where the extension points are incomplete. Those patches cover dtype maps,
fusion patterns, and compile_fx wrapping. The rest of the architecture is out-of-tree.

### SuperDSC: the Spyre tile-level intermediate representation

A GPU backend typically produces Triton kernels or CUDA templates and library calls. Torch-Spyre
produces **SuperDSC** (Super Design Space Config) instead. SuperDSC is a JSON-based intermediate
representation that describes the full tile-level compute graph for the 32 cores of Spyre.
Understanding SuperDSC is required to understand how the Torch-Spyre compiler backend differs
from a GPU backend.

#### What SuperDSC encodes

SuperDSC is not a kernel. It is a self-contained, compiled artifact that describes everything
the Spyre hardware needs to execute a single scheduled operation deterministically. The top-level
structure contains core fold properties, work-slice mappings, and a per-core execution schedule.
A `dscs_` array holds one or more `DesignSpaceConfig` entries. Each entry is a complete
description of one compute configuration, and contains the following elements.

- **Core fold properties** (`coreFoldProp_`, `numWkSlicesPerDim_`, `coreIdToWkSlice_`): how to
  divide the iteration space across 32 cores. For a tensor of shape `(1024, 256)`, this encodes
  how many rows each core processes. The encoding gives each core an equal number of sticks and
  keeps each core within its addressable device memory limit.
- **Tensor descriptors** (`labeledDs_`, `primaryDsInfo_`): for each tensor argument, the tiling
  structure define which dimensions are stick dimensions, how the host-side shape maps to device-side
  tiles, memory residency (HBM vs. LX scratchpad), data format, and which dimensions each tensor
  iterates over fully vs. which are summed over (contracted) as in the K dimension of a matmul.
- **Schedule tree** (`scheduleTree_`): a list of allocate nodes (one per tensor) that specify
  memory placement (HBM or LX scratchpad), dimension ordering, per-core start addresses via fold
  mappings, and coordinate information encoding how each dimension is split across cores with affine
  transformations.
- **Data staging** (`dataStageParam_`): per-core dimension sizes for steady-state and epilogue
  passes, describing how data is partitioned for transfer into scratchpad.
- **Compute operations** (`computeOp_`): one entry per operation, encoding the execution unit (PT
  or SFP), operation name, data format, fidelity, and the input/output tensor references from
  `labeledDs_`.

Folding is a central concept in SuperDSC. A single parameterized artifact can represent multiple
execution variants across time steps and cores without recompilation. Fold properties use affine
transformations (`alpha * index + beta`) to compute per-core coordinates and addresses. One JSON
file describes the behavior of all 32 cores compactly instead of duplicating the description for
each core.

#### The codegen pipeline

Three components cooperate to produce a SuperDSC artifact for each scheduled node.

1. `SpyreKernel` collects the iteration space from the scheduler and builds an RValue AST
   (Abstract Syntax Tree of read-side expressions) representing the computation with node types
   like `TensorAccess`, `PointwiseOp`, `ReductionOp`, and `Constant`. Leaves are tensor reads or constants; internal nodes are operations. This captures the
   mathematical structure in a form the codegen can walk through.
2. `OpSpec` packages the kernel's output into a structured descriptor: the operation name, the
   iteration space encoded as [SymPy](https://www.sympy.org/) symbolic math expressions (which
   Inductor uses throughout for symbolic shape reasoning), the tensor arguments annotated with device
   coordinates (tile index, intra-stick offset), and any auxiliary information like constants or dtype
   conversion rules.
3. `generate_sdsc()` takes the `OpSpec` and emits the final JSON IR. This is where symbolic
   expressions are resolved to concrete loop bounds, tiling parameters are expanded, and the schedule
   tree is assembled. The output is written to JSON files (e.g., `sdsc_0.json`) that the Spyre backend
   compiler (Deeptools) consumes to produce a device binary.

#### Multi-core work division

Two passes run after layout propagation, and together they handle one of the decisions that
differs most from a GPU. That decision is the static partitioning of work across all 32 cores
at compile time. `span_reduction()` analyzes the iteration space of each reduction and determines
how its span can be split. `work_distribution()` then assigns those spans to cores.

The algorithm does four things. It identifies which iteration dimensions can be parallelized
across cores. It uses a two-pass approach that first satisfies the minimum splits required per
dimension (some dimensions have indivisible constraints) and then distributes the remaining
cores by priority. It enforces that each core receives an equal number of sticks, so that there
is no load imbalance. It validates that the data footprint for each core fits within the
per-core addressable device memory range, which is a hardware address space constraint that is
distinct from the 2 MB SRAM scratchpad.

The result is embedded directly in the SuperDSC IR. Each core knows exactly which slice of the
iteration space it owns before the binary reaches the hardware. There is no runtime scheduler,
no dynamic load balancing, and no speculation. This is what makes Spyre latency deterministic.
Every aspect of execution is decided at compile time.

#### Example: SuperDSC for `abs` on a (4, 64) fp16 tensor with 2 cores

The following example shows what Torch-Spyre produces for `torch.abs` on a `(4, 64)` fp16 tensor
with 2 cores. The structural skeleton below shows the key decisions. The field names match the
actual JSON, and the values are simplified for readability. Inductor names each dimension of the
iteration space sequentially. `c0` is the first dimension (rows), `c1` is the second (columns),
and so on for higher-dimensional tensors.

```
{
  "abs": {
    "coreFoldProp_":    { "factor_": 2 },          // 2-core split
    "coreIdToWkSlice_": { "0": {"c0": 0},          // core 0 → rows 0–1
                          "1": {"c0": 1} },         // core 1 → rows 2–3
    "dscs_": [{
      "abs": {
        "N_":              {"c0_": 4, "c1_": 64},   // full iteration space
        "dataStageParam_": {"c0_": 2, "c1_": 64},   // per-core slice
        "labeledDs_":      ["Tensor0", "Tensor1"],   // both dsType "INPUT" (same layout)
        "scheduleTree_":   ["allocate Tensor0 @ core0:0, core1:256",
                            "allocate Tensor1 @ core0:512, core1:768"],
        "computeOp_":      "abs(Tensor0) → Tensor1 on PT unit"
      }
    }]
  }
}
```

Each core receives 2 rows of 64 elements, which is one stick per row. The `dscs_` entry
contains the full description, including tensor descriptors, per-core memory addresses, and the
compute operation. The corelet and row fold fields appear in the full artifact as pass-through
constants (always factor 1) that the backend compiler requires. Torch-Spyre reasons only at core
granularity. (Note: the `hbm` field name in the JSON is a legacy label in the SuperDSC IR that
refers to device memory in general. The actual device memory on Spyre is LPDDR5, not HBM.)

<details>
<summary><strong>Full SuperDSC JSON artifact</strong> (click to expand)</summary>

```json
{
  "abs": {
    "coreFoldProp_": {"factor_": 2, "label_": "core"},
    "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
    "numCoresUsed_": 2,
    "numWkSlicesPerDim_": {"c0": 2, "c1": 1},
    "coreIdToWkSlice_": {
      "0": {"c0": 0, "c1": 0},
      "1": {"c0": 1, "c1": 0}
    },
    "dscs_": [{
      "abs": {
        "numCoresUsed_": 2,
        "coreIdsUsed_": [0, 1],
        "N_": {"name_": "n", "c0_": 4, "c1_": 64},
        "dataStageParam_": {
          "0": {
            "ss_": {"name_": "core", "c0_": 2, "c1_": 64},
            "el_": {"name_": "core", "c0_": 2, "c1_": 64}
          }
        },
        "primaryDsInfo_": {
          "INPUT": {
            "layoutDimOrder_": ["c0", "c1"],
            "stickDimOrder_": ["c1"],
            "stickSize_": [64]
          }
        },
        "labeledDs_": [
          {
            "ldsIdx_": 0, "dsName_": "Tensor0", "dsType_": "INPUT",
            "scale_": [1, 1], "wordLength": 2,
            "dataFormat_": "SEN169_FP16",
            "memOrg_": {"hbm": {"isPresent": 1}, "lx": {"isPresent": 1}}
          },
          {
            "ldsIdx_": 1, "dsName_": "Tensor1", "dsType_": "INPUT",
            "scale_": [1, 1], "wordLength": 2,
            "dataFormat_": "SEN169_FP16",
            "memOrg_": {"hbm": {"isPresent": 1}, "lx": {"isPresent": 1}}
          }
        ],
        "scheduleTree_": [
          {
            "nodeType_": "allocate",
            "name_": "allocate-Tensor0_hbm",
            "ldsIdx_": 0,
            "component_": "hbm",
            "layoutDimOrder_": ["c0", "c1"],
            "startAddressCoreCorelet_": {
              "dim_prop_func": [{"Map": {}}, {"Const": {}}, {"Const": {}}],
              "dim_prop_attr": [
                {"factor_": 2, "label_": "core"},
                {"factor_": 1, "label_": "corelet"},
                {"factor_": 1, "label_": "time"}
              ],
              "data_": {"[0, 0, 0]": "0", "[1, 0, 0]": "256"}
            },
            "coordinates_": {
              "coordInfo": {
                "c0": {
                  "spatial": 3, "temporal": 0, "elemArr": 1,
                  "padding": "nopad",
                  "folds": {
                    "dim_prop_func": [
                      {"Affine": {"alpha_": 2, "beta_": 0}},
                      {"Affine": {"alpha_": 0, "beta_": 0}},
                      {"Affine": {"alpha_": 0, "beta_": 0}},
                      {"Affine": {"alpha_": 1, "beta_": 0}}
                    ],
                    "dim_prop_attr": [
                      {"factor_": 2, "label_": "core_fold"},
                      {"factor_": 1, "label_": "corelet_fold"},
                      {"factor_": 1, "label_": "row_fold"},
                      {"factor_": 2, "label_": "elem_arr_0"}
                    ]
                  }
                },
                "c1": {
                  "spatial": 3, "temporal": 0, "elemArr": 2,
                  "padding": "nopad",
                  "folds": {
                    "dim_prop_func": [
                      {"Affine": {"alpha_": 64, "beta_": 0}},
                      {"Affine": {"alpha_": 0, "beta_": 0}},
                      {"Affine": {"alpha_": 0, "beta_": 0}},
                      {"Affine": {"alpha_": 64, "beta_": 0}},
                      {"Affine": {"alpha_": 1, "beta_": 0}}
                    ],
                    "dim_prop_attr": [
                      {"factor_": 1, "label_": "core_fold"},
                      {"factor_": 1, "label_": "corelet_fold"},
                      {"factor_": 1, "label_": "row_fold"},
                      {"factor_": 1, "label_": "elem_arr_1"},
                      {"factor_": 64, "label_": "elem_arr_0"}
                    ]
                  }
                }
              }
            }
          },
          {
            "nodeType_": "allocate",
            "name_": "allocate-Tensor1_hbm",
            "ldsIdx_": 1,
            "component_": "hbm",
            "layoutDimOrder_": ["c0", "c1"],
            "startAddressCoreCorelet_": {
              "dim_prop_func": [{"Map": {}}, {"Const": {}}, {"Const": {}}],
              "dim_prop_attr": [
                {"factor_": 2, "label_": "core"},
                {"factor_": 1, "label_": "corelet"},
                {"factor_": 1, "label_": "time"}
              ],
              "data_": {"[0, 0, 0]": "512", "[1, 0, 0]": "768"}
            },
            "coordinates_": {
              "coordInfo": {
                "c0": {
                  "spatial": 3, "temporal": 0, "elemArr": 1,
                  "padding": "nopad",
                  "folds": {
                    "dim_prop_func": [
                      {"Affine": {"alpha_": 2, "beta_": 0}},
                      {"Affine": {"alpha_": 0, "beta_": 0}},
                      {"Affine": {"alpha_": 0, "beta_": 0}},
                      {"Affine": {"alpha_": 1, "beta_": 0}}
                    ],
                    "dim_prop_attr": [
                      {"factor_": 2, "label_": "core_fold"},
                      {"factor_": 1, "label_": "corelet_fold"},
                      {"factor_": 1, "label_": "row_fold"},
                      {"factor_": 2, "label_": "elem_arr_0"}
                    ]
                  }
                },
                "c1": {
                  "spatial": 3, "temporal": 0, "elemArr": 2,
                  "padding": "nopad",
                  "folds": {
                    "dim_prop_func": [
                      {"Affine": {"alpha_": 64, "beta_": 0}},
                      {"Affine": {"alpha_": 0, "beta_": 0}},
                      {"Affine": {"alpha_": 0, "beta_": 0}},
                      {"Affine": {"alpha_": 64, "beta_": 0}},
                      {"Affine": {"alpha_": 1, "beta_": 0}}
                    ],
                    "dim_prop_attr": [
                      {"factor_": 1, "label_": "core_fold"},
                      {"factor_": 1, "label_": "corelet_fold"},
                      {"factor_": 1, "label_": "row_fold"},
                      {"factor_": 1, "label_": "elem_arr_1"},
                      {"factor_": 64, "label_": "elem_arr_0"}
                    ]
                  }
                }
              }
            }
          }
        ],
        "computeOp_": [{
          "exUnit": "pt",
          "opFuncName": "abs",
          "attributes_": {"dataFormat_": "SEN169_FP16", "fidelity_": "regular"},
          "location": "Inner",
          "inputLabeledDs": ["Tensor0-idx0"],
          "outputLabeledDs": ["Tensor1-idx1"]
        }]
      }
    }]
  }
}
```

</details>

![SuperDSC example: abs on a (4, 64) tensor split across 2 cores](../_static/images/how-torch-spyre-works/fig4b-sdsc-example.svg)

*Figure 4b: A (4, 64) fp16 tensor split across 2 cores. There is one allocate node per tensor
and one JSON artifact for both cores.*

The table below shows what Torch-Spyre decides and what it encodes into SuperDSC for this
example.

| Decision | SuperDSC Field | Value |
|---|---|---|
| Core count | `coreFoldProp_` | 2 cores |
| Row assignment | `coreIdToWkSlice_` | Core 0 → rows 0–1, Core 1 → rows 2–3 |
| Per-core slice | `dataStageParam_.ss_` | (2, 64). Each core receives 2 rows with 1 stick each. |
| Memory addresses | `startAddressCoreCorelet_` | Core 0 → byte 0, Core 1 → byte 256 (per tensor) |
| Tensors | `labeledDs_` | Tensor0 (input) and Tensor1 (output), both fp16, both `dsType_: "INPUT"`. This label identifies the tiling layout rather than the role of the tensor. Both tensors share the label here because `abs` is pointwise and both tensors have identical tiling. |
| Operation | `computeOp_` | `abs` on PT unit, Tensor0 → Tensor1 |

One JSON file describes execution for both cores. The fold algebra, which includes the affine
transforms in `coordinates_` and the `Map` function in `startAddressCoreCorelet_`, parameterizes
the artifact so that each core derives its own slice without a separate description. At 32
cores, this compactness matters.

#### Why JSON

We chose JSON as the wire format for SuperDSC because we needed to read and diff these
artifacts constantly during development. JSON is also straightforward to cache through the
`torch.compile` artifact system. When an operation produced wrong results on a specific core
layout, opening the artifact in a text editor and inspecting the address mapping for that core
was often the fastest way to diagnose the problem.

#### From SuperDSC to KTIR

SuperDSC was designed to get Torch-Spyre working quickly while emitting a pragmatic IR that
matches the Spyre hardware model closely. We are transitioning to KernelTile IR (KTIR), an
MLIR-based representation that generalizes the SuperDSC concepts (compute tiles, scratchpad
staging, and compile-time core partitioning) into a community specification for any dataflow
accelerator. The structure of SuperDSC informed the design of KTIR. The
[KTIR RFC](https://github.com/torch-spyre/rfcs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md) is
available in our public repository.

---

## Challenge 4: covering ops in a model forward pass

Once the compiler infrastructure was in place, we ran into a different problem. Any model a user
targets will use hundreds of distinct operations in its forward pass, and every one of those
operations needs a path to Spyre hardware. A missing op causes a graph break. The compiled graph
stops, data round-trips to the CPU, the op executes there, and the data comes back. A single
graph break in the hot path can remove the performance gains from everything above.

We ran into this with Granite 3.3 8B.

We approached op coverage in layers. Each layer handles a different kind of gap between what
PyTorch produces and what Spyre can execute.

**Native ops** are ATen ops that Deeptools supports directly. These include pointwise ops such
as `add`, `relu`, and `sigmoid`, and matrix ops such as `mm` and `bmm`. Each native op maps to a
single SuperDSC that references an existing Deeptools OpFunc.

**Custom ops** are Spyre-specific ops that we register through `torch.library.custom_op`. The
user-facing ops are `spyre::rms_norm`, `spyre::layer_norm`, `spyre::gelu`, `spyre::softplus`,
`spyre::clamp`, `spyre::topkvalue`, `spyre::topkindex`, `spyre::full`, `spyre::ones_scalar`,
`spyre::logical_not`, and `spyre::constant`. There are also a few infrastructure ops
(`restickify`, `overwrite`, and `copy_from_d2d`). Each custom op has a `@register_fake` for
shape and dtype inference during tracing, along with a lowering rule that emits SuperDSCs. We
need custom ops because the default PyTorch behavior is to decompose ops such as `rms_norm` into
a sequence of multiplies, means, and reciprocals. Lowering that sequence op by op produces a
SuperDSC that does not match what the hardware actually runs. Registering `spyre::rms_norm` as a
single named op produces a clean lowering target, so that the SuperDSC reflects the real
computation that happens in the hardware.

**Decompositions** are FX graph rewrites that connect ATen ops to the first two layers.
`aten.rms_norm` decomposes into `spyre::rms_norm`, which is a custom op. `aten.addmm` decomposes
into `matmul + scale + add`, which are all native ops. A more subtle case is scalar constants.
The Spyre hardware does not support immediate scalar constants, so
`convert_constant_with_graph_node` rewrites the FX graph to replace every scalar with a size-1
tensor (using `spyre::constant`) before it reaches the compiler.

**CPU fallbacks** cover the long tail. The current set is `embedding`, `arange`, `sin`, `cos`,
`tril`, `triu`, `isin`, `normal_`, `argmax`, `bitwise_or`, `bitwise_xor`, and the int64 variants
of `max`. The runtime automatically transfers data to the CPU, executes the op, and returns the
result to Spyre. This is transparent to the model, but these ops are off the hot path.

An ATen op flows through this pipeline in the following way. Decomposition rewrites it into
either a native op or a custom op. Custom ops then lower to SuperDSCs built from native OpFuncs.
Anything left over falls back to the CPU. The goal is single-graph compilation, with no graph
breaks that force CPU round-trips in the forward pass.

![Operation enablement layers](../_static/images/how-torch-spyre-works/fig5-op-layers.svg)

*Figure 5: The three layers of op handling on Spyre, along with a CPU fallback path for the
long tail. Decompositions rewrite ATen ops into native ops or custom ops. Custom ops lower to
SuperDSCs built from native OpFuncs.*

We validate at four levels. The first level runs the PyTorch upstream test suite, which includes
`test_view_ops` and the OpInfo-based tests through `instantiate_device_type_tests`, directly
against the `spyre` device. This is the most important compatibility signal. If the PyTorch
framework agrees that Spyre tensors behave correctly, the integration is sound at the API level.
We maintain an allowlist of which test variants pass and we update it with each PR. The second
level tests individual ops against CPU reference outputs in `test_inductor_ops.py`. The third
level tests transformer building blocks, such as attention heads, FFN, and normalization, as
composed subgraphs in `test_building_blocks.py`. The fourth level tests full model forward
passes with real Granite weights against YAML-specified tolerance profiles in
`test_model_ops.py`. All four levels run in CI on every pull request.

![Testing pyramid](../_static/images/how-torch-spyre-works/fig6-testing-pyramid.svg)

*Figure 6: The four validation levels run in CI on every PR. All four tiers currently require
Spyre hardware. There is no emulated mode yet.*

---

## What we learned

The extension points exist. These are PrivateUse1, the scheduler pass hooks, the TorchInductor
backend registration, and a custom `TensorImpl`. We built everything out-of-tree, but we still
monkey-patch a few internal APIs where the extension points are incomplete. The three patches
cover dtype computation maps, fusion patterns, and compile_fx wrapping. The device registration
layer is well documented. `torch.accelerator` and OpenReg give a clear starting point. The
deeper hooks, which are the scheduler passes and codegen, are less well documented. For those,
we spent a lot of time reading the source code of other backends to figure out the patterns.

Staying out-of-tree was worth the discipline it imposed. Every time we were tempted to edit a
core file directly, the constraint pushed us toward finding the proper hook instead. That is not
a comfortable way to develop. It did give us independent release cadence, clean CI, and no risk
of breaking PyTorch for anyone else.

The stride model gap is real and unsolved at the framework level. `FixedTiledLayout` is a local
fix. Any scratchpad-based accelerator will run into the same mismatch, and we think a
generalized version should live in Inductor.

Covering operations was also a long task. We had to handle edge cases in dtype promotion,
layout propagation, view semantics, and error messaging across dozens of operations.

---

## What is next

**KTIR.** We are transitioning from SuperDSC to an MLIR-based IR that is designed as a
community specification rather than a Spyre-specific format. The
[KTIR RFC](https://github.com/torch-spyre/rfcs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md) is
available in our public repository.

**Scratchpad optimization.** We are designing and building a `ScratchPadAllocator` that plans
what data lives in the 2 MB LX scratchpad and what stays in DDR. This is the Spyre equivalent of
what a GPU cache does implicitly. Without it, compute units stall while they wait for data.

**Broader model and serving coverage.** Llama 3.1 8B, vision models, and speech models are
next. We are also integrating with vLLM as a platform plugin alongside CUDA and ROCm.

**Distributed inference.** We are extending `torch.compile` to generate collective operations
for multi-card Spyre configurations, using the standard PyTorch distributed communication APIs.

**Profiling.** We are building `torch.profiler` integration through `ProfilerActivity.PrivateUse1`
so that transfer latency, scratchpad pressure, and pipeline utilization are visible to users.
Early scaffolding is in place. See our
[profiling RFC](https://github.com/torch-spyre/rfcs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md).

**Upstream contributions.** We plan to contribute a generalized `FixedTiledLayout` for Inductor,
OpenReg primitives for standardized out-of-tree backend testing, and documented CI patterns for
hardware teams that follow us.

---

## Getting started

Torch-Spyre is available for users who have access to IBM Spyre hardware. Install from source.

```bash
git clone https://github.com/torch-spyre/torch-spyre
cd torch-spyre && pip install .
```

PyPI distribution (`pip install torch-spyre`) is coming soon.

- GitHub: [github.com/torch-spyre/torch-spyre](https://github.com/torch-spyre/torch-spyre)
- RFCs: [github.com/torch-spyre/rfcs](https://github.com/torch-spyre/rfcs)
- Documentation: [torch-spyre.readthedocs.io](https://torch-spyre.readthedocs.io)

---

## Appendix: extension point reference for out-of-tree PyTorch backends

For teams that are building out-of-tree PyTorch backends, the tables below list every hook we
used. The tables are organized by the challenge that each hook addresses.

### Challenge 1: device registration (PrivateUse1)

| Component | PyTorch Hook | Purpose | Key Detail |
|---|---|---|---|
| `torch.utils.rename_privateuse1_backend("spyre")` | PrivateUse1 | Makes `"spyre"` a valid device name in PyTorch | Must be called before any device use; rewires dispatch tables and `.to()` routing |
| `torch._register_device_module("spyre", ...)` | PrivateUse1 | Attaches the Python device module | Provides `torch.spyre.*` namespace |
| `SpyreGuardImpl` | `DeviceGuardImplInterface` | Device context management | Registered via `C10_REGISTER_GUARD_IMPL(PrivateUse1, SpyreGuardImpl)`; uses thread-local storage; supports 10+ dtypes including fp8 variants |
| `SpyreAllocator` | `REGISTER_ALLOCATOR` | Custom device memory allocator | Registered for `c10::DeviceType::PrivateUse1`; wraps flex runtime memory in `SharedOwnerCtx` without exposing raw pointers (IBM Z security requirement) |
| `SpyreTensorImpl` | `at::TensorImpl` | Carries tiled layout metadata alongside the PyTorch tensor | Adds `SpyreTensorLayout` with device-side tiled size, a stride map from host dims to device dims, and device dtype. `BYTES_IN_STICK=128` (64 fp16 elements) |
| `SpyreHooksInterface` | `PrivateUse1HooksInterface` | Reports device availability and primary context status | Queried by PyTorch runtime during device enumeration |
| `SpyreStream` | `torch.cuda.Stream` semantics | Asynchronous execution and CPU↔Spyre overlap | Implements `torch.spyre.Stream`, `current_stream()`, `synchronize()`; enables pipelining batch N+1's input transfer behind batch N's compute |

Initialization is lazy and thread-safe. Importing `torch_spyre` registers the device name and
module. `flex::initializeRuntime()` starts only on the first device use.

### Challenge 2: tiled tensor layout (FixedTiledLayout)

| Component | PyTorch Hook | Purpose | Key Detail |
|---|---|---|---|
| `FixedTiledLayout` | `inductor.ir.FixedLayout` (subclass) | Bridges the PyTorch stride model and Spyre tiled memory | Carries a `SpyreTensorLayout` descriptor with tiled device shape, dimension mapping (stick dims appear twice, once as tile index and once as intra-stick offset), stride mapping, and 128-byte padding |
| `propagate_spyre_tensor_layouts()` | `CustomPreSchedulingPasses` (runs before `Scheduler` construction through a `GraphLowering._update_scheduler` monkey-patch) | Converts `FixedLayout` to `FixedTiledLayout` across the operation graph | Topological traversal of IR operations. Pointwise ops inherit the input layout. Matmul and reduction require special handling for the contracted output dimension. External kernels use a generic stick format. |
| `span_reduction()` and `work_distribution()` | `CustomPreSchedulingPasses` (run after `propagate_spyre_tensor_layouts` in the same pre-scheduling pass) | Partitions iteration dimensions across 32 cores at compile time | Constrained by `SENCORES=32`. `span_reduction` determines how the span of each reduction can be split. `work_distribution` assigns spans to cores. Each core receives an equal number of sticks. Enforces the per-core addressable device memory limit (a hardware constraint that is distinct from the 2 MB LX SRAM scratchpad). Two-pass algorithm: minimum splits first, then remaining cores by priority. |

*Why the scheduler phase.* Tiled layouts must be resolved before codegen because they are
needed for memory planning and core work division. Propagating through Dynamo and FakeTensor
would require invasive core changes. Deferring to codegen is too late.

### Challenge 3: Inductor backend for dataflow compilation

| Component | PyTorch Hook | Purpose | Key Detail |
|---|---|---|---|
| `enable_spyre_context()` | Context manager | Activates all Spyre-specific Inductor configuration | Registers decompositions, lowerings, `mm_to_bmm_pass` (2D matmul to 3D bmm for better core utilization), Inductor config overrides for the dataflow model, and fusion heuristics |
| `SuperDSCScheduling` | Inductor backend scheduling class | Decides how to group and order operations | Replaces Triton scheduling. Registered at import time. |
| `SpyrePythonWrapperCodegen` | Inductor backend codegen class | Generates the Python wrapper that allocates tiled buffers and launches kernels | Calls `spyre_empty_with_layout()` for buffer allocation. Wraps kernel dispatch through `async_compile.sdsc()`. |
| `SpyreDeviceOpOverrides` | Inductor device op overrides | Device-specific operation implementations | Registered at import time alongside the scheduling and codegen classes |
| `SpyreKernel`, `OpSpec`, `generate_sdsc()` | Custom codegen pipeline | Produces the SuperDSC JSON IR per scheduled node | `SpyreKernel` builds an RValue AST. `OpSpec` encodes the iteration space as sympy expressions with device coordinates. `generate_sdsc()` emits JSON with `dscs_` entries that contain `labeledDs_`, `primaryDsInfo_`, `scheduleTree_`, `dataStageParam_`, and `computeOp_`. Folding through affine transforms keeps the artifact compact across all 32 cores. |

### Challenge 4: op coverage

| Mechanism | PyTorch Hook | Purpose | Key Detail |
|---|---|---|---|
| Native ops (Deeptools-supported) | Direct mapping to OpFuncs | Pointwise and matrix ops supported natively by the hardware | Pointwise: `add`, `sub`, `mul`, `truediv`, `relu`, `sigmoid`, `abs`, `neg`, `exp`, `log`, `sqrt`, `rsqrt`, `reciprocal`, `tanh`, `floor`, `eq`, `ne`, `ge`, `le`, `lt`, `gt`, `square`, `where`, `logical_and`, `to_dtype`, plus lowered forms of `clamp`/`gelu`/`softplus`/`layernormscale`/`layernormnorm`. Matrix: `mm`, `bmm` (`matmul` is decomposed to these by Inductor upstream). Matrix ops need custom layout propagation for the contracted dimension, but both kinds map to single SDSCs. |
| `torch.library.custom_op` | Custom op registration | Layer 3 normalization, activation, and ops that need a single named lowering target | Each op requires `@register_fake` for shape/dtype inference during tracing, plus a lowering rule mapping to Spyre primitives. User-facing ops: `spyre::rms_norm`, `spyre::layer_norm`, `spyre::gelu`, `spyre::softplus`, `spyre::clamp`, `spyre::topkvalue`, `spyre::topkindex`, `spyre::full`, `spyre::ones_scalar`, `spyre::logical_not`, `spyre::constant`. Infrastructure ops: `spyre::restickify`, `spyre::overwrite` / `spyre::overwrite_f`, `spyre::copy_from_d2d`. |
| Decompositions | Registered in `enable_spyre_context()` | Layer 4 ops not natively available | `logical_not` → `eq(input, ne(input, input))` (bool), `addmm` → `matmul+scale+add`, `linear` → `matmul+add` (with weight transpose), `rms_norm` → `spyre::rms_norm`, `layer_norm` → `spyre::layer_norm`, `gelu` → `spyre::gelu`, `softplus` → `spyre::softplus`, `topk` → `spyre::topkvalue`/`spyre::topkindex`, `max.dim` → split value/index decomp, `scaled_dot_product_attention` → explicit Q·K^T·V, `cat`, `constant_pad_nd`, `ones` → `spyre::ones_scalar+expand+clone`, `new_ones`, `full` → `spyre::full`, `bitwise_not`, `bitwise_and` |
| CPU fallback (auto-transfer) | PyTorch fallback dispatch | Infrequent ops that are off the critical path | `embedding`, `arange`, `sin`, `cos`, `tril`, `triu`, `isin`, `normal_`, `argmax`, `bitwise_or`, `bitwise_xor`, int64 variants of `max`. Data auto-transfers to the CPU, executes, and returns to Spyre. Transparent to the model. |

### Profiling (in progress)

| Component | PyTorch Hook | Purpose | Key Detail |
|---|---|---|---|
| kineto-spyre extension | `ProfilerActivity.PrivateUse1` | Spyre-specific metrics in `torch.profiler` | Target metrics are transfer latency, scratchpad pressure, pipeline utilization, and inter-core communication overhead. Early scaffolding is in place. See the [profiling RFC](https://github.com/torch-spyre/rfcs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md). |

---

## Acknowledgments

This work is a collaboration across multiple teams at IBM Research. We thank the PyTorch team
for building the extensibility mechanisms that made this possible without forking the core. We
also thank the broader community for the reference implementations (Ascend NPU, Intel GPU, and
OpenReg) that helped us understand what an out-of-tree backend could look like in practice.
