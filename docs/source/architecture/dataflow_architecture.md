# Dataflow Accelerator Architecture

This document provides a reference overview of the dataflow accelerator
model as implemented in the IBM Spyre AI Card. It is intended both as
context for Torch-Spyre developers and as a general reference for the
dataflow accelerator design pattern.

:::{note}
**Key Concepts** — terms used throughout this page:

- **Tile** — a contiguous sub-tensor assigned to a single core.
- **Scratchpad** — fast on-core SRAM. The compiler manages it directly; there is no hardware cache.
- **DMA** — Direct Memory Access. On Spyre this is the PCIe path that carries tensor data between host memory and the device's LPDDR5.
- **Load/store** — the compiler-emitted instructions used by the on-core load/store units to move tiles between LPDDR5 and the LX scratchpad.
- **SPMD** — Single Program, Multiple Data. Every core runs the same program on its own slice of the data, picked by core ID.
- **SuperDSC** — the Spyre-specific kernel descriptor format. A single JSON describes one scheduled kernel operation across all cores.
- **DCI** — Data Conversion Information. The `DataConversionInfo` struct (built by `generate_dci()` in `spyre_mem.cpp`) that bundles loop ranges, host and device strides, and dtype info; the runtime feeds it to `copyAsync` to drive a host ↔ LPDDR5 DMA transfer.
:::

## What is a Dataflow Accelerator?

Traditional von Neumann processors execute instructions sequentially,
fetching data from memory on demand. **Dataflow accelerators** invert
this model: computation is expressed as a directed acyclic graph (DAG)
of operations, and each node fires as soon as all its input operands
are ready. There is no program counter — the data itself drives
execution order. This eliminates most control-flow overhead and enables
deeply pipelined, high-throughput execution
\[[Dennis 1974](#ref-dennis1974), [Veen 1986](#ref-veen1986)\].

Key characteristics:

- Operations are scheduled by data availability, not program counters.
- Tensors are staged in local scratchpad memories close to compute units.
- The compiler is responsible for all data movement and scheduling.

:::{figure} ../_static/images/dataflow-dag.svg
:alt: Dataflow execution graph showing operations firing when their input tensors are ready
:width: 680px
:align: center

Dataflow execution of two independent branches (Linear + LayerNorm) that merge at a Concat node. **Green (solid border) = ready** — all inputs available, fires immediately. **Yellow (dashed border) = waiting** — blocked on upstream output. Both MatMuls fire in parallel because their inputs are independent. No program counter controls execution order — data availability does.
:::

### Dataflow Firing Rule

Execution follows the **dataflow firing rule**: an operation is eligible
to execute as soon as all of its input operands are available
\[[Dennis 1974](#ref-dennis1974)\]. Operands are modeled as **tokens**
that propagate through the graph, activating downstream operations. This
is the fundamental principle that distinguishes dataflow from
control-flow execution. In AI accelerators, dataflow execution operates
at the granularity of operators (or tiles), unlike instruction-level
dataflow in CPUs. Nodes with multiple inputs act as synchronization
points, potentially introducing backpressure that limits throughput.

In the Spyre implementation, a **SuperDSC** (kernel descriptor) is what
"fires" once the compiler-emitted load/store instructions have staged
the necessary **tiles** into the scratchpad. The compiler schedules
this statically: load/store instructions move tiles from LPDDR5 into
the scratchpad, and the SuperDSC kernel runs once all inputs are
resident there.

### Static vs Dynamic Dataflow

There are two major variants of dataflow execution
\[[Veen 1986](#ref-veen1986)\]:

- **Static dataflow** — dependencies are fixed at compile time. The
  hardware executes a pre-determined graph with known shapes and
  scheduling. This is simpler and more efficient for regular workloads.

- **Dynamic dataflow** — uses tagged tokens to track dependencies at
  runtime, supporting multiple concurrent instances of the same
  operation. This is more flexible but significantly more complex in
  hardware.

Modern AI accelerators (including Spyre) typically implement a **static
dataflow** model optimized for workloads where operator graphs and
tensor shapes are known ahead of time.

### Relationship to Out-of-Order Execution

Modern CPUs implement a limited form of dataflow through **out-of-order
(OOO) execution**: instructions execute when their operands are ready,
within a bounded hardware window. Dependencies are tracked dynamically
by the reorder buffer \[[Tomasulo 1967](#ref-tomasulo1967)\].

Dataflow accelerators take this principle much further:

| Aspect | OOO CPU | Dataflow Accelerator |
|--------|---------|---------------------|
| Scope | Small instruction window (hundreds) | Entire computation graph |
| Dependency tracking | Dynamic, in hardware | Static, at compile time |
| Parallelism | Limited by window size | Limited by graph structure |
| Overhead | Significant (register renaming, speculation) | Minimal (pre-scheduled) |

This connection helps explain why dataflow architectures can be so
efficient for regular workloads: they trade the generality of dynamic
OOO scheduling for the performance of static, whole-graph optimization.

### Why Dataflow is Effective for Deep Learning

Deep neural networks exhibit properties that are particularly
well-suited to dataflow execution
\[[Chen 2017](#ref-chen2017), [Sze 2017](#ref-sze2017)\]:

- **Regular computation patterns** — operations like GEMM, convolution,
  and element-wise activations are highly predictable
- **High data reuse opportunities** — weights and activations are
  accessed repeatedly across layers and batches
- **Static execution graphs** — during inference (and often training),
  the operator graph is fixed and shapes are known

Dataflow architectures exploit these properties by:

- Maximizing data reuse in local scratchpad, avoiding redundant
  off-chip memory accesses
- Enabling pipeline parallelism across layers and operations
- Pre-planning most data movement at compile time, which keeps
  runtime allocation overhead low (though execution timing of the
  staged transfers still depends on runtime conditions)

**Data movement — not compute — is often the dominant cost** in DNN
execution \[[Sze 2017](#ref-sze2017)\]. This has been dramatically
demonstrated in the transformer era by FlashAttention
\[[Dao 2022](#ref-dao2022)\], which achieved large speedups purely by
restructuring attention computation to minimize memory reads/writes.
Dataflow optimization targets this bottleneck directly by keeping active
data close to compute units and minimizing DDR round-trips.

:::{note}
**For Torch-Spyre developers:** The dataflow model has direct
implications for how PyTorch operations are lowered. Because all data
movement is explicit and compiler-managed, the backend has to fix the
tiling strategy, the load/store schedule that stages tiles into
scratchpad, and the kernel descriptors (SuperDSC) at compile time —
there is no runtime fallback to a hardware cache. See
[Compiler Architecture](../compiler/architecture.md) and
[Work Division Planning](../compiler/work_division_planning.md) for
details.
:::

## Spyre Architecture Highlights

:::{figure} ../_static/images/telum2-spyre-chip.jpg
:alt: IBM Spyre Accelerator chip (left) and IBM Telum II processor (right)
:width: 680px
:align: center

The IBM Spyre Accelerator chip (left) and IBM Telum II processor (right). *Image credit: [IBM Newsroom](https://newsroom.ibm.com/ai-on-z).*
:::

| Feature | Detail |
|---------|--------|
| Cores | 32 AI accelerator cores |
| Technology | 5 nm |
| Memory per card | Up to 128 GB LPDDR5 |
| Peak performance | >300 TOPS per card |
| Supported data types | int4, int8, fp8, fp16 |
| Power envelope | 75 W per card |
| Host interface | PCIe |
| Max card cluster | 8 cards / 1 TB memory [^cluster] |

[^cluster]: Multiple Spyre cards can be clustered in a single IBM Z I/O drawer, sharing memory across cards for larger model capacity.

:::{figure} https://research-website-prod-cms-uploads.s3.us.cloud-object-storage.appdomain.cloud/IBM_AIU_PCIE_05_d6a1bd0d18.jpg
:alt: IBM Spyre Accelerator PCIe card
:width: 560px
:align: center

The IBM Spyre Accelerator PCIe card (reverse side), showing the physical form factor for IBM Z and Power systems. *Image credit: [IBM Newsroom](https://newsroom.ibm.com/ai-on-z).*
:::

Spyre implements a **hybrid dataflow** architecture: dataflow execution
drives the compute kernels, while control-flow mechanisms handle host
interaction, kernel sequencing, and device orchestration. This is
consistent with the design of most modern accelerators — pure dataflow
machines were largely unsuccessful historically
\[[Veen 1986](#ref-veen1986)\], and practical systems combine the
efficiency of dataflow execution with the flexibility of control flow
for coordination.

## Memory Hierarchy

Spyre exposes two levels of memory visible to the compiler.
Understanding this hierarchy is critical for performance: LX Scratchpad
is significantly lower-latency than DDR, so minimizing DDR round-trips
through effective tiling is the primary lever for optimizing kernel
throughput.

1. **DDR (device DRAM)** — large, off-core storage for full tensors.
2. **LX Scratchpad** — fast, on-core storage for tiles actively being
   processed. The compiler emits load/store instructions to stage
   tiles between DDR and the scratchpad; there is no hardware cache
   to fall back on.

:::{figure} ../_static/images/spyre-memory-hierarchy.svg
:alt: Spyre two-level memory hierarchy showing DDR and LX Scratchpad
:width: 600px
:align: center

Spyre memory hierarchy. Full tensors live in off-chip LPDDR5 — 128 GB physical, of which 16 GB is reserved for ECC, leaving roughly 112 GB usable. Before a kernel runs, the compiler issues load/store instructions to stage the tiles it needs into the on-core LX Scratchpad. The PCIe link between the host and LPDDR5 is the DMA path. Tiling that keeps traffic on-chip is the main lever for performance.
:::

The end-to-end data path is:

```
Host → DDR → LX Scratchpad → Compute Units → LX Scratchpad → DDR → Host
```

The compiler generates two kinds of artifacts to drive this pipeline:

- **Load/store instructions** that move tiles between DDR and the LX
  scratchpad at the right time.
- **SuperDSC** — JSON kernel descriptors that specify the computation
  performed on each tile once it arrives in scratchpad. SuperDSC is
  being superseded by KTIR, a tile-based MLIR intermediate
  representation — see [RFC 0682](../rfcs/index.md).

## Execution Model

Each Spyre core executes a **kernel** — a self-contained computation
on a **tile** (a contiguous sub-tensor region) of data. All three of
the following are determined **at compile time** — there is no global
instruction scheduler. Execution is driven by compile-time planning
and data readiness, with only minimal runtime control for transfer
sequencing and resource arbitration:

1. **Work division** — how to split a tensor operation across cores
   (see [Work Division Planning](../compiler/work_division_planning.md))
2. **Data staging** — when, and with what load/store instructions, to
   move tiles between LPDDR5 and the LX scratchpad
3. **Kernel specification** — the SuperDSC (Spyre kernel descriptor)
   JSON that fully describes the operation: operand shapes, data
   types, tiling parameters, and the sequence of compute instructions
   for the PT array.

Cores execute in SPMD (Single Program, Multiple Data) fashion: cores
follow a common program structure but operate on different tiles and
may execute different kernel phases over time, identified by their
core ID.

:::{figure} ../_static/images/spyre-core-microarchitecture.png
:alt: Spyre core microarchitecture showing PT units, PE, SFP, LX Scratchpad, and device memory
:width: 50%
:align: center

Spyre core microarchitecture. Each Spyre core contains two corelets (Corelet 0 and Corelet 1) that share a single 2 MB LX scratchpad (SRAM). A corelet is built from an 8 × 8 systolic Processing Element (PE) array, used for matrix-style compute on the PT execution unit, plus a 1D Special Function Unit (SFU) vector unit that handles non-linear activations such as GELU and softmax. Compiler-emitted load/store instructions move tiles between the LX scratchpad and off-chip LPDDR5; there is no hardware cache. Cores talk to each other over a bi-directional ring interconnect at 128 B per cycle per direction. The architecture descends from IBM's RaPiD AI accelerator (Venkataramani et al., ISCA 2021, [DOI:10.1109/ISCA52012.2021.00021](https://doi.org/10.1109/ISCA52012.2021.00021)).
:::

## Comparison with GPU and Other Accelerators

Spyre's dataflow model differs from GPU execution in several
fundamental ways that directly affect how Torch-Spyre lowers PyTorch
operations:

| Aspect | GPU (CUDA) | Spyre Dataflow |
|--------|-----------|----------------|
| Scheduling | Warp-level SIMT | Data-driven, core SPMD |
| Memory model | Shared memory + global | Scratchpad + DDR |
| Data movement | Implicit caching | Explicit, compiler-scheduled load/store between DDR and scratchpad |
| Supported precision | fp32 / bf16 / fp16 / int8 | int4 / int8 / fp8 / fp16 |
| Compiler model | Hybrid (AOT kernels with optional JIT from PTX) | Primarily AOT (scheduling and data movement planned at compile time) |
| Parallelism granularity | Thread blocks [^gpu] | Core tiles [^gpu] |

[^gpu]: GPU thread blocks and Spyre core tiles serve analogous roles (distributing work across parallel units) but differ in how scheduling is performed: thread blocks are dispatched by a hardware scheduler at runtime, while core tile assignment is fixed at compile time in Spyre.

Dataflow accelerators also differ from **systolic arrays** (as used in
Google TPUs \[[Jouppi 2017](#ref-jouppi2017),
[Jouppi 2023](#ref-jouppi2023)\]): systolic arrays move data through a
fixed pipeline of processing elements in a regular, predetermined
pattern, while dataflow architectures schedule execution based on graph
dependencies, allowing more flexible communication patterns at the cost
of more complex compilation \[[Xu 2024](#ref-xu2024)\]. Formal
frameworks such as MAESTRO \[[Kwon 2020](#ref-kwon2020)\] provide
analytical tools for reasoning about the data reuse, bandwidth, and
energy trade-offs across these architectural styles.

### Modern Dataflow-Inspired Accelerators

Spyre is part of a broader trend of dataflow-inspired AI accelerators:

| Accelerator | Organization | Key characteristic |
|-------------|-------------|-------------------|
| IBM Spyre | IBM | Static dataflow with explicit scratchpad management |
| SambaNova RDU | SambaNova Systems | Reconfigurable dataflow architecture \[[Prabhakar 2017](#ref-prabhakar2017)\] |
| Cerebras WSE | Cerebras Systems | Spatial/dataflow execution across wafer-scale fabric |
| Graphcore IPU | Graphcore | Bulk Synchronous Parallel model with explicit data movement |

Each makes different trade-offs in programmability, scalability, and
hardware complexity, but all share the principle of optimizing data
movement as the primary lever for performance and energy efficiency.
For a comprehensive survey of hardware accelerators in the LLM era, see
\[[Kachris 2025](#ref-kachris2025)\].

## Limitations and Challenges

While dataflow architectures offer strong parallelism and efficiency
for regular workloads, they face several challenges:

- **Compiler complexity** — extracting, scheduling, and mapping
  dataflow graphs to hardware requires sophisticated compilation
  passes. The compiler must handle tiling, data layout, memory
  allocation, and multi-core scheduling — all at compile time.

- **Irregular workloads** — dynamic control flow (e.g., variable-length
  sequences, conditional branching) and variable tensor shapes reduce
  the effectiveness of static dataflow scheduling. Operations that
  cannot be pre-planned may require fallback mechanisms (e.g., host
  execution or alternative kernels).

- **Placement and scheduling** — efficiently mapping large operator
  graphs onto a fixed number of cores with limited scratchpad memory
  remains a hard optimization problem, especially as model sizes grow.

- **Synchronization and backpressure** — operations that require
  multiple inputs (e.g., concatenation, residual additions) act as
  synchronization points where all upstream paths must complete before
  execution can proceed. The Torch-Spyre compiler mitigates this
  through tiling strategies that balance work across cores and minimize
  idle time at synchronization boundaries (see
  [Work Division Planning](../compiler/work_division_planning.md)).

- **Data movement overhead** — dataflow cuts down on redundant
  movement, but every tile that has to come from DDR still costs
  latency on the way into and out of the scratchpad. At scale the
  communication network is the bottleneck, especially for ops with
  poor data locality.

- **Limited general-purpose adoption** — dataflow architectures have
  been most successful in domain-specific applications (AI inference,
  signal processing) where computation patterns are predictable. They
  are not well-suited for general-purpose workloads with irregular
  memory access patterns \[[Veen 1986](#ref-veen1986)\].

These challenges are active areas of research in the Torch-Spyre
project. The [Compiler Architecture](../compiler/architecture.md)
documentation describes the specific strategies used to address tiling,
scheduling, and fallback handling.

## References

(ref-dennis1974)=
- **\[Dennis 1974\]** J. B. Dennis and D. P. Misunas, "A Preliminary Architecture for a Basic Data-Flow Processor," *Proc. 2nd Annual Symposium on Computer Architecture (ISCA)*, 1974. [DOI:10.1145/641675.642111](https://doi.org/10.1145/641675.642111)

(ref-veen1986)=
- **\[Veen 1986\]** A. H. Veen, "Dataflow Machine Architecture," *ACM Computing Surveys*, vol. 18, no. 4, pp. 365–396, 1986. [DOI:10.1145/27633.28055](https://doi.org/10.1145/27633.28055)

(ref-tomasulo1967)=
- **\[Tomasulo 1967\]** R. M. Tomasulo, "An Efficient Algorithm for Exploiting Multiple Arithmetic Units," *IBM Journal of Research and Development*, vol. 11, no. 1, pp. 25–33, 1967. [DOI:10.1147/rd.111.0025](https://doi.org/10.1147/rd.111.0025)

(ref-chen2017)=
- **\[Chen 2017\]** Y.-H. Chen *et al.*, "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks," *IEEE Journal of Solid-State Circuits*, vol. 52, no. 1, pp. 127–138, 2017. [DOI:10.1109/JSSC.2016.2616357](https://doi.org/10.1109/JSSC.2016.2616357)

(ref-sze2017)=
- **\[Sze 2017\]** V. Sze *et al.*, "Efficient Processing of Deep Neural Networks: A Tutorial and Survey," *Proceedings of the IEEE*, vol. 105, no. 12, pp. 2295–2329, 2017. [DOI:10.1109/JPROC.2017.2761740](https://doi.org/10.1109/JPROC.2017.2761740)

(ref-jouppi2017)=
- **\[Jouppi 2017\]** N. P. Jouppi *et al.*, "In-Datacenter Performance Analysis of a Tensor Processing Unit," *Proc. 44th Annual International Symposium on Computer Architecture (ISCA)*, 2017. [DOI:10.1145/3079856.3080246](https://doi.org/10.1145/3079856.3080246)

(ref-prabhakar2017)=
- **\[Prabhakar 2017\]** R. Prabhakar *et al.*, "Plasticine: A Reconfigurable Architecture for Parallel Patterns," *Proc. 44th Annual International Symposium on Computer Architecture (ISCA)*, 2017. [DOI:10.1145/3079856.3080256](https://doi.org/10.1145/3079856.3080256)

(ref-kwon2020)=
- **\[Kwon 2020\]** H. Kwon *et al.*, "MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings," *IEEE Micro*, vol. 40, no. 3, pp. 20–29, 2020. [DOI:10.1109/MM.2020.2985963](https://doi.org/10.1109/MM.2020.2985963)

(ref-dao2022)=
- **\[Dao 2022\]** T. Dao *et al.*, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

(ref-jouppi2023)=
- **\[Jouppi 2023\]** N. P. Jouppi *et al.*, "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings," *Proc. 50th Annual International Symposium on Computer Architecture (ISCA)*, 2023. [DOI:10.1145/3579371.3589350](https://doi.org/10.1145/3579371.3589350)

(ref-xu2024)=
- **\[Xu 2024\]** R. Xu *et al.*, "A Survey of Design and Optimization for Systolic Array-based DNN Accelerators," *ACM Computing Surveys*, vol. 56, no. 1, 2024. [DOI:10.1145/3604802](https://doi.org/10.1145/3604802)

(ref-kachris2025)=
- **\[Kachris 2025\]** C. Kachris, "A Survey on Hardware Accelerators for Large Language Models," *Applied Sciences*, vol. 15, no. 2, art. 586, 2025. [DOI:10.3390/app15020586](https://doi.org/10.3390/app15020586)

## Further Reading

- [IBM Spyre Accelerator Overview](spyre_accelerator.md)
- [Compiler Architecture](../compiler/architecture.md)
- [Tensor Layouts](../user_guide/tensors_and_layouts.md)
