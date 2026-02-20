# Kernel Tile Data Parallel (`ktdp`) Intermediate Representation

**Authors:**

Prasanth Chatarasi, Bardia Mahjour, Alberto Mannari, Swagath Venkataramani, David Grove, Olivier Tardieu, Mori Ohara, Takuya Nakaike, Kazuaki Ishizaki, Eri Ogawa, Michihiro Horie, Mudhakar Srivatsa, Viji Srinivasan

Point of contact: Prasanth Chatarasi

## **Summary**
This RFC defines a new interface between compiler frontend (TorchInductor, custom kernels via. Triton, and others) and compiler backend (Deeptools)

## **Motivation**
Transtion from SuperDSC-bundle into tile-based interface built over open-source MLIR compiler infrastructure. It enables sufficient expressivity to capture data-parallel mappings from: 1) Torch inductor compiler frontend 2) Custom Spyre kernels written in Triton/Helion

## **Proposed Implementation**

## Introduction

`ktdp` is a tile-based, block-structured intermediate representation (IR) designed to express programs targeting multi-core accelerator architectures. It embodies a data-parallel abstraction of the accelerator shown in Figure 1. The accelerator contains multiple cores, with each core comprising a compute engine and an on-chip scratchpad memory associated with the compute unit. The cores are attached together through an on-chip interconnect fabric, which also interfaces with one or more off-chip memory banks. 

<p align="center">
  <img src="ktdp_hw_abstraction.png" alt="ktdp_hw_abstraction" width="450"/>
</p>
<p align="center">
  Figure 1. Hardware abstraction of multi-core accelerator embodied in `ktdp` IR 
</p>  

`ktdp` programs allow tensors to be placed in a distributed fashion across multiple memory elements (on-chip and/or off-chip memory). In the same vein, compute operations can be split into compute tiles and assigned to the compute engine within each core. `ktdp` allows each compute tile to have global view of all on-chip and off-chip memory elements via the on-chip interconnect fabric.

`ktdp` is built to represent complex computation kernels (e.g., attention in LLMs) mapped onto multi-core accelerators, going beyond a single operation (e.g., tensor Add). Kernels span multiple operations that are composed sequentially, interleaved with structured control-flow such as loops, conditionals, and multi-stage pipelines. `ktdp` includes construts to capture kernel execution semantics including dependencies, synchronization points, and tensor liveliness. Intermediate tensors within the kernel could be placed on on-chip memory elements and reused across producer-consumer operations. With each compute tile having global access to all memory elements, individual compute operations can be flexibly tiled along different dimensions.

#### Who Produces `ktdp` IR? 
`ktdp` is mid-level IR in the compilation pipeline. Starting from a graph-level description or expert-written versions of the kernel, higher-level compiler frameworks (e.g., TorchInductor, Triton) are expected to optimize data-parallel work partitioning and memory management and generate `ktdp` IR. `ktdp` can express the already established parallel decomposition from the front-end compiler and facilitates further optimizations within that decomposition.

#### Who Consumes `ktdp` IR? 
Kernel scheduler operates on `ktdp` programs and performs data-flow scheduling. The scheduler is responsible for efficient data-flow mapping of the kernel onto the hardware preserving correctness.

## Table of Contents

- [A) Operations in `ktdp` IR](#a-operations-in-ktdp-ir)
    - [1. Layouts, Access Tiles, Loads, and Stores](#1-layouts-access-tiles-loads-and-stores)
    - [2. Compute Operations](#2-compute-operations)
    - [3. Control-Flow Operations](#3-control-flow-operations)
    - [Overall Design Perspective](#overall-design-perspective)

- [B) Example: Tile-Parallel Matrix Add in `ktdp`](#b-example-tile-parallel-matrix-add-in-ktdp)

- [C) Operations within KTDP dialect](#c-operations-within-ktdp-dialect)
    - [1. `ktdp.get_compute_tile_id`](#1-ktdpget_compute_tile_id-ktdpgetcomputetileid)
    - [2. `ktdp.construct_memory_view`](#2-ktdpconstruct_memory_view-ktdpconstructmemoryviewop)
    - [3. `ktdp.construct_distributed_memory_view`](#3-ktdpconstruct_distributed_memory_view-ktdpconstructdistributedmemoryviewop)
    - [4. `ktdp.construct_access_tile`](#4-ktdpconstruct_access_tile-ktdpconstructaccesstilesop)
    - [5. `ktdp.construct_indirect_access_tile`](#5-ktdpconstruct_indirect_access_tile-ktdpconstructindirectaccesstilesop)
    - [6. `ktdp.load`](#6-ktdpload-ktdploadop)
    - [7. `ktdp.store`](#7-ktdpstore-ktdpstoreop)

- [D) Types within KTDP dialect](#d-types-within-ktdp-dialect)
    - [1. AccessTileType](#1-accesstiletype)

- [E) Attributes within KTDP dialect](#e-attributes-within-ktdp-dialect)
    - [1. SpyreMemorySpaceAttr](#1-spyrememoryspaceattr)
  
## A) Operations in `ktdp` IR
The operations within `ktdp` IR can be organized into several functional categories, reflecting the different roles they play in expressing kernel semantics and execution structure.

### 1. Layouts, Access Tiles, Loads, and Stores
`ktdp` IR introduces a dedicated dialect named KTDP to model all memory-view and access–related abstractions. This dialect provides operations for describing layouts that define logical views over memory regions allocated in specific memory spaces, constructing distributed memory or tensor views that span multiple physical memories, and representing access tiles that encode structured collections of logical tensor coordinates. It also includes the corresponding load and store primitives that consume these access tiles to perform explicit data movement. Together, these operations make memory, locality, and access patterns first-class entities in the IR.

### 2. Compute Operations
For arithmetic and numerical computation, `ktdp` reuses operations from existing MLIR dialects such as Arith, Math, and LinAlg. These dialects already provide a rich set of well-defined semantics for scalar, vector, and tensor computation. When kernel-specific functionality is required that is not directly expressible using the standard dialect operations, `ktdp` extends them in a disciplined manner through auxiliary variants (e.g., suffixed with _Ext).

Arith dialect - https://mlir.llvm.org/docs/Dialects/ArithOps/

Math dialect - https://mlir.llvm.org/docs/Dialects/MathOps/

LinAlg dialect - https://mlir.llvm.org/docs/Dialects/Linalg/ 


### 3. Control-Flow Operations
Structured control flow in `ktdp` is represented using operations borrowed from MLIR’s SCF dialect, which supports constructs such as conditionals, loops having region-based iterations with explicit loop-carried variables. This allows `ktdp` programs to describe complex kernel, including multi-stage pipelines, conditional execution paths, and iterative computations, while retaining analyzable structure for optimization and scheduling.

SCF dialect - https://mlir.llvm.org/docs/Dialects/SCFDialect/ 


### Overall Design Perspective
In summary, `ktdp` is a compositional IR that integrates established MLIR dialects—such as Arith, Math, LinAlg, MemRef, and SCF—for computation, memory representation, and control flow, while introducing the KTDP dialect to capture specialized abstractions for layouts, distributed views, access tiles, and explicit memory accesses. This multi-dialect design enables `ktdp` to leverage existing MLIR infrastructure and optimizations while providing the domain-specific constructs required to model tile-level execution on data-parallel accelerator architectures with distributed scratchpads and global memory.

## B) Example: Tile-Parallel Matrix Add in `ktdp`

This example demonstrates how `ktdp` operations express a data-parallel tensor addition kernel executing across multiple compute tiles over an accelerator.

```

// An example of two tensors of sizes 96x64 allocated on HBM.
// Each compute tile works at a granularity of 3x64 with total number of compute tiles being 32
// matching with number of cores.
module {
  func.func @add() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tile_size = arith.constant 3 : index
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    // Construct a memory view of A from a given address
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Construct a memory view of B from a given address
    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Construct a memory view of C from a given address
    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Looping over tile size with each iteration working over 1x64 fp16
    scf.for %i = %c0 to %tile_size step %c1 {

        // Construct an access tile set from the memory view of A
        %A_access_tile = ktdp.construct_access_tile %A_view[%start_row + %i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
        } : memref<96x64xf16> -> !ktdp.tile<1x64xindex>

        // Construct an access tile set from the memory view of B
        %B_access_tile = ktdp.construct_access_tile %B_view[%start_row + %i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
        } : memref<96x64xf16> -> !ktdp.tile<1x64xindex>

        // Load data from the corresponding access tile
        %A_data_tile = ktdp.load %A_access_tile : !ktdp.tile<1x64xindex> -> tensor<1x64xf16>

        %B_data_tile = ktdp.load %B_access_tile : !ktdp.tile<1x64xindex> -> tensor<1x64xf16>

        // Perform add operation on the data tiles.
        %C_data_tile = tensor.empty() : tensor<1x64xf16>
        linalg.add ins(%A_data_tile, %B_data_tile : tensor<1x64xf16>, tensor<1x64xf16>)
                    outs(%C_data_tile: tensor<1x64xf16>) -> tensor<1x64xf16>

        // Construct an access tile set from the memory view of C
        %C_access_tile = ktdp.construct_access_tile %C_view[%start_row + %i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
        } : memref<96x64xf16> -> !ktdp.tile<1x64xindex>

        // Store data into the access tile.
        ktdp.store %C_data_tile, %C_access_tile : tensor<1x64xf16>, !ktdp.tile<1x64xindex>
    }

    return
  }
}

```

The above IR implements a tile-parallel elementwise addition kernel using KTDP dialect's memory and access abstractions combined with standard MLIR's compute and control-flow dialects. It operates on three tensors—A, B, and C—each of size 96×64 with FP16 elements stored in HBM memory. The computation is distributed across 32 compute tiles, corresponding to the available cores, where each tile is responsible for processing a distinct 3×64 slice of the tensors. This partitioning ensures that the workload is evenly divided and that each tile works on a non-overlapping region, enabling parallel execution without synchronization conflicts.

At the start of execution, each compute tile determines its identity using the `ktdp.get_compute_tile_id` operation. This ID is used to compute the starting row of the tile’s assigned region by multiplying it with the tile height (3 rows). This establishes a deterministic mapping from tiles to data regions and reflects `ktdp`’s design principle that work partitioning is decided prior to scheduling. Consequently, the IR assumes that decomposition is already fixed and focuses solely on expressing how each tile should execute its portion efficiently.

The program then constructs logical memory views for tensors A, B, and C using `ktdp.construct_memory_view`. These operations interpret raw memory addresses as structured tensors by specifying sizes, strides, and coordinate bounds. They do not allocate or move data; instead, they define how the underlying memory should be viewed. This separation between memory interpretation and data movement is a key KTDP abstraction that allows the compiler to reason precisely about layout, locality, and address computation.

Within each tile, a loop iterates over the three rows assigned to that tile. During each iteration, the program constructs access tiles for A and B using `ktdp.construct_access_tile`. These access tiles symbolically describe the coordinates corresponding to a single row slice of width 64. Rather than performing memory reads immediately, the operation produces a structured representation of the coordinates that will be accessed. This explicit representation of access regions enables advanced analyses, scheduling, and transformations before any actual memory traffic occurs.

Loading is then performed explicitly using `ktdp.load`, which reads values from the specified access tiles of A and B and produces data tiles containing tensor values. The computation itself is carried out using the standard `linalg.add` operation, which adds the two data tiles elementwise to produce a result tile. By delegating arithmetic to existing MLIR compute dialects, `ktdp` avoids redefining compute semantics.

After computation, another access tile is constructed for tensor C that describes the destination coordinates for the result. The result tile is written back using `ktdp.store`, which maps the computed values to their corresponding memory locations through the symbolic access description. This explicit store operation ensures that all writes are well-defined and aligned with the tile’s assigned region.

Overall, this sample IR illustrates `ktdp`’s fundamental design philosophy: computation, memory interpretation, and memory access are represented as separate but composable abstractions. Logical views define how memory is seen, access tiles define data to be accessed, loads and stores perform memory operation, and compute dialects perform arithmetic. Together, these constructs form a tiled data-parallel execution suitable for scheduling and lowering onto spatial accelerators.

The following sections describe the operations and types specific to the KTDP dialect. Operations originating from other dialects are documented in the referenced links above and are not repeated here.

## C) Operations within KTDP dialect

### 1. `ktdp.get_compute_tile_id` (ktdp::GetComputeTileId)

_Gets the id of the current compute tile_


Syntax:

```
operation ::= `ktdp.get_compute_tile_id` attr-dict `:` type(results)
```

The `ktdp.get_compute_tile_id` operation returns the identifier of the
currently executing compute tile.

This operation models a query to the execution environment that exposes
the logical ID of the compute tile on which the operation is executed.
The returned value uniquely identifies the tile within the device’s
execution grid or topology and can be used to specialize computation,
index into distributed data structures, or select tile-specific memory
regions. for, e.g.,
  ```
  %tile_id = ktdp.get_compute_tile_id : index
  ```

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | index

### 2. `ktdp.construct_memory_view` (ktdp::ConstructMemoryViewOp)

_Operation to construct memory view._


Syntax:

```
operation ::= `ktdp.construct_memory_view` $offset `` `,`  `sizes` `` `:`
              custom<DynamicIndexList>($sizes, $static_sizes)
              `` `,` `strides` `` `:`
              custom<DynamicIndexList>($strides, $static_strides)
              attr-dict `:` type($result)
```

The `ktdp.construct_memory_view` operation creates a memref view that
represents a tensor located in a specified memory space at a given base
address, assuming allocation has already been done.

This operation materializes a logical view from a raw memory
location. The `offset` operand specifies the start address in the memory space,
while the `sizes` and `strides` operands define the logical shape and layout of the view.
Static components of sizes and strides may be encoded in the `static_sizes` and
`static_strides` attributes, while dynamic components are supplied as SSA
operands. Together, these describe a strided layout mapping logical tensor
coordinates to memory addresses.

The result type is a memref whose element type and memory space are
determined by the result type annotation. The operation itself does not
allocate memory; it only constructs a view over an existing region of
memory.

The `coordinate_set` attribute specifies the subset of logical
coordinates associated with this view relative to a larger global or
distributed tensor. This allows multiple memory views to represent
disjoint or overlapping regions of a conceptual global tensor view when
building distributed or partitioned views. When no special distribution
semantics are required, the coordinate set typically corresponds directly
to the full index space defined by `sizes`, representing a standard dense
tensor view.

The `memory_space` attribute describes the *physical memory space*
associated with return value (e.g., a memref view).

Conceptually, this operation separates *memory interpretation* from
*memory allocation*: it defines how a region of memory should be viewed as
a tensor, but does not impose any constraints on how that memory was
created or managed.

for, e.g.,
```
  #set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 64 >= 0)>
  %A_view = ktdp.construct_memory_view %A_start_address, sizes: [32, 64], strides: [64, 1] {
      coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>
  } : memref<32x64xf16>
```

Traits: `AttrSizedOperandSegments`, `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>static_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>static_strides</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>memory_space</code></td><td>::mlir::Attribute</td><td></td></tr>
<tr><td><code>coordinate_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `offset` | index
| `sizes` | variadic of index
| `strides` | variadic of index

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | memref of any type values

### 3. `ktdp.construct_distributed_memory_view` (ktdp::ConstructDistributedMemoryViewOp)

_Operation to construct distributed memory view over multiple memref objects_


Syntax:

```
operation ::= `ktdp.construct_distributed_memory_view` `(` ($memrefs^ `:` type($memrefs))? `)`  attr-dict `:` type(results)
```

The `ktdp.construct_distributed_memory_view` operation constructs a single
logical memref view by composing multiple per-partition memory views, each
typically produced by `ktdp.construct_memory_view`.

Each input memref represents a view into a physical memory region, often
residing in a particular memory space (e.g., per-tile scratchpad, HBM
partition) and covering a subset of the coordinates of a
conceptual global tensor. The association between an input memref and its
covered region in the global index space is defined by the `coordinate_set`
carried by the corresponding `construct_memory_view` that created it.

This operation combines these individual views into a unified distributed
view whose global coordinate domain is the union of the coordinate sets of
the inputs. Conceptually, the result behaves like a single memref indexed
over the global tensor dimensions, while its physical storage is distributed
across the provided underlying memrefs.

The operation does not allocate or move data. It only establishes a logical
mapping from global coordinates to one of the constituent memref views. If
coordinate sets overlap, the behavior is unspecified unless additionally
constrained by dialect rules (e.g., requiring disjointness) or by program
semantics.

The result type determines the element type, global shape, and memory-space
abstraction used to represent the distributed view. Lowering is expected to
resolve global indexing into selection of the appropriate underlying memref
and a local coordinate computation, potentially producing explicit address
calculation and communication when required by the target architecture.

This operation is intended as a construction primitive for representing
globally indexed tensors whose storage is partitioned across multiple memory
spaces, enabling explicit modeling of distributed scratchpads and other
non-uniform memory organizations in the IR.

for, e.g.,
```
%A_dview = ktdp.construct_distributed_memory_view (%A0_view, %A1_view : memref<32x64xf16>, memref<32x64xf16>) : memref<64x64xf16>
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `memrefs` | variadic of memref of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
&laquo;unnamed&raquo; | memref of any type values

### 4. `ktdp.construct_access_tile` (ktdp::ConstructAccessTilesOp)

_Operation to construct access tiles over a tensor or memref objects._

The `access_tiles` operation models a logical collection of coordinates
derived from a source memref or tensor. It constructs an *access tile*,
which represents a structured subset of indices into the underlying data
object.

The tile is defined relative to a base coordinate computed from an affine
map applied to a list of operands, which may include induction variables,
symbols, or constants. This base coordinate acts as the anchor point for
the tile. The region of coordinates contained in the tile is then specified
by an IntegerSet attribute that describes the valid points relative to this
base. Using an IntegerSet allows the tile to represent general polyhedral
regions—not just rectangular blocks—but also skewed, strided, triangular,
or otherwise constrained coordinate sets.

Conceptually, this operation separates *address computation* from *data
access*: it materializes a symbolic set of coordinates without performing
any memory read or write. The resulting value is a tile whose elements
encode coordinate tuples describing the selected region. This coordinate
tile is intended to be consumed by subsequent operations (e.g., loads,
stores, gathers, or scatters) that interpret it as an explicit access
specification.

for, e.g.,
```
  %A_access_tile = ktdp.construct_access_tile %A_view[%c0, %c0] {
      access_tile_set = #set
  } : memref<32x64xf16> -> !ktdp.tile<32x64xindex>
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>base_map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
<tr><td><code>access_tile_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `base` | memref of any type values or tensor of any type values
| `indices` | variadic of index

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | Multi-dimensional tile with a fixed number of dimensions and index element type

### 5. `ktdp.construct_indirect_access_tile` (ktdp::ConstructIndirectAccessTilesOp)

_Operation to construct access tiles over a tensor or memref objects._

The `ktir.construct_indirect_access_tile` operation constructs an *access tile*
for indirect (a.k.a. gather/scatter-style) indexing into a base memref/tensor.

The operation defines a collection of logical coordinates into `$base` by
composing (1) one *memory view* per base dimension, and (2) an affine
subscript expression per dimension on to that memory view.
The number of base dimensions must match the number of entries
in `$memory_view_names` and `$memory_view_subscripts`.

Each entry in `$memory_view_names` identifies the memref/tensor view that
provides indices for that base dimension (e.g., an index matrix such as `IDX1`
or `IDX2`). If a dimension does not have a corresponding index view (or is
represented with a sentinel meaning “no index”), that dimension is treated as
*direct* access and its coordinate is taken directly from the variables
referenced by the corresponding subscript.

The `$memory_view_subscripts` attribute is an array of affine expressions,
one per base dimension, that computes the per-dimension index. These
expressions may reference the `$common_variables` operands (typically SSA
values representing induction variables, symbols, or constants) and may also
reference *temporary variables* defined in the operation’s (hidden) region.
The same common variables may be shared across multiple dimension subscripts,
enabling coupled indexing patterns (e.g., both `IDX1[m,k]` and `IDX2[m,k]`
using the same `(m,k)` iterators).

The `$variables_space_set` integer set constrains the domain of all temporary
variables used by the subscripts (introduced by the hidden region).
It defines the set of legal values these intermediate
variables may take while forming the access tile. Conceptually, the operation
enumerates points in this variable space and, for each point, evaluates the
per-dimension subscript expressions to produce a coordinate tuple into `$base`.

Indirect dimensions implicitly read from their associated index views: for a
given variable-space point, the operation evaluates the subscript for that
dimension to form an index into the corresponding memory view, loads the
index value from that view, and uses it as the coordinate for the `$base`
dimension. Direct dimensions bypass this indirection and use the evaluated
subscript result directly as the coordinate.

The result is an access-tile value (modeled as a tile of indices)
that can be consumed by subsequent load/store operations. The operation itself
does not perform any access to `$base`; it only materializes the coordinate
collection describing where accesses should occur.
Lowering is expected to resolve this into explicit address computation and
the required loads from index views.

Example: for `Y[m,k] = X[ IDX1[m,k], IDX2[m,k] ]`, the access tile is formed by
iterating over `(m,k)` in the specified domain, loading `IDX1[m,k]` and
`IDX2[m,k]`, and using the loaded values as the two-dimensional coordinates
into `X`.

 ```

 %Y = ktir.construct_indirect_access_tile %X[%IDX1[m, k], %IDX2[m, k]] {
                access_tile_set = {(m, k) | 0 <= m < 2, 0 <= k < 64},
              } : memref<64x64xfp16>, memref<64x64xfp16>, memref<64x64xfp16>, !ktdp.tile<64x64xindex>
 ```

Traits: `AttrSizedOperandSegments`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>memory_view_subscripts</code></td><td>::mlir::ArrayAttr</td><td>array attribute</td></tr>
<tr><td><code>variables_space_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `base` | memref of any type values or tensor of any type values
| `memory_view_names` | variadic of index
| `common_variables` | variadic of index

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | Multi-dimensional tile with a fixed number of dimensions and index element type

### 6. `ktdp.load` (ktdp::LoadOp)

_Operation to load data based on the coordinates from access_tile._


Syntax:

```
operation ::= `ktdp.load` $access_tile attr-dict `:` type($access_tile) `->` type(results)
```

The `load` performs a logical read from a source memref/tensor
using an *access tile* that encodes a collection of coordinates.

The access tile provides the index tuples to be read. Conceptually, the op
iterates over the coordinates described by the access tile and gathers the
corresponding elements from the source, producing a tensor containing the
loaded values.

The result is always a tensor. Its shape corresponds to the shape of the
access tile’s coordinate collection, and its element type matches the
source element type.

Note: This op is intended to consume the result of an access-tile
construction op.
for, e.g.,
```
  %A_data_tile = ktdp.load %A_access_tile : !ktdp.tile<32x64xindex> -> tensor<32x64xf16>
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `access_tile` | Multi-dimensional tile with a fixed number of dimensions and index element type

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | tensor of any type values

### 7. `ktdp.store` (ktdp::StoreOp)

_Operation to load data based on the coordinates from access_tile._


Syntax:

```
operation ::= `ktdp.store` $data_tile `,` $access_tile attr-dict `:` type($data_tile) `,` type($access_tile)
```

The `store` operation performs a logical write into a destination
memref/tensor using an *access tile* that encodes a collection of coordinates,
and a *data tile* that provides the values to be written.

The access tile (typically produced by `construct_access_tiles`) specifies the set of
index tuples that define the write footprint. The data tile (typically
produced by `tiled_load` and/or subsequent compute operations) supplies one
value per coordinate in the access tile. Conceptually, the operation iterates
over the coordinate collection described by the access tile and stores the
corresponding element from the data tile into the destination at that
coordinate.

This operation models explicit data movement to memory: it has write effects
on the destination. The destination element type must match the element type
of the data tile. The shape of the data tile must match the logical iteration
shape of the access tile (i.e., the number and structure of coordinate points
described by the tile), establishing a one-to-one correspondence between
coordinates and stored values.

The `store` op does not define an ordering among writes beyond the
dialect’s semantics for the access tile.

for, e.g.,
```
ktdp.store %A_data_tile, %A_access_tile : tensor<32x64xf16>, !ktdp.tile<32x64xindex>
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data_tile` | tensor of any type values
| `access_tile` | Multi-dimensional tile with a fixed number of dimensions and index element type



[//]: # (## Attributes)

[//]: # ()
[//]: # (### KTDPMemorySpaceAttr)

[//]: # ()
[//]: # (attributes representing memory spaces in the KTDP IR)

[//]: # ()
[//]: # (Syntax:)

[//]: # ()
[//]: # (```)

[//]: # (#ktdp.memory_space<)

[//]: # (  ::KTDPMemorySpace   # value)

[//]: # (>)

[//]: # (```)

[//]: # ()
[//]: # (Enum cases:)

[//]: # (* unspecified &#40;`unspecified`&#41;)

[//]: # (* LX &#40;`LX`&#41;)

[//]: # (* HBM &#40;`HBM`&#41;)

[//]: # ()
[//]: # (#### Parameters:)

[//]: # ()
[//]: # (| Parameter | C++ type | Description |)

[//]: # (| :-------: | :-------: | ----------- |)

[//]: # (| value | `::KTDPMemorySpace` | an enum of type KTDPMemorySpace |)

[//]: # ()
[//]: # (## Enums)

[//]: # ()
[//]: # (### KTDPMemorySpace)

[//]: # ()
[//]: # (attributes representing memory spaces in the KTDP IR)

[//]: # ()
[//]: # (#### Cases:)

[//]: # ()
[//]: # (| Symbol | Value | String |)

[//]: # (| :----: | :---: | ------ |)

[//]: # (| unspecified | `0` | unspecified |)

[//]: # (| LX | `1` | LX |)

[//]: # (| HBM | `2` | HBM |)

[//]: # ()




## D) Types within KTDP dialect

### 1. AccessTileType

Multi-dimensional tile with a fixed number of dimensions and index element type

Syntax:

```
tile-type ::= `tile` `<` dimension-list `index` `>`
dimension-list ::= (dimension `x`)*
dimension ::= `?` | decimal-literal
```

The tile type is similar to the builtin ranked tensor type but is specific
to the KTDP dialect. It represents aggregate N-dimensional data values with
a fixed rank with a list of dimensions. Each dimension may be a static
non-negative decimal constant or be dynamically determined (indicated by `?`).

Unlike the builtin ranked tensor type, the tile type only supports the
`index` type as its element type. This constraint is enforced during both
parsing and verification.

Example:

```mlir
// Known rank but unknown dimensions.
tile<? x ? x index>

// Partially known dimensions.
tile<? x 64 x index>

// Full static shape.
tile<1 x 64 x index>

// Tile with rank zero. Represents a scalar.
tile<index>

// Zero-element dimensions are allowed.
tile<0 x 42 x index>
```

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| shape | `::llvm::ArrayRef<int64_t>` |  |
| elementType | `Type` |  |


## E) Attributes within KTDP dialect

### 1. SpyreMemorySpaceAttr



Syntax:

```
#ktdp.spyre_memory_space<
  ::mlir::ktdp::KTDP_Spyre_MemoryType,   # memory_type
  int32_t   # core 
  >
```

The `ktdp.spyre_memory_space` attribute describes the *physical memory space*
associated with a value (e.g., a memref view) when targeting the Spyre
architecture. It is used to explicitly model where data resides in the
memory hierarchy, enabling compilation and scheduling decisions that are
sensitive to locality, placement, and cross-core data movement.

The attribute is parameterized by a `KTDP_Spyre_MemoryType` enumeration and
an optional `core` identifier. The memory type specifies which class of
storage the value is associated with:
- `LX`: a core-local scratchpad.
- `HBM`: global high-bandwidth memory shared across cores.

The optional `core` parameter refines the memory space by identifying the
affine with a specific compute core whose local memory is referenced (relevant for `LX`).

The textual form of the attribute is:

`#ktdp.spyre_memory_space<HBM>`
`#ktdp.spyre_memory_space<LX, core = 7>`

Verification ensures that the memory type is a valid Spyre memory kind and
that the `core` field, when specified, is well-formed and consistent with
the chosen memory type (e.g., core is meaningful only for core-local spaces).

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| memory_type | `::mlir::ktdp::KTDP_Spyre_MemoryType` |  |
| core | `int32_t` |  |


## **Metrics **
Expressivity to capture data-parallel mappings for a wide range of operations

## **Drawbacks**

## **Alternatives**

## **Prior Art**

## **How we teach this**

## **Unresolved questions**

## Resolution

### Level of Support
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.

#### Additional Context

### Next Steps
Will implement it.

#### Tracking issue
https://github.com/torch-spyre/torch-spyre/issues/682

#### Exceptions

