# Compile-Time Collective Decomposition Design for Torch-Spyre

## Executive Summary

This document proposes a design for moving collective operation decomposition from runtime to compile time in the Spyre backend. The goal is to enable the Inductor compiler to reason about and optimize decomposed collective operations (e.g., allreduce as tree-based reduce + broadcast), allowing better scheduling, fusion, and communication-compute overlap.

---

## Current Architecture Overview

### Current State (Runtime Decomposition)

```
User Code: dist.allreduce(tensor, group_name='default')
    ↓
_c10d_functional.allreduce (functional op)
    ↓ (Dynamo Tracing)
Pass-through Kernel (kernels.py)
    ↓ (Inductor Lowering)
SpyreAlloredAsyncFallback IR Node
    ↓ (Codegen)
torch.ops.spyre.allreduce_async(tensor, group_name)
    ↓ (Runtime - C++ Dispatcher)
spyre_distributed.cpp: spyre_allreduce_async_impl()
    ↓ (RUNTIME DECOMPOSITION HERE)
spyre_comms::allreduce() → decomposes internally
    ├── Reduce locally to rank 0
    ├── Broadcast result from rank 0
    └── (Compiler has NO visibility into these steps)
    ↓
Hardware Communication
```

**Problems with Runtime Decomposition:**

1. **No compiler visibility**: Inductor sees `allreduce` as a single atomic op, can't reason about sub-operations

2. **No scheduling opportunities**: Can't interleave compute with reduce phase and broadcast phase separately

3. **No fusion opportunities**: Reduce operations aren't fused with preceding compute, broadcast isn't fused with following compute

4. **Inefficient work division**: Compiler can't co-optimize barrier/collective boundaries with compute kernels

---

## Proposed Compile-Time Decomposition Architecture

### High-Level Design

```
User Code: dist.allreduce(tensor, group_name='default')
    ↓
_c10d_functional.allreduce (functional op)
    ↓ (Dynamo Tracing)
Pass-through Kernel (kernels.py) - unchanged
    ↓ (Inductor LOWERING - NEW STEP)
┌─────────────────────────────────────────────────────┐
│ COMPILE-TIME DECOMPOSITION                          │
│ lowering.py: lower_allreduce_decomposed()          │
│                                                     │
│ Calls: spyre_collective_decomposer.decompose(...)  │
│ Returns: [ReduceIR, BroadcastIR] IR nodes          │
│ (Inductor now sees the full operation tree)         │
└─────────────────────────────────────────────────────┘
    ↓
SpyreReduceAsyncFallback IR Node
SpyreBroadcastAsyncFallback IR Node
    ↓ (Codegen)
torch.ops.spyre.reduce_async(...)
torch.ops.spyre.broadcast_async(...)
    ↓ (Runtime - C++ Dispatcher)
Hardware Communication (optimized scheduling)
```

**Benefits:**

1. ✓ Compiler has full visibility of operation graph

2. ✓ Scheduler can interleave compute with reduce and broadcast phases

3. ✓ Reduction can be fused with preceding compute

4. ✓ Broadcast can be fused with following compute

5. ✓ Better work division planning across cores

---

## Detailed Architecture

### 1\. Collective Decomposition Layer (`torch_spyre/_inductor/collective_decomposer.py`)

**Purpose**: Encapsulate all collective decomposition logic in a Python module that can be called at compile time.

```python
# File: torch_spyre/_inductor/collective_decomposer.py

from typing import List, Tuple, Optional, Any
import torch
from torch._inductor import ir

class CollectiveDecomposition:
    """Result of decomposing a collective operation."""
    
    def __init__(self, op_name: str, world_size: int, rank: int):
        self.op_name = op_name
        self.world_size = world_size
        self.rank = rank
        self.operations: List[Tuple[str, dict]] = []
    
    def add_operation(self, op_type: str, params: dict):
        """Add a decomposed operation (reduce, broadcast, barrier, etc.)."""
        self.operations.append((op_type, params))
    
    def __repr__(self):
        return f"CollectiveDecomposition({self.op_name}, {len(self.operations)} ops)"


class SpyreCollectiveDecomposer:
    """
    Decomposes collective operations into primitive operations.
    
    Knows about:
    - Spyre-specific optimizations (e.g., broadcast tree topology)
    - World size and rank constraints
    - Available tensor operations for decomposition
    
    Examples:
        allreduce(x) → reduce(x, root=0) + broadcast(x, root=0)
        allgather(x) → gather_to_all(x, root=0)
        reduce_scatter(x) → scatter(x, root=0)
    """
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
    
    @staticmethod
    def decompose_allreduce(
        tensor_shape: Tuple[int, ...],
        tensor_dtype: torch.dtype,
        reduction_op: str = "sum",
        group_name: str = "default",
    ) -> CollectiveDecomposition:
        """
        Decompose allreduce into reduce + broadcast.
        
        allreduce(x, op=SUM, group=g) →
          y = reduce(x, op=SUM, root=0, group=g)  # All ranks → rank 0
          z = broadcast(y, root=0, group=g)        # Rank 0 → all ranks
        
        Args:
            tensor_shape: Shape of tensor being reduced
            tensor_dtype: Data type
            reduction_op: "sum", "prod", "max", "min", "avg"
            group_name: Process group name
        
        Returns:
            CollectiveDecomposition with [reduce, broadcast] operations
        """
        decomp = CollectiveDecomposition("allreduce", 
                                        world_size=...,
                                        rank=...)
        
        decomp.add_operation("reduce", {
            "tensor_shape": tensor_shape,
            "tensor_dtype": tensor_dtype,
            "reduction_op": reduction_op,
            "root_rank": 0,
            "group_name": group_name,
        })
        
        decomp.add_operation("broadcast", {
            "tensor_shape": tensor_shape,
            "tensor_dtype": tensor_dtype,
            "root_rank": 0,
            "group_name": group_name,
        })
        
        return decomp
    
    @staticmethod
    def decompose_allgather(
        tensor_shape: Tuple[int, ...],
        tensor_dtype: torch.dtype,
        group_name: str = "default",
    ) -> CollectiveDecomposition:
        """
        Decompose allgather into gather-to-all.
        
        allgather(x, group=g) →
          y = gather_to_all(x, group=g)  # All ranks send to all ranks
        """
        decomp = CollectiveDecomposition("allgather",
                                        world_size=...,
                                        rank=...)
        
        decomp.add_operation("gather_to_all", {
            "local_tensor_shape": tensor_shape,
            "global_tensor_shape": (tensor_shape[0] * ...,) + tensor_shape[1:],
            "tensor_dtype": tensor_dtype,
            "group_name": group_name,
        })
        
        return decomp
    
    @staticmethod
    def decompose_reduce_scatter(
        tensor_shape: Tuple[int, ...],
        tensor_dtype: torch.dtype,
        reduction_op: str = "sum",
        group_name: str = "default",
    ) -> CollectiveDecomposition:
        """
        Decompose reduce_scatter into reduce + scatter.
        
        reduce_scatter(x, op=SUM, group=g) →
          y = reduce(x, op=SUM, root=0, group=g)   # All ranks → rank 0
          z = scatter(y, root=0, group=g)           # Rank 0 → all ranks
        """
        decomp = CollectiveDecomposition("reduce_scatter",
                                        world_size=...,
                                        rank=...)
        
        decomp.add_operation("reduce", {
            "tensor_shape": tensor_shape,
            "tensor_dtype": tensor_dtype,
            "reduction_op": reduction_op,
            "root_rank": 0,
            "group_name": group_name,
        })
        
        decomp.add_operation("scatter", {
            "tensor_shape": tensor_shape,
            "tensor_dtype": tensor_dtype,
            "root_rank": 0,
            "group_name": group_name,
        })
        
        return decomp


# Global decomposer instance
_decomposer_instance: Optional[SpyreCollectiveDecomposer] = None

def get_decomposer() -> SpyreCollectiveDecomposer:
    """Get or create the global decomposer."""
    global _decomposer_instance
    if _decomposer_instance is None:
        # Get rank/world_size from distributed context
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        _decomposer_instance = SpyreCollectiveDecomposer(world_size, rank)
    return _decomposer_instance
```

### 2\. New Lowering Rules in `lowering.py`

Add compile-time decomposition lowering rules:

```python
# File: torch_spyre/_inductor/lowering.py (additions)

from torch_spyre._inductor.collective_decomposer import (
    get_decomposer,
    CollectiveDecomposition,
)

@register_spyre_lowering(torch.ops._c10d_functional.allreduce.default)
def lower_c10d_allreduce_decomposed(tensor, op, group_name):
    """
    Decompose allreduce into reduce + broadcast at compile time.
    
    This lowering rule runs during Inductor's lowering phase and converts
    the functional allreduce into a sequence of IR nodes (reduce, then broadcast)
    that Inductor can reason about for scheduling and fusion.
    
    Architecture:
        _c10d_functional.allreduce
            ↓ (This lowering rule)
        [SpyreReduceAsyncFallback, SpyreBroadcastAsyncFallback]
            ↓ (Codegen generates)
        torch.ops.spyre.reduce_async(...) 
        + torch.ops.spyre.broadcast_async(...)
    
    Args:
        tensor: Input tensor IR node
        op: Reduction operation ("sum", "prod", "max", "min", "avg")
        group_name: Process group name
    
    Returns:
        TensorBox containing the final broadcast result
    """
    tensor.realize()
    
    # Get collective decomposer
    decomposer = get_decomposer()
    
    # Decompose allreduce at compile time
    decomp = decomposer.decompose_allreduce(
        tensor_shape=tensor.get_size(),
        tensor_dtype=tensor.get_dtype(),
        reduction_op=op,
        group_name=group_name,
    )
    
    logger.info(
        f"Decomposing allreduce (op={op}, group={group_name}) into "
        f"{len(decomp.operations)} operations at compile time"
    )
    
    # Create IR nodes for each decomposed operation
    current_node = tensor
    for op_type, params in decomp.operations:
        if op_type == "reduce":
            logger.debug(f"  Creating reduce IR node: {params}")
            current_node = _create_reduce_ir_node(current_node, **params)
        elif op_type == "broadcast":
            logger.debug(f"  Creating broadcast IR node: {params}")
            current_node = _create_broadcast_ir_node(current_node, **params)
        else:
            raise ValueError(f"Unknown decomposed op: {op_type}")
    
    return current_node


def _create_reduce_ir_node(
    tensor,
    tensor_shape,
    tensor_dtype,
    reduction_op,
    root_rank,
    group_name,
):
    """Helper to create a SpyreReduceAsyncFallback IR node."""
    tensor.realize()
    return ir.TensorBox.create(
        SpyreReduceAsyncFallback(
            torch.ops.spyre.reduce_async.default,
            tensor,
            reduction_op=reduction_op,
            root_rank=root_rank,
            group_name=group_name,
        )
    )


def _create_broadcast_ir_node(
    tensor,
    tensor_shape,
    tensor_dtype,
    root_rank,
    group_name,
):
    """Helper to create a SpyreBroadcastAsyncFallback IR node."""
    tensor.realize()
    return ir.TensorBox.create(
        SpyreBroadcastAsyncFallback(
            torch.ops.spyre.broadcast_async.default,
            tensor,
            src_rank=root_rank,
            group_name=group_name,
        )
    )


@register_spyre_lowering(torch.ops._c10d_functional.allgather.default)
def lower_c10d_allgather_decomposed(tensor, group_name):
    """Similar pattern for allgather."""
    # Implementation follows same pattern as allreduce
    pass


@register_spyre_lowering(torch.ops._c10d_functional.reduce_scatter.default)
def lower_c10d_reduce_scatter_decomposed(tensor, op, group_name):
    """Similar pattern for reduce_scatter."""
    # Implementation follows same pattern as allreduce
    pass
```

### 3\. New IR Nodes in `ir.py`

Add reduce-specific IR nodes (broadcast already exists):

```python
# File: torch_spyre/_inductor/ir.py (additions)

class SpyreReduceAsyncFallback(ir.ExternKernel):
    """IR node for spyre.reduce_async — emits a runtime call to async reduce.
    
    This starts a reduce operation asynchronously and returns immediately,
    allowing computation to proceed while communication is in progress.
    """
    
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        """Generate code to call torch.ops.spyre.reduce_async at runtime."""
        input_tensor = self.inputs[0]
        input_name = input_tensor.codegen_reference()
        
        reduction_op = self.reduction_op
        root_rank = self.root_rank
        group_name = self.group_name
        
        output_name = self.get_name()
        generated_code = (
            f"{output_name} = torch.ops.spyre.reduce_async("
            f"{input_name}, '{reduction_op}', {root_rank}, '{group_name}')"
        )
        
        logger.debug(
            f"Codegen reduce_async: {input_name} -> {output_name} "
            f"(op={reduction_op}, root={root_rank}, group='{group_name}')"
        )
        
        wrapper.writeline(generated_code)
    
    def should_allocate(self) -> bool:
        return False
    
    def get_mutation_names(self) -> Sequence[str]:
        return []
    
    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()
    
    def __init__(
        self,
        op_overload: torch._ops.OpOverload,
        x: IRNode,
        reduction_op: str,
        root_rank: int,
        group_name: str,
    ) -> None:
        # Reduce returns a tensor with the same layout as input
        layout = x.get_layout()
        super().__init__(
            None,
            layout,
            [x],
            (reduction_op, root_rank, group_name),
            python_kernel_name="torch.ops.spyre.reduce_async",
            op_overload=op_overload,
        )
        self.reduction_op = reduction_op
        self.root_rank = root_rank
        self.group_name = group_name
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
```

### 4\. New Custom Ops in `spyre_library.py`

Register reduce and wait operations:

```python
# File: torch_spyre/_inductor/distributed/spyre_library.py (additions)

@torch.library.custom_op("spyre::reduce_async", mutates_args=())
def reduce_async(
    x: torch.Tensor,
    reduction_op: str = "sum",
    root_rank: int = 0,
    group_name: str = "default",
) -> torch.Tensor:
    """Async reduce operation - returns immediately, communication in background."""
    raise RuntimeError(
        "This should never be called - C++ dispatcher should handle all calls."
    )


@reduce_async.register_fake
def _(x, reduction_op="sum", root_rank=0, group_name="default"):
    """Fake implementation for shape inference during compilation."""
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)


# Similarly for scatter_async, gather_to_all_async, etc.
```

### 5\. C++ Runtime Implementation

Add reduce dispatcher in `spyre_distributed.cpp`:

```cpp
// torch_spyre/csrc/distributed/spyre_distributed.cpp (additions)

at::Tensor spyre_reduce_async_impl(
    const at::Tensor& input,
    const std::string& reduction_op,
    int64_t root_rank,
    const std::string& group_name) {
  
  DEBUGINFO("spyre::reduce_async called with reduction_op=", reduction_op,
            ", root_rank=", root_rank, ", group=", group_name);
  
  // Get world context
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    spyre_comms::initialize_library();
    context = spyre_comms::get_world_context();
  }
  
  // Validate root_rank
  TORCH_CHECK(
      root_rank >= 0 && root_rank < static_cast<int64_t>(context->getSize()),
      "root_rank out of range: ", root_rank);
  
  // Create output tensor (only root rank receives data)
  at::Tensor output = at::empty_like(input);
  
  // Convert reduction op string to spyre_comms enum
  auto reduce_op = string_to_reduce_op(reduction_op);
  
  // Start reduce (non-blocking)
  auto work_schedule = context->reduce(
      output,
      input,
      reduce_op,
      static_cast<spyre_comms::process_id_t>(root_rank),
      group_name
  );
  
  TORCH_CHECK(work_schedule != nullptr,
              "Reduce operation failed to create work schedule");
  
  work_schedule->start();  // Start but DON'T wait
  
  // Store WorkSchedule in map
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
        output.storage().data_ptr().get_context());
    pending_work_map_.emplace(ctx, PendingWork{std::move(work_schedule)});
  }
  
  return output;  // Return immediately without waiting
}

// Register the reduce operation
TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m) {
  m.impl("reduce_async", &spyre::spyre_reduce_async_impl);
  // ... other ops
}
```

---

## Execution Flow Example: Compile-Time Decomposed Allreduce

```python
@torch.compile
def distributed_sum(x):
    # Standard PyTorch functional collective
    return torch.ops._c10d_functional.allreduce(x, "sum", "default")

# Create Spyre tensor
x = torch.ones(1024, device='spyre:0')

# First call triggers compilation
result = distributed_sum(x)
```

### Detailed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Tracing Phase (Dynamo)                                  │
├─────────────────────────────────────────────────────────────────┤
│ Dynamo intercepts the allreduce call                            │
│ Executes: torch.ops._c10d_functional.allreduce(x, "sum", "...")│
│ PyTorch dispatcher → kernels.py pass-through                    │
│ Returns: x (unchanged, just for tracing)                        │
│ FX graph records: allreduce node                                │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Lowering Phase (Inductor) - COMPILE-TIME DECOMPOSITION │
├─────────────────────────────────────────────────────────────────┤
│ Inductor's lowering phase receives FX graph                    │
│ Node: _c10d_functional.allreduce(x, "sum", "default")         │
│     ↓                                                            │
│ lower_c10d_allreduce_decomposed() fires                        │
│     ↓                                                            │
│ Calls: get_decomposer().decompose_allreduce(...)              │
│     ↓ (COMPILE TIME - not runtime!)                           │
│ Returns: CollectiveDecomposition([reduce, broadcast])          │
│     ↓                                                            │
│ Creates IR nodes:                                               │
│   1. SpyreReduceAsyncFallback(x, "sum", root=0, "default")    │
│      → reduce_result                                            │
│   2. SpyreBroadcastAsyncFallback(reduce_result, root=0, ...)  │
│      → broadcast_result                                         │
│     ↓                                                            │
│ Inductor now sees full decomposed graph:                       │
│   - Can reason about reduce operation → fuse with prior compute│
│   - Can reason about broadcast → fuse with following compute   │
│   - Can schedule reduce and broadcast separately               │
│   - Can plan barrier/collective boundaries                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Codegen Phase (Inductor)                               │
├─────────────────────────────────────────────────────────────────┤
│ SpyreReduceAsyncFallback.codegen()                             │
│   Generates: buf1 = torch.ops.spyre.reduce_async(buf0, ...)   │
│ SpyreBroadcastAsyncFallback.codegen()                          │
│   Generates: buf2 = torch.ops.spyre.broadcast_async(buf1, ...) │
│                                                                  │
│ Final compiled code:                                            │
│   buf1 = torch.ops.spyre.reduce_async(x, "sum", 0, "default") │
│   buf2 = torch.ops.spyre.broadcast_async(buf1, 0, "default")  │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Runtime Execution                                       │
├─────────────────────────────────────────────────────────────────┤
│ Execute: torch.ops.spyre.reduce_async(x, "sum", 0, "default") │
│   → PyTorch dispatcher → spyre_reduce_async_impl()             │
│   → spyre_comms::reduce() starts non-blocking                  │
│   → Returns immediately                                        │
│     ↓                                                            │
│ Execute: torch.ops.spyre.broadcast_async(buf1, 0, "default")  │
│   → PyTorch dispatcher → spyre_broadcast_async_impl()          │
│   → spyre_comms::broadcast() starts non-blocking               │
│   → Returns immediately                                        │
│     ↓                                                            │
│ Hardware: Reduce completes, then broadcast completes           │
│ (Compiler could have scheduled compute between phases)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1\. **Decomposition at Compile Time, Execution at Runtime**

- **Compile time**: Determine *what* operations are needed (reduce, broadcast)

- **Runtime**: Execute those operations asynchronously

This separation enables:

- Compiler visibility and optimization (lowering phase)

- Efficient hardware execution (spyre-comms)

- Potential communication-compute overlap

### 2\. **Decomposer as Separate Module**

Created `collective_decomposer.py` as a pure Python module that:

- Has no dependencies on Inductor IR (except for return types)

- Can be unit tested independently

- Can be extended with new decompositions easily

- Encapsulates all decomposition logic

### 3\. **IR Nodes Represent Primitive Operations**

After decomposition, IR nodes represent:

- `reduce_async`: Single collective reduce operation

- `broadcast_async`: Single collective broadcast operation (already exists)

This allows Inductor to:

- Fuse reduce with preceding pointwise operations

- Fuse broadcast with following pointwise operations

- Schedule barriers appropriately

### 4\. **Async Pattern with Wait**

Following the existing broadcast pattern:

```
tensor_after_reduce = reduce_async(tensor)       # Non-blocking, returns immediately
wait_work(tensor_after_reduce)                   # Blocks until complete
```

This enables:

- Communication-compute overlap (schedule compute between reduce_async and wait)

- Pipelining (multiple reduces in flight)

- Better resource utilization

### 5\. **World Size and Rank from Runtime Context**

Decomposer reads `dist.get_rank()` and `dist.get_world_size()` at compile time because:

- These are stable during compilation

- Compilation happens once per process

- Allows compile-time assertions (e.g., "root rank must be < world_size")

Future optimization: Cache decompositions by (rank, world_size, op_name) signature to enable multi-rank compilation scenarios.

---

## Integration Points

### Files Modified/Created

| File                     | Change   | Purpose                                                  |
|------------------------|--------|--------------------------------------------------------|
| collective_decomposer.py | NEW      | Decomposition logic, pure Python                         |
| lowering.py              | MODIFIED | Add decomposed lowering rules                            |
| ir.py                    | MODIFIED | Add SpyreReduceAsyncFallback and other IR nodes          |
| spyre_library.py         | MODIFIED | Register reduce_async, scatter_async, etc.               |
| spyre_distributed.cpp    | MODIFIED | Implement reduce_async, scatter_async, etc. C++ dispatch |

### Backward Compatibility

- Existing code paths (immediate allreduce, non-decomposed) continue to work

- New decomposed paths can be enabled via config flag initially

- Gradual migration to compile-time decomposition

### Configuration and Control

Proposed env var to control decomposition:

```bash
# Use compile-time decomposition (NEW)
export TORCH_SPYRE_COLLECTIVE_DECOMPOSE_COMPILE_TIME=1

# Use runtime decomposition (OLD, default for now)
export TORCH_SPYRE_COLLECTIVE_DECOMPOSE_COMPILE_TIME=0
```

Lowering rules check this flag:

```python
@register_spyre_lowering(torch.ops._c10d_functional.allreduce.default)
def lower_c10d_allreduce_decomposed(tensor, op, group_name):
    if not os.getenv("TORCH_SPYRE_COLLECTIVE_DECOMPOSE_COMPILE_TIME"):
        # Fall back to atomic allreduce
        return lower_c10d_allreduce_atomic(tensor, op, group_name)
    
    # New compile-time decomposition path
    ...
```

---

## Example: Allreduce Decomposition Details

### Decomposition Pattern

```
allreduce(x, op=SUM, group=g) →

PHASE 1 - REDUCE:
  For rank r < world_size:
    if r == 0:
      y = reduce(x, op=SUM, root=0, group=g)  # Receive reductions from all
    else:
      y = reduce(x, op=SUM, root=0, group=g)  # Send x to rank 0

PHASE 2 - BROADCAST:
  For rank r < world_size:
    z = broadcast(y, root=0, group=g)         # Rank 0 sends y, others receive
```

### What the IR Graph Looks Like (Before Decomposition)

```
Input: x (shape=[1024], dtype=float32, device=spyre)
  ↓
[allreduce node]
  ↓
Output: result (shape=[1024], dtype=float32)
```

### What the IR Graph Looks Like (After Decomposition)

```
Input: x (shape=[1024], dtype=float32, device=spyre)
  ↓
[reduce_async node]
  │ (reduce all ranks' x to rank 0)
  ↓
reduced_tensor (shape=[1024], dtype=float32)
  ↓
[broadcast_async node]
  │ (broadcast reduced result from rank 0 to all)
  ↓
Output: result (shape=[1024], dtype=float32)
```

### Inductor's View

With decomposition, Inductor can:

1. **Fuse reduce with preceding compute**:

```python
# User code:
y = x * 2
z = dist.allreduce(y)

# Inductor sees:
y = pointwise_mul(x, 2)  # Could be fused with...
y_reduced = reduce_async(y, ...)  # ...the reduce operation
y_final = broadcast_async(y_reduced, ...)
```

2. **Schedule barriers properly**:

```python
# Inductor knows:
# - Reduce phase must have all reduce_async calls before any broadcast
# - Barrier between reduce and broadcast phases
# - Broadcast phase can start after all reduces complete
```

3. **Plan work division**:

```python
# Inductor can coordinate:
# - Core assignment for reduce trees
# - Core assignment for broadcast trees
# - Synchronization points for collectives
```

---

## Testing Strategy

### Unit Tests (`collective_decomposer_test.py`)

```python
def test_decompose_allreduce():
    decomp = decomposer.decompose_allreduce(
        tensor_shape=(1024,),
        tensor_dtype=torch.float32,
        reduction_op="sum",
        group_name="default",
    )
    
    assert len(decomp.operations) == 2
    assert decomp.operations[0][0] == "reduce"
    assert decomp.operations[1][0] == "broadcast"

def test_decompose_allgather():
    decomp = decomposer.decompose_allgather(
        tensor_shape=(1024,),
        tensor_dtype=torch.float32,
        group_name="default",
    )
    
    assert len(decomp.operations) == 1
    assert decomp.operations[0][0] == "gather_to_all"
```

### Integration Tests (`test_collective_lowering.py`)

```python
def test_allreduce_lowering_decomposed():
    """Verify allreduce is decomposed at compile time."""
    
    @torch.compile
    def allreduce_fn(x):
        return dist.allreduce(x)
    
    x = torch.ones(128, device='spyre:0')
    result = allreduce_fn(x)
    
    # Verify result is correct
    assert torch.allclose(result, torch.full_like(result, 128.0))
    
    # Check that lowering fired (via logger capture)
    assert "Decomposing allreduce" in captured_logs

def test_allreduce_vs_naive():
    """Compare decomposed allreduce with naive reduce+broadcast."""
    
    x = torch.ones(128, device='spyre:0')
    
    # Method 1: Decomposed (compile time)
    @torch.compile
    def decomposed(x):
        return dist.allreduce(x)
    
    result1 = decomposed(x)
    
    # Method 2: Naive (compute reduce + broadcast manually)
    def naive(x):
        y = dist.reduce(x, dst=0)
        z = dist.broadcast(y, src=0)
        return z
    
    result2 = naive(x)
    
    # Results should match
    assert torch.allclose(result1, result2)
```

---

## Performance Implications

### Expected Benefits

1. **Compiler Visibility**: Inductor sees all operations, enabling:

   - Fusion between compute and collectives

   - Better work division planning

   - Barrier optimization

2. **Reduced Idle Time**: With visibility, Inductor can:

   - Interleave computation between reduce and broadcast phases

   - Schedule other ranks' work during collective phases

   - Optimize synchronization points

3. **Memory Efficiency**: Compiler can:

   - Reuse temporaries between operations

   - Plan buffer allocation better

   - Reduce peak memory usage

### Potential Overheads

1. **Decomposition CPU Cost**: Small (< 1ms per collective in compile phase)

2. **IR Size**: Slightly larger (3 IR nodes instead of 1), but still negligible

3. **Generated Code**: Slightly longer, but C++ inlining mitigates this

---

## Future Extensions

### Phase 2: Multi-Algorithm Decompositions

Different decomposition strategies for different scenarios:

```python
@staticmethod
def decompose_allreduce_ring(tensor, op, group_name):
    """Ring-based allreduce: N-1 steps, lower bandwidth waste."""
    ...

@staticmethod
def decompose_allreduce_tree(tensor, op, group_name):
    """Tree-based allreduce: log(N) steps, good for small N."""
    ...

@staticmethod
def decompose_allreduce_butterfly(tensor, op, group_name):
    """Butterfly pattern: optimized for specific topologies."""
    ...
```

### Phase 3: Topology-Aware Decomposition

Query Spyre device topology and choose optimal algorithms:

```python
decomposer = SpyreCollectiveDecomposer(
    world_size=4,
    rank=0,
    topology="mesh_2d",  # Query from Spyre device
    bandwidth_matrix=...,  # Measured inter-rank bandwidth
)

decomp = decomposer.decompose_allreduce(...)  # Chooses algorithm
```

### Phase 4: Communication-Compute Overlap Scheduling

Inductor's scheduler directly interleaves compute with collective phases:

```
Rank 0:
  reduce_async(x)  // Start collective
  // Can do compute here while collective in flight
  compute_something()
  wait_work(x)
  
Rank 1:
  reduce_async(y)
  compute_something_else()
  wait_work(y)
```

---

## Summary

This design moves collective decomposition from runtime (in spyre-comms) to compile time (in Torch-Spyre's Inductor lowering), enabling:

1. **Compiler Visibility**: Inductor sees the full decomposed operation graph

2. **Better Optimization**: Fusion, scheduling, work division planning

3. **Performance**: Potential for communication-compute overlap and reduced idle time

4. **Flexibility**: Easy to add new decomposition algorithms and strategies

The architecture cleanly separates:

- **Decomposition Logic** (pure Python in `collective_decomposer.py`)

- **Lowering Rules** (in `lowering.py` using decomposer)

- **IR Nodes** (in `ir.py` representing decomposed operations)

- **C++ Runtime** (in `spyre_distributed.cpp` executing operations)

This enables incremental adoption, testing, and future optimization.

### Notes from Sukriti

- this is a high level design prepared by claude , and has not been reviewed or revised by me. Sharing this to have an initial idea of the problem and a possible solution that can be adapted.

- Next step: We need to understand the decomposition API Spyre-Comms provides and fit it in the lowering phase. The API will be in C++, so we could regiter another custom op for it to call from python, like torch.ops.broadcast or something like that (will have to redesign how to call C++ api from python lowering)

- This design would also call decomposition per collective at compile time . We eventually want a phase to read entire graph and call decomposition once - this needs to be redesigned

- We can start with per collective lowering and decomposition at compile time and then extend to graph level planning