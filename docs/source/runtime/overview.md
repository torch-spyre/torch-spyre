# Runtime Overview

The Torch-Spyre runtime layer manages device lifecycle, memory
allocation, and kernel execution at inference time.

## Responsibilities

- **Device registration** — registering `spyre` as a PyTorch device type
- **Tensor memory management** — allocating and freeing device DRAM (DDR)
  for `SpyreTensorImpl` objects
- **DMA transfers** — moving tensor data between host (CPU) memory and
  device (DDR) memory via the `to()` / `from_device()` APIs
- **Kernel dispatch** — loading compiled program binaries and
  orchestrating their execution across Spyre cores

:::{figure} ../_static/images/pytorch-dispatcher.png
:alt: PyTorch Dispatcher routing a Spyre tensor operation through the dispatch table
:width: 45%
:align: center

The PyTorch Dispatcher routes each operation to the correct device implementation. When a `torch.add` call carries Spyre tensors, the Dispatcher looks up `SPYRE` in its dispatch table and calls the registered `spyre__add_Tensor` kernel. Torch-Spyre registers all its eager runtime kernels in this table via `TORCH_LIBRARY_IMPL`.
:::

## Device Registration

Torch-Spyre registers `spyre` as a PyTorch device using the
`PrivateUse1` mechanism — the standard PyTorch pathway for out-of-tree
accelerators:

```python
torch.utils.rename_privateuse1_backend("spyre")
torch._register_device_module("spyre", make_spyre_module())
```

This gives the device a human-readable name (`"spyre"`) without
requiring any upstream PyTorch changes. A custom
`SpyreGuardImpl` implements `c10::impl::DeviceGuardImplInterface`
to handle device management and synchronization.

## Key C++ Components

| File | Responsibility |
|------|---------------|
| `csrc/module.cpp` | PyTorch extension entry point and device registration |
| `csrc/spyre_tensor_impl.cpp` | `SpyreTensorImpl` — the device tensor backing store |
| `csrc/spyre_mem.cpp` | Device memory allocation and DMA |
| `csrc/spyre_views.cpp` | Tensor view and striding support on device |
| `csrc/spyre_guard.cpp` | `SpyreGuardImpl` — device guard and synchronization |
| `csrc/spyre_stream.cpp` | Stream management for asynchronous execution |
| `csrc/attn_utils.cpp` | SDPA dispatch — routes `scaled_dot_product_attention` to the Spyre backend |

## Python Entry Point

`torch_spyre/__init__.py` is loaded automatically by PyTorch via the
`torch.backends` entry point declared in `pyproject.toml`. This triggers
device and backend registration without requiring an explicit import.

:::{figure} ../_static/images/spyre-device-allocator.png
:alt: Spyre device allocator call chain from torch.empty to Flex::TryAllocate
:width: 40%
:align: center

The Spyre device allocator call chain. A `torch.empty(..., device="spyre")` call flows through `spyre_empty_strided` into `SpyreAllocator::allocate`, which delegates to the underlying `Flex::TryAllocate` for physical device memory.
:::

## Memory Model

Spyre tensors live in device DRAM (DDR). Their layout is described by
`SpyreTensorLayout` (see
[Tensors and Layouts](../user_guide/tensors_and_layouts.md)), which
encodes tiling, padding, and stick dimensions.

### SpyreTensorImpl

Because PyTorch's standard `size()` and `strides()` cannot fully
represent tiled device layouts, Torch-Spyre defines `SpyreTensorImpl`
— a subclass of `TensorImpl`. It embeds a `SpyreTensorLayout` instance
that captures:

- `device_size` — shape on device, with extra tiling dimensions and padded values
- `stride_map` — host stride for each device dimension (-1 for synthetic/padded dimensions)
- `device_dtype` — on-device data format (e.g. `SEN169_FP16`)

Tensor handles returned to Python **do not contain raw physical
pointers** — a security requirement on IBM Z systems.

### SpyreAllocator: PF and VF Modes

Torch-Spyre supports two allocator modes, selected at startup:

| Mode | Description |
|------|-------------|
| **PF (Physical Frame)** | Allocates device memory eagerly per tensor based on its stickified size. Each tensor maps directly to a `DeviceMemoryAllocationPtr`. No limit on concurrent allocations. |
| **VF (Virtual Frame)** | Allocates one large memory region on device startup. Virtual allocations are sub-divided within this region — similar to the CUDA cache allocator. Limits total concurrent allocations but reduces per-tensor runtime overhead. |

The large chunk in VF mode is allocated via `Flex::TryAllocate`. Within
the `SpyreAllocator`, sub-regions are assigned to tensors and freed
back to the pool when the Python garbage collector releases them.

## Eager Operations

Torch-Spyre registers eager kernels for ATen operations via
`TORCH_LIBRARY_IMPL`:

```cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", TORCH_FN(spyre__add_Tensor));
    m.impl("mm", TORCH_FN(spyre__mm));
    // ...
}
```

Many eager ops are themselves compiled via `torch.compile` (AOT path)
so they do not need to be hand-written. The compiled artifacts are
cached using the standard `torch.compile` cache, so subsequent
invocations do not re-compile.

## Streams

Torch-Spyre supports stream-based asynchronous execution, following the
same API pattern as `torch.cuda` streams:

| API | Description |
|-----|-------------|
| `torch.spyre.Stream()` | Create a new Spyre stream |
| `torch.spyre.current_stream()` | Get the current stream for the device |
| `torch.spyre.default_stream()` | Get the default stream for the device |
| `torch.spyre.synchronize()` | Wait for all operations on all streams to complete |

Streams are implemented in `torch_spyre/streams.py` (Python) and
`csrc/spyre_stream.cpp` (C++).

## Multi-Card Support

Ensembles of up to 8 Spyre cards deliver up to 1 TB of aggregate device
memory. Multi-card communication follows the standard PyTorch collective
communications API (all-reduce, all-gather, reduce-scatter) via a
custom `ProcessGroup` implementation.

## TODO

- Document kernel launch sequence and Control Block Stream design
- Document error handling and device reset
