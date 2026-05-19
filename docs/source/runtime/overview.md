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
accelerators. Registration happens in `torch_spyre/__init__.py`'s
`_autoload()`:

```python
torch.utils.rename_privateuse1_backend("spyre")
torch._register_device_module("spyre", make_spyre_module())
```

This gives the device a human-readable name (`"spyre"`) without
requiring any upstream PyTorch changes. A custom
`SpyreGuardImpl` implements `c10::impl::DeviceGuardImplInterface`
to handle device management and synchronization.

### Device Enumeration

`torch.spyre.device_count()` is handled by the PrivateUse1 hooks in `csrc/spyre_hooks.cpp`, which look up the visible-device set from a small group of environment variables read in `csrc/spyre_device_enum.cpp`:

| Variable | Effect |
|---|---|
| `AIU_WORLD_SIZE` | Overrides the visible device count. |
| `SPYRE_DEVICES` | Comma-separated list of device indices to expose. |
| `FLEX_DEVICE` | Selects the underlying flex runtime mode (PF or VF). |

The count itself comes from `flex::getNumDevices`.

## Key C++ Components

| File | Responsibility |
|------|---------------|
| `csrc/module.cpp` | pybind11 entry point for the `_C` extension module. Device registration itself happens in `torch_spyre/__init__.py::_autoload()`. |
| `csrc/spyre_tensor_impl.cpp` | `SpyreTensorImpl`, the device tensor backing store. |
| `csrc/spyre_mem.cpp` | Device memory allocation and DMA, including graph-free DMA and FlexAllocator support. |
| `csrc/spyre_allocator.cpp` | `SpyreAllocator`, which bridges PyTorch's `c10::Allocator` to `flex::FlexAllocator`. |
| `csrc/spyre_storage_impl.cpp` | `SpyreStorageImpl`, the storage object backing `SpyreTensorImpl`. |
| `csrc/spyre_views.cpp` | Tensor view and striding support on device, including `_reshape_alias`. |
| `csrc/spyre_guard.cpp` | `SpyreGuardImpl`, device guard and synchronization. |
| `csrc/spyre_stream.cpp` | Stream management for asynchronous execution. |
| `csrc/spyre_hooks.cpp` | `PrivateUse1HooksInterface`, wires PyTorch's PrivateUse1 hooks to Spyre. |
| `csrc/spyre_device_enum.cpp` | Visible-device enumeration. Reads `AIU_WORLD_SIZE`, `SPYRE_DEVICES`, `FLEX_DEVICE`. |
| `csrc/spyre_sendnn_utils.cpp` | Eager-mode helpers, including the `EAGER_MODE` env var. |
| `csrc/logging.cpp` | C++ debug logging, gated on `TORCH_SPYRE_DEBUG`. |
| `csrc/profiler/` | PyTorch Profiler (PrivateUse1) integration. |
| `csrc/attn_utils.cpp` | SDPA dispatch. Routes `scaled_dot_product_attention` to the Spyre backend, with GQA support. |

## Python Entry Point

`torch_spyre/__init__.py` is loaded automatically by PyTorch via the
`torch.backends` entry point declared in `pyproject.toml`. This triggers
device and backend registration without requiring an explicit import.

:::{figure} ../_static/images/spyre-device-allocator.png
:alt: Spyre device allocator call chain from torch.empty through SpyreAllocator to flex::FlexAllocator::allocate
:width: 40%
:align: center

The Spyre device allocator call chain. A `torch.empty(..., device="spyre")` call flows through `spyre_empty_strided` into `SpyreAllocator::allocate`, which calls `flex::FlexAllocator::allocate(nbytes)` ([`spyre_allocator.cpp:137`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/csrc/spyre_allocator.cpp#L137)).
:::

## Memory Model

Spyre tensors live in off-chip LPDDR5. Before any kernel runs, the compiler stages the tiles it needs into a much smaller on-core LX scratchpad and the kernel reads from there. The runtime, though, only deals with the LPDDR5 side. Everything below is about how a Spyre tensor in Python turns into a real LPDDR5 allocation, and how that allocation eventually finds its way back to the pool.

:::{figure} ../_static/images/spyre-memory-hierarchy.svg
:alt: Spyre memory hierarchy showing host CPU, LPDDR5 device memory, and LX scratchpad
:width: 75%
:align: center

The two levels of memory the device sees. Full tensors stay in LPDDR5. The compiler emits load/store instructions that stage active tiles into the per-core LX scratchpad just in time for each kernel. The runtime owns the LPDDR5 allocation that backs every Spyre tensor.
:::

For the layout that lets the runtime actually walk one of those tensors, see [Tensors and Layouts](../user_guide/tensors_and_layouts.md). The next two sections cover what the C++ side of that looks like and how the lifetime ends.

### SpyreTensorImpl

A standard PyTorch `(size, stride)` pair cannot describe a tiled device tensor, so Torch-Spyre defines `SpyreTensorImpl` as a subclass of `TensorImpl`. The subclass adds one piece of data, a `SpyreTensorLayout`, that captures everything the runtime needs:

- `device_size` — the tensor's shape on device, including the extra tiling and padding dims.
- `stride_map` — the host stride for each device dim. A `-1` here means the dim is synthetic or fully padded.
- `device_dtype` — the on-device data format, for example `SEN169_FP16`.
- `dma_sizes` and `dma_strides` — a host-shape DMA descriptor used when copying views back to the host. They drive `copyAsync()` in `spyre_stream.cpp`.

Note that the handles returned to Python never carry a raw device pointer. That is a hard requirement on IBM Z.

:::{figure} ../_static/images/spyre-tensor-impl-anatomy.png
:alt: Nested boxes showing at::Tensor wrapping TensorImpl wrapping SpyreTensorImpl wrapping SpyreTensorLayout
:width: 80%
:align: center

What is behind a Spyre tensor, drawn as a stack of layers. Python only ever sees the outermost `at::Tensor` handle. Underneath, `c10::TensorImpl` carries the standard tensor metadata, and the Spyre subclass adds a `SpyreTensorLayout` that holds the device shape, the `stride_map`, the device dtype, and the DMA descriptor.
:::

### SpyreAllocator

`SpyreAllocator` (`csrc/spyre_allocator.cpp`) is a thin bridge between PyTorch's `c10::Allocator` and `flex::FlexAllocator`. Every `allocate(nbytes)` call passes straight through to `flex_alloc->allocate(nbytes)` and returns a `c10::DataPtr` with a `ReportAndDelete` callback wired in as its deleter. When the tensor's storage refcount hits zero, that deleter runs, updates the `DeviceStats` counters, and hands the allocation back to flex. The trigger is PyTorch's own refcount: Python's garbage collector is not in this loop at all.

:::{figure} ../_static/images/spyre-tensor-lifetime.png
:alt: Five-step flowchart showing how a Python tensor going out of scope frees a Spyre allocation
:width: 75%
:align: center

What happens between a Python tensor going out of scope and the device allocation returning to the flex pool. The piece that connects the two ends is the `ReportAndDelete` callback that `SpyreAllocator` installs on every `c10::DataPtr` it hands out.
:::

Physical-frame (PF) and virtual-frame (VF) execution are *not* allocator strategies inside `SpyreAllocator`. The mode is picked by the `FLEX_DEVICE` environment variable, which configures the underlying flex runtime (see `csrc/spyre_device_enum.cpp`):

| Mode | Selection | Description |
|------|-----------|-------------|
| PF (Physical Frame) | `FLEX_DEVICE` set to a PF device | Direct hardware execution path. |
| VF (Virtual Frame) | `FLEX_DEVICE` set to a VF device | Virtualized hardware, used in multi-tenant deployments. |

## Eager Operations

Eager kernels reach the Spyre dispatch key from two Python sources.

The first is manual registrations in [`torch_spyre/ops/eager.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/ops/eager.py), which use `torch.library.register_kernel` to wire up ops like `mm`, `silu`, `mish`, `fill_.Scalar`, `normal_`, `uniform_`, `_local_scalar_dense`, and `_copy_from`.

The second is CPU fallbacks in [`torch_spyre/ops/fallbacks.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/ops/fallbacks.py), registered through `@register_fallback` (or the `register_fallback_default` helper for plain pass-throughs). These cover the long tail: `arange`, `embedding`, `cumsum`, `tril`/`triu`, `isin`, `bitwise_xor`/`bitwise_or`, `argmax`, and similar.

Five Inductor decompositions registered through `register_spyre_decomposition` also dispatch eagerly: `rms_norm`, `layer_norm`, `softplus`, `linear`, and `_scaled_dot_product_fused_attention_overrideable`.

C++ kernels can still be registered through the usual `TORCH_LIBRARY_IMPL` block, but most of the public eager surface today comes from the Python sources above.

## Streams

Torch-Spyre supports stream-based asynchronous execution, following the
same API pattern as `torch.cuda` streams:

| API | Description |
|-----|-------------|
| `torch.spyre.Stream()` | Create a new Spyre stream |
| `torch.spyre.stream(s)` | Pass-through helper used inside `with` blocks; the current-stream swap is performed by `Stream.__enter__/__exit__` |
| `torch.spyre.current_stream()` | Get the current stream for the device |
| `torch.spyre.default_stream()` | Get the default stream for the device |
| `torch.spyre.synchronize()` | Wait for all operations on all streams to complete |

Streams are implemented in `torch_spyre/streams.py` (Python) and
`csrc/spyre_stream.cpp` (C++).

### Stream Pool

Each device keeps a fixed pool of streams (see `csrc/spyre_stream.cpp`). Stream `0` is the default. Streams `1` through `32` form the low-priority pool (`priority == 0`); streams `33` through `64` form the high-priority pool (any non-zero priority). Each pool holds 32 streams per device and allocates round-robin.

Note that `priority` is a binary switch: `0` selects the low-priority pool and any non-zero value selects the high-priority pool. There is no graded scale of priority levels.

## Multi-Card Support

Ensembles of up to 8 Spyre cards deliver up to 1 TB of aggregate device
memory.

:::{admonition} Planned
:class: note

Multi-card collective communications (all-reduce, all-gather, reduce-scatter) using the standard PyTorch `ProcessGroup` API are planned but not yet implemented in this repository. The C++ tree currently has no `ProcessGroup` source files.
:::

## TODO

- Document kernel launch sequence and Control Block Stream design
- Document error handling and device reset
