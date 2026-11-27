# Buffer transfer API

Buffer transfer APIs let safetensors loaders prepare Spyre-formatted weights in
bounded host buffers and transfer them into existing device allocations without
whole-tensor staging.

## Asynchronous pinned physical-range H2D

`copy_buffer_ranges` copies a batch of raw byte ranges from pinned host buffers
into existing Spyre device allocations on the caller-supplied stream and returns
after enqueue, without waiting or performing layout or data-format conversion.

```python
copy_buffer_ranges([(source_bytes, composite_address, offset, length)], stream=stream)
```

`source_bytes` exports nonempty contiguous one-dimensional bytes. Allocate through
the existing `torch.empty(n, dtype=torch.uint8, device="cpu", pin_memory=True)`
factory and export with `memoryview(host.numpy())`. Interior page-aligned views
are allowed. `SpyrePinnedAllocator::retain(pointer)` provides the existing
mapping, offset, capacity and owner. Both export length and backing capacity are
validated. Cache capacity/idle bytes need not equal the requested size.

`composite_address` is the existing storage-owning `get_composite_address(tensor)`
handle. Offset/length refer to physical allocation **bytes**, not logical Tensor
slices. Only a single chunk <= 4 GiB is admitted; base, offset and length must be
128-byte aligned, and the host pointer must be OS-page aligned. A subaddress is
bounded in both physical start and size before pinned DMA submission. No layout
conversion, pageable fallback or internal staging occurs.

`stream` is an existing Python `torch_spyre.Stream` or its existing `_C` stream
object. Capture on the caller and pass to the worker; no worker current-stream
lookup is used. The complete batch is validated before launch. A successful return
means submitted, **not completed**. An exception can leave earlier entries in
flight. This primitive does not drain, serialize callers, quarantine or recover.

The caller must keep source buffer exports, their pinned host storage, destination
storage-owning handles and the stream alive and unchanged until completion. Use
the existing `stream.synchronize()` and check the existing
`stream_get_error(stream._cdata)` (or the raw stream) for `SpyreStreamError.Success`
before and after waiting. Only then consume the destination, release exports or
reuse a host slot. The submitted operation also carries the existing pinned
mapping owner through the typed launch path; it does not retain Python exports
or destination handles after this function returns.

On enqueue failure the caller must still drain prior submissions while retaining
all owners. If a stream error or failed wait makes completion unknown, quarantine
those owners for the process lifetime and reject further transfers. Restart the
process; do not retry, reset, replace a library or reuse storage. fastsafetensors
implements this policy in its adapter and retains its blocking host-reader
contract, captured stream and two-slot bounded pipeline. Other callers control
their own concurrency and waits subject to the same lifetime obligations.

Use one device per process. Handles/mappings must belong to the live runtime and
process, on the stream's device; handle storage cannot be resized/rebound.
These are **caller obligations**, not generation/identity checks: the unchanged
existing handle and pinned allocator do not expose creation device, PID, runtime
generation or storage identity. Only the stream's device is checked against the
current runtime. Do not pass handles/mappings inherited across fork or left over
from a reset/reinitialization, even if the runtime pointer was reused.
Reset/teardown with live handles/mappings/operations and concurrent buffer/storage
mutation are forbidden. Keeping the existing typed handle, buffer export and
pinned owner alive protects ordinary allocation lifetime, but neither detects
storage resize/rebind nor synchronizes arbitrary caller mutations.

## Buffer-only conversion

`convert_buffers` encodes dense safetensors Tensor bytes into the standard Spyre
representation, writing into caller-provided host buffers without allocating
Tensors or device memory; callers supply only logical shape and safetensors dtype.

```python
convert_buffers([(output_bytes, input_bytes, logical_shape, dtype)])
```

`input_bytes` contains a contiguous row-major Tensor in little-endian safetensors
encoding. `logical_shape` is its ordinary shape in outermost-first order, e.g.
`[2, 33]`, not a padded or physical shape. Use `[]` for a scalar. Rank is at most
16, and extents are positive integers (not booleans) less than 2**31. Empty
Tensors are not supported. `dtype` is one of `"F16"`, `"BF16"`, `"F32"`, `"I32"`.
It describes the input's logical dtype, not an SDK data-format name.

The backend derives physical shape, padding, stick size, data formats and the
conversion walk using the existing `SpyreTensorLayout(logical_shape, dtype)`.
F16/BF16 encode as dlfloat16; F32/I32 retain their element formats and receive
the standard Spyre layout. Arbitrary destination layouts, stride descriptions,
SDK format pairs and conversion descriptors are deliberately not supported.

`output_bytes` is a writable contiguous byte buffer with enough capacity for that
layout. The existing layout object's `device_size` and `device_dtype` describe
the storage; no new allocation or sizing API is introduced. Input capacity must
cover the dense Tensor. Required input/output sizes are at most 4 GiB. Exports
may have extra capacity: only the required output range is zeroed and written,
including zero padding; trailing capacity is untouched. Input exports may alias
each other, but outputs must not overlap any input or other output in the batch.
The whole batch is validated before any output mutation, and buffer exports stay
held until the call exits, including exceptions. Conversion is synchronous and
sequential on the calling thread with the GIL released: there is no `num_threads`
argument or internal worker pool. The caller controls parallelism across calls
using disjoint outputs. If conversion throws, outputs may be partially written,
but no background conversion survives the call; later batch entries need not run.

For bounded conversion a loader may pass compact dense tiles with their logical
shape and dtype, directing each result to a preplanned output interval. The API
does not read files or plan placement of tiles in a larger Tensor; fastsafetensors
keeps that I/O and frozen checkpoint-profile knowledge internally. It does not
need to construct physical shapes or conversion walks for this call. A whole
Tensor shape with an undersized output buffer is rejected, never staged or
silently retried. No Tensor or runtime/device allocation is required. Format
special values, rounding and cross-SDK equivalence still require independent
numerical qualification.
