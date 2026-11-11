# Sharded Tensors

Sharding gives a Spyre tensor *one storage that is physically split
across the accelerator's cores*, while the tensor itself keeps its global
logical shape. A sharded tensor is created with
`torch.spyre.to_sharded(x, Shard(dim))`, copied back with
`.to("cpu")` / `.cpu()`, and consumed by compiled kernels that address
each core's own
shard directly. Everything that cannot honor that storage model fails
closed — it raises instead of silently reading part of the tensor.

This page describes the design for contributors: the user-facing contract,
the module boundaries, the runtime data structures, the compiler changes,
and the invariants the tests pin. It covers the whole vertical slice; the
per-file detail lives in the source, linked throughout.

## Scope

The Spyre 1p5 memory architecture gives each core its own addressable
allocation. Sharded tensors are the smallest faithful implementation of that model:

* **Equal split.** The sharded dimension divides into exactly `num_shards`
  (8 or 16) equal, non-empty shards.
* **Identity slot-to-core map.** Shard *i* lives in placement slot *i*, and
  slot *i* backs core *i*. There is no placement freedom yet; the extension
  points for a non-identity map exist but are pinned to identity (see
  [Roadmap extension points](#roadmap-extension-points)).
* **Pointwise ops compiled end-to-end.** Any same-shape pointwise op runs
  as a sharded kernel (`add` is the one the test suite pins); chaining
  works across compiled calls, but a multi-op graph with a sharded
  intermediate is rejected. Everything else that would touch sharded
  storage is rejected at compile time with a message saying why.
* **Host copies only.** Sharded tensors copy to and from the CPU
  (whole-tensor, no dtype change). Device-to-device copies and fills are
  rejected because they take a single device address.

Beyond the op set, the compiled path admits only a narrow shape regime;
everything outside it fails at compile time with a specific message:

* Every sharded operand must share one placement and have the op's exact
  shape — no views, slices, reshapes, broadcasts or reindexing (views of
  sharded storage are additionally rejected at construction; see
  [Runtime](#runtime-storage-and-addressing-c)).
* Shard-local strides must equal the global ones (DeepTools strides each
  core's tiles from the global shape): the shard boundary must fall on
  whole sticks for a last-dim shard, exactly one (the shard) dim may be
  divided, and for a shard on a non-last dim rows may not span multiple
  sticks — a 32×128 fp16 `Shard(0)` tensor fails because the second stick
  of every row would go unwritten (`_check_local_strides` in
  `sharded_hbm.py`).
* Static shapes only (`dynamic=False`), one dtype across operands and
  output, and `BUNDLE_SYMBOLIC_ARGS=1` (each core's start address is a
  bundle `input_arg`).
* Each sharded input must be a base tensor (`check_sharded_input` in
  `sharded_hbm.py`) whose kernel-argument layout is the shard-local
  `SpyreTensorLayout`, and the committed device layout must be the default
  global one — a restickified or otherwise relaid tensor cannot use chunk
  addressing (`_non_unified_shard_bytes` in `spyre_kernel.py`).
* Sharded intermediates and in-place writes into sharded storage are
  rejected — only graph inputs and outputs may be sharded.

`num_shards` must equal the active core count (`SENCORES`), so the storage
and the kernels that consume it agree on how many cores participate.

## User-facing API

```python
import torch
from torch.distributed.tensor import Shard

y = torch.spyre.to_sharded(x, Shard(0))                # CPU -> sharded spyre
z = y + y                                              # compiled sharded add
x_back = y.cpu()                                       # gather every shard

torch.spyre.is_sharded(y)                              # True
info = torch.spyre.sharding_info(y)
info.placement, info.num_shards, info.receipt          # Shard(0), 8, entries
```

The API reuses the Tensor Parallel *vocabulary* — the `Shard(dim)` placement
type and its single-element list idiom — but deliberately avoids the
`torch.distributed.tensor` entry-point names: `distribute_tensor` returns a
DTensor there and a plain `torch.Tensor` here, and sharing the name would
invite that confusion. `to_sharded` instead follows the `to_sparse` /
`to_mkldnn` family — a plain tensor with a different storage layout — and the
docstrings state the divergence at the point of use: no DTensor, no mesh, no
collectives, no autograd, equal shards only, with every wrong expectation
failing loudly rather than silently.

### `to_sharded`

```python
torch.spyre.to_sharded(tensor, placement, *, num_shards=None)
    -> torch.Tensor
```

Copies a CPU tensor into a sharded Spyre tensor. Arguments:

* **`tensor`** — the source, a CPU `torch.Tensor` or `torch.nn.Parameter`
  (gradient-free): contiguous, base-like, compact-storage, non-scalar, of a
  supported dtype (`bool`, `uint8`, `int32`, `float16`, `bfloat16`,
  `float32`). Its shape and dtype become the global shape and dtype of the
  result.
* **`placement`** — a `torch.distributed.tensor.Shard(dim)` naming the
  dimension to split, or the single-element list `[Shard(dim)]`. Exactly one
  placement is admitted (there is no multi-dimensional mesh); `dim` may be
  negative. The named dimension must divide into `num_shards` equal,
  non-empty shards: its size must be a multiple of `num_shards`. The shard
  count is never reduced to fit; an unshardable dimension is rejected before
  any allocation, and the error lists the dimensions that would divide.
  For example, `Shard(0)` on a `[3, 32, 64]` tensor with 8 shards is rejected
  (3 elements cannot split into 8 shards) and suggests dims 1 and 2.
* **`num_shards`** — keyword-only. Must be 8 or 16 and must equal the active
  core count (`SENCORES`), so storage and the kernels that consume it agree
  on how many cores participate. Omitted, it defaults to the active core
  count; since the default `SENCORES` is 32, a default environment must set
  `SENCORES` to 8 or 16 first.

Returns a plain `torch.Tensor` on the `spyre` device carrying the global
shape; its storage holds `num_shards` placement slots with shard *i*
(the *i*-th equal slice along `placement`'s dimension) in slot *i*, bound
to core *i*.

The whole request is validated before the runtime is touched: the placement
must be exactly `torch.distributed.tensor.Shard` (or a single-element list
of one), the shard count must be 8 or 16 and match `SENCORES`, and the
source must be a contiguous, base-like, gradient-free CPU tensor of a
supported dtype with compact storage. Degenerate shapes and out-of-range
placement dims are rejected by the planner before allocation. If validation
passes but allocation or any shard copy fails, no tensor is returned — the
result is dropped and freed; nothing falls back to unified storage.

### `is_sharded`

```python
torch.spyre.is_sharded(tensor) -> bool
```

Returns whether `tensor` is a sharded Spyre tensor — `False` for any other
value, including non-Spyre tensors; never raises.

### `sharding_info`

```python
torch.spyre.sharding_info(tensor) -> ShardedTensorInfo
```

Returns the read-only sharding record of a sharded Spyre tensor: the
`placement` (`Shard(dim)`), `num_shards`, `shard_device_bytes`, and
`receipt`. The receipt's entries carry `(shard_ordinal, core_id,
byte_length)` plus an opaque physical mapping for inspection; the receipt is
*generation-neutral*: it asserts identity, core binding, and equal shard
extents, and never exposes backend-specific placement as part of the
tensor's semantic identity. Raises `TypeError` for anything that is not a
sharded Spyre tensor.

The public functions live in
[_sharded_api.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_sharded_api.py)
and are exported onto the `torch.spyre` namespace during device autoload.

## Module boundaries

The Python side is three modules with a strict dependency direction:

```
_sharded_api.py  ──►  _native_placement.py  ──►  sharding.py
     (public API)         (generation-aware          (neutral semantics)
                           adapter)
```

* [sharding.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/sharding.py)
  is **decision- and generation-independent**. It defines the
  `SpyreShardingSpec` (placement + shard count), the equal-shard planner
  (`plan_equal_shards`, backed by `compute_shard_range` and PyTorch's
  `Shard.local_shard_size_and_offset` helper), the `ShardRange` value type,
  and the neutral receipt contract (`ShardedReceipt`,
  `ShardPlacementEntry`, `validate_sharded_receipt`). Nothing in it knows
  how the backend physically places a shard.
* [_native_placement.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_native_placement.py)
  is the **single generation-aware adapter**. It is the only module that
  knows 1p0-emulation facts: each logical chunk is placed into one of the
  six dispatchable Tensor segments (1–6), logical chunk identity is
  distinct from physical segment identity, and extents must be 128-byte
  aligned. `sharded_receipt_from_native` validates the native per-chunk
  records, proves the per-segment extents disjoint (the native allocator
  does not), and builds the neutral receipt. Moving this adapter into Flex
  is planned; when that happens, this file is deleted and the
  import in `_sharded_api` is retargeted.
* [_sharded_api.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_sharded_api.py)
  is the **public surface**. It validates a transfer request, plans it,
  initializes the runtime, allocates, validates the receipt of the real
  allocation, and copies. The extension (`torch_spyre._C`) is imported
  lazily inside the functions, so importing the module has no side
  effects — pinned by a subprocess test.

The split is load-bearing: `_native_placement` imports the neutral types
from `sharding`, and the API module consumes both, so the API cannot fold
into `sharding` without creating a cycle.

## Runtime: storage and addressing (C++)

The sharded storage lives in the Spyre allocator,
[spyre_allocator.cpp](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/csrc/spyre_allocator.cpp):

* `allocateShardedStorage(shard_dim, num_shards, shard_device_bytes)`
  allocates **one owning plain allocation per shard** and returns a `DataPtr`
  whose context is a `SharedOwnerCtx`.
* `SharedOwnerCtx` carries the per-shard allocations in `shard_addrs`
  (indexed by shard ordinal: shard *i* backs core *i*) and a hand-built,
  **non-owning** `composite_addr` spanning all shards' chunks, so the
  whole-storage view reports a total size without ever freeing anything in
  its destructor. A `sharding` record (`ShardedStorageInfo`: shard dim,
  shard count, per-shard bytes, the global shape the storage was allocated
  for) marks the storage as sharded; a `shard_view`
  back-pointer lets a view's `DataPtr` deleter free its enclosing
  `ShardViewCtx`. Each shard's allocation is placed round-robin: shard *i*
  is bound to Flex memory domain `i % num_memory_domains()`, spreading the
  shards across the device's domains.

On top of that,
[spyre_mem.cpp](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/csrc/spyre_mem.cpp)
implements:

* `spyre_empty_sharded` — builds the tensor with the global shape, the
  shard-local `SpyreTensorLayout`, and contiguous global strides.
* `copy_sharded` — host↔device copy, one shard at a time through
  `spyre_shard_view`, a single-chunk view of one shard's allocation that
  keeps the parent storage alive. Copies are blocking by contract.
* `shard_device_address` — the launch-time resolver: given a sharded
  tensor and a shard index, returns the device address of that shard's
  allocation. It cross-checks the tensor's actual shard count and
  per-shard bytes against what the kernel was compiled for, so a stale
  kernel or a differently-sharded tensor fails at launch, not silently.
* Fail-closed guards: `set_`, `resize_`, and fills reject sharded storage
  (they would reinterpret or partially write it); the view constructors and
  in-place `as_strided_` reject any view of sharded storage that does not
  reproduce the base tensor exactly — same sizes, strides, and storage
  offset — because a view's shape no longer matches the shard mapping
  (`check_sharded_view` in
  [spyre_views.cpp](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/csrc/spyre_views.cpp);
  identity views such as `detach` are allowed); and `copy_sharded` checks
  the tensor's sizes against the recorded `global_shape` before mapping
  elements onto shards;
  [spyre_composite_address.cpp](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/csrc/spyre_composite_address.cpp)
  rejects sharded storage anywhere a single `CompositeAddress` is
  consumed (kernels, fills, device copies).

Addresses are **resolved per launch and never baked into compiled
artifacts** — a compiled kernel names its sharded arguments symbolically
and the runner resolves them against the actual tensor at call time.

The pybind surface (`_spyre_empty_sharded`, `_spyre_is_sharded`,
`_spyre_sharding_info`, `num_memory_domains`) is declared in
[module.cpp](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/csrc/module.cpp)
and mirrored in
[_C.pyi](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_C.pyi).

## Compiler: planning and codegen

The Inductor side is led by
[sharded_hbm.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/sharded_hbm.py),
whose `plan_sharded_hbm` admits a graph for sharded compilation or raises
`Unsupported`. Its checks, in order:

1. Only same-shape pointwise ops are admitted (the op-kind check is
   generic — not limited to `add`); mixed dtypes, dynamic shapes, and
   views of sharded inputs are rejected with specific messages.
2. Every sharded input must agree with the others on placement (shard dim,
   count) — an operand-vs-operand check; each operand's own storage was
   already validated against its receipt when the tensor was created.
3. Only pinned ops may read sharded storage, including
   non-`ComputedBuffer` consumers (fallback kernels), which a separate
   loop catches.

Downstream, the sharding plan pins work division — a sharded op's core
split is fixed by its shard plan (the scratchpad allocator's "shard plan"
branch), never renegotiated by the work-division passes — and codegen
emits **one SDSC (kernel invocation) with the non-unified
`nonUnifiedAllocInHBM_` form and one `input_arg` per (sharded tensor,
core) pair** — every core executes the same kernel on its own shard:

* [bundle.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/codegen/bundle.py)
  introduces the `kShardAddress` symbol kind: a sharded kernel argument
  is emitted as N symbolic shard-address args (one per core), deduplicated
  by `(arg_index, core)`. Ordinary kernel args and dimension symbols are
  deduplicated in the same pass.
* [compute_ops.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/codegen/compute_ops.py)
  builds the non-unified kernel form for sharded consumption and verifies
  (via `verify_non_unified_split` in `sharded_hbm.py`) that each core's
  slice is exactly one shard of every sharded argument.
* [kernel_runner.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/execution/kernel_runner.py)
  resolves each shard-address symbol at launch with
  `shard_device_address`, passing the compiled `num_shards` down as a
  cross-check.
* [_monkey_patch.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_monkey_patch.py)
  folds sharding into the compiled-artifact cache key, so sharded and
  unsharded tensors of the same shape never share a kernel.

## Invariants the tests pin

The sharding suite lives in `tests/sharding/`; every file there exercises
production code (there are no self-referential reference models). The
load-bearing contracts, and where they are pinned:

* **Validation precedes runtime.** Every rejection in `to_sharded`
  happens before the runtime is initialized
  (`tests/sharding/test_to_sharded_api.py`, via a fixture that fails the
  test if the runtime is touched).
* **Importing the API has no side effects** — no extension load, no
  `torch.distributed`, no device registration (subprocess test in the
  same file).
* **The planner is fail-closed against a misbehaving helper.**
  `compute_shard_range` re-proves the geometry PyTorch's
  `Shard.local_shard_size_and_offset` returns: negative, out-of-range, or
  non-covering geometry is rejected
  (`tests/sharding/test_shard_planning.py` monkeypatches the helper).
* **The adapter proves disjointness.** Per-segment extent overlap is
  rejected in `sharded_receipt_from_native` — the only place it is
  checked (`tests/sharding/test_segment_emulated_receipt.py`).
* **The neutral receipt contract holds for every backend**
  (`tests/sharding/test_sharding_spec.py`).
* **The compiled path is correct and isolated**: one shard per core,
  each core writes only its own shard, sharded and unsharded inputs never
  share a kernel, and unsupported ops/shardings fail at compile with
  stable messages (`tests/sharding/test_sharded_add.py`).

`test_sharded_add.py` needs a device or the emulation shape (`SENCORES`
in (8, 16) and a Flex device mode that admits sharded allocations —
`FLEX_DEVICE=EMUL1p5` — at process start);
`test_device_suite_in_emulation_shape` re-runs the file in each shape via
a child pytest. The mode opens the card through the VF PCI path, so the
wrapper probes it in a throwaway process first and skips with
the underlying error on a host that cannot back it. Each test file must be
listed in a config under `tests/configs/torch_spyre_tests/sharding/` (see
`tests/scripts/check_test_coverage.sh`).

When adding or changing behavior, follow the repo test rules in
`AGENTS.md`: a deliberate rejection is an assertion (`pytest.raises` with
a message fragment), not an xfail; a bug's xfail cites an open issue and
is removed by the PR that fixes it.

## Roadmap extension points

The current implementation keeps several extension points deliberately
pinned to identity or unused, for the multi-domain memory compiler
milestone 1:

* `ShardPlacementEntry.physical` — the opaque per-entry physical mapping
  (today: domain, region, segment, offset, filled from the live region
  map at allocation time; shards spread round-robin across domains).
  Non-identity *placement policies* would land here.
* `SpyreShardingSpec.bindings()` / `ShardChunkBinding` — binds each
  semantic shard to a placement slot and core. Today the map is identity;
  a non-identity slot-to-core map lands here without changing the public
  API.
* `validate_sharded_receipt` — the neutral invariants every backend must
  uphold for a receipt; the adapter's own checks are stricter today, so
  this is the contract a future backend is held to.
* Moving the generation-aware adapter into Flex — the end-state that
  deletes the adapter file.
