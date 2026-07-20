# SpyreStream Ownership and Synchronization

## Ownership
Three components are involved. Each has a distinct role:

### flex::RuntimeContext
`flex::RuntimeContext` (via `GlobalRuntime`) creates and owns all `flex::RuntimeStream` instances. torch-spyre accesses this through `GlobalRuntime::get()`, which returns a `const std::shared_ptr<flex::RuntimeContext>&` — callers copy this reference to take shared ownership. torch-spyre never destroys a `flex::RuntimeStream`.

The runtime context is stored in a function-local static `std::shared_ptr<flex::RuntimeContext>`. It persists for the lifetime of the process unless explicitly replaced or reset via `GlobalRuntime::set()` or `GlobalRuntime::reset()`.

torch-spyre never allocates, deallocates, or manages the lifetime of a `flex::RuntimeStream`. The flex runtime is solely responsible for stream lifecycle.

### StreamPool
`StreamPool` holds non-owning `flex::RuntimeStream*` pointers in `stream_handle_map`, keyed by `c10::StreamId`. It is responsible for lookup and round-robin assignment. It owns nothing — all pointers it stores are borrowed from the flex runtime.

### SpyreStream
`SpyreStream` is a value type wrapping a `c10::Stream`. It holds no pointer to a `flex::RuntimeStream`. It resolves the underlying handle on every operation by calling `resolveRuntimeHandle()`.

```
┌─────────────────────────────────────────────────────────────┐
│ FLEX RUNTIME                                                │
│  flex::RuntimeContext ──owns──► flex::RuntimeStream*        │
└─────────────────────────────────────────────────────────────┘
          ▲
          │ torch-spyre calls createStream()
          │
┌─────────┴───────────────────────────────────────────────────┐
│ TORCH-SPYRE                                                 │
│                                                             │
│  GlobalRuntime::get() ──returns──►                          │
│         const std::shared_ptr<flex::RuntimeContext>&        │
│         │                                                   │
│         │ calls getDefaultStream() / createStream()         │
│         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ StreamPool::stream_handle_map                        │   │
│  │   ┌────────────────────────────────────────────┐     │   │
│  │   │ mutable std::shared_mutex mutex            │     │   │
│  │   └────────────────────────────────────────────┘     │   │
│  │                                                      │   │
│  │   WRITES (unique_lock):                              │   │
│  │   • initializeStreamPoolImpl() ─────────────────┐    │   │
│  │     - Writes default stream (ID 0)              │    │   │
│  │     - Populates low/high priority ID lists      │    │   │
│  │                                                 │    │   │
│  │   • getStreamFromPool() ────────────────────────┤    │   │
│  │     - Always updates round-robin index          │    │   │
│  │     - Inserts new RuntimeStream* on ID miss     │    │   │
│  │                                                 │    │   │
│  │   READS (shared_lock):                          │    │   │
│  │   • resolveRuntimeHandle() ─────────────────────┤    │   │
│  │   • getDefaultStreamRuntimeHandle() ────────────┤    │   │
│  │   • synchronizeDevice() ────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                                                   │
│         │ used as LUT by (shared_lock for reads)            │
│         ▼                                                   │
│  SpyreStream::resolveRuntimeHandle()                        │
│         └──► pool.stream_handle_map.find(id())              │
└─────────────────────────────────────────────────────────────┘
```

## Synchronization

> **Note:** "Synchronization" in this section refers to `StreamPool`'s internal locking and thread-safety model, not stream execution ordering or completion synchronization. Event-based stream synchronization is not yet implemented.

`StreamPool` uses `std::shared_mutex` declared as mutable on the struct:

```cpp
struct StreamPool {
  mutable std::shared_mutex mutex;
  ...
};
```

The locking discipline is:
| Caller                           | Lock type           | Why                                                    |
|----------------------------------|---------------------|--------------------------------------------------------|
| `resolveRuntimeHandle()`         | `std::shared_lock`  | Read-only lookup, concurrent reads safe                |
| `getDefaultStreamRuntimeHandle()`| `std::shared_lock`  | Read-only lookup                                       |
| `synchronizeDevice()`            | `std::shared_lock`  | Collects handles, releases before synchronizing        |
| `initializeStreamPoolImpl()`     | `std::unique_lock`  | Writes default stream entry and ID lists               |
| `getStreamFromPool()`            | `std::unique_lock`  | Always writes round-robin index; may insert new handle |

The `shared_lock` guards handle resolution only — it protects reads and writes to `stream_handle_map`, not submission of work to the flex runtime. Multiple threads resolving handles for **different** streams may do so concurrently. However, concurrent submission of operations to the **same** `flex::RuntimeStream` must be serialized by the caller; the lock provides no protection there.

Writes occur in two scenarios:
1. **Pool initialization** (`initializeStreamPoolImpl()`): Initializes per-device stream pool state: default stream mapping, priority stream ID lists, and round-robin indices. Executed once per device.
2. **Stream assignment** (`getStreamFromPool()`): Always acquires `unique_lock` and always writes the round-robin index (`next_low_priority_idx` or `next_high_priority_idx`) on every call. Additionally, if the selected `c10::StreamId` has no flex handle yet, it inserts a new `flex::RuntimeStream*` into `stream_handle_map` (at most once per stream ID).

`synchronizeDevice()` deliberately releases the lock before calling `handle->synchronize()` — the handles snapshot is taken under `shared_lock`, then the lock is dropped before doing any blocking work on the flex runtime. This avoids holding the read lock across a potentially long hardware sync.

Handle resolution calls `TORCH_CHECK` and throw if the requested stream ID has no entry in `stream_handle_map` (both `resolveRuntimeHandle()` and `getDefaultStreamRuntimeHandle()` follow this pattern). `query()` and `synchronize()` propagate any error returned by the flex runtime without additional wrapping.

## Invariants

**Single device per process.** torch-spyre follows a one-process-one-device model. `startRuntime()` calls `flex::initializeRuntime` exactly once per process (via `std::call_once`) and binds `GlobalRuntime` to a single logical device, selected from `tls_idx`, `LOCAL_RANK`, or defaulting to `0`. Multi-device workloads are handled by `torchrun`, which spawns one process per device. This is why `stream_handle_map` is keyed by `c10::StreamId` alone with no device dimension — there is only ever one device's worth of stream IDs in the map.

`stream_handle_map` is append-only. Once a `c10::StreamId` is mapped to a `flex::RuntimeStream*`, that entry is never mutated or removed. This is what makes concurrent `shared_lock` reads safe — readers never observe a partial update or a removed entry.

Stream IDs are never reassigned. A given `c10::StreamId` always maps to the same `flex::RuntimeStream*` for the lifetime of the process. This invariant holds because the flat `StreamId` namespace is safe under the single-device-per-process guarantee above.

`StreamPool` never deletes a `flex::RuntimeStream*`. Lifetime is managed entirely by the flex runtime via `GlobalRuntime`.

## Runtime Shutdown

`freeRuntime()` calls `GlobalRuntime::reset()`, which drops torch-spyre's `shared_ptr<flex::RuntimeContext>`. Flex manages the context's actual lifetime independently, so stream handles are not immediately invalidated by this call. However, `StreamPool` is not cleared and `device_init_flags` are not reset, and `startRuntime()` cannot be called again because its `std::once_flag` is already spent. **Runtime reset after stream initialization is not supported.**
