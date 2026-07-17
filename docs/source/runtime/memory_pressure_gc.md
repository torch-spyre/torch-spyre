# Memory Pressure and Python GC

The `SpyreAllocator` can register a callback with `flex::FlexAllocator` that is
invoked whenever the allocator exhausts its memory regions. Torch-Spyre uses
this hook to trigger Python garbage collection, freeing tensors whose Python
references have been dropped but whose C++ storage has not yet been released.

This page describes the Python-specific design: the GIL interaction, the
lock-ordering contract, the per-call-site rules, and the free-threaded Python
considerations. The underlying C++ callback contract (mutex release/re-acquire
around any blocking work) is documented in
[`flex/docs/lock_ordering.md`](https://github.com/torch-spyre/flex/blob/main/docs/lock_ordering.md).

## Why GC Can Free Device Memory

A Spyre tensor's device allocation is freed by the `ReportAndDelete` deleter
installed on the `c10::DataPtr` inside `SpyreAllocator::allocate()`. That
deleter runs when the tensor's storage refcount reaches zero. PyTorch's refcount
tracks C++ `shared_ptr` ownership, not Python references, so a tensor that is
still reachable only through a Python cyclic reference will not be freed until
Python's cyclic GC runs `PyGC_Collect()`.

Calling `PyGC_Collect()` under memory pressure breaks those cycles, drops the
Python reference, and lets the `shared_ptr` refcount fall to zero — releasing
the underlying device allocation back to flex.

## The Two Locks

| Lock | Owner | Purpose |
|------|-------|---------|
| Python GIL | CPython | Serialises Python thread execution and C-API calls |
| `FlexAllocator::allocator_mutex_` | flex | Protects allocator internal state |

## Lock Ordering Rule

```text
allocator_mutex_ → GIL  (NEVER  GIL → allocator_mutex_)
```

The mutex must be **released** before the GIL is acquired. Both locks must
never be held simultaneously. See the [Execution Paths](#execution-paths)
section for how this plays out in practice.

## Execution Paths

### Path 1 — Python thread allocation (bridge path)

**Entry**: Python code calls `torch.randn(..., device='spyre')`.

| Step | State |
|------|-------|
| Python thread runs; GIL is already held | GIL ✓ |
| Crosses the pybind11 boundary into C++ | GIL ✓ |
| `SpyreAllocator::allocate()` → `FlexAllocator::allocate()` acquires `allocator_mutex_` | GIL ✓, mutex ✓ |
| Normal allocation succeeds; mutex released | — |

**Lock order**: `GIL → allocator_mutex_`

**Why it is safe**: The GIL is *already held on entry* — it is never acquired
while the mutex is held. If memory pressure fires, the callback releases the
mutex before touching the GIL (Path 2 below).

### Path 2 — Memory pressure callback

**Entry**: `FlexAllocator::allocateInDomain()` exhausts all regions.

| Step | State |
|------|-------|
| `allocateInDomain()` holds `allocator_mutex_` | mutex ✓ |
| Callback `memory_pressure_callback_(lock)` is invoked | mutex ✓ |
| `lock.unlock()` — mutex released | — |
| `PyGILState_Ensure()` — GIL acquired | GIL ✓ |
| `PyGC_Collect()` runs | GIL ✓ |
| `PyGILState_Release()` — GIL released | — |
| `lock.lock()` — mutex re-acquired | mutex ✓ |
| Returns to `allocateInDomain()`; allocation retried | mutex ✓ |

**Lock order**: `allocator_mutex_ → (release) → GIL → (release) → allocator_mutex_`

**Why it is safe**: The mutex is *not held* when the GIL is acquired.

### Path 3 — C++ direct allocation (off-GIL path)

**Entry**: C++ code calls `FlexAllocator::allocate()` without holding the GIL.

Examples: `TimestampCalibrator` (setup path), compiled Inductor kernels,
sendnn operations, other internal C++ callers.

The lock sequence is identical to Path 2 when memory pressure fires:

1. `FlexAllocator::allocate()` acquires `allocator_mutex_`.
2. Callback releases `allocator_mutex_`.
3. Callback acquires GIL via `PyGILState_Ensure()`.
4. `PyGC_Collect()` runs.
5. GIL released; `allocator_mutex_` re-acquired.

**Safety**: same as Path 2 — mutex is not held while acquiring the GIL.

## Deadlock Scenarios

### What prevents deadlock

**Scenario 1 — Python thread vs memory pressure**

- T1 (Python): holds GIL, acquires `allocator_mutex_` → allocation succeeds.
- T2 (allocator): holds `allocator_mutex_`, hits pressure → **releases mutex**,
  then acquires GIL.
- No deadlock: T2 releases the mutex before it needs the GIL.

**Scenario 2 — Two Python threads**

- T1: holds GIL, acquires mutex.
- T2: blocked on GIL; cannot even attempt to acquire mutex.
- No deadlock: the GIL serialises Python threads before they reach the allocator.

**Scenario 3 — C++ thread vs Python thread**

- T1 (C++): acquires mutex, hits pressure → releases mutex, acquires GIL.
- T2 (Python): holds GIL, waits for mutex.
- No deadlock: T1 releases the mutex before acquiring the GIL, unblocking T2.

### What causes deadlock

```cpp
// WRONG — will deadlock!
PyGILState_STATE gstate = PyGILState_Ensure();        // Acquire GIL first
std::lock_guard<std::mutex> lock(allocator_mutex_);   // Then try for mutex
// If another thread holds mutex and is waiting for the GIL → deadlock.
```

```cpp
// CORRECT
// (allocator_mutex_ is already held via unique_lock& lock passed to the callback)
lock.unlock();                                  // Release mutex
PyGILState_STATE gstate = PyGILState_Ensure(); // Acquire GIL
PyGC_Collect();
PyGILState_Release(gstate);                    // Release GIL
lock.lock();                                   // Re-acquire mutex
```

## Implementation in SpyreAllocator

The callback is implemented in
[`torch_spyre/csrc/spyre_allocator.cpp`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/csrc/spyre_allocator.cpp).
It is registered with `FlexAllocator` during `SpyreAllocator` construction and
follows the contract above exactly.

## Callback Requirements Checklist

Any memory-pressure callback that acquires the GIL must:

1. **Accept lock by reference**: `void callback(std::unique_lock<std::mutex>& lock)`
2. **Release mutex** before acquiring the GIL:

   ```cpp
   lock.unlock();
   ```

3. **Acquire GIL**:

   ```cpp
   PyGILState_STATE gstate = PyGILState_Ensure();
   ```

4. **Run GC**:

   ```cpp
   PyGC_Collect();
   ```

5. **Release GIL**:

   ```cpp
   PyGILState_Release(gstate);
   ```

6. **Re-acquire mutex** before returning:

   ```cpp
   lock.lock();
   ```

7. **Exception safety**: the `unique_lock` destructor handles an already-unlocked
   mutex correctly if the callback throws — no double-unlock or missed re-lock.

## Call-site Rules

| Call-site context | Action required |
|---|---|
| From Python (GIL already held) | None — GIL is already held on entry; callback will release the mutex before re-acquiring the GIL. |
| From C++ without GIL | None — callback acquires and releases the GIL internally. |
| From C++ with GIL explicitly held | Ensure `allocator_mutex_` is not held when acquiring the GIL elsewhere in the same frame. |

## Free-Threaded Python (≥ 3.13 `--disable-gil`, 3.14+)

On free-threaded Python builds `PyGILState_Ensure()` is a no-op and the GIL no
longer exists as a single global lock. The specific deadlock this page describes
— one thread holding `allocator_mutex_` while blocked on the GIL, another
holding the GIL while blocked on `allocator_mutex_` — cannot occur on these
builds.

The mutex-release contract around `PyGC_Collect()` remains correct and useful:
free-threaded GC uses a stop-the-world pause that suspends all threads, so
releasing `allocator_mutex_` before calling `PyGC_Collect()` avoids holding the
mutex across that pause unnecessarily.

The implementation (`lock.unlock()` → `PyGC_Collect()` → `lock.lock()`) is safe
and well-behaved on both GIL and no-GIL builds.

## Concurrency Design Choice

The implementation uses the **hold-mutex-during-callback** approach:

- The callback is invoked while `allocator_mutex_` is held by the caller frame.
- The callback itself releases and re-acquires the mutex around GIL operations.
- This serialises all memory-pressure events automatically.
- It is simpler than the alternative (release mutex, use a condition variable)
  and is appropriate for the low-concurrency workload: one device per process,
  allocations during `JobPlan` construction.

## Testing

| Test | Location | What it covers |
|------|----------|----------------|
| C++ unit tests | `flex/flex/tests/allocator/memory_pressure_callback_test.cpp` | Callback invocation, retry logic, registration/unregistration |
| Python integration tests | `torch-spyre/tests/test_memory_pressure_gc.py` | GIL safety with concurrent threads, memory pressure triggering GC, no-deadlock scenarios |
| TSan | `TSAN_OPTIONS="detect_deadlocks=1" ./run_tests` | Data races and lock-ordering violations |

## References

- **C++ callback contract**: `flex/docs/lock_ordering.md`
- **SpyreAllocator memory model**: [Runtime — Memory Model](index.md#memory-model)
- **Implementation**: [`torch_spyre/csrc/spyre_allocator.cpp`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/csrc/spyre_allocator.cpp)
