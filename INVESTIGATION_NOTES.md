# Issue #476: Segmentation Fault During Python Interpreter Shutdown - Investigation

## Problem Summary
VF allocator tests pass successfully, but Python process crashes with **SIGSEGV (signal 11)** during interpreter shutdown/cleanup, after all test assertions pass.

## Root Cause Analysis

### The Core Issue: Global Object Destruction Race Condition

The segmentation fault stems from a **use-after-free vulnerability** in the global allocator lifecycle during Python interpreter shutdown.

#### The Call Chain:
1. `SpyreAllocator::instance()` creates a `static std::unique_ptr<SpyreAllocator>` allocator (line 915-929)
2. This allocator is registered globally: `REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &SpyreAllocator::instance())`
3. Tensors/Storage objects hold DataPtr with `ReportAndDelete` callback
4. During Python shutdown, global objects destroy in **unpredictable order**

#### The Race:
1. When Python shuts down, the global `unique_ptr<SpyreAllocator>` may be destroyed
2. **BUT** some tensor storage objects still have pending `ReportAndDelete` callbacks
3. These callbacks try to access the already-destroyed allocator's members:
   ```cpp
   static void ReportAndDelete(void* ctx_void) {
     VFSpyreAllocator* allocator = instance_ptr.load();
     // <-- allocator might be nullptr here (OK)
     // BUT if it's not nullptr, it might point to freed memory (NOT OK)
     std::lock_guard<std::mutex> lock(allocator->allocator_mutex);  // SEGFAULT!
   }
   ```

### VFSpyreAllocator-Specific Issues

The VF allocator is particularly vulnerable because:

1. **Complex State Management** (lines 536-551):
   - `std::vector<MemorySegment> segments;` - Destroyed in destructor
   - `std::unordered_map<SharedOwnerCtx*, MemorySegment*> block_to_segment;` - Contains pointers INTO segments
   - `std::mutex allocator_mutex;` - Used in ReportAndDelete callback

2. **Pointer Invalidation**:
   - `MemorySegment` contains `std::set<MemoryBlock>` and `std::unordered_map<..., MemoryBlock*>`
   - When VFSpyreAllocator's segments vector is destroyed, all MemoryBlock pointers become invalid
   - ReportAndDelete tries to use these stale pointers

3. **Destructor Implementation** (lines 851-853):
   ```cpp
   ~VFSpyreAllocator() override {
     instance_ptr.store(nullptr, std::memory_order_release);  // Just sets to nullptr
     // NO cleanup of segments, ctx_to_block, or other state!
   }
   ```
   - Destructor only clears `instance_ptr`, doesn't clean up resources properly
   - Doesn't drain pending operations or callbacks

## Files Involved

1. **`torch_spyre/csrc/spyre_mem.cpp`** (1132 lines):
   - `VFSpyreAllocator` class definition (lines 536-912)
   - ReportAndDelete callback (lines 561-582)
   - Factory method `SpyreAllocator::instance()` (lines 915-929)
   - Static registration (line 931)

2. **`torch_spyre/csrc/spyre_allocator.h`**:
   - `MemoryBlock` struct (lines 28-46)
   - `MemorySegment` struct (lines 49-66)
   - `SharedOwnerCtx` struct (in module.h)

## Proposed Fixes

### Fix 1: Prevent Access to Destroyed Allocator (CRITICAL)
**File**: `torch_spyre/csrc/spyre_mem.cpp`, ReportAndDelete (lines 561-582)

**Problem**: Accessing `allocator->allocator_mutex` and `allocator->` members after allocator destruction

**Solution**: Add robust null/validity checks:
```cpp
static void ReportAndDelete(void* ctx_void) {
  if (!ctx_void) return;
  
  auto* ctx = static_cast<SharedOwnerCtx*>(ctx_void);
  VFSpyreAllocator* allocator = instance_ptr.load(std::memory_order_acquire);
  
  // CRITICAL FIX: Don't access allocator if it's been destroyed
  if (!allocator) {
    delete ctx;  // Still cleanup the context
    return;
  }
  
  // Add a validity check (optional but safer):
  // Store an atomic<bool> to mark when destructor starts
  if (!allocator->is_valid()) {
    delete ctx;
    return;
  }
  
  // Now safe to access allocator members
  std::lock_guard<std::mutex> lock(allocator->allocator_mutex);
  // ... rest of deallocation logic
}
```

### Fix 2: Explicit Cleanup in Destructor (IMPORTANT)
**File**: `torch_spyre/csrc/spyre_mem.cpp`, ~VFSpyreAllocator (lines 851-853)

**Problem**: Destructor doesn't clean up resources properly

**Solution**: Implement proper cleanup:
```cpp
~VFSpyreAllocator() override {
  {
    std::lock_guard<std::mutex> lock(allocator_mutex);
    // Clear data structures to prevent any pending callbacks from accessing them
    segments.clear();
    block_to_segment.clear();
    fallback_sizes.clear();
  }
  instance_ptr.store(nullptr, std::memory_order_release);
}
```

### Fix 3: Add Validity Tracking (SAFER)
**File**: `torch_spyre/csrc/spyre_mem.cpp`, VFSpyreAllocator class

**Problem**: No way to know if allocator is in process of being destroyed

**Solution**: Add a validity flag:
```cpp
struct VFSpyreAllocator final : public SpyreAllocator {
 private:
  std::atomic<bool> is_valid{false};  // Add this
  // ... other members ...
  
 public:
  VFSpyreAllocator(size_t max_seg = MAX_SEGMENTS)
      : SpyreAllocator(), segments_locked(false), max_segments(max_seg) {
    fallback_sizes = {12ULL * 1024 * 1024 * 1024, 8ULL * 1024 * 1024 * 1024,
                      4ULL * 1024 * 1024 * 1024};
    instance_ptr.store(this, std::memory_order_release);
    is_valid.store(true, std::memory_order_release);  // Mark as valid
  }

  ~VFSpyreAllocator() override {
    is_valid.store(false, std::memory_order_release);  // Mark as invalid FIRST
    {
      std::lock_guard<std::mutex> lock(allocator_mutex);
      segments.clear();
      block_to_segment.clear();
    }
    instance_ptr.store(nullptr, std::memory_order_release);
  }
  
  bool is_valid_allocator() const {
    return is_valid.load(std::memory_order_acquire);
  }
};
```

### Fix 4: Add Allocator Lifetime Management (BEST)
**File**: `torch_spyre/csrc/spyre_mem.cpp`, Consider global allocator guard

**Problem**: Global unique_ptr destruction is unpredictable

**Solution**: Use a controlled shutdown mechanism:
```cpp
// Add a function to safely shutdown the allocator
void safe_shutdown_allocator() {
  VFSpyreAllocator* allocator = instance_ptr.load(std::memory_order_acquire);
  if (allocator) {
    {
      std::lock_guard<std::mutex> lock(allocator->allocator_mutex);
      allocator->segments.clear();
      allocator->block_to_segment.clear();
    }
    instance_ptr.store(nullptr, std::memory_order_release);
  }
}
```

## Implementation Priority

1. **HIGH**: Fix #1 (Prevent invalid access) - Quick fix, minimal risk
2. **HIGH**: Fix #2 (Explicit cleanup) - Ensures resources are freed properly
3. **MEDIUM**: Fix #3 (Validity tracking) - Makes code more robust
4. **MEDIUM**: Fix #4 (Lifetime management) - Better long-term solution

## Testing

After implementing fixes, verify with:
```bash
FLEX_DEVICE=VF python -m pytest tests/test_vf_allocator_standalone.py -v
FLEX_DEVICE=VF python tests/test_vf_allocator_standalone.py TestVFAllocatorStandalone.test_realistic_allocation_pattern
```

All tests should pass without segfault on exit.
