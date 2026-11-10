# PyTorch 2.15 compatibility audit

This audit covers the Python monkey-patches in `torch_spyre`, including the
Inductor patches brought in by merging upstream main at `2c2cda371f` into
the 2.15 upgrade. Upstream comparisons use PyTorch `v2.14.1` and
`v2.15.0` (`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`). Runtime checks use
Python 3.12 and torch 2.15.0+cpu.

## Required changes

- Move the counted-loop scheduler release gate to 2.15. Upstream removed
  `GraphLowering.additional_star_deps`; loop dependencies must be forwarded
  through `additional_buffer_deps`, which `Scheduler.compute_dependencies`
  still reads. The ordering regression now exercises that table.
- Use Dynamo's new keyword-only `supports_input_input_aliasing` policy in
  the scan patch. The upgrade branch already contains this adaptation.
  Preserve the upstream policy for other higher-order operators, including
  FlexAttention, and retain rejection of aliased inputs requiring gradients.

## Scan and for_each_tile

The public scan implementation, `ScanOp.__call__`, `trace_scan`,
`generic_scan`, `scan_functionalize`, and Dynamo's
`ScanHigherOrderVariable._call_function` are unchanged between these tags.
`for_each_tile` continues to pass a flattened body, carry leaves, scanned
operand leaves, and lifted invariant operands to `scan_op`.

The Dynamo alias checker changed its signature and now independently controls
input-input aliasing. This replaces the need to override tracer alias detection.
The functionalization checker has not gained the equivalent policy: its
`_check_alias_and_mutation` wrapper remains necessary. It permits read-only
input aliases for scan while rejecting input-output aliases, output-output
aliases, mutations, and failed alias analysis. Other operators delegate to
their original checkers. Tests cover shared K/V storage, aliased carry inputs,
gradient restrictions, nested conditional operators, and FlexAttention's native
input-alias policy.

`decompose_scan_to_while_loop` now returns `(list(outputs),)` from its
replacement callback, including the zero-trip branch. This is upstream's FX
replacement structure; Spyre does not replace that callback or unwrap it.
The scan output convention consumed by `for_each_tile` is unchanged.
`for_each_tile` rejects zero tiles before invoking scan. Its direct `make_fx`
path still checks carry structure and metadata, mutation, output aliases,
captured traced tensors, and nested bodies.

`resolve_shape_to_proxy` remains unchanged and still misses expression keys
such as `FloorDiv(symbol, tile_size)`. The expression-key workaround for
[pytorch#203212](https://github.com/pytorch/pytorch/issues/203212) remains
necessary. Its tests verify the upstream failure, the patched result, signature
guarding, and restoration after exceptions.

Scan cache opt-in remains necessary. The existing salt and body/constant hashes
still distinguish graphs. CPU tests check cache hits, changed bodies, changed
captured constants, and cache bypass when disabled. The scan implementation is
unchanged, so this upgrade does not require a new scan salt.

## Remaining patch inventory

| Location / upstream surface | 2.15 assessment |
| --- | --- |
| `loop_scheduler.py`: four replaced scheduler methods and dependent IR interfaces | The surface comparison reports 27 unchanged, two changed, none missing. `_init` adds fusion-memory state while preserving the front-half ordering. `compute_dependencies` removes the obsolete table; fixed above. All four replaced methods are unchanged. |
| `patches.py`: dtype computation map | Same upstream map and lookup contract. Keep Spyre's native-width policy scoped to the compile context. |
| `patches.py`: `GraphLowering.placeholder` | Unchanged; upstream still initially gives input layouts offset zero. Preserve Spyre input offsets before views are lowered. |
| `patches.py`: `GraphLowering.register_operation` | Unchanged; naming by list length still needs protection after graph edits. |
| `patches.py`: `Loops.has_large_inner_fn`, joint-graph patterns | Signatures and selected softmax patterns unchanged; retain realization and fusion controls. |
| `patches.py`: `GraphLowering._update_scheduler`, `SchedulerNode.has_side_effects` | Unchanged. Wrappers retain upstream behavior and protect loop copy-out/mutation operations from elimination. |
| `patches.py`: addmm fusion and `GraphTransformObserver.apply_graph_pass` | Fusion registration is still found; observer signature is unchanged and the wrapper delegates to upstream. |
| `lowering.py`: lowering/fallback registries | Registry contracts remain valid; overlays save and restore original entries, including nested use. |
| `wrapper.py`: `SizeVarAllocator._simplify_loops_impl` | Unchanged. Per-graph override remains needed because upstream does not see Spyre's physical layout. |
| `ir.py`: `is_wait` in utils/comms/scheduler | Unchanged exact-type check still excludes Spyre's subclass. Keep the wrapper and by-name bindings. |
| `pass_counters.py`: `ComputedBuffer.get_read_writes` | Unchanged; wrapper forwards arguments and restores the original. |
| `torch_phases.py`: Chromium event start/end | Positional argument order is unchanged, including start metadata at position three. Required-phase tests pass. |
| `_inductor/__init__.py`: `compile_fx`, `compile_fx_inner` | New compile-option handling stays in upstream, which the wrapper calls. The picklable decomposition callable and thread-local nested-region context remain necessary. |
| `_monkey_patch.py`: Tensor.to/repr/layout and torch.empty | Existing forwarding contracts remain valid. Device layout extensions remain Spyre-specific; factory cache is warmed before replacement. |
| `_monkey_patch.py`: `GuardBuilder.TENSOR_MATCH` and reuse metadata | New current-device and dimension guards are preserved by calling upstream first. `GuardCheckSpec` and tensor metadata contracts are unchanged. |
| `_monkey_patch.py`: nested-region decomposition extraction | Upstream extractor is unchanged; inherited Spyre decompositions still need threading into nested bodies. |
| `_monkey_patch.py`: `FxGraphHashDetails.__init__` | New upstream custom-backend hashes are preserved by calling the original initializer before adding layout, offset, and sharding fields. |
| `_monkey_patch.py`: safetensors entry points | Independent of the changed torch internals; CPU passthrough and loading-hook tests validate the installed safetensors version. |
| `model_utils.py`: `nn.Module.to` | Upstream `Module.to` and `_apply` are unchanged. Keep Spyre loading/layout handling and upstream delegation for other devices. |
| `__init__.py`: lazy profiler initializer | 2.15 adds activity configuration and renames CUDA profiler internals. The wrapper forwards all arguments to the original initializer and does not copy those internals. |
| `__init__.py`: allocator initialization query | The zero-argument query remains present; wrapper only handles the existing non-DeviceAllocator error and propagates others. |
| Device/distributed registration and eager nested-copy guard | Registration APIs remain available. `_use_eager_on_nested_compile` and its predicate are unchanged. These are integration interfaces rather than replaced implementations. |

## Validation and limits

The torch-spyre native extension rebuilt against 2.15. The focused CPU audit
passed 162 tests and 89 subtests across loop scheduling, scan shape proxies,
scan aliasing, nested tile tracing, recursion limits, scan caches, pass counters,
Dynamo guards, and compile timing. All 56 safetensors hook/passthrough tests
also passed with safetensors 0.8.0. The 17 tile-lowering tests and 21 subtests
passed, checking generated while-loop counts, materializations, nested loops,
carry values, pytree outputs, and attention numerics on CPU. These tests used
`--noconftest -o addopts=`
and `IS_INDUCTOR_SPAWNED_SUBPROCESS=1` to avoid the device-poisoning fixture
and hardware stream discovery; they validate CPU/compiler behavior only.

The normal test path failed during device initialization with
`RAS::PCI::PCIeFailure`, before executing the tests. Device building blocks,
loop-scheduler equivalence, fused-QKV attention, offset/layout guards, model
loading, distributed waits, profiler collection, and frontend work-bound
calibration therefore require the all-platform CI run. No expectations or
tolerances were weakened to accommodate that hardware failure.

Downstream images also need torchvision 0.29.2, whose stable torch ABI supports
torch >=2.14. There is no torchvision 0.30 release. CPU NMS and image resize
were validated against torch 2.15; Power/Z source builds remain CI coverage.
