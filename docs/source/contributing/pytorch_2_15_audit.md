# PyTorch 2.15 compatibility audit

This audit covers the Python monkey-patches in `torch_spyre`, including the
Inductor patches brought in by merging upstream main at `48a7e872eb` into
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
- Preserve a sliced tile's explicit extent with non-overlapping `unfold`.
  In 2.15, `unflatten` re-derives a literal tile width as
  `S // (S // tile_width)`, even after narrowing to whole tiles. This breaks
  scan output-shape resolution and later integer-only loop geometry.
- Check the bare `mark_dynamic` call instead of expecting an empty
  `_dynamo_dynamic_range` entry. 2.15 no longer records a range when neither
  bound is supplied; Spyre's separate reservation contract is unchanged.

## Scan and for_each_tile

The public scan implementation, `ScanOp.__call__`, `trace_scan`,
`generic_scan`, `scan_functionalize`, and Dynamo's
`ScanHigherOrderVariable._call_function` are unchanged between these tags.
`for_each_tile` continues to pass a flattened body, carry leaves, scanned
operand leaves, and lifted invariant operands to `scan_op`.

The sliced operand view now supplies its tile width and step explicitly.
The existing divisibility check guarantees the view covers the input exactly;
the view retains noncontiguous strides and storage offsets. This avoids
depending on `unflatten`'s symbolic shape simplification. Relative indexing
also preserves the runtime offset of an enclosing scan; a raw `as_strided`
view is insufficient for nested CPU-compiled scans. The symbolic-loop,
dynamic-reservation and cross-kernel reservation tests check numerical results
at every declared size, output metadata, rejection of undeclared dimensions,
and reuse of a single graph and cached kernel.

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
| `loop_scheduler.py`: `refresh_group_node_dependencies` | Unchanged in 2.15. The new wrapper preserves the loop's unpruned fake dependencies when upstream rebuilds the group from its children, and restores the original function on exit. |
| `patches.py`: stride-first contiguity | The copied upstream function is unchanged in 2.15; upstream still introduces the size-one guard before checking strides. Keep the workaround for pytorch#203781. |
| `_monkey_patch.py`: `FakeTensorMode.avoid_device_init` | The property is unchanged in 2.15; the upstream PrivateUse1 opt-out is still absent. Keep the device-initialization workaround for pytorch#203190. |
| `_inductor/__init__.py`, guards and FX hashes: declared dimensions | GraphArg example/fake-tensor access retains its contract. Preserve reservation metadata in cache keys, reuse guards and compiled outputs. The unflatten change above is required to keep local tile geometry concrete. |
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
| `_monkey_patch.py`: `mark_dynamic` for reserved tensors | Bare marking still makes the dimension dynamic, but no longer creates an empty range record. The test now checks the real call and dynamic marking. |
| `_monkey_patch.py`: `GuardBuilder.TENSOR_MATCH` and reuse metadata | New current-device and dimension guards are preserved by calling upstream first. `GuardCheckSpec` and tensor metadata contracts are unchanged. |
| `_monkey_patch.py`: nested-region decomposition extraction | Upstream extractor is unchanged; inherited Spyre decompositions still need threading into nested bodies. |
| `_monkey_patch.py`: `FxGraphHashDetails.__init__` | New upstream custom-backend hashes are preserved by calling the original initializer before adding layout, offset, and sharding fields. |
| `_monkey_patch.py`: safetensors entry points | Independent of the changed torch internals; CPU passthrough and loading-hook tests validate the installed safetensors version. |
| `model_utils.py`: `nn.Module.to` | Upstream `Module.to` and `_apply` are unchanged. Keep Spyre loading/layout handling and upstream delegation for other devices. |
| `__init__.py`: lazy profiler initializer | 2.15 adds activity configuration and renames CUDA profiler internals. The wrapper forwards all arguments to the original initializer and does not copy those internals. |
| `__init__.py`: allocator initialization query | The zero-argument query remains present; wrapper only handles the existing non-DeviceAllocator error and propagates others. |
| Device/distributed registration and eager nested-copy guard | Registration APIs remain available. `_use_eager_on_nested_compile` and its predicate are unchanged. These are integration interfaces rather than replaced implementations. |

## Validation and limits

The November 18 refresh uses the current sentient Deeptools, runtime/Flex,
libaiupti and spyre-comms artifacts, and the installed external senlib runtime.
The native extension rebuilt against torch 2.15.0+cpu with profiler support.
The refresh incorporates upstream's AIUPTI `device_index`, eager `masked_fill`,
fp8 zero-fill, and partial-view fill-guard fixes.

The focused compiler/device suite passed 280 tests and 156 subtests, including
scan aliasing, shape proxies, nested tile lowering, frontend work bounds,
cost-model checks, fused-QKV views, and the scalar rejection cases. After the
final `unfold` change, 38 tests and 23 subtests passed for symbolic tile shapes,
noncontiguous/offset views, nested matmul tilings, dynamic reservations,
symbolic loops, cross-kernel reservations, and bare dynamic marking. The
reservation tests check numerical results across declared sizes and reuse of
one graph and its cached kernel. Cache tests used a library-version file
containing hashes of the installed compiler and Flex library.

The stable local-window slice-fill limitation is now asserted explicitly for
eager and compiled execution, with the corresponding exception and message;
it remains tracked by torch-spyre#4502. No numerical tolerances were loosened.

Full pre-commit checks and generated requirements consistency passed. OOT
configuration validation found no duplicate, missing, or unlabelled tests;
182 existing dead-pattern warnings remain. The scheduler surface comparison
reports 29 names: 27 unchanged, two reviewed changes, and none missing.
The earlier CPU audit also passed all 56 safetensors hook/passthrough tests.

Local validation covers one x86_64 Spyre device. Power/Z source builds,
multi-device distributed paths, and the full model matrix require the
all-platform CI run.

Downstream images also need torchvision 0.29.2, whose stable torch ABI supports
torch >=2.14. There is no torchvision 0.30 release. CPU NMS and image resize
were validated against torch 2.15; Power/Z source builds remain CI coverage.
