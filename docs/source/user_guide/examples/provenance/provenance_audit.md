# Provenance Audit: `SimpleMLP` — Metadata Across the Compilation Pipeline

> Generated: 2026-06-24 19:17 &nbsp;|&nbsp; Issue: [torch-spyre#2574](https://github.com/torch-spyre/torch-spyre/issues/2574)

Measured in-process during one cache-defeated `torch.compile` (compile-path objects only). This report is **measurement-only**; interpretation is a separate deliverable.

| Quantity | Value |
| --- | --- |
| FX pre-grad compute nodes | 3 |
| FX post-grad compute nodes | 7 |
| LoopLevelIR operations | 5 |
| OpSpec ops created | 5 |
| SuperDSC kernels | 2 |
| `sdsc_*.json` files | 5 |
| `OpSpec` declared fields | `['op', 'is_reduction', 'iteration_space', 'args', 'op_info', 'tiled_symbols']` |

## Stage × Field Matrix

✅ present & non-empty on **all** instances &nbsp; ◐ on some (n/total) &nbsp; ❌ reachable here but measured **empty/absent** &nbsp; ➖ not applicable here (no such slot, or carried indirectly via other fields).

Every column tests **population** (the field exists *and* carries non-empty content; `0` counts as content, `None`/`[]`/`{}`/`""` do not). These cells are measurements only; interpreting each absence is the separate analysis deliverable (`provenance_analysis.md`).

The **Layer** column marks whether a field lives on the FX node (`FX`) or the IR `ComputedBuffer` (`IR`). The two IR columns are the same LoopLevelIR before and after the Spyre pre-scheduling passes: **LoopLevelIR (pre-pass)** is the lowered IR entering them, **LoopLevelIR (post-pass)** is after they mutate it in place (e.g. inserting `restickify` buffers). These map to issue #2574's "Inductor passes" → "LoopLevelIR".

| Layer | Field | FX Graph (pre-grad) | FX Graph (post-grad) | LoopLevelIR (pre-pass) | LoopLevelIR (post-pass) | OpSpec | SuperDSC JSON |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FX | `stack_trace` | ✅ | ◐ 3/7 | ➖ | ➖ | ➖ | ➖ |
| FX | `nn_module_stack` | ◐ 2/3 | ◐ 2/7 | ➖ | ➖ | ➖ | ➖ |
| FX | `source_fn_stack` | ✅ | ◐ 3/7 | ➖ | ➖ | ➖ | ➖ |
| FX | `original_aten` | ➖ | ✅ | ➖ | ➖ | ➖ | ➖ |
| FX | `from_node` | ➖ | ◐ 3/7 | ➖ | ➖ | ➖ | ➖ |
| IR | `origins` | ➖ | ➖ | ✅ | ✅ | ❌ | ❌ |
| IR | `origin_node` | ➖ | ➖ | ✅ | ✅ | ❌ | ❌ |
| IR | `traceback` | ➖ | ➖ | ❌ | ❌ | ❌ | ❌ |
| IR | `get_stack_traces` | ➖ | ➖ | ◐ 3/5 | ◐ 3/5 | ❌ | ❌ |

## Stage 2 — FX Graph (pre-grad): 3 compute nodes

Cell = observed `type` of the field, or ❌ if absent.

| Node | target | `stack_trace` | `nn_module_stack` | `source_fn_stack` | `original_aten` | `from_node` | source line |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `x` | `<built-in function linear>` | `str` | `dict` | `list` | ❌ | ❌ | `x = self.fc1(x)` |
| `x_1` | `<built-in method relu of type object at 0x7f9edd8058a0>` | `str` | ❌ | `list` | ❌ | ❌ | `x = torch.relu(x)` |
| `x_2` | `<built-in function linear>` | `str` | `dict` | `list` | ❌ | ❌ | `x = self.fc2(x)` |

## Stage 2 — FX Graph (post-grad): 7 compute nodes

Cell = observed `type` of the field, or ❌ if absent.

| Node | target | `stack_trace` | `nn_module_stack` | `source_fn_stack` | `original_aten` | `from_node` | source line |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `permute` | `aten.permute.default` | `str` | `dict` | `list` | `OpOverload` | `list` | `x = self.fc1(x)` |
| `mm_default_1` | `aten.mm.default` | ❌ | ❌ | ❌ | `OpOverload` | ❌ | — |
| `add_tensor_1` | `aten.add.Tensor` | ❌ | ❌ | ❌ | `OpOverload` | ❌ | — |
| `relu` | `aten.relu.default` | `str` | ❌ | `list` | `OpOverload` | `list` | `x = torch.relu(x)` |
| `permute_1` | `aten.permute.default` | `str` | `dict` | `list` | `OpOverload` | `list` | `x = self.fc2(x)` |
| `mm_default` | `aten.mm.default` | ❌ | ❌ | ❌ | `OpOverload` | ❌ | — |
| `add_tensor` | `aten.add.Tensor` | ❌ | ❌ | ❌ | `OpOverload` | ❌ | — |

## Stage 3 — LoopLevelIR (pre-pass): 5 operations

The lowered IR entering the Spyre pre-scheduling passes.

| Op | `origins` | `origin_node` | `traceback` | `get_stack_traces` |
| --- | --- | --- | --- | --- |
| `op0` | `mm_default_1`, `permute` | `mm_default_1` | ❌ | ✅ |
| `op1` | `add_tensor_1` | `add_tensor_1` | ❌ | ❌ |
| `op2` | `relu` | `relu` | ❌ | ✅ |
| `op3` | `mm_default`, `permute_1` | `mm_default` | ❌ | ✅ |
| `op4` | `add_tensor` | `add_tensor` | ❌ | ❌ |

## Stage 4 — LoopLevelIR (post-pass): 5 operations

The same IR after the pre-scheduling passes mutate it in place.

| Op | `origins` | `origin_node` | `traceback` | `get_stack_traces` |
| --- | --- | --- | --- | --- |
| `op0` | `mm_default_1`, `permute` | `mm_default_1` | ❌ | ✅ |
| `op1` | `add_tensor_1` | `add_tensor_1` | ❌ | ❌ |
| `op2` | `relu` | `relu` | ❌ | ✅ |
| `op3` | `mm_default`, `permute_1` | `mm_default` | ❌ | ✅ |
| `op4` | `add_tensor` | `add_tensor` | ❌ | ❌ |

## Stage 5 — OpSpec: 5 ops

`OpSpec` declared fields: `['op', 'is_reduction', 'iteration_space', 'args', 'op_info', 'tiled_symbols']` — no provenance field. The `origins` below are what is *available on the input `ComputedBuffer`* at `create_op_spec`; the `OpSpec` object itself declares no field to hold them.

| Spyre op | buffer | `origins` | `origin_node` |
| --- | --- | --- | --- |
| `batchmatmul` | `op0` | `mm_default_1`, `permute` | `mm_default_1` |
| `add` | `op1` | `add_tensor_1` | `add_tensor_1` |
| `relufwd` | `op2` | `relu` | `relu` |
| `batchmatmul` | `op3` | `mm_default`, `permute_1` | `mm_default` |
| `add` | `op4` | `add_tensor` | `add_tensor` |

## Stage 6 — SuperDSC: 5 `sdsc_*.json` files (2 kernels)

Provenance field present in any emitted `sdsc_*.json`: ❌

### `sdsc_fused_addmm_linear_relu_0`

- buffers (4): `op0`, `op1`, `op2`, `op3`
- fx origins: `add_tensor_1`, `mm_default`, `mm_default_1`, `permute`, `permute_1`, `relu`
- kernel metadata: `# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.linear, aten.addmm, aten.relu]`
- `sdsc_*.json` files: 4 &nbsp; provenance in JSON: ❌

### `sdsc_fused_addmm_1`

- buffers (1): `op4`
- fx origins: `add_tensor`
- kernel metadata: `# Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]`
- `sdsc_*.json` files: 1 &nbsp; provenance in JSON: ❌
