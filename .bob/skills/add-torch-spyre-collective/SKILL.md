---
name: add-torch-spyre-collective
description: Use when the user wants to add, enable, or wire up a new collective (e.g. reduce, reduce_scatter, gather, all_to_all) in torch-spyre for torch.compile/Inductor, or add a new c10d_functional lowering, SpyreCollectiveKernel subclass, or spyre::*_plan/*_run op. Covers IR node, lowering, C++ plan/run ops, fake kernels, graph passes, and tests. Assumes spyre-comms already exposes the plan/apply API; otherwise use the expose-spyre-comms-collective-api skill first.
---

# Add a Collective to torch-spyre

Wires one new collective through torch-spyre's compiled path. Existing collectives are the
templates; copy their shape rather than inventing a new one:

| Template | Style | Use when the new op... |
|---|---|---|
| `AllReduceKernel` | in-place (`create_inplace`, `NoneLayout`) | mutates and returns its input |
| `AllGatherKernel` / `BroadcastKernel` | out-of-place (`create_out_of_place`) | produces a new output buffer |

Scope: only touch `torch-spyre/`. Do not edit `spyre-comms/` here; if an API is missing, stop
and switch to the `expose-spyre-comms-collective-api` skill.

## Step 0 — Gather facts

Use `ask_followup_question` only for what you cannot determine from the code:

1. Which `_c10d_functional` op is being lowered (check `torch.ops._c10d_functional.<op>`)?
   Note its schema: non-tensor args, in order. They become `constant_args`.
2. In-place or out-of-place? Does output shape differ from input (as in all_gather)?
3. Does spyre-comms already expose the three calls on `spyre_comms::Context`? Grep
   `spyre-comms/include/spyre_comms.hpp` for `<op>(const TensorInfo&`, and `<op>_applyTensor`/`_applyTensors`.
   All three (eager, plan, apply) are needed. If missing, stop and use the other skill.

Then read, before editing, the template closest to the new op:
- `torch_spyre/_inductor/ir.py` — `SpyreCollectiveKernel` and the subclasses
- `torch_spyre/csrc/distributed/spyre_distributed.cpp` — `*_plan_impl` / `*_run_impl`
- `torch_spyre/_inductor/lowering.py` — "Direct c10d Lowerings" section

## Step 1 — IR node (`torch_spyre/_inductor/ir.py`)

Subclass `SpyreCollectiveKernel`. Set a unique `plan_prefix` (used in the plan variable name) and implement:
- `_plan_call(num_elems, dtype_code)`: returns the `torch.ops.spyre.<op>_plan(...)` expression.
  Unpack `self.constant_args` in the same order as the `_c10d_functional` schema.
  List args (e.g. `all_to_all_single`'s split sizes) are **flattened** into `constant_args`, so
  `all_to_all_single(x, [4, 4], [4, 4], "g")` gives `(4, 4, 4, 4, "g")`. Print `constant_args` from a
  unit test before unpacking; a tuple-unpack that assumes the schema shape fails at compile time.
- `_run_call(input_name, plan_var)`: returns the `torch.ops.spyre.<op>_run(...)` expression.

The base `codegen()` already handles emit-once plan lines and in-place vs out-of-place binding. Do
not override it unless the op needs more than one tensor operand.

## Step 2 — Lowering (`torch_spyre/_inductor/lowering.py`)

Add a `@register_spyre_lowering(_c10d.<op>.default)` function in the "Direct c10d Lowerings"
section. Import the new kernel at the top of the file.
- Out-of-place: `return ir.TensorBox.create(<Kernel>.create_out_of_place(_c10d.<op>.default, tensor, *args))`
- In-place: call `<Kernel>.create_inplace(_c10d.<op>_.default, tensor, *args)` and `return tensor`.
  If the functional (non-underscore) variant exists, also lower it on `clone(tensor)` as
  `lower_c10d_all_reduce` does, and lower the in-place `_` variant separately.

No lowering is needed for `wait_tensor`: `WaitKernel` is generic.

## Step 3 — C++ plan/run ops (`torch_spyre/csrc/distributed/spyre_distributed.cpp`)

Mirror the closest existing op:
1. Add a value to `enum class CollectiveKind` and `enum class PlanKind`.
2. If the op needs a parameter not in `CachedPlan` (`rank_param`, `reduce_op`, `group_size`),
   reuse a field where its meaning fits, or add a field. Update `cache_lookup()` to compare it,
   and update every `CachedPlan{...}` initializer (they are positional).
3. In `ensure_wsi()` add the `case PlanKind::<Op>` that calls the spyre-comms **plan** overload
   (`context->op(*plan.tensor_info, ...)`).
4. Add `spyre_<op>_plan_impl(...)`: `ensure_context()`, validate args with `TORCH_CHECK`,
   `cache_lookup` under `wsi_cache_mutex_`, otherwise push a `CachedPlan` and call `ensure_wsi`.
5. Add `spyre_<op>_run_impl(...)`: validate the tensor (`is_privateuseone`, `is_contiguous`,
   non-zero `nbytes`), build a `spyre_comms::Tensor` from `*plan.tensor_info` with
   `SetSpyreDeviceAddressBorrowed(get_composite_address(t))`, call `context-><op>_applyTensor(s)`,
   `work_schedule->start()`, build `make_tensor_keep_alive(...)` for every borrowed tensor, and
   register a `PendingWork` in `pending_work_map_` keyed by the output's `SharedOwnerCtx*`.
6. In `spyre_wait_work_impl`, add post-processing only if the output is not already in the tensor
   (all_gather assembles from per-rank buffers; in-place and broadcast need nothing). Update the
   "No pending async work" error message to name the new `*_run` op.
7. Register in `TORCH_LIBRARY(spyre, m)`: `m.def("<op>_plan(...) -> int", &spyre::spyre_<op>_plan_impl)`
   and `m.def("<op>_run(...) -> Tensor")`; and `m.impl("<op>_run", ...)` under `TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m)`.
   Use `Tensor(a!)` / `-> Tensor(a)` for in-place runs, as `allreduce_run` does.

Keep the reduce-op parser (`parse_reduce_op`) in mind: only `sum` is accepted today.

## Step 4 — Fake kernel (`torch_spyre/_inductor/distributed/spyre_library.py`)

Inside the `if torch._C._dispatch_has_kernel("spyre::broadcast_run"):` block, add
`@torch.library.register_fake("spyre::<op>_run")` with the same signature as the schema:
return `x` for in-place, `torch.empty_strided(...)` for same-shape out-of-place, or
`torch.empty(<new shape>)` for shape-changing ops. Plan ops need no fake kernel.

## Step 5 — Graph passes (usually no change)

Passes already treat any `_CollectiveKernel` generically. Check each; edit only if it applies:
- `propagate_layouts.py` (the `_CollectiveKernel` branch): out-of-place kernels fall into the
  generic `SpyreCollectiveKernel` branch (`generic_layout`). Add an explicit branch (like
  `BroadcastKernel`, which copies the input's layouts) only when output layout must follow the input.
- `work_division.py` and `scratchpad/lx_context_switching.py`: collectives are already skipped.
- `scratchpad/graph_editor.py` `_REWRITABLE_COLLECTIVES`: add the kernel only if it is in-place,
  reads exactly `inputs[0]`, and has a test showing a loop carry feeding it. Otherwise leave it.
- `scratchpad/allocator.py` and `hbm_pool_planning.py` have collective-specific comments/logic
  (HBM operands, in-place results). Grep them for `all_reduce` and confirm the new op's operand/result
  placement is covered.

## Step 6 — Tests

Add, modelled on the existing allreduce files:
1. `tests/inductor/test_collective_kernels.py`: a unit test that builds the node via the factory and
   asserts the emitted plan/run lines. Runs without spyre-comms or hardware.
2. `tests/distributed/test_<op>.py`: multi-rank pytest (uses `torchrun`; see `test_allreduce.py`).
   Add a compiled variant `test_<op>_compiled.py` if the op is expected under `torch.compile`.
3. `tests/configs/distributed_tests/test_<op>.yaml`: copy `test_allreduce.yaml`, change the `path`.
   Register the new yaml in `.github/workflows/_test_matrix.yaml`, next to the existing
   `config: distributed_tests/test_allreduce.yaml` entries (around line 1212).

Then, if `spyre-comms` mirrors torch-spyre distributed tests, tell the user to run the
`mirror-test-coverage` skill in `spyre-comms` for the new `test_<op>.py`.

## Step 7 — Optional follow-ups (mention, do not do unprompted)

- Eager backend: `csrc/distributed/spyre_ccl.cpp` / `spyre_ccl.hpp` (`SpyreCCLBackend::<op>`).
- Profiler names: `csrc/profiler/AiuptiActivityHandlers.cpp` maps `AIUPTI_RUNTIME_TRACE_CBID_COLL_*`
  to display names. The cbid itself lives in `libaiupti`; add a mapping only if the cbid exists.
- Docs: `docs/source/user_guide/` and `docs/source/user_guide/examples/distributed/`.

## Validation

Run what exists for the files touched; report anything you could not run:
- `python -m pytest tests/inductor/test_collective_kernels.py` (no hardware needed)
- The repo's lint/format config (check `pyproject.toml` / `.pre-commit-config.yaml`) on changed files
- A C++ build of the extension if the environment has one; multi-rank tests need AIU hardware and
  `torchrun`, so state clearly when they were not run.
