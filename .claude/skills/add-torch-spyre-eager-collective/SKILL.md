---
name: add-torch-spyre-eager-collective
description: "Use when the user wants an eager (non-compiled) torch.distributed collective to work on Spyre tensors with the spyreccl backend, e.g. dist.reduce_scatter_tensor, dist.all_to_all_single, dist.all_gather_into_tensor, dist.scatter, or when an eager call fails with SpyreCCLNotSupportedException or \"does not support\". Covers SpyreCCLBackend methods in csrc/distributed/spyre_ccl.cpp and their multi-rank tests. Not for torch.compile lowerings."
---

# Add an Eager Collective to torch-spyre

An eager `torch.distributed` call on a Spyre tensor ends in one method of `SpyreCCLBackend`
(`torch_spyre/csrc/distributed/spyre_ccl.cpp` / `.hpp`), a `c10d::Backend` subclass. Enabling
an eager collective means implementing that method on top of the spyre-comms **eager** call
(`group_context_-><op>(...)`, the overload taking `Tensor&`, not `TensorInfo`). No Inductor code
is involved. Uncompiled `torch.ops._c10d_functional.*` calls also run through the process group,
so they reach the same method.

For the compiled path (`torch.compile`), use the `add-torch-spyre-collective` skill instead.
The two are independent: doing one does not enable the other.

Paths are relative to the torch-spyre repo root. spyre-comms is checked out alongside it at
`../spyre-comms`. Only touch `torch-spyre/`. If the spyre-comms eager call is missing, stop and
tell the user it must be added to spyre-comms first.

## Step 0 — Find the backend method the user's call reaches

Python API to backend method, for torch 2.14 (verify against the installed
`torch/distributed/distributed_c10d.py`: grep `group\.<method>(` in the API's body):

| Python API (`dist.*`) | Backend method it calls | Paired alias (see Step 1) | spyre-comms eager call |
|---|---|---|---|
| `broadcast` | `broadcast` | | `broadcast(Tensor&, root)` |
| `all_reduce` | `allreduce` | | `allreduce(Tensor&, op)` |
| `all_reduce_coalesced`, coalescing manager | `allreduce_coalesced` | | none; loop or ask |
| `all_gather` (list) | `allgather` | | `allgather(vector<Tensor>&, Tensor&)` |
| `all_gather_into_tensor`, `all_gather_single` | `all_gather_single` | `_allgather_base` | `allgather(Tensor& input, Tensor& output)` |
| `all_gather_into_tensor` in coalescing manager | `all_gather_single_coalesced` | `allgather_into_tensor_coalesced` | none; loop or ask |
| `reduce` | `reduce` | | `reduce(Tensor&, op, root)` |
| `reduce_scatter` (list) | `reduce_scatter` | | `reduce_scatter(Tensor& input, Tensor& output, op)` on a packed input (see Step 4) |
| `reduce_scatter_tensor`, `reduce_scatter_single` | `reduce_scatter_single` | `_reduce_scatter_base` | `reduce_scatter(Tensor& input, Tensor& output, op)` |
| `gather` (list) | `gather` | | `gather(vector<Tensor>&, Tensor&, root)` |
| `gather_single`, `gather_into_tensor` | `gather_single` | `gather_into_tensor` | `gather(Tensor&, Tensor&, root)` |
| `scatter` | `scatter` | | none in the header today; stop and ask |
| `all_to_all_single` | `all_to_all_single` | `alltoall_base` | `alltoallw(out, in, out_splits, in_splits)` |
| `all_to_all` (list) | `alltoall` | | none; build on `all_to_all_single` (Step 4) |
| `barrier` | `barrier` | | `barrier()` |
| `send`/`isend`, `recv`/`irecv` | `send`, `recv` | | `send`, `recv` |

The user's list of collectives may name them by the functional or c10d API; map each to a row.
Then check the current state:

```bash
grep -n -E 'SpyreCCLBackend::[a-z_]+\(' torch_spyre/csrc/distributed/spyre_ccl.cpp
grep -n -B3 'SpyreCCLNotSupportedException(getBackendName' torch_spyre/csrc/distributed/spyre_ccl.cpp
grep -n 'virtual std::unique_ptr<WorkSchedule> ' ../spyre-comms/include/spyre_comms.hpp
```

A method that throws `SpyreCCLNotSupportedException` is a stub to replace. A method that is
absent falls back to the `c10d::Backend` default (throws "does not support", or forwards to its
alias). Read the spyre-comms doc comment for the eager call: its argument order and in-place
semantics often differ from c10d (the single-buffer `allgather` and `reduce_scatter` take input
first).

Ask the user only about what the code cannot answer: which reduce ops and dtypes they need
beyond what spyre-comms supports, and whether a missing spyre-comms call should block the work.

## Step 1 — Paired aliases: implement one, never stub the other

In torch 2.14, each single-tensor collective has two virtual methods that forward to each other
(`all_gather_single` ↔ `_allgather_base`, `reduce_scatter_single` ↔ `_reduce_scatter_base`,
`all_to_all_single` ↔ `alltoall_base`, `gather_single` ↔ `gather_into_tensor`,
`all_gather_single_coalesced` ↔ `allgather_into_tensor_coalesced`,
`reduce_scatter_single_coalesced` ↔ `reduce_scatter_tensor_coalesced`). See
`torch/include/torch/csrc/distributed/c10d/Backend.hpp`.

- Implement the **new name** (the left one); `dist.*` calls it.
- Delete any override of the alias that throws. If the alias throws, the new name's default
  forwards to it and the call still fails. `_allgather_base` throws like this today, so
  `dist.all_gather_into_tensor` fails even though the list `allgather` works.
- If you override neither, the default guard raises at runtime.

## Step 2 — Device layout: host chunks are not device slabs

This step decides whether the op returns correct data. Do it before writing code.

spyre-comms works on the **flat device buffer**. A single-buffer call (single-tensor
`allgather`, `reduce_scatter`, `alltoallw`, single-buffer `gather`, scatter) gives rank r the
r-th contiguous slab of that buffer. Under the default stick layout, host dim 0 is not the
outermost device dim (`(4, 128)` fp16 is `device_size=[2, 4, 64]`), so slab r is **not**
`tensor.chunk(g)[r]`. spyre-comms raises no error: it moves the wrong elements. Per-rank slabs
must also be whole 128-byte sticks; spyre-comms has no algorithm for partial sticks and can
abort the process instead of throwing.

The compiled path has solved this for each op. Reuse its approach and its checks:

| Op | Compiled solution to copy | Where |
|---|---|---|
| reduce_scatter | stage input `(g, A, B)` as `(A, g, B)` with `.contiguous()`, so `g` is the outermost device dim; output is a default-layout `(A, B)` | `_reduce_scatter` in `_inductor/lowering.py`; checks in `spyre_reducescatter_run_impl` (`spyre_distributed.cpp`) |
| all_to_all_single | accept only layouts whose dim-0 rows are contiguous on device; splits become device elements (`rows * row_device_elems`), each a multiple of 128 bytes | `_dim0_row_device_elems` and `AllToAllKernel._plan_call` in `_inductor/ir.py`; units in `spyre_alltoall_plan_impl` |
| all_gather_into_tensor | gather into per-rank buffers, then `output.narrow(0, r * chunk, chunk).copy_(buf[r])` | `CollectiveKind::AllGather` in `spyre_wait_work_impl` (`spyre_distributed.cpp`) |
| list forms (`allgather`, `gather`) | one device tensor per rank; no slab issue | existing eager methods |

Units: `prepare_tensor` / `getTensorInfo` describe every tensor to spyre-comms as a **flat 1-D
buffer of device elements**, stick padding included. So where a spyre-comms doc comment says
"entries of the outermost dimension", it means device elements here. That is why the compiled
all_to_all converts row splits to `rows * row_device_elems`.

Read every `TORCH_CHECK` and `TODO` in the compiled `spyre_<op>_plan_impl` / `_run_impl` and in
the kernel's `_plan_call` before writing checks. They record spyre-comms limits that crash
instead of throwing (for example zero-size all_to_all splits segfault; see
`to_alltoall_splits`). The eager method needs the same checks.

Share the code, do not copy it. The C++ helpers in `spyre_distributed.cpp` are `static`, and the
layout checks for some ops exist only in Python (`_dim0_row_device_elems` in `_inductor/ir.py`).
Move a C++ helper you need into `spyre_collective_utils.h` and call it from both files. Port a
Python check into that header too, and leave a comment in both places that they must stay in
sync.

In eager C++ you have real tensors, so check layouts with `spyre::get_spyre_tensor_layout(t)`,
compare against `SpyreTensorLayout(sizes, dtype)` and `device_size`, and check device bytes with
`spyre::get_device_size_in_bytes`. Reject anything the chosen approach does not cover with a
`TORCH_CHECK` that names the layout. Do not pass a tensor straight to a single-buffer call
because its host shape divides evenly.

Staging (`permute(...).contiguous()`, `copy_` into a temporary) runs Spyre eager kernels from
inside the backend. Check them standalone on the target shapes first. A temporary that must be
copied back after completion only works for `asyncOp == false` today (`SpyreCCLWork::wait()`
has no post-completion hook). Either run synchronously in that case and say so in a comment,
or add the hook.

## Step 3 — Declare in `spyre_ccl.hpp`

Add (or keep) the `override` declaration in `SpyreCCLBackend`, with the exact signature and
default opts from `Backend.hpp` (`const <X>Options& opts = <X>Options()`). Remove the
declaration of any alias you deleted in Step 1. If the op keeps a staged or temporary tensor
alive, add a field to `SpyreCCLWork` next to `input_tensor_` and update its comment.

## Step 4 — Implement in `spyre_ccl.cpp`

Copy the closest existing method; they all have the same shape:

```cpp
c10::intrusive_ptr<Work> SpyreCCLBackend::reduce(
    std::vector<at::Tensor>& tensors, const ReduceOptions& opts) {
  check_vector_tensor(tensors, 1, 1);           // count + contiguous + dense
  // op-specific TORCH_CHECKs: reduceOp == SUM, fp16/bf16 for reductions,
  // list sizes == group_context_->getSize(), split sizes, shapes
  spyre_comms::Tensor tensor;
  prepare_tensor(tensors[0], &tensor);          // borrows the device address
  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::REDUCE);
  work->work_schedule_ = group_context_->reduce(
      tensor, convert_reduce_op_type(opts.reduceOp), opts.rootRank);
  work->work_schedule_->start();
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}
```

| Template | Use when the new op... |
|---|---|
| `allreduce`, `reduce` | reduces; copy the SUM and fp16/bf16 checks verbatim |
| `broadcast` | is in-place on one tensor |
| `gather` | has a per-rank output list, valid only on the root |
| `allgather` | has a per-rank output list on every rank, or needs the padded-input handling |
| `send`, `recv` | has no opts struct (no `asyncOp`; always returns without waiting) |

Rules:
- Validate every tensor with `check_single_tensor` / `check_vector_tensor`, and every list size
  against `group_context_->getSize()`. Error messages start with `"[" + getBackendName() + "]: "`.
- Reductions: `convert_reduce_op_type` maps only `SUM`; reject other ops with `TORCH_CHECK`, as
  `allreduce` does, rather than passing `UNSUPPORTED` to spyre-comms.
- Convert argument types explicitly where spyre-comms differs (for example `int64_t` split sizes
  to `std::vector<int>`). `dist.all_to_all_single` passes `None` splits as empty vectors, which
  mean an even split: build them from `size(0) / group size` after checking divisibility.
- List-form ops over a single-buffer spyre-comms call (list `reduce_scatter`, list `all_to_all`):
  build them on the single-tensor version (pack the inputs, then apply Step 2 to the packed
  tensor). Implement the single-tensor API first.
- Keep any tensor that spyre-comms reads after `start()` alive: store it in the work
  (`work->input_tensor_`, as `allgather` does) when you create a temporary (clone, view).
- Use the matching `OpType` (`OpType::REDUCE_SCATTER`, `OpType::ALLTOALL_BASE`, `OpType::SCATTER`,
  ...). Honour `opts.asyncOp` exactly as above.
- Ops there is no plan to support keep `throw SpyreCCLNotSupportedException(getBackendName(),
  __func__);` with a `// Do not intend to support: <reason>` comment.

## Step 5 — Tests (`tests/distributed/`)

1. `tests/distributed/test_<op>.py`: first `ls tests/distributed/`. Some names are taken by tests
   of the compiled path's ops (`test_all_to_all_ops.py` tests `spyre::alltoall_plan/run`); pick
   a name that does not collide and say in its docstring that it tests the eager `dist.*` API.
   Copy `test_reduce.py` (or `test_gather.py`). Keep the
   license header, the module-level `RANK`/`WORLD_SIZE` skips, and `setUpClass`
   (`init_process_group("cpu:gloo,spyre:spyreccl")`). Cover:
   - each user-facing API from the Step 0 row (for example both `reduce_scatter_tensor` and the
     list `reduce_scatter` if both were enabled), `async_op=False` and `True` (call
     `work.wait()` before reading the result);
   - non-root ranks for rooted ops, 1-D and 2-D shapes, fp16 and bf16 (`_native_bfloat16_tensor`,
     `_collective_rank_input`, `_collective_expected` from `utils.py`);
   - shapes that break a naive slab split: a 2-D last dim above one stick (`(g * 4, 128)` fp16)
     and a padded last dim (`(g * 3, 100)`). If the Step 2 approach supports them, they are
     correctness tests: compare every rank's result with the CPU reference (for reduce_scatter
     `sum_r(input_r).chunk(g)[rank]`). If it rejects them, they are `assertRaisesRegex` tests.
     A layout bug passes on `(g * 64,)`, so never test only that;
   - the uncompiled functional op (`torch.ops._c10d_functional.<op>` + `wait_tensor`), which
     reaches the same method;
   - each rejection you added (`assertRaisesRegex` on the message).
   Use values that are exact in fp16 (the existing tests assert `<= 1024`), compare on the host.
2. `tests/configs/distributed_tests/test_<op>.yaml`: copy `test_reduce.yaml`, change the `path`.
3. Register it in `.github/workflows/_test_matrix.yaml` next to the other
   `config: distributed_tests/test_*.yaml` multi-Spyre entries.
4. Run `make check-all-configs` from the repo root.
5. Add the file to the list in `tests/distributed/README.md`.

If spyre-comms mirrors torch-spyre distributed tests, tell the user the new test should be
mirrored there as well.

## Validation

Report what you ran and what you could not:
- Build the C++ extension (a compile error in `spyre_ccl.cpp` breaks the whole backend).
- `pre-commit run --files <changed files>` (clang-format, cpplint).
- `cd tests && torchrun --nproc-per-node 2 -m pytest distributed/test_<op>.py -v`. This needs
  at least 2 AIUs; say so clearly when it was not run.

## Common mistakes

| Mistake | Result |
|---|---|
| Implementing `_allgather_base`-style alias only, or leaving it throwing | `dist.*_into_tensor` / `*_single` still fails |
| Passing a Spyre tensor straight to a single-buffer spyre-comms call | rank r gets the wrong elements, no error |
| Testing only 1-D or one-stick shapes | layout bugs pass the tests |
| Passing c10d `(output, input)` order to a spyre-comms call that takes `(input, output)` | wrong data or size error |
| Passing a non-SUM op through `convert_reduce_op_type` | `UNSUPPORTED` reaches spyre-comms |
| Forgetting `if (!opts.asyncOp) work->wait();` | sync callers read stale data |
| Test not added to a yaml and `_test_matrix.yaml` | test never runs in CI |
| Editing `lowering.py` / `ir.py` | that is the compiled path; not needed here |
