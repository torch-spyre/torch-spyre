---
name: find-numerical-divergence
description: "Find the first kernel where a program that is right one way and wrong another (a config flag, a commit, an environment variable) goes numerically wrong on Spyre, using the OpSpec Lab's capture, compare.py and replay, then tell a frontend bug from a backend one."
---

# Finding where a wrong run goes wrong

Use this when a program, test or model gives wrong numbers on Spyre one way and right
numbers another: a config flag on or off, a commit, an environment variable. A model can
run dozens of kernels. This finds the first one whose inputs agree between the two runs
but whose outputs do not, and checks whether that kernel is wrong on its own.

The tools are in `tests/op_specs/`: `capture.py` writes every kernel a program runs as a
standalone script, `compare.py` compares two captures, and each captured script replays
its kernel. "Comparing two runs" in `docs/source/user_guide/debugging/op_spec_lab.md`
walks through a worked example, `tests/op_specs/programs/05_cat_after_matmul.py`, with
the commands and what each prints. It takes about three minutes on a Spyre machine and
reproduces on main until the bug it shows is fixed. Run it first to see what each step
below should look like.

## Steps

1. **Make the program capturable.** `capture.py` runs a script. For a pytest test, write
   one that ends in `sys.exit(pytest.main([...]))`. Keep every compile and launch in that
   process; for vLLM, as in spyre-inference, set `VLLM_ENABLE_V1_MULTIPROCESSING=0`. A
   test plugin that ends the process with `os._exit` after the exit handlers (as
   spyre-inference's does) still gets its capture written.
2. **Capture both runs** with `--save-inputs`, into two directories. `capture.py` turns
   Inductor's caches off itself; for vLLM also point `VLLM_CACHE_ROOT` at an empty
   directory, or its compile cache can skip the compile and the capture comes back empty.
3. **Run `compare.py good bad`.** A kernel's `inputs` are the args it reads, as they were
   going in, and its `outputs` the args its launch changed; each number is the largest
   elementwise difference between the runs, relative to the largest magnitude in the good
   tensor. Small inputs with large outputs is the kernel that introduced the difference;
   large inputs means it was handed one. Exit 3 means a kernel parts, and it is named.
   Exit 4 means none does but the runs are not shown to agree, and names the first kernel
   in the way: one only one run launched (`unpaired:`, often a fusion difference), a pair
   whose element arrangements differ ("not comparable"), or a pair whose inputs already
   differ, so the difference came from outside the paired kernels (the program's own
   inputs, a copy to the device). Exit 0 means every kernel pairs and agrees.
4. **Replay the kernel it names** with the printed commands (bad script, good inputs). A
   large `max|got - observed|` means the kernel is wrong on correct inputs. A small one
   means the difference comes from outside it: another kernel, or memory planning.
5. **Diff the two scripts** (their decoded headers show layouts, iteration space and
   offsets) and their SDSC, from `python <script> --stage bundle --out-dir <dir>` on each.
   An OpSpec or SDSC that does not describe the computation is a frontend bug. If both
   describe it correctly and the result is still wrong, suspect the backend.
6. **Check a fix on the captured kernel first.** A fix below the OpSpec (SuperDSC, the
   bundle, the backend) applies to the captured script as it is: replay the bad script on
   the good inputs, and `max|got - observed|` should drop to what replaying the good
   script shows, which is 0 for pure data movement like the example's cat and a rounding
   difference for a kernel that computes. A fix above it (lowering, layout, fusion)
   changes the OpSpec itself, so capture the bad run again with the fix in place and
   compare that.
7. **Then run every suite the change can reach**, not only the obvious subset. #6265's
   first version fixed the cat and broke 58 storage-offset tests in `test_inductor_ops.py`
   that `-k "cat or slice or pad or narrow"` never selected.

## Caveats

- Only each kernel's first launch is recorded. Under vLLM that is a warm-up pass, which
  can miss a bug that depends on the data of a later one.
- A capture copies every arg to the host around each kernel's first launch, so first
  launches run one at a time. A bug that depends on timing may not show.
- The relative difference can hide one wrong small element among large ones (`--tol`,
  default `1e-2`).
- Kernels pair by name and argument shapes, not script name: the `_1` suffix follows
  compile order, which two runs need not share.

## On a model

The same steps work on a whole model. The bug in the worked example was found this way
from spyre-inference's
`tests/e2e/test_clip_embeddings.py::test_clip_image_embedding_matches_hf`: 47 kernels
under vLLM, about five minutes per capture, the cat at launch 35. #6265's description
walks through it.
