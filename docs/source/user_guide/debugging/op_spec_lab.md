# The OpSpec Lab

**Scope:** attributing a wrong number to one side of the frontend/backend
boundary, and learning to read an OpSpec while you are there.

An OpSpec list is the last IR the Torch-Spyre frontend produces before
SDSC/MLIR codegen. It is the contract with the DeepTools back-end: everything
upstream of it is lowering, fusion, stickification and layout; everything
downstream executes what it says. That makes it the one place where a wrong
result can be *split in two* — but only if you can run one in isolation, which
is what the tools under
[`tests/op_specs/`](https://github.com/torch-spyre/torch-spyre/tree/main/tests/op_specs)
are for.

| Tool | Job |
|------|-----|
| `capture.py` | Point it at a PyTorch script; get one self-contained, runnable Python file per compiled kernel |
| `runner.py` | The run loop that takes an OpSpec list to a device launch with no `torch.compile` involved; `capture.py` inlines it into every file it emits |
| `explain.py` | Render a resolved OpSpec as a readable summary instead of a wall of symbolic expressions |

If you would rather learn these by using them, skip to the
[four worked examples](#walkthrough-four-worked-examples) and come back to the
field reference when a line in one of them does not make sense.

This picks up where [Step 4 of the debugging
workflow](index.md#step-4--examine-compiler-artifacts) leaves off: once the
artifacts show what the compiler produced, replaying one OpSpec tells you which
side of the boundary is wrong, with no model, Dynamo or Inductor left in it.

---

## The workflow

### 1. Capture

Your script runs unmodified. It does not have to call `torch.compile` — many
eager Spyre ops are themselves implemented by compiling, so eager code produces
capturable kernels too.

```bash
python tests/op_specs/capture.py my_repro.py --out captured/
```

```
captured 1 kernel(s); writing 1
  captured/sdsc_fused__softmax_0.py  (6 OpSpec(s), 2 arg(s))
```

A **kernel** here is one `SpyreAsyncCompile.sdsc()` call — the unit that
compiles and launches atomically. It generally holds several OpSpecs, so one
emitted file is one kernel, not one operation. The softmax above is six.

| Option | Use it when |
|---|---|
| `--kernel NAME` | Only emit kernels whose name contains `NAME` |
| `--save-inputs` | Dump recorded argument values to a `.pt`, for a replay whose inputs are the real ones. Without it, integer args replay as zeros |
| `--no-execute` | Stub the backend compile and launch, so no device is needed |
| `--no-explain-header` | Omit the decoded explanation from each emitted file |

The stale-`fxgraph`-cache trap is handled for you: `capture.py` patches
`force_disable_caches` on for the duration, because a warm cache skips
recompilation entirely and would make the capture silently produce nothing. Torch
announces that on stderr every run, and it is expected, not a problem:

```
UserWarning: dynamo_pgo force disabled by torch.compiler.config.force_disable_caches
```

If the target raises partway through, `capture.py` prints the traceback and
still writes the kernels compiled before it, telling you the later ones are
missing. So a program that crashes on the kernel you are chasing is still
capturable.

### 2. Read it

Every emitted file opens with provenance and a decoded explanation of its own
spec, embedded at capture time so the file self-documents on a machine with no
torch-spyre installed. This is the header of
[example 01](#01--elementwise-add), so you can reproduce it exactly:

```
# Captured OpSpec kernel: sdsc_fused_add_0
#
# Source program:  <your checkout>/tests/op_specs/programs/01_add.py
# Kernel:          1 of 1 compiled from that program
# Environment:     torch 2.13.0+cpu, SENCORES=32, bundle_symbolic_args=True
# Kernel args:     3 in the spec, 3 observed at .run()
#
# Run it:
#     python <this file>                 # compile and launch on the device
#     python <this file> --stage bundle  # sdsc_N.json + bundle.mlir only

# ==============================================================================
#  sdsc_fused_add_0                          1 OpSpec - 3 kernel args - no pool
# ==============================================================================
#
# KERNEL ARGS                           what .run() receives, in arg_index order
#   arg0  input   (128, 256) float16     -> device [4, 128, 64]  SEN169_FP16
#   arg1  input   (128, 256) float16     -> device [4, 128, 64]  SEN169_FP16
#   arg2  output  (128, 256) float16     -> device [4, 128, 64]  SEN169_FP16
#
# OP 1/1  add                                              pointwise - unit: sfp
#   origin      aten.add.Tensor  @  01_add.py:27
#   iteration   c0 =    128    split over 32 cores -> 4 each
#               c1 =    256    not split
#               kernel uses 32 core(s)
#   dim labels  c0 -> mb   c1 -> out
#               positional, from iteration_space insertion order
#   sticks      64 elems/stick at SEN169_FP16 (128-byte sticks)
#               stick dim c1 = 256 -> ceil(256/64) = 4 stick(s)
#               device_size [4, 128, 64]; last dim is always elems/stick
#   args
#     # role   allocation layout scales     strides
#     0 input  hbm @ 0    L0     mb=1 out=1 mb=8192 out=32768
#     1 input  hbm @ 1    L0     mb=1 out=1 mb=8192 out=32768
#     2 output hbm @ 2    L0     mb=1 out=1 mb=8192 out=32768
#   L0 = dim_order [mb, out], stick dim [out], [64] elem/stick
#        raw SDSC label "OUTPUT"
#        a layout label is an equivalence class shared by args with
#        identical layouts -- "OUTPUT" here is NOT an input/output role
```

The `Environment` line reports the config in force when the *spec was built*,
not when the file was written — a program that builds its specs inside a
`config.patch` is reported as it really compiled.

`bundle_symbolic_args` is also *pinned*, as `BUNDLE_SYMBOLIC_ARGS`: the spec's
allocations were baked under it, so a replay has to bundle under the same value.
If your environment disagrees, the script says so and uses the captured one.

### 3. Run it

```bash
python captured/sdsc_fused__softmax_0.py                 # compile and launch
python captured/sdsc_fused__softmax_0.py --stage bundle  # artifacts only
```

`--stage bundle` writes `sdsc_N.json` and `bundle.mlir` and stops. It needs no
device and no `dxp_standalone`, so it is the stage to use when the question is
*"what did my OpSpec turn into?"* rather than *"what does it compute?"*.

Then edit a field and re-run. That is the whole point of the file being
standalone: change a `device_size`, a work division or a coordinate expression,
diff the resulting `bundle.mlir`, and you are probing the backend directly.

### 4. Interpret

| Captured script's result | Where the bug is |
|---|---|
| Also wrong | The **back-end** mis-executes a spec that faithfully describes the intent. You now have a minimal reproducer with no model or graph in it. |
| Correct | The bug is **upstream** — in lowering, fusion, stickification or layout. The spec is not what you meant. |

---

## Why a captured file needs nothing but itself

A captured script imports `torch` and `torch_spyre` and nothing else from this
repo — not even `runner.py`, which is where its run loop comes from. `capture.py`
inlines those helpers into every file it emits, reading them off the live
functions with `inspect.getsource` rather than keeping a copy of the source text,
so an emitted script cannot carry a stale version of the loop and a fix to
`runner.py` reaches the next capture automatically.

That is what makes the file portable: attach it to a ticket, and whoever opens it
can run it on any machine with torch-spyre installed, from any directory, with no
`PYTHONPATH` to set. Everything it needs to allocate the tensors, bring the
runtime up, generate the bundle and launch is in the file.

The one thing that is *not* inlined is `explain.py`. `--explain` on a captured
script re-renders the decoded view from the live `ops` object — so it reflects an
edit the moment you make one, where the comment header at the top of the file is
frozen at capture time. It needs `tests/op_specs/` importable to do that:

```bash
PYTHONPATH=$PWD/tests/op_specs python captured/sdsc_fused_add_0.py --explain
```

Without it, `--explain` says so and carries on. That split is deliberate:
`--stage run` must never depend on a repo import, whereas `--explain` is a
convenience that is allowed to degrade.

The two views differ in one place: host shapes and dtypes were observed at
capture time, so `KERNEL ARGS` shows them in the frozen header but not under
`--explain`.

---

## Field reference

### `OpSpec`

| Field | Meaning |
|---|---|
| `op` | The **backend** op name, not the ATen name you wrote. `aten.mm` arrives as `batchmatmul`. Use `debug_handle` to get back to user code. |
| `is_reduction` | Set whenever a dimension is contracted — including for a matmul, where nobody wrote a reduction but K is one. |
| `iteration_space` | `{symbol: (extent, work_division)}`. **Insertion order is load-bearing**: codegen renames these positionally. |
| `args` | `TensorArg`s, positional. Index in this list is the op's own arg order, *not* `arg_index`. |
| `debug_handle` | Provenance: `aten_op`, `source` file/line, and `fused_from` when an op fuses several origins. A null `aten_op` on a fused op is normal. |
| `tiled_symbols` | Per-level tile-advance symbols, innermost level first. Only present under coarse tiling. |
| `tiled_symbol_trip_counts` | `{symbol: count}` for the above. |

### `TensorArg`

| Field | Meaning |
|---|---|
| `is_input` | Read or written by *this op*. The same `arg_index` can be both across a kernel. |
| `arg_index` | Position in the kernel's argument list, and in `.run()`'s, including when the kernel uses an HBM pool -- the pool is allocated inside the bundle, not passed in. Negative means it is not a kernel arg. |
| `device_dtype` | On-device format, e.g. `SEN169_FP16`. This is what sets the stick size. |
| `device_size` | The tiled device shape, `[..., sticks, rows, elems_per_stick]`. The last dim is always elements per stick. |
| `device_coordinates` | One sympy expression per device dim, mapping an iteration point to a position. An `IndirectAccess` here means the coordinate is a runtime-loaded value. |
| `allocation` | `{"hbm": n}` for a kernel arg, `{"lx": offset}` for a scratchpad intermediate, `{"hbm_pool": offset}` for one that spilled. An empty dict is never valid. |

### `LoopSpec`

Wraps a body of specs in a `count` trip count. Only coarse tiling produces
these, and only via `spyre_hint(num_tiles_per_dim=...)` — see
[Coarse Tiling Loops](../../compiler/coarse_tiling_loops.md).

## Walkthrough: four worked examples

Four programs live in
[`tests/op_specs/programs/`](https://github.com/torch-spyre/torch-spyre/tree/main/tests/op_specs/programs).
Each is about fifteen lines, and each produces an OpSpec shape the others do not.
Work down them in order and you will have met every field in the reference above.

Every one is the same four beats:

1. **Run the program** — is the compiled answer right?
2. **Capture its OpSpec** — what was the back-end actually asked to do?
3. **Read the header** — the decoded explanation, embedded in the emitted file.
4. **Change one thing and capture again** — watch a number move.

Beat 4 is the point. Reading a spec teaches you the field names; changing an input
and diffing the result is what teaches you which fields *mean* something. Captures
land in `./captured/` unless you pass `--out`, and that directory is git-ignored.

A capture never overwrites an existing file — a second one lands as `<name>_1.py`
beside the first, so you keep both to diff rather than losing one. Example 4
captures the same program twice, so its second capture passes its own `--out` to
keep the filenames quoted below matching what you see.

### 01 — Elementwise add

`x + y` over two 128x256 fp16 tensors. The floor: one OpSpec, three args, no
reduction, nothing in the scratchpad or the pool.

**1. Run it.**

```bash
python tests/op_specs/programs/01_add.py
```

```
transfer floor      = 0.00048828125
max|got - expected| = 0.001953125
```

Read those in that order, always. The first has no arithmetic in it at all — it is
what the fp16 round-trip costs on its own, because `torch.float16` is stored on
device as `SEN169_FP16` and keeps one mantissa bit fewer than IEEE fp16. The
second is that plus one add. A diff at the floor is not a bug.

**2. Capture it.**

```bash
python tests/op_specs/capture.py tests/op_specs/programs/01_add.py
```

```
captured 1 kernel(s); writing 1
  captured/sdsc_fused_add_0.py  (1 OpSpec(s), 3 arg(s))
```

**3. Read the header.** `captured/sdsc_fused_add_0.py` opens with the provenance
block shown under [Read it](#2-read-it) above — that example *is* this kernel — and
then the op itself:

```
# OP 1/1  add                                              pointwise - unit: sfp
#   origin      aten.add.Tensor  @  01_add.py:27
#   iteration   c0 =    128    split over 32 cores -> 4 each
#               c1 =    256    not split
#               kernel uses 32 core(s)
#   dim labels  c0 -> mb   c1 -> out
#               positional, from iteration_space insertion order
#   sticks      64 elems/stick at SEN169_FP16 (128-byte sticks)
#               stick dim c1 = 256 -> ceil(256/64) = 4 stick(s)
#               device_size [4, 128, 64]; last dim is always elems/stick
#   args
#     # role   allocation layout scales     strides
#     0 input  hbm @ 0    L0     mb=1 out=1 mb=8192 out=32768
#     1 input  hbm @ 1    L0     mb=1 out=1 mb=8192 out=32768
#     2 output hbm @ 2    L0     mb=1 out=1 mb=8192 out=32768
#   L0 = dim_order [mb, out], stick dim [out], [64] elem/stick
#        raw SDSC label "OUTPUT"
```

**4. Change one thing.** Edit `01_add.py` so both tensors are `(128, 320)` instead
of `(128, 256)`, and capture again. The `sticks` line does the arithmetic in front
of you: `ceil(320/64) = 5`, so `device_size` becomes `[5, 128, 64]` and every
stride on the args scales with it. Nothing else about the spec changes — same op,
same unit, same single layout class.

**What to notice**

- Three args for a two-operand add: the output is an argument too, and `.run()`
  receives all three in `arg_index` order.
- All three share layout `L0`, so there is one layout class. Compare 03, which has
  three.
- Work division is per-dimension: `c0` splits over 32 cores, `c1` does not split.

### 02 — Softmax: one op becomes six

`softmax(x, dim=-1)` over a 64x256 fp16 tensor. One `aten._softmax.default`
becomes `identity, max, sub, exp, sum, realdiv` in a single kernel — back-end op
names, so the last one is not spelled `div`.

**1. Run it.**

```bash
python tests/op_specs/programs/02_softmax.py
```

```
max|got - expected| = 4.1961669921875e-05
```

**2. Capture it.**

```bash
python tests/op_specs/capture.py tests/op_specs/programs/02_softmax.py
```

```
captured 1 kernel(s); writing 1
  captured/sdsc_fused__softmax_0.py  (6 OpSpec(s), 2 arg(s))
```

Six OpSpecs, but only **two** kernel args — the four values flowing between the
ops are never arguments at all.

**3. Read the header.** Five of the six carry the same `origin`, which is what
ties them back to the single line you wrote; the leading `identity` is a copy-in
the backend inserted rather than anything you asked for, so it has none at all.
The reductions are where `scales` stops being decoration:

```
# OP 2/6  max                                              reduction - unit: sfp
#   origin      aten._softmax.default  @  02_softmax.py:26
#   iteration   d0 =     64    split over 32 cores -> 2 each
#               d1 =    256    not split
#   dim labels  d0 -> mb   d1 -> out
#   args
#     # role   allocation layout scales      strides
#     0 input  lx @ 256   L0     mb=1 out=1  mb=4096 out=16384
#     1 output lx @ 0     L0     mb=1 out=-2 mb=4096 out=4096
#   L0 = dim_order [mb, out], stick dim [out], [64] elem/stick
#        raw SDSC label "OUTPUT"
#   out=-2 -> reduced along the stick dim: sparse output, 1 elem per stick
```

Note the symbols are `d0`/`d1` here, where 01's were `c0`/`c1`. The prefix is not
part of the contract — only the *order* is, which is why the `dim labels` line is
the thing to read rather than the symbol names.

That output's `device_size` drops to `[1, 64, 64]` — one stick, because the
reduction collapsed the stick dimension.

**4. Change one thing.** Make the tensor `(32, 256)` instead of `(64, 256)` and
capture again. The reduction output's `device_size` follows the row count to
`[1, 32, 64]` while the stick dimension stays at 64 elements — the collapse is in
the *stick* dim, not the row dim, and halving the rows shows you which is which.

**What to notice**

- `allocation` is `lx @ offset`, not `hbm @ n`: these are scratchpad
  intermediates, so they never reach HBM and never become kernel args.
- One source line, six spec entries. `origin` is the only thing that links them,
  and a backend-inserted op has none — a missing `origin` is not a bug.
- A negative scale is the reduction. `out=-2` specifically means *reduced along
  the stick dim*; any other negative value is an ordinary reduced dimension.

### 03 — Matmul: a different unit, and the *other* label lists

`a @ b`, `(64, 128) @ (128, 256)`, fp16. Still one OpSpec, and almost nothing
else matches 01.

**1. Run it.**

```bash
python tests/op_specs/programs/03_matmul.py
```

```
max|got - expected| = 0.125
```

**2. Capture it.**

```bash
python tests/op_specs/capture.py tests/op_specs/programs/03_matmul.py
```

```
captured 1 kernel(s); writing 1
  captured/sdsc_fused_mm_0.py  (1 OpSpec(s), 3 arg(s))
```

**3. Read the header.**

```
# OP 1/1  batchmatmul                                       reduction - unit: pt
#   origin      aten.mm.default  @  03_matmul.py:27
#   iteration   c0 =     64    split over 4 cores -> 16 each
#               c1 =    256    split over 4 cores -> 64 each
#               c2 =    128    split over 2 cores -> 64 each
#               kernel uses 32 core(s)
#   dim labels  c0 -> mb   c1 -> out   c2 -> in
#   sticks      64 elems/stick at SEN169_FP16 (128-byte sticks)
#               stick dim c1 = 256 -> ceil(256/64) = 4 stick(s)
#               device_size [4, 64, 64]; last dim is always elems/stick
#   args
#     # role   allocation layout scales     strides
#     0 input  hbm @ 0    L0     mb=1 in=1  mb=4096 in=8192
#     1 input  hbm @ 1    L1     in=1 out=1 in=8192 out=32768
#     2 output hbm @ 2    L2     mb=1 out=1 mb=4096 out=16384
#   L0 = dim_order [mb, in], stick dim [in], [64] elem/stick
#        raw SDSC label "INPUT"
#   L1 = dim_order [in, out], stick dim [out], [64] elem/stick
#        raw SDSC label "KERNEL"
#   L2 = dim_order [mb, out], stick dim [out], [64] elem/stick
#        raw SDSC label "OUTPUT"
```

Three layout classes for three arguments, and none of the raw labels lines up with
the argument's role: the *input* `a` is `"INPUT"`, but only because a matmul reads
`MATMUL_LAYOUT_LABELS`. Compare 01, where the sole class on all three args —
including the output — was raw-labelled `"OUTPUT"`.

**4. Change one thing.** Take K from 128 to 256 — `a` becomes `(64, 256)` and `b`
becomes `(256, 256)` — and capture again. `c2` is the contracted dimension, so its
extent tracks K. Watch what the compiler does with the *work division* on that
line: how a reduction dimension gets split is policy, not arithmetic you can
predict from the shape, which is exactly why it is worth looking at.

**What to notice**

- It is a `reduction` although nobody wrote one: K is contracted. And it runs on
  `unit: pt`, the matrix engine, not the `unit: sfp` every elementwise op gets.
- The op is named `batchmatmul`. `aten.mm` appears only in `origin` — OpSpec op
  names are back-end names.
- Three iteration dims, and `c2 -> in`, where `"in"` is *not a member of*
  `INPUT_DIM_LABELS`. A matmul draws from `MATMUL_DIM_LABELS`, so the legend you
  learned from 01 does not carry over.
- Three layout classes, and the first raw label is `"INPUT"`, not `"OUTPUT"` —
  because a matmul uses `MATMUL_LAYOUT_LABELS`, whose order differs. A raw label
  is an equivalence class over layouts, never a role.
- Per-dim work division is not the kernel's core count: `c2` splits over 2 while
  the kernel still uses 32.

### 04 — Spilling to the HBM pool

`softmax(a @ b) @ c`. An intermediate that has to survive across the kernel, and
is too large or too long-lived for the scratchpad, cannot be `lx`.

**1. Run it.**

```bash
python tests/op_specs/programs/04_pool_chain.py
```

```
max|got - expected| = 0.005859375
```

**2. Capture it.**

```bash
python tests/op_specs/capture.py tests/op_specs/programs/04_pool_chain.py
```

```
captured 1 kernel(s); writing 1
  captured/sdsc_fused__softmax_mm_0.py  (7 OpSpec(s), 4 arg(s), 32768-byte pool)
```

The summary line gains a pool note, and the emitted file's provenance block gains
a line the first three do not have:

```
# Kernel args:     4 in the spec, 4 observed at .run()
# Pool:            32768 bytes, allocated by the bundle
```

**3. Read the header.** Seven OpSpecs this time, and the intermediates that could
not fit the scratchpad carry `hbm_pool @ offset` where 02's carried `lx @ offset`.

```
# OP 1/7  batchmatmul                                       reduction - unit: pt
#   origin      aten.mm.default  @  04_pool_chain.py:25
#   iteration   c0 =     64    split over 4 cores -> 16 each
#               c1 =    256    split over 4 cores -> 64 each
#               c2 =    128    split over 2 cores -> 64 each
#               kernel uses 32 core(s)
#   dim labels  c0 -> mb   c1 -> out   c2 -> in
#               positional, from iteration_space insertion order
#   sticks      64 elems/stick at SEN169_FP16 (128-byte sticks)
#               stick dim c1 = 256 -> ceil(256/64) = 4 stick(s)
#               device_size [4, 64, 64]; last dim is always elems/stick
#   args
#     # role   allocation   layout scales     strides
#     0 input  hbm @ 0      L0     mb=1 in=1  mb=4096 in=8192
#     1 input  hbm @ 1      L1     in=1 out=1 in=8192 out=32768
#     2 output hbm_pool @ 0 L2     mb=1 out=1 mb=4096 out=16384
#   L0 = dim_order [mb, in], stick dim [in], [64] elem/stick
#        raw SDSC label "INPUT"
#   L1 = dim_order [in, out], stick dim [out], [64] elem/stick
#        raw SDSC label "KERNEL"
#   L2 = dim_order [mb, out], stick dim [out], [64] elem/stick
#        raw SDSC label "OUTPUT"
```

**4. Change one thing.** Shrink the chain — take the inner dimension from 256 down
to 64, so `b` is `(128, 64)` and `c` is `(64, 128)` — and capture again, into its
own directory so the first capture survives to diff against:

```bash
python tests/op_specs/capture.py tests/op_specs/programs/04_pool_chain.py \
    --out captured/04_small
```

The pool does *not* go away; it scales:

```
  captured/04_small/sdsc_fused__softmax_mm_0.py  (7 OpSpec(s), 4 arg(s), 8192-byte pool)
```

32768 bytes down to 8192, with the same seven ops and the same four args. Worth
doing precisely because the intuition it corrects is the tempting one: spilling is
not a threshold this shape crosses, it is a property of the chain, and shrinking
the tensors buys a smaller pool rather than no pool.

**What to notice**

- The pool is **not** an argument. The bundle allocates it itself, as
  `%pool = sdscbundle.device_mem_allocate 32768 bytes` on the first line of
  `bundle.mlir`, so `.run()` receives the four kernel args and nothing else.
- A replay needs only `POOL_SIZE`, never the offset map: the size reaches
  `sdsc()` and the offsets in the spec take care of themselves.
- Three allocation kinds have now appeared across the four examples: `hbm @ n` for
  a kernel arg (01), `lx @ offset` for a scratchpad intermediate (02), and
  `hbm_pool @ offset` for one that spilled (04).

### Regenerating

Captures are git-ignored rather than committed, and it is worth knowing why,
because the reasons apply to any captured file you might think of attaching to a
branch. Each embeds `runner.py`'s helpers verbatim, so one edit to that module
invalidates every copy. `DebugHandle.source` records the absolute path of the
program on the capturing machine, and `DebugHandle.id` is a hash *of* that source,
so the bytes differ per checkout. And the OpSpec literal comes from the compiler's
own printer at a few hundred columns, which `ruff-format` would reflow and the next
capture would undo. Attach a capture to a *ticket*, not to the repo.

Re-capture after a compiler change and diff against what you had. Every block above
is real codegen from a real device run, so patching a number by inference is how they
go stale: an `origin` line moves whenever its program does, and the op indices move
whenever fusion does. Re-run the four programs and paste, rather than adjusting a
digit that looks wrong.

---

## See Also

- [Debugging overview](index.md) — the outer workflow this plugs into
- [Inductor artifacts](inductor_artifacts.md) — the stages upstream of the OpSpec
- [Tensors and layouts](../tensors_and_layouts.md) — sticks, tiling, `device_size`
- [Coarse tiling loops](../../compiler/coarse_tiling_loops.md) — what produces a `LoopSpec`
- [Indirect access](../../compiler/indirect_access.md) — how a gather becomes coordinates
