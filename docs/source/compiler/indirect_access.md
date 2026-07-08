# Indirect Access (Gather)

This page describes how Torch-Spyre compiles *indirect* tensor accesses —
operations like `aten.index` where the row of a data tensor is selected by a
value loaded at runtime from a separate index tensor.

**Quick navigation:**

- [What is indirect access?](#what-is-indirect-access)
- [How Inductor represents indirect access](#how-inductor-represents-indirect-access)
- [The challenge: coordinates that depend on runtime values](#the-challenge-coordinates-that-depend-on-runtime-values)
- [Solution: range and substitution for indirect symbols](#solution-range-and-substitution-for-indirect-symbols)
- [Pipeline walkthrough](#pipeline-walkthrough)
- [Stick compatibility for index tensors](#stick-compatibility-for-index-tensors)
- [Op spec layout](#op-spec-layout)
- [Fusion](#fusion)
- [Current limitations](#current-limitations)

---

## What is indirect access?

A gather reads a value tensor at a row determined by a *runtime-loaded* index:

```python
# x: float16 [M, N], i: int32 [P, Q]
# result: float16 [P, Q, N] — row i[p,q] of x, for each (p,q)
result = x[i]         # aten.index
result = x[i].exp()   # aten.index fused with aten.exp
```

---

## How Inductor represents indirect access

Inductor lowers `aten.index` to a `Pointwise` node whose `inner_fn` calls
`ops.indirect_indexing()` to convert a loaded value into a loop index:

```
load(index_tensor, ...)        → tmp0   (int32 value)
indirect_indexing(tmp0, ...)   → i_sym  (used as row address)
load(value_tensor, i_sym * N + c2)
```

In the resulting `MemoryDep`, the index expression for the value tensor
contains a symbol (e.g. `tmp0`) that is **not** present in `dep.ranges` —
it has no static loop bound because it is resolved only at runtime.
`MemoryDep.is_indirect()` returns `True` for such deps.

---

## The challenge: coordinates that depend on runtime values

Every stage of the Spyre compilation pipeline — stick compatibility checking,
`normalize_coordinates`, `align_tensors`, and op spec generation — works with
*device-coordinate expressions*: symbolic formulas that map iteration variables
`(c0, c1, ...)` to positions in the on-device tiled layout.

For a direct access these formulas contain only loop variables.  For an
indirect access, one coordinate of the value tensor contains `tmp0`, which
represents a value that only the *hardware* will know at runtime.  Without
special handling every stage that computes or checks coordinates would either
crash, silently skip the symbol, or produce wrong layout decisions.

---

## Solution: range and substitution for indirect symbols

`compute_coordinates` needs two things to handle an indirect symbol correctly:
its **range** (to place it in the right device dimension) and a **substitution**
(to replace the opaque symbol with a named expression for later stages).
These are handled by two separate optional parameters.

### Range for indirect symbols: `indirect_sizes`

`compute_coordinates` accepts an optional `indirect_sizes: dict[Symbol, int]`
parameter.  When a free symbol in the index is not in `var_ranges`,
the function looks it up here to get its range.

Indirect symbols have no entry in `dep.ranges` — their range is the `size`
argument passed to `ops.indirect_indexing()` at lowering time, baked into the
`inner_fn` closure and not visible in the printed IR.  How `indirect_sizes` is
built differs between pipeline stages:

| Stage | Symbol name | Source |
|---|---|---|
| Pre-scheduler (`propagate_layouts`) | `tmp0`, `tmp1`, … (Inductor-assigned) | `indirect_sizes_from_op(op)` in `pass_utils.py` — re-executes `inner_fn` via `_IndirectIndexFinder`, which intercepts `indirect_indexing()` calls and captures the size argument |
| Post-scheduler (`SpyreKernel`) | `indirect0`, `indirect1`, … (Spyre-assigned) | `SpyreKernel.indirect_sizes` — populated by `SpyreKernelOpsHandler.indirect_indexing()` during kernel body execution; passed to `compute_coordinates` from `create_tensor_arg` |

If `indirect_sizes=None` is passed — as is the case for structural callers that
only check stick compatibility or layout shape — the indirect symbol is silently
skipped and its contribution is left as zero in the output coordinates. If
`indirect_sizes` is a dict but does not contain a symbol that appears in the
index, `Unsupported` is raised immediately rather than producing silently wrong
coordinates.

### Naming indirect symbols: `IndirectAccess`

Once `tmp0` is placed in the correct device dimension, the coordinate
expression still contains the raw symbol.  Later stages and the backend need
to know which buffer supplies the runtime value.  The solution is
`IndirectAccess`, a sympy `Function` subclass defined in
[`op_spec.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/op_spec.py):

```python
class IndirectAccess(Function):
    """IndirectAccess(tensor_name) — value loaded from that tensor at the current point."""
```

`IndirectAccess('arg1_1')` means *"the value loaded from the buffer named `arg1_1`
at the current iteration point"*.  It is an opaque sympy atom that survives
`sympify` round-trips and can be carried through arithmetic expressions without
being evaluated.

#### Building the substitution dict

Two helpers in
[`pass_utils.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py)
produce a `{indirect_sym → IndirectAccess(name)}` substitution dict, depending on
the pipeline stage:

| Helper | When to use | How it works |
|---|---|---|
| `indirect_access_subs_from_op(op)` | Pre-scheduler (`propagate_layouts`) | Re-executes `inner_fn` via `_IndirectIndexFinder` to discover which buffer's load fed each indirect symbol |
| `indirect_access_subs_from_kernel(indirect_vars)` | Post-scheduler (`SpyreKernel`) | Reads `SpyreKernel.indirect_vars`, which `SpyreKernelOpsHandler.indirect_indexing()` populates live during codegen |

Both return the same dict shape, so every downstream caller sees a uniform
interface regardless of when it runs.

#### Applying the substitution

`compute_coordinates` (in
[`views.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/views.py))
accepts an optional `indirect_access_subs` parameter.  After computing the
coordinate expressions in the normal way it applies the substitution:

```python
if indirect_access_subs:
    coordinates = [c.xreplace(indirect_access_subs) for c in coordinates]
```

This replaces `tmp0` with `IndirectAccess('arg1_1')` in the affected coordinate,
giving a gather-aware expression that all later stages can handle.

The `device_coordinates` wrapper in `pass_utils.py` forwards the parameter
to `compute_coordinates`, so callers in `propagate_layouts.py` and
`spyre_kernel.py` only need to pass the dict through one function.

---

## Pipeline walkthrough

The gather `x[i]` with `x: float16 [128, 256]` and `i: int32 [3, 192]` passes
through the following stages.

### 1. propagate_layouts (pre-scheduler)

`compute_layouts` is called for the `Pointwise` node.  Both
`indirect_sizes_from_op` and `indirect_access_subs_from_op` re-execute `inner_fn`
via `_IndirectIndexFinder` to recover the size and the source buffer name for
`tmp0`.

The resulting device coordinates (logged when `TORCH_LOGS="+inductor"` is set)
show the substitution explicitly:

```
input[1] name=arg0_1  (value tensor x)
  device_coordinates=[floor(d2/64), tmp0, Mod(d2, 64)]
    ->  [floor(d2/64), IndirectAccess(arg1_1), Mod(d2, 64)]
```

The `IndirectAccess` coordinate is passed through `normalize_coordinates` as an
opaque `Term(var=None, offset=IndirectAccess(...))` and through `align_tensors`
unchanged.

### 2. SpyreKernel codegen (post-scheduler)

`SpyreKernelOpsHandler.indirect_indexing()` intercepts the `indirect_indexing`
call during `inner_fn` execution and stores both the source `TensorAccess` and
the size:

```python
sym = sympy_index_symbol(f"indirect{self.kernel._indirect_var_count}")
self.kernel._indirect_var_count += 1
self.kernel.indirect_vars[sym] = index_var   # TensorAccess for arg1_1
self.kernel.indirect_sizes[sym] = int(size)
```

`indirect_access_subs_from_kernel(self.indirect_vars)` converts `indirect_vars`
to `{sym: IndirectAccess(Symbol('arg1_1'))}`, and `self.indirect_sizes` is
passed directly — both go to `compute_coordinates` from `create_tensor_arg`.

### 3. Op spec generation

`SpyreKernel` emits the index tensor as the **first** `TensorArg` in the op
spec with `name='arg1_1'` set.  This name is what `IndirectAccess('arg1_1')` refers
to in the value tensor's coordinates:

```python
TensorArg(
    is_input=True, arg_index=0, device_dtype=DataFormats.IEEE_INT32,
    device_size=[1, 6, 3, 32],
    device_coordinates=[0, floor(c1/32), c0, Mod(c1, 32)],
    name='arg1_1',
),
TensorArg(
    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
    device_size=[1, 4, 128, 64],
    device_coordinates=[0, floor(c2/64), IndirectAccess('arg1_1'), Mod(c2, 64)],
),
```

The backend compiler reads `IndirectAccess('arg1_1')` as: *"load this tensor's row
index from the tensor named `arg1_1` at the current iteration point"*.

### 4. Wrapper serialization

`SpyreKernel._codegen_op_spec_list` serializes `IndirectAccess` expressions
using a dedicated branch of `sympy_str`:

```python
if isinstance(x, IndirectAccess):
    name_sym = x.args[0]
    return f"IndirectAccess('{name_sym}')"
```

The generated wrapper imports `IndirectAccess` from `op_spec` so the op spec
survives `eval`/`sympify` round-trips at kernel load time.

---

## Stick compatibility for index tensors

The standard stick-compatibility check (`compute_restickify_needed`) compares
stick-dimension expressions across input and output tensors to decide whether
a restickify pass is needed.  Index tensors must be excluded from this check:
their loaded *values* determine an address, not a position in the output, so
constraining their stick layout to match the output would be incorrect.

`compute_restickify_needed` accepts an optional `op` parameter.  When
provided, it calls `indirect_index_dep_names(op)` and returns `(False, None)`
immediately for any dep whose name is in that set:

```python
if op is not None and in_dep.name in indirect_index_dep_names(op):
    return False, None
```

The same exclusion is applied in `_multi_arg_pointwise_layouts` when
collecting stick expressions:

```python
indirect_index_names = indirect_index_dep_names(op)
stick_exprs = {
    device_coordinates(stl, arg.dep)[-1]
    for arg in args
    for stl in arg.layouts
    if arg.dep.name not in indirect_index_names
    and device_coordinates(stl, arg.dep)[-1] != 0
}
```

---

## Op spec layout

For an unfused gather the scheduler produces **two** op specs:

1. **identity** — copies gathered rows from the value tensor into a
   temporary buffer using `IndirectAccess` coordinates.
2. **exp** (or whichever unary follows) — applies the unary to the temporary
   buffer using direct coordinates.

When fusion is enabled (currently disabled pending backend support — see the
flag in `patches.py`), the two ops collapse into a single fused op spec with
the index tensor as a named input.

The argument ordering rule for op specs that contain an index tensor is:

1. Index tensor(s), in buffer-name order, each with `name` set.
2. Value tensor(s), in the order they appear in the computation.
3. Output tensor.

The helper `_is_indirect_index_arg` identifies index-role args post-hoc by
scanning other args' coordinates for `IndirectAccess` atoms whose name matches.

---

## Fusion

Pointwise fusion with the gather is currently **disabled** in `patches.py`
because `IndirectAccess` coordinate expressions are not yet handled in SuperDSC
generation. When enabled, the identity op and the downstream pointwise op
merge into one op spec with a single `IndirectAccess` coordinate expression in
the input arg.

---

## Current limitations

- Only 1-D index tensors (a single indirect symbol per data dep) are supported.
- Scatter index tensors (`aten.scatter_`, `aten.index_put_`) are correctly
  detected via `_find_scatter_index_buf_names` and excluded from stick
  compatibility checks. However, `IndirectAccess` coordinates on output args
  (the codegen side of scatter) are not yet wired up in SuperDSC generation.
- The fused (single op spec) path is disabled because `IndirectAccess` coordinates
  are not yet handled in SuperDSC generation.
