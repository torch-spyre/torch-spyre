# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Render an OpSpec list as a human-readable explanation of what it means.

Everything is derived from ``parse_op_spec()`` -- the ``SDSCSpec`` for resolved
layout/scales/strides and the ``symbol_mapping`` for the dim-label legend -- so
this view cannot drift from what SDSC codegen does.

The point is to decode rather than dump.  ``SDSCSpec.__str__`` prints every
field at ~14 lines per argument, and leaves two things unexplained that
routinely mislead readers:

1. Iteration symbols are renamed positionally: ``c0``/``c1`` arrive as
   ``mb``/``out``, names appearing nowhere in the OpSpec.  Hence the "dim
   labels" legend.
2. ``layout=OUTPUT`` shows up on *input* arguments, because labels are handed
   out per distinct layout rather than per role.  Hence relabelling to
   L0/L1/... with the raw name in a footnote.

``verbose=True`` appends the raw ``str(SDSCSpec)`` per op.  Output is plain
ASCII wrapped to 78 columns: it gets embedded as a comment header inside
generated Python files and pasted into tickets.
"""

import os
import textwrap

from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp
from torch_spyre._inductor.codegen.superdsc import parse_op_spec
from torch_spyre._inductor.spyre_kernel import _iter_op_specs

WIDTH = 78


# ---------------------------------------------------------------------------
# Small formatting helpers
#
# These four -- _labelled, _wrapped, _fit and _table -- are what hold the render
# inside WIDTH at any nesting depth, and each has broken that at least once.
# test_hostile_names_stay_within_width in test_op_spec_lab.py is the tripwire:
# re-run it after touching anything here, since a short flat kernel exercises
# almost none of this.
# ---------------------------------------------------------------------------


def _num(expr) -> str:
    """Render a possibly-symbolic sympy extent compactly."""
    try:
        return str(int(expr))
    except (TypeError, ValueError):
        return str(expr)


def _alloc(arg: TensorArg) -> str:
    """ "hbm @ 0" / "hbm_pool @ 1048576" / "(unset)" for an allocation dict."""
    if not arg.allocation:
        return "(unset)"
    key = next(iter(arg.allocation))
    return f"{key} @ {arg.allocation[key]}"


def _role(arg: TensorArg) -> str:
    return "input" if arg.is_input else "output"


def _kv(mapping) -> str:
    """Render a {symbol: value} dict as "a=1 b=2"."""
    return " ".join(f"{k}={v}" for k, v in mapping.items())


def _kernel_arg_roles(specs) -> dict:
    """Map arg_index -> "input" / "output" / "in+out" across every op.

    An index can be both when a kernel updates a tensor in place, so this is
    computed over all ops rather than taken from the first one seen.
    """
    seen: dict = {}
    for op in _iter_op_specs(specs):
        for arg in op.args:
            if isinstance(arg, TensorArg) and arg.arg_index >= 0:
                prev = seen.get(arg.arg_index)
                role = _role(arg)
                seen[arg.arg_index] = "in+out" if prev and prev != role else role
    return seen


def _labelled(label: str, parts: list, pad: str = "", joiner: str = "   ") -> list:
    """Render ``  label  part joiner part ...`` under a 14-column label gutter.

    Wraps *between* whole parts rather than at spaces: "c0 -> mb" pairs and
    dotted op names are unreadable split down the middle.  ``pad`` is the
    caller's own indent, charged against the budget so nesting stays inside 78.
    """
    budget = max(WIDTH - len(pad) - 14, 20)
    rows: list = []
    current = ""
    for part in parts:
        candidate = f"{current}{joiner}{part}" if current else str(part)
        if current and len(candidate) > budget:
            rows.append(current)
            current = str(part)
        else:
            current = candidate
    rows.append(current)
    # A single part can exceed the budget alone. Break at whitespace but never
    # mid-token: a split identifier no longer names anything.
    broken: list = []
    for row in rows:
        broken.extend(
            textwrap.wrap(
                row, width=budget, break_long_words=False, break_on_hyphens=False
            )
            or [row]
        )
    return [
        (f"  {label}".ljust(14) if i == 0 else " " * 14) + row
        for i, row in enumerate(broken)
    ]


def _wrapped(head: str, body: str, pad: str = "") -> list:
    """``head + body`` wrapped to the column budget, continuation lines aligned.

    For unbounded text -- an exception message, a tile-advance symbol.
    Truncating removes the tail, which is where an AttributeError names the
    attribute.
    """
    budget = max(WIDTH - len(pad) - len(head), 20)
    chunks = textwrap.wrap(
        body, width=budget, break_long_words=False, break_on_hyphens=False
    )
    return [
        (head if i == 0 else " " * len(head)) + c
        for i, c in enumerate(chunks or [body])
    ]


def _fit(label: str, body: str, pad: str = "") -> list:
    """A ``label`` + ``body`` row under the same 14-column gutter, wrapped if long.

    Rows that fit are emitted byte for byte: the aligned columns in the iteration
    and sticks sections carry meaning, and ``textwrap`` would collapse the runs of
    spaces that produce them.  Only an overflowing row gets rewrapped.
    """
    head = f"  {label}".ljust(14)
    if len(pad) + len(head) + len(body) <= WIDTH:
        return [head + body]
    return _wrapped(head, body, pad)


def _table(rows: list, headers: list, indent: str, pad: str = "") -> list:
    """Render an aligned table, spilling wide trailing cells onto their own line.

    Columns are sized to their content; trailing columns that would run past
    WIDTH move to a continuation line rather than being truncated.  ``pad`` is
    charged against the budget but not prefixed -- the caller adds it.
    """
    if not rows:
        return []
    widths = [
        max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))
    ]

    def build(cells, is_header=False):
        out, spill, budget = [], [], WIDTH - len(indent) - len(pad)
        for i, cell in enumerate(cells):
            piece = str(cell).ljust(widths[i])
            # Once one column spills, every later one must too, or a narrow
            # trailing cell lands under the wrong header.
            if out and (spill or len(" ".join(out + [piece])) > budget):
                # A spilled cell names its own column, except on the header row,
                # where the cell already *is* the column name ("scales scales").
                # Every cell is padded to the shared widths, so the header line is
                # exactly as wide as the widest row and spills whenever one does.
                spill.append(str(cell) if is_header else f"{headers[i]} {cell}".strip())
            else:
                out.append(piece)
        line = indent + " ".join(out).rstrip()
        return [line] + ([indent + "    " + "  ".join(spill)] if spill else [])

    lines = build(headers, is_header=True)
    for row in rows:
        lines += build(row)
    return lines


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _title(kernel_name, specs, args, pool_size) -> list:
    n_ops = sum(1 for _ in _iter_op_specs(specs))
    n_args = len(args) if args else len(_kernel_arg_roles(specs))
    facts = f"{n_ops} OpSpec{'s' if n_ops != 1 else ''} - {n_args} kernel args"
    facts += " - no pool" if not pool_size else f" - {pool_size}-byte pool"
    name = kernel_name or "(unnamed OpSpec list)"
    gap = WIDTH - 2 - len(name) - len(facts)
    if gap < 1:
        # A fused kernel name can fill 78 columns alone, so right-align the facts
        # underneath rather than overflow the rule. The name forgoes the leading
        # space: it is unbreakable, and that column can be the difference.
        header = [name, f" {facts:>{WIDTH - 2}}"]
    else:
        header = [f" {name}{' ' * gap}{facts}"]
    return ["=" * WIDTH, *header, "=" * WIDTH, ""]


def _kernel_args_section(specs, args, pool_size) -> list:
    """The tensors .run() receives, in arg_index order.

    ``args`` is optional: capture.py has the host shapes observed at launch,
    where a hand-written spec has only what the TensorArgs carry.
    """
    roles = _kernel_arg_roles(specs)
    if not roles and not pool_size:
        return []

    caption = "what .run() receives, in arg_index order"
    header = "KERNEL ARGS" + " " * max(WIDTH - 11 - len(caption), 1) + caption
    lines = [header]

    if pool_size:
        lines.append(
            f"  pool           {pool_size} bytes -- allocated by the bundle, not an arg"
        )

    # Fall back to the TensorArgs when no observed-launch info was supplied.
    by_index: dict = {}
    for op in _iter_op_specs(specs):
        for arg in op.args:
            if isinstance(arg, TensorArg) and arg.arg_index >= 0:
                by_index.setdefault(arg.arg_index, arg)

    for idx in sorted(roles):
        role = roles[idx].ljust(7)
        rec = args[idx] if args and idx < len(args) else None
        if rec is not None:
            host = f"{tuple(rec.shape)} {str(rec.dtype).removeprefix('torch.')}"
            dev = f"device {list(rec.device_size)}  {rec.device_dtype_name}"
            lines.append(f"  arg{idx}  {role} {host:<22} -> {dev}")
        else:
            arg = by_index.get(idx)
            dev = (
                f"device {arg.device_size}  {_enum_name(arg.device_dtype)}"
                if arg is not None
                else "(no layout information)"
            )
            lines.append(f"  arg{idx}  {role} {dev}")
    return lines + [""]


def _enum_name(value) -> str:
    """Bare member name of a pybind enum ("SEN169_FP16")."""
    name = getattr(value, "name", None)
    return name if name is not None else str(value).rsplit(".", 1)[-1]


def _origin(op_spec, pad: str = "") -> list:
    """The source location and ATen op this OpSpec came from, if recorded.

    A null ``source``/``aten_op`` is normal for an op fusing several origins, so
    fall back to ``fused_from`` rather than printing None.  Name and location are
    separate parts so a long qualified name drops the location to its own line.
    """
    handle = getattr(op_spec, "debug_handle", None)
    if handle is None:
        return []
    where = ""
    if handle.source is not None:
        base = os.path.basename(handle.source.file)
        where = f"@  {base}:{handle.source.start_line}"
    if handle.aten_op:
        parts = [handle.aten_op, where or "(no source location)"]
        return _labelled("origin", parts, pad, joiner="  ")
    if handle.fused_from:
        ops = sorted({h.aten_op for h in handle.fused_from if h.aten_op})
        head = f"fused from {len(handle.fused_from)} origins"
        if not ops:
            return _labelled("origin", [head], pad)
        # First op rides with the head so the colon stays attached to it; the
        # rest are separate parts, so a long fusion wraps between op names.
        return _labelled("origin", [f"{head}: {ops[0]}", *ops[1:]], pad, joiner=", ")
    return []


def _iteration(op_spec, cores, pad: str = "") -> list:
    """Per-dim extent and how it is divided across cores."""
    lines = []
    for i, (sym, (extent, wd)) in enumerate(op_spec.iteration_space.items()):
        note = "not split"
        if wd and wd > 1:
            try:
                ext = int(extent)
                per = -(-ext // wd)
                note = f"split over {wd} cores -> {per} each"
                if ext % wd:
                    note += " (uneven)"
            except (TypeError, ValueError):
                note = f"split over {wd} cores"
        lines += _fit(
            "iteration" if i == 0 else "", f"{sym} = {_num(extent):>6}    {note}", pad
        )
    if cores:
        lines += _fit("", f"kernel uses {cores} core(s)", pad)
    return lines


def _dim_labels(symbol_mapping, pad: str = "") -> list:
    """The positional rename legend -- see trap 1 in the module docstring.

    Only actual renames: a tiled kernel's tile-advance symbols map to themselves
    and run past fifty characters, saying nothing and blowing the budget.
    """
    renames = [(k, v) for k, v in (symbol_mapping or {}).items() if str(k) != str(v)]
    if not renames:
        return []
    pairs = [f"{k} -> {v}" for k, v in renames]
    return _labelled("dim labels", pairs, pad) + [
        " " * 14 + "positional, from iteration_space insertion order"
    ]


def _sticks(op_spec, sdsc_spec, symbol_mapping, pad: str = "") -> list:
    """Explain how the logical extent turns into sticks for the output arg.

    Deliberately does not guard ``elems_per_stick()``: swallowing that would make
    the whole section vanish silently, where letting it raise gets it reported as
    a ``!!`` line by the section wrapper in :func:`_op_block`.
    """
    out = next((a for a in op_spec.args if not a.is_input), None)
    if out is None or not out.device_size:
        return []
    eps = out.device_dtype.elems_per_stick()
    fmt = _enum_name(out.device_dtype)
    lines = _fit("sticks", f"{eps} elems/stick at {fmt} (128-byte sticks)", pad)

    # Reverse the rename so the stick dim is named in the user's own symbols.
    reverse = {v: k for k, v in (symbol_mapping or {}).items()}
    label = None
    for sdsc_arg in sdsc_spec.args if sdsc_spec is not None else []:
        if sdsc_arg.arg_index == out.arg_index:
            label = sdsc_arg.layout
            break
    # stick_dim_order went from a bare Symbol to a list in #2286. Accept either.
    raw = None
    if label is not None and label in sdsc_spec.layouts:
        raw = sdsc_spec.layouts[label].get("stick_dim_order")
    stick_dims = list(raw) if isinstance(raw, (list, tuple)) else [raw]

    for sdsc_dim in stick_dims:
        # An unhashable dim, a missing dim and a symbolic extent all just mean
        # "no stick line for this dim", never a traceback.
        try:
            own = reverse.get(sdsc_dim, sdsc_dim)
            extent = int(op_spec.iteration_space[own][0])
            count = -(-extent // eps)
        except (TypeError, ValueError, KeyError):
            continue
        lines += _fit(
            "",
            f"stick dim {own} = {extent} -> ceil({extent}/{eps}) = {count} stick(s)",
            pad,
        )
    lines += _fit(
        "", f"device_size {out.device_size}; last dim is always elems/stick", pad
    )
    return lines


def _tiled(op_spec, pad: str = "") -> list:
    """Loop-tiling symbols and trip counts, innermost level first."""
    if not op_spec.tiled_symbols:
        return []
    lines = []
    for level, syms in enumerate(op_spec.tiled_symbols):
        if not syms:
            body = f"lvl{level}: (loop-invariant)"
        else:
            named = ", ".join(
                f"{s} (x{op_spec.tiled_symbol_trip_counts.get(s, '?')})" for s in syms
            )
            body = f"lvl{level}: {named}"
        lines += _fit("tiled" if level == 0 else "", body, pad)
    return lines


def _args_table(op_spec, sdsc_spec, layout_names, pad: str = "") -> list:
    """One row per argument: role, allocation, layout class, scales, strides."""
    if sdsc_spec is None:
        rows = [
            [i, _role(a), _alloc(a), "-", "-", "-"] for i, a in enumerate(op_spec.args)
        ]
    else:
        rows = []
        for i, (arg, sdsc_arg) in enumerate(zip(op_spec.args, sdsc_spec.args)):
            rows.append(
                [
                    i,
                    _role(arg),
                    _alloc(arg),
                    layout_names.get(sdsc_arg.layout, sdsc_arg.layout),
                    _kv(sdsc_arg.scales),
                    _kv(sdsc_arg.strides),
                ]
            )
    lines = ["  args"] + _table(
        rows, ["#", "role", "allocation", "layout", "scales", "strides"], "    ", pad
    )

    # Extras only when they carry information, so the common case stays narrow.
    if sdsc_spec is not None:
        for i, sdsc_arg in enumerate(sdsc_spec.args):
            extra = []
            if sdsc_arg.backGap:
                extra.append(f"backGap {_kv(sdsc_arg.backGap)}")
            if any(v for v in sdsc_arg.offsets.values()):
                extra.append(f"offsets {_kv(sdsc_arg.offsets)}")
            if any(v != -1 for v in sdsc_arg.max_dim_sizes.values()):
                extra.append(f"max_dim {_kv(sdsc_arg.max_dim_sizes)}")
            # Whether this reference advances across a coarse-tile loop: #3567
            # made this expression the single source of truth, replacing the
            # per-buffer per_tile_fixed flag. None means it stays put.
            if sdsc_arg.device_tile_advance_expr is not None:
                extra.append(f"tile advance {sdsc_arg.device_tile_advance_expr}")
            if extra:
                # A tile-advance symbol runs to 48 characters, so wrap.
                lines += _wrapped(f"      arg {i}: ", "  ".join(extra), pad)
    return lines


def _layout_footnotes(op_spec, sdsc_spec, layout_names) -> list:
    """Spell out each layout class, and that the raw label is not a role.

    See trap 2 in the module docstring: LAYOUT_LABELS[0] is the string
    "OUTPUT", so an input argument routinely reads layout=OUTPUT.
    """
    if sdsc_spec is None:
        return []
    lines = []
    misleading = False
    for raw, display in layout_names.items():
        info = sdsc_spec.layouts.get(raw, {})
        dims = ", ".join(str(d) for d in info.get("dim_order", []))
        lines.append(
            f"  {display} = dim_order [{dims}], stick dim"
            f" {info.get('stick_dim_order')}, {info.get('stick_size')} elem/stick"
        )
        users = [
            op_spec.args[i]
            for i, a in enumerate(sdsc_spec.args)
            if a.layout == raw and i < len(op_spec.args)
        ]
        roles = {_role(a) for a in users}
        if len(roles) > 1 or (raw == "OUTPUT" and roles == {"input"}):
            misleading = True
        lines.append(f'       raw SDSC label "{raw}"')
    if misleading:
        lines.append(
            "       a layout label is an equivalence class shared by args with"
        )
        lines.append(
            '       identical layouts -- "OUTPUT" here is NOT an input/output role'
        )
    return lines


def _scale_decode(sdsc_spec) -> list:
    """Explain negative scales, which is where a reduction's meaning lives."""
    if sdsc_spec is None:
        return []
    seen: dict = {}
    for arg in sdsc_spec.args:
        for dim, scale in arg.scales.items():
            if isinstance(scale, int) and scale < 0:
                seen[(dim, scale)] = True
    lines = []
    for dim, scale in sorted(seen, key=lambda t: str(t[0])):
        if scale == -2:
            what = "reduced along the stick dim: sparse output, 1 elem per stick"
        else:
            what = "reduced dimension"
        lines.append(f"  {dim}={scale} -> {what}")
    return lines


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _op_block(op_spec, op_idx, n_ops, depth, verbose) -> list:
    pad = "  " * depth
    try:
        sdsc_spec, symbol_mapping = parse_op_spec(op_spec)
        failure = None
    except Exception as exc:  # never let explaining break the thing explained
        sdsc_spec, symbol_mapping, failure = None, {}, exc

    kind = "reduction" if op_spec.is_reduction else "pointwise"
    unit = getattr(sdsc_spec, "execution_unit", None)
    right = f"{kind}" + (f" - unit: {unit}" if unit else "")
    left = f"OP {op_idx}/{n_ops}  {op_spec.op}"
    lines = [left + " " * max(WIDTH - len(left) - len(right) - len(pad), 1) + right]

    if failure is not None:
        # Wrapped, not truncated: an exception message is unbounded.
        detail = f"parse_op_spec failed: {type(failure).__name__}: {failure}"
        lines += _wrapped("  !! ", detail, pad)
        lines.append("  !! showing OpSpec fields only; resolved view unavailable")

    layout_names = {}
    if sdsc_spec is not None:
        layout_names = {raw: f"L{i}" for i, raw in enumerate(sdsc_spec.layouts)}

    # Each section is isolated: they reach into resolved-view internals whose
    # shape drifts (#2286), and one breaking used to take the whole render with
    # it. One bad section should cost one section.
    #
    # Sections return [] when they have nothing to say. _sticks is the deliberate
    # exception: it lets a failed elems_per_stick() raise, because an absent
    # sticks block is indistinguishable from "this op has no sticks", where the
    # !! line names what broke. New sections should return [] unless they have
    # the same ambiguity.
    _cores = getattr(sdsc_spec, "num_cores", None)
    sections = (
        ("origin", lambda: _origin(op_spec, pad)),
        ("iteration", lambda: _iteration(op_spec, _cores, pad)),
        ("dim labels", lambda: _dim_labels(symbol_mapping, pad)),
        ("sticks", lambda: _sticks(op_spec, sdsc_spec, symbol_mapping, pad)),
        ("tiled", lambda: _tiled(op_spec, pad)),
        ("args", lambda: _args_table(op_spec, sdsc_spec, layout_names, pad)),
        ("layouts", lambda: _layout_footnotes(op_spec, sdsc_spec, layout_names)),
        ("scales", lambda: _scale_decode(sdsc_spec)),
    )
    for section_name, render_section in sections:
        try:
            lines += render_section()
        except Exception as exc:
            # Wrapped, not sliced: an AttributeError names the attribute at the
            # very end, which is what truncating to WIDTH cut away.
            detail = f"{section_name} section failed: {type(exc).__name__}: {exc}"
            lines += _wrapped("  !! ", detail, pad)

    if verbose and sdsc_spec is not None:
        lines.append("  raw SDSCSpec")
        lines += [f"    {ln}" for ln in str(sdsc_spec).splitlines()]

    return [pad + ln if ln else "" for ln in lines] + [""]


def _walk(specs, depth, counter, n_ops, verbose) -> list:
    lines = []
    for item in specs:
        if isinstance(item, LoopSpec):
            pad = "  " * depth
            lines.append(f"{pad}LOOP  trip count {_num(item.count)}")
            lines += _walk(item.body, depth + 1, counter, n_ops, verbose)
        elif isinstance(item, OpSpec):
            lines += _op_block(item, counter[0], n_ops, depth, verbose)
            counter[0] += 1
        elif isinstance(item, UnimplementedOp):
            lines.append(f"{'  ' * depth}UNIMPLEMENTED  {item.op}")
            lines.append("")
    return lines


def render(specs, *, kernel_name=None, args=None, pool_size=0, verbose=False) -> str:
    """Return a human-readable explanation of an OpSpec list.

    ``specs`` is a list of OpSpec / LoopSpec / UnimplementedOp.  ``args``
    optionally carries what was observed at launch (objects with ``shape``,
    ``dtype``, ``device_size``, ``device_dtype_name``).
    """
    specs = list(specs)
    n_ops = sum(1 for _ in _iter_op_specs(specs))
    lines = _title(kernel_name, specs, args, pool_size)
    lines += _kernel_args_section(specs, args, pool_size)
    # 1-based: "OP 0/7" against an "of 7" denominator reads as an off-by-one.
    lines += _walk(specs, 0, [1], n_ops, verbose)
    return "\n".join(lines).rstrip() + "\n"


def render_comment_block(specs, **kwargs) -> str:
    """:func:`render` as a Python comment block, for embedding in a script."""
    return "\n".join(
        ("# " + line).rstrip() for line in render(specs, **kwargs).splitlines()
    )
