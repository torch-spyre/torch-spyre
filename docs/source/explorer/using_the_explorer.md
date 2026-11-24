# Using the knowledge graph explorer

The {doc}`index` is a generated view of the Torch-Spyre codebase. Sphinx
rebuilds it from the source tree on every documentation build, so it
matches the code at the commit it was built from. Every node records the
file and line it came from, and the panel below the graph links to that
location.

This page describes what you can do with it. The first half is for people
running models on Spyre. The second is for people working on the backend.

## What it is for

Torch-Spyre consists of more than a hundred Python modules and a C++ runtime. Most of
the useful facts are relationships: a PyTorch op is handled by a lowering,
that lowering is part of a pass group, and the pass group reads an
environment variable. These relationships are spread across decorators,
class definitions, and import statements, so reading them one file at a
time is slow. A diagram maintained by hand goes out of date as soon as
someone merges a PR.

The explorer avoids that. It reads the structure directly from the code
with Python's `ast` module at build time, so the graph shows the current
code rather than a description of it.

## How to read each view

Every view uses the same grammar: nodes are things, edges are the relationship
between them, and each edge is labeled with its relationship name. A node's
color encodes its type, and the legend above the graph lists the colors for the
current view. The four views below each have one worked example.

### Operations

:::{figure} ../_static/images/decomposition-chain.svg
:alt: aten.bitwise_and has a decomposed_by edge to the bitwise_and decomposition, which has decomposes_into edges to aten.logical_and, aten.bitwise_not, and aten.bitwise_or.
:width: 100%
:align: center

The decomposition chain for `aten.bitwise_and`. The `decomposed_by` edge
reaches the decomposition function, and the `decomposes_into` edges reach the
ops it rewrites into.
:::

Start at the op node `aten.bitwise_and`, follow its `decomposed_by` edge to the
decomposition `bitwise_and`, then follow the `decomposes_into` edges to see that
it rewrites into `aten.logical_and`, `aten.bitwise_not`, and `aten.bitwise_or`.
A different op might instead carry a `lowered_by` edge to a lowering, a
`falls_back_to` edge (dashed) to a CPU fallback, or an `eager_via` edge to an
eager kernel; the outgoing edge is how the backend handles that op. In the live
graph the `decomposes_into` edges are drawn in green, so the expansion stands
apart from the `decomposed_by` link.

This answers two questions for a user at once: whether `bitwise_and` runs on the
accelerator and what it actually becomes, which matters when a result or a
performance number is surprising. For a developer adding a related op, it shows
the existing rewrite to follow.

### Lowering Paths

Select an op to trace its path to the device. For `aten.gelu.default` the
trace is `aten.gelu.default` to the decomposition `spyre_gelu`, to
`spyre.gelu`, to the lowering `lower_gelu`, to the Inductor op `ops.gelu`, to
the SDSC opfunc `gelufwd`. The information panel lists the opfuncs the selected
node reaches.

The edges have these meanings:

| Edge | Meaning | Read from |
|---|---|---|
| `has_overload` | An op name to one of its overloads (`aten.bmm` to `aten.bmm.default`) | op ids |
| `decomposed_by`, `decomposes_into` | The decomposition of an op, and the ops it produces | `decompositions.py` |
| `lowered_by` | The Spyre lowering registered for an op | `lowering.py` |
| `delegates_to` | A Spyre lowering that calls another Spyre lowering | `lowering.py` |
| `emits` | An Inductor op or SDSC opfunc a lowering produces | `lowering.py` |
| `maps_to` | The SDSC opfunc an Inductor op becomes | `SpyreOpFuncs` in `spyre_kernel.py` |
| `inductor_default` | An aten op with no Spyre path, to the Inductor op of the same name | op names |
| `rewritten_by` | An FX graph pass that matches or creates the op | `temp_passes.py`, `passes.py` |

Dashed edges are inferred by name rather than read from a call. An
`inductor_default` edge assumes that Inductor's default lowering of `aten.X`
emits `ops.X`. An inferred `emits` edge comes from a reference to
`lowering.X`, Inductor's lowering of that name. A reduction opfunc can also
receive a `nonstick` suffix at code generation (`_get_op_func` in
`codegen/superdsc.py`), which the graph does not show.

For a developer, the trace shows which lowering and which Inductor op an
operation depends on before a change to either. For a user, an op that reaches
no SDSC opfunc either moves data only (a copy, `cat`, or a collective), runs
through an eager kernel or a CPU fallback, or has a path the extractor cannot
read; check the Operations view and the source link.

### Compiler Passes

A pass-group node such as `CustomPreSchedulingPasses` has a `contains_pass` edge
to each pass it runs, for example `splice_while_loops`, `deadcode_elimination`,
and `propagate_named_dims`, laid out top to bottom in execution order.

The group is the pipeline and each child is one pass; top to bottom is the order
they run. This is mostly a developer view. Use it to see where a transformation
happens and what runs before it, before adding or reordering a pass.

### Architecture

Class nodes carry `inherits_from` edges to their base classes, for example
`AllReduceKernel` inherits from `SpyreCollectiveKernel`, which in turn
inherits from the Inductor class `ir._CollectiveKernel`. Module nodes carry `imports` edges to the
modules they import, and dataclass nodes carry `contains_field` edges to the
dataclasses they reference.

Follow `inherits_from` up to find the base class a change would affect, and
trace `imports` inbound to find the modules that depend on the one you are
changing. This is the view for learning an unfamiliar subsystem and for judging
the reach of a refactor before starting it.

### Configuration

An environment-variable node such as `LX_PLANNING` is connected by a
`reads_env` edge to each module that reads it, for example `_inductor.config`.

Pick the variable, follow `reads_env` to the module, then open the source link
to see the accepted values and the default. For a user this is the fastest way
to find the knob that controls a behavior; for a developer it shows which
modules depend on a variable before renaming or retiring it.

The sections below put these views to work for specific user and developer
tasks.

## For users

If you run models through `torch.compile`, the two most useful views are
Operations and Configuration.

### Checking whether an op is supported

Open the Operations view and search for the op, for example `mm` or
`softmax`. The node color and its outgoing edge show how the backend
handles it:

- Decomposition: the op is rewritten into smaller ops before it runs on
  the hardware.
- Lowering: the op maps to a Spyre kernel during compilation.
- Custom op: a hand-written Spyre operator implements it.
- CPU fallback (dashed red edge): the op runs on the host instead of the
  accelerator. Its inputs are copied to the host and its result is copied
  back to the device, so check this view when a model is slower than you
  expect.
- Eager kernel: the op has a direct implementation that also works outside
  `torch.compile`.

If an op is not in the graph, the backend has no registration for it. That
is a concrete answer, and a useful detail to include when you file an
issue.

### Finding the variable that controls a behavior

The Configuration view shows every environment variable the code reads
(`SENCORES`, `LX_PLANNING`, `TORCH_SPYRE_DEBUG`, and the rest) and the
module that reads it. To change runtime or compilation behavior, use this
view to find the variable, then follow the source link to the code that
reads it, where you can see the accepted values and the default.

### Sharing a node in a bug report

Selecting a node writes a link into the page URL. You can paste that link,
for example `.../explorer/index.html#ops/op::aten.embedding.default`, into an issue,
and it opens on that node and view.

## For developers

For contributors, the explorer helps with two things: learning an
unfamiliar area, and checking what a change affects.

### Exploring an unfamiliar subsystem

The Architecture view shows module dependencies and class inheritance for
the `torch_spyre` package. When you start working in a subsystem you have
not seen before, find the class you were pointed to, turn on Focus to hide
everything else, and read its base classes and the modules that import it.
That is usually enough to decide where to set a breakpoint.

### Opening the source

Every node with a single definition site is a link. Click it to see the
file and line in the panel, or double-click to open that definition on
GitHub in a new tab. The link uses the commit the graph was built from, so
the line number stays correct even after the code moves.

### Checking what a change affects

Two relationships are worth tracing before you refactor:

- `imports` edges (Architecture view) show which modules depend on the one
  you are changing. Focus the module node to see the inbound edges.
- `inherits_from` edges show the subclasses a base-class change affects.

This does not replace running the tests, but it shows you where to look.

### Learning the op registration patterns

When you add an operation, the Operations view shows the ops already
implemented with each pattern. Filter to the op family you are working on
and open the existing decomposition, lowering, or custom-op definitions.
Use this with the {doc}`../compiler/adding_operations` guide: the guide
explains the registration patterns, and the graph shows every op that uses
each one, with a link to its code.

### Viewing the compiler pipeline

The Compiler Passes view shows each `Custom*Passes` group and the pass
functions it runs, in pipeline order from top to bottom. Use it to see
where a transformation happens and what runs before it, before you add or
reorder a pass.

## Accuracy and limits

Three properties are worth relying on:

- The graph is rebuilt on every documentation build. A Sphinx extension
  runs `docs/source/_ext/extract_graph.py` and writes a fresh
  `graph.json`, so there is no stored copy that can go out of date.
- Extraction is syntactic. It parses the source with `ast` and imports
  nothing, so it runs without Spyre hardware or the C++ extensions and is
  not affected by import-time side effects.
- Source links are commit-pinned. They use the commit the graph was built
  from, and use the default branch only when the build runs outside a git
  checkout.

The graph covers registrations, class hierarchies, imports, dataclass
fields, environment-variable reads, and the lowering paths described above. It
does not trace runtime call graphs or data flow, and it only parses the files
listed in `build_graph()`. A lowering path is read from the lowering's own body
and from module-level helpers up to three calls deep; an op chosen at run time
(from a variable or a dictionary) does not appear. If a new registration pattern or source file is not
showing up, the extractor needs updating. The {doc}`../contributing/index`
notes and the `check-docs` checklist explain how.

## See also

- {doc}`index`: the explorer, with the navigation reference.
- {doc}`../compiler/adding_operations`: the op registration patterns the
  Operations view lists.
- {doc}`../compiler/architecture`: the compilation pipeline the Compiler
  Passes view shows.
