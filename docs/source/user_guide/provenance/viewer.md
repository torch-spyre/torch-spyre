# Spyre provenance viewer

The provenance viewer answers one focused question:

> Which recorded source and compiler operations contributed to this saved
> Spyre profiler event?

The generator validates one `spyre_provenance.json` sidecar, optionally pairs
individual activities from one Kineto Chrome trace, and writes a deterministic
self-contained HTML file. The page has two selectors, compact event facts, and
six linked evidence panels.

## Quick start

### 1. Capture the sidecar and trace

Choose an explicit sidecar path so the trace and its matching artifact are easy
to archive together:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
INDUCTOR_PROVENANCE=1 \
TORCH_SPYRE_PROVENANCE_PATH=/path/to/capture/spyre_provenance.json \
python my_profiled_model.py
```

`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` ensures this run compiles instead of
silently reusing an older cache entry. `INDUCTOR_PROVENANCE=1` asks Inductor
to retain its version 2.0 node mappings. Torch-Spyre attaches those mappings to
the sidecar's compile records. `TORCH_SPYRE_PROVENANCE_PATH` controls where
the sidecar is published; without it, publication uses the run-scoped Inductor
debug directory.

Export a Chrome trace from the same profiled run:

```python
import torch
from torch.profiler import ProfilerActivity, profile

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
) as prof:
    compiled_model(example_input)

prof.export_chrome_trace("/path/to/capture/spyre_trace.json")
```

The trace is optional for the viewer, but it supplies individual runtime
occurrences, timestamps, durations, and JobPlan command suffixes.

### 2. Generate the HTML

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python -m torch_spyre.provenance_viewer \
  /path/to/capture/spyre_provenance.json \
  --kineto-trace /path/to/capture/spyre_trace.json \
  --output /path/to/capture/spyre_provenance_viewer.html
```

Backend autoload is disabled because this command reads saved JSON only. It
does not need to initialize the Spyre backend.

For sidecar-only inspection, omit `--kineto-trace`:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python -m torch_spyre.provenance_viewer \
  /path/to/capture/spyre_provenance.json \
  --output /path/to/capture/spyre_provenance_viewer.html
```

Open the output directly in a browser. No local web server is required.

The viewer does not accept a `tlparse` directory or raw compiler artifacts.
The sidecar already carries the structured source, ATen, handle, compile,
alias, and upstream mapping evidence required by the six panels. The existing
`tlparse --inductor-provenance` workflow remains useful when full generated
FX or wrapper text is needed, but it is a separate view.

## Why export standalone HTML?

The HTML freezes the validated evidence used for the analysis, including the
presentation logic. This makes the result reproducible, easy to archive, and
viewable without a device or backend environment. It is also convenient to
share after reviewing it for sensitive source paths and model structure.

Regenerate the HTML when either input changes. The embedded input paths,
sizes, and SHA-256 digests make accidental artifact mixing diagnosable.

## Selectors and event facts

The **Profiler event** selector chooses one persisted finalized-bundle
identity. Its event-name base has this form:

```text
spyre_kernel_v1_<summary>_<provenance-key>
```

The 16-character provenance key is the join into
`kernelIdentities`. The readable summary is descriptive and is not used to
resolve evidence. It contains only sorted, deduplicated ATen packet names, so
many structurally different bundles can share the same readable
`fused_<operations>` summary. Their keys differ because the key fingerprints
the complete finalized `OpSpec | LoopSpec` tree, including the directly
attached handle at every OpSpec position.

In this viewer, a profiler event is that unique persisted event-name base and
bundle identity. It is not one execution of the bundle.

The **Runtime occurrence** selector chooses one concrete trace activity.
Repeated activities stay separate even when they have the same event name and
bundle identity.

Runtime occurrence remains a selector rather than a static header field
because each execution can have a different timestamp, duration, correlation
record, or `#<step>` suffix. Choosing one occurrence updates those facts while
leaving the bundle evidence unchanged. A static field would have to discard or
arbitrarily choose among repeated executions.

The event facts distinguish these scopes:

- **Exact observed event name** is the trace name, or the persisted base name
  in sidecar-only mode.
- **Timestamp** and **duration** belong to the selected trace activity.
- **JobPlan step** is the `#<step>` suffix: a static command index in the
  finalized backend plan. It is not a token, generation iteration, stage, or
  OpSpec index.
- **Compile candidates** counts retained `kernelOccurrences` with the
  selected identity key. A compile candidate is not a runtime occurrence or a
  JobPlan step, and the viewer does not choose one when several are valid.

A device event represents one `ComputeOnDevice` activity. One complete
JobPlan execution can produce several device events with different suffixes,
and repeated executions can produce the same suffix repeatedly. Correlation
IDs can pair runtime activities and can support a later stage analysis, but
they do not map an OpSpec to a proprietary backend command.

## How rows are constructed

The resolution path is:

```text
trace activity
  -> native args.provenance_key and/or key in the event name
  -> kernelIdentities[key]
  -> direct handle IDs and recursive fused_from handles
  -> source, ATen, lower-IR, and SpecPath evidence
  -> matching kernelOccurrences
  -> compile-scoped aliases and upstream v2 FX mappings
```

When both native and name-derived keys exist, they must agree. Native
`debug_handles` are compared with the sidecar's direct handle list but never
replace it. Carrier conflicts are errors; malformed optional trace input and
missing optional evidence are reported without invalidating the sidecar view.

Each panel has exactly one clickable row unit. Runtime and compile facts stay in
the selected-event facts; panel headers define their row unit and show compact
counts or coverage:

| Panel | One row represents |
| --- | --- |
| Python source locations | One unique structured `SourceLoc` |
| Recorded ATen identities | One unique `DebugHandle.aten_op` value |
| FX pre-grad nodes | One `(compile ID, pre-grad node name)` |
| FX post-grad nodes | One `(compile ID, post-grad node name)` |
| Recorded lower-IR lineage by handle | One handle and all its recorded lineage |
| Direct OpSpec bindings | One ordered `(position, SpecPath, handle ID)` binding |

## Panel meanings

### Python source locations

This panel deduplicates equal structured source ranges while retaining all
contributing handle references and their count. A `SourceLoc` contains a file,
start line and column, and optional end line and column.

It is a location, not a source snapshot. The viewer does not read local source
files or embed literal Python text because a saved path may be missing, changed,
private, or from another machine. Stack traces locate frames, and generated FX
text describes compiler graphs; neither is an authoritative copy of the
original source. Adding portable source text would require a separately
reviewed bounded snapshot input, not a change to sidecar v1.

### Recorded ATen identities

This panel shows unique operation identities recorded on recursive debug
handles. Equal display values share a row, but contributing handle
multiplicity remains visible. No operation is promoted as representative of a
fused bundle.

ATen and post-grad FX are not redundant. For example, one recorded
`aten.linear.default` operation can correspond to pre-grad `linear` and then
decompose into post-grad `permute`, `mm`, and `add` nodes.

### FX pre-grad and post-grad nodes

FX rows are compile-scoped because graph names can be reused or change between
compilations. The sidecar stores PyTorch Inductor mapping version 2.0, where:

- `cppCodeToPost` maps a registered compiler alias to post-grad node names;
- `postToPre` maps a post-grad node to pre-grad nodes;
- `preToPost` is the reverse pre-grad relationship; and
- `postToCppCode` is the reverse alias relationship.

For example, an alias can map through `cppCodeToPost` to post-grad `add`,
then through `postToPre` to pre-grad `linear`. The alias-to-post
relationship can be exact even when the sidecar only supports bundle-level,
derived handle attribution for that node.

The Spyre provenance viewer intentionally does not embed generated FX definitions or
raw compiler text. Use the saved `tlparse` output for full graph text.

### Recorded lower-IR lineage by handle

Each row contains one debug handle's complete ordered `ir_chain` and
`transform_history`. Transformation details are subordinate to the handle;
they are not separate peer rows.

The current producer does not record stable typed entities for every lower-IR
stage or exact old-to-new rewrite edges. This panel is therefore recorded
lineage, not a LoopLevelIR graph and not proof of a fabricated edge sequence.

### Direct OpSpec bindings

Each row is one ordered `specHandleBinding`, including its binding position,
SpecPath, and directly attached handle. A SpecPath is a nonempty zero-based
path through the finalized `OpSpec | LoopSpec` tree:

- `[0]` means the first top-level OpSpec;
- `[1, 0]` means the first OpSpec inside the second top-level LoopSpec.

A SpecPath is not an OpSpec ID, kind, runtime command, or proprietary backend
program location. Two paths can bind the same handle and remain distinct rows.

The `#<step>` runtime command remains attributed to the complete finalized
bundle. The current backend interface does not export a command-to-OpSpec
subset, so the viewer must not imply one.

## Evidence states and interaction

**Exact** means a validated field or mapping directly supports the
relationship. **Derived** means the row follows a declared mapping or a
documented bundle-level fallback. For example, pre-grad `linear` reached by
`alias -> post-grad add -> postToPre -> linear` is derived.

**Multiple candidates** is separate from evidence strength. It means two or
more valid compile occurrences remain and the retained evidence cannot select
one. A row can therefore be both derived and associated with multiple
candidates. A relation is not ambiguous merely because it contains an
explicitly recorded set of several members.

Clicking one evidence row highlights every related row and dims unrelated rows.
The clicked row, keyboard focus, clicked panel scroll, and page position stay
fixed. Other panels center their first related row while retaining every
related highlight. Click the focused row again to restore the full union.
Changing the profiler event also clears row focus.

Headers, summaries, and empty states are not interactive.

## Sidecar-only mode and limits

Without a trace, the viewer can still show every persisted identity and all
sidecar evidence. Runtime timestamps, durations, command suffixes, and
trace-to-sidecar pairing coverage are unavailable. FX panels can still be
populated when the sidecar contains the accepted upstream mappings; they do
not require raw graph files.

An empty panel means the producer did not retain that evidence. The viewer does
not infer missing source, FX nodes, typed lower-IR edges, scheduler decisions,
or command-to-OpSpec relationships. Provenance can identify a recorded
divergence boundary, but it cannot explain a fusion or split decision that no
producer recorded.

The sidecar and optional trace are each bounded to 256 MiB. Each panel displays
at most 10,000 deterministically ordered evidence rows, and the export retains
at most the earliest 10,000 resolved runtime occurrences by trace-event index.
Panel headers and the run summary show displayed and total counts when either
limit is reached. The presentation becomes ``partial`` and records explicit
truncation diagnostics; the validated sidecar and trace remain the complete
evidence sources.

Optional trace failures leave the validated sidecar view available with
diagnostics. Invalid or unsupported sidecars fail generation because showing
unvalidated attribution would be misleading.

The export can reveal file paths, line numbers, operation names, and compiler
structure. Review it before sharing.
