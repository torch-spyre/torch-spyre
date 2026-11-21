---
name: lx-allocation-maps
description: "Capture CP-SAT LX buffer allocations for compiled Spyre graphs, validate and plot them, and author an agent-only Markdown report. Use when collecting or visualizing LX allocation watermarks."
---

# LX Allocation Maps

Capture static CP-SAT buffer placement records from any model compile, parse the
JSONL into graph summaries, and generate per-graph allocation maps and operation
ledgers. Capture, parse, and plot are standalone CLIs under
`tools/lx_planning/allocation_maps/`; users can run them directly without an
agent. Only the final experiment Markdown interpretation is agent-authored.

## Important semantics

- A capture record is one allocator solve / compiled graph, **not** one runtime
  invocation. This skill covers static allocation capture and is not the place to
  collect runtime invocation counts or profiler traces. Do not infer call counts
  from allocation records.
- Plotted lifetimes and addresses are the frontend planner's static solve. They
  do not show emitted bundle instruction order, runtime occupancy, HBM traffic,
  or a model-wide peak across graphs.
- The plotted watermark is the highest resident buffer end address at an
  allocator tick, not the sum of resident buffer sizes. New CP-SAT captures
  use `solver_slot` time: graph operation `i` writes in slot `2*i`, and a
  coarse-tiling cut's optional copy runs at `2*i+1`. The plotted half-open
  lifetime `[start_time, end_time)` is the solver-selected placement box; it
  may differ from the buffer's original operation-index lifetime when coarse
  tiling changes the schedule. `uses` records graph access at `2*i`, not copy
  instructions. Older captures used operation index `i`; fixed-division
  relayout captures can use operation `2*i+1` with a transfer at `2*i`.
  Always read `allocation_time_unit` and `timeline[].allocation_tick`.
  The first use of a computed buffer is its producing write, not necessarily
  a read. A plot dot is one recorded access, not every emitted load.
- Capture distinguishes host tensor shape, dtype, and strides from physical
  `SpyreTensorLayout` where available: `device_size`, `stride_map`,
  `device_dtype`, `element_arrangement`, and `valid_elements`. Unknown/symbolic
  values remain explicit; a synthetic relayout destination does not inherit
  its source's physical layout. `logical_size_bytes` is a legacy field holding
  the solver's total buffer footprint, **not** `numel * dtype.itemsize`; use
  `logical_tensor_bytes` for host logical bytes when available. The captured
  `reserved_size_bytes` is the selected division's candidate per-core LX
  footprint; it represents actual LX reservation only when an LX address was
  assigned.
- `capacity_bytes` is the frontend allocator's available range for that compile
  configuration, not necessarily the physical 2 MiB LX size.
- These scripts instrument the CP-SAT solve's slot-span extraction and
  `CoOptimizingAllocator._post_solve`. They capture the joint-placement
  allocator, not placement-only `ScratchpadAllocator` solves.
- The hook gathers FX source metadata and writes JSON during compilation. That
  adds compile-time and disk overhead; **never include hook-enabled timings in a
  latency comparison**. For timing, use a separate hook-free run with matching
  compiled cache policy.
- The capture hook is loaded only for the child command: `capture.py` prepends its
  own directory to `PYTHONPATH`. Do not install `sitecustomize.py` globally or
  in a model environment.
- Capture reads AOT context from PyTorch internals and imports Spyre Inductor
  internals. Use the same Python environment and checkout used to compile.

## Manual CLI workflow

Run from the torch-spyre checkout. Set `TOOLS` to the checked-out tool
directory, and choose an experiment directory under the repository. Preserve
all outputs there; do not use `/tmp` for experiment data.

```bash
TOOLS="$PWD/tools/lx_planning/allocation_maps"
RUN="$PWD/agentic-experiments/$(date -u +%Y%m%dT%H%M%SZ)-allocation-map"
mkdir -p "$RUN"
```

### 1. Capture while compiling / running the target

The target command must actually compile the graphs of interest. For a Python
script that runs a compiled model:

```bash
python "$TOOLS/capture.py" --output-dir "$RUN/capture" -- \
  python path/to/run_model.py --your-model-args
```

Pass the normal compile configuration to the child process through exported
environment variables. For CP-SAT joint placement, the defaults are
`LAYOUT_SOLVER=cpsat`, `CO_OPTIMIZING_LX_PLANNING=1`, and `LX_PLANNING=1`;
verify them in `capture/run.json`. Require the child to use the target Python
runtime and repository. Use a distinct compiler cache when the allocation
configuration changes; preserve it under the experiment directory if graph
matching is needed. Spyre supports one device process at a time: run jobs
serially.

For vLLM, launch from its checkout using that checkout's environment and wrap
the actual benchmark command. Example one-request capture:

```bash
cd /path/to/spyre-inference
TOOLS=/path/to/torch-spyre/tools/lx_planning/allocation_maps
RUN="$PWD/agentic-experiments/<new-timestamp>-allocation-map"
mkdir -p "$RUN"
export TORCHINDUCTOR_CACHE_DIR="$RUN/torchinductor-cache"
env SPYRE_NUM_CPUS=8 \
  uv run --no-sync python "$TOOLS/capture.py" \
    --output-dir "$RUN/capture" -- \
    uv run --no-sync vllm bench latency \
      --model google/gemma-4-26B-A4B-it \
      --input-len 1024 --output-len 64 --batch-size 1 \
      --max-model-len 1088 --num-iters-warmup 1 --num-iters 1
```

Adjust model and shapes to the user's workload. This one-warmup/one-measurement
example is for capture smoke only, not a stable latency benchmark. Record the
requested and resolved concurrency/token-budget settings from vLLM logs.



### Startup warnings, model support and run issues
Preserve and classify the full `capture/command.log`; do not suppress warnings or
assume a warning means capture failed. vLLM can warn that a model/shape is not
in the Spyre model registry while proceeding; report it as unvalidated even if
the run succeeds. Scheduler warnings can describe a requested
`max_num_batched_tokens` later overridden by the Spyre platform; record both
requested and resolved values. Triton/CPU-Triton fallback and Hugging Face Hub
authentication/rate-limit warnings are distinct from CP-SAT capture status;
record them and state whether they affected completion. Successful execution is
not an independent numerical correctness check.

Keep an issue log with the step/command, observed message or failure, diagnosis,
resolution/workaround, and proposed skill improvement. Record compiler warnings
(including work-division/provenance warnings) rather than treating captured
allocations as correctness evidence.
For repeatable measurements, pin the compile cache and solver settings explicitly
before launching the CLI; for example:

```bash
export LAYOUT_SOLVER=cpsat CO_OPTIMIZING_LX_PLANNING=1 LX_PLANNING=1
export DXP_LX_FRAC_AVAIL=0.45 SENCORES=32 SPYRE_NUM_CPUS=8
export TORCHINDUCTOR_CACHE_DIR="$RUN/torchinductor-cache"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="$RUN/torch_compile_debug"
```

Pass the exact interpreter and module path expected by the model:

```bash
python "$TOOLS/capture.py" --output-dir "$RUN/capture" -- \
  /path/to/python /path/to/run_model.py --your-model-args
```

Use a new empty `--output-dir` per capture run so records from multiple
processes/configurations cannot be mistaken for one capture set.

`capture.py` writes:

- `capture/allocations.<pid>.jsonl`: one complete JSON object per allocator
  solve, with graph identity, solver, per-buffer host and Spyre layout metadata
  where available, producer/consumer graph-op indices, operation origins,
  allocator-tick timeline, address watermark, capacity, and invariant flags;
- `capture/processes.<pid>.jsonl`: confirms which Python processes loaded the
  hook; a `capture_failed` entry with traceback flags an omitted graph record
  without interrupting compilation;
- `capture/run.json`: command, working directory, Python executable and key
  compile/allocation environment values;
- `capture/command.log`: target stdout and stderr.

If the script never triggers `CoOptimizingAllocator` or does not compile, no
allocation JSONL will be produced. Check the command log and process marker;
do not report an empty capture as a valid zero-allocation result.

### 2. Parse and validate the raw records

```bash
python "$TOOLS/parse.py" "$RUN/capture/allocations.<pid>.jsonl" \
  --output-dir "$RUN/parsed"
```

The command validates JSON/schema, solver identity, graph operation counts,
resident-buffer spans, global watermark, and per-operation watermarks at their
allocator ticks. A transfer-only tick can set a global peak above the maximum
sampled at graph operations. A valid run exits zero and writes:

- `parsed/summary.json`: one row per captured graph with operation/source
  coverage, resident/nonresident counts (nonresident does not imply HBM
  placement), watermark, capacity and fraction;
- `parsed/graphs.tsv`: compact spreadsheet-friendly view.

Invalid records produce an error and nonzero exit; fix or explain them before
plotting or reporting. `plot.py --record N` selects a capture record by its
zero-based line order in the JSONL.

Select the largest recorded watermark from the parsed summary rather than
guessing from the order in the command's output:

```bash
python -c 'import json,sys; rows=json.load(open(sys.argv[1]))["records"]; print(max(rows,key=lambda row:row["global_watermark_bytes"])["capture_record_index"])' "$RUN/parsed/summary.json"
```

### 3. Plot every graph or select individual records

Render every valid record:

```bash
python "$TOOLS/plot.py" "$RUN/capture/allocations.<pid>.jsonl" \
  --output-dir "$RUN/plots"
```

Render one record with `--record N`; repeat the option in the **same invocation**
to render a subset into one indexed output directory:

```bash
python "$TOOLS/plot.py" "$RUN/capture/allocations.<pid>.jsonl" \
  --output-dir "$RUN/plots" --record 7 --record 12
```

The plotter uses only Python's standard library plus the project's `regex`
dependency; it does not require Matplotlib. Each rendered graph produces an
interactive offline HTML allocation map, HTML operation ledger, TSV operation
ledger, and JSON graph summary.
`plots/manifest.json` indexes the records selected in that invocation; a later
invocation using the same output directory replaces the manifest even though
previously rendered files remain. Use one invocation for the full selection or
separate output directories. The plotter uses the parser's CP-SAT, watermark,
timeline, and capacity validation before rendering. Blue
rectangles are ordinary LX-resident buffers; orange marks ordinary buffers
reaching the highest LX address, while teal identifies synthetic relayout
buffers (with an orange outline if at the highest address). The red line is
the highest live LX address at each tick; the dashed line is planner capacity.
Both plots display numeric allocator ticks on aligned x-axes. The operation
strip uses a separate repeating palette keyed to operation type. The operation
ledger includes active resident buffers, operation watermark, FX source files,
code, and operation names where available.

#### Inspecting tensor metadata and the interactive map

The JSONL buffer entries preserve the legacy `logical_size_bytes` field as the
solver buffer size, and include `allocated_size_bytes` with the same meaning.
`logical_tensor_bytes` is the host shape × dtype itemsize when concretely
available; `reserved_size_bytes` is the per-core LX footprint of the selected
division. Do not equate these three numbers. The details view shows one
**Reason** when spill and residency reasons match; distinct values are labeled
**Solver outcome** and **LX eligibility**. Host shape/dtype/strides and
structured Spyre layout may be unavailable. These are **planner pre-commit**
layouts, not guaranteed to be the final emitted bundle layout. An explicit
unknown is not proof of any particular layout. Synthetic relayout copies carry
source metadata separately and do not claim their source as destination layout.

Open the interactive HTML map locally: operation/tick position is the x-axis,
LX address is the y-axis. `Fit full graph` fits the complete timeline into the
map viewport. The slider and `+`/`−` buttons preserve the viewport center; with
the pointer over the plot, **W/S** zoom in/out around the pointer's horizontal
position and **A/D** pan left/right. Both plots are displayed at full height
and scroll horizontally only; there is no vertical scrollbar.
The separate nonresident **graph-buffer** plot shares the same allocator-tick
x-axis and zoom: up-arrows mark producer operations and down-arrows mark
consumer operations when operation metadata exists. Otherwise, small dots mark
recorded reference/use ticks; a larger outlined dot marks lifetime start. The thin
line spans the half-open lifetime; the y-axis is log₂ solver footprint bytes. A
hollow point in the bottom gutter means the solver footprint is unknown or
nonpositive; it does not imply a zero-sized tensor or establish HBM allocation.
Select an arrow, use dot, or outlined lifetime-start point to inspect its buffer.
Click/focus the plot pane to enable keyboard controls. Modified browser
shortcuts and typing in controls are not intercepted.
Buffers with a known empty `consumer_op_indices` list are omitted from the
nonresident plot and table because the graph records no consuming op for them.
The omitted count is shown; entries remain in raw capture. Unknown (`null`) consumer
metadata is retained rather than treated as an empty list.
`RelayoutCopyBuffer` candidates: a copy without an LX address means its
LX-to-LX shuffle was **not selected**, not that a shuffle spilled to HBM.
These rejected candidates remain in the raw JSONL for solver analysis but
are excluded from the nonresident plot, table and operation-linked lists;
the viewer reports their omitted count separately. This is distinct from the
known-empty-consumer filter above. Selected LX relayout copies are shown by
default in the LX map, with a checkbox to hide them; they are solver planning
entries, not additional graph operations.

The `-1` solver-size sentinel means the allocator could not compute an LX
placement footprint (for example, a mutation alias or a buffer without a fixed
Spyre device layout); it is not a measurement of the buffer's logical or
physical size. `logical_tensor_bytes` may still be known, and a captured
`device_size` may be present for some entries. A nonresident planner record
means only that this solve did not assign an LX address. Downstream codegen
later assigns ordinary HBM arguments or an `hbm_pool` offset to eligible tensor
arguments, but these pre-commit records do not capture that final allocation
key/address. Therefore neither a positive size nor an unsized record alone
proves a real HBM allocation; inspect final bundle `allocation` metadata for
HBM/HBM-pool placement. Neither establishes measured runtime HBM traffic.
The searchable/filterable companion table is sorted lexicographically by its
operation-links text (then by buffer name) and has a bounded-height region
with independent vertical scrolling and sticky column headers. The details panel
can be collapsed entirely to free the column; a compact `Show selected details`
button reopens it. When open, it is capped at the plot panel's height; long
selections scroll vertically inside the details text without stretching the plots
row. Detail fields with unknown or null values are hidden.
Generated HTML has pre-rendered charts, log-size y-labels, numeric x-axes
and table rows even when local JavaScript is disabled; hover/focus tooltips,
zoom and filtering require JavaScript.
Use the operation ledger for FX source context and JSONL for downstream analysis. Both plots show static frontend planning only:
nonresident points do not prove measured HBM traffic.
Older captures without added metadata remain inspectable with unknown
fields shown as unavailable.


### 4. Agent-authored Markdown report — agent-dependent

Do not add a Markdown-report generator to the standalone CLIs. When the user
asks for a written report, inspect `parsed/summary.json`, the raw capture JSONL,
`capture/run.json`, and relevant compile logs. Write `report.md` manually,
recording:

- workload, shapes/dtypes, compile controls, device/runtime, repository
  revisions, commands, and number of capture runs;
- captured graph inventory and operation/source coverage;
- highest watermarks and capacities, with links to the interactive HTML, ledger,
  and TSV;
- CP-SAT and invariant validation status;
- interpretation limits: static planner address extents, no inference of runtime
  LX occupancy, final instruction schedule, dynamic transactions, or aggregate
  live peak across graphs.

Never treat plot generation or non-empty captures as correctness proof. State
whether numerical correctness was independently checked and with what method.
