# Trace Analysis

**Stack:** torch-spyre (new, Inductor-based).

Traces written by `torch.profiler` (see [PyTorch Profiler](pytorch_profiler.md))
are Chrome-trace JSON files. They open in any of three viewers:

- **PyTorch Profiler TensorBoard Plugin** (preferred): AIU-aware
  views on top of the raw trace. Source and install instructions:
  <https://github.com/IBM/kineto-spyre/tree/main/tb_plugin>
- **Perfetto**: drag and drop the JSON onto <https://ui.perfetto.dev/>
- **Chrome Trace Viewer**: `chrome://tracing` in Chrome

## Quick start

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    record_shapes=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/my_run"),
) as prof:
    output = model(inputs)
```

`tensorboard_trace_handler` writes the trace to `./logs/my_run/` for
TensorBoard. For Perfetto / Chrome, call `prof.export_chrome_trace("trace.json")`
and load the file directly.

## What the trace contains

The exported file is [Chrome Trace Event Format][ctf] JSON: a top-level
object with a `traceEvents` array. Each entry is one timed span. A
trimmed excerpt looks like this:

```json
{
  "traceEvents": [
    {"ph": "X", "cat": "user_annotation", "name": "iteration_0",
     "pid": 3153, "tid": 3153, "ts": 1707279511678052, "dur": 1800},
    {"ph": "X", "cat": "cpu_op", "name": "aten::mm",
     "pid": 3153, "tid": 3153, "ts": 1707279511678215, "dur": 40},
    {"ph": "X", "cat": "kernel", "name": "spyre_matmul",
     "pid": 3153, "tid": 7, "ts": 1707279511678260, "dur": 96,
     "args": {"device": 0,
              "cycles_ts": [3463433774,3463433774,3463434222,3463440085,3463440094]}}
  ]
}
```

The values above are illustrative. The fields that matter when reading a
trace by hand:

- `cat` classifies the span. `user_annotation` marks a `record_function`
  region, `cpu_op` is a host-side ATen call, and `kernel` is a device-side
  Spyre kernel.
- `ts` and `dur` are the start time and duration in microseconds.
- `tid` separates the trace rows in the viewer. Compute kernels render on
  device rows named `Stream {N}`. Host-side runtime activities, including
  `aiuSubmitToHardware` and the `aiuComms*` collective spans emitted by
  spyre-comms, render on host-thread rows named `Thread {N}`.

Filtering `traceEvents` to `cat == "kernel"` isolates device-side work
for scripted analysis.

### Collective metadata args

Device-side `kernel` and `gpu_memcpy` events that belong to a collective carry extra `args` naming the collective they serve:

| Arg | Type | Meaning |
|---|---|---|
| `coll_group` | string | Collective type, such as `"Allreduce"` or `"Allgather"`. |
| `coll_algo` | string | Collective algorithm, such as `"Ring"`. |
| `coll_bytes` | number or `""` | Total transfer size of the collective, in bytes; `""` when unknown. |

```json
{"ph": "X", "cat": "kernel", "name": "...",
 "args": {"coll_group": "Allreduce", "coll_algo": "Ring", "coll_bytes": 256}}
```

When each arg appears:

- Events outside a collective carry none of the `coll_*` args.
- Every collective event carries `coll_group` and `coll_bytes`.
- `coll_algo` is present only when the control block carried the full
  `[Coll,Algo,Bytes]` annotation. A control block labelled with a bare
  collective name has no `coll_algo`, and its `coll_bytes` is `""`
  because the runtime cannot recover the size.
- A numeric `coll_bytes` of `0` is a real zero-byte transfer, not an
  unknown size.

## `aiu-trace-analyzer`

[`aiu-trace-analyzer`][ata] is an open-source post-processing tool for
traces from the PyTorch profiler. The repository README is the
authoritative guide; minimum setup follows.

Install from source:

```bash
git clone https://github.com/IBM/aiu-trace-analyzer.git
cd aiu-trace-analyzer
pip install --editable .
```

Run the workload with profiling enabled (see
[PyTorch Profiler](pytorch_profiler.md)) and the runtime env vars that
expose compiler exports:

```bash
export DTCOMPILER_KEEP_EXPORT=true
export DEEPRT_EXPORT_DIR=<workload-directory>
export DTCOMPILER_EXPORT_DIR=<workload-directory>
export DT_DEEPRT_VERBOSE=0

python3 workload.py > logs.txt
```

Post-process the trace:

```bash
acelyzer -i <trace_file_json> -c logs.txt
```

### Known issues (from torch-spyre-docs)

- On the new stack, `logs.txt` can end up empty, in which case the
  processed output files are created but contain no additional
  information beyond the input trace.

## See also

- [PyTorch Profiler](pytorch_profiler.md): generating the traces
- [Performance analysis methodology](performance_analysis_methodology.md):
  using a loaded trace

[ata]: https://github.com/IBM/aiu-trace-analyzer
[ctf]: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
