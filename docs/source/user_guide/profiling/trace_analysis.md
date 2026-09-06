# Trace Analysis

**Stack:** torch-spyre (new, Inductor-based).

Traces written by `torch.profiler` (see [PyTorch Profiler](pytorch_profiler.md))
are Chrome-trace JSON files. They open in any of three viewers:

- **PyTorch Profiler TensorBoard Plugin** (preferred) — AIU-aware
  views on top of the raw trace. Source and install instructions:
  <https://github.com/IBM/kineto-spyre/tree/main/tb_plugin>
- **Perfetto** — drag and drop the JSON onto <https://ui.perfetto.dev/>
- **Chrome Trace Viewer** — `chrome://tracing` in Chrome

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
     "args": {"device": 0}}
  ]
}
```

The values above are illustrative. The fields that matter when reading a
trace by hand:

- `cat` classifies the span. `user_annotation` marks a `record_function`
  region, `cpu_op` is a host-side ATen call, and `kernel` is a device-side
  Spyre kernel.
- `ts` and `dur` are the start time and duration in microseconds.
- `tid` separates the host thread from device streams, so device kernels
  render on their own row in the viewer.

Filtering `traceEvents` to `cat == "kernel"` isolates device-side work
for scripted analysis.

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

- [PyTorch Profiler](pytorch_profiler.md) — generating the traces
- [Performance analysis methodology](performance_analysis_methodology.md) —
  using a loaded trace

[ata]: https://github.com/IBM/aiu-trace-analyzer
[ctf]: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
