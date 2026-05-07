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
