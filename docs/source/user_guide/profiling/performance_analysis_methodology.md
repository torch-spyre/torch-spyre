# Performance Analysis Methodology

**Stack:** torch-spyre (new, Inductor-based).

The high-value pattern today is capturing a time-bounded
`torch.profiler` trace, breaking it down into kernel and transfer time,
comparing kernel time against the compiler's ideal-cycle estimate, and
reading the result alongside `aiu-smi` telemetry. Bottleneck
classification and multi-rank analysis will follow as the
[RFC 0601][rfc-0601] tooling matures. Contributions welcome.

## 1. Bound the measured region

Use `torch.profiler`'s `schedule` + `record_function` to avoid
measuring compile/warmup cost and to make iterations easy to select
in the viewer:

```python
from torch.profiler import profile, ProfilerActivity, schedule, record_function

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
) as prof:
    for step in range(10):
        with record_function(f"iteration_{step}"):
            output = model(inputs)
        prof.step()

prof.export_chrome_trace("spyre_trace.json")
```

See the upstream [PyTorch profiler documentation][torch-profiler-docs]
for the full `schedule` / `record_function` API.

## 2. Extract kernel and transfer time from the trace

A trace captured with `ProfilerActivity.PrivateUse1` is written in the
Chrome Trace Event Format, where every event carries a category (`cat`)
and a duration (`dur`) in microseconds. Two categories give the
device-side breakdown:

- `cat == "kernel"` events are device compute. Summing their `dur`
  yields total kernel time.
- `cat in ("gpu_memcpy", "gpu_memset")` events are host-to-device and
  device-to-host transfers and memory initialization. Summing their
  `dur` yields total transfer time.

Reading the exported JSON directly keeps the metric independent of the
printed summary table:

```python
import json

def kernel_and_transfer_ms(trace_path, n_iters):
    with open(trace_path) as f:
        events = json.load(f).get("traceEvents", [])
    kernel_us = sum(e.get("dur", 0) for e in events if e.get("cat") == "kernel")
    transfer_us = sum(
        e.get("dur", 0) for e in events
        if e.get("cat") in ("gpu_memcpy", "gpu_memset")
    )
    return kernel_us / n_iters / 1000.0, transfer_us / n_iters / 1000.0
```

Divide by the iteration count captured in the `active` window so the
result is per-iteration milliseconds. A high transfer fraction relative
to kernel time points to host-device movement on the critical path
rather than device compute.

## 3. PT-active utilization

PT-active utilization compares the theoretical minimum time for a kernel
against its measured execution time. It answers how close a kernel runs
to the hardware bound.

The numerator comes from a compiler artifact. When the compiler runs
with `SENPERFORMANCE=2`, it writes an `ideal_cycles.json` file per
kernel under `inductor-spyre/sdsc.bundle.mlir/perf/` in the Inductor
cache directory (`TORCHINDUCTOR_CACHE_DIR`). The `TOTAL` entry in that file
gives the kernel's ideal cycle count. Convert cycles to time with the
core clock frequency:

```text
ideal_ms  = ideal_cycles / core_frequency_mhz / 1000
actual_ms = measured kernel time for that kernel (from the trace)
pt_active_utilization = ideal_ms / actual_ms * 100
```

Sum `ideal_ms` and `actual_ms` across kernels, excluding memcpy and
memset entries from the compute total, for a whole-model figure.
`SENPERFORMANCE` is a compiler environment variable rather than a
torch-spyre setting, so the availability and exact format of
`ideal_cycles.json` follow the compiler build in use.

## 4. Pair the trace with `aiu-smi`

Run `aiu-smi` in a second shell during the profiling window (see
[Device monitoring](device_monitoring.md)). Both timestamps are
wall-clock, so you can line up a region of the trace with the
corresponding sample lines.

Which `aiu-smi` columns to look at depends on the question you're
asking — consult `aiu-smi --help` for the current column set. Note
that on the current new-stack build `rsvmem` and `pt_act` are not
captured correctly.

For post-processing the captured trace (additional statistics, trace
enrichment), see [`aiu-trace-analyzer`](trace_analysis.md#aiu-trace-analyzer)
([public repository][ata]).

## 5. Filing a performance report

When opening an issue, include:

- [ ] Minimal reproducer script and iteration count
- [ ] PyTorch version and torch-spyre commit SHA
- [ ] `aiu-smi` output covering at least one full active iteration
- [ ] `spyre_trace.json` or the TensorBoard log directory
- [ ] Summary table printed by `prof.key_averages().table(...)`
- [ ] What you expected vs. what you saw (latency or throughput)
- [ ] **For a performance regression**, cite the previous metric — the
  numeric value, the build date or commit SHA it was measured on, and
  the workload type — so the regression window is unambiguous.

## See also

- [PyTorch Profiler](pytorch_profiler.md) — generating traces
- [Device monitoring](device_monitoring.md) — `aiu-smi` telemetry
- [Trace analysis](trace_analysis.md) — viewer mechanics
- [RFC 0601][rfc-0601] — planned toolkit

[rfc-0601]: https://github.com/torch-spyre/rfcs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md
[torch-profiler-docs]: https://pytorch.org/docs/stable/profiler.html
[ata]: https://github.com/IBM/aiu-trace-analyzer
