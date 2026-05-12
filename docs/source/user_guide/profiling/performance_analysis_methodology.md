# Performance Analysis Methodology

**Stack:** torch-spyre (new, Inductor-based).

:::{admonition} Stub
:class: warning

This page is a scaffold. Methodology examples — bottleneck
classification, kernel drill-down, category breakdowns, multi-rank
analysis — will land here as real new-stack traces become available
and are validated against [RFC 0601][rfc-0601] tooling. Contributions
welcome.
:::

The high-value pattern today is capturing a time-bounded
`torch.profiler` trace alongside `aiu-smi` telemetry and reading them
together.

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

## 2. Pair the trace with `aiu-smi`

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

## 3. Filing a performance report

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
