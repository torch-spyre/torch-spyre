# Toolkit Usage Matrix

**Stack:** torch-spyre (new, Inductor-based).

**Scope:** performance only. For correctness / compiler-artifact
questions, see [Debugging](../debugging/index.md).

| You want to know… | Tool | Where |
|---|---|---|
| CPU-side time per ATen op | `torch.profiler` with `ProfilerActivity.CPU` | [PyTorch Profiler](pytorch_profiler.md#cpu-only-no-extra-install) |
| Device-side time per Spyre kernel | `torch.profiler` with `PrivateUse1` + [kineto-spyre][kineto-spyre] | [PyTorch Profiler](pytorch_profiler.md#cpu--privateuse1) |
| Device power / temperature / bandwidth | `aiu-smi` | [Device monitoring](device_monitoring.md) |
| Post-processed trace metrics | [aiu-trace-analyzer][ata] | [Trace analysis](trace_analysis.md#aiu-trace-analyzer) |
| Memory APIs (`torch.spyre.memory_allocated()` / peak) | *Planned* | [RFC 0601][rfc-0601] |
| Scratchpad utilization | *Planned* | [RFC 0601][rfc-0601] |
| IR-instrumentation profiling | *Planned* | [RFC 0601][rfc-0601] |

[kineto-spyre]: https://github.com/IBM/kineto-spyre
[ata]: https://github.com/IBM/aiu-trace-analyzer
[rfc-0601]: https://github.com/torch-spyre/rfcs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md
