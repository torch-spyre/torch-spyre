# Profiling

```{toctree}
:hidden:
:maxdepth: 2

environment_variables
pytorch_profiler
device_monitoring
trace_analysis
performance_analysis_methodology
toolkit_matrix
end_to_end_example
```

**Stack:** torch-spyre (new, Inductor-based).

**Scope:** performance — *why is it slow?* For correctness questions
(*why is the result wrong?*) see [Debugging](../debugging/index.md).

Torch-Spyre provides tooling to measure the performance of PyTorch
workloads running on the Spyre accelerator. The full design of the
planned toolkit is in
[RFC 0601 — Spyre Profiling Toolkit][rfc-0601].

The in-tree `torch_spyre.profiler` package is currently a scaffold —
`torch_spyre.profiler.is_available()` returns `False`, and there is no
public API yet. Profiling today goes through `torch.profiler` plus the
external integrations described on this page (`kineto-spyre`,
`aiu-smi`, `aiu-trace-analyzer`); the in-tree API will be populated as
RFC 0601 lands.

## What can be profiled today

| Capability | Status | Where |
|---|---|---|
| Compiler pipeline logs | Available | [Environment variables](environment_variables.md) |
| CPU-side timing with `torch.profiler` | Available | [PyTorch Profiler](pytorch_profiler.md) |
| Device telemetry (power, temperature, bandwidth) | Available — PF and VF mode (IBM-internal distribution; public release tracked in [#1335][issue-1335]) | [Device monitoring](device_monitoring.md) |
| Device-side kernel timing via `ProfilerActivity.PrivateUse1` | Preview (requires [`kineto-spyre`][kineto-spyre] wheel) | [PyTorch Profiler](pytorch_profiler.md) |
| Trace post-processing (aiu-trace-analyzer) | Available, known gaps | [Trace analysis](trace_analysis.md) |
| `torch.spyre.memory.memory_allocated()` / `max_memory_allocated()` | Available — delegates to [`torch.accelerator.memory`][accelerator-memory] (PR [#770][pr-770]) | [Quick example](#memory-api-quick-example) |
| Kineto bridge (`SpyreActivityProfiler`) | In progress — in-tree Kineto integration for `ProfilerActivity.PrivateUse1` device-side events (PR [#1856][pr-1856]) | upstream Kineto integration |
| Scratchpad utilization metrics | Planned | [RFC 0601][rfc-0601] |
| IR-instrumentation-based fine-grained profiler | Planned | [RFC 0601][rfc-0601] |

### Memory API quick example

`torch.spyre.memory` re-exports `torch.accelerator.memory`, so the
same memory-query calls used on CUDA apply to Spyre. The example
below allocates a tensor, frees it, and reads the current and peak
allocations:

```python
import torch

# Reset the peak counter so max_memory_allocated() starts from zero.
torch.spyre.memory.reset_peak_memory_stats()

# Allocate on the device; memory_allocated() reflects the new total.
x = torch.rand((64, 64), dtype=torch.float16, device="spyre")
print(torch.spyre.memory.memory_allocated())     # bytes currently allocated

# Free the tensor. memory_allocated() drops back, but the peak persists.
del x
print(torch.spyre.memory.memory_allocated())     # current allocation
print(torch.spyre.memory.max_memory_allocated()) # peak since reset
```

The module also exposes `reset_accumulated_memory_stats()` and
`memory_stats()`.

## Toolkit layers

| Layer | Tool | Granularity |
|---|---|---|
| Application / PyTorch | `torch.profiler` + [kineto-spyre][kineto-spyre] | Kernel-level |
| Compiler frontend | Inductor logging | Pass-level |
| Compiler backend | IR instrumentation *(planned)* | Intra-kernel |
| Runtime | `libaiupti` kernel + memory events | Kernel + memory |
| Device / HW | `aiu-smi` | Device-level telemetry |
| Post-processing | [aiu-trace-analyzer][ata] | Derived metrics |

## Profiling topics

- [Environment variables](environment_variables.md) — logging, device
  enumeration, runtime/driver variables used by `aiu-smi` and
  `aiu-trace-analyzer`
- [PyTorch Profiler](pytorch_profiler.md) — `torch.profiler` usage, CPU
  today, device-side preview
- [Device monitoring](device_monitoring.md) — `aiu-smi` setup
- [Trace analysis](trace_analysis.md) — Chrome / Perfetto / TensorBoard
  viewing and `aiu-trace-analyzer` post-processing
- [Performance analysis methodology](performance_analysis_methodology.md) —
  bounding a region and pairing traces with telemetry
- [Toolkit usage matrix](toolkit_matrix.md) — which tool for which metric
- [End-to-end example](end_to_end_example.md) — profiling a Granite
  model on Spyre, gluing all four tools into one workflow

## See also

- [Debugging](../debugging/index.md) — correctness-focused workflow,
  including `TORCH_COMPILE_DEBUG` artifacts and the `sendnn` bisect
- [Running Models](../running_models.md) — `torch.compile` usage
- [Compiler Architecture](../../compiler/architecture.md) — pipeline
  overview
- [RFC 0601][rfc-0601] — full profiling toolkit design
- [Contributing to the Profiler](../../contributing/profiling.md) —
  branch / commit conventions, build flag, test layout, and review
  process for the profiling squad

:::{admonition} Work in Progress
:class: warning

Some subsystems above are labelled **Planned** and are under active
development as part of [RFC 0601][rfc-0601]. The APIs reflect planned
design and may change.
:::

[rfc-0601]: https://github.com/torch-spyre/rfcs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md
[kineto-spyre]: https://github.com/IBM/kineto-spyre
[ata]: https://github.com/IBM/aiu-trace-analyzer
[issue-1335]: https://github.com/torch-spyre/torch-spyre/issues/1335
[pr-770]: https://github.com/torch-spyre/torch-spyre/pull/770
[pr-1856]: https://github.com/torch-spyre/torch-spyre/pull/1856
[accelerator-memory]: https://docs.pytorch.org/docs/stable/accelerator.html
