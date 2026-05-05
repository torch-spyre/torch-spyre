# PyTorch Profiler on Spyre

**Stack:** torch-spyre (new, Inductor-based).

`torch.profiler.profile` is the entry point for per-op timing on Spyre.
Two modes are available:

1. **CPU-only** — no extra install; measures host-side Python and
   `torch.compile` activity.
2. **CPU + PrivateUse1** — measures CPU *and* Spyre-side kernel activity;
   requires the [`kineto-spyre`][kineto-spyre] PyTorch wheel.

## CPU-only (no extra install)

```python
import torch
from torch.profiler import profile, ProfilerActivity

compiled = torch.compile(model, backend="spyre")

with profile(activities=[ProfilerActivity.CPU]) as prof:
    output = compiled(x_spyre)

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

This captures CPU wall-clock for every ATen call and every Dynamo /
Inductor stage.

## CPU + PrivateUse1

Install a matching [`kineto-spyre`][kineto-spyre] wheel for your
PyTorch version (check the [releases page][kineto-spyre-releases] for
the current combination). Example URL for PyTorch 2.10.0:

```bash
uv pip install --no-deps --force-reinstall \
  https://github.com/IBM/kineto-spyre/releases/download/torch-2.10.0.aiu.kineto.1.1.1/torch-2.10.0+aiu.kineto.1.1.1-cp312-cp312-linux_x86_64.whl
```

Then profile with `ProfilerActivity.PrivateUse1`:

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    record_shapes=True,
    profile_memory=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/mymodel"),
) as prof:
    compiled_result = compiled(x_device).cpu()
```

### Print aggregates

```python
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10).replace("CUDA", "AIU"))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10).replace("CUDA", "AIU"))
```

The `.replace("CUDA", "AIU")` is a cosmetic workaround — the profiler's
internal column category is still named after CUDA; native renaming is
on the roadmap.

### Export a trace for viewers

```python
prof.export_chrome_trace("spyre_trace.json")
```

See [Trace analysis](trace_analysis.md) for viewing.

## Advanced features

Full reference lives in the upstream
[PyTorch profiler documentation][torch-profiler-docs]:

- `record_function` — annotate named spans
- `schedule` — skip warmup, sample a bounded window
- `on_trace_ready` — stream to TensorBoard-compatible JSON
- `with_stack` — include file and line for Python ops

## Known issues (from torch-spyre-docs)

- **Multi-AIU communication profiling is not supported yet.**

## See also

- [Trace analysis](trace_analysis.md) — viewers for the traces
- [Device monitoring](device_monitoring.md) — `aiu-smi` telemetry
  alongside `torch.profiler`

[kineto-spyre]: https://github.com/IBM/kineto-spyre
[kineto-spyre-releases]: https://github.com/IBM/kineto-spyre/releases
[torch-profiler-docs]: https://pytorch.org/docs/stable/profiler.html
