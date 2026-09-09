# PyTorch Profiler on Spyre

**Stack:** torch-spyre (new, Inductor-based).

`torch.profiler.profile` is the entry point for per-op timing on Spyre.
Two modes are available:

1. **CPU-only** — no extra install; measures host-side Python and
   `torch.compile` activity.
2. **CPU + PrivateUse1** — measures CPU *and* Spyre-side kernel activity.
   No extra install either: the AIUPTI/Kineto activity-tracing support is
   built directly into `torch_spyre`'s native extension (gated by the
   `USE_SPYRE_PROFILER` build flag, on by default).

## CPU-only (no extra install)

```python
import torch
from torch.profiler import profile, ProfilerActivity

compiled = torch.compile(model)

with profile(activities=[ProfilerActivity.CPU]) as prof:
    output = compiled(x_spyre)

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

This captures CPU wall-clock for every ATen call and every Dynamo /
Inductor stage.

## CPU + PrivateUse1

Profile with `ProfilerActivity.PrivateUse1`:

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
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="device_time_total", row_limit=10))
```

The table groups time by operator. `Self CPU` is host-side time spent
in the operator itself. `SPYRE total` is the device-side time attributed
to it. The layout looks like this:

```text
---------------------------  ------------  ------------  ------------  ------------
                       Name     Self CPU     CPU total   SPYRE total    # of Calls
---------------------------  ------------  ------------  ------------  ------------
             aten::mm          1.20ms         4.80ms        9.30ms            96
      aten::scaled_dot_...     0.40ms         2.10ms        3.70ms            48
             aten::add         0.30ms         0.90ms        0.80ms           192
    TorchDynamo Cache Lookup   0.05ms         0.05ms        0.00ms             1
---------------------------  ------------  ------------  ------------  ------------
Self CPU time total: 6.40ms
Self SPYRE time total: 14.10ms
```

The values above are illustrative. Absolute numbers depend on the model,
the batch and sequence configuration, and the build. Read the shape, not
the magnitudes: a large `SPYRE total` next to a small `Self CPU` marks a
device-bound operator (`aten::mm` here), which is the expected profile
for compute-heavy matmul layers.

### Export a trace for viewers

```python
prof.export_chrome_trace("spyre_trace.json")
```

See [Trace analysis](trace_analysis.md) for viewing.

### Compiled-kernel provenance names

Compiled Spyre compute events use a versioned name that carries a stable
bundle identity:

```text
spyre_kernel_v1_<fused-aten-summary>_<16-character-key>#<step>
```

The key is the first 80 bits of a SHA-256 fingerprint over the finalized
`OpSpec` and `LoopSpec` bundle, encoded as 16 lowercase base32 characters. An
80-bit key keeps collision probability negligible for the kernels in one
compile while leaving more of the AIUPTI event-name budget for the display-only
ATen summary. Source paths and line numbers are not written in
plaintext, avoiding disclosure of private paths. The fingerprint does include
direct `debug_handle` IDs, which derive from source metadata; moving the same
model to a different path can therefore change the opaque key.

The readable component intentionally derives from stable ATen packet names
instead of reusing Inductor's generated `kernel_name`, whose numeric suffix
depends on compilation order. The formatter conservatively reserves enough
space for `#<step>` even when the step index is the largest value supported by
the process ABI. A typical plan uses much smaller indices, but keeping this
reservation guarantees that Python-generated names remain valid after the C++
runtime appends the suffix.

The name describes bundle-level attribution:

- Every `ComputeOnDevice` step produced from the bundle receives the same key.
  The `#<step>` suffix is the JobExecPlan command index, so compute suffixes
  need not be contiguous. It distinguishes commands but does not claim that the
  proprietary backend assigned a particular subset of operations to that step.
- A compiler-generated provenance name deliberately replaces an existing
  SpyreCode compute label so every compute event retains the stable join key.
  Plans without a provenance name keep their previous labels and fallbacks.
- The associated descriptor lists only `debug_handle` IDs attached directly
  to finalized `OpSpec` records. Recursive `fused_from` records provide the
  constituent source and ATen lineage; the readable summary may use those
  constituents without adding their IDs to the direct list.
- Each `debug_handle` ID is a versioned 63-bit content hash of its complete
  structured source range, ATen op, ordered IR chain, and ordered fused
  constituent IDs. Transformation history is excluded because it describes how
  the operation was produced rather than which operation it represents.
- A valid bundle with no handles still receives a key and uses
  `fused_unknown` as its display summary.

For compiler-prepared provenance events, Kineto also carries structured
metadata:

- `args.provenance_key` is the 16-character join key as a string.
- `args.debug_handles` is a JSON array of directly attached handle IDs. The
  IDs are strings, and JavaScript consumers must keep them as strings rather
  than coercing them to numbers.

The key-bearing event name remains the compatibility join for raw traces and
name-only consumers. The trace does not embed source locations or full
transformation lineage. Durable source attribution requires pairing it with
`spyre_provenance.json`, which a separate artifact-publication change writes.

The v1 key is a trace-to-sidecar join key only. It is not a compilation cache
key or a cross-machine artifact identifier; source-path, schema, or toolchain
changes can produce a different key. A content hash is used instead of a
compile-local counter so a warm-cache wrapper replay can still join a fresh
trace to its persisted sidecar; ordering changes would make a counter risk a
wrong attribution rather than an explicit missing join.

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

[torch-profiler-docs]: https://pytorch.org/docs/stable/profiler.html
