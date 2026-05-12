# End-to-End Example: Profiling a Granite Model on Spyre

**Stack:** torch-spyre (new, Inductor-based).

This page shows how to capture a `torch.profiler` trace of a
Granite-class model running on Spyre, paired with `aiu-smi` device
telemetry. It uses today's tooling: `torch.profiler` +
[`kineto-spyre`][kineto-spyre] + [`aiu-smi`](device_monitoring.md) +
[`aiu-trace-analyzer`](trace_analysis.md).

The Granite end-to-end path on Spyre today goes through the
[Foundation Model Stack][fms] and
[`aiu-fms-testing-utils`][aiu-fms] — **not** HuggingFace
`AutoModelForCausalLM` directly. Spyre support for both currently
exists on the `eager_spyre` branch of each repo, so install them from
source off that branch rather than from PyPI.

:::{admonition} Where this page is going
:class: note

The script below wires `torch.profiler` around `fms.get_model(...)`
explicitly. Once [RFC 0601][rfc-0601] lands, the in-tree
`torch_spyre.profiler` API will replace this glue and the script will
shrink. This page will be revised to the in-tree API once it ships.
:::

## What you need

| Piece | Source | Sample install (verify against the upstream README) |
|---|---|---|
| `foundation-model-stack` (`fms`) | [github.com/foundation-model-stack/foundation-model-stack][fms] (`eager_spyre` branch) | `git clone -b eager_spyre <repo>.git && uv pip install -e ./foundation-model-stack` |
| `aiu-fms-testing-utils` | [github.com/foundation-model-stack/aiu-fms-testing-utils][aiu-fms] (`eager_spyre` branch) | `git clone -b eager_spyre <repo>.git && uv pip install -e ./aiu-fms-testing-utils` |
| `kineto-spyre` | [github.com/IBM/kineto-spyre][kineto-spyre] | `uv pip install --no-deps <release-wheel-url-matching-your-pytorch>` (see [releases page][kineto-spyre-releases]) |
| `aiu-trace-analyzer` (optional) | [github.com/IBM/aiu-trace-analyzer][ata] | `pip install aiu-trace-analyzer` |
| Granite checkpoint | [huggingface.co/ibm-granite/granite-3.3-8b-instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) | `huggingface-cli download ibm-granite/granite-3.3-8b-instruct --local-dir /tmp/models/granite-3.3-8b-instruct` |

The sample commands above are starting points; each upstream README is
the source of truth and may require additional steps (extras, source
installs, branch selection) depending on your platform and PyTorch
build.

## Setup

```bash
# Install fms and aiu-fms-testing-utils from the eager_spyre branch
# (the branch that registers the Spyre device backend today).
git clone -b eager_spyre https://github.com/foundation-model-stack/foundation-model-stack.git
git clone -b eager_spyre https://github.com/foundation-model-stack/aiu-fms-testing-utils.git
uv pip install -e ./foundation-model-stack
uv pip install -e ./aiu-fms-testing-utils

# Cache HuggingFace artifacts and download the Granite checkpoint.
export HF_HOME=/tmp/models/hf_cache
huggingface-cli download ibm-granite/granite-3.3-8b-instruct \
  --local-dir /tmp/models/granite-3.3-8b-instruct

# Example kineto-spyre wheel for PyTorch 2.10.0 + Python 3.12 on x86_64
# Linux. Pick the wheel that matches your stack from the releases page.
uv pip install --no-deps --force-reinstall \
  https://github.com/IBM/kineto-spyre/releases/download/torch-2.10.0.aiu.kineto.1.1.1/torch-2.10.0+aiu.kineto.1.1.1-cp312-cp312-linux_x86_64.whl
```

Useful environment variables (see [Environment variables](environment_variables.md)
for the full list):

```bash
export PYTHONUNBUFFERED=1
export SENCORES=32                 # full accelerator (1–32; default 32)
# Inductor visibility — uncomment when investigating compile-time issues:
# export TORCH_LOGS=ir_post_fusion,output_code,graph,aot_graphs,post_grad_graphs
# export TORCH_LOGS_FORMAT=short
# export TORCH_SPYRE_DEBUG=1
```

## The script

Save as `profile_granite.py`.

```python
import time
from statistics import mean, median

import torch
from torch.profiler import ProfilerActivity, profile
from fms.models import get_model
from transformers import AutoTokenizer

DEVICE = torch.device("spyre")
DTYPE = torch.float16  # Spyre's default dtype
MODEL_PATH = "/tmp/models/granite-3.3-8b-instruct"

# 1. Load Granite via FMS.
model = get_model(
    architecture="hf_pretrained",
    model_path=MODEL_PATH,
    device_type="spyre",
    data_type=DTYPE,
    unfuse_weights=True,
).eval().to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 2. Compile through the Spyre Inductor backend.
model.compile()

# 3. Build a prefill input. Replace torch.randint with a real-prompt
#    encoding (tokenizer(..., padding="max_length", max_length=512))
#    for non-toy runs.
inputs = torch.randint(
    0, tokenizer.vocab_size, (1, 512), dtype=torch.int64,
).to(DEVICE)

# 4. Warm up — first call triggers torch.compile + kernel codegen.
with torch.no_grad():
    model(inputs)

# 5. Profile a steady-state forward pass.
N_RUNS = 5
wall_clock_ms = []
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    record_shapes=True,
    profile_memory=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/granite"),
) as prof:
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(inputs)
        wall_clock_ms.append((time.perf_counter() - t0) * 1000)
        prof.step()

# 6. Two timing signals: wall-clock (what the user feels) and
#    profiler-derived CPU time (host-side overhead). The gap between
#    them ≈ device-side work.
cpu_per_run_ms = sum(e.self_cpu_time_total for e in prof.events()) / 1000 / N_RUNS

print(prof.key_averages().table(sort_by="device_time_total", row_limit=10))
print(f"wall-clock ms: mean={mean(wall_clock_ms):.3f} median={median(wall_clock_ms):.3f}")
print(f"profiler-derived CPU ms (per run): {cpu_per_run_ms:.3f}")
```

Three patterns to call out:

- **Run iteration 1 outside the timed loop.** First call carries the
  `torch.compile` cost; including it skews the average.
- **Two orthogonal timing signals.** Wall-clock from
  `time.perf_counter()` is what the user feels; profiler-derived CPU
  time (`sum(e.self_cpu_time_total)`) is host-side overhead. The
  difference between them ≈ device-side work.
- **`tensorboard_trace_handler(log_dir)` over `export_chrome_trace`.**
  Per-step JSON files isolate compile-time warmup from steady-state
  runs and open in TensorBoard *and* Chrome / Perfetto.

See [PyTorch Profiler](pytorch_profiler.md).

## Inspect the trace

The `logs/granite/` directory will contain one JSON per profiler step.
Open in any of:

- `chrome://tracing` — built into Chromium / Chrome.
- [Perfetto UI](https://ui.perfetto.dev/) — drag-and-drop the file.
- TensorBoard — `tensorboard --logdir=logs/granite`.

Then post-process with `aiu-trace-analyzer` to extract derived metrics
(kernel durations, gap analysis, idle bubbles). See
[Trace analysis](trace_analysis.md).

## Run with telemetry alongside

`aiu-smi` requires the senlib config file environment variable to be
set before it can talk to the device. Set it (and any other
device-discovery env vars your environment requires) in the same shell
before launching `aiu-smi`.

In one terminal:

```bash
export SENLIB_DEVEL_CONFIG_FILE=/path/to/senlib_config.json
aiu-smi dmon | tee /tmp/aiu-smi.log
```

In another:

```bash
python profile_granite.py
```

`aiu-smi dmon` samples once a second and streams power, temperature,
PT-array utilization, device-memory and PCIe bandwidth. Pair its
timestamps with the trace timeline to attribute idle gaps to either
host-side work or device-side stalls. See
[Device monitoring](device_monitoring.md).

## What to look for

For a Granite-class transformer the typical signals are:

| Symptom | Likely cause | Where to dig |
|---|---|---|
| First iteration much slower than the rest | `torch.compile` warmup | Expected — discard the first iteration. |
| Wall-clock ≫ profiler CPU | Device-side work dominates (good for compute-bound layers like MLP / large matmul) | Cross-check with `aiu-smi` PT-array util. |
| Wall-clock ≈ profiler CPU | Host-side bottleneck — Python or Dynamo overhead | `TORCH_LOGS="+inductor"` |
| Per-layer kernel gaps | Tile staging between LPDDR5 and LX scratchpad | [Performance analysis methodology](performance_analysis_methodology.md) |
| Low PT-array utilization in `aiu-smi` | Work-division inefficiency, stick-alignment padding | [Compiler work division](../../compiler/work_division_planning.md) |
| Long Inductor pass times in stderr | Compile-time regression | [Inductor debug artifacts](../debugging/inductor_artifacts.md) |
| Idle bubbles between consecutive kernels | Reconfiguration latency or DMA stalls | `aiu-trace-analyzer` gap analysis |

## See also

- [PyTorch Profiler](pytorch_profiler.md) — `torch.profiler` reference
- [Device monitoring](device_monitoring.md) — `aiu-smi` setup
- [Trace analysis](trace_analysis.md) — viewers and `aiu-trace-analyzer`
- [Performance analysis methodology](performance_analysis_methodology.md) —
  bounding a region and pairing traces with telemetry
- [Environment variables](environment_variables.md) — full list of
  logging and runtime flags
- [Running Models](../running_models.md) — `torch.compile` basics
- [RFC 0601][rfc-0601] — full profiling toolkit design

[fms]: https://github.com/foundation-model-stack/foundation-model-stack
[aiu-fms]: https://github.com/foundation-model-stack/aiu-fms-testing-utils
[kineto-spyre]: https://github.com/IBM/kineto-spyre
[kineto-spyre-releases]: https://github.com/IBM/kineto-spyre/releases
[ata]: https://github.com/IBM/aiu-trace-analyzer
[rfc-0601]: https://github.com/torch-spyre/rfcs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md
