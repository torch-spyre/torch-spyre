# End-to-End Example: Profiling a Granite Model on Spyre via FMS

**Stack:** Torch-Spyre (new, Inductor-based).

This page shows how to capture a `torch.profiler` trace of a
Granite-class model running on Spyre, paired with `aiu-smi` device
telemetry. It uses today's tooling: `torch.profiler` + [`aiu-smi`](device_monitoring.md) + [`aiu-trace-analyzer`](trace_analysis.md).

The Granite end-to-end path on Spyre today goes through the
[Foundation Model Stack][fms] and
[`aiu-fms-testing-utils`][aiu-fms] — **not** HuggingFace
`AutoModelForCausalLM` directly. Spyre support for both currently
exists on the `eager_spyre` branch of each repo, so install them from
source off that branch rather than from PyPI.

This example profiles Granite 3.3-8B-Instruct on Spyre in eager mode so torch.compile is not required.

:::{admonition} Where this page is going
:class: note

The script below wires `torch.profiler` around `fms.get_model(...)`
explicitly. Once [RFC 0601][rfc-0601] lands, the in-tree
`torch_spyre.profiler` API will replace this glue and the script will
shrink. This page will be revised to the in-tree API once it ships.
:::

## What you need

**Prerequisites Step**

Ensure that you have access a pod with spyre accelerator and torch spyre and spyre software stack installed.

**Prerequisites Step**

Ensure that you have access a pod with spyre accelerator and torch spyre and spyre software stack installed.

| Piece | Source | Sample install (verify against the upstream README) |
|---|---|---|
| `foundation-model-stack` (`fms`) | [github.com/foundation-model-stack/foundation-model-stack][fms] (`eager_spyre` branch) | `git clone -b eager_spyre <repo>.git && uv pip install -e ./foundation-model-stack` |
| `aiu-fms-testing-utils` | [github.com/foundation-model-stack/aiu-fms-testing-utils][aiu-fms] (`eager_spyre` branch) | `git clone -b eager_spyre <repo>.git && uv pip install -e ./aiu-fms-testing-utils` |
| `kineto-spyre` (not required for torch >= 2.13)| [github.com/IBM/kineto-spyre][kineto-spyre] | `uv pip install --no-deps <release-wheel-url-matching-your-pytorch>` (see [releases page][kineto-spyre-releases]) |
| `aiu-trace-analyzer` (optional) | [github.com/IBM/aiu-trace-analyzer][ata] | `pip install aiu-trace-analyzer` |
| Granite checkpoint | [huggingface.co/ibm-granite/granite-3.3-8b-instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) | `hf download ibm-granite/granite-3.3-8b-instruct --local-dir /tmp/models/granite-3.3-8b-instruct` |
| Granite checkpoint | [huggingface.co/ibm-granite/granite-3.3-8b-instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) | `hf download ibm-granite/granite-3.3-8b-instruct --local-dir /tmp/models/granite-3.3-8b-instruct` |

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

#Install huggingface hub
pip install -U huggingface_hub

#Download granite
hf download ibm-granite/granite-3.3-8b-instruct --local-dir /tmp/models/granite-3.3-8b-instruct

# If your environment uses PyTorch 2.11, install the matching kineto-spyre wheel. It is not required for PyTorch 2.13 and later
# See below an example of 2.11 kineto Wheel on x86
uv pip install --no-deps --force-reinstall \
  https://github.com/IBM/kineto-spyre/releases/download/torch-2.11.0.aiu.kineto.1.1.2/torch-2.11.0+aiu.kineto.1.1.2-cp312-cp312-linux_x86_64.whl
  https://github.com/IBM/kineto-spyre/releases/download/torch-2.11.0.aiu.kineto.1.1.2/torch-2.11.0+aiu.kineto.1.1.2-cp312-cp312-linux_x86_64.whl
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
import os
import time
import torch
from statistics import mean, median
from torch.profiler import profile, ProfilerActivity
from fms.models import get_model
from fms.utils.generation import pad_input_ids
from transformers import AutoTokenizer

DEVICE = torch.device("spyre")
DTYPE = torch.float16  # Spyre's default dtype
MODEL_PATH = "/tmp/models/granite-3.3-8b-instruct"

print("=" * 42)
print("Loading".center(42))
print("=" * 42)

# 1. Load Granite via FMS.
model = get_model(
    architecture="hf_pretrained",
    model_path=MODEL_PATH,
    device_type="spyre",
    data_type=DTYPE,
    unfuse_weights=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

# 2. Build a prefill input. Replace torch.randint with a real-prompt
#    encoding (tokenizer(..., padding="max_length", max_length=512))
#    for non-toy runs.
raw_ids = torch.randint(
    0, tokenizer.vocab_size, (512,), dtype=torch.int32,
)

ids, kwargs = pad_input_ids(
    [raw_ids],
    min_pad_length=512,
    pad_token_id=pad_token_id,
)

position_ids = kwargs["position_ids"]
alpha = model.base_model.rot_emb.compute_freqs_cis(DEVICE, ids.shape[1])
selected_freqs = model.base_model.rot_emb.cached_freqs[0][alpha][position_ids].to(DEVICE)
mask = kwargs["mask"].to(dtype=torch.float16).to(DEVICE)

# 3Warmup — first run is always slower due to JIT/compilation
print("=" * 42)
print("Warming up.".center(42))
print("=" * 42)

for _ in range(2):
    with torch.no_grad():
      model(ids.to(DEVICE), position_ids=position_ids.to(DEVICE), mask=mask, selected_freqs=selected_freqs)

print("=" * 42)
print("Starting Inferencing".center(42))
print("=" * 42)


# 4. Profile a steady-state forward pass.
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
          model(ids.to(DEVICE), position_ids=position_ids.to(DEVICE), mask=mask, selected_freqs=selected_freqs)
          model(ids.to(DEVICE), position_ids=position_ids.to(DEVICE), mask=mask, selected_freqs=selected_freqs)
        wall_clock_ms.append((time.perf_counter() - t0) * 1000)
        prof.step()

# 5. Two timing signals: wall-clock (what the user feels) and
#    profiler-derived CPU time (host-side overhead). The gap between
#    them ≈ device-side work.
cpu_per_run_ms = sum(e.self_cpu_time_total for e in prof.events()) / 1000 / N_RUNS

print("=" * 42)
print("Profiling Granite Running on Spyre".center(42))
print("=" * 42)
print("=" * 42)
print("Profiling Granite Running on Spyre".center(42))
print("=" * 42)
print(prof.key_averages().table(sort_by="device_time_total", row_limit=10))
print(f"wall-clock ms: mean={mean(wall_clock_ms):.3f} median={median(wall_clock_ms):.3f}")
print(f"profiler-derived CPU ms (per run): {cpu_per_run_ms:.3f}")
```

**Note**

Ensure the profiler report includes Spyre/device time; otherwise the trace does not confirm the accelerator execution

Three patterns to call out:

- **Run warmup iterations outside the timed loop**. The first few executions can include one-time runtime initialization, lazy setup, cache population, or other startup overhead. Excluding them keeps steady-state measurements representative.
- **Use two orthogonal timing signals**. Wall-clock time from time.perf_counter() represents end-to-end latency observed by the caller, while profiler-derived CPU time provides a view of host-side activity. Comparing them can help identify whether latency is dominated by host-side work or accelerator execution, but their difference should not be treated as a direct measurement of device time.
- **`tensorboard_trace_handler(log_dir)` over `export_chrome_trace`.** Per-step JSON files make it easier to distinguish warmup executions from steady-state runs and inspect each profiler step independently.
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
| First iteration much slower than the rest | the first iterations as runtime/device initialization or general warmup.
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
- [RFC 0601][rfc-0601] — full profiling toolkit design

[fms]: https://github.com/foundation-model-stack/foundation-model-stack
[aiu-fms]: https://github.com/foundation-model-stack/aiu-fms-testing-utils
[kineto-spyre]: https://github.com/IBM/kineto-spyre
[kineto-spyre-releases]: https://github.com/IBM/kineto-spyre/releases
[ata]: https://github.com/IBM/aiu-trace-analyzer
[rfc-0601]: https://github.com/torch-spyre/rfcs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md
