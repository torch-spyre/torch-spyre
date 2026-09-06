# End-to-End Example: Profiling a Granite Model on Spyre via FMS

**Stack:** torch-spyre

This page shows how to capture a `torch.profiler` trace of a
Granite-class model via FMS running on Spyre, paired with `aiu-smi` device
telemetry. It uses today's tooling: `torch.profiler` + [`aiu-smi`](device_monitoring.md) + [`aiu-trace-analyzer`](trace_analysis.md).

The Granite end-to-end path on Spyre today goes through the
[Foundation Model Stack][fms] and
[`aiu-fms-testing-utils`][aiu-fms] — **not** HuggingFace
`AutoModelForCausalLM` directly. Spyre support for both currently
exists on the `eager_spyre` branch of each repo, so install them from
source off that branch rather than from PyPI.

This example profiles Granite 3.3-8B-Instruct on Spyre in eager mode so `torch.compile` is not required.

## What you need

**Prerequisites Step**

Ensure that you have access to a pod with a Spyre accelerator and the torch-spyre and Spyre software stack installed.

| Piece | Source | Sample install (verify against the upstream README) |
|---|---|---|
| `foundation-model-stack` (`fms`) | [github.com/foundation-model-stack/foundation-model-stack][fms] (`eager_spyre` branch) | `git clone -b eager_spyre <repo>.git && uv pip install -e ./foundation-model-stack` |
| `aiu-fms-testing-utils` | [github.com/foundation-model-stack/aiu-fms-testing-utils][aiu-fms] (`eager_spyre` branch) | `git clone -b eager_spyre <repo>.git && uv pip install -e ./aiu-fms-testing-utils` |
| `aiu-trace-analyzer` (optional) | [github.com/IBM/aiu-trace-analyzer][ata] | `pip install aiu-trace-analyzer` |
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
```

Useful environment variables (see [Environment variables](environment_variables.md)
for the full list):

```bash
export PYTHONUNBUFFERED=1
export SENCORES=32                 # full accelerator (1–32; default 32)
```

## The script

Save as `profile_granite.py`.

```python
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

# NOTE: Precomputing the rotary frequencies and passing selected_freqs=
# as a forward kwarg is a temporary workaround. It reaches into FMS
# internals and will be removed once the upstream fix lands.
position_ids = kwargs["position_ids"]
alpha = model.base_model.rot_emb.compute_freqs_cis(DEVICE, ids.shape[1])
selected_freqs = model.base_model.rot_emb.cached_freqs[0][alpha][position_ids].to(DEVICE)
mask = kwargs["mask"].to(dtype=torch.float16).to(DEVICE)

# 3. Warmup — first run is always slower due to runtime and device initialization
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
        wall_clock_ms.append((time.perf_counter() - t0) * 1000)
        prof.step()

# 5. Two timing signals: wall-clock (end-to-end latency) and
#    profiler-derived CPU time (host-side activity). Comparing them
#    shows whether latency is host-bound or device-bound.
cpu_per_run_ms = sum(e.self_cpu_time_total for e in prof.events()) / 1000 / N_RUNS

print("=" * 42)
print("Profiling Granite Running on Spyre".center(42))
print("=" * 42)

print(prof.key_averages().table(sort_by="device_time_total", row_limit=10))
print(f"wall-clock ms: mean={mean(wall_clock_ms):.3f} median={median(wall_clock_ms):.3f}")
print(f"profiler-derived CPU ms (per run): {cpu_per_run_ms:.3f}")
```

**Note**

Confirm the profiler table reports Spyre or device time. If it does not, the trace is not capturing accelerator execution.

Three patterns to call out:

- **Run warmup iterations outside the timed loop**. The first runs carry one-time runtime and device initialization. Excluding them keeps the steady-state numbers representative.
- **Use two orthogonal timing signals**. Wall-clock from time.perf_counter() is end-to-end latency; profiler-derived CPU time is host-side activity. Comparing them shows whether latency is host-bound or device-bound, though the difference is not a direct measurement of device time.
- **`tensorboard_trace_handler(log_dir)` over `export_chrome_trace`.** Per-step JSON files make it easier to distinguish warmup executions from steady-state runs, and they open in TensorBoard as well as Chrome / Perfetto.

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
| First iteration much slower than the rest | Runtime and device initialization or general warmup | Expected. Discard the early iterations. |
| Wall-clock ≫ profiler CPU | Device-side work dominates (good for compute-bound layers like MLP / large matmul) | Cross-check with `aiu-smi` PT-array util. |
| Wall-clock ≈ profiler CPU | Host-side bottleneck — Python or Dynamo overhead | `TORCH_LOGS="+inductor"` |
| Per-layer kernel gaps | Tile staging between LPDDR5 and LX scratchpad | [Performance analysis methodology](performance_analysis_methodology.md) |
| Low PT-array utilization in `aiu-smi` | Work-division inefficiency, stick-alignment padding | [Compiler work division](../../compiler/work_division_planning.md) |
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
[ata]: https://github.com/IBM/aiu-trace-analyzer
[rfc-0601]: https://github.com/torch-spyre/rfcs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md
