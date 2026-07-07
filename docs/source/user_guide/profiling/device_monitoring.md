# Device Monitoring with `aiu-smi`

**Stack:** torch-spyre (new, Inductor-based).

`aiu-smi` is a command-line monitoring tool for Spyre devices. It
reads hardware performance counters and periodically prints metrics
such as PT-array utilization, power, temperature, device-memory and
PCIe bandwidth. No code changes are needed in the workload.

For the full metric list, CLI flags, and output format, consult the
tool directly — `aiu-smi --help` or `aiu-smi dmon --help`.

## Install

`aiu-monitor` is published on IBM SWG Artifactory under
`sys-power-hpc-pypi-local`; access to that repository is required.
**Wheel versions, package names, Python tags, and supported
architectures evolve** — always browse the live tree before you copy
an install command:

[`aiu-monitor/` on Artifactory](https://na.artifactory.swg-devops.com/ui/repos/tree/General/sys-power-hpc-pypi-local/aiu-monitor)

The tree is organised as `<arch>/{stable,dev}/<version>/<wheel>.whl`.
Pick the wheel that matches your CPU architecture and the Python
version of your venv — the wheel filename encodes both
(e.g. `…-py312-none-linux_x86_64.whl`).

The snippet below is **illustrative of the install pattern**, not a
pinned recipe; the URLs will go stale as new releases land. At the
time of writing, the latest stable was `1.2.1`:

```bash
# x86_64, Python 3.12 — torch-spyre-tagged build
uv pip install \
  https://na.artifactory.swg-devops.com/artifactory/sys-power-hpc-pypi-local/aiu-monitor/x86_64/stable/1.2.1/ibm_aiu_monitor-1.2.1+torch.spyre-py312-none-linux_x86_64.whl

# ppc64le, Python 3.9 (no torch-spyre build tag at this version)
uv pip install \
  https://na.artifactory.swg-devops.com/artifactory/sys-power-hpc-pypi-local/aiu-monitor/ppc64le/stable/1.2.1/ibm_aiu_monitor-1.2.0-py39-none-linux_ppc64le.whl

uv pip install psutil
```

A `dev/` channel also exists per arch (e.g.
`ppc64le/dev/latest_ubi10/` for the UBI 10 development build) — use
it only if you're chasing a fix that hasn't reached `stable/` yet.

## Two-terminal workflow

`aiu-smi` runs in its own shell alongside the workload.

**Workload shell:**

```bash
export DTCOMPILER_KEEP_EXPORT=true
export SENLIB_DEVEL_CONFIG_FILE=<path-to-venv>/etc/senlib_config_aiusmi.json
python my_workload.py
```

**`aiu-smi` shell:**

```bash
export DEEPRT_EXPORT_DIR=<workload-directory>
aiu-smi
```

See [Environment variables](environment_variables.md) for the variables
above.

## Known issues

- PF mode only.
- `rsvmem` and `pt_act` are **not captured correctly** on the current
  new-stack build.

## See also

- [Environment variables](environment_variables.md) — the variables
  that affect `aiu-smi`
- [Performance analysis methodology](performance_analysis_methodology.md) —
  pairing `aiu-smi` samples with trace-viewer timelines
