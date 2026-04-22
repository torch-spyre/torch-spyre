# Device Monitoring with `aiu-smi`

**Stack:** torch-spyre (new, Inductor-based).

`aiu-smi` is a command-line monitoring tool for Spyre devices. It
reads hardware performance counters and periodically prints metrics
such as PT-array utilization, power, temperature, device-memory and
PCIe bandwidth. No code changes are needed in the workload.

For the full metric list, CLI flags, and output format, consult the
tool directly — `aiu-smi --help` or `aiu-smi dmon --help`.

## Install

Install the wheel matching your platform. Access to IBM SWG Artifactory
is required.

```bash
uv pip install \
  https://na.artifactory.swg-devops.com/artifactory/api/pypi/sys-power-hpc-pypi-local/aiu-monitor/x86_64/stable/1.0.0/aiu_monitor-1.0.0-py39-none-linux_x86_64.whl
uv pip install psutil
```

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

