# Device Monitoring with `aiu-smi`

**Stack:** torch-spyre (new, Inductor-based).

`aiu-smi` is a command-line monitoring tool for Spyre devices. It
reads hardware performance counters and periodically prints metrics
such as PT-array utilization, power, temperature, device-memory and
PCIe bandwidth. No code changes are needed in the workload.

For the full metric list, CLI flags, and output format, consult the
tool directly — `aiu-smi --help` or `aiu-smi dmon --help`.

## Install

`aiu-monitor` ships as a pre-built wheel from your internal IBM package
mirror. Ask your Spyre enablement contact for the mirror location and
access. The steps here describe the install pattern. They do not pin a
specific URL.

Wheel versions, package names, Python tags, and supported architectures
change over time, so browse the live package index before you copy an
install command. The wheels are organised as
`<arch>/{stable,dev}/<version>/<wheel>.whl`. Pick the wheel that matches
your CPU architecture and the Python version of your venv. The wheel
filename encodes both (e.g. `ibm_aiu_monitor-...-py312-none-linux_x86_64.whl`).
Prefer the `stable/` channel. A `dev/` channel exists per arch for
chasing a fix that has not reached `stable/` yet.

Install with the URL or local path your mirror provides, for example:

```bash
# x86_64, Python 3.12, torch-spyre-tagged build
uv pip install <mirror>/aiu-monitor/x86_64/stable/<version>/ibm_aiu_monitor-<version>+torch.spyre-py312-none-linux_x86_64.whl

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
