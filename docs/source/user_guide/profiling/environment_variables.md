# Environment Variables for Profiling

**Stack:** torch-spyre (new, Inductor-based).

Variables that affect profile capture, telemetry, and observability.
Debug-oriented variables (`TORCH_SPYRE_DEBUG`, `TORCH_COMPILE_DEBUG`,
`TORCHINDUCTOR_FORCE_DISABLE_CACHES`, `INDUCTOR_PROVENANCE`,
`TORCH_TRACE`) live under [Debugging](../debugging/index.md); the FFDC
table below re-lists `TORCH_COMPILE_DEBUG` only to note its effect on
captured artifacts.

## Logging

| Variable | Effect |
|---|---|
| `SPYRE_INDUCTOR_LOG=1` | *Deprecated*. Use `TORCH_LOGS="torch_spyre.inductor"`. Enables Spyre-specific Inductor logging (INFO level) |
| `SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` | *Deprecated*. Use `TORCH_LOGS="+torch_spyre.inductor"`. Sets Spyre Inductor log verbosity to DEBUG |
| `SPYRE_LOG_FILE=path/to/file.log` | *Deprecated*. Mapped to the top-level `spyre` logger file handler. Redirects Spyre Inductor log output to a file |
| `TORCH_LOGS="+torch_spyre.inductor"` | Preferred logging control (DEBUG level). Accepts `torch_spyre.*` namespaces |
| `TORCH_LOGS="torch_spyre.inductor"` | Same as above but at INFO level (no `+` prefix) |
| `TORCH_LOGS="-torch_spyre.inductor"` | Sets to ERROR level (suppresses INFO/DEBUG) |
| `TORCH_LOGS="+inductor"` | Verbose PyTorch Inductor logging |
| `TORCH_SPYRE_DOWNCAST_WARN=0` | Suppress `int64 → int32` downcast warnings |

### Programmatic Configuration

For log levels not supported by `TORCH_LOGS` (WARNING, CRITICAL, DISABLED), use the
programmatic API:

```python
from torch_spyre import logging_config

# Set any log level programmatically
logging_config.set_log_level('spyre.inductor', 'CRITICAL')
logging_config.set_log_level('spyre.runtime', 'WARNING')
logging_config.disable('spyre.execution')  # DISABLED level

# Convenience functions
logging_config.enable('spyre.inductor')   # INFO level
```

**Per-pass DEBUG logging** requires setting both the log level and pass filter:

```python
from torch_spyre import logging_config

# Enable DEBUG level for passes
logging_config.set_log_level('spyre.inductor.passes', 'DEBUG')

# Configure which passes to log
logging_config.set_log_passes('all')                              # All passes
logging_config.set_log_passes('split_multi_ops,insert_restickify') # Specific passes
logging_config.set_log_passes('')                                  # Disable

# Query current configuration
level = logging_config.get_log_level('spyre.inductor.passes')
log_passes = logging_config.get_log_passes()
```

Available levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, `DISABLED`

**Note:** Use internal `spyre.*` namespace in programmatic calls, not `torch_spyre.*`.
The `torch_spyre.*` namespace is only for the `TORCH_LOGS` environment variable.

## Compiler configuration

| Variable | Effect |
|---|---|
| `SENCORES=<1..32>` | Number of Spyre cores to target (default 32) |

## FFDC (First Failure Data Capture)

| Variable | Effect |
|---|---|
| `TORCH_SPYRE_FFDC=1` | Opt in to automatic FFDC JSON reports on Spyre frontend-compile / backend-compile / runtime / unimplemented failures. Retrieve with `torch.spyre.get_diagnostic_report()`. Separate from `USE_SPYRE_PROFILER` (the `setup.py` Kineto build flag); this env var alone gates capture at runtime and is not set by default on pods. |
| `TORCH_COMPILE_DEBUG=1` | Optional. Writes `torch_compile_debug/` artifacts that FFDC links into `artifacts.paths` (see [FFDC user guide](ffdc.md)). Not required for capture. |
| `DUMP_SPYRE_CODE=1` | Optional. Emits `sdsc_*.json` and `*.mlir` bundle files that FFDC can reference. Not required for capture. |

See the [FFDC user guide](ffdc.md) for the full workflow, report locations,
and pod/CI usage.

## Device enumeration

Honored by the `flex` library itself (not read directly by torch-spyre)
when [`spyre_device_enum.cpp`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/csrc/spyre_device_enum.cpp)
calls `flex::getNumDevices()` to determine how many Spyre devices are
visible to the process:

| Variable | Effect |
|---|---|
| `FLEX_DEVICE` | Device type: `PF`, `VF`, or `MOCK`. Selects how device count is determined |
| `AIU_WORLD_SIZE` | Number of devices to use; caps the total device count (or is returned directly under `FLEX_DEVICE=MOCK`) |
| `SPYRE_DEVICES` | Comma-separated list of device indices to use (e.g., `0,2,3`); overrides the default enumeration |

Read directly by torch-spyre
([`spyre_guard.cpp`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/csrc/spyre_guard.cpp))
to pick the device for the current process:

| Variable | Effect |
|---|---|
| `LOCAL_RANK` | Per-process rank set by `torchrun`; used to select the device for each child process (defaults to 0 if unset) |

Set by the OpenShift AIU operator (or manually); not read directly by
torch-spyre's device-enumeration code:

| Variable | Effect |
|---|---|
| `PCIDEVICE_IBM_COM_AIU_PF` | Comma-separated list of PCI bus IDs assigned to the container; consumed by [`tests/oot_framework/run_test.sh`](https://github.com/torch-spyre/torch-spyre/blob/main/tests/oot_framework/run_test.sh) |
| `AIU_WORLD_RANK_<N>` | PCI bus ID bound to rank `N`; not consumed in-tree — it is scraped back out of pod logs after the fact by [`.github/scripts/parse_hw_failures.py`](https://github.com/torch-spyre/torch-spyre/blob/main/.github/scripts/parse_hw_failures.py) |

## Runtime / driver (for `aiu-smi` and `aiu-trace-analyzer`)

| Variable | Effect |
|---|---|
| `SENLIB_DEVEL_CONFIG_FILE=<path>` | Point the Spyre driver (`senlib`) at a config file enabling hardware-counter collection; required for `aiu-smi` |
| `DTCOMPILER_KEEP_EXPORT=true` | Keep compiler export directories around after a run; required for `aiu-smi` to report `rsvmem` and for `aiu-trace-analyzer` post-processing |
| `DEEPRT_EXPORT_DIR=<dir>` | Where the runtime / compiler write export artifacts; set to the same path in the workload and monitoring shells |
| `DTCOMPILER_EXPORT_DIR=<dir>` | Override the compiler export location (defaults to CWD when unset) |
| `DT_DEEPRT_VERBOSE=0` | Quiet runtime logs when capturing traces for `aiu-trace-analyzer` |

## Quick-reference recipes

### `aiu-smi` workload shell

```bash
export DTCOMPILER_KEEP_EXPORT=true
export SENLIB_DEVEL_CONFIG_FILE=$HOME/.local/etc/senlib_config_aiusmi.json
# Optional: co-locate compiler exports and aiu-smi lookups
export DEEPRT_EXPORT_DIR=$PWD
```

### `aiu-smi` monitoring shell (run in parallel)

```bash
export DEEPRT_EXPORT_DIR=$PWD   # matches the workload shell
aiu-smi
```
