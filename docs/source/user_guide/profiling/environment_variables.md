# Environment Variables for Profiling

**Stack:** torch-spyre (new, Inductor-based).

Variables that affect profile capture, telemetry, and observability.
Debug-oriented variables (`TORCH_SPYRE_DEBUG`, `TORCH_COMPILE_DEBUG`,
`TORCHINDUCTOR_FORCE_DISABLE_CACHES`, `INDUCTOR_PROVENANCE`,
`TORCH_TRACE`) live under [Debugging](../debugging/index.md).

## Logging

| Variable | Effect |
|---|---|
| `SPYRE_INDUCTOR_LOG=1` | Enable Spyre-specific Inductor logging |
| `SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` | Set Spyre Inductor log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `SPYRE_LOG_FILE=path/to/file.log` | Redirect Spyre Inductor log output to a file |
| `TORCH_LOGS="+inductor"` | Verbose PyTorch Inductor logging |
| `TORCH_SPYRE_DOWNCAST_WARN=0` | Suppress `float32 → float16` downcast warnings |

## Compiler configuration

| Variable | Effect |
|---|---|
| `SENCORES=<1..32>` | Number of Spyre cores to target (default 32) |
| `LX_PLANNING=1` | Enable LX scratchpad memory planning |

## Device enumeration

Read by torch-spyre
([`spyre_device_enum.cpp`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/csrc/spyre_device_enum.cpp))
at startup to discover the Spyre devices visible to the process.

| Variable | Effect |
|---|---|
| `PCIDEVICE_IBM_COM_AIU_PF` | Comma-separated list of PCI bus IDs assigned to the container (set by the OpenShift AIU operator or manually) |
| `AIU_WORLD_RANK_<N>` | PCI bus ID bound to rank `N` |
| `SPYRE_VISIBLE_DEVICES` | Override the device list explicitly (takes priority over `PCIDEVICE_IBM_COM_AIU_PF`) |
| `LOCAL_RANK` | Per-process rank set by `torchrun`; used to select the device for each child process |

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
