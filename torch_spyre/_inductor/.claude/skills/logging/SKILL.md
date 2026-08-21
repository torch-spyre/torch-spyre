---
name: logging
description: "Use when you want log/debug output from torch-spyre (Python or C++) and aren't sure which TORCH_LOGS setting, component name, or env var to use, or when TORCH_LOGS raises 'Invalid log settings', logs silently don't appear, or a dynamic logger doesn't respond to TORCH_LOGS."
---

# Getting Log Output from torch-spyre (Unified Logging Framework)

Full reference: `docs/source/user_guide/debugging/unified_logging_framework.md`.

## Quick recipes

```bash
# Everything, at DEBUG
export TORCH_LOGS="+torch_spyre"

# Just the inductor compiler, at DEBUG
export TORCH_LOGS="+torch_spyre.inductor"

# Inductor at INFO, but silence its passes sub-component
export TORCH_LOGS="torch_spyre.inductor,-torch_spyre.inductor.passes"

# Runtime (C++: allocator/streams/distributed) at DEBUG
export TORCH_LOGS="+torch_spyre.runtime"

# Spyre + PyTorch's own inductor tracing together
export TORCH_LOGS="+torch_spyre,+inductor"
```

## The two rules that explain most failures

1. **Always spell the env var as `torch_spyre.*`, never bare `spyre.*`.**
   PyTorch validates every `TORCH_LOGS` entry with `find_spec()` before Spyre
   code runs; `spyre.*` isn't a real importable package and raises
   `Invalid log settings`. (Internally, everything — Python loggers, C++
   components, the programmatic API — uses `spyre.*`. Only the `TORCH_LOGS`
   string uses the `torch_spyre.*` form.)
2. **Prefix controls level, there is no `component:LEVEL` syntax** (colons
   aren't allowed by PyTorch's parser):
   - `+<component>` → DEBUG
   - `<component>` (no prefix) → INFO
   - `-<component>` → ERROR (suppress)

   A parent setting cascades to children unless a child entry overrides it,
   e.g. `+torch_spyre.inductor,-torch_spyre.inductor.passes` gives everything
   under inductor DEBUG except `passes`, which is ERROR.

## If logs still don't appear

- No prefix means INFO, not DEBUG — `logger.debug(...)` needs `+component`.
- **Dynamic loggers can't be targeted directly.** Anything created via
  `get_inductor_logger(name)` (most per-pass loggers — `lowering`,
  `dedup_constants`, `scheduler`, `work_division`, etc.) has no on-disk
  package stub, so naming it directly in `TORCH_LOGS`
  (`+torch_spyre.inductor.dedup_constants`) raises an exception at import
  time. Enable the parent instead: `+torch_spyre.inductor`. Full list of
  dynamic logger names and which file registers them: see "Dynamic Loggers"
  in the full reference doc.
- **C++ logs missing but Python logs work:** C++ config is pushed during
  `_lazy_init()` (first device op). If you only `import torch_spyre` without
  running anything on-device, call
  `torch_spyre.logging_config._sync_cpp_config()` explicitly.
- Legacy vars (`SPYRE_INDUCTOR_LOG`, `SPYRE_INDUCTOR_LOG_LEVEL`,
  `TORCH_SPYRE_DEBUG`, `SPYRE_LOG_FILE`) still work but emit deprecation
  warnings — see the migration table in the full reference doc for their
  `TORCH_LOGS` equivalents.

## Finer control than TORCH_LOGS allows (WARNING/CRITICAL, log-to-file)

`TORCH_LOGS` only reaches DEBUG/INFO/ERROR. For WARNING/CRITICAL/DISABLED,
per-pass filtering, or file output, use the programmatic API instead
(note: `spyre.*` namespace here, not `torch_spyre.*`):

```python
from torch_spyre import logging_config

logging_config.set_log_level("spyre.inductor.lowering", "DEBUG")
logging_config.set_log_passes("split_multi_ops,insert_restickify")  # or "all"
logging_config.set_log_file("/tmp/spyre.log")   # Python + C++ both write here
logging_config.get_effective_config()           # introspect current levels
```

## Predefined components

| Component | Covers |
| --- | --- |
| `spyre` | root — everything |
| `spyre.inductor` | all compiler passes/codegen (`torch_spyre/_inductor/`) |
| `spyre.inductor.lowering` | ATen → Spyre IR lowering |
| `spyre.inductor.codegen` | code generation |
| `spyre.inductor.stickify` | stickification passes |
| `spyre.inductor.passes` | general compiler passes |
| `spyre.runtime` | C++ runtime: allocator, streams, distributed |

Everything else under `spyre.inductor.*` (e.g. `scheduler`, `padding`,
`work_division`, `dedup_constants`) is a dynamic logger reachable only via
the `spyre.inductor` parent or the programmatic API — see above.

## Adding logging to new code

```python
from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("my_new_pass")  # -> spyre.inductor.my_new_pass
logger.debug("detail: %s", value)
```

```cpp
#include "logging_config.h"

SPYRE_LOG("spyre.runtime", INFO) << "New feature initialized";
// or, for the common runtime component:
SPYRE_RUNTIME_DEBUG() << "Allocated " << nbytes << " bytes";
```

`SPYRE_LOG`/`SPYRE_RUNTIME_*` are zero-cost when disabled — safe in hot
paths. For expensive diagnostic computation, gate it explicitly:

```cpp
if (SPYRE_LOG_ENABLED("spyre.runtime", torch_spyre::logging::LogLevel::DEBUG)) {
    SPYRE_RUNTIME_DEBUG() << expensive_state_dump();
}
```
