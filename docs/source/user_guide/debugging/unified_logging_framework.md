# Spyre Unified Logging Framework (ULF) Guide

<!-- markdownlint-disable MD024 -->

## Table of Contents

- [Spyre Unified Logging Framework (ULF) Guide](#spyre-unified-logging-framework-ulf-guide)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [1. Overview](#1-overview)
    - [Migrating from Legacy Variables](#migrating-from-legacy-variables)
  - [2. TORCH\_LOGS Syntax for Spyre Components](#2-torch_logs-syntax-for-spyre-components)
    - [Syntax Forms](#syntax-forms)
    - [Namespace Normalization](#namespace-normalization)
    - [Hierarchy Rules](#hierarchy-rules)
    - [Common Recipes](#common-recipes)
  - [3. Component Names](#3-component-names)
    - [The Naming Rule](#the-naming-rule)
    - [Why torch\_spyre.\* in TORCH\_LOGS and spyre.\* Internally](#why-torch_spyre-in-torch_logs-and-spyre-internally)
  - [4. Available Components](#4-available-components)
    - [Predefined Components](#predefined-components)
    - [Dynamic Loggers](#dynamic-loggers)
  - [5. Python Usage](#5-python-usage)
    - [Getting a Logger](#getting-a-logger)
    - [Adding Logging to New Code](#adding-logging-to-new-code)
    - [Log Format](#log-format)
    - [Enabling multiple components](#enabling-multiple-components)
    - [File Output](#file-output)
  - [6. C++ Usage](#6-c-usage)
    - [Architecture {#architecture-c}](#architecture-architecture-c)
    - [Include Files](#include-files)
    - [Available Macros](#available-macros)
    - [Adding Logging to New C++ Code](#adding-logging-to-new-c-code)
    - [Using the Macros](#using-the-macros)
    - [The Logger Class](#the-logger-class)
    - [Performance: Level-Gated Logging](#performance-level-gated-logging)
    - [Legacy DEBUGINFO](#legacy-debuginfo)
    - [Log Format (C++)](#log-format-c)
    - [File Output (C++)](#file-output-c)
  - [7. Python–C++ Synchronization](#7-pythonc-synchronization)
    - [Initialization Flow](#initialization-flow)
    - [Runtime Reconfiguration](#runtime-reconfiguration)
    - [Generation Counter and Thread-Local Cache](#generation-counter-and-thread-local-cache)
  - [8. Troubleshooting](#8-troubleshooting)
  - [Appendix A: Python Programmatic API](#appendix-a-python-programmatic-api)
    - [Setting Levels](#setting-levels)
    - [Per-Pass Logging](#per-pass-logging)
    - [Introspection](#introspection)
    - [File Output (API)](#file-output-api)
    - [Reset (for testing)](#reset-for-testing)
  - [Appendix B: C++ Direct API](#appendix-b-c-direct-api)
    - [LoggingConfig Singleton](#loggingconfig-singleton)
    - [Checking if a Level is Enabled](#checking-if-a-level-is-enabled)
    - [Querying Configuration](#querying-configuration)
    - [Setting Levels (testing only)](#setting-levels-testing-only)
    - [Output Sink Management](#output-sink-management)
    - [Thread Safety](#thread-safety)
  - [Appendix C: C++ LogLevel Enum Reference](#appendix-c-c-loglevel-enum-reference)
  - [Appendix D: Design Rationale — Naming](#appendix-d-design-rationale--naming)

---

## Quick Start

```bash
# Enable all spyre logging at DEBUG
export TORCH_LOGS="+torch_spyre"

# Enable just the inductor compiler logging at DEBUG
export TORCH_LOGS="+torch_spyre.inductor"

# Enable inductor at INFO (no prefix), suppress passes to ERROR
export TORCH_LOGS="torch_spyre.inductor,-torch_spyre.inductor.passes"

# Enable spyre runtime at DEBUG only
export TORCH_LOGS="+torch_spyre.runtime"

# Full debug — all spyre + PyTorch inductor
export TORCH_LOGS="+torch_spyre,+inductor"
```

See [Common Recipes](#common-recipes) for more configuration patterns.

---

## 1. Overview

The **Unified Logging Framework (ULF)** consolidates all torch-spyre logging
(Python and C++) behind a hierarchical component system configured via the
`TORCH_LOGS` environment variable. ULF replaces the legacy
`SPYRE_INDUCTOR_LOG`, `SPYRE_INDUCTOR_LOG_LEVEL`, and `TORCH_SPYRE_DEBUG`
variables with a single, per-component configuration that is consistent
across Python and C++ code paths.

A **component** is a dot-separated identifier (e.g.,
`spyre.inductor.lowering`) that names a logical subsystem within
torch-spyre. Components form a hierarchy: setting a level on a parent
propagates to all its children unless a more-specific child entry overrides
it.

Spyre components coexist with PyTorch's own `TORCH_LOGS` components
(e.g., `+inductor`, `dynamo`). Non-spyre entries are handled by PyTorch's
logging system; `torch_spyre.*` entries are intercepted, normalized to
`spyre.*`, and routed through ULF.

Configuration priority (highest wins):

1. `TORCH_LOGS` environment variable (primary, recommended)
2. Legacy env vars (deprecated, emit warnings)
3. Programmatic API (`logging_config.set_log_level(...)`)
4. Defaults (all components at WARNING)

### Migrating from Legacy Variables

The following legacy environment variables are deprecated and emit warnings
on use. Migrate to their `TORCH_LOGS` equivalents:

| Old variable | New equivalent |
| --- | --- |
| `SPYRE_INDUCTOR_LOG=1` | `TORCH_LOGS="torch_spyre.inductor"` (INFO) |
| `SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` | `TORCH_LOGS="+torch_spyre.inductor"` (DEBUG) |
| `TORCH_SPYRE_DEBUG=1` | `TORCH_LOGS="+torch_spyre"` (DEBUG) |
| `SPYRE_LOG_FILE=/path` | `logging_config.set_log_file("/path")` |

---

## 2. TORCH_LOGS Syntax for Spyre Components

### Syntax Forms

`TORCH_LOGS` accepts a **comma-separated list** of entries. Entries that
begin with `torch_spyre` are handled by ULF. Each spyre entry takes one of
three forms:

| Syntax | Effect | Example |
| --- | --- | --- |
| `+<component>` | Enable at **DEBUG** | `+torch_spyre.inductor` |
| `<component>` (no prefix) | Enable at **INFO** | `torch_spyre.inductor` |
| `-<component>` | Suppress (level = **ERROR**) | `-torch_spyre.inductor.passes` |

> **Why `torch_spyre.*` and not bare `spyre.*`?** PyTorch's `TORCH_LOGS`
> parser validates entries using `importlib.util.find_spec()` during
> `import torch`. A bare `spyre.*` target fails validation (it is not a
> real importable package) and causes PyTorch to raise
> `Invalid log settings` before any Spyre code runs. The `torch_spyre.*`
> namespace stubs exist on disk to pass this validation. Internally, ULF
> normalizes `torch_spyre.*` onto `spyre.*`.
>
> **Note:** The `component:LEVEL` syntax (e.g., `spyre.inductor:DEBUG`) is
> **not supported**. PyTorch's `TORCH_LOGS` parser does not permit colons.
> For finer-grained level control (e.g., setting a specific component to
> WARNING or CRITICAL), use the programmatic API described in
> [Appendix A](#appendix-a-python-programmatic-api).

Non-spyre entries in the same `TORCH_LOGS` value are passed through to
PyTorch's logging system unchanged:

```bash
# Both PyTorch inductor tracing AND spyre runtime:
export TORCH_LOGS="+inductor,+torch_spyre.runtime"
```

### Namespace Normalization

Users spell `torch_spyre.*` on the command line. Internally, ULF normalizes
this to `spyre.*` so the configured level lands on the Python logger and
C++ component that actually emit records:

| CLI target (`TORCH_LOGS`) | Internal logger |
| --- | --- |
| `torch_spyre` | `spyre` (root) |
| `torch_spyre.inductor` | `spyre.inductor` |
| `torch_spyre.inductor.passes` | `spyre.inductor.passes` |
| `torch_spyre.runtime` | `spyre.runtime` |

The programmatic API uses the **internal** `spyre.*` namespace directly —
do not pass `torch_spyre.*` to `logging_config.set_log_level()`.

### Hierarchy Rules

A parent setting propagates to children unless a more-specific entry
overrides. Setting `+torch_spyre.inductor` cascades DEBUG down to all child
components like `spyre.inductor.codegen`, `spyre.inductor.lowering`, etc.
A child override (`-torch_spyre.inductor.passes`) takes precedence over the
inherited parent level.

### Common Recipes

**Enable all spyre logging at DEBUG:**

```bash
export TORCH_LOGS="+torch_spyre"
```

| Component | Effective level |
| --- | --- |
| all `spyre.*` | DEBUG |

**Enable only inductor logging at INFO:**

```bash
export TORCH_LOGS="torch_spyre.inductor"
```

| Component | Effective level |
| --- | --- |
| `spyre.inductor.*` | INFO |
| `spyre.runtime` | WARNING (unaffected) |

**Enable inductor at DEBUG, suppress passes:**

```bash
export TORCH_LOGS="+torch_spyre.inductor,-torch_spyre.inductor.passes"
```

| Component | Effective level |
| --- | --- |
| `spyre.inductor.lowering` | DEBUG |
| `spyre.inductor.passes` | ERROR |
| `spyre.runtime` | WARNING (unaffected) |

**Enable spyre runtime and PyTorch inductor together:**

```bash
export TORCH_LOGS="+inductor,+torch_spyre.runtime"
```

| Component | Effective level |
| --- | --- |
| PyTorch `inductor` | DEBUG (PyTorch convention) |
| `spyre.runtime` | DEBUG |

**Suppress all spyre logging (ERROR only):**

```bash
export TORCH_LOGS="-torch_spyre"
```

| Component | Effective level |
| --- | --- |
| all `spyre.*` | ERROR |

---

## 3. Component Names

### The Naming Rule

ULF uses two namespaces — a **public CLI namespace** (`torch_spyre.*`) for
`TORCH_LOGS` and an **internal namespace** (`spyre.*`) for everything else:

| Context | Namespace | Example |
| --- | --- | --- |
| `TORCH_LOGS` environment variable | `torch_spyre.*` | `TORCH_LOGS="+torch_spyre.inductor"` |
| Python `logging` logger | `spyre.*` | `logging.getLogger("spyre.inductor.lowering")` |
| C++ component string | `spyre.*` | `SPYRE_LOG("spyre.runtime", INFO)` |
| Programmatic API | `spyre.*` | `logging_config.set_log_level("spyre.inductor", "DEBUG")` |
| Log output | `spyre.*` | `[INFO] [spyre.inductor.lowering] ...` |

The normalization layer translates `torch_spyre.*` → `spyre.*` at parse
time. Once past the environment variable, the internal `spyre.*` name is
used everywhere.

### Why torch_spyre.\* in TORCH_LOGS and spyre.\* Internally

The Python **package** is named `torch_spyre` (installed as `torch-spyre`),
and source files live under `torch_spyre/_inductor/...`. The logging
namespace is `spyre.*` — deliberately shorter:

1. **Brevity** — `spyre.inductor` is shorter than `torch_spyre._inductor`
   and appears in every log line.
2. **Clean hierarchy** — the `_inductor` internal package path and leading
   `torch_` prefix are implementation details that don't belong in log
   output or env vars.
3. **Consistency** — PyTorch itself uses short, clean names (`inductor`,
   `dynamo`) rather than exposing `torch._inductor`.

However, the `TORCH_LOGS` CLI must use `torch_spyre.*` because PyTorch
validates entries with `importlib.util.find_spec()` before Spyre code runs.
Namespace stub packages (`torch_spyre/inductor/`, `torch_spyre/runtime/`,
etc.) exist solely to pass this validation.

---

## 4. Available Components

### Predefined Components

These components are defined in `DEFAULT_LOG_LEVELS` in
`torch_spyre/logging_config.py` and can be targeted directly in
`TORCH_LOGS`:

| Component | What it controls | Primary source |
| --- | --- | --- |
| `spyre` | Root — all Spyre logging | — |
| `spyre.inductor` | All Inductor compiler passes and codegen | `torch_spyre/_inductor/` |
| `spyre.inductor.lowering` | Op lowering (ATen → Spyre IR) | `_inductor/lowering.py` |
| `spyre.inductor.codegen` | Code generation (parent) | `_inductor/codegen/bundle.py` |
| `spyre.inductor.stickify` | Tensor stickification passes | `_inductor/insert_restickify.py` |
| `spyre.inductor.passes` | General compiler passes | `_inductor/passes.py` |
| `spyre.runtime` | C++ runtime (allocator, streams, distributed) | `torch_spyre/csrc/` |
| `spyre.execution` | (reserved) | — |
| `spyre.device` | (reserved) | — |

### Dynamic Loggers

Not all loggers are listed in `DEFAULT_LOG_LEVELS`. Any call to
`get_inductor_logger(name)` creates a Python logger named
`spyre.inductor.<name>` on first use. These **dynamic loggers** do not
require pre-registration — they inherit their log level from the nearest
configured ancestor in the hierarchy (typically `spyre.inductor`).

Example — `compile_fx_wrapper` in `torch_spyre/_inductor/__init__.py`:

```python
logger = get_inductor_logger("compile_fx_wrapper")
# Creates: logging.getLogger("spyre.inductor.compile_fx_wrapper")
```

This logger is not in `DEFAULT_LOG_LEVELS`, but it responds to parent
configuration:

```bash
# Via parent — enables all inductor loggers including compile_fx_wrapper:
export TORCH_LOGS="+torch_spyre.inductor"
```

> **Warning: Do not use dynamic logger names directly in `TORCH_LOGS`.**
> PyTorch validates every `TORCH_LOGS` entry with
> `importlib.util.find_spec()` at import time. Only components that have a
> corresponding importable package stub pass validation. Dynamic loggers
> like `torch_spyre.inductor.dedup_constants` have no on-disk package and
> will cause PyTorch to raise an exception. To control dynamic loggers,
> enable their parent (e.g., `+torch_spyre.inductor`) or use the
> programmatic API (`logging_config.set_log_level("spyre.inductor.dedup_constants", "DEBUG")`).

The full list of dynamic loggers in the codebase:

| `get_inductor_logger()` arg | Internal logger name | Source file |
| --- | --- | --- |
| `"compile_fx_wrapper"` | `spyre.inductor.compile_fx_wrapper` | `_inductor/__init__.py` |
| `"model_utils"` | `spyre.inductor.model_utils` | `model_utils.py` |
| `"enforce_indirect_access_layout"` | `spyre.inductor.enforce_indirect_access_layout` | `_inductor/enforce_indirect_access_layout.py` |
| `"HBM_POOL_PLANNING"` | `spyre.inductor.HBM_POOL_PLANNING` | `_inductor/hbm_pool_planning.py` |
| `"optimize_restickify"` | `spyre.inductor.optimize_restickify` | `_inductor/optimize_restickify.py` |
| `"insert_restickify"` | `spyre.inductor.insert_restickify` | `_inductor/insert_restickify.py` |
| `"propagate_hints"` | `spyre.inductor.propagate_hints` | `_inductor/propagate_hints.py` |
| `"dedup_constants"` | `spyre.inductor.dedup_constants` | `_inductor/dedup_constants.py` |
| `"split_multi_ops"` | `spyre.inductor.split_multi_ops` | `_inductor/split_multi_ops.py` |
| `"ir"` | `spyre.inductor.ir` | `_inductor/ir.py` |
| `"padding"` | `spyre.inductor.padding` | `_inductor/padding.py` |
| `"scheduler"` | `spyre.inductor.scheduler` | `_inductor/scheduler.py` |
| `"pass_utils"` | `spyre.inductor.pass_utils` | `_inductor/pass_utils.py` |
| `"sdsc_compile"` | `spyre.inductor.sdsc_compile` | `_inductor/codegen/bundle.py` |
| `"codegen.superdsc"` | `spyre.inductor.codegen.superdsc` | `_inductor/codegen/superdsc.py` |
| `"kernel_runner"` | `spyre.inductor.kernel_runner` | `_inductor/execution/kernel_runner.py` |
| `"scratchpad.allocator"` | `spyre.inductor.scratchpad.allocator` | `_inductor/scratchpad/allocator.py` |
| `"scratchpad.plan_solver"` | `spyre.inductor.scratchpad.plan_solver` | `_inductor/scratchpad/plan_solver.py` |
| `"scratchpad.greedy_solver"` | `spyre.inductor.scratchpad.greedy_solver` | `_inductor/scratchpad/greedy_solver.py` |
| `"assign_dim_hints"` | `spyre.inductor.assign_dim_hints` | `_inductor/wsr/coarse_tile_hints.py` |
| `"coarse_tile"` | `spyre.inductor.coarse_tile` | `_inductor/wsr/coarse_tile.py` |
| `"propagate_named_dims"` | `spyre.inductor.propagate_named_dims` | `_inductor/wsr/propagate_named_dims.py` |
| `"span_overflow_hint_analysis"` | `spyre.inductor.span_overflow_hint_analysis` | `_inductor/wsr/span_overflow_hint_analysis.py` |
| `"propagate_layouts"` | `spyre.inductor.propagate_layouts` | `_inductor/propagate_layouts.py` |
| `"spyre_kernel"` | `spyre.inductor.spyre_kernel` | `_inductor/spyre_kernel.py` |
| `"work_division"` | `spyre.inductor.work_division` | `_inductor/work_division.py` |

All of these respond to `TORCH_LOGS="+torch_spyre.inductor"` (parent inheritance).

---

## 5. Python Usage

### Getting a Logger

```python
from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("lowering")  # → "spyre.inductor.lowering"
logger.info("mm: x%s @ y%s -> %s", x.shape, y.shape, out.shape)
```

### Adding Logging to New Code

```python
# In torch_spyre/_inductor/my_new_pass.py:
from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("my_new_pass")

# Use standard Python logging methods:
logger.debug("Detailed trace: %s", detail)
logger.warning("Unexpected condition: %s", condition)
```

The logger `spyre.inductor.my_new_pass` is automatically controlled by
`TORCH_LOGS="+torch_spyre.inductor"` via parent inheritance.

### Log Format

```text
[INFO] [spyre.inductor.lowering] mm: x[2,3] @ y[3,4] -> [2,4]
```

### Enabling multiple components

```bash
export TORCH_LOGS="+torch_spyre.inductor,-torch_spyre.inductor.passes"
```

```python
from torch_spyre._inductor.logging_utils import get_inductor_logger

lowering_log = get_inductor_logger("lowering")
passes_log = get_inductor_logger("passes")
stickify_log = get_inductor_logger("stickify")

lowering_log.debug("This prints (DEBUG enabled by +torch_spyre.inductor)")
passes_log.info("This does NOT print (ERROR level set by -)")
stickify_log.debug("This prints (DEBUG inherited from +torch_spyre.inductor)")
```

### File Output

Both Python and C++ logging can be directed to a file. The C++ sink
follows Python's configuration automatically:

```python
from torch_spyre import logging_config
logging_config.set_log_file("/tmp/spyre.log")
```

This configures the top-level `spyre` logger's file handler (Python) and
calls `LoggingConfig::set_log_file()` on the C++ side — both languages write
to the same file.

---

## 6. C++ Usage

### Architecture {#architecture-c}

The C++ logging system is implemented in `torch_spyre/csrc/` and consists of
three layers:

| File | Role |
| --- | --- |
| `logging_config.h` / `.cpp` | `LoggingConfig` singleton, `Logger` class, convenience macros |
| `logging_bindings.h` / `.cpp` | pybind11 bindings exposing C++ logging to Python |
| `logging_legacy.h` | `DEBUGINFO(...)` compatibility shim |
| `logging.h` / `.cpp` | Umbrella header re-exporting the public interface |

All C++ logging state lives in the `torch_spyre::logging` namespace.

### Include Files

For most C++ code in `torch_spyre/csrc/`, include the umbrella header:

```cpp
#include "logging.h"
```

This gives you access to:

- All `SPYRE_LOG` / `SPYRE_RUNTIME_*` macros
- The `DEBUGINFO(...)` legacy macro
- The `Logger`, `LoggingConfig`, and `LogLevel` types

If you only need the new logging system (no legacy `DEBUGINFO`), you can
include `logging_config.h` directly.

### Available Macros

| Macro | Component | Level |
| --- | --- | --- |
| `SPYRE_RUNTIME_DEBUG()` | `spyre.runtime` | DEBUG |
| `SPYRE_RUNTIME_INFO()` | `spyre.runtime` | INFO |
| `SPYRE_RUNTIME_WARNING()` | `spyre.runtime` | WARNING |
| `SPYRE_RUNTIME_ERROR()` | `spyre.runtime` | ERROR |
| `SPYRE_RUNTIME_CRITICAL()` | `spyre.runtime` | CRITICAL |
| `SPYRE_LOG(component, LEVEL)` | any | any |
| `SPYRE_LOG_ENABLED(component, level)` | any | any (returns bool) |
| `DEBUGINFO(...)` | `spyre.runtime` | DEBUG (legacy) |

The `SPYRE_LOG` macro is **zero-cost when disabled**: it checks
`SPYRE_LOG_ENABLED` first (a thread-local cache hit) and short-circuits
the entire `Logger` construction and stream operations when the level is
not enabled.

### Adding Logging to New C++ Code

```cpp
#include "logging_config.h"

// Use SPYRE_LOG with your component name:
SPYRE_LOG("spyre.runtime", INFO) << "New feature initialized";

// For conditional expensive computation:
if (SPYRE_LOG_ENABLED("spyre.runtime", torch_spyre::logging::LogLevel::DEBUG)) {
    auto stats = compute_expensive_stats();
    SPYRE_RUNTIME_DEBUG() << "Stats: " << stats;
}
```

### Using the Macros

```cpp
#include "logging_config.h"

SPYRE_RUNTIME_DEBUG() << "Allocated " << nbytes << " bytes";
SPYRE_RUNTIME_INFO() << "Kernel launched on device " << dev_id;

// Generic form for any component:
SPYRE_LOG("spyre.inductor.codegen", DEBUG) << "Generating op: " << op_name;
```

Example — enable the runtime component and observe output:

```bash
export TORCH_LOGS="+torch_spyre.runtime"
```

```cpp
#include "logging_config.h"

// With TORCH_LOGS="+torch_spyre.runtime" (sets spyre.runtime to DEBUG):
SPYRE_RUNTIME_DEBUG() << "This prints (DEBUG enabled by +)";
SPYRE_RUNTIME_INFO() << "This also prints (INFO < DEBUG threshold)";
SPYRE_LOG("spyre.inductor.lowering", INFO) << "This does NOT print (no entry, defaults to WARNING)";
```

### The Logger Class

For advanced use cases, you can instantiate the `Logger` class directly
rather than using macros:

```cpp
#include "logging_config.h"

using torch_spyre::logging::Logger;
using torch_spyre::logging::LogLevel;

void my_function() {
    Logger log("spyre.runtime", LogLevel::INFO);
    if (log.is_enabled()) {
        log.info() << "Kernel ready: " << kernel_name;
    }
}
```

The `Logger` class provides stream methods for each level: `debug()`,
`info()`, `warning()`, `error()`, and `critical()`. Each returns a
`LogStream` RAII object that:

1. Buffers the message in a thread-local `std::ostringstream`
2. On destruction, if enabled, formats and emits the complete log record
   as a single atomic write

A fast-path constructor (`Logger::AlreadyEnabled`) is used by the
`SPYRE_LOG` macro to avoid the redundant `get_log_level()` call when the
macro has already verified the level is enabled.

### Performance: Level-Gated Logging

The `SPYRE_LOG` macro uses an `if`-gate pattern so that when logging is
disabled, **no `Logger` object is constructed and no stream operations
execute**:

```cpp
// Macro expansion (simplified):
if (auto enabled = SPYRE_LOG_ENABLED(component, level); !enabled) {
    /* nothing */
} else
    Logger(component, level, AlreadyEnabled{}).level() << ...
```

This means you can freely place `SPYRE_LOG` calls in hot paths without
measurable overhead when the component is at its default WARNING level.

For expensive argument computation, use the explicit check:

```cpp
if (SPYRE_LOG_ENABLED("spyre.runtime", torch_spyre::logging::LogLevel::DEBUG)) {
    std::string dump = expensive_state_dump();
    SPYRE_RUNTIME_DEBUG() << dump;
}
```

### Legacy DEBUGINFO

```cpp
#include "logging.h"

DEBUGINFO("Allocating ", nbytes, " bytes on Spyre", device);
// Equivalent to: SPYRE_RUNTIME_DEBUG() << __func__ << ": Allocating " << ...
```

`DEBUGINFO` maps to component `spyre.runtime` at DEBUG level. It
automatically prepends the calling function name (`__func__`).

The `DEBUGINFO` macro is defined in `logging_legacy.h` and delegates to
the new `Logger` class internally — it is not a separate logging system.
It exists solely for backward compatibility with existing C++ code. New
code should use `SPYRE_LOG` or `SPYRE_RUNTIME_*` macros instead.

### Log Format (C++)

```text
[DEBUG] [spyre.runtime] 2026-08-06 14:30:22 allocate_tensor: Allocated 1024 bytes
```

The C++ format includes a timestamp (unlike Python's default format):

```text
[LEVEL] [component] YYYY-MM-DD HH:MM:SS message
```

The timestamp uses `localtime_r` (thread-safe) and the system's local
timezone.

### File Output (C++)

C++ log output follows the file path configured from Python. There is no
separate C++ file configuration — call `logging_config.set_log_file("/path")`
from Python and C++ output goes to the same file.
See [File Output in section 5](#file-output).

Internally, `LoggingConfig::set_log_file()` opens the file in append mode
and atomically swaps the sink pointer. When no file is configured, output
goes to `std::cerr`.

---

## 7. Python–C++ Synchronization

### Initialization Flow

The C++ `LoggingConfig` singleton does not read environment variables
itself. Instead, Python drives all configuration:

```text
torch_spyre import
    → logging_config.py module executes initialize()
        → _resolve_config() parses TORCH_LOGS + legacy vars
        → configure_python_logging() sets up Python loggers
    → _lazy_init() (first device op or explicit init)
        → imports torch_spyre._C (loads the pybind11 extension)
        → logging_config._sync_cpp_config()
            → LoggingConfig::initialize_from_python(config)
            → LoggingConfig::set_log_file(path)
```

Before `_sync_cpp_config()` is called, C++ code that logs will see all
components at WARNING (the uninitialized default). In practice this is
safe because:

1. C++ runtime code runs only after `_lazy_init()` starts the runtime
2. `_sync_cpp_config()` is called within `_lazy_init()` before
   `start_runtime()`

### Runtime Reconfiguration

When you call `logging_config.set_log_level()` from Python at runtime,
it automatically pushes the updated configuration to C++:

```python
from torch_spyre import logging_config
logging_config.set_log_level("spyre.runtime", "DEBUG")
# C++ LoggingConfig now also has spyre.runtime=DEBUG
```

The sync is immediate — C++ threads see the new level on their next
log-check (after the generation counter propagates to their thread-local
cache).

### Generation Counter and Thread-Local Cache

The C++ `LoggingConfig` uses a lock-free caching strategy for
steady-state performance:

1. **Generation counter** (`std::atomic<uint64_t>`) — incremented on
   every `set_log_level` or `initialize_from_python` call
2. **Thread-local cache** — each thread maintains a 4-slot direct-mapped
   cache of `(component, level, generation)` tuples
3. **Fast path** — on a cache hit (same component + same generation),
   `get_log_level()` returns immediately with no lock or hash-map lookup
4. **Slow path** — on a cache miss, acquires a shared lock and walks the
   hierarchy; the result is cached for subsequent calls

This means that in the common case (configuration is stable, threads
repeatedly check the same few components), the overhead of
`SPYRE_LOG_ENABLED` is a thread-local array comparison — no atomics,
no locking, no memory allocation.

---

## 8. Troubleshooting

**Logs don't appear:**

- Verify the component is enabled. `TORCH_LOGS="torch_spyre.inductor"`
  (no prefix) sets level to INFO, so `logger.debug(...)` calls won't
  appear. Use `+torch_spyre.inductor` for DEBUG-level output, or use the
  programmatic API
  (see [Appendix A](#appendix-a-python-programmatic-api)).
- Use the `torch_spyre.*` namespace in `TORCH_LOGS`, not bare `spyre.*`.
  PyTorch rejects `spyre.*` during validation.
- **Do not use `component:LEVEL` syntax** — PyTorch's `TORCH_LOGS` parser
  does not permit colons.

**C++ logs don't appear but Python logs do:**

- Ensure the runtime has been initialized. C++ logging config is pushed
  during `_lazy_init()`. If you only import `torch_spyre` without
  triggering device operations, call `logging_config._sync_cpp_config()`
  explicitly after importing `torch_spyre._C`.

**Dynamic logger not responding:**

- Dynamic loggers inherit from their nearest configured ancestor. Enable
  the parent to reach all children:
  `TORCH_LOGS="+torch_spyre.inductor"`
- **Do not target dynamic loggers directly in `TORCH_LOGS`** (e.g.,
  `TORCH_LOGS="+torch_spyre.inductor.dedup_constants"`). They have no
  on-disk package stub, so PyTorch's `find_spec()` validation will raise
  an exception at import time. Use the parent component or the
  programmatic API instead.

**Legacy env var deprecation warnings:**

- See [Migrating from Legacy Variables](#migrating-from-legacy-variables) in
  section 1 for the equivalent `TORCH_LOGS` settings.

**File output not working:**

- File configuration must happen before C++ threads start logging. Call
  `logging_config.set_log_file("/path")` early in your script, before any
  compiled model execution.
- The file is opened in append mode. Verify the path is writable and the
  directory exists.

**Thread-safety concerns:**

- C++ log *reads* (level checks) are lock-free in steady state.
- C++ log *writes* (message emission) use single-write semantics and are
  safe on Linux/POSIX (serialized via `flockfile` internally).
- Reconfiguring the log file path (`set_log_file`) while C++ threads are
  actively logging is **not safe** — do it at initialization time only.

---

## Appendix A: Python Programmatic API

The `torch_spyre.logging_config` module provides runtime control over logging
configuration. Use this API when you need finer-grained control than what
`TORCH_LOGS` offers — for example, setting individual components to WARNING
or CRITICAL level.

**Important:** The programmatic API uses the internal `spyre.*` namespace,
not `torch_spyre.*`. The `torch_spyre.*` prefix is only for the `TORCH_LOGS`
environment variable.

### Setting Levels

```python
from torch_spyre import logging_config

# Set an explicit level (any valid level name)
logging_config.set_log_level("spyre.inductor.lowering", "DEBUG")

# Shorthand: enable at INFO
logging_config.enable("spyre.runtime")

# Shorthand: disable entirely
logging_config.disable("spyre.inductor.passes")
```

Valid level names: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`,
`DISABLED`

### Per-Pass Logging

For detailed compiler pass output, set the log level AND configure which
passes emit output:

```python
from torch_spyre import logging_config

# Enable DEBUG level for passes
logging_config.set_log_level("spyre.inductor.passes", "DEBUG")

# Configure which passes to log
logging_config.set_log_passes("all")                                # All passes
logging_config.set_log_passes("split_multi_ops,insert_restickify")  # Specific passes
logging_config.set_log_passes("")                                   # Disable

# Query current configuration
logging_config.get_log_passes()  # → "all", "split_multi_ops", or ""
```

### Introspection

```python
from torch_spyre import logging_config

# Get effective config for all components
logging_config.get_effective_config()
# → {"spyre": "WARNING", "spyre.inductor": "INFO", ...}

# Get source of a component's config
logging_config.get_config_source("spyre.inductor")
# → "TORCH_LOGS" | "legacy:SPYRE_INDUCTOR_LOG" | "legacy:TORCH_SPYRE_DEBUG" | "programmatic" | "default"

# List all predefined components
logging_config.list_components()
# → ["spyre", "spyre.inductor", "spyre.inductor.lowering", ...]
```

### File Output (API)

```python
from torch_spyre import logging_config

# Direct all logging (Python + C++) to a file
logging_config.set_log_file("/tmp/spyre.log")

# Query current setting
logging_config.get_log_file()  # → "/tmp/spyre.log" or None
```

### Reset (for testing)

```python
from torch_spyre import logging_config

# Re-read environment variables and reinitialize all state
logging_config.reset()
```

This also pushes the refreshed configuration to C++ via
`_sync_cpp_config()`, ensuring both Python and C++ are in sync after a
test modifies environment variables.

---

## Appendix B: C++ Direct API

The C++ logging API is defined in `torch_spyre/csrc/logging_config.h`. Most
developers will use the `SPYRE_LOG` / `SPYRE_RUNTIME_*` macros (section 6).
The direct API below is for advanced use cases.

### LoggingConfig Singleton

```cpp
#include "logging_config.h"

using torch_spyre::logging::LoggingConfig;

// Access the global singleton
LoggingConfig& config = LoggingConfig::instance();
```

The singleton is constructed on first access (Meyers' singleton pattern)
and persists for the lifetime of the process.

### Checking if a Level is Enabled

```cpp
#include "logging_config.h"

using torch_spyre::logging::LoggingConfig;
using torch_spyre::logging::LogLevel;

if (LoggingConfig::instance().is_enabled("spyre.runtime", LogLevel::DEBUG)) {
    // Only compute expensive diagnostics when DEBUG is active
    auto stats = compute_memory_stats();
    SPYRE_RUNTIME_DEBUG() << "Memory: " << stats;
}
```

The `is_enabled()` method is inlined and delegates to `get_log_level()`,
which hits the thread-local cache on the fast path.

### Querying Configuration

```cpp
// Get the effective level for a component
LogLevel level = LoggingConfig::instance().get_log_level("spyre.runtime");

// List all configured components
std::vector<std::string> components = LoggingConfig::instance().get_components();
```

### Setting Levels (testing only)

```cpp
// Override a component's level at runtime
LoggingConfig::instance().set_log_level("spyre.runtime", LogLevel::DEBUG);
```

This increments the generation counter, invalidating all thread-local
caches. Subsequent `get_log_level()` calls will re-resolve from the
updated config map.

### Output Sink Management

```cpp
// Redirect all C++ log output to a file (append mode)
LoggingConfig::instance().set_log_file("/tmp/spyre_debug.log");

// Revert to stderr
LoggingConfig::instance().set_log_file("");

// Access the current output stream (for direct writes — not typical)
std::ostream& out = LoggingConfig::instance().sink();
```

The `sink()` accessor is lock-free (atomic pointer read). It returns
`std::cerr` when no file is configured, or the active `std::ofstream`
otherwise.

### Thread Safety

| Operation | Mechanism | Contention |
| --- | --- | --- |
| Config reads (`get_log_level`, `is_enabled`) | Generation-validated thread_local cache | Lock-free (fast path) |
| Config reads (cache miss) | `std::shared_lock<std::shared_mutex>` | Shared/reader lock |
| Config writes (`set_log_level`, `initialize_from_python`) | `std::unique_lock<std::shared_mutex>` | Exclusive/writer lock |
| Sink reads (`sink()`) | `std::atomic<std::ostream*>` load | Lock-free |
| Sink writes (`set_log_file`) | Exclusive lock + atomic store | Exclusive lock |
| Log output (write to sink) | Single `ostream::write` per record | POSIX `flockfile` serialization |

---

## Appendix C: C++ LogLevel Enum Reference

The `torch_spyre::logging::LogLevel` enum matches Python's `logging`
module numeric values:

| Enumerator | Numeric value | Meaning |
| --- | --- | --- |
| `LogLevel::NOTSET` | 0 | Unset — falls through to parent |
| `LogLevel::DEBUG` | 10 | Detailed diagnostic output |
| `LogLevel::INFO` | 20 | Normal operational messages |
| `LogLevel::WARNING` | 30 | Potential issues (default level) |
| `LogLevel::ERROR` | 40 | Errors that don't halt execution |
| `LogLevel::CRITICAL` | 50 | Fatal conditions |

These values are exposed to Python via pybind11 as
`torch_spyre._C._logging.LogLevel`.

Utility functions:

```cpp
// Convert enum to string: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
const char* log_level_to_string(LogLevel level);

// Convert string to enum (case-sensitive, must be uppercase)
LogLevel string_to_log_level(const std::string& level_str);
```

---

## Appendix D: Design Rationale — Naming

This appendix explains why ULF uses `torch_spyre.*` in `TORCH_LOGS` and
`spyre.*` internally.

In upstream PyTorch, `TORCH_LOGS` uses **short aliases** that are distinct
from the internal Python logger names:

| TORCH_LOGS alias | Internal Python logger |
| --- | --- |
| `inductor` | `torch._inductor` |
| `dynamo` | `torch._dynamo` |
| `aot` | `torch._functorch.aot_autograd` |

ULF takes a **two-layer approach**: `torch_spyre.*` is the public CLI
namespace (required by PyTorch's `find_spec()` validation), while `spyre.*`
is the internal logger namespace. A normalization function
(`_normalize_component`) maps the former to the latter at parse time.

| Layer | Name | Example |
| --- | --- | --- |
| CLI (`TORCH_LOGS`) | `torch_spyre.*` | `TORCH_LOGS="+torch_spyre.inductor"` |
| Python package (import path) | `torch_spyre` | `from torch_spyre._inductor.lowering import ...` |
| Internal loggers, macros, output | `spyre.*` | `[INFO] [spyre.inductor.lowering] ...` |
| Programmatic API | `spyre.*` | `logging_config.set_log_level("spyre.inductor", "DEBUG")` |

Rationale for the internal `spyre.*` short form:

1. **Brevity** — `spyre.inductor` appears in every log line; shorter is
   better for readability and grepping.
2. **Clean hierarchy** — `_inductor` and `torch_` are implementation
   details, not user-facing concepts.
3. **No confusion** — one internal namespace, one rule, no normalization
   edge cases in Python loggers or C++ code.

Rationale for `torch_spyre.*` in `TORCH_LOGS`:

1. **PyTorch validation** — `import torch` validates every `TORCH_LOGS`
   entry with `importlib.util.find_spec()`. Only real importable packages
   pass. Namespace stub modules (`torch_spyre/inductor/`,
   `torch_spyre/runtime/`, etc.) exist to satisfy this check.
