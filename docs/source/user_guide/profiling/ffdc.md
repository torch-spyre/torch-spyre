# FFDC (First Failure Data Capture)

**Stack:** torch-spyre (new, Inductor-based).

When a Spyre workload fails during compilation, kernel launch, or
dispatch of an unimplemented operation, the diagnostic evidence is easy
to lose: processes exit, pods restart, and compiler temp directories are
cleaned up. FFDC captures a structured JSON snapshot at the moment of
failure — exception, environment, nearby compiler artifacts, runtime
context, and hardware availability — before the original error
propagates.

Capture is opt-in via `TORCH_SPYRE_FFDC=1`. Retrieval via
`torch.spyre.get_diagnostic_report()` does **not** require that
variable. For field-level API details see
[`get_diagnostic_report`](../../api/torch_spyre.rst).

This gate is intentionally separate from `USE_SPYRE_PROFILER` (the
`setup.py` Kineto profiler build flag). FFDC capture does not require a
profiler build, and pods do not enable `TORCH_SPYRE_FFDC` by default.

Auto-capture hooks landed in
[PR #2704](https://github.com/torch-spyre/torch-spyre/pull/2704); the
retrieve API landed in
[PR #3305](https://github.com/torch-spyre/torch-spyre/pull/3305). This
guide is the field/support path: enable → fail → retrieve → triage.

## Workflow

1. **Enable capture** for the failing run:

   ```bash
   export TORCH_SPYRE_FFDC=1
   export TORCH_COMPILE_DEBUG=1   # optional; more artifact paths
   ```

   `TORCH_COMPILE_DEBUG` is optional enrichment. It is not required
   for capture. Bundle files (`sdsc_*.json`, `bundle.mlir`) are
   written by `generate_bundle()` without an extra env var.
   `DUMP_SPYRE_CODE` is snapshotted into the report if set; torch-spyre
   does not read it.

2. **Reproduce the failure.** With the gate set, frontend-compile,
   backend-compile, runtime-launch, and unimplemented-operation hooks
   write a JSON report automatically. FFDC never changes program
   behaviour: hooks use nested `try/except` so a collection failure
   cannot mask the original exception.

3. **Retrieve** the newest valid report. Use the same search directory
   the failing process used (`TORCHINDUCTOR_CACHE_DIR` / `TMPDIR`). A
   new interpreter does **not** inherit an isolated cache unless you
   export it **before** the first Inductor `cache_dir()` call (often
   during `import torch`). That helper assigns
   `TORCHINDUCTOR_CACHE_DIR` to a default when it is unset, and
   `get_diagnostic_report()` will miss reports written under a
   different root.

   ```python
   import torch
   import torch_spyre

   report = torch.spyre.get_diagnostic_report()
   if report is not None:
       print(report["failure"]["category"])
       print(report["failure"]["file"], report["failure"]["lineno"])
       print(report["failure"]["message"])
       print(report["_report_path"])
   ```

   The same function is available as
   `torch_spyre.profiler.get_diagnostic_report`. Retrieval does not
   require `TORCH_SPYRE_FFDC`.

4. **Triage.** Start with `failure.category`, then the fields in
   [Field / support interpretation](#ffdc-interpretation).
   Confirm `metadata.pid` / `metadata.timestamp` match this run —
   `get_diagnostic_report()` returns the newest valid file in the
   directory, not “this process.”

Reports are **local to the host** that produced them. CI web UIs do
not show them unless a workflow prints `_report_path` or uploads the
report directory as an artifact. Logging the path on capture and CI
upload are tracked in
[#4052](https://github.com/torch-spyre/torch-spyre/issues/4052); they
are not shipped yet.

## Failure categories and hook locations

Auto-hooks write one of four labels into `failure.category`. Empty
input to `collect()` is stored as `unknown`. Custom non-empty strings
are stored as-is (the on-disk filename sanitizes characters outside
`[A-Za-z0-9_-]` and truncates the category to 32 characters).

| `failure.category` | Layer | When it fires | Hook location |
|---|---|---|---|
| `compile_frontend` | Frontend | `torch.compile()` / Spyre Inductor frontend fails (including decomposition and lowering paths that surface through `compile_fx`) | `torch_spyre/_inductor/__init__.py` (`compile_fx`) |
| `compile_backend` | Backend | `dxp_standalone` (`sdsc`) or `dbo-opt` (`_compile_ktir_with_dbo`, including missing spyrecode) fails after artifact emit | `torch_spyre/execution/async_compile.py` |
| `runtime_launch` | Runtime | `launch_jobplan` fails on device | `torch_spyre/execution/kernel_runner.py` (`SpyreSDSCKernelRunner.run`) |
| `unimplemented` | Runtime | An op reaches the runtime without a Spyre lowering | `torch_spyre/execution/kernel_runner.py` (`SpyreUnimplementedRunner.run`) |
| `unknown` | — | Manual `collect()` / `try_collect()` that omit `failure_category`, or empty category input | `torch_spyre/profiler/_ffdc.py` (no auto-hook) |

When a backend hook (`sdsc` / `dbo-opt`) captures and re-raises, the
outer `compile_fx` wrapper does **not** write a second
`compile_frontend` report for that same exception, a wrapper raised
`from` it (`__cause__`), or a wrapper whose `__context__` is that
exception (bare `raise New` inside `except`). The inner category is
kept.

(ffdc-interpretation)=

### Field / support interpretation

Use this section when triaging a report from a customer pod or CI
host. Start with `failure.category`, then the fields called out for
that category.

#### `compile_frontend`

- **Trigger path:** Spyre Inductor frontend compile via
  `try_collect(..., failure_category=compile_frontend)` in
  `torch_spyre/_inductor/__init__.py` (`compile_fx`).
- **Owning hook:** frontend only (does not attach `runtime.kernel_name`
  / `runtime.code_dir`).
- **Triage first:** `failure.exception_type` / `message` / `traceback`;
  then `artifacts.paths` (`fx_graph_readable.py`,
  `fx_graph_transformed.py`, `ir_pre_fusion.txt`, `ir_post_fusion.txt`,
  `output_code.py`) and `environment.TORCH_COMPILE_DEBUG`.
- **Interpretation:** usually a graph / decomposition / lowering
  failure that surfaced through `compile_fx`. Also covers unhooked
  emit, pre-tool checks, and `prepare_kernel` failures when they
  bubble to `compile_fx`. This hook does not set `runtime.code_dir`;
  use the traceback and any on-disk compiler trees.

#### `compile_backend`

- **Trigger path:** `try_collect(..., failure_category=compile_backend)`
  around `dxp_standalone` in `SpyreAsyncCompile.sdsc` and around
  `dbo-opt` in `SpyreAsyncCompile._compile_ktir_with_dbo` (including
  dbo-opt exit 0 with missing spyrecode) in
  `torch_spyre/execution/async_compile.py`.
- **Owning hook:** backend tool invoke; passes `runtime.kernel_name`
  and `runtime.code_dir`.
- **Triage first:** `failure.*`; `runtime.kernel_name` /
  `runtime.code_dir` (open that tree for `sdsc_*.json`, `*.mlir`,
  `*.ktir`, `spyreCodeDir/` — FFDC artifact search does not walk
  `code_dir`); then any matching paths already under `artifacts.paths`
  and `environment.TORCH_COMPILE_DEBUG`.
- **Interpretation:** the backend compiler tool (`dxp_standalone` /
  `dbo-opt`) was invoked. Prefer `runtime.code_dir` / tool artifacts
  over FX graphs. The JSON may capture the **inner** tool exception
  (`TimeoutExpired`, `CalledProcessError`) even when the process
  then re-raises a wrapping `RuntimeError`. Failures in
  `generate_bundle` / `generate_ktir`
  **before** that tool runs are not this category (see
  [Deferred category gaps](#ffdc-deferred-gaps)).

#### `runtime_launch`

- **Trigger path:** `SpyreSDSCKernelRunner.run` decorated with
  `runtime_launch`.
- **Owning hook:** `torch_spyre/execution/kernel_runner.py`.
- **Triage first:** `failure.*`; `runtime.kernel_name` and
  `runtime.code_dir` (path strings on the report — FFDC does not
  inventory that tree into `artifacts.paths`);
  `hardware_state.spyre_available`; then any matching paths already
  listed under `artifacts.paths`, or open `code_dir` on the host that
  produced the report.
- **Interpretation:** the compiled kernel existed far enough to
  attempt device launch (`launch_jobplan`). `prepare_kernel` in
  `SpyreSDSCKernelRunner.__init__` is **not** this category (see
  [Deferred category gaps](#ffdc-deferred-gaps)). Prefer hardware
  and runtime context over Inductor IR when the traceback is inside
  the runner.

#### `unimplemented`

- **Trigger path:** `SpyreUnimplementedRunner.run` decorated with
  `unimplemented`.
- **Owning hook:** `torch_spyre/execution/kernel_runner.py`.
- **Triage first:** `failure.message` (names the missing op) and
  `runtime.kernel_name`; `code_dir` is not attached by this hook.
- **Interpretation:** an op reached runtime without a Spyre lowering.
  This is usually a coverage / lowering gap, not a device-driver
  fault.

#### `unknown`

- **Trigger path:** manual `collect()` / `try_collect()` calls that
  omit `failure_category`, or empty category input.
- **Owning code:** `torch_spyre/profiler/_ffdc.py` (no auto-hook emits
  this on its own).
- **Triage first:** `failure.*` and `collector.*`; treat other
  sections as best-effort context only.
- **Interpretation:** do not assume a compile or launch failure.
  Confirm whether capture was manual or whether a caller omitted the
  category.

(ffdc-deferred-gaps)=

### Deferred category gaps

No additional auto-hook labels exist beyond the four hook-emitted
values above. The following are **explicitly deferred** (documented
gaps, not missing vocabulary entries):

- Device-side failure category beyond `runtime_launch`
- Profiling / capture-pipeline failure category
- Finer per-pass frontend categories (decomposition / lowering still
  share `compile_frontend` when they surface through `compile_fx`)
- `generate_bundle` / `generate_ktir` (and persisting `.ktir` to disk)
  before `dxp_standalone` / `dbo-opt` — unhooked at the emit site
  today; if they surface through `compile_fx`, they are labeled
  `compile_frontend`
- Pre-tool checks such as `_check_ktir_device_prerequisites()`
  outside the `dbo-opt` `try_collect` — same bubble behavior as emit
  helpers
- `SpyreSDSCKernelRunner.__init__` / `prepare_kernel` — after backend
  tools succeed, before `run()`; if they surface through `compile_fx`,
  they are labeled `compile_frontend`

## Where reports are stored

Default directory (from Inductor `cache_dir()`, **not**
`~/.cache/torch/inductor`):

```
<tempdir>/torchinductor_<user>/torch-spyre/ffdc_reports/
```

`<tempdir>` is Python's `tempfile.gettempdir()`: typically `/tmp` on
Linux. Override it with `TMPDIR` (or `TEMP`/`TMP` on some platforms).
When `TORCHINDUCTOR_CACHE_DIR` is set, reports land under
`$TORCHINDUCTOR_CACHE_DIR/torch-spyre/ffdc_reports/` instead — that
env var replaces the Inductor cache root entirely.

Example on Linux with no overrides:
`/tmp/torchinductor_<user>/torch-spyre/ffdc_reports/`.

If resolving that Inductor cache root fails for any reason, reports
fall back to:

```
<tempdir>/torch-spyre-ffdc/
```

The directory is created with mode `0o700` and each report file with
`0o600` on POSIX so captured env, host, paths, and tracebacks are not
world-readable. Owner-only modes are POSIX-only: on Windows,
`os.chmod` does not change DACLs, so reports inherit the destination
directory's ACL and this privacy restriction is not applied.

Each successful capture writes a **new** file; earlier reports are not
overwritten. Filenames follow:

```
ffdc_<category>_<YYYYMMDDTHHMMSS>_<microseconds>_<pid>.json
```

Example: `ffdc_compile_frontend_20250101T120000_123456_42.json`

**Retention:** the directory keeps the newest **50** reports (by file
modification time) and deletes older ones.

(ffdc-selecting-reports)=

## Selecting the newest report

`get_diagnostic_report()` walks candidates newest-first by the UTC
timestamp **embedded in the filename** (not `st_mtime`) **across all
runs in the directory** — not scoped to the process that just called
it — skips unreadable or structurally invalid files (including FIFOs
and symlinks whose names look like reports), and returns `None` when
no valid report remains. An unreadable search directory returns
`None` rather than raising.

If your script did not actually fail, this may return a leftover
report from an earlier failure; compare `metadata.pid` or
`metadata.timestamp` against your current run when that distinction
matters. You can also identify a report by `metadata.host` and
`failure.category` inside the JSON, or by the filename itself.

Capture and retrieval can use different environment variables:

- `TORCH_SPYRE_FFDC` only gates **writing**. Retrieval works even if
  that variable is unset in a later session.
- The **search directory** must still match. If
  `TORCHINDUCTOR_CACHE_DIR` or `TMPDIR` changed between capture and
  retrieval, pass the original path as `output_dir`. If a tool or
  test isolated the cache, export the printed
  `TORCHINDUCTOR_CACHE_DIR` before calling
  `get_diagnostic_report()` in a new interpreter.

## Report schema

Top-level keys on the dict **returned** by `collect()` and
`get_diagnostic_report()`: `metadata`, `failure`, `environment`,
`artifacts`, `runtime`, `hardware_state`, `collector`, `_report_path`.

The on-disk JSON is that dict **without** `_report_path`.
`collect()` attaches `_report_path` only on the in-memory return
value after a successful write. `get_diagnostic_report()` injects
it when loading a file.

When `TORCH_SPYRE_FFDC` is not `1`, `collect()` still returns that
shape but skips filesystem work: sections are empty or placeholder,
`collector.disabled` is `true`, and `_report_path` is `null`.
`get_diagnostic_report()` can still read reports written earlier.

Look at `failure` first, then `collector` (did capture itself work?),
then the category-specific fields above.

### `metadata`

| Field | Meaning |
|---|---|
| `timestamp` | UTC ISO-8601 time of capture |
| `host` | `platform.node()` |
| `pid` | Process id that wrote the report |
| `python_version` | `sys.version` |
| `torch_version` | `torch.__version__`, or `"unavailable"` |
| `torch_spyre_version` | `torch_spyre.version.__version__`, or `"unavailable"` |
| `platform` | `platform.platform()` |

### `failure`

| Field | Meaning |
|---|---|
| `category` | Hook label (see table above) |
| `exception_type` | `type(exc).__name__` of the object passed to `collect()` (not walked via `__cause__`); `null` for manual `collect()` with no exception |
| `message` | `str(exc)`, or `"manual collection (no exception)"` when `exc` is omitted |
| `traceback` | Joined traceback text; `null` when `exc` is omitted |
| `file` | Filename of the innermost raise site |
| `lineno` | Line of the innermost raise site |

`file` / `lineno` are the innermost frame of that exception object,
not the hook wrapper. A backend tool hook may capture the inner
subprocess error, then re-raise a wrapping `RuntimeError` — the
report type/message will not match the exception the caller sees.

### `environment`

Captured as strings (`""` if unset — completeness treats `""` as
present, not missing). `TORCHINDUCTOR_CACHE_DIR` and `TMPDIR` are
**not** in this snapshot; they still control where reports are
written (see [Where reports are stored](#where-reports-are-stored)).

| Key | Role |
|---|---|
| `TORCH_SPYRE_FFDC` | Capture gate (`1` to write reports) |
| `TORCH_COMPILE_DEBUG` | Enables `torch_compile_debug/` trees that artifact search links |
| `DUMP_SPYRE_CODE` | Snapshotted if set; torch-spyre does not read this variable |
| `TORCH_SPYRE_DEBUG` | Legacy runtime/build debug gate (deprecated; prefer `TORCH_LOGS`) |
| `SPYRE_INDUCTOR_LOG` | Deprecated Inductor log gate (still captured) |
| `SPYRE_INDUCTOR_LOG_LEVEL` | Deprecated Inductor log level (still captured) |
| `TORCH_LOGS` | PyTorch logging selector |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES` | Forces full recompile |
| `SENCORES` | Spyre core count |

### `artifacts`

| Field | Meaning |
|---|---|
| `searched` | `true` if the artifact walk ran |
| `found_count` | Unique paths found before the cap (when search ran) |
| `paths` | Up to 20 unique paths (when search ran) |
| `error` | Present when search timed out or failed |

Search is best-effort and bounded (2.0s). A timeout sets
`searched: false` and records `artifacts: timed out` in
`collector.collector_errors`. See
[Compiler artifacts FFDC searches for](#compiler-artifacts-ffdc-searches-for).

### `runtime`

| Field | Meaning |
|---|---|
| `kernel_name` | Set by backend and runtime hooks; `null` for frontend |
| `code_dir` | Set by backend and `runtime_launch`; `null` for frontend and `unimplemented` |

These are path **strings**. FFDC does not copy or inventory `code_dir`.

### `hardware_state`

| Field | Meaning |
|---|---|
| `spyre_available` | Result of `torch.spyre.is_available()` (1.0s timeout) when `torch.spyre` is registered |
| `note` | Set when the probe times out, finds no Spyre hardware, or the check fails |

The helper swallows exceptions into `note`; an `error` key is not
used on the current path.

Off-pod, when `torch.spyre` is registered, `spyre_available` is
typically `false` with a short `note`. If the Spyre backend was never
loaded, this section may be only `{spyre_available: false}` with no
`note`. It is not device telemetry.

### `collector`

| Field | Meaning |
|---|---|
| `capture_latency_ms` | Wall time of this `collect()` call |
| `missing_fields` | Required dotted paths that were `null` |
| `collector_errors` | Best-effort errors from individual sections or the write |
| `success` | `true` when `collector_errors` is empty |
| `completeness_pct` | Share of required fields that were present |
| `disabled` | Present and `true` only when capture was gated off |

Required fields for `completeness_pct`: `metadata.timestamp`,
`metadata.torch_version`, `metadata.python_version`,
`failure.category`, `failure.exception_type`, `failure.message`,
`failure.traceback`, `environment.TORCH_COMPILE_DEBUG`,
`environment.TORCH_SPYRE_DEBUG`, `environment.SPYRE_INDUCTOR_LOG`,
`artifacts.searched`. Manual `collect()` with no exception leaves
`exception_type` and `traceback` null, so completeness is not 100%.

Do not treat `collector.success` as “the user’s failure was
diagnosed.” Artifact search can time out on a complete-enough
report; the original exception still propagated.

### `_report_path`

Present on the dict returned by `collect()` (after a successful
write) and on the dict returned by `get_diagnostic_report()`.
**Not** stored inside the JSON file. `null` when capture was
disabled or the write failed.

### Compiler artifacts FFDC searches for

When `TORCH_COMPILE_DEBUG=1` is set, Inductor writes a
`torch_compile_debug/` tree. FFDC looks for that tree under each of
four fixed locations — the current working directory, the torch-spyre
repo root, `/dev/shm`, and `/tmp` — and, in each one that exists,
searches only that location's **newest** `run_*` subdirectory for:

- `fx_graph_readable.py`, `fx_graph_transformed.py`
- `ir_pre_fusion.txt`, `ir_post_fusion.txt`
- `output_code.py`, `graph_diagram.html`
- `aot_model_*` logs and other `*.log` files (e.g.
  `torchdynamo/debug.log`)
- `sdsc_*.json`, `*.mlir`, `*.ll` (when present under that `run_*`
  tree; bundle emit also writes `sdsc_*.json` / `bundle.mlir` under
  the Spyre kernel cache searched below)

Because all four locations are checked, a stale `torch_compile_debug/`
tree left over from an earlier, unrelated run can still surface in
`artifacts.paths`.

It also searches the newest kernel directory under the Spyre Inductor
cache (`$TORCHINDUCTOR_CACHE_DIR/inductor-spyre/` when set, else
`<tempdir>/torchinductor_<user>/inductor-spyre/`) for bundle artifacts.

To keep capture best-effort and bounded, the search is intentionally
limited:

- only the newest `run_*` directory *per search location* is scanned
- only the newest Spyre kernel-cache directory is scanned
- up to 5 matches are kept per filename pattern
- up to 20 unique paths are returned in `artifacts.paths`

## Pod workflow

On a Spyre pod, run a workload with capture enabled:

```bash
cd /dev/shm/workdir/torch-spyre
TORCH_SPYRE_FFDC=1 TORCH_COMPILE_DEBUG=1 python your_script.py
```

Runtime hooks fire from `SpyreSDSCKernelRunner.run` (`launch_jobplan`)
and `SpyreUnimplementedRunner.run`. A fake `code_dir` that fails in
`prepare_kernel` (`__init__`) is **not** `runtime_launch` and will not
write that category.

`tools/ffdc_trigger.py` is an internal hardware experiment, not the
field-triage path. If you use it and it prints
`export TORCHINDUCTOR_CACHE_DIR=...`, a new interpreter must export
that value before `get_diagnostic_report()` or it will search the
default cache and miss this run's reports.

Print the exact report path that `get_diagnostic_report()` would
return (same cache dir as the failing process):

```bash
# export TORCHINDUCTOR_CACHE_DIR=<path used by the failing process>
python - <<'PY'
import torch
import torch_spyre

report = torch.spyre.get_diagnostic_report()
print(report["_report_path"] if report is not None else "No FFDC report found")
PY
```

Copy it to your laptop (replace `<pod>` and use the exact path printed
above — the filename's category prefix depends on which hook fired:
`compile_frontend`, `compile_backend`, `runtime_launch`, or
`unimplemented`):

```bash
oc cp <pod>:/exact/path/from/_report_path ./ffdc_report.json
```

## Known limitations

- FFDC is opt-in. If `TORCH_SPYRE_FFDC=1` was not set when the failure
  happened, no new report is written.
- Capture happens only at hooked call sites: `compile_fx` (frontend);
  `async_compile.sdsc` / `dxp_standalone` and
  `async_compile._compile_ktir_with_dbo` / `dbo-opt` (backend tools);
  `SpyreSDSCKernelRunner.run` / `launch_jobplan` (`runtime_launch`);
  and `SpyreUnimplementedRunner.run` (`unimplemented`). Emit helpers
  (`generate_bundle` / `generate_ktir`), pre-tool checks, and
  `prepare_kernel` in `SpyreSDSCKernelRunner.__init__` are not
  separately hooked; if they surface through `compile_fx`, they are
  labeled `compile_frontend`. There is no separate per-pass category
  today.
- `hardware_state` is intentionally lightweight today: it records
  `spyre_available` plus a short note when the probe times out,
  errors, or simply finds no Spyre hardware available (the common
  case off-pod).
- Artifact capture links raw files; it does not yet parse IR / bundle
  files into a machine-generated root-cause summary.
- Successful capture does not yet log the report path next to the
  traceback ([#4052](https://github.com/torch-spyre/torch-spyre/issues/4052)).

## Planned enhancements

- Logging category + absolute `_report_path` on successful write, and
  CI artifact upload of `ffdc_reports/` when capture is enabled
  ([#4052](https://github.com/torch-spyre/torch-spyre/issues/4052))
- Parsing IR / bundle artifacts to populate an `error_summary` field
  in the JSON report
- Additional hooks for device-side and profiling failures (see
  [Deferred category gaps](#ffdc-deferred-gaps))
- Formal KPI targets (MTTRC, zero-repro diagnostics) — design goals,
  not current guarantees

## Related work

These items are **in review or not started**. This guide documents
hooks and `collect()` as they exist on `main`; it does not treat the
following as shipped:

- [PR #3626](https://github.com/torch-spyre/torch-spyre/pull/3626)
  ([#3551](https://github.com/torch-spyre/torch-spyre/issues/3551)) —
  preferred-vocabulary constants and stricter empty-category
  normalization. Auto-hook labels on `main` already match the table
  above.
- [PR #3896](https://github.com/torch-spyre/torch-spyre/pull/3896)
  ([#3554](https://github.com/torch-spyre/torch-spyre/issues/3554)) —
  hardware integration tests and `ffdc_trigger.py` cache isolation.
  The retrieve/`TORCHINDUCTOR_CACHE_DIR` rule above is already true
  of Inductor `cache_dir()` on `main`.
- [#4052](https://github.com/torch-spyre/torch-spyre/issues/4052) —
  log report path + CI artifacts (not started).

## See also

- [API: `get_diagnostic_report`](../../api/torch_spyre.rst) — parameter
  defaults and return contract
- [Profiling environment variables](environment_variables.md) —
  `TORCH_SPYRE_FFDC`
- [Debugging guide](../debugging/index.md) — manual
  `TORCH_COMPILE_DEBUG` artifact inspection
- [Contributing to the Profiler](../../contributing/profiling.md) —
  test layout and review process
- Auto-capture hooks:
  [PR #2704](https://github.com/torch-spyre/torch-spyre/pull/2704)
- Retrieve API:
  [PR #3305](https://github.com/torch-spyre/torch-spyre/pull/3305)
