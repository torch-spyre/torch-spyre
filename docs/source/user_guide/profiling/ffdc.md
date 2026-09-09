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

Auto-capture hooks landed in
[PR #2704](https://github.com/torch-spyre/torch-spyre/pull/2704); this
guide covers the retrieval API and report schema.

## Quick start

```bash
export TORCH_SPYRE_FFDC=1
export TORCH_COMPILE_DEBUG=1   # optional; adds torch_compile_debug/ paths
export DUMP_SPYRE_CODE=1       # optional; adds sdsc_*.json and *.mlir paths
python your_script.py
```

Then inspect the newest valid report:

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

`TORCH_SPYRE_FFDC=1` controls whether new reports are written.
`get_diagnostic_report()` can still read already-written reports even if
that variable is unset later.

This gate is intentionally separate from `USE_SPYRE_PROFILER` (the
`setup.py` Kineto profiler build flag). FFDC capture does not require a
profiler build, and pods do not enable `TORCH_SPYRE_FFDC` by default.
Set it explicitly for the failing run.

With `TORCH_SPYRE_FFDC=1` set, frontend-compile, backend-compile,
runtime-launch, and unimplemented-operation hooks write a report
automatically. `TORCH_COMPILE_DEBUG=1` and `DUMP_SPYRE_CODE=1` are
optional enrichment flags (often already available in debug workflows);
they are not required for capture, but they give FFDC more artifact
paths to link into the report.

## Failure categories and hook locations

| `failure.category` | Layer | When it fires | Hook location |
|---|---|---|---|
| `compile_frontend` | Frontend | `torch.compile()` / Spyre Inductor frontend fails (including decomposition and lowering paths that surface through `compile_fx`) | `torch_spyre/_inductor/__init__.py` |
| `compile_backend` | Backend | `dxp_standalone` / bundle compilation fails while generating op-spec / MLIR artifacts | `torch_spyre/execution/async_compile.py` |
| `runtime_launch` | Runtime | Kernel launch or `launch_jobplan` fails on device | `torch_spyre/execution/kernel_runner.py` (`SpyreSDSCKernelRunner`) |
| `unimplemented` | Runtime | An op reaches the runtime without a Spyre lowering | `torch_spyre/execution/kernel_runner.py` (`SpyreUnimplementedRunner`) |
| `unknown` | — | Manual collection or empty category input normalized at capture time | `torch_spyre/profiler/_ffdc.py` |

FFDC never changes program behaviour: hooks use nested `try/except` so
a collection failure cannot mask the original exception.

When a backend hook (`sdsc` / `dbo-opt`) captures and re-raises, the
outer `compile_fx` wrapper does **not** write a second
`compile_frontend` report for that same exception, a wrapper raised
`from` it (`__cause__`), or a wrapper whose `__context__` is that
exception (bare `raise New` inside `except`). The inner category is
kept.

The same function is available as
`torch_spyre.profiler.get_diagnostic_report`.

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

If resolving that Inductor cache root fails for any reason, reports fall
back to:

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
  retrieval, pass the original path as `output_dir`.

## JSON report triage

| Section | What to check |
|---|---|
| `metadata` | `timestamp`, `host`, `pid`, `python_version`, `torch_version`, `torch_spyre_version`, `platform` — identifies which run produced this report |
| `failure` | `category` (encodes layer: frontend / backend / runtime), `exception_type`, `message`, `file`, `lineno` (innermost raise site), `traceback` — start here |
| `artifacts` | `paths` lists compiler files found near the failure (`fx_graph_readable.py`, `fx_graph_transformed.py`, `ir_*.txt`, `output_code.py`, `sdsc_*.json`, `*.mlir`, logs) |
| `environment` | Values of `TORCH_SPYRE_FFDC`, `TORCH_COMPILE_DEBUG`, `DUMP_SPYRE_CODE`, `SENCORES`, and other captured env vars |
| `runtime` | `kernel_name` and `code_dir` when the failure happened during kernel execution |
| `hardware_state` | `spyre_available` and any probe notes |
| `collector` | `completeness_pct`, `missing_fields`, `collector_errors` — whether capture itself succeeded |
| `_report_path` | Absolute path to the loaded JSON file on the host that produced it |

### Compiler artifacts FFDC searches for

When `TORCH_COMPILE_DEBUG=1` is set, Inductor writes a `torch_compile_debug/`
tree. FFDC looks for that tree under each of four fixed locations — the
current working directory, the torch-spyre repo root, `/dev/shm`, and
`/tmp` — and, in each one that exists, searches only that location's
**newest** `run_*` subdirectory for:

- `fx_graph_readable.py`, `fx_graph_transformed.py`
- `ir_pre_fusion.txt`, `ir_post_fusion.txt`
- `output_code.py`, `graph_diagram.html`
- `aot_model_*` logs and other `*.log` files (e.g. `torchdynamo/debug.log`)
- `sdsc_*.json`, `*.mlir`, `*.ll` (when present, e.g. with `DUMP_SPYRE_CODE=1`)

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

To exercise runtime hooks on hardware without a full model failure:

```bash
TORCH_SPYRE_FFDC=1 TORCH_COMPILE_DEBUG=1 python tools/ffdc_trigger.py
```

Print the exact report path that `get_diagnostic_report()` would return:

```bash
python - <<'PY'
import torch
import torch_spyre

report = torch.spyre.get_diagnostic_report()
print(report["_report_path"] if report is not None else "No FFDC report found")
PY
```

Copy it to your laptop (replace `<pod>` and use the exact path printed above —
the filename's category prefix depends on which hook fired:
`compile_frontend`, `compile_backend`, `runtime_launch`, or
`unimplemented`):

```bash
oc cp <pod>:/exact/path/from/_report_path ./ffdc_report.json
```

Reports are **local to the host** that produced them. CI web UIs do not
show them unless a workflow prints `_report_path` or uploads the report
directory as an artifact.

## Known limitations

- FFDC is opt-in. If `TORCH_SPYRE_FFDC=1` was not set when the failure
  happened, no new report is written.
- Frontend decomposition / op-spec failures are captured only when they
  surface through the hooked call sites above (`compile_fx` for
  frontend, `async_compile.sdsc` for backend bundle generation). There
  is no separate per-pass category today.
- `hardware_state` is intentionally lightweight today: it records
  `spyre_available` plus a short note when the probe times out, errors,
  or simply finds no Spyre hardware available (the common case off-pod).
- Artifact capture links raw files; it does not yet parse IR / bundle
  files into a machine-generated root-cause summary.

## Planned enhancements

- Parsing IR / bundle artifacts to populate an `error_summary` field in
  the JSON report
- Additional hooks for device-side and profiling failures
- Formal KPI targets (MTTRC, zero-repro diagnostics) — design goals,
  not current guarantees

## See also

- [API: `get_diagnostic_report`](../../api/torch_spyre.rst) — parameter
  defaults and return contract
- [Profiling environment variables](environment_variables.md) —
  `TORCH_SPYRE_FFDC`
- [Debugging guide](../debugging/index.md) — manual `TORCH_COMPILE_DEBUG`
  artifact inspection
- [Contributing to the Profiler](../../contributing/profiling.md) — test
  layout and review process
- Auto-capture hooks:
  [PR #2704](https://github.com/torch-spyre/torch-spyre/pull/2704)
