# Op Coverage Report

Before you bring a new model up on Spyre, find out which of its ops
torch-spyre already handles. The op coverage report answers that for any
PyTorch model. It lists every op the model calls and, for each one, says
whether the compiled path lowers it to a Spyre kernel, decomposes it,
runs it on the CPU, or cannot handle it yet. It also gives the reason.

The report is generated, not written by hand. It reads torch-spyre's own
tables (decompositions, lowerings, CPU fallbacks, eager kernels), so it
stays current as ops are added. No Spyre device is needed, except for the
optional `--probe` and `--e2e` checks.

## Quick Start

Run the tool from a torch-spyre checkout:

```bash
# A YOLO model, built from its Ultralytics config (no weight download)
python tests/op_coverage/op_coverage.py ultralytics yolov5n.yaml

# YOLO26 with its NMS-free head, which adds top-k post-processing
python tests/op_coverage/op_coverage.py ultralytics yolo26n.yaml --opt end2end=true

# A Hugging Face LLM: two layers, prefill and three decode steps
python tests/op_coverage/op_coverage.py hf ibm-granite/granite-3.3-2b-instruct \
    --opt layers=2 --opt mode=generate --opt max_new_tokens=3

# Any model: a function that returns (model, args)
python tests/op_coverage/op_coverage.py module my_package.models:build

# An op YAML that hf-adapters' utils/model_ops collected on a GPU host
python tests/op_coverage/op_coverage.py model-ops-yaml \
    tests/configs/model_ops_tests/Meta-Llama-3.1-8B-Instruct.yaml \
    --cast-float float16
```

Each run writes `<model>.md` and `<model>.json` to `op_coverage_reports/`
(change it with `-o`). Run `python tests/op_coverage/op_coverage.py --help` for the
full option list.

## How It Works

The tool runs in four stages:

1. **Capture.** The model runs once on the CPU under `torch.compile`, with
   a Dynamo backend that records each graph and then runs it eagerly. Each
   torch-level call, such as `torch.conv2d` or `torch.nn.functional.silu`,
   is recorded with the shapes and dtypes of its arguments. Op names
   follow the hf-adapters `model_ops` collector, so both sources produce
   comparable reports.
2. **Trace.** Each unique call is traced with `make_fx` on fake tensors on
   the Spyre device, through Spyre's decomposition table and
   functionalization, the way AOTAutograd prepares a Spyre compile. No
   device memory is touched. Every decomposition that fires is recorded,
   so the report shows the exact path from the torch op to the ATen ops
   that reach Inductor.
3. **Classify.** Each remaining ATen op is checked the way Inductor's
   graph lowering would handle it. The checks cover Spyre's CPU
   fallbacks, Spyre lowerings, in-tree lowerings, and the pointwise ops
   and reduction types that Spyre code generation implements. The eager
   kernel that Inductor falls back to is checked too. A torch op gets the
   worst status of its ATen ops; view ops count only when nothing else is
   left.
4. **Render.** The tool writes Markdown for people and JSON for scripts.
   The rows of
   [Supported Operations](../../docs/source/user_guide/supported_operations.md)
   are joined
   in as documented notes. Curated known issues, such as open PRs, are
   attached as links.

### Statuses

| Status | Meaning |
|---|---|
| ✅ Spyre | Lowered to a Spyre kernel |
| 🟦 View | View op, handled by layout propagation; some shapes are not stick-aligned |
| 🟨 Eager fallback | No lowering; Inductor calls the Spyre eager kernel and splits the graph |
| 🟧 CPU fallback | Runs on the CPU, with copies to and from the device |
| ⚠️ Needs validation | An in-tree Inductor lowering exists, but nothing shows Spyre supports it |
| ❓ Unknown | The op could not be traced; the report shows the error |
| ❌ Not supported | No lowering and no Spyre kernel, or the lowering emits an op Spyre cannot generate |

An in-tree lowering counts as supported when Spyre implements the ops it
emits. It also counts when the supported-operations page lists the op as
compiled on Spyre.

## Reading the Report

The Markdown report has these sections:

- **Summary**: op counts per status, and whether a compile is expected to
  fail.
- **What does the model require?**: one row per torch op, with call count,
  compiled-path and eager status, and where it runs (Spyre, CPU, or
  both). The notes name the decompositions and the ATen ops that are not
  plainly on Spyre.
- **Gaps**: the ops to work on, worst first, with the ATen op at fault,
  example shapes, the module that calls it, and linked issues.
- **Op details**: for each op, the decomposition tree, eager behavior,
  documented notes, calling modules and source lines, and a table of
  signatures. Results can differ by shape; for example, some
  convolutions lower directly while others go through the CPU.
- **ATen ops that reach Inductor**: every leaf op with its status and
  which torch ops use it.
- **Graph breaks**: where Dynamo split the model, and why. The code at a
  break runs eagerly and its ops are not in the report.

## Options

| Option | Effect |
|---|---|
| `--opt KEY=VALUE` | Recipe option, repeatable; see the recipes below |
| `--name NAME` | Model name in the report and the file names |
| `-o DIR`, `--output-dir DIR` | Output directory (default `op_coverage_reports`) |
| `--format md,json` | Formats to write |
| `--notes FILE` | Extra known issues, YAML or JSON, repeatable |
| `--docs-table PATH`, `--no-docs` | Use another supported-operations page, or none |
| `--cast-float DTYPE` | For `model-ops-yaml`: treat float tensors as this dtype |
| `--probe` | Compile and run each signature on the device (needs a Spyre card) |
| `--e2e` | Compile and run the whole model on the device (recipe sources) |
| `--device NAME` | Device for `--probe` and `--e2e` (default `spyre`) |
| `--probe-timeout SEC`, `--max-probes N` | Per-case time limit, and a cap on the number of probed signatures |
| `--fail-on STATUS` | Exit with 1 if any op has this status or worse, for CI |
| `-q`, `--quiet` | Print nothing but errors |

### Recipes

| Source | Target | Options |
|---|---|---|
| `module` | `package.module:function` returning `(model, args)`, `(model, args, kwargs)` or a `ModelSpec` | passed to the function as keyword arguments |
| `hf` | Hugging Face hub id or local path | `pretrained=false`, `layers=N`, `seq_len=32`, `batch=1`, `mode=forward\|generate`, `max_new_tokens=4`, `image_size`, `attn_implementation=sdpa`, `trust_remote_code=false`, `dtype=float16` |
| `ultralytics` | model YAML or `.pt` checkpoint | `fuse=true`, `end2end=true\|false`, `image_size=640`, `batch=1`, `dtype=float16` |
| `torchvision` | model name | `image_size=224`, `batch=1`, `dtype=float16` |
| `timm` | model name | `batch=1`, `dtype=float16` |
| `model-ops-yaml` | path to a `model_ops` YAML | — |
| `json` | a saved report | re-renders it only |

By default, the `hf` recipe builds the model from its config with random
weights, so only the config is downloaded. Ops repeat from layer to
layer, so `layers=2` gives the same op list as the full model at a
fraction of the cost.

The model's own libraries (transformers, ultralytics, torchvision, timm)
are imported only by their recipe and are not torch-spyre dependencies.

## On a Spyre Host: `--probe` and `--e2e`

The static analysis cannot see everything. It misses layout and
stick-alignment failures, numerical differences, and graph-level FX
passes that rewrite several ops at once. With `--probe`, each unique
signature is compiled and run on the device with random inputs and
compared with the CPU (`atol=rtol=0.1`). The outcome is one of `pass`,
`mismatch`, `compile_error`, `runtime_error`, `crash`, `timeout`, or
`skipped`, and any CPU fallback warnings are recorded with it. With
`--e2e`, the whole model is compiled and run the same way.

Probes run in a worker process. A case that kills the process or runs
past `--probe-timeout` is recorded, and the worker restarts at the next
case.

## Extending the Tool

- **A new model family**: register a recipe in
  `tests/op_coverage/spyre_op_coverage/recipes.py` with
  `@register_recipe(name, help)`. The function returns a `ModelSpec`, and
  its library is imported inside it.
- **Known issues**: shared entries live in `KNOWN_ISSUES` in
  `tests/op_coverage/spyre_op_coverage/knowledge.py`, keyed by torch op or ATen
  op name. Remove an entry when its issue closes. For team-local notes,
  pass `--notes FILE`:

  ```yaml
  aten.gather:
    note: Gather lowering in progress.
    links: [https://github.com/torch-spyre/torch-spyre/pull/1234]
  ```

- **Support itself**: nothing to do. When an op gains a lowering,
  decomposition or fallback in torch-spyre, the next report shows it.

## Python API

The package lives in this directory and is not installed, so put the
directory on `sys.path`. Import `torch` before the tool so the Spyre
backend loads first:

```python
import sys

sys.path.insert(0, "tests/op_coverage")  # from the repository root

import torch
from spyre_op_coverage import analyze_model, to_markdown

report = analyze_model(model, (example_input,), name="my-model")
print(to_markdown(report))
```

`report.to_dict()` gives the JSON form, and
`CoverageReport.from_dict()` reads it back.

## Limitations

- The model runs on the CPU for capture. A model that takes a different
  code path on a Spyre device can produce different graphs.
- Each op is traced on its own, so graph-level FX passes are not modeled.
  Use `--probe` or `--e2e` to cover them.
- Ops at a graph break run eagerly and are not analyzed. The report lists
  where each break happens.
