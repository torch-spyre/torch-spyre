---
name: model-op-coverage
description: "Generate a Spyre op coverage report for a model (Hugging Face, Ultralytics YOLO, torchvision, timm, any nn.Module, or an hf-adapters model_ops YAML): which ops it needs, and whether each lowers to Spyre, decomposes, falls back to CPU, or is unsupported. Use when asked whether a model can run on Spyre, what blocks it, or to plan a model bring-up."
---

# Model Op Coverage Report

`tests/op_coverage/op_coverage.py` captures a model's ops on the CPU and judges each one
against torch-spyre's own registries. It needs no Spyre device, except for
`--probe` and `--e2e`. User documentation:
`tests/op_coverage/README.md`. Code:
`tests/op_coverage/spyre_op_coverage/`.

## 1. Pick the source and run it

| The model is | Command |
|---|---|
| On the Hugging Face hub | `python tests/op_coverage/op_coverage.py hf <id> --opt layers=2` (add `--opt mode=generate` for decode) |
| An Ultralytics YOLO | `python tests/op_coverage/op_coverage.py ultralytics yolov5n.yaml` (`--opt end2end=true` for YOLO26/YOLOv10 heads) |
| torchvision / timm | `python tests/op_coverage/op_coverage.py torchvision resnet50` |
| Anything else | write `def build(): return model, (inputs,)` and run `python tests/op_coverage/op_coverage.py module pkg.mod:build` |
| Already captured by hf-adapters `utils/model_ops` | `python tests/op_coverage/op_coverage.py model-ops-yaml <file.yaml> --cast-float float16` |

- Run it from the repo root. The launcher imports `torch` before
  `torch_spyre`; `python -m torch_spyre...` fails in installed environments.
- Use `layers=2` for LLMs. Ops repeat per layer, and the default random
  init downloads only the config.
- Outputs go to `op_coverage_reports/<model>.md` and `.json`.

## 2. Read the result

Start with the **Gaps** section; it lists the worst problems first.

| Status | What to do |
|---|---|
| ❌ Not supported | Add a lowering or a decomposition; see the `add-spyre-operation` skill. A mechanism that reads "emits `ops.X`, which SpyreOpFuncs does not implement" needs a method on `SpyreOpFuncs` |
| ❓ Unknown | Read the error in Op details. It is often a tracing limitation, not a Spyre gap |
| ⚠️ Needs validation | An in-tree lowering exists. On a Spyre host, run with `--probe`, then write a test with the `write-spyre-op-test` skill |
| 🟧 CPU fallback | Works, but costs host round trips. Note the ATen op and the issue in its notes |
| 🟨 Eager fallback | Works, but splits the compiled graph |

- A torch op can show different statuses for different shapes ("varies by
  shape"). Check its signature table in Op details.
- **Graph breaks** list code that runs eagerly and is not analyzed.

## 3. Confirm on a device (Spyre host only)

`--probe` compiles and runs every unique signature on the device and
compares it with the CPU. `--e2e` does the same for the whole model. Use
them to catch layout and stick-alignment failures, which the static analysis
cannot see.

## 4. Keep it accurate

- Support facts come from torch-spyre's registries, so a new lowering
  shows up in the next report without any change to the tool.
- Curated notes (open issues, PRs in flight) live in `KNOWN_ISSUES` in
  `tests/op_coverage/spyre_op_coverage/knowledge.py`. Remove an entry when its
  issue closes.
- A new model family needs a recipe in
  `tests/op_coverage/spyre_op_coverage/recipes.py` (`@register_recipe`).
- Tests: `python3 -m pytest tests/op_coverage/test_op_coverage.py`.
