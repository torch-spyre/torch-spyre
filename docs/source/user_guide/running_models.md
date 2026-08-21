# Running Models on Spyre

This page explains how to run full PyTorch models on the Spyre device
using `torch.compile` and the Torch-Spyre backend.

To run a stock HuggingFace Transformers checkpoint without writing your own
model code, see [Running HuggingFace models on Spyre](running_hf_models.md),
which covers the [hf-adapters](https://github.com/torch-spyre/hf-adapters)
project.

## Using `torch.compile`

Torch-Spyre registers itself as an Inductor backend for the `spyre`
device. Any model compiled with `torch.compile` and targeting the
`spyre` device is automatically routed through the Torch-Spyre compiler.

```python
import torch

DEVICE = torch.device("spyre")

model = MyModel().to(DEVICE)
compiled_model = torch.compile(model)

x = torch.rand(1, 3, 224, 224, dtype=torch.float16).to(DEVICE)
output = compiled_model(x)
```

## Supported Operations

For the full list of supported operations, see
[Supported Operations](supported_operations.md).

To add support for a new operation, see
[Adding Operations](../compiler/adding_operations.md).

## Configuration

Work division (core parallelism) is controlled by the `SENCORES`
environment variable:

```bash
SENCORES=32 python my_script.py
```

Valid values: 1–32 (default: 32). See
[Work Division Planning](../compiler/work_division_planning.md) for details.

## Examples

Full working examples are listed on the [Examples](examples/index.md)
page. It has single-op scripts (`tensor_allocate.py`, `softmax.py`,
`gelu.py`, `mean.py`, `mul.py`, `softplus.py`, `spyre_hints.py`), a
`distributed/` set covering the collective ops (allgather, allreduce,
broadcast, gather, reduce, barrier) plus a multi-rank broadcast
walkthrough, and a `scratchpad/` set that models the LX layout solver in
isolation. The scripts are under
[examples/](https://github.com/torch-spyre/torch-spyre/tree/main/docs/source/user_guide/examples).

## Troubleshooting

When a model fails to compile or produces unexpected results, the following
resources cover the common cases:

- [Debugging](debugging/index.md) explains how to enable compiler logging,
  dump Inductor artifacts, and inspect intermediate representations with
  `TORCH_LOGS`, `TORCH_COMPILE_DEBUG`, and `TORCH_SPYRE_DEBUG`.
- [Supported Operations](supported_operations.md) lists which operations run on
  Spyre and which fall back to the CPU. An unsupported operation in the model
  forces a graph break or a CPU fallback.
- [Profiling](profiling/index.md) shows how to measure where time is spent once
  a model runs correctly.
