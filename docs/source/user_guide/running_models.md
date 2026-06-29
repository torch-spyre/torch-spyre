# Running Models on Spyre

This page explains how to run full PyTorch models on the Spyre device
using `torch.compile` and the Torch-Spyre backend.

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
`gelu.py`, `mean.py`, `mul.py`, `softplus.py`, `spyre_hints.py`) and a
`distributed/` set covering the collective ops (allgather, allreduce,
broadcast, gather, reduce, barrier). The scripts are under
[examples/](https://github.com/torch-spyre/torch-spyre/tree/main/docs/source/user_guide/examples).

## Troubleshooting

> **TODO:** Document common errors and how to resolve them.
