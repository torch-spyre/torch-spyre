# Examples

The
[examples/](https://github.com/torch-spyre/torch-spyre/tree/main/examples)
directory in this repository contains self-contained scripts demonstrating
common Torch-Spyre use cases.

## Available Examples

| Script | Description |
|--------|-------------|
| `tensor_allocate.py` | Creating and allocating tensors on the Spyre device |
| `softmax.py` | Computing softmax on Spyre |
| `gelu.py` | Computing GELU activation on Spyre |
| `mean.py` | Computing mean reduction on Spyre |
| `mul.py` | Element-wise multiplication on Spyre |
| `softplus.py` | Computing softplus activation on Spyre |

## Running an Example

```bash
python examples/tensor_allocate.py
python examples/softmax.py
```

## Writing Your Own Example

A minimal Torch-Spyre script follows this pattern:

```python
import torch

DEVICE = torch.device("spyre")

# Move data to device
x = torch.rand(512, 1024, dtype=torch.float16).to(DEVICE)

# Run computation (optionally with torch.compile)
output = torch.some_op(x)

# Move result back to CPU for inspection
print(output.cpu())
```

## See Also

- [Quickstart](../getting_started/quickstart.md)
- [Running Models](running_models.md)
