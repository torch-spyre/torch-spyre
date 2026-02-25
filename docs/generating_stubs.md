# Updating pybind11 Stubs

The C++ extension modules (`torch_spyre._C` and `torch_spyre._hooks`) expose
their interfaces to Python via pybind11.  Type stubs (`.pyi` files) for these
modules are generated with
[pybind11-stubgen](https://github.com/sizmailov/pybind11-stubgen).

## Prerequisites

Install the stub generator:

```bash
pip install pybind11-stubgen
```

## Generating the stubs

PyTorch must be imported before running the stub generator so that custom types
registered by PyTorch (e.g. `torch.Tensor`, `torch.dtype`) are available to the
introspection machinery.

From a Python shell:

```python
import torch
import pybind11_stubgen

pybind11_stubgen.main(["torch_spyre._C"])
pybind11_stubgen.main(["torch_spyre._hooks"])
```

This writes the stub files into the current working directory.  Review the
generated `.pyi` files and copy them into `torch_spyre/`:

```
torch_spyre/_C.pyi
torch_spyre/_hooks.pyi
```
