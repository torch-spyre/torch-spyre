# Torch Spyre Device Enablement

This project contains the PyTorch layer C++ and Python code for supporting the [IBM Spyre device](./docs/source/architecture/spyre_accelerator.md) as a new device, named `spyre`, in PyTorch.

## Documentation

Full documentation: <https://torch-spyre.readthedocs.io/>

To build the docs locally:

```bash
pip install -r docs/requirements.txt
cd docs && make html
```

See the [Documentation Contributor Guide](./docs/README.md) for details.

## Setup and Build

Building this project currently requires a development build of the IBM Spyre Software Stack.
Internal build instructions are available to IBM employees through internal documentation channels.

## How to Try It Out

Non-interactive, simple script:

```
python3 -m pytest tests/

python3 examples/tensor_allocate.py

python3 examples/softmax.py
```

Interactive:

```
python3
>>> import torch
>>> x = torch.tensor([1,2], dtype=torch.float16, device="spyre")
>>> x.device
device(type='spyre', index=0)
```

Controlling logging:

* `TORCH_SPYRE_DEBUG=1` to enable debug logging
* `TORCH_SPYRE_DOWNCAST_WARN=0` to disable downcast warning (accept: 0/1, true/false, on/off)
* `SPYRE_INDUCTOR_LOG=1` to enable Spyre Inductor logging
* `SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` to set Spyre Inductor log verbosity (DEBUG, INFO, WARNING, ERROR)
* `DT_DEEPRT_VERBOSE=-1` to reduce Spyre stack logging
* `DTLOG_LEVEL=error` to reduce Spyre stack logging

## Description

This implementation of a PyTorch backend for IBM Spyre device is based on the self-contained example of a PyTorch out-of-tree backend leveraging the "PrivateUse1" backend from core. For that project, you can visit this [link](https://github.com/pytorch/pytorch/tree/v2.9.1/test/cpp_extensions/open_registration_extension).

Unlike open_registration_extension, most of the code for this will be done in C++ utilizing the lower level spyre repositories.

## Folder Structure

This project contains 2 main folders for development:

* `torch_spyre`: This will contain all required Python code to enable eager (currently this is being updated). This [link](https://github.com/pytorch/pytorch/tree/v2.9.1/test/cpp_extensions/open_registration_extension) describes the design principles we follows. For the most part, all that will be necessary from a Python standpoint is registering the device with PrivateUse1.

* `torch_spyre/csrc`: This will be where all of the Spyre-specific implementations of PyTorch tensor ops / management functions will be.

## Profiling

Profiling support is under active development. See `torch_spyre/profiler/` — requires the kineto-spyre wheel (version matching the PyTorch install).

The kineto-spyre wheel install is required currently for `profiler/__init__.py` and `profiler/_spyre_activity.py`.
