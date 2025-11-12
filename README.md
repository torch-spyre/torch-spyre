# Torch Spyre Device Enablement

This repo will hold the required Pytorch C++ and Python layer code for supporting the [IBM Spyre device](https://www.ibm.com/new/announcements/ibm-spyre-accelerator-and-telum-ii-processor-capturing-ai-value-at-a-trusted-enterprise-level) as the new device, named `spyre`, in PyTorch.

## Setup

### Pytorch Version

To properly set this up, you will need to have a specific version of pytorch that compiles using C++11 ABI.

* torch-2.7.1+cpu.cxx11.abi-cp311-cp311-linux_x86_64.whl
* build your own with C++11 ABI enabled

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu-cxx11-abi/
```

### How to Install

To install Torch Spyre, clone the repository and from the cloned directory run the following command:

```bash
pip3 install -e . --no-deps --no-build-isolation -vvv --verbose
```

### How to Try It Out

Non-interactive, simple script:

```
python3 -m pytest tests/

python3 examples/tensor_allocate.py
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
* `DT_DEEPRT_VERBOSE=-1` to reduce Spyre stack logging
* `DTLOG_LEVEL=error` to reduce Spyre stack logging

## Description

This implementation of eager mode for IBM Spyre device is based on the self-contained example of a Pytorch out-of-tree backend leveraging the "PrivateUse1" backend from core. For that project, you can visit this [link](https://github.com/pytorch/pytorch/tree/v2.7.1/test/cpp_extensions/open_registration_extension).

Unlike open_registration_extension, most of the code for this will be done in C++ utilizing the lower level spyre repositories.

## Folder Structure

This project contains 2 main folders for development:

* `torch_spyre`: This will contain all required Python code to enable eager (currently this is being updated). This [link](https://github.com/pytorch/pytorch/tree/v2.7.1/test/cpp_extensions/open_registration_extension) describes the design principles we follows. For the most part, all that will be necessary from a Python standpoint is registering the device with PrivateUse1.

* `torch_spyre/csrc`: This will be where all of the Spyre-specific implementations of pytorch tensor ops / management functions will be.
