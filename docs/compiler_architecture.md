# Overview

This document describes the high-level compiler architecture used in `torch-spyre`.

Compiling PyTorch programs for execution on the IBM Spyre accelerator involves two
separate compilers. The *front-end* compiler is an open source PyTorch Inductor extension
that is implemented as part of torch-spyre. The *back-end* compiler
is a proprietary compiler called `DeepTools` that is invoked by the front-end compiler
to generate program binaries.

The front-end compiler is responsible for the following tasks:
+ Op selection: mapping AI models from FX Graphs of `torch` operations to Spyre operations.
+ Work division: breaking tensors into tiles and distributing those tiles across Spyre cores
+ Memory management: assigning tensors to DDR and scratchpad memories
+ Host code: generating the code that orchestrates device/host data transfers and kernel launches

The back-end compiler is responsible for taking the core-level specifications
produced by the front-end compiler, mapping them to optimized Spyre dataflows, and producing
executable programs binaries.

# Background

A working knowledge of PyTorch's compilers is essential for understanding our front-end compiler.
Some useful resources are:
+ The [ASPLOS'24 paper on PyTorch2](https://docs.pytorch.org/assets/pytorch2-2.pdf)
+ The [ASPLOS'24 tutorial on PyTorch2](https://github.com/meta-pytorch/workshops/tree/master/ASPLOS_2024). In particular,
the portion on [Inductor](https://github.com/pytorch/workshops/tree/master/ASPLOS_2024/inductor.pdf)
+ General documentation on [torch.compiler](https://docs.pytorch.org/docs/stable/torch.compiler.html#)

# Front-end Compiler Overview

The front-end compiler works by registering itself with PyTorch as the Inductor backend
for the `spyre` device.  This causes compilation of any FX Graphs that are targeted
to a `spyre` device to be routed to the front-end compiler.  Compilation then proceeds
per the normal Dynamo/Inductor compilation flow through the following stages:
+ The program is traced and an FX Graph constructed.
+ Inductor performs a number of FX Graph rewrite passes, including decompositions of
complex operations into a smaller core ATen set of operations.
+ The FX Graph of core operations is lowered into LoopLevelIR.
+ A number of analysis and optimization passes are performed over the LoopLevelIR.
+ A final codegen pass over the LoopLevelIR generates Python code the contains both the kernels
that will be compiled by the backend compiler into device binaries and the host code that will
orchestrate their execution.

Some key entry points to the front-end compiler are:
+ [init.py](../torch_spyre/_inductor/__init__.py) registers the compiler and customizes the configuration of Inductor.
+ [decompositions.py](../torch_spyre/_inductor/decompositions.py) is where we add Spyre-specific decompositions of existing high-level ATen operations.
+ [custom_ops.py](../torch_spyre/_inductor/customops.py) is where we define new Spyre-specific operations.
+ [passes.py](../torch_spyre/_inductor/passes.py) is where we add Spyre-specific compiler passes into the three exported
extension points of Inductor. It supports adding passes to both the FX Graph and LoopLevelIR stages of compilation.
+ [spyre_kernel.py](../torch_spyre/_inductor/spyre_kernel.py) defines our compilation from LoopLevelIR into `KernelSpec`, our
high-level description of a single operation to be performed on the device.
+ [codegen](../torch_spyre/_inductor/codegen/) defines the compilation from a `KernelSpec` into a lower-level SuperDSC json file which is the input to the backend compiler.

## Additional Topics

+[adding_operations.md](./adding_operations.md) describes the steps to add a new supported operation to the front-end compiler.
