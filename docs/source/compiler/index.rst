Compiler Stack
==============

This section describes the full compilation pipeline that transforms
PyTorch models into programs executable on the Spyre hardware.

The pipeline consists of two compilers:

* **Inductor front-end** — an open-source PyTorch Inductor extension
  implemented as part of Torch-Spyre. It maps FX graphs to Spyre
  operations and generates SuperDSC specifications.
* **DeepTools back-end** — a proprietary compiler that translates
  SuperDSC into optimized Spyre program binaries.

.. toctree::
   :maxdepth: 2

   architecture
   inductor_frontend
   backend
   work_division_planning
   work_division_codegen
   adding_operations
