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

The documentation is organized in four parts:

* **Overview** — the big-picture architecture of the compilation
  pipeline.
* **Compilers** — deep dives on the Inductor front-end and the DeepTools
  back-end.
* **Operations** — how to add new operations to the Spyre backend.
* **Optimization passes** — the pre-scheduling transformations applied
  by the front-end. These are presented in pipeline order: coarse-tiling
  (the IR rewrite that enables working-set reduction), then
  work-division across cores, then scratchpad placement.

For the project workflow around enabling and triaging new ops (issues,
test coverage, bug classification), see :doc:`/contributing/op_enablement`.

.. toctree::
   :maxdepth: 2
   :caption: Overview

   architecture

.. toctree::
   :maxdepth: 2
   :caption: Compilers

   inductor_frontend
   backend

.. toctree::
   :maxdepth: 2
   :caption: Operations

   adding_operations

.. toctree::
   :maxdepth: 2
   :caption: Optimization passes

   coarse_tiling_loops
   work_division_planning
   scratchpad_planning
