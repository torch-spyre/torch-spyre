.. Torch-Spyre documentation master file

Torch-Spyre Documentation
==========================

**Torch-Spyre** is the PyTorch backend for the `IBM Spyre AI Accelerator
<https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator>`_.
It enables standard PyTorch models to run on the Spyre device with full
``torch.compile`` support via a custom Inductor backend.

.. admonition:: New to Torch-Spyre?
   :class: tip

   Start with :doc:`getting_started/how_torch_spyre_works` for a walkthrough
   of how Torch-Spyre integrates with PyTorch as an out-of-tree backend,
   including the four design challenges we hit and the PyTorch extension
   mechanism that addressed each one.

.. toctree::
   :caption: For Users
   :maxdepth: 3

   getting_started/index
   user_guide/index
   api/index

.. toctree::
   :caption: For Developers
   :maxdepth: 3

   architecture/index
   compiler/index
   runtime/index
   contributing/index
   rfcs/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
