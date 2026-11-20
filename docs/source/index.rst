.. Torch-Spyre documentation master file

Torch-Spyre Documentation
==========================

**Torch-Spyre** is the PyTorch backend for the `IBM Spyre AI Accelerator
<https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator>`_.
It enables standard PyTorch models to run on the Spyre device with full
``torch.compile`` support via a custom Inductor backend.

.. admonition:: New to Torch-Spyre?
   :class: tip

   Choose a starting point based on what you need:

   * **Just want to run a model?** Start with :doc:`getting_started/quickstart`.
   * **Need the mental model?** Read :doc:`getting_started/key_concepts`,
     a short primer on dataflow execution, sticks and tiled tensors,
     the LX scratchpad, the eager and compiled paths, and graph breaks.
   * **Want the design story?** :doc:`getting_started/how_torch_spyre_works`
     describes the four integration challenges and the PyTorch extension
     mechanisms that address each one.

   For a one-line definition of a specific term, jump to the
   :doc:`getting_started/glossary`.

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

.. toctree::
   :caption: Explorer
   :maxdepth: 2

   explorer/index
   explorer/using_the_explorer

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
