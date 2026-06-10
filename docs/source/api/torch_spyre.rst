torch\_spyre
============

When the ``torch_spyre`` package is installed, PyTorch picks it up
through the ``torch.backends`` autoload entry point — no explicit
``import torch_spyre`` is needed. The Spyre backend registers itself
on first use of ``torch`` and the public API is available under
``torch.spyre``, mirroring the ``torch.cuda`` surface.

.. code-block:: python

   import torch

   torch.spyre.is_available()
   torch.spyre.device_count()

Device Management
-----------------

.. function:: torch.spyre.is_available() -> bool

   Returns ``True`` if at least one Spyre device is available.

   .. code-block:: python

      >>> torch.spyre.is_available()
      True

.. function:: torch.spyre.device_count() -> int

   Returns the number of Spyre devices available.

   .. code-block:: python

      >>> torch.spyre.device_count()
      1

.. function:: torch.spyre.current_device() -> int

   Returns the index of the currently selected Spyre device.

   .. code-block:: python

      >>> torch.spyre.current_device()
      0

.. function:: torch.spyre.set_device(idx)

   Sets the current Spyre device.

   :param int idx: Device index to set as current.

.. function:: torch.spyre.is_initialized() -> bool

   Returns ``True`` if the Spyre runtime has been initialized.

.. function:: torch.spyre.get_amp_supported_dtype() -> list[torch.dtype]

   Returns the dtypes supported by ``torch.autocast`` on Spyre. Used by the
   PyTorch AMP machinery to validate the autocast dtype.

   .. code-block:: python

      >>> torch.spyre.get_amp_supported_dtype()
      [torch.float16, torch.bfloat16]

.. note::

   ``torch.spyre.get_device_properties()`` is not yet exposed on the public
   ``torch.spyre`` namespace. The ``SpyreDeviceProperties`` dataclass and
   ``SpyreInterface.get_device_properties()`` exist internally and are used
   by the Inductor device interface (see ``torch_spyre/device/interface.py``).

Random Number Generation
------------------------

**Preferred (device-agnostic):** Use the PyTorch ``torch.accelerator`` API so
that your code is portable across backends (CUDA, Spyre, etc.):

.. code-block:: python

   torch.accelerator.manual_seed(42)      # current device
   torch.accelerator.manual_seed_all(42)  # all devices

**Backend-specific alternative:**

.. function:: torch.spyre.manual_seed(seed)

   Sets the seed for generating random numbers on the current Spyre device.

   :param int seed: The desired seed.

   .. note::

      The public binding accepts a single ``seed`` argument. To target a
      specific device, either call ``set_device`` first, or use
      ``torch.spyre.manual_seed_all``, which seeds every visible Spyre
      device.

.. function:: torch.spyre.manual_seed_all(seed)

   Sets the seed for generating random numbers on all Spyre devices.

   :param int seed: The desired seed.

Streams
-------

Streams allow overlapping execution of operations. The API mirrors
``torch.cuda`` streams.

.. class:: torch.spyre.Stream(device=None, priority=0)

   Wrapper around a Spyre stream.

   A stream is a linear sequence of execution that belongs to a specific
   device. Operations on different streams can run concurrently. The
   ``Stream`` object is itself a context manager: putting it in a
   ``with`` block sets it as the current stream for that block.

   :param device: Device for the stream. Accepts ``torch.device``,
       ``int``, or a string like ``"spyre"`` or ``"spyre:0"``. If
       ``None``, the current device is used.
   :type device: torch.device or int or str, optional
   :param int priority: Priority class for the stream. ``0`` selects
       the low-priority pool; any non-zero value selects the
       high-priority pool. Each pool has 32 streams per device,
       allocated round-robin. Default: ``0``.

       The constructor input and the ``.priority`` getter use different
       conventions: a stream constructed with ``priority=5`` is placed
       in the high-priority pool, and its ``.priority`` attribute then
       reports ``-1`` rather than ``5``. See the ``priority`` attribute
       below.

   .. code-block:: python

      >>> s = torch.spyre.Stream()
      >>> with torch.spyre.stream(s):
      ...     x = torch.randn(100, device="spyre", dtype=torch.float16)

   .. method:: synchronize()

      Wait for all operations on this stream to complete.

   .. method:: query() -> bool

      Returns ``True`` if all operations on this stream have completed.

   .. method:: device() -> torch.device

      Returns the device associated with this stream. Unlike
      ``torch.cuda.Stream.device``, this is a method, not a property.

   .. attribute:: id
      :type: int

      The stream ID (read-only). ``0`` is the default stream, ``1`` to
      ``32`` are the low-priority streams, and ``33`` to ``64`` are the
      high-priority streams.

   .. attribute:: priority
      :type: int

      The stream priority class (read-only). Reports ``0`` for low-priority
      streams (IDs 0--32) and ``-1`` for high-priority streams (IDs 33--64),
      matching the convention used by ``torch.cuda.Stream.priority``. The
      attribute does not echo the integer passed to the constructor.

.. function:: torch.spyre.stream(stream)

   Pass-through helper for use inside a ``with`` block. The actual swap
   of the current stream is done by ``Stream.__enter__`` and
   ``Stream.__exit__``; calling ``stream(s)`` just returns ``s`` so the
   ``with`` form reads naturally.

   :param Stream stream: The stream to use.

   .. code-block:: python

      >>> s = torch.spyre.Stream()
      >>> with torch.spyre.stream(s):
      ...     x = torch.randn(100, device="spyre", dtype=torch.float16)

.. function:: torch.spyre.current_stream(device=None) -> Stream

   Returns the currently active stream for the given device.

   :param device: Device to query. If ``None``, uses the current device.
   :type device: torch.device or int, optional

.. function:: torch.spyre.default_stream(device=None) -> Stream

   Returns the default stream (stream ID 0) for the given device.

   :param device: Device to query. If ``None``, uses the current device.
   :type device: torch.device or int, optional

.. function:: torch.spyre.synchronize(device=None)

   Waits for all operations on all streams to complete. If a device
   is specified, synchronizes only that device.

   :param device: Device to synchronize. If ``None``, synchronizes all
       devices.
   :type device: torch.device or int or str, optional

   .. code-block:: python

      >>> torch.spyre.synchronize()          # sync all devices
      >>> torch.spyre.synchronize("spyre:0") # sync device 0

Distributed
-----------

Torch-Spyre registers a ``c10d::Backend`` named ``spyreccl`` for cross-card
collective communication. Standard PyTorch distributed setup applies:

.. code-block:: python

   import torch
   import torch.distributed as dist

   dist.init_process_group(backend="cpu:gloo,spyre:spyreccl")

   x = torch.zeros(1024, dtype=torch.float16, device="spyre")
   dist.broadcast(x, src=0)

The backend follows a one-device-per-process model: each rank attaches to a
single Spyre device and reuses the rank's existing flex runtime instance.
Supported collectives, the list of process-group entries that raise
``SpyreCCLNotSupportedException``, and the placement of
``SpyreCCLBackend`` in the runtime stack are documented in
:doc:`../runtime/index`.

Memory
------

``torch.spyre.memory`` re-exports ``torch.accelerator.memory``, so the
standard accelerator memory API is available against Spyre devices:

.. code-block:: python

   torch.spyre.memory.memory_allocated()        # bytes currently allocated
   torch.spyre.memory.max_memory_allocated()    # peak since the last reset
   torch.spyre.memory.reset_peak_memory_stats()

A worked example is in :doc:`../user_guide/profiling/index`.

Profiler
--------

.. function:: torch_spyre.profiler.is_available() -> bool

   Returns ``True`` when the Spyre profiler integration is built into the
   current package and the device can be profiled. Returns ``False`` in the
   default build today; the in-tree profiler package is a scaffold whose
   collection backends are still landing. See
   :doc:`../user_guide/profiling/index` for the current state and the
   profiling tooling that is available in the meantime.

Tensor Operations
-----------------

Spyre tensors are created using the ``device="spyre"`` argument:

.. code-block:: python

   # Create a tensor on Spyre
   x = torch.tensor([1, 2], dtype=torch.float16, device="spyre")

   # Move an existing tensor to Spyre
   y = cpu_tensor.to("spyre")

   # Move back to CPU
   z = x.cpu()

The default dtype for Spyre is ``torch.float16``. See
:doc:`../user_guide/tensors_and_layouts` for details on how tensors are
laid out in device memory.

Compilation
-----------

Spyre models are compiled using ``torch.compile`` with the ``"spyre"``
backend:

.. code-block:: python

   model = MyModel().to("spyre")
   compiled = torch.compile(model, backend="spyre")
   output = compiled(inputs)

See :doc:`../user_guide/running_models` for details and
:doc:`../user_guide/supported_operations` for the list of supported ops.

Tensor Layouts
--------------

Spyre uses a tiled memory layout that differs from PyTorch's standard
strided layout. The following classes and functions allow inspection and
manipulation of device tensor layouts. See
:doc:`../user_guide/tensors_and_layouts` for background.

.. class:: torch_spyre._C.SpyreTensorLayout

   Describes how a tensor is laid out in Spyre device memory. Each
   ``SpyreTensorLayout`` captures the tiling, padding, and dimension
   mapping required by the hardware.

   Can be constructed in two ways:

   .. code-block:: python

      # From host tensor metadata (automatic layout computation)
      layout = SpyreTensorLayout(host_size=[4, 128], dtype=torch.float16)

      # From explicit device layout parameters
      layout = SpyreTensorLayout(
          device_size=[4, 2, 64],
          stride_map=[128, 64, 1],
          device_dtype=DataFormats.SEN169_FP16,
      )

   .. attribute:: device_size
      :type: list[int]

      Shape on device, including tiling dimensions and padding.

   .. attribute:: stride_map
      :type: list[int]

      Host stride for each device dimension. A value of -1 indicates a
      synthetic or padded dimension with no corresponding host stride.

   .. attribute:: device_dtype
      :type: DataFormats

      The on-device data format (e.g., ``SEN169_FP16``).

   .. method:: elems_per_stick() -> int

      Returns the number of elements per stick for this layout's dtype.

.. class:: torch_spyre._C.DataFormats

   Enumeration of Spyre on-device data formats. Each format defines the
   bit-level encoding used in device memory.

   Common values:

   .. attribute:: SEN169_FP16

      Spyre native 16-bit floating point (default for ``torch.float16``).

   .. attribute:: IEEE_FP32

      IEEE 754 single-precision floating point.

   .. attribute:: IEEE_FP16

      IEEE 754 half-precision floating point.

   .. attribute:: BFLOAT16

      Brain floating-point 16-bit format.

   .. attribute:: SEN143_FP8

      Spyre native 8-bit floating point (E4M3 variant).

   .. attribute:: SEN152_FP8

      Spyre native 8-bit floating point (E5M2 variant).

   .. attribute:: SENINT8

      Spyre native 8-bit integer.

   .. method:: elems_per_stick() -> int

      Returns the number of elements that fit in a single 128-byte stick
      for this data format.

.. function:: torch_spyre._C.get_spyre_tensor_layout(tensor) -> SpyreTensorLayout

   Returns the ``SpyreTensorLayout`` for a tensor that resides on a Spyre
   device.

   :param torch.Tensor tensor: A Spyre device tensor.
   :returns: The device layout of the tensor.
   :rtype: SpyreTensorLayout

   .. code-block:: python

      >>> x = torch.randn(4, 128, dtype=torch.float16, device="spyre")
      >>> layout = torch_spyre._C.get_spyre_tensor_layout(x)
      >>> print(layout.device_size)
      [4, 2, 64]

.. function:: torch_spyre._C.set_spyre_tensor_layout(tensor, layout)

   Sets the ``SpyreTensorLayout`` on a Spyre device tensor.

   :param torch.Tensor tensor: A Spyre device tensor.
   :param SpyreTensorLayout layout: The layout to assign.

Warnings
--------

.. function:: torch_spyre._C.get_downcast_warning() -> bool

   Returns whether float32 → float16 downcast warnings are enabled.

.. function:: torch_spyre._C.set_downcast_warning(enabled)

   Enable or disable float32 → float16 downcast warnings.

   :param bool enabled: ``True`` to enable warnings, ``False`` to suppress.

   Can also be controlled via the ``TORCH_SPYRE_DOWNCAST_WARN`` environment
   variable.

Constants
---------

.. data:: torch_spyre.constants.DEVICE_NAME
   :value: "spyre"

   The device name string used to register Spyre with PyTorch.

.. data:: torch_spyre.constants.DISTRIBUTED_BACKEND_NAME
   :value: "spyreccl"

   The backend name used to register the Spyre distributed backend with
   ``torch.distributed``. Pass this string to ``init_process_group(backend=...)``.

Environment Variables
---------------------

**Spyre runtime and compiler:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Purpose
   * - ``TORCH_SPYRE_DEBUG=1``
     - Enable C++ debug logging and ``-O0`` builds
   * - ``TORCH_SPYRE_DOWNCAST_WARN=0``
     - Suppress float32 → float16 downcast warnings
   * - ``SPYRE_INDUCTOR_LOG=1``
     - Enable Spyre Inductor logging
   * - ``SPYRE_INDUCTOR_LOG_LEVEL=DEBUG``
     - Set Spyre Inductor log verbosity (DEBUG, INFO, WARNING, ERROR)
   * - ``SPYRE_LOG_FILE=path``
     - Redirect Spyre Inductor logs to a file
   * - ``TORCH_SENDNN_LOG``
     - SendNN library logging level (default: ``CRITICAL``)
   * - ``DT_DEEPRT_VERBOSE``
     - DeepTools runtime verbosity (default: ``-1``, disabled)
   * - ``DTLOG_LEVEL``
     - DeepTools log level (default: ``error``)

**Compiler / Inductor configuration** (``torch_spyre/_inductor/config.py``):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Purpose
   * - ``SENCORES``
     - Number of Spyre cores (1--32, default 32)
   * - ``DXP_LX_FRAC_AVAIL``
     - Fraction of LX scratchpad available to the planner (default ``0.2``)
   * - ``LX_PLANNING``
     - Enable LX scratchpad planning (default ``1``; set ``0`` to skip the
       ``scratchpad_planning`` pass)
   * - ``CO_OPTIMIZING_LX_PLANNING``
     - Use the co-optimizing LX allocator strategy (default ``0``)
   * - ``CHUNK_LARGE_TENSORS``
     - Run the ``chunk_large_tensors`` pass to split tensors that exceed
       the per-core span (default ``0``)
   * - ``GLOBAL_STICK_OPTIMIZER``
     - Enable the global stick-dimension optimizer (default ``1``)
   * - ``SPYRE_CORE_ID_K_FAST_EMISSION``
     - Permute physical core IDs at SDSC emission so K-collaborator cores
       sit on adjacent ring positions, reducing PSUM chain hops (default
       ``1``)
   * - ``BUNDLE_SYMBOLIC_ARGS``
     - Emit LPDDR5 tensor addresses as runtime symbols rather than baked
       integers (default ``0``)
   * - ``UNROLL_LOOPS``
     - Fully unroll ``LoopSpec`` nodes into flat ``OpSpec``\s before bundle
       generation (default ``1``; set ``0`` to keep the
       ``scf.for`` / ``affine.apply`` path)

**Device enumeration** (``torch_spyre/csrc/spyre_device_enum.cpp``):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Purpose
   * - ``AIU_WORLD_SIZE``
     - Override the visible Spyre device count
   * - ``SPYRE_DEVICES``
     - Comma-separated list of device indices to expose
   * - ``FLEX_DEVICE``
     - Select the underlying flex runtime mode (PF / VF)

**Internal:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Purpose
   * - ``IS_INDUCTOR_SPAWNED_SUBPROCESS``
     - Marker set by Inductor when spawning compile subprocesses

**Useful PyTorch knobs (not defined by torch-spyre):**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Purpose
   * - ``TORCH_LOGS="+inductor"``
     - Verbose PyTorch Inductor logging
   * - ``TORCH_COMPILE_DEBUG=1``
     - Dump Inductor debug artifacts
