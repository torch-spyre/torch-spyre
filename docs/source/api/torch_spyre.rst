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

      The stream priority class (read-only). ``0`` for low-priority,
      anything non-zero for high-priority.

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
   * - ``LX_PLANNING=1``
     - Enable LX scratchpad memory planning during the pre-scheduling pass
   * - ``DXP_LX_FRAC_AVAIL``
     - Fraction of LX scratchpad available to the planner

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
