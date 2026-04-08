torch\_spyre
============

After importing ``torch_spyre``, the Spyre backend is registered and
accessible via ``torch.spyre``. The public API follows the same patterns
as ``torch.cuda``.

.. code-block:: python

   import torch
   import torch_spyre  # registers the "spyre" device

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

.. function:: torch.spyre.get_device_properties(device=None) -> SpyreDeviceProperties

   Returns the properties of the given Spyre device.

   :param device: Device to query. If ``None``, uses the current device.
   :type device: torch.device or int, optional
   :returns: A dataclass with the following fields:

      - ``type`` (*str*) — hardware type identifier (e.g., ``"dd2"``)
      - ``index`` (*int*) — device index
      - ``multi_processor_count`` (*int*) — number of cores (up to 32)

   .. code-block:: python

      >>> props = torch.spyre.get_device_properties(0)
      >>> props.multi_processor_count
      32

Random Number Generation
------------------------

**Preferred (device-agnostic):** Use the PyTorch ``torch.accelerator`` API so
that your code is portable across backends (CUDA, Spyre, etc.):

.. code-block:: python

   torch.accelerator.manual_seed(42)      # current device
   torch.accelerator.manual_seed_all(42)  # all devices

**Backend-specific alternative:**

.. function:: torch.spyre.manual_seed(seed, device=None)

   Sets the seed for generating random numbers on the specified Spyre device.

   :param int seed: The desired seed.
   :param device: Device index to set the seed on. If ``None``, uses the
       current device.
   :type device: int, optional

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
   device. Operations on different streams can execute concurrently.

   :param device: Device for the stream. If ``None``, uses the current device.
   :type device: torch.device or int, optional
   :param int priority: Priority of the stream. Lower numbers indicate
       higher priority. Default: ``0``.

   .. code-block:: python

      >>> s = torch.spyre.Stream()
      >>> with torch.spyre.stream(s):
      ...     x = torch.randn(100, device="spyre", dtype=torch.float16)

   .. method:: synchronize()

      Wait for all operations on this stream to complete.

   .. method:: query() -> bool

      Returns ``True`` if all operations on this stream have completed.

   .. method:: device() -> torch.device

      Returns the device associated with this stream.

   .. attribute:: id
      :type: int

      The stream ID (read-only).

   .. attribute:: priority
      :type: int

      The stream priority (read-only). Lower numbers indicate higher priority.

.. function:: torch.spyre.stream(stream)

   Context manager that sets the given stream as the current stream.
   All operations within the context will be dispatched to the specified
   stream.

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
          dim_map=[0, 1, 1],
          stride_map=[1, 1, 1],
          device_dtype=DataFormats.SEN169_FP16,
      )

   .. attribute:: device_size
      :type: list[int]

      Shape on device, including tiling dimensions and padding.

   .. attribute:: dim_map
      :type: list[int]

      Mapping from each device dimension back to the corresponding
      PyTorch (host) dimension.

   .. attribute:: stride_map
      :type: list[int]

      Stride mapping for device dimensions.

   .. attribute:: device_dtype
      :type: DataFormats

      The on-device data format (e.g., ``SEN169_FP16``).

   .. method:: elems_per_stick() -> int

      Returns the number of elements per stick for this layout's dtype.

   .. method:: host_stick_dim() -> int

      Returns the host dimension index that corresponds to the stick
      (innermost) dimension on device.

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

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Purpose
   * - ``TORCH_SPYRE_DEBUG=1``
     - Enable C++ debug logging and ``-O0`` builds
   * - ``SENCORES``
     - Number of Spyre cores (1--32, default 32)
   * - ``TORCH_SPYRE_DOWNCAST_WARN=0``
     - Suppress float32 → float16 downcast warnings
   * - ``SPYRE_INDUCTOR_LOG=1``
     - Enable Spyre Inductor logging
   * - ``SPYRE_INDUCTOR_LOG_LEVEL=DEBUG``
     - Set Spyre Inductor log verbosity (DEBUG, INFO, WARNING, ERROR)
   * - ``SPYRE_LOG_FILE=path``
     - Redirect Spyre Inductor logs to a file
   * - ``TORCH_LOGS="+inductor"``
     - Verbose PyTorch Inductor logging
   * - ``TORCH_COMPILE_DEBUG=1``
     - Dump Inductor debug artifacts
   * - ``TORCH_SENDNN_LOG``
     - SendNN library logging level (default: ``CRITICAL``)
   * - ``DT_DEEPRT_VERBOSE``
     - DeepTools runtime verbosity (default: ``-1``, disabled)
   * - ``DTLOG_LEVEL``
     - DeepTools log level (default: ``error``)
