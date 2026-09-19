import torch
from typing import Optional
from torch_spyre import _C  # C++ extension module
from torch_spyre.constants import DEVICE_NAME

# Expose in module
__all__ = [
    "Stream",
    "stream",
    "current_stream",
    "default_stream",
    "synchronize",
]


class Stream:
    """
    Wrapper around a Spyre stream.

    A stream is a linear sequence of execution that belongs to a specific device.
    Operations on different streams can execute concurrently.

    Args:
        device (torch.device, optional): Device for the stream. Default: current device
        priority (int, optional): Priority of the stream. Lower numbers = higher priority.
                                  Default: 0

    Example:
        >>> dev = torch.device("spyre")
        >>> stream = torch.Stream(dev) //modern use
        >>> with torch.stream(stream):
        ...     x = torch.randn(100, device='spyre')

        >>> stream = torch_spyre.Stream()
        >>> with torch_spyre.stream(stream):
        ...     x = torch.randn(100, device='spyre')
    """

    def __init__(self, device: Optional[torch.device] = None, priority: int = 0):
        if device is None:
            # Use current device
            device = torch.device(DEVICE_NAME, torch.spyre.current_device())
        elif isinstance(device, int):
            device = torch.device(DEVICE_NAME, device)
        elif isinstance(device, str):
            device = torch.device(device)

        # Get stream from pool via C++ binding
        self._cdata = _C.get_stream_from_pool(device, priority)
        # Stack of streams displaced by __enter__, restored in __exit__.
        # A stack (rather than a single slot) makes the context manager safe
        # to re-enter with the same Stream instance, e.g. via
        # `with s: with s: ...` or `torch_spyre.stream(s)`, which hands back
        # the same object rather than a fresh context manager.
        self._prev_streams: list[_C._SpyreStreamBase] = []

    @classmethod
    def _from_cdata(cls, cdata) -> "Stream":
        """Wrap an existing C++ stream handle without going through
        __init__ (which would allocate a new stream from the pool)."""
        stream_obj = cls.__new__(cls)
        stream_obj._cdata = cdata
        stream_obj._prev_streams = []
        return stream_obj

    def __enter__(self):
        """Enter stream context - set as current stream"""
        # Save previous stream
        self._prev_streams.append(_C.current_stream(self.device()))
        # Set this stream as current
        _C.set_current_stream(self._cdata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit stream context - restore previous stream"""
        if not self._prev_streams:
            raise RuntimeError("Stream.__exit__ called without a matching __enter__")
        # Restore previous stream
        _C.set_current_stream(self._prev_streams.pop())
        return False

    def synchronize(self):
        """Wait for all operations on this stream to complete"""
        self._cdata.synchronize()

    def query(self) -> bool:
        """Check if all operations on this stream have completed"""
        return self._cdata.query()

    def device(self) -> torch.device:
        """Get the device associated with this stream"""
        return self._cdata.device()

    @property
    def id(self) -> int:
        """Get the stream ID"""
        return self._cdata.id()

    @property
    def priority(self) -> int:
        """Get the stream priority"""
        return self._cdata.priority()

    def __repr__(self):
        return self._cdata.__repr__()

    def __eq__(self, other):
        if not isinstance(other, Stream):
            return False
        return self.id == other.id and self.device() == other.device()

    def __hash__(self):
        return hash((self.device(), self.id))


def stream(stream: Stream):
    """
    Context manager for stream.

    All operations in the context will be executed on the specified stream.

    Args:
        stream (Stream): The stream to use

    Example:
        >>> s = torch_spyre.Stream()
        >>> with torch.stream(s):
        ...     x = torch.randn(100, device='spyre')
    """
    return stream  # Stream class already has __enter__/__exit__


def current_stream(device: Optional[torch.device] = None) -> Stream:
    """
    Get the current stream for a device.

    Args:
        device (torch.device, optional): Device to query. Default: current device

    Returns:
        Stream: The current stream

    Example:
        >>> s = torch.spyre.current_stream()
        >>> s = torch_spyre.current_stream()
        >>> print(s)
    """
    if device is None:
        device = torch.device(DEVICE_NAME, torch.spyre.current_device())
    elif isinstance(device, int):
        device = torch.device(DEVICE_NAME, device)

    cdata = _C.current_stream(device)

    # Wrap in Python Stream object
    return Stream._from_cdata(cdata)


def default_stream(device: Optional[torch.device] = None) -> Stream:
    """
    Get the default stream for a device.

    Args:
        device (torch.device, optional): Device to query. Default: current device

    Returns:
        Stream: The default stream (stream ID 0)
    """
    if device is None:
        device = torch.device(DEVICE_NAME, torch.spyre.current_device())
    elif isinstance(device, int):
        device = torch.device(DEVICE_NAME, device)

    cdata = _C.default_stream(device)

    return Stream._from_cdata(cdata)


def synchronize(device: Optional[torch.device] = None):
    """
    Synchronize all streams on a device.

    Args:
        device (torch.device, optional): Device to synchronize.
                                        If None, synchronizes all devices.

    Example:
        >>> stream.synchronize()  # Sync given stream
        >>> torch_spyre.synchronize()  # Sync all devices
        >>> torch_spyre.synchronize('spyre:0')  # Sync device 0
    """
    if device is not None:
        if isinstance(device, int):
            device = torch.device(DEVICE_NAME, device)
        elif isinstance(device, str):
            device = torch.device(device)

    _C.synchronize(device)
