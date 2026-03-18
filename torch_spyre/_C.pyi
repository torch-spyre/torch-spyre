"""
Spyre C++ bindings
"""

from __future__ import annotations
import collections.abc
import torch
import typing

__all__: list[str] = [
    "DataFormats",
    "SpyreTensorLayout",
    "_SpyreStreamBase",
    "current_stream",
    "default_stream",
    "get_stream_from_pool",
    "set_current_stream",
    "synchronize",
    "as_strided_with_layout",
    "convert_artifacts",
    "empty_with_layout",
    "encode_constant",
    "free_runtime",
    "get_device_dtype",
    "get_downcast_warning",
    "get_elem_in_stick",
    "get_spyre_tensor_layout",
    "launch_kernel",
    "set_downcast_warning",
    "set_spyre_tensor_layout",
    "spyre_empty_with_layout",
    "start_runtime",
    "to_with_layout",
]

class DataFormats:
    """
    Members:

      SEN169_FP16

      IEEE_FP32

      INVALID

      SEN143_FP8

      SEN152_FP8

      SEN153_FP9

      SENINT2

      SENINT4

      SENINT8

      SENINT16

      SENINT24

      IEEE_INT64

      IEEE_INT32

      SENUINT32

      SENUINT2

      IEEE_FP16

      BOOL

      BFLOAT16

      SEN18F_FP24
    """

    BFLOAT16: typing.ClassVar[DataFormats]  # value = <DataFormats.BFLOAT16: 17>
    BOOL: typing.ClassVar[DataFormats]  # value = <DataFormats.BOOL: 16>
    IEEE_FP16: typing.ClassVar[DataFormats]  # value = <DataFormats.IEEE_FP16: 15>
    IEEE_FP32: typing.ClassVar[DataFormats]  # value = <DataFormats.IEEE_FP32: 1>
    IEEE_INT32: typing.ClassVar[DataFormats]  # value = <DataFormats.IEEE_INT32: 12>
    IEEE_INT64: typing.ClassVar[DataFormats]  # value = <DataFormats.IEEE_INT64: 11>
    INVALID: typing.ClassVar[DataFormats]  # value = <DataFormats.INVALID: 2>
    SEN143_FP8: typing.ClassVar[DataFormats]  # value = <DataFormats.SEN143_FP8: 3>
    SEN152_FP8: typing.ClassVar[DataFormats]  # value = <DataFormats.SEN152_FP8: 4>
    SEN153_FP9: typing.ClassVar[DataFormats]  # value = <DataFormats.SEN153_FP9: 5>
    SEN169_FP16: typing.ClassVar[DataFormats]  # value = <DataFormats.SEN169_FP16: 0>
    SEN18F_FP24: typing.ClassVar[DataFormats]  # value = <DataFormats.SEN18F_FP24: 18>
    SENINT16: typing.ClassVar[DataFormats]  # value = <DataFormats.SENINT16: 9>
    SENINT2: typing.ClassVar[DataFormats]  # value = <DataFormats.SENINT2: 6>
    SENINT24: typing.ClassVar[DataFormats]  # value = <DataFormats.SENINT24: 10>
    SENINT4: typing.ClassVar[DataFormats]  # value = <DataFormats.SENINT4: 7>
    SENINT8: typing.ClassVar[DataFormats]  # value = <DataFormats.SENINT8: 8>
    SENUINT2: typing.ClassVar[DataFormats]  # value = <DataFormats.SENUINT2: 14>
    SENUINT32: typing.ClassVar[DataFormats]  # value = <DataFormats.SENUINT32: 13>
    __members__: typing.ClassVar[
        dict[str, DataFormats]
    ]  # value = {'SEN169_FP16': <DataFormats.SEN169_FP16: 0>, 'IEEE_FP32': <DataFormats.IEEE_FP32: 1>, 'INVALID': <DataFormats.INVALID: 2>, 'SEN143_FP8': <DataFormats.SEN143_FP8: 3>, 'SEN152_FP8': <DataFormats.SEN152_FP8: 4>, 'SEN153_FP9': <DataFormats.SEN153_FP9: 5>, 'SENINT2': <DataFormats.SENINT2: 6>, 'SENINT4': <DataFormats.SENINT4: 7>, 'SENINT8': <DataFormats.SENINT8: 8>, 'SENINT16': <DataFormats.SENINT16: 9>, 'SENINT24': <DataFormats.SENINT24: 10>, 'IEEE_INT64': <DataFormats.IEEE_INT64: 11>, 'IEEE_INT32': <DataFormats.IEEE_INT32: 12>, 'SENUINT32': <DataFormats.SENUINT32: 13>, 'SENUINT2': <DataFormats.SENUINT2: 14>, 'IEEE_FP16': <DataFormats.IEEE_FP16: 15>, 'BOOL': <DataFormats.BOOL: 16>, 'BFLOAT16': <DataFormats.BFLOAT16: 17>, 'SEN18F_FP24': <DataFormats.SEN18F_FP24: 18>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    def elems_per_stick(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SpyreTensorLayout:
    __hash__: typing.ClassVar[None] = None  # type: ignore
    def __eq__(self, arg0: SpyreTensorLayout) -> bool: ...  # type: ignore
    @typing.overload
    def __init__(
        self,
        host_size: collections.abc.Sequence[typing.SupportsInt],
        dtype: torch.dtype,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        host_size: collections.abc.Sequence[typing.SupportsInt],
        dtype: torch.dtype,
        dim_order: collections.abc.Sequence[typing.SupportsInt],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        device_size: collections.abc.Sequence[typing.SupportsInt],
        dim_map: collections.abc.Sequence[typing.SupportsInt],
        device_dtype: DataFormats,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def elems_per_stick(self) -> int: ...
    def host_stick_dim(self) -> int: ...
    def similar_dim_order(self, arg0: typing.SupportsInt) -> list[int]: ...
    @property
    def device_dtype(self) -> DataFormats: ...
    @property
    def device_size(self) -> list[int]: ...
    @property
    def dim_map(self) -> list[int]: ...

class _SpyreStreamBase:
    """
    C++ SpyreStream wrapper class.

    Represents a stream of execution on a Spyre device.
    """
    def synchronize(self) -> None:
        """Wait for all operations on this stream to complete"""
        ...

    def query(self) -> bool:
        """Check if all operations on this stream have completed"""
        ...

    def device(self) -> torch.device:
        """Get the device associated with this stream"""
        ...

    def id(self) -> int:
        """Get the stream ID"""
        ...

    def priority(self) -> int:
        """Get the stream priority"""
        ...

    def __repr__(self) -> str: ...

def get_stream_from_pool(device: torch.device, priority: int = 0) -> _SpyreStreamBase:
    """
    Get a stream from the pool for the specified device and priority.

    Args:
        device: The device for which to get a stream
        priority: Stream priority (lower = higher priority). Default: 0

    Returns:
        A SpyreStream object from the pool
    """
    ...

def current_stream(device: torch.device) -> _SpyreStreamBase:
    """
    Get the current stream for a device.

    Args:
        device: The device to query

    Returns:
        The current stream for the device
    """
    ...

def default_stream(device: torch.device) -> _SpyreStreamBase:
    """
    Get the default stream for a device.

    Args:
        device: The device to query

    Returns:
        The default stream (stream ID 0) for the device
    """
    ...

def set_current_stream(stream: _SpyreStreamBase) -> None:
    """
    Set the current stream for the stream's device.

    Args:
        stream: The stream to set as current
    """
    ...

def synchronize(device: torch.device | None = None) -> None:
    """
    Synchronize all streams on a device or all devices.

    Args:
        device: The device to synchronize. If None, synchronizes all devices.
    """
    ...

def as_strided_with_layout(
    arg0: torch.Tensor,
    arg1: tuple[int, ...],
    arg2: tuple[int, ...],
    arg3: typing.SupportsInt | None,
    arg4: SpyreTensorLayout,
) -> torch.Tensor: ...
def convert_artifacts(arg0: str) -> None: ...
def empty_with_layout(
    arg0: tuple[int, ...],
    arg1: SpyreTensorLayout,
    arg2: torch.dtype | None,
    arg3: torch.device | None,
    arg4: bool | None,
    arg5: torch.memory_format | None,
) -> torch.Tensor: ...
def encode_constant(arg0: typing.SupportsFloat, arg1: DataFormats) -> int: ...
def free_runtime() -> None: ...
def get_device_dtype(arg0: torch.dtype) -> DataFormats: ...
def get_downcast_warning() -> bool:
    """
    Return whether downcast warnings are enabled.
    """

def get_elem_in_stick(arg0: torch.dtype) -> int: ...
def get_spyre_tensor_layout(arg0: torch.Tensor) -> SpyreTensorLayout: ...
def launch_kernel(arg0: str, arg1: collections.abc.Sequence[torch.Tensor]) -> None: ...
def set_downcast_warning(arg0: bool) -> None:
    """
    Enable/disable downcast warnings for this process.
    """

def set_spyre_tensor_layout(arg0: torch.Tensor, arg1: SpyreTensorLayout) -> None: ...
def spyre_empty_with_layout(
    arg0: tuple[int, ...],
    arg1: tuple[int, ...],
    arg2: torch.dtype,
    arg3: SpyreTensorLayout,
) -> torch.Tensor: ...
def start_runtime() -> None: ...
def to_with_layout(arg0: torch.Tensor, arg1: SpyreTensorLayout) -> torch.Tensor: ...
