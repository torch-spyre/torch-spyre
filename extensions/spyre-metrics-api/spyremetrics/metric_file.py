# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Library to read Spyre metric files.

File format (64bit aligned):
- Magic (32bit) + File size (32bit)
- Version (32bit) + Num. Sections (32bit)
- Last update time (64bit)
- Reserved (64bit)
- Section size table:
  - Type ID of Section #0 (32bit) + Size of Section #0 (32bit)
  - Type ID of Section #1 (32bit) + Size of Section #1 (32bit)
  - ...
- Section header and contents of Section #0
- Section header and contents of Section #1
- ...

Section header format (64bit aligned):
- Section type ID (32bit) + Section size (32bit)
- Sequential number (32bit) + Data array length (32bit)
- Last update time (64bit)
- Update interval (32bit) + Reserved (32bit)
- Data type array:
  - Type ID of Metric data #0 (32bit) + Number of words for Metric data #0 (32bit)
  - Type ID of Metric data #1 (32bit) + Number of words for Metric data #1 (32bit)
  - ...
- Metric data #0 (array of 64bit)
- Metric data #1 (array of 64bit)
- ...
"""

import numpy as np
import os
import psutil
import sys

from enum import IntEnum, auto, unique
from dataclasses import dataclass, field
from datetime import datetime
from numpy import memmap as np_memmap
from pathlib import Path
from traceback import print_tb
from typing import ClassVar, Iterable, Optional, Callable, override

# Import section_types module with loading config json files
try:
    from .section_type_defs import (
        SectionType,
        MetricDataType,
    )
except ImportError as e:
    print(f"Failed to import section_types module: {e}", file=sys.stderr)
    ## Notify error to the caller as a RuntimeError
    raise RuntimeError from e

# Begin
SUPPORTED_FORMAT_VERSION_UPPER = [1, 0, 255, 255]
SUPPORTED_FORMAT_VERSION_LOWER = [1, 0, 0, 0]

FILE_HEADER_DTYPE = np.dtype(
    [
        ("magic", "u1", 4),
        ("size", "u4"),
        ("version", "u1", 4),
        ("nsections", "u4"),
        ("timestamp", "u8"),
        ("reserved", "u8"),
    ]
)

SECTION_HEADER_DTYPE = np.dtype(
    [
        ("typeid", "u4"),
        ("size", "u4"),
        ("seqnum", "u4"),
        ("ndata", "u4"),
        ("timestamp", "u8"),
        ("interval", "u4"),
        ("reserved", "u4"),
    ]
)

TABLE_ENTRY_DTYPE = np.dtype([("id", "u4"), ("size", "u4")])

MAGIC_BYTES = "AiuM"
MAGIC_UPDATING = "!"
FILE_HEADER_MIN_SIZE = 32

MAGIC_BUSID = "%BUSID"


@unique
class MetricFileFormat(IntEnum):
    """
    Enum to indicate the format of a metric file is NEW or OLD, if the metric file exists.

    UNKNOWN is the value when the metric file is not create yet.
    """

    UNKNOWN = 0
    OLD = 1
    NEW = 2


@dataclass(frozen=True)
class MetricFetcher:
    """
    Each instance is responsible to fetch read metric data of a given type from a metric file.
    It holds the offset from the beginning of the file, so that it can directory load the data
    without walking through the file and section headers.
    """

    data_type: MetricDataType
    index: int  # uint64 index of this data from the beginning of memmap array
    len: int = field(default=1)

    def fetch_detail(self, mmap: np_memmap) -> list[int | float]:
        return [
            self.data_type.value_type.adjust(mmap[self.index + i])
            for i in range(self.len)
        ]

    def fetch(self, mmap: np_memmap) -> int | float:
        return self.data_type.summarizer_type.summarizer(self.fetch_detail(mmap))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.data_type.name}):index={self.index},len={self.len}"


@dataclass(frozen=True)
class HostMetricFetcher(MetricFetcher):
    """
    Pseudo MetricFetcher to get host metrics and returns the values as if it is loaded from a metric file.
    """

    @override
    def fetch_detail(self, mmap: np_memmap) -> list[int | float]:
        match self.data_type.name:
            case "hostcpu":
                val = sum(psutil.cpu_percent(percpu=True))
            case "hostmem":
                val = psutil.virtual_memory().percent
            case _:
                val = 0
        return [self.data_type.value_type.adjust(val)]

    @override
    def fetch(self, mmap: np_memmap) -> int | float:
        return self.data_type.summarizer_type.summarizer(self.fetch_detail(mmap))

    @classmethod
    def fetchers(cls) -> list[MetricFetcher]:
        return [
            HostMetricFetcher(MetricDataType.find_by_name("hostcpu"), index=0, len=0),
            HostMetricFetcher(MetricDataType.find_by_name("hostmem"), index=0, len=0),
        ]


@dataclass
class SectionHeader:
    """
    Data class for the fixed-part of section header structure, i.e., section header except data table.

    Use a static method try_parse_header() to create an instance from the section header in a metric file.
    """

    section_type: SectionType
    section_size: int  # BYte size of this section
    header_size: int  # Byte size of the section header including the metric data table
    seqnum: int
    interval: int
    last_update: datetime  # Last modified time of the file
    data_table: list[tuple[int, int]]  # metric data table
    valid: bool = field(default=True)

    def __post_init__(self) -> None:
        self.nmetrics: int = len(
            self.data_table
        )  # Number of metric types in this section
        self.nwords: int = sum([x[1] for x in self.data_table], 0)

    @classmethod
    def try_parse_header(
        cls, expect_type: SectionType, mmap: np_memmap, start_offset: int
    ) -> "SectionHeader":
        """Parse file header from bytes.

        Args:
            expect_type:  Section type in the section table. This must be the same as the type in the section header.
            mmap:         Memmap to the raw 64bit words containing the file header.
            start_offset: Byte offset of the section header from the beginning of the metric file.

        Returns:
            FileHeader instance
        """
        # Assuming file header format: nsections (uint32) + padding
        # Adjust struct format based on actual binary format

        header = np.recarray(
            shape=1,
            buf=mmap,
            offset=start_offset,
            dtype=SECTION_HEADER_DTYPE,
            aligned=False,
        )[0]

        typeid = header["typeid"]
        size = header["size"]
        seqnum = header["seqnum"]
        ndata = header["ndata"]
        timestamp = datetime.fromtimestamp(header["timestamp"] / 1000.0)
        interval = header["interval"]

        dat_tbl = np.recarray(
            shape=ndata,
            buf=mmap,
            offset=start_offset + header.nbytes,
            dtype=TABLE_ENTRY_DTYPE,
            aligned=False,
        )
        header_size = header.nbytes + dat_tbl.nbytes
        data_table = dat_tbl.tolist()

        section_type = SectionType.find_by_id(typeid)

        return cls(
            section_type=section_type,
            section_size=size,
            header_size=header_size,
            seqnum=seqnum,
            interval=interval,
            last_update=timestamp,
            data_table=data_table,
            valid=(section_type == expect_type),
        )


class MetricSection:
    """Represents a section in the metric file with an iterator for data words."""

    def __init__(
        self,
        section_type: SectionType,
        mmap: np_memmap,
        section_offset: int,
        section_header: Optional[SectionHeader] = None,
        fetchers: Optional[list[MetricFetcher]] = None,
    ):
        """Initialize a metric section.

        Args:
            header: The section header
            mmap_array: Memory-mapped numpy array of the entire file
            data_offset: Offset in bytes where section data starts
        """
        self._sec_type: SectionType = section_type  # Type of the section
        self._mmap: np_memmap = mmap
        self._section_offset: int = section_offset
        # Calculate word offset (divide byte offset by 8 for uint64)
        self._header: SectionHeader = (
            section_header
            if section_header
            else SectionHeader.try_parse_header(
                expect_type=section_type, mmap=mmap, start_offset=section_offset
            )
        )
        self._data_start_index = (section_offset + self._header.header_size) // 8
        self._metric_fetchers: list[MetricFetcher] = (
            fetchers if fetchers else self._build_fetchers()
        )

    def _build_fetchers(self) -> list[MetricFetcher]:
        """
        Register a list of MetricFetcher objects to read data from this section.
        Unneeded data types will be skipped if filter is set.

        Default is all available data types in this section.
        """
        f_list: list[MetricFetcher] = []
        index = self._data_start_index
        for id, len in self._header.data_table:
            dtype = MetricDataType.find_by_id(id)
            if dtype.id == 0:
                index += len
                continue
            f_list.append(MetricFetcher(data_type=dtype, index=index, len=len))
            index += len
        return f_list

    @classmethod
    def _create_host_metric_section(cls, mmap: np_memmap) -> "MetricSection":
        """
        Create a pseudo section for host metrics when the metrics are not stored in metric files
        and expected to be generated by this Python API.
        """
        fetchers: list[MetricFetcher] = HostMetricFetcher.fetchers()
        sec_type = fetchers[0].data_type.section_type
        header = SectionHeader(
            section_type=sec_type,
            section_size=0,
            header_size=0,
            seqnum=1,
            interval=100,
            last_update=datetime.now(),
            data_table=[(0, 0)],
        )
        return MetricSection(
            section_type=sec_type,
            mmap=mmap,
            section_offset=0,
            section_header=header,
            fetchers=fetchers,
        )

    @property
    def type(self) -> SectionType:
        """Get the section type object."""
        return self._sec_type

    @property
    def type_id(self) -> int:
        """Get the section type ID as integer."""
        return self._sec_type.id

    @property
    def name(self) -> str:
        """Get the section type name."""
        return self._sec_type.name

    @property
    def len(self) -> int:
        """Get the number of metric data types in this section."""
        return len(self._metric_fetchers)

    def keys(self) -> list[MetricDataType]:
        """
        Enlist MetricDataType objects that will be read from this section.

        Yields:
            MetricDataType supported in this section. Unneeded data types will be skipped
            if filter is set.
        """
        return [x.data_type for x in self._metric_fetchers]

    def items(self) -> Iterable[tuple[MetricDataType, int | float]]:
        """
        Read metric data in this section. Unneeded data will be skipped if filter is set.

        Yields:
            A pair of MetricDataType and the data read from the metric file.
        """
        for f in self._metric_fetchers:
            yield f.data_type, f.fetch(self._mmap)

    def __getitem__(self, key: MetricDataType) -> int | float | None:
        """
        Access metric data by metric data type.

        Args:
            key: MetricDataType to get the data for

        Returns:
            Metric data of the specified data type
        """
        for x in self._metric_fetchers:
            if x.data_type == key:
                return x.fetch(self._mmap)
        return None

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._sec_type.name}): valid={self._header.valid}, size={self._header.section_size}, "
            f"header-size={self._header.header_size}, offset={self._section_offset}, data_start_index={self._data_start_index}\n"
            f"    ndata={len(self._header.data_table)}, data_table={self._header.data_table}\n"
            f"    defined-fetchers={', '.join((str(f) for f in self._metric_fetchers))}"
        )


@dataclass
class FileHeader:
    """
    Data class for the fixed-part of file header structure, i.e., section header except section table.

    Use a static method try_parse_header() to create an instance from the contents of a metric file.
    """

    file_format: MetricFileFormat  # True if the metric file is the new format
    version: list[
        int
    ]  # version number list [ 'major', 'minor', 'rev', 'fix' ] (uint8 each)
    header_size: int  # Byte size of the file header including the section table
    file_size: int  # File size
    last_update: datetime  # Last modified time of the file
    section_table: list[tuple[int, int]]  # section table
    null_file_header: ClassVar["FileHeader"]
    skip_old_header: ClassVar["FileHeader"]

    @property
    def is_old_format(self) -> bool:
        """True if the file format is OLD. Note that False is returned if format is UNKNOWN."""
        return self.file_format == MetricFileFormat.OLD

    @classmethod
    def try_parse_header(
        cls, filepath: Path, mmap: np_memmap, new_format_only: bool = False
    ) -> "FileHeader":
        """Parse file header from bytes.

        Args:
            mmap: Raw bytes containing the file header

        Returns:
            FileHeader instance
        """
        # Assuming file header format: nsections (uint32) + padding
        # Adjust struct format based on actual binary format

        header = np.recarray(shape=1, buf=mmap, dtype=FILE_HEADER_DTYPE, aligned=False)[
            0
        ]
        st = filepath.stat()
        magic = header["magic"]
        if not any(magic):  ## check if all bytes are 0
            # old format
            if new_format_only:
                return FileHeader.skip_old_header

            st = filepath.stat()
            return cls(
                file_format=MetricFileFormat.OLD,
                version=[0, 2, 1, 1],
                file_size=st.st_size,
                header_size=0,
                last_update=datetime.fromtimestamp(st.st_mtime),
                section_table=[(0, 0)],
            )

        if any([magic[i] != ord(ch) for i, ch in enumerate(MAGIC_BYTES)]):
            # If magic bytes is not "AiuM", check if it is still updating header by comparing the magic byte with "!"
            if any([magic[i] != ord(ch) for i, ch in enumerate(MAGIC_UPDATING)]):
                print(f"ERROR: Unexpected magic bytes: {magic}", file=sys.stderr)
            return FileHeader.null_file_header

        version = [
            int(x) for x in header["version"]
        ]  # convert from list[np.uint8] to list[int] for comparison
        if (
            version > SUPPORTED_FORMAT_VERSION_UPPER
            or version < SUPPORTED_FORMAT_VERSION_LOWER
        ):
            print(
                f"ERROR: Unsupported metric file version: {'.'.join([str(x) for x in version])}",
                file=sys.stderr,
            )
            return FileHeader.null_file_header

        file_size = header["size"]
        nsections = header["nsections"]
        timestamp = datetime.fromtimestamp(header["timestamp"] / 1000.0)

        sec_tbl = np.recarray(
            shape=nsections,
            buf=mmap,
            offset=header.nbytes,
            dtype=TABLE_ENTRY_DTYPE,
            aligned=False,
        )
        header_size = header.nbytes + sec_tbl.nbytes
        section_table = sec_tbl.tolist()

        return cls(
            file_format=MetricFileFormat.NEW,
            version=version,
            file_size=file_size,
            header_size=header_size,
            last_update=timestamp,
            section_table=section_table,
        )

    @classmethod
    def init_null_header(cls):
        """Initializer of static variables, as Python data classes cannot define static initializers."""
        cls.null_file_header = FileHeader(
            file_format=MetricFileFormat.UNKNOWN,
            version=[0, 0, 0, 0],
            file_size=0,
            header_size=0,
            last_update=datetime.fromtimestamp(0),
            section_table=[(0, 0)],
        )
        cls.skip_old_header = FileHeader(
            file_format=MetricFileFormat.OLD,
            version=[0, 0, 0, 0],
            file_size=0,
            header_size=0,
            last_update=datetime.fromtimestamp(0),
            section_table=[(0, 0)],
        )


# Initialize class variables null_file_header an skip_old_header here
FileHeader.init_null_header()


## Avoid  duplicated import when libaiumonitor.so is missing
failedImportForOldFormat: bool = False


class MetricFile:
    """
    Class to represent a metric file, and provides properties and iterators to access metric data.
    This is the main class of spyremetrics API.

    Iterators:
    - read_metrics: Read all available metric data, or the set of data filtered by set_filters().
    - metric_types: Enlist MetricDataType objects that will be read from a metric file.
    - sections:     Iterate through sections in this metric file.

    Properties:
    - ready:              True if the file is ready to read (or ready to skip if new_format_only is set and the file format is old)
    - metric_file_format: NEW or OLD if the metric file exists, or UNKNOWN if not created yet.
    - is_old_format:      True if the file format is OLD. Note that False is returned if format is UNKNOWN.
    - path:               pathlib.Path object of the metric file.
    - nsections:          Number of sections in the metric file. Note that generated host_metric section is not included.
    - version:            Version numbers as a 4-entry list of uint8_t: [ major, minor, mod, fix ].
    """

    @unique
    class State(IntEnum):
        """
        Enum to represent the state of the metric file.

        Note: Stale is unused as a file state at this moment, but it is used as the border if the metric file is OK to read
        """

        Unknown = 0
        NoFile = auto()  #  No metric file exists
        Error = auto()  #   Incorrect metric file or runtime environment
        #                    (e.g., no libaiumonitor.so)
        SkipOld = auto()  # Metric file exists and it is the old format,
        #                    but do not use syremetrics API
        Stale = auto()  #   Metric file exists, but it is stale
        Ready = auto()  #   Metric file exists and ready to read metric data

    ## Static variable for empty mmap region
    _empty_mmap: np_memmap = np.memmap("/dev/zero", dtype=np.uint64, mode="r", shape=1)

    def __init__(
        self,
        filepath: Path | str,
        filters: Optional[list[str | int | MetricDataType]] = None,
        *,
        local_host_metrics: bool = False,
        new_format_only: bool = False,
        state: "Optional[MetricFile.State]" = None,
    ):
        """Initialize a metric file reader.

        Args:
            filepath:        Path to the metric file
            filters:         List of metric data type to filter out other metrics. The list element can be metric name
                             in string, metric type ID, or MetricDataType object.
        Keyword-only Args:
            local_host_metrics: True if client allows this API to insert host_metrics pseudo section if it is not in
                                the metric file. The data of host_metrics pseudo section are measured in by this API,
                                not in the workload process.
            new_format_only:    True if MetricFile does not handle a metric file in the old format
            state:              Force to set the metric file state (Useful for seamless error handling)
        """
        self._filepath: Path = Path(filepath)
        self._file_state: MetricFile.State = (
            state if state else MetricFile.State.Unknown
        )
        self._local_host_metrics: bool = local_host_metrics

        if self._file_state == MetricFile.State.Error:
            self._mmap: np_memmap = MetricFile._empty_mmap
        else:
            # Check if valid metric file exists, and mmap if possible
            self._mmap: np_memmap = self._try_memmap_metric_file()

        if self._file_state <= MetricFile.State.Stale:  # No metric file or a stale file
            self._file_header: FileHeader = FileHeader.null_file_header
        else:
            self._file_header: FileHeader = FileHeader.try_parse_header(
                self._filepath, self._mmap, new_format_only
            )
            if self._file_header == FileHeader.skip_old_header:
                self._file_state = MetricFile.State.SkipOld
            elif self._file_header == FileHeader.null_file_header:
                self.close()
                self._file_state = MetricFile.State.Stale

        self._filters: list[MetricDataType] = MetricDataType.normalize(filters)
        self._sections: list[MetricSection] = self._load_sections()
        self._metric_fetchers: list[MetricFetcher] = self._collect_fetchers()
        self._start_time: datetime = datetime.now()

    def _try_memmap_metric_file(self) -> np_memmap:
        """
        Check if a metric file exists, and memmap if it exists.

        Set self._file_state to NoFile or Ready accordingly.
        """
        if not self._filepath.exists():
            self._file_state = MetricFile.State.NoFile
            return MetricFile._empty_mmap

        file_size = self._filepath.stat().st_size
        if file_size % 8 != 0:
            print(
                f"ERROR: Metric file size is not a multiple of sizeof(uint64_t): {self._filepath}",
                file=sys.stderr,
            )
            self._file_state = MetricFile.State.Error
            return MetricFile._empty_mmap

        if file_size < FILE_HEADER_MIN_SIZE:
            print(
                f"ERROR: Metric file size is shorter than minimum required size={FILE_HEADER_MIN_SIZE}: "
                f"{self._filepath}",
                file=sys.stderr,
            )
            self._file_state = MetricFile.State.Error
            return MetricFile._empty_mmap

        self._file_state = MetricFile.State.Ready
        return np.memmap(
            self._filepath, dtype=np.uint64, mode="r", shape=file_size // 8
        )

    def _ensure_metric_file(self, force_reopen: bool = False) -> bool:
        """
        If the metric file is not memmapped, check if it exists and memmap it if exists.

        If force_reopen is True, memmap again even if the file is already memmapped.

        Returns True if memmap is ready, otherwise False
        """
        if self._file_state > MetricFile.State.Stale and not force_reopen:
            return True

        self._mmap = self._try_memmap_metric_file()  # try to mmap

        if self._file_state <= MetricFile.State.Stale:  # No metric file or a stale file
            return False  # memmap failed

        self._file_header = FileHeader.try_parse_header(self._filepath, self._mmap)
        if self._file_header == FileHeader.null_file_header:
            self._file_state = MetricFile.State.Stale
            self.close()
            return False  # File is being updated

        self._sections = self._load_sections()
        self._metric_fetchers = self._collect_fetchers()
        return True

    def _load_sections(self) -> list[MetricSection]:
        """
        Iterate through the section table and load section headers.

        Returns an empty list if the metric file is not ready.
        """
        sections: list[MetricSection] = []

        if self._file_state <= MetricFile.State.Stale:
            return sections

        id_set: set[int] = set()
        off = self._file_header.header_size
        for id, size in self._file_header.section_table:
            sec_type = SectionType.find_by_id(id)
            if sec_type.id == 0:
                # print(f"Skip invalid section ID: {id}", file=sys.stderr)
                pass
            elif sec_type.id in id_set:
                # print(f"Skip duplicated section ID: {id}={sec_type.id}", file=sys.stderr)
                pass
            else:
                section = MetricSection(
                    section_type=sec_type, mmap=self._mmap, section_offset=off
                )
                if section._header.valid:
                    sections.append(section)
                    id_set.add(sec_type.id)
                else:
                    print(
                        f"Warning: Skip section marked as invalid: {id}",
                        file=sys.stderr,
                    )
            off += size

        if self._local_host_metrics and not any(
            [s.name == "host_metrics" for s in sections]
        ):
            sections.insert(
                0, MetricSection._create_host_metric_section(mmap=self._mmap)
            )

        return sections

    def _collect_fetchers(self) -> list[MetricFetcher]:
        """
        Build the list of MetricFetchers, with applying filter list.
        """
        fetchers: list[MetricFetcher] = []
        for s in self._sections:
            for f in s._metric_fetchers:
                if not self._filters or f.data_type in self._filters:
                    fetchers.append(f)
        return fetchers

    def set_filters(self, filters: list[str | int | MetricDataType]) -> None:
        """
        Register a list of metric data types that are interested in the program using this API.
        This can be an optimization to filter out unneeded metric data types.

        Default is all available data types in the metric file.
        """
        self._filters = MetricDataType.normalize(filters)
        self._metric_fetchers = self._collect_fetchers()

    @property
    def ready(self) -> bool:
        """True if the file is ready to read (or ready to skip if new_format_only is set and the file format is old)"""
        self._ensure_metric_file()
        return self._file_state in (MetricFile.State.Ready, MetricFile.State.SkipOld)

    @property
    def metric_file_format(self) -> MetricFileFormat:
        """NEW or OLD if the metric file exists, or UNKNOWN if not created yet."""
        self._ensure_metric_file()
        return self._file_header.file_format

    @property
    def is_old_format(self) -> bool:
        """True if the file format is OLD. Note that False is returned if format is UNKNOWN."""
        self._ensure_metric_file()
        return self._file_header.is_old_format

    @property
    def path(self) -> Path:
        """pathlib.Path object of the metric file."""
        return self._filepath

    @property
    def nsections(self) -> int:
        """Number of sections in the metric file. Note that generated host_metric section is not included."""
        self._ensure_metric_file()
        return len(self._sections)

    @property
    def version(self) -> list[int]:
        """Version numbers as a 4-entry list of uint8_t: [ major, minor, mod, fix ]."""
        self._ensure_metric_file()
        return self._file_header.version

    def metric_types(self) -> Iterable[MetricDataType]:
        """
        Enlist MetricDataType objects that will be read from a metric file.

        Yields:
            MetricDataType supported by the current metric file.
        """
        self._ensure_metric_file()
        yield from [f.data_type for f in self._metric_fetchers]

    def read_metrics(self) -> Iterable[tuple[MetricDataType, int | float]]:
        """
        Read all available metric data, or the set of data filtered by set_filters().

        Yields:
            A pair of MetricDataType and the data read from the metric file.
        """
        self._ensure_metric_file()

        if self._file_state <= MetricFile.State.Error:
            print(f"ERROR: Invalid metric file: {self._filepath}", file=sys.stderr)
            return  # Finish generator with not data

        if self._file_state <= MetricFile.State.Stale:
            return  # Finish generator with not data

        if self._file_header.is_old_format:
            # Check if importing old_metric_file has failed
            global failedImportForOldFormat
            if failedImportForOldFormat:
                return  # Finish generator with not data

            # Old format: Use libaiumonitor.so to read it
            try:
                from .old_metric_file import read_old_metrics

                yield from read_old_metrics(self._filepath)
            except RuntimeError:  # as e:
                failedImportForOldFormat = True
                # print(f"ERROR: Failed to import read_old_metrics(): {str(e)}", file=sys.stderr)
                self._file_state = MetricFile.State.Error
                return

        for f in self._metric_fetchers:
            yield (f.data_type, f.fetch(self._mmap))

    def sections(self) -> Iterable[MetricSection]:
        """
        Iterate through sections in this metric file.

        Yields:
            MetricSection for the sections in the file.
        """
        self._ensure_metric_file()
        if self._file_state <= MetricFile.State.Stale:
            return
        yield from self._sections

    @classmethod
    def open_metrics(
        cls,
        metric_file_str: str,
        filters: Optional[list[str | int | MetricDataType]] = None,
        *,
        local_host_metrics: bool = False,
        new_format_only: bool = False,
    ) -> list["MetricFile"]:
        """A factory method to create MetricFile, with expanding "%BUSID" magic keyword. Since "%BUSID" can be expanded to
        multiple device IDs, this method returns a list of MetricFile objects.

        Args:
            filepath:        Path string to the metric file. Accept "%BUSID" keyword.
            filters:         List of metric data type to filter out other metrics. The list element can be metric name in string,
                             metric type ID, or MetricDataType object.
        Keyword-only Args:
            local_host_metrics: True if client allows this API to insert host_metrics pseudo section if it is not in the metric file
                                The data of host_metrics pseudo section are measured in by this API, not in the workload process.
            new_format_only:    True if MetricFile does not handle a metric file in the old format
        """
        files = []
        metric_path = Path(metric_file_str)

        if MAGIC_BUSID not in metric_path.name:
            files.append(metric_path)
        else:
            busids: list[str] = []
            if envStr := os.environ.get("PCIDEVICE_IBM_COM_AIU_PF"):
                busids = envStr.split(",")
            elif envStr := os.environ.get("AIU_WORLD_SIZE"):
                try:
                    world_size = int(envStr)
                except ValueError:
                    print(
                        "ERROR: AIU_WORLD_SIZE is set to non-integer value",
                        file=sys.stderr,
                    )
                    return []

                for rank in range(world_size):
                    envStr = os.environ.get(f"AIU_WORLD_RANK_{rank}")
                    if envStr:
                        busids.append(envStr)
                    else:
                        busids.append(f"UNKNOWN_BUSID_RANK_{rank}")
            else:
                print(
                    "ERROR: AIU_WORLD_SIZE is not set and cannot get world size",
                    file=sys.stderr,
                )
                return []

            if busids:
                for bid in busids:
                    fname = metric_path.name.replace(MAGIC_BUSID, bid, 1)
                    files.append(metric_path.parent / fname)
            else:
                files.append(metric_path)

        return [
            cls(
                p,
                filters=filters,
                local_host_metrics=local_host_metrics,
                new_format_only=new_format_only,
            )
            for p in files
        ]

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}: file={self._filepath}, old-format={self._file_header.is_old_format}, "
            f"skip-old-format={self._file_header.skip_old_header}, "
            f"version={'.'.join((str(v) for v in self._file_header.version))}, state={self._file_state.name}, "
            f"nsections={self.nsections}\n"
            f"    header-size={self._file_header.header_size}, section-table={self._file_header.section_table}\n"
            f"    filters={self._filters}\n"
            f"    mmap={self._mmap.dtype} x {self._mmap.size} = {self._mmap.nbytes} bytes, {self._mmap.filename}\n"
            f"    active-fetchers={', '.join((str(f) for f in self._metric_fetchers))}\n"
            f"    sections:\n{'\n'.join((str(s) for s in self._sections))}"
        )

    def close(self) -> None:
        """Close the memory-mapped file."""
        if self._mmap.size > 1:
            self._mmap = MetricFile._empty_mmap  # Existing mapping will be GCed

    def __enter__(self) -> "MetricFile":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Destructor to ensure memmap is cleaned up."""
        self.close()
