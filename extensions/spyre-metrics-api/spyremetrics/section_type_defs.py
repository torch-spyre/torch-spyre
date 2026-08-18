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
Data classes to define attributes of elements in Spyre metric files.
"""

import sys

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable, Callable, ClassVar, Type, cast


# Classes for types in a metric file
@dataclass(frozen=True)
class SectionType:
    JSON_KEY: ClassVar = "section_type"
    MAP_NAME: ClassVar = "_section_type_map"
    name: str
    long_name: str
    id: int
    id_map: ClassVar[dict[int, "SectionType"]] = {}
    name_map: ClassVar[dict[str, "SectionType"]] = {}

    def __post_init__(self) -> None:
        if isinstance(self.id, str):
            object.__setattr__(
                self, "id", int(cast(str, self.id), 0)
            )  # convert 'id' from str to int

    @classmethod
    def find_by_id(cls, id: int) -> "SectionType":
        return _lookup_by_id(cls.id_map, id)

    @classmethod
    def find_by_name(cls, name: str) -> "SectionType":
        return _lookup_by_name(cls.name_map, name)

    @classmethod
    def add_id_map(cls, map: dict[int, "SectionType"]) -> None:
        _add_mapping_by_id(cls.id_map, map)

    @classmethod
    def add_name_map(cls, map: dict[str, "SectionType"]) -> None:
        _add_mapping_by_name(cls.name_map, map)

    @classmethod
    def items(cls) -> Iterable["SectionType"]:
        yield from cls.id_map.values()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.name}: id={self.id}, desc={self.long_name}"


@dataclass(frozen=True)
class ValueType:
    JSON_KEY: ClassVar = "value_type"
    MAP_NAME: ClassVar = "_value_type_map"
    name: str
    long_name: str
    id: int
    python_type: Type[int] | Type[float] = int
    scale: int = 1
    id_map: ClassVar[dict[int, "ValueType"]] = {}
    name_map: ClassVar[dict[str, "ValueType"]] = {}

    def __post_init__(self) -> None:
        if isinstance(self.id, str):
            object.__setattr__(
                self, "id", int(cast(str, self.id), 0)
            )  # convert 'id' from str to int
        match self.name:
            case "invalid":
                object.__setattr__(self, "python_type", int)
                object.__setattr__(self, "scale", 1)
            case "int":
                object.__setattr__(self, "python_type", int)
                object.__setattr__(self, "scale", 1)
            case "KiB":
                object.__setattr__(self, "python_type", int)
                object.__setattr__(self, "scale", 1024)
            case "MiB":
                object.__setattr__(self, "python_type", int)
                object.__setattr__(self, "scale", 1024 * 1024)
            case "GiB":
                object.__setattr__(self, "python_type", int)
                object.__setattr__(self, "scale", 1024 * 1024 * 1024)
            case "float":
                object.__setattr__(self, "python_type", float)
                object.__setattr__(self, "scale", 1.0)
            case "fdec1":
                object.__setattr__(self, "python_type", float)
                object.__setattr__(self, "scale", 0.1)
            case _:
                raise ValueError(f"Unexpected value_type name: {self.name}")

    def adjust(self, data) -> int | float:
        return data * self.scale

    @classmethod
    def find_by_id(cls, id: int) -> "ValueType":
        return _lookup_by_id(cls.id_map, id)

    @classmethod
    def find_by_name(cls, name: str) -> "ValueType":
        return _lookup_by_name(cls.name_map, name)

    @classmethod
    def add_id_map(cls, map: dict[int, "ValueType"]) -> None:
        _add_mapping_by_id(cls.id_map, map)

    @classmethod
    def add_name_map(cls, map: dict[str, "ValueType"]) -> None:
        _add_mapping_by_name(cls.name_map, map)

    @classmethod
    def items(cls) -> Iterable["ValueType"]:
        yield from cls.id_map.values()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}: {self.name}: scale={self.scale}, "
            f"type={self.python_type}, id={self.id}, desc={self.long_name}"
        )


@dataclass(frozen=True)
class SummarizerType:
    JSON_KEY: ClassVar = "summarizer_type"
    MAP_NAME: ClassVar = "_summarizer_map"
    name: str
    long_name: str
    id: int
    summarizer: Callable[[list[int | float] | tuple[int | float, ...]], int | float] = (
        lambda x: 0
    )
    id_map: ClassVar[dict[int, "SummarizerType"]] = {}
    name_map: ClassVar[dict[str, "SummarizerType"]] = {}

    def __post_init__(self) -> None:
        if isinstance(self.id, str):
            object.__setattr__(
                self, "id", int(cast(str, self.id), 0)
            )  # convert 'id' from str to int
        match self.name:
            case "invalid":
                object.__setattr__(self, "summarizer", lambda x: 0)
            case "first":
                object.__setattr__(self, "summarizer", lambda x: x[0])
            case "last":
                object.__setattr__(self, "summarizer", lambda x: x[-1])
            case "sum":
                object.__setattr__(self, "summarizer", lambda x: sum(x))
            case "avg":
                object.__setattr__(self, "summarizer", lambda x: mean(x))
            case "max":
                object.__setattr__(self, "summarizer", lambda x: max(x))
            case _:
                raise ValueError(f"Unexpected summarizer_type name: {self.name}")

    @classmethod
    def find_by_id(cls, id: int) -> "SummarizerType":
        return _lookup_by_id(cls.id_map, id)

    @classmethod
    def find_by_name(cls, name: str) -> "SummarizerType":
        return _lookup_by_name(cls.name_map, name)

    @classmethod
    def add_id_map(cls, map: dict[int, "SummarizerType"]) -> None:
        _add_mapping_by_id(cls.id_map, map)

    @classmethod
    def add_name_map(cls, map: dict[str, "SummarizerType"]) -> None:
        _add_mapping_by_name(cls.name_map, map)

    @classmethod
    def items(cls) -> Iterable["SummarizerType"]:
        yield from cls.id_map.values()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}: {self.name}: id={self.id}, "
            f"summarizer={repr(self.summarizer)}, desc={self.long_name}"
        )


@dataclass(frozen=True)
class MetricDataType:
    JSON_KEY: ClassVar = "metric_type"
    MAP_NAME: ClassVar = "_metric_type_map"
    name: str
    long_name: str
    id: int
    summarizer: str
    type: str
    section: str
    section_type: SectionType = field(
        default_factory=lambda: SectionType(
            name="invalid", id=0, long_name="Invalid section type"
        )
    )
    value_type: ValueType = field(
        default_factory=lambda: ValueType(
            name="invalid", id=0, long_name="Invalid value type"
        )
    )
    summarizer_type: SummarizerType = field(
        default_factory=lambda: SummarizerType(
            name="invalid", id=0, long_name="Invalid summarizer"
        )
    )
    id_map: ClassVar[dict[int, "MetricDataType"]] = {}
    name_map: ClassVar[dict[str, "MetricDataType"]] = {}

    def __post_init__(self) -> None:
        if isinstance(self.id, str):
            object.__setattr__(
                self, "id", int(cast(str, self.id), 0)
            )  # convert 'id' from str to int
        object.__setattr__(self, "section_type", SectionType.find_by_name(self.section))
        object.__setattr__(self, "value_type", ValueType.find_by_name(self.type))
        object.__setattr__(
            self, "summarizer_type", SummarizerType.find_by_name(self.summarizer)
        )

    def summary(self, t: tuple[int | float, ...]) -> int | float:
        if len(t) == 1:
            return t[0]
        elif len(t) == 0:
            if self.value_type.python_type is int:
                return 0
            elif self.value_type.python_type is float:
                return 0.0
            else:
                print(
                    f"ERROR: Unexpected list element type: {self.value_type.python_type}",
                    file=sys.stderr,
                )
                raise ValueError(self.value_type.python_type)
        else:
            return self.summarizer_type.summarizer(t)

    def add(self, v1: int | float, v2: int | float) -> int | float:
        return self.summarizer_type.summarizer([v1, v2])

    @classmethod
    def find_by_id(cls, id: int) -> "MetricDataType":
        return _lookup_by_id(cls.id_map, id)

    @classmethod
    def find_by_name(cls, name: str) -> "MetricDataType":
        return _lookup_by_name(cls.name_map, name)

    @classmethod
    def add_id_map(cls, map: dict[int, "MetricDataType"]) -> None:
        _add_mapping_by_id(cls.id_map, map)

    @classmethod
    def add_name_map(cls, map: dict[str, "MetricDataType"]) -> None:
        _add_mapping_by_name(cls.name_map, map)

    @classmethod
    def normalize(
        cls, type_list: list["str|int|MetricDataType"] | None
    ) -> list["MetricDataType"]:
        if not type_list:
            return cast(list[MetricDataType], [])

        s: set[MetricDataType] = set()
        for x in type_list:
            match x:
                case str():
                    m = MetricDataType.find_by_name(x)
                    if m.id != 0:
                        s.add(m)
                    else:
                        print(f"Invalid MetricDataType: {x}", file=sys.stderr)
                case int():
                    m = MetricDataType.find_by_id(x)
                    if m.id != 0:
                        s.add(m)
                    else:
                        print(f"Invalid MetricDataType: {x}", file=sys.stderr)
                case MetricDataType():
                    if x.id != 0:
                        s.add(x)
                    else:
                        print(f"Invalid MetricDataType: {x}", file=sys.stderr)
                case _:
                    print(f"Invalid MetricDataType: {x}", file=sys.stderr)
        return list(s)

    @classmethod
    def items(cls) -> Iterable["MetricDataType"]:
        yield from cls.id_map.values()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}: {self.name}: id={self.id}, desc={self.long_name}\n"
            + f"   section={self.section_type}\n"
            + f"   type={self.value_type}\n"
            + f"   summarizer={self.summarizer_type}"
        )


### Helper functions to add or look up the global config map
# Note: map[0] is "invalid" entry
def _lookup_by_id[T: SectionType | ValueType | SummarizerType | MetricDataType](
    map: dict[int, T], _id: int
) -> T:
    if not map:
        raise ValueError("Type singleton map is not initialized yet")
    return map.get(_id, map[0])


def _lookup_by_name[T: SectionType | ValueType | SummarizerType | MetricDataType](
    map: dict[str, T], name: str
) -> T:
    if not map:
        raise ValueError("Type singleton map is not initialized yet")
    return map.get(name, map["invalid"])


def _add_mapping_by_id[T: SectionType | ValueType | SummarizerType | MetricDataType](
    dst: dict[int, T], src: dict[int, T]
) -> None:
    dst |= src


def _add_mapping_by_name[T: SectionType | ValueType | SummarizerType | MetricDataType](
    dst: dict[str, T], src: dict[str, T]
) -> None:
    dst |= src
