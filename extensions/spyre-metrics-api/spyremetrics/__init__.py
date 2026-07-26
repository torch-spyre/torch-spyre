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

from .metric_file import (
    FileHeader,
    MetricFile,
    MetricSection,
    SectionHeader,
)

from .section_type_defs import (
    MetricDataType,
    SectionType,
    SummarizerType,
    ValueType,
)

from .section_types import (
    CONFIG_FILE_LIST,
    config_version,
)

## TODO: Should refer to the package version
__version__ = "0.5.0"  ## Version for release
#__version__ = "0.5.8"  ## Version for uploading to PyPI test server
__all__ = [
    "CONFIG_FILE_LIST",
    "FileHeader",
    "MetricFile",
    "MetricSection",
    "SectionHeader",
    "MetricDataType",
    "SectionType",
    "SummarizerType",
    "ValueType",
    "config_version",
]
