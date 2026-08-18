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
Test script to check if section and metric data type definitions can be loaded correctly.
"""

import sys
from traceback import print_tb, print_exception

"""
Try to import generated map of SectionType etc. objects
"""
try:
    from spyremetrics.generated_section_types import (
        MetricDataType,
        SectionType,
        SummarizerType,
        ValueType,
        VERSION,
        ## Singleton objects are added in the global map in generated_section_types.py
    )

    config_version = VERSION
    print("Loaded generated_section_types.py", file=sys.stderr)  # For debug
except ImportError as e:
    # For debug
    print(
        "Debug: Failed to import pre-converted config file. Fall back to load from JSON file",
        file=sys.stderr,
    )
    # print_tb(e.__traceback__, file=sys.stderr)
    print_exception(e, file=sys.stderr)

    """
    Load the config json file into a global map, so that SectionType etc. objects can be singleton
    """
    try:
        from spyremetrics import (
            MetricDataType,
            SectionType,
            SummarizerType,
            ValueType,
            config_version,
        )

        print("Loaded type definitions from JSON config", file=sys.stderr)  # For debug
    except Exception as e:
        print(f"ERROR: Exception is thrown: {str(e)}", file=sys.stderr)
        print_tb(e.__traceback__, file=sys.stderr)
        sys.exit(1)

print(f"Config version: {config_version}")
for n, c in {
    "Section": SectionType,
    "Metric": MetricDataType,
    "Value": ValueType,
    "Summarizers": SummarizerType,
}.items():
    print(f"=== {n} type ===")
    print(*[str(e) for e in c.items()], sep="\n")
