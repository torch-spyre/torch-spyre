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
Load definitions of section and data types either from generated Python code or JSON file.
"""

import json
import os
import sys

from pathlib import Path
from traceback import print_tb, print_exception

from .section_type_defs import (
    MetricDataType,
    SectionType,
    SummarizerType,
    ValueType,
)

# Global parameters
"""List of pre-defined config file paths"""
CONFIG_FILE_LIST = [
    "~/.section_types.json",
    "~/.local/etc/section_types.json",
    str(Path(__file__).parent / "section_types.json"),
    "/opt/ibm/spyre/runtime/etc/section_types.json",
]

"""Control whether to ignore json load error"""
ignore_json_error = False  # For debug


"""
Config file version
"""
_config_version: tuple[int, int, int, int] = (0, 0, 0, 0)


def config_version() -> tuple[int, int, int, int]:
    return _config_version


### Config JSON loader
def _load_section_types(
    config_files: list[str], env_name: str, cls_list: list[type]
) -> None:
    """Load config json file to set up all supported data types in a metric file"""

    # Add user config file to the list
    config_env = os.environ.get(env_name)
    if config_env and Path(config_env).exists():
        config_files.append(config_env)

    _raw_config_map = {}  # Temporary json object, but define as a global variable for debugging
    abs_path = None
    # Read config files
    for path_str in config_files:
        #        print(f"Checking {path_str}", file=sys.stderr)
        abs_path = Path(path_str).expanduser().resolve()
        if not abs_path.exists():
            abs_path = None
            continue
        with abs_path.open() as f:
            try:
                #                print(f"Load {path_str}", file=sys.stderr)
                _raw_config_map = json.load(f)
                break  # Use the first available config file
            except json.JSONDecodeError as e:
                print(
                    f"ERROR: Failed to load section type config file: {str(abs_path)}\n  MSG: {e.msg}",
                    file=sys.stderr,
                )
                if ignore_json_error:
                    print("Warning: This file is ignored, and go on", file=sys.stderr)
                    continue
                raise e

    if not _raw_config_map:  # Check if config file is loaded
        if not abs_path:
            msg = "Failed to find JSON config file."
        else:
            msg = f"Failed to load elements from JSON config: {abs_path}"
        raise ValueError(msg)

    # Extract config version
    if "version" in _raw_config_map:
        vers: list[int] = [0] * 4
        for i, elm in enumerate(_raw_config_map["version"].split(".")):
            if i >= 4:
                break
            if elm and elm.isdigit():
                vers[i] = int(elm)
            # Leave vers[i] to be 0 if the element is not a decimal string

        global _config_version
        _config_version = tuple(vers)  # len(vers) is always 4

    # Convert to data classes
    global _section_type_map, _metric_type_map, _value_type_map, _summarizer_map
    for cls in cls_list:
        if not hasattr(cls, "JSON_KEY"):
            raise ValueError(f"{str(cls)} does not have JSON_KEY attribute")

        category = cls.JSON_KEY
        entries = _raw_config_map.get(category, [])
        if not entries:
            # print(f"Warning: \"{category}\" key is empty in section type json", file=sys.stderr)  # For debug
            continue

        map = {int(x["id"], 0): cls(**x) for x in entries}

        if not hasattr(cls, "MAP_NAME"):
            raise ValueError(f"{str(cls)} does not have MAP_NAME attribute")

        match cls.MAP_NAME:
            case "_section_type_map":
                SectionType.add_id_map(map)
                SectionType.add_name_map({e.name: e for e in map.values()})

            case "_value_type_map":
                ValueType.add_id_map(map)
                ValueType.add_name_map({e.name: e for e in map.values()})

            case "_summarizer_map":
                SummarizerType.add_id_map(map)
                SummarizerType.add_name_map({e.name: e for e in map.values()})

            case "_metric_type_map":
                MetricDataType.add_id_map(map)
                MetricDataType.add_name_map({e.name: e for e in map.values()})


"""
Try to import generated map of SectionType etc. objects
"""
try:
    from .generated_section_types import (
        VERSION,
        ## Singleton objects are added in the global map in generated_section_types.py
    )

    _config_version = VERSION
    # print("Loaded generated_section_types.py", file=sys.stderr)  # For debug
except (ImportError, ValueError) as e:
    ## For debug
    print(
        "Debug: Failed to import pre-converted config file. Fall back to load from JSON file",
        file=sys.stderr,
    )

    if isinstance(e, ValueError):
        print(
            "Warning: Error occurred while loading the pre-converted config file.",
            file=sys.stderr,
        )
        ## print_tb(e.__traceback__, file=sys.stderr)
        print_exception(e, file=sys.stderr)

    ## Load the config json file into a global map, so that SectionType etc. objects can be singleton
    try:
        _load_section_types(
            CONFIG_FILE_LIST,
            "SECTION_TYPE_CONFIG_FILE",
            [SectionType, ValueType, SummarizerType, MetricDataType],
        )  # MetricDataType must be the last
        # print("Loaded type definitions from JSON config", file=sys.stderr)  # For debug
    except ValueError as e:
        print(
            f"ERROR: Failed to load definitions of SectionType, etc.\n{str(e)}",
            file=sys.stderr,
        )
        raise ImportError from e
    except Exception as e:
        raise ImportError(
            "Unexpected error while loading definitions of SectionType, etc."
        ) from e

# print("Loaded by section_types.py", file=sys.stderr)  # For debug
