#!/usr/bin/env python3

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
Generate Python and C++ literal containers from a Spyre metrics JSON file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from datetime import datetime


PYTHON_OUTPUT = "generated_section_types.py"
CPP_HEADER_OUTPUT = "generated_section_types.hpp"
CPP_SOURCE_OUTPUT = "generated_section_types.cpp"
SCRIPT_NAME = "convert_section_types.py"

PYTHON_COPYRIGHT = """\
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
# limitations under the License."""

CPP_COPYRIGHT = PYTHON_COPYRIGHT.replace("#", "//")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a JSON file like util/spyremetrics/section_types.json and "
            "generate Python and C++ source files with literal container data."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for generated files. Defaults to this script directory.",
    )
    parser.add_argument(
        "--namespace",
        default="spyremetrics",
        help="C++ namespace for generated files. Defaults to 'spyremetrics'.",
    )
    parser.add_argument(
        "--target",
        choices=("python", "cpp", "both"),
        default="python",
        help="Select generated output target. Defaults to 'python'.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON value must be an object.")
    return data


def ensure_list_of_objects(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"JSON key {key!r} must contain a list.")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} in {key!r} must be an object.")
        result.append(item)
    return result


def parse_hex_id(value: Any, field_name: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"Field {field_name!r} must be a string.")
    try:
        parsed = int(value, 16)
    except ValueError as exc:
        raise ValueError(f"Field {field_name!r} must be a hexadecimal string.") from exc
    if parsed < 0 or parsed > 0xFFFFFFFF:
        raise ValueError(f"Field {field_name!r} is out of range for uint32_t.")
    return parsed


def parse_version_parts(version: Any) -> tuple[int, int, int, int] | None:
    if version is None:
        return None
    if not isinstance(version, str):
        raise ValueError("JSON key 'version' must be a string when present.")

    parts = version.split(".")
    if len(parts) > 4:
        raise ValueError(
            "JSON key 'version' must contain at most 4 dot-separated parts."
        )

    parsed_parts: list[int] = []
    for index, part in enumerate(parts):
        if not part or not part.isdigit():
            raise ValueError(f"Version part {index} must be a non-negative integer.")
        parsed_parts.append(int(part))

    while len(parsed_parts) < 4:
        parsed_parts.append(0)

    return (parsed_parts[0], parsed_parts[1], parsed_parts[2], parsed_parts[3])


def validate_schema(data: dict[str, Any]) -> dict[str, Any]:
    section_types = ensure_list_of_objects(data, "section_type")
    metric_types = ensure_list_of_objects(data, "metric_type")
    value_types = ensure_list_of_objects(data, "value_type")
    summarizer_types = ensure_list_of_objects(data, "summarizer_type")

    section_names = set()
    value_names = set()
    summarizer_names = set()

    for item in section_types:
        for field_name in ("name", "id", "long_name"):
            if field_name not in item:
                raise ValueError(f"Missing {field_name!r} in section_type item.")
        if not isinstance(item["name"], str) or not isinstance(item["long_name"], str):
            raise ValueError("section_type name and long_name must be strings.")
        parse_hex_id(item["id"], "section_type.id")
        section_names.add(item["name"])

    for item in value_types:
        for field_name in ("name", "id", "long_name"):
            if field_name not in item:
                raise ValueError(f"Missing {field_name!r} in value_type item.")
        if not isinstance(item["name"], str) or not isinstance(item["long_name"], str):
            raise ValueError("value_type name and long_name must be strings.")
        parse_hex_id(item["id"], "value_type.id")
        value_names.add(item["name"])

    for item in summarizer_types:
        for field_name in ("name", "id", "long_name"):
            if field_name not in item:
                raise ValueError(f"Missing {field_name!r} in summarizer_type item.")
        if not isinstance(item["name"], str) or not isinstance(item["long_name"], str):
            raise ValueError("summarizer_type name and long_name must be strings.")
        parse_hex_id(item["id"], "summarizer_type.id")
        summarizer_names.add(item["name"])

    for item in metric_types:
        for field_name in ("section", "name", "id", "type", "summarizer", "long_name"):
            if field_name not in item:
                raise ValueError(f"Missing {field_name!r} in metric_type item.")
        for field_name in ("section", "name", "type", "summarizer", "long_name"):
            if not isinstance(item[field_name], str):
                raise ValueError(f"metric_type.{field_name} must be a string.")
        parse_hex_id(item["id"], "metric_type.id")
        if item["section"] not in section_names:
            raise ValueError(
                f"Unknown section name in metric_type: {item['section']!r}"
            )
        if item["type"] not in value_names:
            raise ValueError(f"Unknown value type in metric_type: {item['type']!r}")
        if item["summarizer"] not in summarizer_names:
            raise ValueError(
                f"Unknown summarizer type in metric_type: {item['summarizer']!r}"
            )

    return {
        "version": parse_version_parts(data.get("version")),
        "section_type": section_types,
        "metric_type": metric_types,
        "value_type": value_types,
        "summarizer_type": summarizer_types,
    }


def py_literal(value: Any) -> str:
    return repr(value)


def cpp_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def generate_python(module_name: str, input_path: Path, schema: dict[str, Any]) -> str:
    lines: list[str] = [
        PYTHON_COPYRIGHT,
        "#",
        "# DO NOT EDIT THIS FILE. This is a generated file.",
        f"# Generated by {SCRIPT_NAME} at {datetime.now()}.",
        f"# JSON input: {input_path.name} (Last update: {datetime.fromtimestamp(input_path.stat().st_mtime)}).",
        "",
        #        "from __future__ import annotations",
        #        "",
        "from .section_type_defs import (",
        "    MetricDataType,",
        "    SectionType,",
        "    SummarizerType,",
        "    ValueType,",
        ")",
        "",
        f"SOURCE_JSON = {py_literal(module_name)}",
        "",
    ]
    version_parts = schema.get("version")
    if version_parts is not None:
        lines.extend(
            [
                f"VERSION = {py_literal(version_parts)}",
                "",
            ]
        )

    def add_sequence(
        name: str, class_name: str, items: list[dict[str, Any]], fields: list[str]
    ) -> None:
        lines.append(f"{name} = (")
        for item in items:
            parts = []
            for field_name in fields:
                value = item[field_name]
                if field_name == "id":
                    value = parse_hex_id(value, field_name)
                parts.append(f"{field_name}={py_literal(value)}")
            lines.append(f"    {class_name}({', '.join(parts)}),")
        lines.append(")")
        lines.append("")

    add_sequence(
        "SECTION_TYPES",
        "SectionType",
        schema["section_type"],
        ["name", "id", "long_name"],
    )
    lines.extend(
        [
            "SectionType.add_id_map({entry.id: entry for entry in SECTION_TYPES})",
            "SectionType.add_name_map({entry.name: entry for entry in SECTION_TYPES})",
            "",
        ]
    )
    add_sequence(
        "VALUE_TYPES", "ValueType", schema["value_type"], ["name", "id", "long_name"]
    )
    lines.extend(
        [
            "ValueType.add_id_map({entry.id: entry for entry in VALUE_TYPES})",
            "ValueType.add_name_map({entry.name: entry for entry in VALUE_TYPES})",
            "",
        ]
    )
    add_sequence(
        "SUMMARIZER_TYPES",
        "SummarizerType",
        schema["summarizer_type"],
        ["name", "id", "long_name"],
    )
    lines.extend(
        [
            "SummarizerType.add_id_map({entry.id: entry for entry in SUMMARIZER_TYPES})",
            "SummarizerType.add_name_map({entry.name: entry for entry in SUMMARIZER_TYPES})",
            "",
        ]
    )
    add_sequence(
        "METRIC_TYPES",
        "MetricDataType",
        schema["metric_type"],
        ["section", "name", "id", "type", "summarizer", "long_name"],
    )
    lines.extend(
        [
            "MetricDataType.add_id_map({entry.id: entry for entry in METRIC_TYPES})",
            "MetricDataType.add_name_map({entry.name: entry for entry in METRIC_TYPES})",
            "",
        ]
    )

    return "\n".join(lines) + "\n"


def generate_cpp_header(
    namespace: str, input_path: Path, schema: dict[str, Any]
) -> str:
    section_count = len(schema["section_type"])
    value_count = len(schema["value_type"])
    summarizer_count = len(schema["summarizer_type"])
    metric_count = len(schema["metric_type"])

    lines = [
        CPP_COPYRIGHT,
        "//",
        "// DO NOT EDIT THIS FILE. This is a generated file.",
        f"// Generated by {SCRIPT_NAME} at {datetime.now()}.",
        f"// JSON input: {input_path.name} (Last update: {datetime.fromtimestamp(input_path.stat().st_mtime)}).",
        "",
        "#pragma once",
        "",
        "#include <cstdint>",
        "#include <string_view>",
        "#include <unordered_map>",
        "#include <vector>",
        "",
        f"namespace {namespace} {{",
        "",
        "struct SectionType {",
        "    std::string_view name;",
        "    std::uint32_t id;",
        "    std::string_view long_name;",
        "};",
        "",
        "struct ValueType {",
        "    std::string_view name;",
        "    std::uint32_t id;",
        "    std::string_view long_name;",
        "};",
        "",
        "struct SummarizerType {",
        "    std::string_view name;",
        "    std::uint32_t id;",
        "    std::string_view long_name;",
        "};",
        "",
        "struct MetricDataType {",
        "    std::string_view section;",
        "    std::string_view name;",
        "    std::uint32_t id;",
        "    std::string_view type;",
        "    std::string_view summarizer;",
        "    std::string_view long_name;",
        "};",
        "",
        f"inline constexpr std::size_t SECTION_TYPE_COUNT = {section_count};",
        f"inline constexpr std::size_t VALUE_TYPE_COUNT = {value_count};",
        f"inline constexpr std::size_t SUMMARIZER_TYPE_COUNT = {summarizer_count};",
        f"inline constexpr std::size_t METRIC_TYPE_COUNT = {metric_count};",
        "",
        "extern const std::vector<SectionType> SECTION_TYPES;",
        "extern const std::vector<ValueType> VALUE_TYPES;",
        "extern const std::vector<SummarizerType> SUMMARIZER_TYPES;",
        "extern const std::vector<MetricDataType> METRIC_TYPES;",
        "",
        "extern const std::vector<int> VERSION;",
        "",
        "extern const std::unordered_map<std::string_view, const SectionType*> SECTION_TYPES_BY_NAME;",
        "extern const std::unordered_map<std::uint32_t, const SectionType*> SECTION_TYPES_BY_ID;",
        "extern const std::unordered_map<std::string_view, const ValueType*> VALUE_TYPES_BY_NAME;",
        "extern const std::unordered_map<std::uint32_t, const ValueType*> VALUE_TYPES_BY_ID;",
        "extern const std::unordered_map<std::string_view, const SummarizerType*> SUMMARIZER_TYPES_BY_NAME;",
        "extern const std::unordered_map<std::uint32_t, const SummarizerType*> SUMMARIZER_TYPES_BY_ID;",
        "extern const std::unordered_map<std::string_view, const MetricDataType*> METRIC_TYPES_BY_NAME;",
        "extern const std::unordered_map<std::uint32_t, const MetricDataType*> METRIC_TYPES_BY_ID;",
        "",
        "}  // namespace " + namespace,
        "",
    ]
    return "\n".join(lines) + "\n"


def generate_cpp_source(
    namespace: str, input_path: Path, schema: dict[str, Any]
) -> str:
    lines = [
        CPP_COPYRIGHT,
        "//",
        "// DO NOT EDIT THIS FILE. This is a generated file.",
        f"// Generated by {SCRIPT_NAME} at {datetime.now()}.",
        f"// JSON input: {input_path.name} (Last update: {datetime.fromtimestamp(input_path.stat().st_mtime)}).",
        "",
        '#include "generated_section_types.hpp"',
        "",
        "#include <utility>",
        "",
        f"namespace {namespace} {{",
        "",
        "constexpr std::uint32_t u32(unsigned long long value) noexcept {",
        "    return static_cast<std::uint32_t>(value);",
        "}",
        "",
    ]
    version_parts = schema.get("version")
    if version_parts is not None:
        lines.extend(
            [
                f"const std::vector<int> VERSION {{{version_parts[0]}, {version_parts[1]}, {version_parts[2]}, {version_parts[3]}}};",
                "",
            ]
        )

    def add_array(
        name: str, struct_name: str, items: list[dict[str, Any]], fields: list[str]
    ) -> None:
        lines.append(f"const std::vector<{struct_name}> {name} {{")
        for index, item in enumerate(items):
            values: list[str] = []
            for field_name in fields:
                value = item[field_name]
                if field_name == "id":
                    values.append(f"u32({parse_hex_id(value, field_name)})")
                else:
                    values.append(cpp_string(str(value)))
            suffix = "," if index + 1 < len(items) else ""
            lines.append(f"    {struct_name}{{{', '.join(values)}}}{suffix}")
        lines.append("};")
        lines.append("")

    add_array(
        "SECTION_TYPES",
        "SectionType",
        schema["section_type"],
        ["name", "id", "long_name"],
    )
    add_array(
        "VALUE_TYPES", "ValueType", schema["value_type"], ["name", "id", "long_name"]
    )
    add_array(
        "SUMMARIZER_TYPES",
        "SummarizerType",
        schema["summarizer_type"],
        ["name", "id", "long_name"],
    )
    add_array(
        "METRIC_TYPES",
        "MetricDataType",
        schema["metric_type"],
        ["section", "name", "id", "type", "summarizer", "long_name"],
    )

    def add_map(
        map_decl: str, array_name: str, key_expr: str, item_count: int, item_type: str
    ) -> None:
        base_array_name = array_name.removesuffix("_BY_NAME").removesuffix("_BY_ID")
        lines.append(
            f"const std::unordered_map<{map_decl}, const {item_type}*> {array_name} = [] {{"
        )
        lines.append(f"    std::unordered_map<{map_decl}, const {item_type}*> values;")
        lines.append(f"    values.reserve({item_count});")
        lines.append(f"    for (const auto& entry : {base_array_name}) {{")
        lines.append(f"        values.emplace({key_expr}, &entry);")
        lines.append("    }")
        lines.append("    return values;")
        lines.append("}();")
        lines.append("")

    add_map(
        "std::string_view",
        "SECTION_TYPES_BY_NAME",
        "entry.name",
        len(schema["section_type"]),
        "SectionType",
    )
    add_map(
        "std::uint32_t",
        "SECTION_TYPES_BY_ID",
        "entry.id",
        len(schema["section_type"]),
        "SectionType",
    )
    add_map(
        "std::string_view",
        "VALUE_TYPES_BY_NAME",
        "entry.name",
        len(schema["value_type"]),
        "ValueType",
    )
    add_map(
        "std::uint32_t",
        "VALUE_TYPES_BY_ID",
        "entry.id",
        len(schema["value_type"]),
        "ValueType",
    )
    add_map(
        "std::string_view",
        "SUMMARIZER_TYPES_BY_NAME",
        "entry.name",
        len(schema["summarizer_type"]),
        "SummarizerType",
    )
    add_map(
        "std::uint32_t",
        "SUMMARIZER_TYPES_BY_ID",
        "entry.id",
        len(schema["summarizer_type"]),
        "SummarizerType",
    )
    add_map(
        "std::string_view",
        "METRIC_TYPES_BY_NAME",
        "entry.name",
        len(schema["metric_type"]),
        "MetricDataType",
    )
    add_map(
        "std::uint32_t",
        "METRIC_TYPES_BY_ID",
        "entry.id",
        len(schema["metric_type"]),
        "MetricDataType",
    )

    lines.extend(
        [
            "}  // namespace " + namespace,
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def main() -> int:
    args = parse_args()
    input_path = args.json_path.resolve()
    output_dir = args.output_dir.resolve()

    schema = validate_schema(load_json(input_path))
    module_name = input_path.name
    namespace = args.namespace

    if args.target in ("python", "both"):
        write_text(
            output_dir / PYTHON_OUTPUT, generate_python(module_name, input_path, schema)
        )
    if args.target in ("cpp", "both"):
        write_text(
            output_dir / CPP_HEADER_OUTPUT,
            generate_cpp_header(namespace, input_path, schema),
        )
        write_text(
            output_dir / CPP_SOURCE_OUTPUT,
            generate_cpp_source(namespace, input_path, schema),
        )

    print(f"Generated {args.target} output in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
