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
"""Mark every testcase in a JUnit XML as coming from a whole-file retry.

    mark_retried.py <signal|stall|pod> <xml>

A retry overwrites the file's XML, so without this the ingest cannot tell a retried file's
results from a first attempt. Adds `result.retried=<kind>` (the ingest stores `result.*`
properties in the case's props); a file retried more than once lists each kind, innermost first.
"""

import sys
import xml.etree.ElementTree as ET

KINDS = ("signal", "stall", "pod")
PROPERTY = "result.retried"


def mark(xml_path: str, kind: str) -> int:
    """Add `kind` to each testcase's result.retried; returns the number of testcases."""
    tree = ET.parse(xml_path)
    cases = list(tree.getroot().iter("testcase"))
    for tc in cases:
        props = tc.find("properties")
        if props is None:
            props = ET.Element("properties")
            tc.insert(0, props)
        prop = next(
            (p for p in props.findall("property") if p.get("name") == PROPERTY), None
        )
        if prop is None:
            ET.SubElement(props, "property", name=PROPERTY, value=kind)
        elif kind not in prop.get("value", "").split(","):
            prop.set("value", f"{prop.get('value')},{kind}")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return len(cases)


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in KINDS:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(KINDS)}> <xml>")
    kind, xml_path = sys.argv[1], sys.argv[2]
    print(f"[mark_retried] {kind}: {mark(xml_path, kind)} testcase(s) in {xml_path}")


if __name__ == "__main__":
    main()
