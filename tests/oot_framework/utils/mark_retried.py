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

    mark_retried.py <signal|stall|pod> <xml> [<replaced_xml>]

A retry overwrites the file's XML, so without this the ingest cannot tell a retried file's
results from a first attempt. Adds `result.retried=<kind>` (the ingest stores `result.*`
properties in the case's props); a file retried more than once lists each kind, innermost first.
Given the report the retry replaced, a case that failed or errored there also gets
`result.prior_status` and `result.prior_message`, the only record of a failure the retry hid.
"""

import sys
import xml.etree.ElementTree as ET

KINDS = ("signal", "stall", "pod")
PROPERTY = "result.retried"
PRIOR_STATUS = "result.prior_status"
PRIOR_MESSAGE = "result.prior_message"
# Enough to classify the failure; the replaced report's full text is not kept anywhere.
PRIOR_MESSAGE_MAX = 1000


def _prior_failures(xml_path: str) -> dict[tuple[str, str], tuple[str, str]]:
    """(classname, name) -> (status, message) for each failed or errored case."""
    failures = {}
    for tc in ET.parse(xml_path).getroot().iter("testcase"):
        for status in ("failure", "error"):
            el = tc.find(status)
            if el is not None:
                message = el.get("message") or (el.text or "").strip()
                failures[(tc.get("classname", ""), tc.get("name", ""))] = (
                    "failed" if status == "failure" else "error",
                    message[:PRIOR_MESSAGE_MAX],
                )
                break
    return failures


def _set(props: ET.Element, name: str, value: str) -> None:
    prop = next((p for p in props.findall("property") if p.get("name") == name), None)
    if prop is None:
        ET.SubElement(props, "property", name=name, value=value)
    else:
        prop.set("value", value)


def mark(xml_path: str, kind: str, replaced_xml: str | None = None) -> tuple[int, int]:
    """Mark each testcase retried as `kind`; returns (testcases, prior failures recorded)."""
    prior = _prior_failures(replaced_xml) if replaced_xml else {}
    tree = ET.parse(xml_path)
    cases = list(tree.getroot().iter("testcase"))
    recorded = 0
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
        failure = prior.get((tc.get("classname", ""), tc.get("name", "")))
        if failure:
            _set(props, PRIOR_STATUS, failure[0])
            _set(props, PRIOR_MESSAGE, failure[1])
            recorded += 1
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return len(cases), recorded


def main() -> None:
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in KINDS:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(KINDS)}> <xml> [<replaced_xml>]")
    kind, xml_path = sys.argv[1], sys.argv[2]
    replaced_xml = sys.argv[3] if len(sys.argv) == 4 else None
    cases, recorded = mark(xml_path, kind, replaced_xml)
    print(
        f"[mark_retried] {kind}: {cases} testcase(s) in {xml_path}"
        + (f", {recorded} prior failure(s) from {replaced_xml}" if replaced_xml else "")
    )


if __name__ == "__main__":
    main()
