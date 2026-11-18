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
    python -m spyre_clickhouse_ingest.mark_retried ...   (or: spyre-mark-retried ...)

A retry overwrites the file's XML, so without this the ingest cannot tell a retried file's
results from a first attempt. Adds `result.retried=<kind>` (the ingest stores `result.*`
properties in the case's props); a file retried more than once lists each kind, innermost first.
Given the report the retry replaced, a case that failed or errored there also gets
`result.prior_status` and `result.prior_message`, the only record of a failure the retry hid.
A prior mark already on the replaced report carries forward, and a case the retry did not
report at all is copied over from the replaced report with `result.not_rerun=<kind>`.

Stdlib only: the test pods run it by path, outside any install of this package.
"""

import copy
import sys
import xml.etree.ElementTree as ET

KINDS = ("signal", "stall", "pod")
PROPERTY = "result.retried"
PRIOR_STATUS = "result.prior_status"
PRIOR_MESSAGE = "result.prior_message"
NOT_RERUN = "result.not_rerun"
# Enough to classify the failure; the replaced report's full text is not kept anywhere.
PRIOR_MESSAGE_MAX = 1000


def _key(tc: ET.Element) -> tuple[str, str]:
    return tc.get("classname", ""), tc.get("name", "")


def _props(tc: ET.Element) -> ET.Element:
    props = tc.find("properties")
    if props is None:
        props = ET.Element("properties")
        tc.insert(0, props)
    return props


def _get(tc: ET.Element, name: str) -> str | None:
    return next(
        (p.get("value", "") for p in tc.iter("property") if p.get("name") == name), None
    )


def _set(props: ET.Element, name: str, value: str) -> None:
    prop = next((p for p in props.findall("property") if p.get("name") == name), None)
    if prop is None:
        ET.SubElement(props, "property", name=name, value=value)
    else:
        prop.set("value", value)


def _priors(xml_path: str) -> dict[tuple[str, str], tuple[str, str, ET.Element]]:
    """(classname, name) -> (status, message, testcase) for each case with a failure to keep.

    A failure or error in this report wins; else a prior mark the report already carries.
    """
    priors = {}
    for tc in ET.parse(xml_path).getroot().iter("testcase"):
        el = next(
            (e for e in (tc.find("failure"), tc.find("error")) if e is not None), None
        )
        if el is not None:
            message = el.get("message") or (el.text or "").strip()
            status = "failed" if el.tag == "failure" else "error"
            priors[_key(tc)] = (status, message[:PRIOR_MESSAGE_MAX], tc)
        elif _get(tc, PRIOR_STATUS):
            priors[_key(tc)] = (
                _get(tc, PRIOR_STATUS),
                _get(tc, PRIOR_MESSAGE) or "",
                tc,
            )
    return priors


def mark(
    xml_path: str, kind: str, replaced_xml: str | None = None
) -> tuple[int, int, int]:
    """Mark each testcase retried as `kind`.

    Returns (testcases marked, prior failures recorded, cases carried over unrun).
    """
    priors = _priors(replaced_xml) if replaced_xml else {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cases = list(root.iter("testcase"))
    recorded = 0
    for tc in cases:
        props = _props(tc)
        prop = next(
            (p for p in props.findall("property") if p.get("name") == PROPERTY), None
        )
        if prop is None:
            ET.SubElement(props, "property", name=PROPERTY, value=kind)
        elif kind not in prop.get("value", "").split(","):
            prop.set("value", f"{prop.get('value')},{kind}")
        prior = priors.get(_key(tc))
        if prior:
            _set(props, PRIOR_STATUS, prior[0])
            _set(props, PRIOR_MESSAGE, prior[1])
            recorded += 1
    # The ingest reads only the first <testsuite>, so carried cases go there.
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    reported = {_key(tc) for tc in cases}
    carried = 0
    for key, (_, _, replaced) in priors.items():
        if suite is None or key in reported:
            continue
        tc = copy.deepcopy(replaced)
        _set(_props(tc), NOT_RERUN, kind)
        suite.append(tc)
        verdict = next(
            (
                a
                for e, a in (("failure", "failures"), ("error", "errors"))
                if tc.find(e) is not None
            ),
            None,
        )
        for attr in ("tests", verdict):
            if attr and suite.get(attr, "").isdigit():
                suite.set(attr, str(int(suite.get(attr)) + 1))
        carried += 1
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return len(cases), recorded, carried


def main() -> None:
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in KINDS:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(KINDS)}> <xml> [<replaced_xml>]")
    kind, xml_path = sys.argv[1], sys.argv[2]
    replaced_xml = sys.argv[3] if len(sys.argv) == 4 else None
    cases, recorded, carried = mark(xml_path, kind, replaced_xml)
    print(
        f"[mark_retried] {kind}: {cases} testcase(s) in {xml_path}"
        + (
            f", {recorded} prior failure(s) and {carried} unrun case(s)"
            f" carried from {replaced_xml}"
            if replaced_xml
            else ""
        )
    )


if __name__ == "__main__":
    main()
