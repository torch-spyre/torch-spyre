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

"""Versioned Spyre profiler event-name parsing without compiler imports."""

from __future__ import annotations

import dataclasses

import regex


KERNEL_PROVENANCE_KEY_VERSION = 1
KERNEL_PROVENANCE_KEY_BASE32_WIDTH = 16

_SUPPORTED_KEY_WIDTHS = {
    KERNEL_PROVENANCE_KEY_VERSION: KERNEL_PROVENANCE_KEY_BASE32_WIDTH
}
# The separator before the key is unambiguous because ``_`` is not in the
# lowercase base32 alphabet. Every supported version must keep the separator
# outside its declared key alphabet.
_EVENT_KEY_RE = regex.compile(
    r"\Aspyre_kernel_v(?P<version>[0-9]+)_"
    r"[A-Za-z0-9_]+?_"
    r"(?P<key>[a-z2-7]+)"
    r"(?P<suffix>#(?P<step>[0-9]+))?\Z"
)


@dataclasses.dataclass(frozen=True)
class ParsedKernelProvenanceEvent:
    """Validated event transport fields used by compiler and offline readers."""

    name: str
    base_name: str
    key: str
    step: int | None
    step_suffix: str | None


def parse_kernel_provenance_event_name(
    event_name: str,
) -> ParsedKernelProvenanceEvent | None:
    """Parse one supported Spyre provenance event name and command step."""
    match = _EVENT_KEY_RE.match(event_name)
    if match is None:
        return None
    version = int(match.group("version"))
    expected_width = _SUPPORTED_KEY_WIDTHS.get(version)
    key = match.group("key")
    if expected_width is None or len(key) != expected_width:
        return None
    suffix = match.group("suffix")
    return ParsedKernelProvenanceEvent(
        name=event_name,
        base_name=event_name[: -len(suffix)] if suffix else event_name,
        key=key,
        step=int(match.group("step")) if suffix else None,
        step_suffix=suffix,
    )
