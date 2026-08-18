# Copyright 2025 The Torch-Spyre Authors.
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

"""AIUPTI/Kineto event-name transport for Spyre kernel provenance.

Upstream Inductor kernel names are backend-qualified and place a descriptive
fused-op component before a hash or unique suffix. Torch-spyre follows that
convention while reserving space for the JobPlan #<step> suffix. Libaiupti
stores AIUpti_ActivityCompute.name in char[128] and reserves one byte for the
terminating NUL. The source of truth is
libaiupti/include/libaiupti/aiupti_activity.h.
"""

from __future__ import annotations

import ctypes

import regex

from torch_spyre._C import AIUPTI_ACTIVITY_NAME_MAX_BYTES
from torch_spyre._inductor.kernel_provenance import (
    KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
    KERNEL_PROVENANCE_KEY_VERSION,
    KernelProvenanceDescriptor,
)


# PR #2930 represents the JobPlan command position as C++ size_t and its
# fallback spells the disambiguator as #<step_idx>. Python and the extension
# share one process ABI, so derive the maximum suffix width.
_SIZE_T_BITS = ctypes.sizeof(ctypes.c_size_t) * 8
_MAX_COMPUTE_STEP_SUFFIX_BYTES = len(f"#{(1 << _SIZE_T_BITS) - 1}")
_MAX_EVENT_NAME_BASE_BYTES = (
    AIUPTI_ACTIVITY_NAME_MAX_BYTES - _MAX_COMPUTE_STEP_SUFFIX_BYTES
)

_SUPPORTED_KEY_WIDTHS = {
    KERNEL_PROVENANCE_KEY_VERSION: KERNEL_PROVENANCE_KEY_BASE32_WIDTH
}
_EVENT_NAME_PREFIX = f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_"
# The separator before the key is unambiguous because ``_`` is not in the
# lowercase base32 alphabet. Revisit this parser if a future version changes
# the alphabet.
_EVENT_KEY_RE = regex.compile(
    r"\Aspyre_kernel_v(?P<version>[0-9]+)_"
    r"[A-Za-z0-9_]+?_"
    r"(?P<key>[a-z2-7]+)"
    r"(?:#[0-9]+)?\Z"
)
_DISPLAY_COMPONENT_RE = regex.compile(r"[^A-Za-z0-9]+")
_MIN_EVENT_NAME_BASE_BYTES = len(
    f"{_EVENT_NAME_PREFIX}u_{'a' * KERNEL_PROVENANCE_KEY_BASE32_WIDTH}".encode("ascii")
)
if _MIN_EVENT_NAME_BASE_BYTES > _MAX_EVENT_NAME_BASE_BYTES:
    raise RuntimeError("kernel provenance event-name fields exceed AIUPTI limit")


def format_kernel_provenance_event_name(
    descriptor: KernelProvenanceDescriptor,
) -> str:
    """Encode a bounded, human-readable event name for a kernel descriptor."""
    display = _aten_summary(descriptor.aten_ops)
    key_suffix = f"_{descriptor.key}"
    display_budget = max(
        1,
        _MAX_EVENT_NAME_BASE_BYTES
        - len(f"{_EVENT_NAME_PREFIX}{key_suffix}".encode("ascii")),
    )
    display = display[:display_budget].rstrip("_") or "u"
    name = f"{_EVENT_NAME_PREFIX}{display}{key_suffix}"
    assert len(name.encode("ascii")) <= _MAX_EVENT_NAME_BASE_BYTES, name
    return name


def extract_kernel_provenance_key(event_name: str) -> str | None:
    """Extract a kernel-provenance key from a Spyre device event name."""
    match = _EVENT_KEY_RE.match(event_name)
    if match is None:
        return None
    version = int(match.group("version"))
    expected_width = _SUPPORTED_KEY_WIDTHS.get(version)
    key = match.group("key")
    return key if expected_width is not None and len(key) == expected_width else None


def _aten_summary(aten_ops: tuple[str, ...]) -> str:
    """Return an upstream-style, display-only fused ATen packet summary."""
    names = sorted({_aten_packet_name(aten_op) for aten_op in aten_ops})
    return "_".join(["fused", *(names or ["unknown"])])


def _aten_packet_name(aten_op: str) -> str:
    parts = aten_op.split(".")
    packet = parts[1] if len(parts) >= 3 else aten_op
    return _sanitize_component(packet)


def _sanitize_component(value: str) -> str:
    sanitized = _DISPLAY_COMPONENT_RE.sub("_", value).strip("_")
    return sanitized or "unknown"


# The event name remains the compatibility join for raw traces and name-only
# consumers even when Kineto also carries structured provenance metadata.
