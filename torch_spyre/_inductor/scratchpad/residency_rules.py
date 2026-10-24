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


from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol


class ResidencyChecks(Protocol):
    def op_allowed(self) -> Optional[str]: ...
    def sized(self) -> Optional[str]: ...
    def empty(self) -> Optional[str]: ...
    def mutation(self) -> Optional[str]: ...
    def advancing(self) -> Optional[str]: ...
    def restickify(self) -> Optional[str]: ...
    def extern(self) -> Optional[str]: ...
    def multi_output_extern(self) -> Optional[str]: ...
    def indirect(self) -> Optional[str]: ...
    def reinterpret_output(self) -> Optional[str]: ...
    def mutated_output(self) -> Optional[str]: ...
    def partial_read(self) -> Optional[str]: ...
    def core_division(self) -> Optional[str]: ...
    def untiled(self) -> Optional[str]: ...
    def back_gap(self) -> Optional[str]: ...
    def loop_carry(self) -> bool: ...
    def drain(self) -> bool: ...
    def prefix_gather(self) -> bool: ...


@dataclass(frozen=True)
class ResidencyException:
    name: str
    certificate: str
    proves: Callable[[ResidencyChecks], bool]
    requires: tuple[str, ...]
    tests: tuple[str, ...]


@dataclass(frozen=True)
class ResidencyRule:
    name: str
    reason: str
    check: Callable[[ResidencyChecks], Optional[str]]
    requires: tuple[str, ...]
    tests: tuple[str, ...]
    exceptions: tuple[ResidencyException, ...] = ()


PREFIX_GATHER = ResidencyException(
    "Zero-offset linear prefix",
    "prefix_lx_gather_source",
    lambda c: c.prefix_gather(),
    (
        "one reader and one writer",
        "full-backing served copy",
        "physical stride preserved",
        "unsplit K",
        "disjoint LX spans",
    ),
    (
        "test_prefix_lx_contraction_requires_exact_backing_stride",
        "test_prefix_linear_reads_a_separate_full_backing_lx_gather",
    ),
)

# This order is the allocator's first-failure contract. Keep checks lazy.
BUFFER_RESIDENCY_RULES = (
    ResidencyRule(
        "op_allowed",
        "op not allowed",
        lambda c: c.op_allowed(),
        ("supported producer",),
        ("TestLoopCarryLxEligibility",),
    ),
    ResidencyRule(
        "sized",
        "unsized (no device layout)",
        lambda c: c.sized(),
        ("computable footprint",),
        ("TestEmptyLxEligibility",),
    ),
    ResidencyRule(
        "empty",
        "empty tensor",
        lambda c: c.empty(),
        ("nonempty output",),
        ("TestEmptyLxEligibility",),
    ),
    ResidencyRule(
        "mutation",
        "mutation target",
        lambda c: c.mutation(),
        ("no mutation target",),
        ("TestLoopCarryLxEligibility",),
        (
            ResidencyException(
                "Counted-loop carry",
                "_is_loop_carry_storage",
                lambda c: c.loop_carry(),
                ("joint division choice",),
                ("TestLoopCarryLxEligibility",),
            ),
        ),
    ),
    ResidencyRule(
        "advancing",
        "tiled (advancing)",
        lambda c: c.advancing(),
        ("fixed LX address",),
        ("TestLoopCarryLxEligibility",),
    ),
    ResidencyRule(
        "restickify",
        "read by restickify",
        lambda c: c.restickify(),
        ("fixed divisions agree or exact local-read proof",),
        ("TestRestickifyBarrierDeferredOnJointPath",),
    ),
    ResidencyRule(
        "extern",
        "extern kernel user or live across extern kernel",
        lambda c: c.extern(),
        ("extern boundary protected",),
        ("test_allocator_residency_registry_preserves_guard_order",),
    ),
    ResidencyRule(
        "multi_output_extern",
        "live across multi-output extern kernel",
        lambda c: c.multi_output_extern(),
        ("multi-output boundary protected",),
        ("test_allocator_residency_registry_preserves_guard_order",),
    ),
    ResidencyRule(
        "indirect",
        "index tensor or indirectly accessed",
        lambda c: c.indirect(),
        ("direct addressing",),
        ("test_allocator_residency_registry_preserves_guard_order",),
    ),
    ResidencyRule(
        "reinterpret_output",
        "graph output is a ReinterpretView",
        lambda c: c.reinterpret_output(),
        ("rewritable output clone",),
        ("test_allocator_residency_registry_preserves_guard_order",),
    ),
    ResidencyRule(
        "mutated_output",
        "graph output mutated after production",
        lambda c: c.mutated_output(),
        ("output clone sees final value",),
        ("TestLoopCarryLxEligibility",),
        (
            ResidencyException(
                "Validated output drain",
                "validated_drain_plans",
                lambda c: c.drain(),
                ("clone follows counted loop",),
                ("TestLoopCarryLxEligibility",),
            ),
        ),
    ),
    ResidencyRule(
        "partial_read",
        "partial/offset read",
        lambda c: c.partial_read(),
        ("complete read",),
        ("TestRestickifyBarrierDeferredOnJointPath",),
        (PREFIX_GATHER,),
    ),
    ResidencyRule(
        "core_division",
        "core div mismatch: ...",
        lambda c: c.core_division(),
        ("compatible fixed division",),
        ("TestRestickifyBarrierDeferredOnJointPath",),
    ),
    ResidencyRule(
        "untiled",
        "graph output (no clone); no consumer reads it from LX",
        lambda c: c.untiled(),
        ("untiled reuse or boundary clone",),
        ("TestLoopCarryLxEligibility",),
    ),
    ResidencyRule(
        "back_gap",
        "lx back gap",
        lambda c: c.back_gap(),
        ("no unsupported LX gap",),
        ("test_prefix_linear_reads_a_separate_full_backing_lx_gather",),
        (PREFIX_GATHER,),
    ),
)


def buffer_residency_reason(checks: ResidencyChecks) -> Optional[str]:
    for rule in BUFFER_RESIDENCY_RULES:
        reason = rule.check(checks)
        if reason is not None and not any(
            exception.proves(checks) for exception in rule.exceptions
        ):
            return reason
    return None


def residency_rules_markdown() -> str:
    """Generate the output-buffer table from the checks used by the allocator."""
    lines = [
        "| Rule / reason | Requires | Exceptions / certificate | Covering tests |",
        "|---|---|---|---|",
    ]
    for rule in BUFFER_RESIDENCY_RULES:
        exceptions = (
            "; ".join(
                f"{e.name}: `{e.certificate}` ({', '.join(e.requires)})"
                for e in rule.exceptions
            )
            or "None"
        )
        tests = tuple(
            dict.fromkeys(
                rule.tests + tuple(test for e in rule.exceptions for test in e.tests)
            )
        )
        lines.append(
            f"| `{rule.reason}` | {', '.join(rule.requires)} | "
            f"{exceptions} | {'; '.join(f'`{t}`' for t in tests)} |"
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class TransferCertificate:
    """Describe an existing proof; these rows never dispatch admission."""

    name: str
    certificate: str
    requires: str
    predicate_tests: tuple[str, ...]
    integration_tests: tuple[str, ...]


TRANSFER_CERTIFICATES = (
    TransferCertificate(
        "Updated-input cache snapshot",
        "torch_spyre._inductor.scratchpad.lx_relayout._is_activation_source",
        "earlier direct cache mutation; immutable restickified snapshot; "
        "original cache load and dependencies preserved",
        (
            "test_work_division_hint.py::test_lx_relayout_activation_policy_is_source_wide",
        ),
        ("test_solver_relayout_e2e.py::test_attention_cache_transpose_handoff",),
    ),
)


def transfer_certificates_markdown() -> str:
    """List certificates outside the ordered residency/projection dispatch."""
    lines = [
        "| Certificate | Function | Requires | Predicate tests | Integration tests |",
        "|---|---|---|---|---|",
    ]
    for row in TRANSFER_CERTIFICATES:
        tests = [
            "; ".join(f"`{test}`" for test in group) or "None"
            for group in (row.predicate_tests, row.integration_tests)
        ]
        lines.append(
            f"| {row.name} | `{row.certificate}` | {row.requires} | "
            f"{tests[0]} | {tests[1]} |"
        )
    return "\n".join(lines) + "\n"
