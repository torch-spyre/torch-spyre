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

"""Test-only proof that scheduler preflight and codegen finalize the same inputs."""

import copy
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch as mock_patch

import sympy


def _division_signature(division):
    if division is None:
        return None
    core_id = sympy.Symbol("core_id")
    return (
        division.physical_core_count,
        frozenset(
            (
                dim,
                int(split),
                tuple(
                    sympy.sympify(division.core_id_to_work_slice[dim]).subs(
                        core_id, core
                    )
                    for core in range(division.physical_core_count)
                ),
            )
            for dim, split in division.work_slices.items()
            if int(split) > 1
        ),
    )


def _normalize_call(alignment_inputs, divisions, kwargs):
    """Normalize only repeated identical tensor constraints.

    Scheduler dependency sets contain one read for ``x + x`` while codegen
    retains both TensorArgs. The finalizer has a focused idempotence test for
    that case. Alignment descriptors and their order remain exact; division
    expressions compare by the owners they assign, ignoring unsplit dimensions
    exactly as TensorWorkDivision.same_ownership does.
    """

    unique = []
    for tensor, division in zip(alignment_inputs.tensors, divisions):
        signature = _division_signature(division)
        if not any(
            tensor == seen_tensor and signature == seen_signature
            for seen_tensor, seen_signature in unique
        ):
            unique.append((tensor, signature))
    normalized_inputs = replace(
        alignment_inputs,
        tensors=[copy.deepcopy(tensor) for tensor, _ in unique],
    )
    return (
        normalized_inputs,
        tuple(signature for _, signature in unique),
        dict(kwargs),
    )


class LXFinalizerParity:
    """Audit the two real finalizer callers, keyed by scheduler-node identity."""

    def __init__(self):
        import torch_spyre._inductor.scheduler as scheduler_module
        import torch_spyre._inductor.spyre_kernel as spyre_kernel_module

        self.scheduler_module = scheduler_module
        self.spyre_kernel_module = spyre_kernel_module
        self.preflight_states = {}
        self.preflight_lx_names = {}
        self.preflight_failures = []
        self.relayout_demotions = []
        self.codegen_calls = []
        self.created_specs = []
        # One test can compile several graphs. Keep every observed scheduler
        # node alive so CPython cannot recycle its id and accidentally match a
        # later graph's preflight to an earlier graph's codegen call.
        self._node_refs = {}
        self._preflight_node = None
        self._codegen_node = None
        self._spec_nodes = {}
        self._real_finalize = scheduler_module.finalize_core_mapping_pure
        self._real_preflight = scheduler_module._preflight_lx_ownership
        self._real_demote_relayout = scheduler_module.demote_lx_relayout_group
        self._real_create_op_spec = spyre_kernel_module.SpyreKernel.create_op_spec
        self._real_simplify = spyre_kernel_module.simplify_op_spec
        self._patches = ExitStack()

    def _identity(self, node):
        node_id = id(node)
        prior = self._node_refs.setdefault(node_id, node)
        assert prior is node, "live scheduler-node identity was reused"
        operation = getattr(node, "node", None)
        get_operation_name = getattr(operation, "get_operation_name", None)
        operation_name = (
            get_operation_name()
            if callable(get_operation_name)
            else type(operation).__name__
        )
        return (node_id, node.get_name(), operation_name)

    def _record(self, caller, bucket=None):
        def wrapped(alignment_inputs, divisions, **kwargs):
            result = self._real_finalize(alignment_inputs, divisions, **kwargs)
            node = getattr(self, caller)
            if node is not None and any(division is not None for division in divisions):
                normalized = _normalize_call(
                    alignment_inputs,
                    divisions,
                    kwargs,
                )
                if bucket is not None:
                    bucket.append((node, normalized))
                if caller == "_preflight_node":
                    self.preflight_states[node] = normalized
            return result

        return wrapped

    def _preflight(self, node, *, relayout_copy):
        identity = self._identity(node)
        # Preflight runs to a fixed point. A later demotion can remove every
        # LX constraint from a node, in which case the final attempt calls
        # no finalizer. Clear the earlier attempt before making this one so
        # the corpus compares codegen with the stable ownership state only.
        self.preflight_states[identity] = None
        self.preflight_lx_names[identity] = tuple(
            sorted(
                {
                    dep.name
                    for dep in (*node.read_writes.reads, *node.read_writes.writes)
                    if self.scheduler_module._lx_layout(dep.name) is not None
                }
            )
        )
        self._preflight_node = identity
        try:
            return self._real_preflight(
                node,
                relayout_copy=relayout_copy,
            )
        except Exception as exc:
            self.preflight_failures.append(
                (
                    identity,
                    self.preflight_lx_names[identity],
                    relayout_copy,
                    type(exc).__name__,
                    str(exc),
                )
            )
            raise
        finally:
            self._preflight_node = None

    def _demote_relayout(self, graph, source_name, reason):
        self.relayout_demotions.append((source_name, reason))
        return self._real_demote_relayout(graph, source_name, reason)

    def _create_op_spec(self, kernel, *args, **kwargs):
        spec = self._real_create_op_spec(kernel, *args, **kwargs)
        identity = self._identity(kernel.current_node)
        self._spec_nodes[id(spec)] = identity
        self.created_specs.append((identity, copy.deepcopy(spec)))
        return spec

    def _simplify(self, spec, *args, **kwargs):
        self._codegen_node = self._spec_nodes.get(id(spec))
        try:
            return self._real_simplify(spec, *args, **kwargs)
        finally:
            self._codegen_node = None

    def __enter__(self):
        self._patches.enter_context(
            mock_patch.object(
                self.scheduler_module,
                "_preflight_lx_ownership",
                side_effect=self._preflight,
            )
        )
        self._patches.enter_context(
            mock_patch.object(
                self.scheduler_module,
                "demote_lx_relayout_group",
                side_effect=self._demote_relayout,
            )
        )
        self._patches.enter_context(
            mock_patch.object(
                self.scheduler_module,
                "finalize_core_mapping_pure",
                side_effect=self._record("_preflight_node"),
            )
        )
        self._patches.enter_context(
            mock_patch.object(
                self.spyre_kernel_module.SpyreKernel,
                "create_op_spec",
                side_effect=self._create_op_spec,
                autospec=True,
            )
        )
        self._patches.enter_context(
            mock_patch.object(
                self.spyre_kernel_module,
                "simplify_op_spec",
                side_effect=self._simplify,
            )
        )
        self._patches.enter_context(
            mock_patch.object(
                self.spyre_kernel_module,
                "finalize_core_mapping_pure",
                side_effect=self._record("_codegen_node", self.codegen_calls),
            )
        )
        return self

    def __exit__(self, *exc_info):
        return self._patches.__exit__(*exc_info)

    def assert_codegen_covered(self):
        """Every emitted LX finalization must equal a full-node preflight."""

        for codegen_call in self.codegen_calls:
            expected = self.preflight_states.get(codegen_call[0])
            assert expected == codegen_call[1], (
                "codegen finalized LX ownership without the same per-node "
                f"scheduler inputs: {codegen_call[0]}; scheduler saw "
                f"{expected}; codegen={codegen_call[1]}"
            )
        return [
            (identity, state)
            for identity, state in self.preflight_states.items()
            if state is not None and (identity, state) not in self.codegen_calls
        ]

    def assert_complete(self):
        """Every accepted full-node preflight must also reach codegen."""

        unmatched_preflight = self.assert_codegen_covered()
        assert not unmatched_preflight, (
            "scheduler accepted LX ownership that did not reach codegen: "
            f"{unmatched_preflight}; codegen saw "
            f"{[call[0] for call in self.codegen_calls]}"
        )

    def fallback_report(self):
        """Describe every scheduler fallback without changing its behavior."""

        report = []
        for (
            identity,
            lx_names,
            relayout_copy,
            error_type,
            reason,
        ) in self.preflight_failures:
            final_names = self.preflight_lx_names.get(identity, ())
            final_state = self.preflight_states.get(identity)
            report.append(
                "LX_FINALIZER_PREFLIGHT_FALLBACK "
                f"node={identity!r} buffers={lx_names!r} "
                f"relayout_copy={relayout_copy!r} error={error_type}: {reason}; "
                f"final_buffers={final_names!r} "
                f"finalizer_inputs={'present' if final_state is not None else 'empty'}"
            )
        for source_name, reason in self.relayout_demotions:
            report.append(
                "LX_RELAYOUT_STRUCTURAL_DEMOTION "
                f"source={source_name!r} reason={reason}"
            )
        return report
