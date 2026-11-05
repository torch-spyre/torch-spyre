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

"""Capture final OFF plans without changing placement or work division.

Use the same contract and existing fixture on each adjacent source pair. Input
and hint hashes identify the actual fixture data and hints. Keep upstream and
all deterministic compiler settings matched. Capture is not a correctness test.
Buffer names remain significant: an unstable name stops equality rather than
hiding a possible dependency change. No address or owner expression is dropped.
"""

from contextlib import contextmanager
import dataclasses
import enum
import hashlib
import json
import math
from pathlib import Path
import subprocess
from unittest.mock import patch

import sympy
import torch
from torch.utils._ordered_set import OrderedSet


def freeze(value):
    """Encode every supported field; unknown objects stop the comparison."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"float": value.hex()}
    if isinstance(value, sympy.Basic):
        return {"sympy": sympy.srepr(value)}
    if isinstance(value, (torch.dtype, torch.device)):
        return {"torch": str(value)}
    if isinstance(value, enum.Enum):
        return {"enum": type(value).__qualname__, "name": value.name}
    if type(value).__module__ == "torch_spyre._C" and hasattr(value, "value"):
        return {"enum": type(value).__qualname__, "value": int(value.value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "record": type(value).__qualname__,
            "fields": {
                f.name: freeze(getattr(value, f.name))
                for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, dict):
        pairs = [(freeze(key), freeze(item)) for key, item in value.items()]
        return {"mapping": pairs}
    if isinstance(value, (tuple, list)):
        return {"sequence": [freeze(item) for item in value]}
    if isinstance(value, OrderedSet):
        return {"ordered_set": [freeze(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {
            "set": sorted(
                (freeze(item) for item in value),
                key=lambda item: json.dumps(item, sort_keys=True),
            )
        }
    raise TypeError(
        f"unsupported semantic field: {type(value).__module__}.{type(value).__qualname__}"
    )


_PRE_FLAG_BASELINE = "525dfd8ca6f6acc639238610e23e1483e06f5854"
_PRE_FLAG_MAIN_BASELINE = "10f409548e575dd30833f168d82d98b719def2f4"
_PRE_FLAG_CONFIG_SHA256 = (
    "8bc7e038c6b9b3c984d0a4f337428e9bb6818d5f5a4f8fb4036d9fce2846b236"
)
_MAIN_REFS = (
    "refs/heads/main",
    "refs/remotes/origin/main",
    "refs/remotes/upstream/main",
)


def _is_pre_flag_baseline(config, compiler_head):
    if compiler_head in {_PRE_FLAG_BASELINE, _PRE_FLAG_MAIN_BASELINE}:
        return True
    # Later baselines must be the actual checkout on a known main line, not
    # merely any feature branch descended from the squash merge.
    directory = str(Path(config.__file__).resolve().parent)
    commands = (
        ["rev-parse", "HEAD"],
        ["merge-base", "--is-ancestor", _PRE_FLAG_MAIN_BASELINE, compiler_head],
        [
            "for-each-ref",
            "--contains",
            compiler_head,
            "--format=%(refname)",
            *_MAIN_REFS,
        ],
    )
    try:
        results = [
            subprocess.run(
                ["git", "-C", directory, *command],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for command in commands
        ]
    except (OSError, subprocess.TimeoutExpired):
        return False
    return (
        all(result.returncode == 0 for result in results)
        and results[0].stdout.strip() == compiler_head
        and bool(set(results[2].stdout.splitlines()) & set(_MAIN_REFS))
    )


def _live_setting(config, key, compiler_head):
    """Only verified, unchanged pre-flag sources may omit the rollout switch."""
    if hasattr(config, key):
        return getattr(config, key)
    if key != "lx_extended_ownership" or not _is_pre_flag_baseline(
        config, compiler_head
    ):
        raise AttributeError(f"missing compiler setting: {key}")
    digest = hashlib.sha256(Path(config.__file__).read_bytes()).hexdigest()
    if digest != _PRE_FLAG_CONFIG_SHA256:
        raise ValueError("pre-flag baseline config source does not match")
    return False


def _layout(layout):
    if layout is not None and type(layout).__qualname__ not in {
        "FixedTiledLayout",
        "FixedLayout",
        "MutationLayoutSHOULDREMOVE",
        "NoneLayout",
    }:
        raise TypeError(f"unsupported storage layout: {type(layout).__qualname__}")
    fields = {}
    for name in (
        "size",
        "stride",
        "offset",
        "dtype",
        "device",
        "allocation",
        "lx_view",
        "target",
    ):
        if hasattr(layout, name):
            value = getattr(layout, name)
            if name == "target" and hasattr(value, "get_name"):
                value = {"buffer": value.get_name()}
            fields[name] = freeze(value)
    physical = getattr(layout, "device_layout", None)
    if physical is not None:
        fields["physical"] = {
            name: freeze(getattr(physical, name))
            for name in (
                "device_size",
                "stride_map",
                "device_dtype",
                "element_arrangement",
                "valid_elements",
            )
        }
    return {"layout": type(layout).__qualname__, "fields": fields}


def graph_snapshot(graph):
    """Record final storage, dependencies, copies and physical placement."""
    from torch_spyre._inductor.pass_utils import op_read_writes
    from torch_spyre._inductor.scratchpad.lx_relayout import materialized_lx_relayouts

    operations = []
    for op in graph.operations:
        layout = op.get_layout()
        entry = {
            "name": op.get_name(),
            "operation_name": op.get_operation_name(),
            "kind": type(op).__qualname__,
            "layout": _layout(layout),
            "dependencies": freeze(op_read_writes(op)),
        }
        for name in ("iteration_space_ownership", "op_it_space_splits", "loop_info"):
            if hasattr(op, name):
                entry[name] = freeze(getattr(op, name))
        operations.append(entry)
    return {
        "inputs": [
            (name, _layout(value.get_layout()))
            for name, value in graph.graph_inputs.items()
        ],
        "operations": operations,
        "outputs": freeze(graph.get_output_names()),
        "removed_buffers": freeze(set(graph.removed_buffers)),
        "relayout_registry": freeze(materialized_lx_relayouts(graph)),
    }


def validate_contract(contract):
    required = {
        "fixture",
        "input_sha256",
        "hints_sha256",
        "matched_upstream",
        "settings",
    }
    if (
        set(contract) != required
        or not contract["fixture"]
        or not contract["matched_upstream"]
    ):
        raise ValueError("a complete matched OFF-plan contract is required")
    for key in ("input_sha256", "hints_sha256"):
        value = contract[key]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid {key}")
    required_settings = {
        "lx_extended_ownership",
        "num_cpus",
        "sencores",
        "layout_solver",
        "co_optimizing_lx_planning",
        "lx_planner_relayout",
        "lx_same_tile_group_relayout",
        "ktir_emitter",
    }
    if not required_settings <= contract["settings"].keys():
        raise ValueError("deterministic compiler settings are incomplete")
    if contract["settings"]["num_cpus"] != 1:
        raise ValueError("use one solver worker for matched plan comparisons")
    if contract["settings"].get("lx_extended_ownership") is not False:
        raise ValueError("the comparison requires the rollout flag OFF")


@contextmanager
def capture_off_plan(contract, *, compiler_head):
    """Observe successful final codegen; do not override planner choices.

    The caller must supply the same verified inputs, hints and compiler settings
    on both source trees. A cache hit without emission cannot satisfy this gate.
    Numerical and device checks are separate, unchanged gates.
    """
    from torch._inductor.virtualized import V
    from torch_spyre._inductor import config
    from torch_spyre._inductor.spyre_kernel import SpyreKernel

    validate_contract(contract)
    if _live_setting(config, "lx_extended_ownership", compiler_head):
        raise ValueError("the live compiler rollout flag must be OFF")
    record = {
        "contract": json.loads(json.dumps(contract, allow_nan=False)),
        "compiler_head": compiler_head,
        "kernels": [],
    }
    original = SpyreKernel.codegen_kernel

    def capture(kernel):
        for key, value in contract["settings"].items():
            if _live_setting(config, key, compiler_head) != value:
                raise ValueError(f"live compiler setting differs: {key}")
        result = original(kernel)
        record["kernels"].append(
            {
                "graph": graph_snapshot(V.graph),
                "op_specs": freeze(kernel.op_specs),
                "pool_size": freeze(kernel.pool_size),
                "arguments": freeze(kernel.spyre_kernel_args),
            }
        )
        return result

    with patch.object(SpyreKernel, "codegen_kernel", capture):
        yield record
    if not record["kernels"]:
        raise ValueError(
            "no final kernel was captured; cache hits do not prove equality"
        )


def validate_record(record):
    if set(record) != {"contract", "compiler_head", "kernels"}:
        raise ValueError("incomplete semantic-plan record")
    validate_contract(record["contract"])
    if not record["kernels"]:
        raise ValueError("both plans must contain final kernels")
    for kernel in record["kernels"]:
        if set(kernel) != {"graph", "op_specs", "pool_size", "arguments"}:
            raise ValueError("incomplete final kernel")
        graph = kernel["graph"]
        if (
            set(graph)
            != {
                "inputs",
                "operations",
                "outputs",
                "removed_buffers",
                "relayout_registry",
            }
            or not graph["operations"]
        ):
            raise ValueError("incomplete final graph")
        if not kernel["op_specs"].get("sequence"):
            raise ValueError("missing final operation specifications")


def compare_plans(left, right):
    validate_record(left)
    validate_record(right)
    if left["contract"] != right["contract"]:
        raise AssertionError("unmatched inputs, hints, settings or upstream")
    if left["kernels"] != right["kernels"]:
        raise AssertionError("final semantic plans differ")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    compare_plans(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
    print("Final semantic plans match under the supplied matched contract.")


if __name__ == "__main__":
    main()
