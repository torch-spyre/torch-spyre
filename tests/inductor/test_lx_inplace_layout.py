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

"""
Unit tests for the LX in-place layout promotion in
``_multi_arg_pointwise_layouts`` (propagate_layouts.py).

The block promotes a same-frame input's device layout to the front of the
output's candidate list so the beam commits it on a cost tie, letting the
allocator's positional in-place check reuse the input's LX slot without a
restickify.

Two behaviours are covered:

- ``test_add_reuses_matmul_input_layout`` — in ``matmul(Q * scale, K.T) +
  mask`` the add's committed layout must equal the matmul output's layout
  (the promotion fired). This is the motivating flash-attention pattern.

The reverse regression — promoting an fp8-unpack layout onto a plain output,
which then fails ``copy_tensor`` — is guarded by the footprint check in the
block. Re-testing it here would be redundant: it is covered end-to-end by the
fp8 op tests, e.g.
``tests/inductor/test_fp8_operations.py::TestFP8Operations::test_qfp8ch_basic_conversion``.
"""

import math
from unittest.mock import patch as mock_patch

import torch
import torch._dynamo
import torch._inductor.config as inductor_config

import torch_spyre._inductor.propagate_layouts as propagate_layouts
from torch_spyre._inductor import config, spyre_hint
from utils_inductor import DEVICE

_LAUNCH_JOBPLAN = "torch_spyre.execution.kernel_runner.launch_jobplan"
_PREPARE_KERNEL = "torch_spyre.execution.kernel_runner.prepare_kernel"


def _capture_multi_arg_layouts(original):
    """Wrap ``_multi_arg_pointwise_layouts`` to record, per op, the committed
    (first) output candidate and the layout of each same-frame input.

    Returns ``(wrapper, captured)`` where ``captured`` maps op name ->
    (committed_layout, {arg_name: layout}) and each layout is a
    (device_size, stride_map) tuple.
    """
    captured: dict = {}

    def wrapper(op, output, output_dep, args):
        results = original(op, output, output_dep, args)
        if results:
            committed = (
                tuple(results[0].device_size),
                tuple(results[0].stride_map),
            )
            arg_layouts = {}
            for arg in args:
                stl = next(iter(arg.layouts), None)
                if stl is not None:
                    arg_layouts[arg.dep.name] = (
                        tuple(stl.device_size),
                        tuple(stl.stride_map),
                    )
            captured[op.get_name()] = (committed, arg_layouts)
        return results

    return wrapper, captured


class TestLXInplaceLayout:
    """In-place layout promotion in _multi_arg_pointwise_layouts."""

    def test_add_reuses_matmul_input_layout(self):
        """The add in matmul(Q*scale, K.T) + mask reuses the matmul layout.

        The transpose feeding the matmul makes the add's natural (rebuilt)
        layout diverge from its matmul-output input. With the in-place block
        the input's layout is promoted to the front and committed, so the
        add's committed layout equals its non-broadcast same-frame input's
        layout — enabling in-place reuse without a restickify.
        """
        B, H, S, D = 1, 4, 128, 64
        scale = 1.0 / math.sqrt(math.sqrt(D))

        def spyre_fn(q, k, mask):
            with spyre_hint(work_div={"H": H, "S": S // 4}):
                scores = torch.matmul(q * scale, k.transpose(-1, -2))
                return scores + mask

        q = torch.randn(B, H, S, D, dtype=torch.float16).to(DEVICE)
        k = torch.randn(B, H, S, D, dtype=torch.float16).to(DEVICE)
        mask = torch.zeros(1, 1, S, S, dtype=torch.float16).to(DEVICE)

        wrapper, captured = _capture_multi_arg_layouts(
            propagate_layouts._multi_arg_pointwise_layouts
        )

        with (
            config.patch({"lx_planning": True, "allow_all_ops_in_lx_planning": True}),
            inductor_config.patch({"fx_graph_cache": False}),
            mock_patch.object(
                propagate_layouts, "_multi_arg_pointwise_layouts", wrapper
            ),
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            torch._dynamo.reset()
            cfn = torch.compile(spyre_fn, backend="inductor")
            cfn(q, k, mask)

        assert captured, "no multi-arg pointwise op was lowered"

        # Find the add op: it has a non-broadcast, same-device-footprint input
        # (the matmul output) plus the broadcast mask. Its committed layout
        # must equal that input's layout.
        matched = False
        for _name, (committed, arg_layouts) in captured.items():
            for _arg_name, arg_layout in arg_layouts.items():
                same_footprint = math.prod(
                    [s for s in arg_layout[0] if s > 0]
                ) == math.prod([s for s in committed[0] if s > 0])
                if same_footprint and arg_layout == committed:
                    matched = True
                    break
            if matched:
                break

        assert matched, (
            "Expected an add op whose committed layout equals a same-frame "
            "input's layout (in-place promotion). Captured layouts:\n"
            + "\n".join(
                f"  {n}: committed={c} args={a}" for n, (c, a) in captured.items()
            )
        )
