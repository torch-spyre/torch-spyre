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

from typing import Optional
import torch
from torch._subclasses.fake_tensor import (
    FakeTensorMode,
    FakeTensor,
    FakeTensorConverter,
)
from torch._inductor.ir import FlexibleLayout, Layout, significant_strides_equal
from torch._subclasses.fake_impls import fast_detach
from .stickify import tensor_get_spyre_layout, FixedTiledLayout
from torch_spyre._C import SpyreTensorLayout

orig_from_real_tensor = FakeTensorConverter.from_real_tensor
orig_from_meta_and_device = FakeTensorConverter.from_meta_and_device


def install_spyre_tensors():
    """Extend Tensor and IR Classes for Spyre Stickification"""
    torch.Tensor.get_spyre_layout = tensor_get_spyre_layout
    FakeTensorConverter.from_meta_and_device = spyre_ftc_from_meta_and_device
    FakeTensorConverter.from_real_tensor = spyre_ftc_from_real_tensor
    torch._functorch._aot_autograd.dispatch_and_compile_graph._detach_and_copy_item_memo = spyre_detach_and_copy_item_memo
    torch.fx.experimental.proxy_tensor.snapshot_fake = spyre_snapshot_fake
    torch.fx.passes.fake_tensor_prop.snapshot_fake = spyre_snapshot_fake
    """
    torch._inductor.ir.Buffer.get_layout = spyre_get_layout
    torch._inductor.ir.Buffer.freeze_layout_with_exact_strides = (
        spyre_freeze_layout_with_exact_strides
    )
    """


def spyre_ftc_from_real_tensor(
    self,
    fake_mode,
    t: torch.Tensor,
    make_constant: bool = False,
    shape_env=None,
    *,
    source=None,
    symbolic_context=None,
    trace: bool = True,
) -> FakeTensor:
    res: FakeTensor = orig_from_real_tensor(
        self,
        fake_mode,
        t,
        make_constant=make_constant,
        shape_env=shape_env,
        source=source,
        symbolic_context=symbolic_context,
        trace=trace,
    )
    if t.device.type == "spyre":
        # TODO: Extract SpyreTensorLayout from SpyreTensorImpl (once torch_spyre stores it).
        #       For initial development, synthesize one that encodes generic stick layout.
        res.spyre_layout = SpyreTensorLayout(res.shape, res.dtype)
    return res


def spyre_ftc_from_meta_and_device(
    self, fake_mode: FakeTensorMode, t: torch.Tensor, device: torch.device
) -> FakeTensor:
    res = orig_from_meta_and_device(self, fake_mode, t, device)
    if hasattr(t, "spyre_layout"):
        res.spyre_layout = t.get_spyre_layout()
    return res


def spyre_snapshot_fake(val: torch.Tensor) -> Optional[torch.Tensor]:
    if isinstance(val, FakeTensor):
        res = fast_detach(val.fake_mode, val)
    else:
        res = val.detach()
    # Propagate SpyreDCI to detached copy of val
    if res is not None and hasattr(val, "spyre_layout"):
        res.spyre_layout = val.spyre_layout

    return res


def spyre_detach_and_copy_item_memo(t):
    detached_t = t.detach()
    if hasattr(t, "item_memo"):
        detached_t.item_memo = t.item_memo
    if hasattr(t, "spyre_layout"):
        detached_t.spyre_layout = t.spyre_layout
    return detached_t


def spyre_get_layout(self: torch._inductor.ir.Buffer) -> Layout:
    if isinstance(self.layout, FlexibleLayout):
        for n in self.origins:
            t = n.meta.get("val", None)
            if isinstance(t, torch.Tensor):
                if t.device.type == "spyre":
                    layout = t.get_spyre_layout()
                    if isinstance(layout, SpyreTensorLayout):
                        self.layout = layout.spyre_fixed_layout(
                            t.device, t.size(), t.dtype
                        )
                        return self.layout
            elif isinstance(t, tuple) and (
                n.target == torch.ops.aten.max.dim or n.target == torch.ops.aten.min.dim
            ):
                # TODO: This only works because Spyre implements amax/amin and doesn't implement argmax/argmin
                t = t[0]
                if t.device.type == "spyre":
                    layout = t.get_spyre_layout()
                    if isinstance(layout, SpyreTensorLayout):
                        self.layout = layout.spyre_fixed_layout(
                            t.device, t.size(), t.dtype
                        )
                        return self.layout

        return self.layout
    elif isinstance(self.layout, Layout):
        return self.layout
    raise NotImplementedError(type(self.layout).__name__)


def spyre_freeze_layout_with_exact_strides(  # type: ignore[no-untyped-def]
    self, exact_strides, allow_padding=False
) -> None:
    if isinstance(self.layout, FixedTiledLayout):
        assert significant_strides_equal(
            exact_strides, self.layout.stride, self.layout.size
        )
    else:
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_exact_strides(
            exact_strides, allow_padding=allow_padding
        )
