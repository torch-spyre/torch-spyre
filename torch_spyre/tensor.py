import torch
import torch.utils._pytree as pytree
import os

from ._C import (
    get_spyre_tensor_layout,
    empty_with_layout,
    to_with_layout,
)

aten = torch.ops.aten

_initialized = False


class SpyreTensor(torch.Tensor):
    def __init__(self, t, device_tensor_layout=None):
        orig_layout = None
        while isinstance(t, SpyreTensor):
            orig_layout = orig_layout or t.device_tensor_layout()
            t = t._t

        try:
            orig_layout = get_spyre_tensor_layout(t)
        except RuntimeError:
            pass

        if t.device.type == "spyre":
            if (
                device_tensor_layout is not None
                and orig_layout is not None
                and device_tensor_layout != orig_layout
            ):
                t = to_with_layout(t, device_tensor_layout)
        else:
            if device_tensor_layout is None:
                device_tensor = t.to("spyre")
                device_tensor_layout = device_tensor.device_tensor_layout()
                t = device_tensor._t
            else:
                t = to_with_layout(t, device_tensor_layout)
        assert not isinstance(t, SpyreTensor)
        self._t = t
        self._stl = device_tensor_layout or orig_layout

    @staticmethod
    def __new__(cls, t, device_tensor_layout=None):
        shape = t.shape
        kwargs = {}
        kwargs["strides"] = t.stride()
        kwargs["storage_offset"] = t.storage_offset()
        kwargs["device"] = "spyre"
        kwargs["layout"] = t.layout
        kwargs["requires_grad"] = t.requires_grad
        kwargs["dtype"] = t.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __repr__(self):
        t_repr = repr(self._t)
        if t_repr.startswith("tensor("):
            t_repr = t_repr[len("tensor(") : -1]
        return (
            f"SpyreTensor({t_repr}, device_tensor_layout={self.device_tensor_layout()})"
        )

    def device_tensor_layout(self):
        return self._stl

    def to(self, *args, device_layout=None, **kwargs):
        if (
            device_layout is None
        ):  # use original implementation if no layout is provided
            result = self._t.to(*args, **kwargs)
            if result.device.type == "spyre":
                return SpyreTensor(result)
            else:
                return result
        else:
            return SpyreTensor(to_with_layout(self, device_layout))

    @classmethod
    def empty(
        cls,
        *args,
        device_layout=None,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
        memory_format=torch.contiguous_format,
    ):
        assert device is None or torch.device(device).type == "spyre", (
            f"Device should be None or spyre. Got: {device}"
        )
        if (
            device_layout is None
        ):  # use original implementation if no layout is provided
            return cls(
                torch.empty(
                    *args,
                    out=out,
                    dtype=dtype,
                    layout=layout,
                    device=device or "spyre",
                    requires_grad=requires_grad,
                    pin_memory=pin_memory,
                    memory_format=memory_format,
                )
            )
        else:
            global _initialized
            if not _initialized:
                # TODO: figure out why we need a call here
                _ = torch.randn(2).to("spyre")
                _initialized = True
            # layout_opt is omitted; c10::Layout has no pybind11 type caster,
            # so py_empty_with_layout drops that parameter and always uses
            # the default (Strided).
            return cls(
                empty_with_layout(
                    *args, device_layout, dtype, device, pin_memory, memory_format
                )
            )

    def __tensor_flatten__(self):
        layout = self.device_tensor_layout()
        return ["_t"], {"device_tensor_layout": layout}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, _, __):
        t = inner_tensors["_t"]
        return SpyreTensor(t, meta["device_tensor_layout"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        spyre_tensors = {}

        def unwrap(x):
            if isinstance(x, SpyreTensor):
                spyre_tensors[id(x._t)] = x
                return x._t
            else:
                return x

        def wrap(x):
            if isinstance(x, torch.Tensor) and x.device.type == "spyre":
                found = spyre_tensors.get(id(x))
                if found is not None:
                    return found
                else:
                    return SpyreTensor(x)
            return x

        args_a = pytree.tree_map(unwrap, args)
        kwargs_a = pytree.tree_map(unwrap, kwargs)
        out_a = func(*args_a, **kwargs_a)
        out = pytree.tree_map(wrap, out_a)
        return out


def wrap_spyre_tensor(x):
    if os.environ.get("TORCH_SPYRE_WRAPPER_SUBCLASS", "1") == "0":
        return x
    else:
        return SpyreTensor(x)


def wrap_spyre_tensor_args(*args, **kwargs):
    if os.environ.get("TORCH_SPYRE_WRAPPER_SUBCLASS", "1") == "0":
        return args, kwargs

    def wrap(x):
        return (
            SpyreTensor(x)
            if isinstance(x, torch.Tensor) and not isinstance(x, SpyreTensor)
            else x
        )

    return pytree.tree_map_(wrap, args), pytree.tree_map_(wrap, kwargs)
