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

"""Model recipes: how to build a model and example inputs from a name.

A recipe takes a target (model id, path or ``module:function``) and string
options, and returns a :class:`ModelSpec`. Add a model family by registering a
function::

    @register_recipe("mylib", "Models from mylib (target: model name)")
    def _mylib(target: str, opts: Options) -> ModelSpec:
        model = mylib.build(target)
        return ModelSpec(target, model, (torch.randn(1, 3, 224, 224),))

The model's libraries are imported inside the recipe, so none of them is a
dependency of torch-spyre.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any, Callable, Optional

import torch

from .serialize import dtype_of


@dataclasses.dataclass
class ModelSpec:
    """A model ready to capture."""

    name: str
    model: Any  # nn.Module or callable
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Drives the model (forward compiled) instead of one call with args/kwargs.
    run: Optional[Callable[[Any], Any]] = None
    # What the recipe did to the model, for the report header.
    notes: list[str] = dataclasses.field(default_factory=list)


class Options(dict):
    """``key=value`` recipe options with typed getters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.used: set[str] = set()

    def get_str(self, key: str, default: Optional[str] = None) -> Optional[str]:
        self.used.add(key)
        return self.get(key, default)

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        value = self.get_str(key)
        return default if value is None else int(value)

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self.get_str(key)
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes", "on")

    def dtype(self) -> torch.dtype:
        return dtype_of(self.get_str("dtype", "float16") or "float16")

    def unused(self) -> list[str]:
        return sorted(set(self) - self.used)


Recipe = Callable[[str, Options], ModelSpec]
RECIPES: dict[str, tuple[Recipe, str]] = {}


def register_recipe(name: str, help_text: str) -> Callable[[Recipe], Recipe]:
    def decorator(fn: Recipe) -> Recipe:
        RECIPES[name] = (fn, help_text)
        return fn

    return decorator


def load(recipe: str, target: str, options: dict[str, str]) -> ModelSpec:
    if recipe not in RECIPES:
        raise KeyError(f"unknown recipe {recipe!r}; choose from {sorted(RECIPES)}")
    opts = Options(options)
    spec = RECIPES[recipe][0](target, opts)
    if opts.unused():
        raise ValueError(f"recipe {recipe!r} does not take options {opts.unused()}")
    return spec


def _cast_inputs(value: Any, dtype: torch.dtype) -> Any:
    from torch.utils import _pytree as pytree

    def cast(x: Any) -> Any:
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return x.to(dtype)
        return x

    return pytree.tree_map(cast, value)


def _prepare(model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    return model.eval().to(dtype)


@register_recipe(
    "module",
    "Any model (target: 'package.module:function'). The function takes the "
    "options as keyword arguments and returns a ModelSpec, or (model, args) / "
    "(model, args, kwargs).",
)
def _module(target: str, opts: Options) -> ModelSpec:
    module_name, _, attr = target.partition(":")
    if not attr:
        raise ValueError("target must look like 'package.module:function'")
    factory = getattr(importlib.import_module(module_name), attr)
    opts.used.update(opts)  # every option goes to the factory
    result = factory(**opts)
    if isinstance(result, ModelSpec):
        return result
    model, args, *rest = result
    kwargs = rest[0] if rest else {}
    return ModelSpec(attr, model, tuple(args), dict(kwargs))


@register_recipe(
    "torchvision",
    "torchvision.models (target: model name). Options: image_size=224, batch=1, "
    "dtype=float16.",
)
def _torchvision(target: str, opts: Options) -> ModelSpec:
    import torchvision

    dtype = opts.dtype()
    size = opts.get_int("image_size", 224) or 224
    batch = opts.get_int("batch", 1) or 1
    model = _prepare(torchvision.models.get_model(target, weights=None), dtype)
    return ModelSpec(target, model, (torch.randn(batch, 3, size, size, dtype=dtype),))


@register_recipe(
    "timm",
    "timm models (target: model name). Options: batch=1, dtype=float16.",
)
def _timm(target: str, opts: Options) -> ModelSpec:
    import timm

    dtype = opts.dtype()
    batch = opts.get_int("batch", 1) or 1
    model = _prepare(timm.create_model(target, pretrained=False), dtype)
    size = tuple(model.pretrained_cfg.get("input_size", (3, 224, 224)))
    return ModelSpec(target, model, (torch.randn(batch, *size, dtype=dtype),))


@register_recipe(
    "ultralytics",
    "Ultralytics YOLO (target: model YAML such as yolov5n.yaml or yolo26n.yaml, "
    "or a .pt checkpoint). Options: fuse=true (fold BatchNorm into Conv, as "
    "deployed), end2end=true|false (NMS-free head with top-k post-processing, "
    "YOLO26/YOLOv10), image_size=640, batch=1, dtype=float16.",
)
def _ultralytics(target: str, opts: Options) -> ModelSpec:
    from ultralytics import YOLO

    dtype = opts.dtype()
    size = opts.get_int("image_size", 640) or 640
    batch = opts.get_int("batch", 1) or 1
    model = YOLO(target).model
    notes = []
    end2end = opts.get_str("end2end")
    if end2end is not None:
        head = model.model[-1]
        if not hasattr(head, "end2end"):
            raise ValueError(f"{type(head).__name__} has no end2end mode")
        head.end2end = opts.get_bool("end2end")
        notes.append(f"end2end={head.end2end}")
    if opts.get_bool("fuse", True):
        model = model.fuse(verbose=False)
        notes.append("Conv2d+BatchNorm2d fused with model.fuse()")
    model = _prepare(model, dtype)
    name = target.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return ModelSpec(
        name, model, (torch.randn(batch, 3, size, size, dtype=dtype),), notes=notes
    )


@register_recipe(
    "hf",
    "Hugging Face transformers (target: hub id or local path). Options: "
    "pretrained=false (random weights from the config, no weight download), "
    "layers=N (keep N layers; ops repeat per layer), seq_len=32, batch=1, "
    "mode=forward|generate, max_new_tokens=4, image_size, "
    "attn_implementation=sdpa, trust_remote_code=false, dtype=float16.",
)
def _hf(target: str, opts: Options) -> ModelSpec:
    import transformers

    dtype = opts.dtype()
    trust = opts.get_bool("trust_remote_code", False)
    config = transformers.AutoConfig.from_pretrained(target, trust_remote_code=trust)
    notes = []
    layers = opts.get_int("layers")
    if layers is not None:
        for cfg in (config, getattr(config, "text_config", None)):
            for key in ("num_hidden_layers", "n_layer", "num_layers"):
                if cfg is not None and hasattr(cfg, key):
                    setattr(cfg, key, layers)
        notes.append(f"truncated to {layers} layers")
    attn = opts.get_str("attn_implementation", "sdpa")
    cls = _hf_model_class(config)
    # transformers 4.56 renamed torch_dtype to dtype.
    version = tuple(int(p) for p in transformers.__version__.split(".")[:2])
    dtype_kw = {"dtype" if version >= (4, 56) else "torch_dtype": dtype}
    if opts.get_bool("pretrained", False):
        model = cls.from_pretrained(
            target,
            config=config,
            attn_implementation=attn,
            trust_remote_code=trust,
            **dtype_kw,
        )
    elif cls is transformers.AutoModel:
        model = cls.from_config(
            config, attn_implementation=attn, trust_remote_code=trust, **dtype_kw
        )
        notes.append("random weights (from_config)")
    else:
        model = cls._from_config(config, attn_implementation=attn, **dtype_kw)
        notes.append("random weights (from_config)")
    model = _prepare(model, dtype)

    batch = opts.get_int("batch", 1) or 1
    seq_len = opts.get_int("seq_len", 32) or 32
    kwargs = _hf_inputs(model, config, batch, seq_len, opts.get_int("image_size"))
    kwargs = _cast_inputs(kwargs, dtype)
    run = None
    if opts.get_str("mode", "forward") == "generate":
        max_new = opts.get_int("max_new_tokens", 4) or 4

        def run(m: Any) -> Any:
            return m.generate(**kwargs, max_new_tokens=max_new, do_sample=False)

        notes.append(f"generate, {max_new} new tokens")
    return ModelSpec(target, model, (), kwargs, run=run, notes=notes)


def _hf_model_class(config: Any) -> Any:
    """The config's own architecture class, or AutoModel (base model)."""
    import transformers

    for arch in getattr(config, "architectures", None) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            return cls
    return transformers.AutoModel


def _hf_inputs(
    model: Any, config: Any, batch: int, seq_len: int, image_size: Optional[int]
) -> dict[str, Any]:
    main = getattr(model, "main_input_name", "input_ids")
    if main == "pixel_values":
        vision = getattr(config, "vision_config", config)
        size: Any = image_size or getattr(vision, "image_size", 224)
        if isinstance(size, (list, tuple)):
            height, width = size[0], size[-1]
        elif isinstance(size, dict):
            height, width = size.get("height", 224), size.get("width", 224)
        else:
            height = width = int(size)
        channels = getattr(vision, "num_channels", 3)
        return {"pixel_values": torch.randn(batch, channels, height, width)}
    if main == "input_ids":
        text = getattr(config, "text_config", config)
        vocab = getattr(text, "vocab_size", 1000) or 1000
        input_ids = torch.randint(0, min(vocab, 1000), (batch, seq_len))
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
        }
    dummy = getattr(model, "dummy_inputs", None)
    if dummy:
        return dict(dummy)
    raise ValueError(f"no input recipe for main input {main!r}; use the module recipe")
