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

import os
import threading
import types
import importlib
from .constants import DEVICE_NAME

_runtime_init_lock = threading.Lock()


class _SpyreImpl:
    def __init__(self):
        self._initialized = False
        self._in_bad_fork = False

        # When spawning a supprocess from inductor, ensure that IS_INDUCTOR_SPAWNED_SUBPROCESS=1
        # This will avoid additional initialization when processes are spawned from torch inductor (This happens in Triton pathway)
        # TODO: This may require monkey-patching the method where torch-inductor spawns a subprocess
        if int(os.getenv("IS_INDUCTOR_SPAWNED_SUBPROCESS", "0")):
            # NOTE (tmhoangt): currently, Spyre can't be used by more than one process
            # so, we want only the main process can have access to the actual device
            self._in_bad_fork = True
            self._initialized = True
        try:
            os.register_at_fork(after_in_child=self._mark_after_fork)
        except Exception:
            pass

    def __getattr__(self, name):
        if name == "_C":
            return self.__dict__.get(name, None)
        self._lazy_init()
        return super().__getattribute__(name)

    def _lazy_init(self):
        if self._initialized:
            return
        with _runtime_init_lock:
            if self._initialized:
                return
            # Load the C++ Module
            # put any light, once-per-process setup here
            self._C = importlib.import_module("torch_spyre._C")
            # this will create the allocator
            self._C.start_runtime()
            self._initialized = True

            ## Run patch on import
            from ._monkey_patch import _patch_tensor_for_spyre

            _patch_tensor_for_spyre()

            from torch_spyre._inductor import _autoload as ts_autoload

            ts_autoload()

    def _is_in_bad_fork(self) -> bool:
        return self._in_bad_fork

    def manual_seed(self, seed: int, device: int | None = None) -> None:
        fn = getattr(self._C, "manual_seed", None)
        if fn:
            fn(int(seed), -1 if device is None else int(device))

    def manual_seed_all(self, seed: int) -> None:
        _C = self._C
        if hasattr(_C, "manual_seed_all"):
            _C.manual_seed_all(int(seed))
        else:
            # Otherwise, fan out:
            for idx in range(self.device_count()):
                self.manual_seed(seed, device=idx)

    def is_available(self) -> bool:
        if self._is_in_bad_fork():
            return True
        else:
            return not hasattr(self, "_C") or (
                self._C is not None and getattr(self._C, "is_available", lambda: True)()
            )

    def is_initialized(self):
        return self._initialized and not self._is_in_bad_fork()

    def device_count(self) -> int:
        # TODO(tmhoangt) - invoke the right API to return
        return 1

    def current_device(self) -> int:
        return getattr(self._C, "current_device", lambda: 0)()

    def set_device(self, idx: int) -> None:
        fn = getattr(self._C, "set_device", None)
        if fn:
            fn(int(idx))

    def _mark_after_fork(self):
        self._initialized = True
        self._in_bad_fork = True


def make_spyre_module() -> types.ModuleType:
    """Return a real module object backed by a single _SpyreImpl instance."""
    impl = _SpyreImpl()

    mod = types.ModuleType(DEVICE_NAME)
    mod.__doc__ = "Spyre backend module (wrapped around a stateful implementation)."

    # Expose bound methods directly — they look like plain functions on the module.
    # These are *bound* to `impl`, so `self` is already captured.
    mod._is_in_bad_fork = lambda: impl._is_in_bad_fork()
    mod.manual_seed = lambda s: impl.manual_seed(s)
    mod.manual_seed_all = lambda s: impl.manual_seed_all(s)
    mod.is_available = lambda: impl.is_available()
    mod.is_initialized = lambda: impl.is_initialized()
    mod.device_count = lambda: impl.device_count()
    mod.current_device = lambda: impl.current_device()
    mod.set_device = lambda idx: impl.set_device(idx)
    mod._is_compiled = lambda: True

    # Optional: forward unknown attrs to the impl or _C for convenience
    def __getattr__(name):
        if name in ["__file__", "_C"]:
            # Important: raising AttributeError ensures hasattr() returns False
            # without triggering our lazy loader.
            raise AttributeError(name)
        if hasattr(impl, name):
            return getattr(impl, name)
        if not hasattr(impl, "_C"):
            impl._lazy_init()
        if hasattr(impl._C, name):
            return getattr(impl._C, name)
        raise AttributeError(name)

    mod.__getattr__ = __getattr__
    _OPTIONAL_HOOKS = {
        "Scheduling",
        "GraphLowering",
        "Lowering",
        "Codegen",
        "Compile",
    }
    for _name in _OPTIONAL_HOOKS:
        setattr(mod, _name, None)

    # Keep a hidden handle to the impl (handy for tests/debugging)
    mod._impl = impl

    return mod


def _autoload():
    # guard if autoload may run more than once
    if getattr(_autoload, "_ran", False):
        return
    _autoload._ran = True

    import torch  # noqa: E402
    from . import _hooks  # noqa: F401

    # Set all the appropriate state on PyTorch
    torch.utils.rename_privateuse1_backend(DEVICE_NAME)
    torch._register_device_module(DEVICE_NAME, make_spyre_module())
    import torch_spyre.codegen_ops  # noqa: F401
    from torch_spyre._inductor import _light_autoload

    _light_autoload()

    # Set correct state for dynamo to support eager ops
    import torch._dynamo.config

    # Increasing the default number of graphs from 8 to 1024
    # to have enough cache space for all eager ops
    # You'll get recursion errors if this is exceeded
    torch._dynamo.config.cache_size_limit = 1024

    # set the default backend debugging to quiet
    # enable these if you would like to see runtime/compiler logging
    os.environ.setdefault("TORCH_SENDNN_LOG", "CRITICAL")
    os.environ.setdefault("DT_DEEPRT_VERBOSE", "-1")
    os.environ.setdefault("DTLOG_LEVEL", "error")
