"""
Upstream PyTorch decorator patchers for the Spyre test framework.

These patchers modify upstream PyTorch test decorators at instantiation time
to allow Spyre's privateuse1 backend to run tests that would otherwise be
restricted to specific devices or dtypes.

Each patcher follows the same pattern:
  1. Receive the test method as passed to instantiate_test()
  2. Locate the upstream decorator instance (in the closure or on the function)
  3. Mutate its configuration in-place so the decorator allows privateuse1

PyTorch decorators like @onlyOn and @ops read their configuration at call
time, not at decoration time. So mutating the decorator instance after
decoration but before the test runs is sufficient.

PyTorch deepcopies the test method before calling instantiate_test(), so
each call has its own fresh decorator instances. A one-time global patch
would not affect these copies.
"""

from typing import Set, Optional
import torch


class _OOTOnlyOnPatcher:
    """Patches @onlyOn decorated test methods to also allow privateuse1.

    The already-produced only_fn wrapper closes over the onlyOn instance.
    self.device_type is read at call time, so mutating the instance's
    device_type list after decoration still takes effect.
    """

    _PRIVATEUSE1: str

    def __init__(self, test: object, privateuse1_device_type: str) -> None:
        self._PRIVATEUSE1 = privateuse1_device_type

        # Unwrap bound method to get the underlying function object.
        # Test methods passed to instantiate_test() are bound to their class,
        # so __func__ gives us the raw function whose closure we need to walk.
        self._underlying_fn = (
            test.__func__  # type: ignore[union-attr]
            if hasattr(test, "__func__")
            else test
        )

    def patch(self) -> None:
        """Walk the decorator stack and mutate the onlyOn instance in-place.

        Decorator stacking means @onlyOn may not be the outermost wrapper --
        @suppress_warnings, @skipCUDAIfNotRocm, and @ops are all stacked on
        top of it. We walk the __wrapped__ chain (set by @wraps on each layer)
        until we find a closure cell that holds an onlyOn instance.

        Once found, we append our device name to onlyOn.device_type in-place.
        Because the wrapper reads self.device_type at call time (not at
        decoration time), this update takes effect when the test runs.
        """

        from torch.testing._internal.common_device_type import onlyOn as _onlyOn_cls

        current = self._underlying_fn
        while current is not None:
            # Inspect every cell in this function's closure.
            # Each decorator layer may close over different objects --
            # here we are looking specifically for an onlyOn instance.
            cells = getattr(current, "__closure__", None) or ()
            for cell in cells:
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue

                if not isinstance(val, _onlyOn_cls):
                    # This cell holds something else (e.g. the wrapped function,
                    # a string, or another decorator instance), so continue
                    continue

                # Found the onlyOn instance. Its device_type attribute is what
                # the wrapper checks: `if slf.device_type not in self.device_type`.
                # Update in-place to include our backend name.
                if isinstance(val.device_type, list):
                    if self._PRIVATEUSE1 not in val.device_type:
                        val.device_type.append(self._PRIVATEUSE1)

                # Less common scenario: @onlyOn("cuda") -- single string.
                # Replace with a list containing both the original and ours.
                elif isinstance(val.device_type, str):
                    if val.device_type != self._PRIVATEUSE1:
                        val.device_type = [val.device_type, self._PRIVATEUSE1]
                return

            # This layer had no onlyOn instance in its closure.
            # Move one level deeper via __wrapped__, which @wraps sets
            # to point to the function this decorator wraps.
            current = getattr(current, "__wrapped__", None)

        # If we reach here, that means no @onlyOn was found in the decorator stack.
        # That implies that the test simply did not have @onlyOn.


# ---------------------------------------------------------------------------
# Dtype patcher
# ---------------------------------------------------------------------------


class _OOTDtypePatcher:
    """Patches @ops allowed_dtypes on a bound test method before instantiation.

    Needed because upstream @ops(..., allowed_dtypes=(...)) restricts which dtype
    variants are generated -- dtypes absent here are never instantiated, so they
    cannot be added to the allow_list. We inject extra dtypes before
    super().instantiate_test() calls _parametrize_test.

    Example: if upstream has @ops(binary_ufuncs, allowed_dtypes=(float32,))
        and we want to test float16, the variant
        test_scalar_support_add_privateuse1_float16 is never created unless we
        inject float16 before @ops runs.
    """

    def __init__(self, test, extra_dtypes: set):
        from torch.testing._internal.common_device_type import ops as _ops_cls

        # @ops instance lives at test.__func__.parametrize_fn.__self__
        # Unwrap bound method to access the underlying function.
        # instantiate_test() receives a bound method, so __func__ gives us
        # the raw function object that carries the parametrize_fn attribute.
        underlying_fn = test.__func__ if hasattr(test, "__func__") else test
        p = getattr(underlying_fn, "parametrize_fn", None)

        # Locate the @ops instance.
        # When @ops decorates a test method it attaches a parametrize_fn
        # attribute to the function. parametrize_fn is a bound method of
        # the ops instance, so parametrize_fn.__self__ is the ops instance
        # itself.
        self._ops_instance = (
            p.__self__
            if p is not None
            and hasattr(p, "__self__")
            and isinstance(p.__self__, _ops_cls)
            else None
        )
        self._extra_dtypes = extra_dtypes

    def patch(self) -> None:
        if (
            self._ops_instance is not None
            and self._ops_instance.allowed_dtypes is not None
        ):
            self._ops_instance.allowed_dtypes |= self._extra_dtypes


class _OOTOpListPatcher:
    """Filters @ops.op_list to supported_ops before super().instantiate_test() runs.

    @ops stores its op list as self.op_list = list(op_list) at decoration
    time — a brand new list copied from whatever was passed in. After that,
    mutating the original binary_ufuncs / ops_and_refs lists has no effect
    on self.op_list.

    Access the @ops instance directly via test.__func__.parametrize_fn.__self__
    (the same path _SpyreDtypePatcher uses for allowed_dtypes) and filter
    self.op_list in-place to keep only supported ops.
    """

    def __init__(self, test: object, supported_ops: Set[str]) -> None:
        from torch.testing._internal.common_device_type import ops as _ops_cls

        # Locate the @ops instance via parametrize_fn.__self__
        underlying_fn = (
            test.__func__  # type: ignore[union-attr]
            if hasattr(test, "__func__")
            else test
        )
        p = getattr(underlying_fn, "parametrize_fn", None)
        self._ops_instance = (
            p.__self__
            if p is not None
            and hasattr(p, "__self__")
            and isinstance(p.__self__, _ops_cls)
            else None
        )
        self._supported_ops = supported_ops

    def patch(self) -> None:
        """Filter op_list in-place to keep only supported ops.

        Uses [:] mutation so the list object identity is preserved, though
        in this case identity doesn't matter — what matters is that we modify
        self.op_list before _parametrize_test iterates it.

        If filtering would produce an empty list, we skip the filtering entirely
        and leave op_list untouched. An empty op_list causes @ops to raise
        ValueError at collection time -- it is better to let the variants be
        generated and have _should_run skip them at instantiation time instead.

        This can happen when a test uses a pre-filtered op list that has no
        intersection with supported_ops -- e.g. test_compare_cpu uses
        _ops_and_refs_with_no_numpy_ref which only contains ops where ref is None,
        but add/mul/sub all have refs so the intersection is empty.
        """
        if self._ops_instance is None:
            return

        filtered = [
            op for op in self._ops_instance.op_list if op.name in self._supported_ops
        ]

        if not filtered:
            # Filtering would empty the list -- leave it untouched and let
            # _should_run handle skipping at instantiation time instead.
            # This avoids the ValueError @ops raises on an empty op_list.
            return

        self._ops_instance.op_list[:] = filtered


class _OOTOpDtypeExpander:
    """Expands op.dtypes on each OpInfo in @ops.op_list to include extra dtypes.

    _parametrize_test computes test variants as:
    dtypes = set(op.supported_dtypes(device_type))  # reads op.__dict__["dtypes"]
    if self.allowed_dtypes is not None:
        dtypes = dtypes.intersection(self.allowed_dtypes)

    If apply_op_config_overrides narrowed op.__dict__["dtypes"] to only
    global.supported_dtypes, a dtype in edits.dtypes.include won't survive
    this intersection even if _SpyreDtypePatcher added it to allowed_dtypes.

    Expand op.__dict__["dtypes"] directly on each OpInfo in @ops.op_list
    to include the extra dtypes before super().instantiate_test() runs.
    Writes to __dict__ directly to bypass OpInfo.__setattr__ validation.

    _SpyreDtypePatcher handles @ops.allowed_dtypes (the outer filter).
    _SpyreOpDtypeExpander handles op.dtypes on each OpInfo (the inner filter).
    Both must be patched for a variant to be generated.

    edits.dtypes.include is intentionally NOT bounded by global.supported_dtypes.
    A user may want to test a single dtype on a specific test without adding
    it globally (which would apply it to all tests).
    """

    def __init__(self, test: object, extra_dtypes: Set[torch.dtype]) -> None:
        from torch.testing._internal.common_device_type import ops as _ops_cls

        underlying_fn = (
            test.__func__  # type: ignore[union-attr]
            if hasattr(test, "__func__")
            else test
        )
        p = getattr(underlying_fn, "parametrize_fn", None)
        self._ops_instance = (
            p.__self__
            if p is not None
            and hasattr(p, "__self__")
            and isinstance(p.__self__, _ops_cls)
            else None
        )
        self._extra_dtypes = extra_dtypes

    def patch(self) -> None:
        if self._ops_instance is None:
            return

        for op_info in self._ops_instance.op_list:
            current = op_info.__dict__.get("dtypes")
            if current is not None:
                # op.dtypes was overridden as a frozenset by apply_op_config_overrides.
                # Expand it to include the extra dtypes from edits.dtypes.include.
                op_info.__dict__["dtypes"] = current | self._extra_dtypes
            # If current is None, op.dtypes was not overridden and already
            # contains all upstream dtypes — no expansion needed.

            # Also expand dtypesIfPrivateUse1 for the same reason —
            # _parametrize_test reads supported_dtypes("privateuse1") which
            # checks dtypesIfPrivateUse1 first.
            current_pu1 = op_info.__dict__.get("dtypesIfPrivateUse1")
            if current_pu1 is not None:
                op_info.__dict__["dtypesIfPrivateUse1"] = (
                    current_pu1 | self._extra_dtypes
                )
            elif current is not None:
                # dtypesIfPrivateUse1 was not set but dtypes was — initialize it
                # from the already-expanded dtypes so privateuse1 path sees it too
                op_info.__dict__["dtypesIfPrivateUse1"] = op_info.__dict__["dtypes"]


class _OOTModuleListPatcher:
    """Filters @modules.module_info_list to supported_modules before
    super().instantiate_test() runs.

    Mirrors the behaviour of _OOTOpListPatcher exactly - include injects modules from
    module_db that are not already present, exclude removes them, and
    if the result would be empty the list is left untouched.

    Step 1: inject edits.modules.include (additive -- not bounded by
            global.supported_modules, same semantics as edits.ops.include).
    Step 2: filter to global.supported_modules if specified.
    Step 3: apply edits.modules.exclude
    """

    def __init__(
        self,
        test: object,
        supported_modules: Optional[Set[str]],
        included_modules: Optional[Set[str]] = None,
        excluded_modules: Optional[Set[str]] = None,
    ) -> None:
        from torch.testing._internal.common_modules import modules as _modules_cls

        underlying_fn = test.__func__ if hasattr(test, "__func__") else test
        p = getattr(underlying_fn, "parametrize_fn", None)
        self._modules_instance = (
            p.__self__
            if p is not None
            and hasattr(p, "__self__")
            and isinstance(p.__self__, _modules_cls)
            else None
        )
        self._supported_modules = supported_modules
        self._included_modules: Set[str] = included_modules or set()
        self._excluded_modules: Set[str] = excluded_modules or set()

    def patch(self) -> None:
        if self._modules_instance is None:
            return

        from torch.testing._internal.common_modules import module_db

        module_db_by_name = {m.name: m for m in module_db}

        # inject edits.modules.include
        if self._included_modules:
            existing_names = {m.name for m in self._modules_instance.module_info_list}
            for name in self._included_modules:
                # YAML may use "torch.nn.X" but module_db keys use "nn.X"
                short_name = name.removeprefix("torch.")
                mod_info = module_db_by_name.get(short_name) or module_db_by_name.get(
                    name
                )
                if mod_info is not None and mod_info.name not in existing_names:
                    self._modules_instance.module_info_list.append(mod_info)

        # filter to global.supported_modules
        # included_modules will exist even if not in supported_modules
        if self._supported_modules is not None:
            filtered = [
                m
                for m in self._modules_instance.module_info_list
                if m.name in self._supported_modules
                or m.name in self._included_modules
                or f"torch.{m.name}" in self._supported_modules
                or f"torch.{m.name}" in self._included_modules
            ]
            if filtered:
                self._modules_instance.module_info_list[:] = filtered

        # apply edits.modules.exclude
        if self._excluded_modules:
            filtered = [
                m
                for m in self._modules_instance.module_info_list
                if m.name not in self._excluded_modules
                and f"torch.{m.name}" not in self._excluded_modules  # ← add
            ]
            if filtered:
                self._modules_instance.module_info_list[:] = filtered


class _OOTModuleDtypePatcher:
    """Patches @modules.allowed_dtypes to inject extra dtypes.

    Mirrors the behaviour of _OOTDtypePatcher exactly but targets the @modules decorator.
    Needed when edits.dtypes.include adds dtypes that are not in
    @modules.allowed_dtypes - without this, _parametrize_test's
    intersection would filter them out before variants are generated.
    """

    def __init__(self, test: object, extra_dtypes: Set[torch.dtype]) -> None:
        from torch.testing._internal.common_modules import modules as _modules_cls

        underlying_fn = test.__func__ if hasattr(test, "__func__") else test
        p = getattr(underlying_fn, "parametrize_fn", None)
        self._modules_instance = (
            p.__self__
            if p is not None
            and hasattr(p, "__self__")
            and isinstance(p.__self__, _modules_cls)
            else None
        )
        self._extra_dtypes = extra_dtypes

    def patch(self) -> None:
        if (
            self._modules_instance is not None
            and self._modules_instance.allowed_dtypes is not None
        ):
            self._modules_instance.allowed_dtypes |= self._extra_dtypes


class _OOTPrecisionOverridePatcher:
    """Injects dtype-level precision overrides into fn.precision_overrides and
    fn.tolerance_overrides before super().instantiate_test() runs.

    Uses two upstream mechanisms that instantiate_test reads automatically:
      - fn.precision_overrides: {dtype -> atol}          (@precisionOverride)
      - fn.tolerance_overrides: {dtype -> tol(atol,rtol)} (@toleranceOverride)

    When only atol is specified, precision_overrides is used.
    When rtol is specified (with or without atol), tolerance_overrides is used
    since it is the only upstream mechanism that carries rtol.

    """

    def __init__(
        self,
        test: object,
        global_dtype_precision: dict,  # {torch.dtype -> Precision}
        include_dtype_precision: dict,  # {torch.dtype -> Precision}
    ) -> None:
        self._underlying_fn = test.__func__ if hasattr(test, "__func__") else test
        self._global_dtype_precision = global_dtype_precision
        self._include_dtype_precision = include_dtype_precision

    def patch(self) -> None:
        if not self._global_dtype_precision and not self._include_dtype_precision:
            return

        # Merge: global first (lower priority), include overrides
        merged: dict = {}
        for dtype, prec in self._global_dtype_precision.items():
            if prec is not None:
                merged[dtype] = prec
        for dtype, prec in self._include_dtype_precision.items():
            if prec is not None:
                merged[dtype] = prec

        if not merged:
            return

        try:
            from torch.testing._internal.common_device_type import tol as _tol

            has_tol = True
        except ImportError:
            has_tol = False

        for dtype, prec in merged.items():
            atol = prec.atol
            rtol = prec.rtol

            if rtol is not None and has_tol:
                # rtol specified - use tolerance_overrides (carries both atol+rtol)
                # tolerance_overrides takes precedence over precision_overrides
                # in upstream instantiate_test.
                if not hasattr(self._underlying_fn, "tolerance_overrides"):
                    self._underlying_fn.tolerance_overrides = {}
                self._underlying_fn.tolerance_overrides[dtype] = _tol(
                    atol=atol if atol is not None else 0.0,
                    rtol=rtol,
                )
            elif atol is not None:
                # atol only - use precision_overrides (simpler, matches @precisionOverride)
                if not hasattr(self._underlying_fn, "precision_overrides"):
                    self._underlying_fn.precision_overrides = {}
                # setdefault for global, direct assign for include (already merged above
                # so just assign - include already won priority during merge)
                self._underlying_fn.precision_overrides[dtype] = atol


class _OOTOpMarkerPatcher:
    """Patches @ops._parametrize_test to attach pytest markers directly on
    each test_wrapper as it is yielded, before super().instantiate_test()
    installs it as a class method.

    """

    def __init__(self, test: object) -> None:
        from torch.testing._internal.common_device_type import ops as _ops_cls

        underlying_fn = test.__func__ if hasattr(test, "__func__") else test
        p = getattr(underlying_fn, "parametrize_fn", None)
        self._ops_instance = (
            p.__self__
            if p is not None
            and hasattr(p, "__self__")
            and isinstance(p.__self__, _ops_cls)
            else None
        )
        self._underlying_fn = underlying_fn

    def patch(self) -> None:
        if self._ops_instance is None:
            return

        import pytest
        import regex as re

        original_parametrize_fn = self._underlying_fn.parametrize_fn

        def patched_parametrize_fn(test, generic_cls, device_cls):
            for (
                test_wrapper,
                test_name,
                param_kwargs,
                decorator_fn,
            ) in original_parametrize_fn(test, generic_cls, device_cls):
                op = param_kwargs.get("op")
                dtype = param_kwargs.get("dtype")

                if op is not None:
                    op_safe = re.sub(r"[^a-zA-Z0-9_]", "_", op.name).strip("_")
                    op_safe = re.sub(r"__\d+$", "", op_safe).strip("_")
                    if op_safe:
                        test_wrapper = pytest.mark.__getattr__(f"op__{op_safe}")(
                            test_wrapper
                        )

                if dtype is not None:
                    dtype_safe = re.sub(
                        r"[^a-zA-Z0-9_]", "_", str(dtype).replace("torch.", "")
                    ).strip("_")
                    dtype_safe = re.sub(r"__\d+$", "", dtype_safe).strip("_")
                    if dtype_safe:
                        test_wrapper = pytest.mark.__getattr__(f"dtype__{dtype_safe}")(
                            test_wrapper
                        )

                yield test_wrapper, test_name, param_kwargs, decorator_fn

        # Replacing on the function object itself because this is what the upstream reads.
        self._underlying_fn.parametrize_fn = patched_parametrize_fn


class _OOTModuleMarkerPatcher:
    """Patches @modules.parametrize_fn to attach pytest markers directly on
    each test_wrapper as it is yielded, before super().instantiate_test()
    installs it as a class method.

    Follows _OOTOpMarkerPatcher exactly but targets the @modules decorator.
    Attaches module name and dtype as pytest markers so tests can be filtered
    with -m "nn_Linear" or -m "float16".
    """

    def __init__(self, test: object) -> None:
        from torch.testing._internal.common_modules import modules as _modules_cls

        self._test = test  # store original test object
        underlying_fn = test.__func__ if hasattr(test, "__func__") else test
        p = getattr(underlying_fn, "parametrize_fn", None)
        self._modules_instance = (
            p.__self__
            if p is not None
            and hasattr(p, "__self__")
            and isinstance(p.__self__, _modules_cls)
            else None
        )
        self._underlying_fn = underlying_fn

    def patch(self) -> None:
        if self._modules_instance is None:
            return

        import pytest
        import regex as re

        original_parametrize_fn = self._underlying_fn.parametrize_fn

        def patched_parametrize_fn(test, generic_cls, device_cls):
            for (
                test_wrapper,
                test_name,
                param_kwargs,
                decorator_fn,
            ) in original_parametrize_fn(test, generic_cls, device_cls):
                module_info = param_kwargs.get("module_info")
                dtype = param_kwargs.get("dtype")

                if module_info is not None:
                    module_safe = re.sub(r"[^a-zA-Z0-9_]", "_", module_info.name).strip(
                        "_"
                    )
                    module_safe = re.sub(r"__\d+$", "", module_safe).strip("_")
                    if module_safe:
                        mark = pytest.mark.__getattr__(f"module__{module_safe}")
                        # Set directly on pytestmark list so @wraps copies it
                        if not hasattr(test_wrapper, "pytestmark"):
                            test_wrapper.pytestmark = []
                        test_wrapper.pytestmark = list(test_wrapper.pytestmark) + [mark]

                if dtype is not None:
                    dtype_safe = re.sub(
                        r"[^a-zA-Z0-9_]", "_", str(dtype).replace("torch.", "")
                    ).strip("_")
                    dtype_safe = re.sub(r"__\d+$", "", dtype_safe).strip("_")
                    if dtype_safe:
                        mark = pytest.mark.__getattr__(f"dtype__{dtype_safe}")
                        if not hasattr(test_wrapper, "pytestmark"):
                            test_wrapper.pytestmark = []
                        test_wrapper.pytestmark = list(test_wrapper.pytestmark) + [mark]

                yield test_wrapper, test_name, param_kwargs, decorator_fn

        self._underlying_fn.parametrize_fn = patched_parametrize_fn
        # Also set directly on the test object in case getattr(test, ...)
        # resolves differently from getattr(test.__func__, ...)
        try:
            self._test.parametrize_fn = patched_parametrize_fn  # type: ignore[attr-defined]
        except AttributeError:
            pass
