"""
Shared class and methods for all OOT PyTorch test overrides.

"""

import os
import regex as re
from typing import Dict, List, Optional, Set

import pytest  # type: ignore
import torch

from spyre_test_constants import (
    DEFAULT_FLOATING_PRECISION,
    ENV_TEST_CONFIG,
    MODE_MANDATORY_SUCCESS,
    MODE_SKIP,
    MODE_XFAIL,
    MODE_XFAIL_STRICT,
    UNLISTED_MODE_XFAIL,
)
from spyre_test_matching import (
    extract_dtype_from_name,
    parse_dtype,
)
from spyre_test_parsing import (
    FileEntry,
    apply_op_config_overrides,
    load_yaml_config,
    resolve_current_file,
)

from spyre_upstream_patcher import (
    _OOTDtypePatcher,
    _OOTOnlyOnPatcher,
    _OOTOpDtypeExpander,
    _OOTOpListPatcher,
    _OOTModuleListPatcher,
    _OOTModuleDtypePatcher,
    _OOTPrecisionOverridePatcher,
)
from spyre_test_config_models import (
    OOTTestConfig,
    Precision,
    SupportedOpConfig,
    SupportedModuleConfig,
    TestEntry,
)


# Resolve the actual backend name registered for privateuse1.
# torch._C._get_privateuse1_backend_name() returns e.g. "spyre".
# This is what slf.device_type will be at test runtime.
def _get_privateuse1_device_type() -> str:
    try:
        return torch._C._get_privateuse1_backend_name()
    except Exception:
        return "privateuse1"  # fallback if not registered yet


_SPYRE_DEVICE_TYPE: str = _get_privateuse1_device_type()


# ---------------------------------------------------------------------------
# PrivateUse1TestBase filter
# ---------------------------------------------------------------------------
# TODO: figure out why this filter is needed - expected to use default PrivateUse1TestBase
def remove_builtin_privateuse1_test_base():
    """
    Remove built-in PrivateUse1TestBase from device_type_test_bases.

    This ensures only TorchTestBase handles the privateuse1 device type,
    preventing nondeterministic overwrites when list(set(...)) randomizes order.

    Side effect: Modifies the global device_type_test_bases list in-place.

    TODO: investigate whether this filter will still be needed once the upstream
          PrivateUse1TestBase correctly defers to registered custom backends.
    """
    device_type_test_bases[:] = [  # type: ignore[name-defined] # noqa: F821
        b
        for b in device_type_test_bases  # type: ignore[name-defined] # noqa: F821
        if b is not PrivateUse1TestBase  # type: ignore[name-defined] # noqa: F821
    ]


# Call the filter function to apply the side effect
remove_builtin_privateuse1_test_base()


def _build_test_entry_map(file_entry: FileEntry) -> Dict[str, TestEntry]:
    """Build {method_name -> TestEntry} from file_entry.tests.

    A single TestEntry can cover multiple test ids via name: [list].
    Each method_name in the list gets its own entry in the map pointing
    to the same TestEntry object so _should_run() can look up by method_name.
    """
    result: Dict[str, TestEntry] = {}
    for entry in file_entry.tests:
        for method_name in entry.method_names():
            if method_name in result:
                import warnings

                warnings.warn(
                    f"test method {method_name!r} appears in multiple TestEntry "
                    f"blocks in the YAML. The last entry will take precedence.",
                    stacklevel=2,
                )
            result[method_name] = entry
    return result


def _extract_op_name_from_method(
    method_name: str, base_test_name: str
) -> Optional[str]:
    """Extract the op name from a parametrized method name.

    method_name: test_scalar_support_add_spyre_float16
    base_test_name: test_scalar_support
    returns: "add"

    Returns None if the op name cannot be determined.
    """
    if not method_name.startswith(base_test_name + "_"):
        return None
    remainder = method_name[len(base_test_name) + 1 :]  # "add_spyre_float16"
    # op name is the first segment before the device suffix
    device_type = "spyre"  # or read from _SPYRE_DEVICE_TYPE
    if f"_{device_type}_" in remainder:
        return remainder.split(f"_{device_type}_")[0]  # "add"
    return None


# ---------------------------------------------------------------------------
# TorchTestBase
# ---------------------------------------------------------------------------


# PrivateUse1TestBase injected via globals() by runpy
class TorchTestBase(PrivateUse1TestBase):  # type: ignore[name-defined]  # noqa: F821
    """Base class for OOT Device PyTorch test overrides.

    All configuration is loaded lazily from the YAML file pointed to by
    PYTORCH_TEST_CONFIG.  The YAML is validated by Pydantic on load.
    See spyre_test_config_schema.json for the full schema.
    """

    device_type: str = "privateuse1"
    precision: float = DEFAULT_FLOATING_PRECISION

    TEST_ENTRIES: Dict[str, "TestEntry"] = {}  # {method_name -> TestEntry}
    UNLISTED_TEST_MODE: str = UNLISTED_MODE_XFAIL  # file-level default
    SUPPORTED_OPS_CONFIG: Dict[str, "SupportedOpConfig"] = {}  # {op_name -> config}
    SUPPORTED_MODULES_CONFIG: Dict[
        str, "SupportedModuleConfig"
    ] = {}  # {module_name -> config}
    GLOBAL_SUPPORTED_DTYPES: Optional[Set[torch.dtype]] = None  # None = no filtering
    GLOBAL_DTYPE_PRECISION: Dict[torch.dtype, "Precision"] = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # PrivateUse1TestBase.setUpClass sets cls.device_type = "spyre"
        # (the registered backend name). This mutates the base class's
        # device_type, causing subsequent instantiate_device_type_tests calls
        # to generate class names like TestOldViewOpsSPYRE instead of
        # TestOldViewOpsPRIVATEUSE1, which then get filtered out by
        # PYTORCH_TESTING_DEVICE_ONLY_FOR=privateuse1.
        # Reset TorchTestBase.device_type to "privateuse1" so subsequent
        # calls generate the correct class name.
        TorchTestBase.device_type = "privateuse1"

    # ------------------------------------------------------------------
    # Config loading  (called once per test run via instantiate_test)
    # ------------------------------------------------------------------
    @classmethod
    def _load_test_suite_config(cls) -> None:
        path = os.environ.get(ENV_TEST_CONFIG)
        if not path or getattr(cls, "_yaml_loaded", False):
            return

        config: OOTTestConfig = load_yaml_config(path)

        # global op filtering and overrides
        cls._supported_ops = config.global_config.resolved_supported_ops()
        op_configs = config.global_config.resolved_supported_ops_config()
        if op_configs:
            apply_op_config_overrides(op_configs)
            cls.SUPPORTED_OPS_CONFIG = op_configs

        # global modules filtering and overrides
        cls._supported_modules = config.global_config.resolved_supported_modules()
        module_configs = config.global_config.resolved_supported_modules_config()
        if module_configs:
            cls.SUPPORTED_MODULES_CONFIG = module_configs

        cls.GLOBAL_SUPPORTED_DTYPES = config.global_config.resolved_supported_dtypes()
        cls.GLOBAL_DTYPE_PRECISION = (
            config.global_config.resolved_supported_dtypes_precision()
        )

        file_entry: FileEntry = resolve_current_file(config, path)

        cls.TEST_ENTRIES = _build_test_entry_map(file_entry)
        cls.UNLISTED_TEST_MODE = file_entry.unlisted_test_mode

        cls._yaml_loaded = True

    @classmethod
    def _should_run(
        cls,
        method_name: str,
        base_test_name: str,
        generic_cls_name: str,
    ) -> tuple:
        """Decide the behaviour of test variant based on config modes.

        Returns (enabled: bool, reason: Optional[str], xfail: bool, strict: bool)
        """
        # look up the test entry by base_test_name (method name without op/dtype suffix)
        entry: Optional[TestEntry] = cls.TEST_ENTRIES.get(base_test_name)

        # unlisted_test_mode only applies to tests NOT in TEST_ENTRIES
        if entry is not None:
            effective_mode = entry.mode  # always set, default is mandatory_success
        else:
            effective_mode = cls.UNLISTED_TEST_MODE  # only for truly unlisted tests

        # dtype filtering — extract dtype from method_name and check against supported
        dtype_str = extract_dtype_from_name(method_name)

        if dtype_str:
            try:
                dtype = parse_dtype(dtype_str)

                if entry is not None:
                    excluded = entry.edits.dtypes.resolved_exclude()
                    included = entry.edits.dtypes.resolved_include()
                else:
                    excluded = set()
                    included = set()

                if dtype in excluded:
                    return False, f"Excluded dtype: {dtype_str}", False, False

                # if explicitly included via edits
                # This is the additive path — dtype is IN ADDITION to global.supported_dtypes
                if dtype in included:
                    pass  # allow through regardless of global.supported_dtypes

                # Not explicitly included — apply global ceiling
                # This is the base intersection path:
                # (global.supported_dtypes ∩ op.dtypes ∩ test.allowed_dtypes)
                elif cls.GLOBAL_SUPPORTED_DTYPES is not None:
                    if dtype not in cls.GLOBAL_SUPPORTED_DTYPES:
                        return False, f"Unsupported dtype: {dtype_str}", False, False

            except ValueError:
                pass

        # apply force_xfail from op-level config
        # extract op name from method_name — format: test_name_opname_device_dtype
        # force_xfail only flips mandatory_success → xfail, leaves others unchanged
        op_name = _extract_op_name_from_method(method_name, base_test_name)
        if effective_mode == MODE_MANDATORY_SUCCESS:
            op_cfg = cls.SUPPORTED_OPS_CONFIG.get(op_name) if op_name else None
            if op_cfg is not None and op_cfg.force_xfail:
                effective_mode = MODE_XFAIL

        # resolve final decision
        if effective_mode == MODE_SKIP:
            return False, "Skipped for Spyre", False, False
        elif effective_mode == MODE_XFAIL:
            return True, None, True, False  # run, xfail non-strict
        elif effective_mode == MODE_XFAIL_STRICT:
            return True, None, True, True  # run, xfail strict
        else:  # MODE_MANDATORY_SUCCESS
            return True, None, False, False  # run, must pass

    @classmethod
    def _get_supported_ops(cls) -> Optional[Set[str]]:
        """Return the set of supported op names, or None if no filtering is configured."""
        return getattr(cls, "_supported_ops", None)

    @classmethod
    def _get_supported_modules(cls) -> Optional[Set[str]]:
        """Return the set of supported modules names, or None if no filtering is configured."""
        return getattr(cls, "_supported_modules", None)

    # ------------------------------------------------------------------
    # instantiate_test override
    # ------------------------------------------------------------------
    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):
        _OOTOnlyOnPatcher(test, _SPYRE_DEVICE_TYPE).patch()
        cls._load_test_suite_config()

        # print tags to stderr
        entry = cls.TEST_ENTRIES.get(name)
        tags = entry.tags if entry is not None else []
        # Collect op-level tags from all OpsNamedItem entries in this TestEntry
        # and union them with test-level tags so pytest -m works for both levels.
        op_tags: List[str] = []
        if entry is not None:
            seen_op_tags: set = set()
            for ops_item in entry.edits.ops.include:
                for t in ops_item.tags:
                    if t not in seen_op_tags:
                        seen_op_tags.add(t)
                        op_tags.append(t)

        # Union: test-level tags + op-level tags (deduplicated)
        all_tags = tags + [t for t in op_tags if t not in set(tags)]
        if all_tags:
            os.write(
                2,
                f"[OOTDeviceTestBase] {generic_cls.__name__}::{name} "
                f"tags: [{', '.join(all_tags)}]\n".encode(),
            )

        # op list filtering
        supported_ops = cls._get_supported_ops()
        if supported_ops is not None:
            _OOTOpListPatcher(test, supported_ops).patch()

        # @modules filtering
        supported_modules = cls._get_supported_modules()
        included_modules = (
            entry.edits.modules.included_module_names() if entry is not None else set()
        )
        excluded_modules = (
            entry.edits.modules.excluded_module_names() if entry is not None else set()
        )
        if supported_modules is not None or included_modules or excluded_modules:
            _OOTModuleListPatcher(
                test,
                supported_modules=supported_modules,
                included_modules=included_modules,
                excluded_modules=excluded_modules,
            ).patch()

        op_level_dtypes: Set[torch.dtype] = set()
        if cls.SUPPORTED_OPS_CONFIG:
            from torch.testing._internal.common_device_type import ops as _ops_cls

            underlying_fn = test.__func__ if hasattr(test, "__func__") else test
            p = getattr(underlying_fn, "parametrize_fn", None)
            if (
                p is not None
                and hasattr(p, "__self__")
                and isinstance(p.__self__, _ops_cls)
            ):
                for op_info in p.__self__.op_list:
                    op_cfg = cls.SUPPORTED_OPS_CONFIG.get(op_info.name)
                    if op_cfg is not None:
                        resolved = op_cfg.resolved_dtypes()
                        if resolved is not None:
                            op_level_dtypes |= resolved

        if op_level_dtypes:
            _OOTDtypePatcher(test, op_level_dtypes).patch()

        # module-level dtype injection from SUPPORTED_MODULES_CONFIG
        module_level_dtypes: Set[torch.dtype] = set()
        if cls.SUPPORTED_MODULES_CONFIG:
            from torch.testing._internal.common_modules import modules as _modules_cls

            underlying_fn = test.__func__ if hasattr(test, "__func__") else test
            p = getattr(underlying_fn, "parametrize_fn", None)
            if (
                p is not None
                and hasattr(p, "__self__")
                and isinstance(p.__self__, _modules_cls)
            ):
                for mod_info in p.__self__.module_info_list:
                    mod_cfg = cls.SUPPORTED_MODULES_CONFIG.get(mod_info.name)
                    if mod_cfg is not None:
                        resolved = mod_cfg.resolved_dtypes()
                        if resolved is not None:
                            module_level_dtypes |= resolved

        if module_level_dtypes:
            _OOTModuleDtypePatcher(test, module_level_dtypes).patch()

        if entry is not None:
            extra_dtypes = entry.edits.dtypes.resolved_include()
            if extra_dtypes:
                _OOTDtypePatcher(test, extra_dtypes).patch()
                _OOTOpDtypeExpander(test, extra_dtypes).patch()

        _OOTPrecisionOverridePatcher(
            test,
            global_dtype_precision=cls.GLOBAL_DTYPE_PRECISION,
            include_dtype_precision=(
                entry.edits.dtypes.resolved_include_precision()
                if entry is not None
                else {}
            ),
        ).patch()

        existing_methods = set(cls.__dict__.keys())
        super().instantiate_test(name, test, generic_cls=generic_cls)
        new_methods = set(cls.__dict__.keys()) - existing_methods

        for method_name in new_methods:
            enabled, reason, is_xfail, is_strict = cls._should_run(
                method_name=method_name,
                base_test_name=name,
                generic_cls_name=generic_cls.__name__,
            )

            if not enabled:
                # ------- Delete rather than replace with a skip stub -------
                # Previously this replaced the method with a unittest.SkipTest
                # stub, causing pytest to collect and report the variant as
                # SKIPPED. This happens for dtype-filtered variants (e.g.
                # "Unsupported dtype: complex128") which can produce dozens of
                # SKIPPED lines per test.
                #
                # Deleting the method entirely removes it from the class so
                # pytest never collects it
                delattr(cls, method_name)
                continue

            # Following lines has been commented out to disable generating
            # the skipped tests. If you want to generate, then please uncomment
            # these lines below and comment out the above lines.

            # if not enabled:
            #     @wraps(test)
            #     def _skip(self, _reason=reason or "Skipped for Spyre"):
            #         raise unittest.SkipTest(_reason)

            #     setattr(cls, method_name, _skip)
            #     continue

            # apply pytest tags as marks
            if all_tags:
                existing_fn = cls.__dict__.get(method_name)
                if existing_fn is not None:
                    marked_fn = existing_fn
                    for tag in all_tags:
                        marked_fn = pytest.mark.__getattr__(tag)(marked_fn)
                    setattr(cls, method_name, marked_fn)

            # apply the full method name as its own pytest marker
            existing_fn = cls.__dict__.get(method_name)
            if existing_fn is not None:
                safe_marker = re.sub(r"[^a-zA-Z0-9_]", "_", method_name)
                setattr(
                    cls,
                    method_name,
                    pytest.mark.__getattr__(safe_marker)(existing_fn),
                )

            # apply xfail if needed
            if is_xfail:
                existing_fn = cls.__dict__.get(method_name)
                if existing_fn is not None:
                    setattr(
                        cls,
                        method_name,
                        pytest.mark.xfail(strict=is_strict)(existing_fn),
                    )


TEST_CLASS = TorchTestBase
