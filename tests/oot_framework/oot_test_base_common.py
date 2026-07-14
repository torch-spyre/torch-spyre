"""
Shared class and methods for all OOT PyTorch test overrides.
# Copyright Author: Anubhav Jana (Anubhav.Jana97@ibm.com)

"""

import os
import json
from typing import Dict, List, Optional, Set, Tuple
import warnings

import pytest  # type: ignore
import torch

from oot_framework.oot_test_constants import (
    _DYNAMIC_TAG_PREFIXES,
    DEFAULT_FLOATING_PRECISION,
    ENV_TEST_CONFIG,
    ENV_TEST_TYPE,
    MODE_MANDATORY_SUCCESS,
    MODE_SKIP,
    MODE_XFAIL,
    MODE_XFAIL_STRICT,
    UNLISTED_MODE_XFAIL,
)
from oot_framework.oot_test_matching import (
    extract_dtype_from_name,
    parse_dtype,
)
from oot_framework.oot_test_parsing import (
    FileEntry,
    apply_op_config_overrides,
    load_yaml_config,
    resolve_current_file,
)

from oot_framework.oot_upstream_patcher import (
    _OOTDtypePatcher,
    _OOTModuleMarkerPatcher,
    _OOTOnlyOnPatcher,
    _OOTOpDtypeExpander,
    _OOTOpListPatcher,
    _OOTModuleListPatcher,
    _OOTModuleDtypePatcher,
    _OOTOpMarkerPatcher,
    _OOTPrecisionOverridePatcher,
    _OOTNativeDeviceTypesPatcher,
    _OOTCpuMovePatcher,
    _OOTNoGradPatcher,
    _OOTPlatformMarkerPatcher,
)
from oot_framework.oot_test_config_models import (
    OOTTestConfig,
    Precision,
    SupportedOpConfig,
    SupportedModuleConfig,
    TestEntry,
)
from oot_framework.oot_test_common_methods_invocations import (
    create_module_inputs_func_from_yaml,
    create_module_inputs_func_from_config,
)
from oot_framework.oot_test_utilities import (
    _get_privateuse1_device_type,
    _log_warning,
    _log_error,
    _regex_entries_for_name,
    _build_test_entry_map,
    _select_entry_by_op_index,
    _select_entry_for_variant,
    _extract_op_name_from_method,
)

warnings.filterwarnings("ignore", category=pytest.PytestUnknownMarkWarning)


# Resolve the actual backend name registered for privateuse1.
# torch._C._get_privateuse1_backend_name() returns e.g. "spyre".
# This is what slf.device_type will be at test runtime.
_OOT_DEVICE_TYPE: str = _get_privateuse1_device_type()


# ---------------------------------------------------------------------------
# PrivateUse1TestBase filter
# ---------------------------------------------------------------------------
# TODO: figure out why this filter is needed - expected to use default PrivateUse1TestBase
def remove_builtin_privateuse1_test_base():
    """
    Remove built-in PrivateUse1TestBase from device_type_test_bases.

    This ensures only OOTTestBase handles the privateuse1 device type,
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


# ---------------------------------------------------------------------------
# OOTTestBase
# ---------------------------------------------------------------------------


# PrivateUse1TestBase injected via globals() by runpy
class OOTTestBase(PrivateUse1TestBase):  # type: ignore[name-defined]  # noqa: F821
    """Base class for OOT Device PyTorch test overrides.

    All configuration is loaded lazily from the YAML file pointed to by
    PYTORCH_TEST_CONFIG.  The YAML is validated by Pydantic on load.
    See oot_test_config_schema.json for the full schema.
    """

    device_type: str = "privateuse1"
    precision: float = DEFAULT_FLOATING_PRECISION

    # Exact-name lookup map: {base_method_name -> [TestEntry, ...]}
    # Multiple entries per name arise when two configs target the same test
    # with different tags/dtypes (e.g. same op tested for two different models).
    TEST_ENTRIES: Dict[str, List["TestEntry"]] = {}

    # Regex-pattern store: [(regex_pattern, TestEntry), ...]
    # Populated alongside TEST_ENTRIES from YAML names that contain * ? [ ].
    # Matched against concrete method names in instantiate_test().
    REGEX_ENTRIES: List[Tuple[str, "TestEntry"]] = []

    UNLISTED_TEST_MODE: str = UNLISTED_MODE_XFAIL  # file-level default
    SUPPORTED_OPS_CONFIG: Dict[str, "SupportedOpConfig"] = {}  # {op_name -> config}
    SUPPORTED_MODULES_CONFIG: Dict[
        str, "SupportedModuleConfig"
    ] = {}  # {module_name -> config}
    GLOBAL_SUPPORTED_DTYPES: Optional[Set[torch.dtype]] = None  # None = no filtering
    GLOBAL_DTYPE_PRECISION: Dict[torch.dtype, "Precision"] = {}
    GLOBAL_DTYPE_FORCE_XFAIL: Set[torch.dtype] = set()

    # File-level module filtering (populated during config load)
    # Use None as sentinel to indicate not yet initialized, avoiding shared mutable default
    _FILE_LEVEL_INCLUDED_MODULES: Optional[Set[str]] = None
    _FILE_LEVEL_EXCLUDED_MODULES: Optional[Set[str]] = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # PrivateUse1TestBase.setUpClass sets cls.device_type to the registered
        # backend name (e.g. "spyre").  This mutates the base class's device_type,
        # causing subsequent instantiate_device_type_tests calls to generate class
        # names like TestOldViewOpsSPYRE instead of TestOldViewOpsPRIVATEUSE1,
        # which then get filtered out by PYTORCH_TESTING_DEVICE_ONLY_FOR=privateuse1.
        # Reset OOTTestBase.device_type to "privateuse1" so subsequent
        # calls generate the correct class name.
        OOTTestBase.device_type = "privateuse1"

    # ------------------------------------------------------------------
    # Config loading  (called once per test run via instantiate_test)
    # ------------------------------------------------------------------
    @classmethod
    def _load_test_suite_config(cls) -> None:
        path = os.environ.get(ENV_TEST_CONFIG)
        if not path or getattr(cls, "_yaml_loaded", False):
            return

        # Reset regex store so a fresh load always starts clean.
        cls.REGEX_ENTRIES = []

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
            # Register module input generators for modules with inline inputs
            cls._register_module_input_generators(module_configs)

        cls.GLOBAL_SUPPORTED_DTYPES = config.global_config.resolved_supported_dtypes()
        cls.GLOBAL_DTYPE_PRECISION = (
            config.global_config.resolved_supported_dtypes_precision()
        )
        cls.GLOBAL_DTYPE_FORCE_XFAIL = (
            config.global_config.resolved_supported_dtypes_force_xfail()
        )

        file_entry: FileEntry = resolve_current_file(config, path)

        # Build the exact-name lookup map and the regex-pattern list.
        # Regex patterns (names containing regex metacharacters) go into
        # REGEX_ENTRIES; everything else goes into TEST_ENTRIES keyed by
        # exact method name.
        cls.TEST_ENTRIES, cls.REGEX_ENTRIES = _build_test_entry_map(file_entry)
        cls.UNLISTED_TEST_MODE = file_entry.unlisted_test_mode

        # Initialize file-level module tracking for this config load
        # Create new sets to avoid sharing state between test classes
        cls._FILE_LEVEL_INCLUDED_MODULES = set()
        cls._FILE_LEVEL_EXCLUDED_MODULES = set()

        for entry in file_entry.tests:
            if entry.edits.modules.include:
                cls._register_custom_modules_from_edits(entry.edits.modules.include)
                # Track included module names for filtering
                cls._FILE_LEVEL_INCLUDED_MODULES.update(
                    entry.edits.modules.included_module_names()
                )
            if entry.edits.modules.exclude:
                cls._FILE_LEVEL_EXCLUDED_MODULES.update(
                    entry.edits.modules.excluded_module_names()
                )

        cls._yaml_loaded = True

    @classmethod
    def _register_custom_modules_from_edits(cls, modules_named_items: List) -> None:
        """Register custom modules from edits.modules.include into module_db.

        This allows tests to use modules that aren't in PyTorch's upstream module_db
        by dynamically registering them before the _OOTModuleListPatcher runs.
        """

        try:
            from torch.testing._internal.common_modules import module_db, ModuleInfo
        except ImportError as e:
            _log_warning(
                f"Cannot register custom modules: torch.testing._internal.common_modules "
                f"not available: {e}"
            )
            return

        # Get existing module names to avoid duplicates
        existing_names = {m.name for m in module_db}
        for i, module_item in enumerate(modules_named_items):
            module_name = module_item.name
            # Skip if already registered
            if module_name in existing_names:
                continue

            # Try to import the module class
            module_path = getattr(module_item, "module_path", None)
            if not module_path:
                _log_warning(
                    f"Module '{module_name}' has no module_path, skipping registration"
                )
                continue

            try:
                # Import the module class
                parts = module_path.rsplit(".", 1)
                if len(parts) != 2:
                    _log_error(
                        f"Invalid module_path format for '{module_name}': {module_path}"
                    )
                    continue
                module_pkg, class_name = parts
                pkg = __import__(module_pkg, fromlist=[class_name])
                module_cls = getattr(pkg, class_name)
            except (ImportError, AttributeError) as e:
                _log_error(
                    f"Failed to import module '{module_name}' from {module_path}: "
                    f"{type(e).__name__}: {e}"
                )
                continue

            # Create ModuleInfo and add to module_db
            try:
                # edits.modules.include is additive: ModuleInfo.dtypes (the set
                # of dtype variants @modules will ever generate for this
                # module) is derived from the floating-point dtype(s) actually
                # baked into the YAML's tensor specs, so a dtype absent from
                # the YAML (e.g. float16 for a bfloat16-only module) is never
                # registered and therefore never generated -- regardless of
                # global.supported_dtypes.
                #
                # global.supported_dtypes (GLOBAL_SUPPORTED_DTYPES, applied in
                # _should_run_test_case below) is purely a filter on top of
                # that: it can only skip a dtype variant that was registered
                # here, never add one that wasn't. Fall back to the historical
                # default triple only when no tensor spec can be found at all
                # (e.g. a no-input module), so such a module still gets some
                # coverage.
                resolved_dtypes = module_item.resolved_input_dtypes()
                dtypes = (
                    tuple(sorted(resolved_dtypes, key=str))
                    if resolved_dtypes
                    else (torch.float32, torch.float16, torch.bfloat16)
                )
                module_info = ModuleInfo(
                    module_cls,
                    module_inputs_func=create_module_inputs_func_from_yaml(module_item),
                    skips=(),
                    decorators=None,
                    dtypes=dtypes,
                )
                module_db.append(module_info)
                existing_names.add(module_name)
            except Exception as e:
                _log_error(
                    f"Failed to create ModuleInfo for '{module_name}': "
                    f"{type(e).__name__}: {e}"
                )
                continue

    @classmethod
    def _register_module_input_generators(
        cls, module_configs: Dict[str, SupportedModuleConfig]
    ) -> None:
        """Register module input generators for modules with inline input specs.

        This creates generator functions that follow PyTorch's upstream signature:
        module_inputs_func(module_info, device, dtype, requires_grad, training, **kwargs) -> list[ModuleInput]
        """
        try:
            from torch.testing._internal.common_modules import module_db
        except ImportError as e:
            _log_warning(
                f"Cannot register module input generators: module_db not available: {e}"
            )
            return

        for module_name, module_config in module_configs.items():
            if not module_config.has_inline_inputs():
                continue

            # Find the module in module_db
            matching_modules = [m for m in module_db if m.name == module_name]
            if not matching_modules:
                _log_warning(
                    f"Module '{module_name}' not found in module_db, "
                    f"cannot register input generator"
                )
                continue

            module_info = matching_modules[0]

            # Replace the module's input generator
            module_info.module_inputs_func = create_module_inputs_func_from_config(
                module_config
            )

    @classmethod
    def _should_run(
        cls,
        method_name: str,
        base_test_name: str,
        generic_cls_name: str,
        entry: Optional["TestEntry"] = None,
    ) -> tuple:
        """Decide the behaviour of test variant based on config modes.

        The ``entry`` parameter is the already-resolved TestEntry for this
        specific variant (selected by ``_select_entry_for_variant`` in
        ``instantiate_test``).  Passing it in avoids a second map lookup and
        ensures the correct entry is used when multiple entries share the same
        base test name.

        Returns (enabled: bool, reason: Optional[str], xfail: bool, strict: bool)
        """
        # If entry was not pre-resolved by the caller, fall back to the old
        # single-entry lookup for backward compatibility.
        if entry is None:
            entries = cls.TEST_ENTRIES.get(base_test_name)
            if not entries:
                # Also check regex patterns before giving up.
                entries = (
                    _regex_entries_for_name(
                        base_test_name, getattr(cls, "REGEX_ENTRIES", [])
                    )
                    or None
                )
            if entries:
                entry = _select_entry_for_variant(
                    entries, method_name, cls.GLOBAL_SUPPORTED_DTYPES
                )

        # unlisted_test_mode only applies to tests NOT in TEST_ENTRIES
        if entry is not None:
            effective_mode = entry.mode  # always set, default is mandatory_success
        else:
            effective_mode = cls.UNLISTED_TEST_MODE  # only for truly unlisted tests

        # per-test label filter — skip if this entry's labels don't include the
        # active TEST_TYPE.  Empty labels means "no restriction; run whenever the
        # suite runs" so the check is a no-op for unlabelled entries.
        # suite_<group> values are structural (handled by filter_configs.py at the
        # config-file level) and are ignored here.
        if entry is not None and entry.labels:
            test_type = os.environ.get(ENV_TEST_TYPE, "full")
            if (
                test_type
                and test_type != "full"
                and not test_type.startswith("suite_")
                and test_type not in entry.labels
            ):
                return False, f"Excluded by TEST_TYPE={test_type!r}", False, False

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

                if dtype not in included and cls.GLOBAL_SUPPORTED_DTYPES is not None:
                    if dtype not in cls.GLOBAL_SUPPORTED_DTYPES:
                        return False, f"Unsupported dtype: {dtype_str}", False, False

            except ValueError as e:
                _log_warning(
                    f"Failed to parse dtype '{dtype_str}' in test '{method_name}': {e}"
                )
                # Continue with test execution - dtype filtering is optional

        # apply force_xfail from op-level config
        # extract op name from method_name — format: test_name_opname_device_dtype
        # force_xfail only flips mandatory_success → xfail, leaves others unchanged
        op_name = _extract_op_name_from_method(
            method_name, base_test_name, _OOT_DEVICE_TYPE
        )

        # Op exclusion check (edits.ops.exclude).
        # op_name may include a variant suffix (e.g. "addmm_decomposed" for the
        # "addmm" OpInfo with variant_test_name="decomposed"), so we check both
        # an exact match and a "<excluded>_" prefix match to catch all variants
        # of an excluded op with a single exclude entry.
        if op_name and entry is not None:
            excluded_ops = entry.edits.ops.excluded_op_names()
            if op_name in excluded_ops or any(
                op_name.startswith(exc + "_") for exc in excluded_ops
            ):
                return False, f"Excluded op: {op_name}", False, False

        if effective_mode == MODE_MANDATORY_SUCCESS:
            op_cfg = cls.SUPPORTED_OPS_CONFIG.get(op_name) if op_name else None
            if op_cfg is not None and op_cfg.force_xfail:
                effective_mode = MODE_XFAIL

        if effective_mode == MODE_MANDATORY_SUCCESS and dtype_str:
            try:
                dtype = parse_dtype(dtype_str)
                if dtype in cls.GLOBAL_DTYPE_FORCE_XFAIL:
                    effective_mode = MODE_XFAIL
            except ValueError:
                pass

        # resolve final decision
        if effective_mode == MODE_SKIP:
            return False, "Skipped by OOT config", False, False
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
        _OOTOnlyOnPatcher(test, _OOT_DEVICE_TYPE).patch()
        _OOTNativeDeviceTypesPatcher.patch()
        cls._load_test_suite_config()

        # ------------------------------------------------------------------
        # Retrieve all TestEntry objects for this base test name.
        # There may be multiple when different configs target the same test
        # name with different tags/dtypes (e.g. same op, different models).
        #
        # 1. Exact-name lookup in TEST_ENTRIES
        # 2. Regex-pattern lookup in REGEX_ENTRIES.  A YAML entry like
        #    ``TestOps::test_rope_fms_.*`` stores ``test_rope_fms_.*`` as a
        #    regex pattern; re.fullmatch is used to test whether ``name``
        #    matches.
        #
        # When both sources return entries the exact-name entries come first,
        # preserving the original priority ordering.
        # ------------------------------------------------------------------
        all_entries_for_name: List[TestEntry] = list(cls.TEST_ENTRIES.get(name, []))

        # Regex-pattern lookup: extend with any entries whose pattern matches
        # the current base test name.  Skip entries already in the exact list
        # to avoid double-processing the same TestEntry object.
        regex_matches = _regex_entries_for_name(name, getattr(cls, "REGEX_ENTRIES", []))
        for _re in regex_matches:
            if _re not in all_entries_for_name:
                all_entries_for_name.append(_re)

        # ------------------------------------------------------------------
        # Collect the union of all tags across all entries for collection-time
        # summary logging.  The per-variant tag selection happens later in the
        # new_methods loop where the dtype is known from method_name.
        # ------------------------------------------------------------------
        all_tags_union: List[str] = []
        _seen_union: set = set()
        for _e in all_entries_for_name:
            for _t in _e.tags or []:
                if _t not in _seen_union:
                    _seen_union.add(_t)
                    all_tags_union.append(_t)

        # Collect op-level tags for collection-time summary print ONLY
        op_tags: List[str] = []
        seen_op_tags: set = set()
        for _e in all_entries_for_name:
            for ops_item in _e.edits.ops.include:
                for t in ops_item.tags:
                    if t not in seen_op_tags:
                        seen_op_tags.add(t)
                        op_tags.append(t)

        # Print summary at collection time -- union of all tags
        summary_tags = all_tags_union + [t for t in op_tags if t not in _seen_union]
        if summary_tags:
            if generic_cls is not None:
                os.write(
                    2,
                    f"[OOTDeviceTestBase] {generic_cls.__name__}::{name} "
                    f"tags: [{', '.join(summary_tags)}]\n".encode(),
                )
            else:
                _log_warning(
                    f"Test '{name}' has tags {summary_tags} but generic_cls is None, "
                    f"cannot print tag information"
                )

        # Store union of test-level tags for backward compat (used by print_test_tags_oot)
        cls._TEST_LEVEL_TAGS = all_tags_union

        # op list filtering
        supported_ops = cls._get_supported_ops()
        if supported_ops is not None:
            _OOTOpListPatcher(test, supported_ops).patch()

        # @modules filtering using file-level included/excluded modules
        # Custom modules were already registered during _load_test_suite_config()
        supported_modules = cls._get_supported_modules()

        # Use file-level included/excluded modules (collected from ALL test entries)
        # This ensures filtering applies to ALL instantiate_test() calls, not just the first one
        # Use getattr with set() default to handle None (not yet initialized) case
        included_modules = getattr(cls, "_FILE_LEVEL_INCLUDED_MODULES", None) or set()
        excluded_modules = getattr(cls, "_FILE_LEVEL_EXCLUDED_MODULES", None) or set()

        # Merge in includes/excludes from ALL entries for this test name
        for _e in all_entries_for_name:
            included_modules = (
                included_modules | _e.edits.modules.included_module_names()
            )
            excluded_modules = (
                excluded_modules | _e.edits.modules.excluded_module_names()
            )

        if supported_modules is not None or included_modules or excluded_modules:
            _OOTModuleListPatcher(
                test,
                supported_modules=supported_modules,
                included_modules=included_modules,
                excluded_modules=excluded_modules,
            ).patch()

        # Collect dtype union across all entries for patching
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
                    mod_cfg = cls.SUPPORTED_MODULES_CONFIG.get(
                        mod_info.name
                    ) or cls.SUPPORTED_MODULES_CONFIG.get(f"torch.{mod_info.name}")

                    if mod_cfg is not None:
                        resolved = mod_cfg.resolved_dtypes()
                        if resolved is not None:
                            module_level_dtypes |= resolved

        if module_level_dtypes:
            _OOTModuleDtypePatcher(test, module_level_dtypes).patch()

        # Collect extra dtypes from ALL entries for this test name (union)
        all_extra_dtypes: Set[torch.dtype] = set()
        for _e in all_entries_for_name:
            all_extra_dtypes |= _e.edits.dtypes.resolved_include()

        if all_extra_dtypes:
            _OOTDtypePatcher(test, all_extra_dtypes).patch()
            _OOTOpDtypeExpander(test, all_extra_dtypes).patch()

        # Collect precision overrides: merge global + union across all entries.
        # precision_overrides/tolerance_overrides are dtype-keyed only (an
        # upstream limitation), so -- like the dtype injection above -- these
        # can only vary per dtype, not per op within a shared test method.
        include_dtype_precision: Dict[torch.dtype, Precision] = {}
        for _e in all_entries_for_name:
            include_dtype_precision.update(_e.edits.dtypes.resolved_include_precision())

        _OOTPrecisionOverridePatcher(
            test,
            global_dtype_precision=cls.GLOBAL_DTYPE_PRECISION,
            include_dtype_precision=include_dtype_precision,
        ).patch()

        # Dynamically adds pytest marker to each of ops and dtype passed to @ops
        _OOTOpMarkerPatcher(test).patch()

        # Dynamically adds pytest marker to each of modules and dtype passed to @modules
        _OOTModuleMarkerPatcher(test).patch()

        # Attaches platform__<arch> marker
        _OOTPlatformMarkerPatcher(test).patch()

        existing_methods = set(cls.__dict__.keys())
        super().instantiate_test(name, test, generic_cls=generic_cls)
        new_methods = set(cls.__dict__.keys()) - existing_methods

        # ------------------------------------------------------------------
        # Collect CPU move functions from global config and all test entries
        # for this test name. Then apply the CPU move patcher to move tensor
        # arguments to CPU for specified methods (e.g., assertEqual).
        # ------------------------------------------------------------------
        # Collect CPU move functions from all test entries for this test name.
        # Then apply the CPU move patcher to move tensor arguments to CPU.
        cpu_move_functions: Set[str] = set()
        for _e in all_entries_for_name:
            per_test_funcs = _e.edits.functions.resolved_cpu_move_functions()
            if per_test_funcs:
                cpu_move_functions.update(per_test_funcs)
        if cpu_move_functions:
            _OOTCpuMovePatcher(cls, list(cpu_move_functions), test_name=name).patch()

        _tags_to_write: Dict[str, List[str]] = {}
        for method_name in new_methods:
            # ------------------------------------------------------------------
            # Select the correct TestEntry for THIS variant using dtype matching.
            # Instead of using a single shared entry for all variants, we pick
            # the entry whose dtype set covers the dtype embedded in method_name
            # (e.g. bfloat16 -> bfloat16 entry, float16 -> float16 entry).
            # ------------------------------------------------------------------
            resolved_entry: Optional[TestEntry] = None
            if all_entries_for_name:
                resolved_entry = _select_entry_by_op_index(method_name)
                if resolved_entry is None:
                    resolved_entry = _select_entry_for_variant(
                        all_entries_for_name,
                        method_name,
                        cls.GLOBAL_SUPPORTED_DTYPES,
                    )

            # Tags for this specific variant = tags from the resolved entry only
            variant_tags: List[str] = (
                list(resolved_entry.tags) if resolved_entry else []
            )

            enabled, reason, is_xfail, is_strict = cls._should_run(
                method_name=method_name,
                base_test_name=name,
                generic_cls_name=generic_cls.__name__
                if generic_cls is not None
                else "",
                entry=resolved_entry,
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
            #     def _skip(self, _reason=reason or "Skipped by OOT config"):
            #         raise unittest.SkipTest(_reason)

            #     setattr(cls, method_name, _skip)
            #     continue

            # Collect dynamic markers (op__, dtype__, module__) that the
            # patchers attached to this specific instantiated method, and
            # union them with the variant-specific tags so _XML_INJECT_PY
            # only needs to handle one flat tag list per method.

            existing_fn = cls.__dict__.get(method_name)
            dynamic_tags: List[str] = []
            if existing_fn is not None:
                dynamic_tags = sorted(
                    {
                        m.name
                        for m in getattr(existing_fn, "pytestmark", [])
                        if any(m.name.startswith(p) for p in _DYNAMIC_TAG_PREFIXES)
                    }
                )

            seen = set(variant_tags)
            method_tags = list(variant_tags)
            for t in dynamic_tags:
                if t not in seen:
                    seen.add(t)
                    method_tags.append(t)

            # apply all tags (variant-specific YAML + dynamic) as marks
            if method_tags:
                existing_fn = cls.__dict__.get(method_name)
                if existing_fn is not None:
                    # Store BEFORE marking so the attribute is on the base function
                    existing_fn._oot_method_tags = method_tags
                    marked_fn = existing_fn
                    for tag in method_tags:
                        marked_fn = pytest.mark.__getattr__(tag)(marked_fn)
                    setattr(cls, method_name, marked_fn)
                _tags_to_write[method_name] = method_tags

            # Spyre's custom ops have no registered autograd formula, so a
            # test that builds modules/tensors with ordinary
            # requires_grad=True and never calls backward() must run under
            # torch.no_grad() to avoid AOTAutograd tracing a backward graph
            # at compile time. Opt in per test entry via the YAML `no_grad`
            # flag (e.g. upstream's ModuleInfo-based test_forward). See
            # _OOTNoGradPatcher.
            if resolved_entry is not None and resolved_entry.no_grad:
                existing_fn = cls.__dict__.get(method_name)
                if existing_fn is not None:
                    setattr(cls, method_name, _OOTNoGradPatcher.wrap(existing_fn))

            # apply xfail if needed
            if is_xfail:
                existing_fn = cls.__dict__.get(method_name)
                if existing_fn is not None:

                    def _make_xfail_wrapper(fn, strict):
                        # Factory function to capture fn and strict per-method.
                        # Without this, all closures in the loop would share the
                        # last values of existing_fn and is_strict.
                        def _xfail_wrapper(self, *args, **kwargs):
                            try:
                                fn(self, *args, **kwargs)
                            except BaseException as e:
                                if isinstance(e, pytest.skip.Exception):
                                    # pytest.skip() raised inside the test body (or by
                                    # PyTorch's test_wrapper infrastructure) is caught by
                                    # the unittest runner and reported as SKIPPED, completely
                                    # bypassing the xfail mark. Convert it to AssertionError
                                    # so the xfail mark sees a real failure instead.
                                    raise AssertionError(
                                        f"xfail: converted skip to failure: {e}"
                                    ) from e
                                # All other exceptions (TypeError, RuntimeError, etc.)
                                # propagate normally -- the xfail mark and our
                                # pytest_runtest_makereport hook will rewrite them to XFAIL.
                                raise

                        # Copy essential identity attributes so pytest can identify the
                        # test correctly in output and error messages.
                        _xfail_wrapper.__name__ = fn.__name__
                        _xfail_wrapper.__qualname__ = getattr(
                            fn, "__qualname__", fn.__name__
                        )
                        _xfail_wrapper.__doc__ = fn.__doc__

                        # Carry forward any existing marks (op__, dtype__, model__ tags)
                        # and append the xfail mark. pytest_runtest_makereport in conftest.py
                        # reads this to rewrite SKIPPED/FAILED -> XFAIL and PASSED -> XPASS,
                        # since the unittest runner ignores pytest.mark.xfail on TestCase methods.
                        existing_marks = list(getattr(fn, "pytestmark", []))
                        _xfail_wrapper.pytestmark = existing_marks + [
                            pytest.mark.xfail(strict=strict).mark
                        ]
                        # Copy per-variant tags onto wrapper so the hook can find them
                        # via fn._spyre_method_tags / fn._oot_method_tags when resolving
                        # tags for XFAIL/XPASS report lines.
                        _xfail_wrapper._spyre_method_tags = getattr(
                            fn, "_spyre_method_tags", []
                        )
                        _xfail_wrapper._oot_method_tags = getattr(
                            fn, "_oot_method_tags", []
                        )

                        # Defensively remove __wrapped__ in case it was somehow inherited.
                        # pytest walks __wrapped__ chains to resolve item.obj at collection
                        # time — if present it would resolve to the original function,
                        # losing our pytestmark.
                        if hasattr(_xfail_wrapper, "__wrapped__"):
                            del _xfail_wrapper.__wrapped__

                        return _xfail_wrapper

                    setattr(
                        cls, method_name, _make_xfail_wrapper(existing_fn, is_strict)
                    )

        # Flush {method_name: [tags]} to sidecar for _XML_INJECT_PY.
        # so that XML reads global + op/dtype/module tags in one shot
        if _tags_to_write:
            _cfg = os.environ.get(ENV_TEST_CONFIG, "")
            if _cfg:
                _sidecar = _cfg + ".markers.json"
                _existing_tags: dict = {}
                try:
                    with open(_sidecar) as _sf:
                        _existing_tags = json.load(_sf)
                except Exception:
                    pass
                _existing_tags.update(_tags_to_write)
                try:
                    with open(_sidecar, "w") as _sf:
                        json.dump(_existing_tags, _sf)
                except Exception:
                    pass


TEST_CLASS = OOTTestBase
