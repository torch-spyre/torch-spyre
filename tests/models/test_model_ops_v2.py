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


import logging
import os
import sys
from typing import Any, Dict, List, Optional, Set

import pytest
import regex as re
import torch
import torch.nn as nn

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.opinfo.core import OpInfo, SampleInput

from oot_framework.oot_test_config_models import (
    InputArgTensor,
    InputArgTensorList,
    OOTTestConfig,
    OpsNamedItem,
    TestEntry,
)
from oot_framework.oot_test_constants import ENV_TEST_CONFIG
from oot_framework.oot_test_parsing import load_yaml_config, resolve_current_file
from oot_framework.oot_test_utilities import (
    print_test_tags_oot,
    _format_input_args_shapes,
    _RUNTIME_SHAPES,
)
from op_registry import OP_REGISTRY, OpAdapter
import shared_config

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get("TORCH_SPYRE_DEBUG") else logging.INFO)

# ---------------------------------------------------------------------------
# ModelOpInfo
# ---------------------------------------------------------------------------


class ModelOpInfo(OpInfo):
    """OpInfo carrying per-op test data from edits.ops.include in the YAML config."""

    def __init__(
        self,
        name: str,
        op_name: str,
        *,
        dtypes,
        adapter: OpAdapter,
        ops_item: OpsNamedItem,
        test_entry: TestEntry,
        seed: Optional[int],
    ):
        super().__init__(
            name,
            aten_name=op_name,
            dtypes=dtypes,
            op=adapter.fn,
        )
        self.op_name: str = op_name
        self.adapter: OpAdapter = adapter
        self.ops_item: OpsNamedItem = ops_item  # inputs + description
        self.test_entry: TestEntry = test_entry  # seed, precision, tags
        self.op_tags: List[str] = ops_item.tags
        self.seed: Optional[int] = seed


# ---------------------------------------------------------------------------
# Build model_ops_db from config
# ---------------------------------------------------------------------------


def _build_model_ops_db() -> List[ModelOpInfo]:
    """One ModelOpInfo per edits.ops.include entry for TestSpyreModelOps::test_model_ops_db."""
    path = os.environ.get(ENV_TEST_CONFIG)
    if not path:
        return []

    try:
        config: OOTTestConfig = load_yaml_config(path)
        # seed is now read once from global config
        seed: Optional[int] = config.global_config.input_config.seed
        file_entry = resolve_current_file(config, path)
    except Exception as exc:
        import warnings

        warnings.warn(f"test_model_ops: failed to load config: {exc}")
        return []

    target = "TestSpyreModelOps::test_model_ops_db"
    matching_entries: List[TestEntry] = [
        entry for entry in file_entry.tests if target in entry.names
    ]

    if not matching_entries:
        return []

    db: List[ModelOpInfo] = []
    seen: Set[str] = set()
    idx = 0

    for test_entry in matching_entries:
        for ops_item in test_entry.edits.ops.include:
            op_name = ops_item.name
            if op_name not in OP_REGISTRY:
                import warnings

                warnings.warn(
                    f"test_model_ops: {op_name!r} not in OP_REGISTRY — skipping"
                )
                continue

            safe_op = op_name.replace(".", "_")
            unique_name = f"{safe_op}__{idx}"

            assert unique_name not in seen, f"Duplicate model_ops_db key: {unique_name}"
            seen.add(unique_name)
            idx += 1

            # choose a representative dtype used as a part of test name
            args = ops_item.sample_inputs_func.args
            assert isinstance(args, list)
            if len(args) == 0:
                # use float16 as default
                dtypes = (torch.float16,)
            elif isinstance(args[0], InputArgTensor):
                # use dtype of a tensor at the first arg
                dtypes = (args[0].tensor.resolved_dtype(),)
            elif isinstance(args[0], InputArgTensorList):
                # use dtype of the first tensor in a tensor list at the first arg
                dtypes = (args[0].tensor_list[0].resolved_dtype(),)
            else:
                # use float16 as default
                dtypes = (torch.float16,)

            db.append(
                ModelOpInfo(
                    unique_name,
                    op_name,
                    dtypes=dtypes,
                    adapter=OP_REGISTRY[op_name],
                    ops_item=ops_item,
                    test_entry=test_entry,
                    seed=seed,
                )
            )
            model_ops_entry_by_unique_name[unique_name] = test_entry

    return db


model_ops_db: List[ModelOpInfo] = []
# unique_name (e.g. "torch_add__2") -> originating TestEntry. Used by the
# variant resolver in spyre_test_base_common; dtype heuristics alone can
# pick the wrong entry when merged YAML configs overlap in dtypes.
model_ops_entry_by_unique_name: Dict[str, TestEntry] = {}


def _init_model_ops_db() -> None:
    global model_ops_db
    if not model_ops_db and "pytest" in sys.modules:
        model_ops_db.extend(_build_model_ops_db())
        # Pre-populate _RUNTIME_SHAPES at collection time keyed by op.name
        # key when the test body never ran (runner killed mid-run, fresh-process
        # retry).  Mirrors how _oot_method_tags is stamped at collection time so
        # that [TAGS] always appears even on a second-attempt run.
        for op in model_ops_db:
            shapes_str = _format_input_args_shapes(op.ops_item.sample_inputs_func.args)
            if shapes_str:
                _RUNTIME_SHAPES[op.name] = shapes_str


_init_model_ops_db()


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

_FACTORY_OPS: Set[str] = {
    "torch.tensor",
    "torch.zeros",
    "torch.ones",
    "torch.full",
    "torch.eye",
    "torch.arange",
    "torch.rand",
    "torch.randn",
    "torch.randint",
    "torch.empty",
}


def _normalize_out(out: Any) -> Any:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        return tuple(_normalize_out(x) for x in out)
    return out


def _to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_device(y, device) for y in x)
    return x


def _confirm_device(x: Any, expected: torch.device) -> bool:
    if torch.is_tensor(x):
        return str(expected) in str(x.device)
    if isinstance(x, (tuple, list)):
        return all(_confirm_device(item, expected) for item in x)
    return True


def _assert_close(
    tc: TestCase,
    ref: Any,
    got: Any,
    *,
    atol: float,
    rtol: float,
    case_name: str,
    description: Optional[str],
) -> None:
    ref = _normalize_out(ref)
    got = _normalize_out(got)
    if torch.is_tensor(ref):
        try:
            tc.assertEqual(got, ref, atol=atol, rtol=rtol)
        except AssertionError as e:
            raise AssertionError(
                f"{case_name} FAILED: output not close to reference\n{e}\n"
                f"shape={tuple(ref.shape)} dtype={ref.dtype}\n"
                f"location: {description}\n"
            ) from e
        return
    if isinstance(ref, tuple):
        assert isinstance(got, tuple) and len(got) == len(ref)
        for r, g in zip(ref, got):
            _assert_close(
                tc,
                r,
                g,
                atol=atol,
                rtol=rtol,
                case_name=case_name,
                description=description,
            )
        return
    assert got == ref


class _OpModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def _run_op(fn, sample: SampleInput, device: torch.device, backend: str) -> Any:
    if not backend or device.type == "cpu":
        return fn(sample.input, *sample.args, **sample.kwargs)
    try:
        mod = _OpModule(fn).to(device)
        torch._dynamo.reset_code_caches()
        return torch.compile(mod, backend=backend)(
            sample.input, *sample.args, **sample.kwargs
        )
    except Exception:
        return fn(sample.input, *sample.args, **sample.kwargs)


def _is_cpu_output(op_name: str, test_sample: SampleInput) -> bool:
    if op_name == "torch.to":
        return str(test_sample.kwargs.get("device", "")) == "cpu" or (
            test_sample.args and str(test_sample.args[0]) == "cpu"
        )
    device_str = str(test_sample.kwargs.get("device", ""))
    return device_str == "cpu" or (device_str == "" and op_name in _FACTORY_OPS)


def _get_global_dtype_precision() -> dict:
    """Load global dtype precision map from YAML config."""
    path = os.environ.get(ENV_TEST_CONFIG)
    if not path:
        return {}
    try:
        config: OOTTestConfig = load_yaml_config(path)
        return config.global_config.resolved_supported_dtypes_precision()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# TestSpyreModelOps
#
# OOTTestBase (spyre_test_base_common.py) already handles at instantiate_test:
#   - mode (xfail/skip/mandatory_success)
#   - tags : pytest marks
#   - unlisted_test_mode
#   - pytest -m marker expression
#
# This class owns only what the framework cannot: CLI filtering and op execution.
# ---------------------------------------------------------------------------

seen_case_keys: Set = set()


class TestSpyreModelOps(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    @ops(model_ops_db)
    def test_model_ops_db(self, device: str, dtype: torch.dtype, op: ModelOpInfo):
        # Usage: call `print_test_tags()` from our framework to print tags assosiated per method
        print_test_tags_oot(
            self, op_tags=op.op_tags, input_args=op.ops_item.sample_inputs_func.args
        )
        pytestconfig = shared_config._PYTEST_CONFIG
        assert pytestconfig is not None, (
            "shared_config._PYTEST_CONFIG is None — "
            "make sure conftest.py sets it before tests run"
        )
        selected_models: Set[str] = set(pytestconfig.getoption("--model") or [])
        dedupe_enabled: bool = bool(pytestconfig.getoption("--dedupe", default=True))
        compile_backend: str = str(
            pytestconfig.getoption("--compile-backend") or "inductor"
        ).strip()
        allowed_test_names = pytestconfig.getoption("--test-name")
        device_replace_disabled: bool = bool(
            pytestconfig.getoption("--no-device-replace", default=False)
        )
        assert not device_replace_disabled, (
            "Tentatively, --no-device-replace option is disabled"
        )

        method_name = self._testMethodName
        ops_item: OpsNamedItem = op.ops_item
        op_name: str = op.op_name
        adapter: OpAdapter = op.adapter

        # 1) Model filtering — match against tags (replaces old loadedCase.model)
        if selected_models and not selected_models.intersection(op.op_tags):
            pytest.skip(f"Filtered by --model at op level (op_tags={op.op_tags})")

        # 2) Test name filtering
        if allowed_test_names:
            if not any(n in method_name for n in allowed_test_names):
                pytest.skip("Filtered by --test-name")

        # 3) Cross-op dedupe
        if dedupe_enabled:
            global seen_case_keys
            dedup_key = (
                op_name,
                repr(ops_item.sample_inputs_func.args),
                repr(ops_item.sample_inputs_func.kwargs),
                dtype,  # variants with different runtime dtypes are distinct test cases
            )
            if dedup_key in seen_case_keys:
                pytest.skip(
                    "Duplicate signature already tested (--no-dedupe to disable)"
                )
            seen_case_keys.add(dedup_key)

        if not ops_item.sample_inputs_func.has_inputs():
            pytest.skip(f"No inputs specified for op {op_name!r}")

        # Config values — sourced entirely from TestEntry / OpsNamedItem
        seed: Optional[int] = op.seed
        description: Optional[str] = ops_item.description
        test_device = torch.device(re.sub(r":\d+$", "", device))
        _global_precision_map = (
            shared_config._PYTEST_CONFIG and _get_global_dtype_precision()
        )
        _dtype_prec = (
            _global_precision_map.get(dtype) if _global_precision_map else None
        )
        atol: float = (
            _dtype_prec.atol if _dtype_prec and _dtype_prec.atol is not None else 5e-3
        )
        rtol: float = (
            _dtype_prec.rtol if _dtype_prec and _dtype_prec.rtol is not None else 5e-3
        )

        # Build CPU SampleInput — all construction delegated to spyre_test_config_models
        cpu_sample: SampleInput = ops_item.build_sample_input(
            seed=seed,
            test_device="cpu",
            SampleInput=SampleInput,
        )

        def _to_target_device(x: Any, arg_spec: Optional[Any] = None) -> Any:
            if torch.is_tensor(x):
                if (
                    arg_spec is not None
                    and hasattr(arg_spec, "tensor")
                    and hasattr(arg_spec.tensor, "to_spyre")
                    and arg_spec.tensor.device_layout is not None
                ):
                    logger.debug(
                        f"[LAYOUT] calling to_spyre() for shape={list(x.shape)}"
                    )
                    return arg_spec.tensor.to_spyre(x)
                logger.debug(f"[PLAIN]  plain .to(device) for shape={list(x.shape)}")
                return x.to(test_device)

            if isinstance(x, (list, tuple)):
                # Mirror SampleInput.transform()'s list/tuple recursion, but stay
                # layout-aware: an InputArgTensorList spec's tensor_list entries
                # line up positionally with a list OR tuple of tensors here.
                tensor_specs = getattr(arg_spec, "tensor_list", None)
                result = []
                for i, item in enumerate(x):
                    spec = (
                        tensor_specs[i]
                        if tensor_specs is not None and i < len(tensor_specs)
                        else None
                    )
                    result.append(_to_target_device(item, spec))
                if isinstance(x, tuple):
                    # type(x)(result) breaks for tuple subclasses whose ctor
                    # doesn't accept a single iterable -- namedtuples included
                    # (they need *result, or ._make(result)). Use _make when
                    # available; otherwise fall back to a plain tuple.
                    make = getattr(x, "_make", None)
                    return make(result) if make is not None else tuple(result)
                return type(x)(result)

            if isinstance(x, dict):
                # Mirror transform()'s dict recursion. Positional dict args have no
                # per-key layout spec today (kwargs already forbid tensor/layout
                # specs — see resolved_kwargs()'s NotImplementedError), so this is
                # plain-device-move only, but it at least stops CPU tensors from
                # silently surviving inside a dict.
                return {k: _to_target_device(v) for k, v in x.items()}

            return x

        # Build test_sample with per-tensor layout awareness
        test_args = []
        for i, (cpu_arg, spec_arg) in enumerate(
            zip(
                [cpu_sample.input] + list(cpu_sample.args),
                ops_item.sample_inputs_func.args,
            )
        ):
            test_args.append(_to_target_device(cpu_arg, spec_arg))

        if test_args:
            test_sample = SampleInput(
                test_args[0],
                args=tuple(test_args[1:]),
                # NOTE: kwargs are plain values only (dim, dtype, keepdim, etc.) — no
                # current config passes a tensor/device_layout via kwargs, and
                # resolved_kwargs() will raise if one ever tries to. If that changes,
                # this needs the same arg_spec-based layout lookup as _to_target_device
                # uses for positional args above.
                kwargs={k: _to_target_device(v) for k, v in cpu_sample.kwargs.items()},
            )
        else:
            test_sample = cpu_sample.transform(lambda x: _to_target_device(x))

        # Adapter pre-hook (e.g. dropout sets training=False)
        if adapter.pre is not None:
            cpu_sample = adapter.pre(cpu_sample)
            test_sample = adapter.pre(test_sample)

        # Run
        fn = adapter.fn
        try:
            with torch.no_grad():
                ref_out = fn(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)
                test_out = _run_op(fn, test_sample, test_device, compile_backend)
                if adapter.is_inplace:
                    ref_out = cpu_sample.input
                    test_out = test_sample.input

            expected_device = (
                torch.device("cpu")
                if _is_cpu_output(op_name, test_sample)
                else test_device
            )
            assert _confirm_device(test_out, expected_device), (
                f"Output must be on {expected_device}"
            )

            _assert_close(
                self,
                _to_device(ref_out, torch.device("cpu")),
                _to_device(test_out, torch.device("cpu")),
                atol=atol,
                rtol=rtol,
                case_name=method_name,
                description=description,
            )
        finally:
            torch._dynamo.reset()


instantiate_device_type_tests(TestSpyreModelOps, globals())
