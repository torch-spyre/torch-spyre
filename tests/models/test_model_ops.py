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
import regex as re
import sys

import pytest
import torch

from model_cases_loader import LoadedCase, case_key, load_all_cases
from runner import run_test, parse_dtype, make_SampleInput
import shared_config

from typing import Any, Dict, List

from torch.testing._internal.opinfo.core import (
    OpInfo,
)
from torch.testing._internal.common_device_type import (
    PrivateUse1TestBase,
    ops,
    instantiate_device_type_tests,
)
from op_registry import OP_REGISTRY, OpAdapter


class ModelOpInfo(OpInfo):
    """operator information for model-centric verification."""

    def __init__(
        self,
        name,
        aten_name,
        *,
        dtypes,
        model="",
        defaults: Dict[str, Any] = {},
        loadedCase: LoadedCase,
        adapter: OpAdapter,
        description: str = "",
        # Options from the OpInfo base class
        **kwargs,
    ):
        super().__init__(
            name,
            aten_name=aten_name,
            dtypes=dtypes,
            op=adapter.fn,
            **kwargs,
        )
        self.loadedCase = loadedCase
        self.defaults = defaults
        self.description = description


model_ops_db: list[ModelOpInfo] = []


def add_model_ops_db(loadedCases: List[LoadedCase]):
    seen_test_names = set()
    for loadedCase in loadedCases:
        test_name = loadedCase.case["name"]
        op_name = loadedCase.case["op"]
        adapter = OP_REGISTRY[op_name]
        basename = os.path.basename(loadedCase.source_path)
        test_name = f"{test_name}__{basename}_"
        assert test_name not in seen_test_names
        seen_test_names.add(test_name)

        defaults = loadedCase.defaults
        dtype_str = loadedCase.case.get("dtype", defaults.get("dtype", "fp16"))
        dtype = parse_dtype(dtype_str)

        model_ops_db.append(
            ModelOpInfo(
                test_name,
                aten_name=op_name,
                dtypes=(dtype,),
                loadedCase=loadedCase,
                adapter=adapter,
            )
        )


def _init_model_ops_db():
    """Initialize model_ops_db at module import time."""
    global model_ops_db
    if len(model_ops_db) > 0:
        return

    if "pytest" in sys.modules:
        root = shared_config._PYTEST_CONFIG.rootpath
        loadedCases = load_all_cases(root)
        add_model_ops_db(loadedCases)


_init_model_ops_db()

seen_case_keys = set()


class TestSpyreModelOps(PrivateUse1TestBase):
    @ops(model_ops_db)
    def test_model_ops_db(
        self,
        device,
        dtype,
        op,
    ):
        pytestconfig = shared_config._PYTEST_CONFIG
        selected_models = set(pytestconfig.getoption("--model") or [])
        dedupe_enabled = bool(pytestconfig.getoption("--dedupe", default=True))
        compile_backend = str(
            pytestconfig.getoption("--compile-backend") or "inductor"
        ).strip()
        allowed_test_names = pytestconfig.getoption("--test-name")

        method_name = self._testMethodName

        loadedCase: LoadedCase = op.loadedCase
        model = loadedCase.model
        case: Any = loadedCase.case
        defaults = loadedCase.defaults

        # 1) Model filtering (keeps your "only ops from granite3-speech" capability)
        if selected_models and model not in selected_models:
            pytest.skip(f"Filtered out by --model (selected={sorted(selected_models)})")

        # 2) Test names filtering
        if allowed_test_names:
            matched = any(test_name in method_name for test_name in allowed_test_names)
            if not matched:
                pytest.skip("Filtered out by --test-name list")

        # 3) Check pytest -m marker expression
        mark_expr = pytestconfig.option.markexpr
        if mark_expr:
            from _pytest.mark.expression import Expression

            compiled_expr = Expression.compile(mark_expr)

            # Get marks from YAML case
            case_marks = set()
            marks = case.get("marks")
            if isinstance(marks, str) and marks.strip():
                case_marks.add(marks.strip())
            elif isinstance(marks, (list, tuple)):
                for m in marks:
                    if isinstance(m, str) and m.strip():
                        case_marks.add(m.strip())

            # Evaluate if this test should run based on marks
            if not compiled_expr.evaluate(lambda m: m in case_marks):
                pytest.skip(f"Skipped by marker expression: {mark_expr}")

        # 4) Optional cross-model dedupe (do NOT dedupe at collection time!)
        if dedupe_enabled:
            global seen_case_keys
            k = case_key(case, defaults)
            if k in seen_case_keys:
                pytest.skip(
                    "duplicate signature already tested (enable/disable via --no-dedupe)"
                )
            seen_case_keys.add(k)

        test_device = torch.device(
            re.sub(r":\d+$", "", device)
        )  # remove :digit (e.g. :0)

        # load parameters
        seed = case.get("seed", defaults.get("seed", None))
        rtol = float(case.get("rtol", defaults.get("rtol", 5e-3)))
        atol = float(case.get("atol", defaults.get("atol", 5e-3)))
        description = case.get("description")

        # Build CPU args ONCE, then copy to Spyre (identical values)
        cpu_sample = make_SampleInput(case, seed, dtype)

        def to_spyre(arg):
            if isinstance(arg, list):
                return [x.to(test_device) for x in arg]
            elif isinstance(arg, torch.Tensor):
                return arg.to(test_device)
            else:
                return arg

        test_sample = cpu_sample.transform(to_spyre)

        # run target op
        try:
            run_test(
                op,
                self,
                cpu_sample,
                test_sample,
                test_device,
                compile_backend,
                rtol,
                atol,
                description,
            )
        finally:
            torch._dynamo.reset()


# Instantiate device type tests for the TestSpyreModelOps class
# This is required for @ops decorator to work properly
instantiate_device_type_tests(TestSpyreModelOps, globals())
