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
import logging.handlers

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch_spyre._inductor import config
from torch_spyre._inductor.work_division_hint import work_division_hint


class TestWorkDivisionHint(InductorTestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.logger = logging.getLogger("torch_spyre._inductor.core_division")
        self._original_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.log_handler = logging.handlers.MemoryHandler(capacity=1000)
        self.log_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.log_handler)

    def tearDown(self):
        self.logger.removeHandler(self.log_handler)
        self.logger.setLevel(self._original_level)
        torch._dynamo.reset()
        super().tearDown()

    def _get_log_messages(self) -> list[str]:
        self.log_handler.flush()
        return [self.log_handler.format(record) for record in self.log_handler.buffer]

    @config.patch({"sencores": 8})
    def test_matmul_hint_applied(self):
        """User hint overrides automatic split decisions for matmul."""

        def fn(x, y):
            with work_division_hint([2, 1, 4]):
                return x @ y

        x = torch.randn(128, 256, dtype=torch.float16).to("spyre")
        y = torch.randn(256, 64, dtype=torch.float16).to("spyre")

        cfn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(cfn, x, y)

        logs = self._get_log_messages()
        hint_logs = [m for m in logs if "user-hint" in m]
        self.assertTrue(
            len(hint_logs) > 0,
            f"Expected 'user-hint' in core_division logs, got: {logs}",
        )

    @config.patch({"sencores": 8})
    def test_pointwise_hint_applied(self):
        """User hint overrides automatic split decisions for pointwise ops."""

        def fn(x, y):
            with work_division_hint([4, 1]):
                return x + y

        x = torch.randn(128, 64, dtype=torch.float16).to("spyre")
        y = torch.randn(128, 64, dtype=torch.float16).to("spyre")

        cfn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(cfn, x, y)

        logs = self._get_log_messages()
        hint_logs = [m for m in logs if "pointwise(user-hint)" in m]
        self.assertTrue(
            len(hint_logs) > 0,
            f"Expected 'pointwise(user-hint)' in logs, got: {logs}",
        )

    @config.patch({"sencores": 4})
    def test_hint_wrong_dim_count_falls_back(self):
        """Hint with wrong number of dims is ignored; heuristic is used."""

        def fn(x, y):
            # 2D matmul has 3 iteration dims (M, N, K) but we pass only 2
            with work_division_hint([2, 2]):
                return x @ y

        x = torch.randn(128, 256, dtype=torch.float16).to("spyre")
        y = torch.randn(256, 64, dtype=torch.float16).to("spyre")

        cfn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(cfn, x, y)

        logs = self._get_log_messages()
        warning_logs = [m for m in logs if "Ignoring hint" in m]
        self.assertTrue(
            len(warning_logs) > 0,
            f"Expected 'Ignoring hint' warning in logs, got: {logs}",
        )
        # Should NOT have user-hint logs since it fell back
        hint_logs = [m for m in logs if "user-hint" in m]
        self.assertEqual(len(hint_logs), 0)

    @config.patch({"sencores": 4})
    def test_hint_exceeds_max_cores_warns_but_applies(self):
        """Hint whose product > max_cores still applies with a warning.

        The backend may reject the resulting configuration, so we catch
        compilation errors and verify the warning/hint were logged.
        """

        def fn(x, y):
            # Product = 2*2*2 = 8 > sencores=4
            with work_division_hint([2, 2, 2]):
                return x @ y

        x = torch.randn(128, 256, dtype=torch.float16).to("spyre")
        y = torch.randn(256, 64, dtype=torch.float16).to("spyre")

        cfn = torch.compile(fn, dynamic=False)
        try:
            run_and_get_code(cfn, x, y)
        except Exception:
            pass  # Backend may reject configs exceeding max_cores

        logs = self._get_log_messages()
        over_core_logs = [m for m in logs if "exceeds max_cores" in m]
        self.assertTrue(
            len(over_core_logs) > 0,
            f"Expected 'exceeds max_cores' warning, got: {logs}",
        )
        # Should still apply despite warning
        hint_logs = [m for m in logs if "user-hint" in m]
        self.assertTrue(
            len(hint_logs) > 0,
            f"Expected hint to be applied despite exceeding max_cores, got: {logs}",
        )

    @config.patch({"sencores": 8})
    def test_no_hint_uses_heuristic(self):
        """Without a hint, the normal heuristic path is taken."""

        def fn(x, y):
            return x @ y

        x = torch.randn(128, 256, dtype=torch.float16).to("spyre")
        y = torch.randn(256, 64, dtype=torch.float16).to("spyre")

        cfn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(cfn, x, y)

        logs = self._get_log_messages()
        hint_logs = [m for m in logs if "user-hint" in m]
        self.assertEqual(
            len(hint_logs),
            0,
            f"Expected no 'user-hint' logs without a hint, got: {logs}",
        )

    @config.patch({"sencores": 8, "ignore_work_division_hints": True})
    def test_ignore_hints_flag_suppresses_hint(self):
        """Setting ignore_work_division_hints=True makes the planner ignore hints."""

        def fn(x, y):
            with work_division_hint([2, 1, 4]):
                return x @ y

        x = torch.randn(128, 256, dtype=torch.float16).to("spyre")
        y = torch.randn(256, 64, dtype=torch.float16).to("spyre")

        cfn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(cfn, x, y)

        logs = self._get_log_messages()
        hint_logs = [m for m in logs if "user-hint" in m]
        self.assertEqual(
            len(hint_logs),
            0,
            f"Expected no 'user-hint' logs when ignore_work_division_hints=True, got: {logs}",
        )

    @config.patch({"sencores": 8, "ignore_work_division_hints": False})
    def test_ignore_hints_false_applies_hint(self):
        """Setting ignore_work_division_hints=False (default) still applies hints."""

        def fn(x, y):
            with work_division_hint([2, 1, 4]):
                return x @ y

        x = torch.randn(128, 256, dtype=torch.float16).to("spyre")
        y = torch.randn(256, 64, dtype=torch.float16).to("spyre")

        cfn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(cfn, x, y)

        logs = self._get_log_messages()
        hint_logs = [m for m in logs if "user-hint" in m]
        self.assertTrue(
            len(hint_logs) > 0,
            f"Expected 'user-hint' logs when ignore_work_division_hints=False, got: {logs}",
        )

    @config.patch({"sencores": 8})
    def test_hint_only_applies_inside_context(self):
        """Ops outside the context manager use the heuristic."""

        def fn(x, y, z):
            a = x + z  # should use heuristic
            with work_division_hint([2, 1, 4]):
                b = x @ y  # should use user-hint
            return a, b

        x = torch.randn(128, 256, dtype=torch.float16).to("spyre")
        y = torch.randn(256, 64, dtype=torch.float16).to("spyre")
        z = torch.randn(128, 256, dtype=torch.float16).to("spyre")

        cfn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(cfn, x, y, z)

        logs = self._get_log_messages()
        hint_logs = [m for m in logs if "user-hint" in m]
        # Only the matmul should have the user-hint, not the add
        for msg in hint_logs:
            self.assertIn("reduction(user-hint)", msg)

    @config.patch({"sencores": 8})
    def test_multiple_hint_blocks(self):
        """Different hint blocks for mm and bias-add both propagate."""

        def fn(x, w, b):
            with work_division_hint([4, 2, 1]):
                mm_out = x @ w.T
            with work_division_hint([4, 2]):
                out = mm_out + b
            return out

        x = torch.randn(512, 128, dtype=torch.float16).to("spyre")
        w = torch.randn(256, 128, dtype=torch.float16).to("spyre")
        b = torch.randn(256, dtype=torch.float16).to("spyre")

        cfn = torch.compile(fn, options={"epilogue_fusion": False}, dynamic=False)
        _, source_codes = run_and_get_code(cfn, x, w, b)

        logs = self._get_log_messages()
        reduction_hints = [m for m in logs if "reduction(user-hint)" in m]
        pointwise_hints = [m for m in logs if "pointwise(user-hint)" in m]
        self.assertTrue(
            len(reduction_hints) > 0,
            f"Expected reduction(user-hint) for mm, got: {logs}",
        )
        self.assertTrue(
            len(pointwise_hints) > 0,
            f"Expected pointwise(user-hint) for add, got: {logs}",
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
