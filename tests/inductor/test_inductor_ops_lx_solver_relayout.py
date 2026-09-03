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

"""LX planning sweep with the CP-SAT co-optimizing solver and in-solver relayout.

Re-wraps the two LX-planning wrap classes from
``test_inductor_ops_lx_planning.py`` with three additional config patches:

- ``layout_solver = "cpsat"``: joint core-division + LX placement solving
- ``co_optimizing_lx_planning = True``: divisions chosen with the cost model
- ``lx_solver_relayout = True``: pinned on explicitly (it is the default;
  the pin keeps this lane meaningful even if the default ever changes)

Everything else (the two-op wraps, canonical-subset selection via
``TEST_LX_PLANNING_FULL``, tolerances) matches the LX-planning wrapper, so any
difference between this suite and the LX-planning suite is attributable to
the solver configuration alone.

Structure notes, hard-won against the OOT CI runner; mirror the shape of
``test_inductor_ops_lx_planning.py`` exactly:

- The classes must be plain ``class`` statements whose DIRECT base name
  contains "TestCase" or ends in "TestBase": the runner's AST analyzer only
  recognizes those, and an unrecognized file runs raw with its YAML shard
  config silently ignored.
- The base must carry NO test methods. ``instantiate_device_type_tests``
  deletes the generic class's OWN tests after building the gated device
  class, but tests INHERITED from a concrete parent survive through the MRO
  and are collected ungated alongside the gated ones.
- The patched tests are therefore installed as OWN attributes here, and
  ``wrap`` / ``_wrap_atol_floor`` are copied from the concrete LX-planning
  classes (``make_test_cls_with_patches`` cannot carry them: it builds from
  the source class's bases, and the abstract base's ``wrap`` raises
  NotImplementedError).
"""

import os
import sys

import torch_spyre

from torch._dynamo.testing import make_test_cls_with_patches

_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_test_dir)

import inductor.test_inductor_ops_lx_planning as _lx  # noqa: E402

_SOLVER_RELAYOUT_PATCHES = (
    (torch_spyre._inductor.config, "layout_solver", "cpsat"),
    (torch_spyre._inductor.config, "co_optimizing_lx_planning", True),
    (torch_spyre._inductor.config, "lx_solver_relayout", True),
)


class SolverRelayoutTwoOpPointwiseAdditionTest(_lx._LxPlanningTwoOpTestBase):
    pass


class SolverRelayoutTwoOpReductionTest(_lx._LxPlanningTwoOpTestBase):
    pass


def _build_wrap_class(dst_cls, src_cls):
    """Give ``dst_cls`` the two-op wrap behavior of ``src_cls`` plus the
    solver-relayout config patches around every test, all as own attributes.

    Carries the LX-planning file's INHERITED_TEST_ATTRIBUTES as well: those
    helpers (dtype support, the invalid-dim case lists) live on the concrete
    wrap classes rather than the abstract base, and tests call them through
    self, so a wrap class without them fails with AttributeError."""
    for attr in ("wrap", "_wrap_atol_floor", *_lx.INHERITED_TEST_ATTRIBUTES):
        # hasattr guard mirrors _copy_inherited_methods: not every helper in
        # INHERITED_TEST_ATTRIBUTES exists on every wrap class.
        if hasattr(src_cls, attr):
            setattr(dst_cls, attr, getattr(src_cls, attr))
    patched = make_test_cls_with_patches(
        src_cls, "SolverRelayout", "", *_SOLVER_RELAYOUT_PATCHES
    )
    for name, value in list(patched.__dict__.items()):
        if name.startswith("test_"):
            setattr(dst_cls, name, value)


_build_wrap_class(
    SolverRelayoutTwoOpPointwiseAdditionTest, _lx.LxPlanningTwoOpPointwiseAdditionTest
)
_build_wrap_class(SolverRelayoutTwoOpReductionTest, _lx.LxPlanningTwoOpReductionTest)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
