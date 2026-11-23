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

"""Apply a case's host.cpu_pinning request when `run` launches its command.

`prepare` turns the request into a plan for this host (pinning_plan.resolve) and finds the launcher tool; `wrap` puts
the launcher prefix in front of the command (one process tree) and writes the per-rank wrapper behind the
`{pin_wrapper}` token (torchrun-style layouts: `torchrun --no-python {pin_wrapper} python script.py`). A request that
cannot be applied here (no Linux, no tool, unknown device NUMA node) runs the command unpinned and records
`cpu_set_source=unresolved`, `pinning_verified=unknown` and a note; the observer then caps a `require_verified` run.
"""

import os
import shutil
import stat
import sys
from dataclasses import replace
from pathlib import Path

from spyre_bench import pinning_plan

WRAPPER_TOKEN = "pin_wrapper"
WRAPPER_NAME = "pin-wrapper.sh"


class Prepared:
    def __init__(self, plan, tool, notes):
        self.plan, self.tool, self.notes = plan, tool, notes

    @property
    def observes(self):
        """Whether the process tree is judged: any pinning requested, or verification demanded."""
        return self.plan.policy != "none" or self.plan.require_verified

    @property
    def launches(self):
        return self.plan.apply and bool(self.tool)

    def wrapper_path(self, repeat_dir):
        return str(Path(repeat_dir) / WRAPPER_NAME)

    def observer_request(self):
        return self.plan.observer_request()

    def markers(self, repeat_dir):
        """Wrapper paths the observer treats as the launcher stage; None when the engine does not launch the tree."""
        if not self.launches:
            return None
        return (self.wrapper_path(repeat_dir), *self.plan.stage_prefixes(self.tool))

    def wrap(self, argv, repeat_dir, uses_wrapper):
        """argv behind the launcher prefix; writes the wrapper file when the command names {pin_wrapper}."""
        if uses_wrapper:
            path = Path(self.wrapper_path(repeat_dir))
            path.write_text(self.plan.wrapper(self.tool or ""))
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return self.plan.prefix(self.tool) + list(argv) if self.launches else list(argv)

    def recorded(self, observed):
        """run.exec.* for a repeat: the observation when there was one, else what was requested."""
        if observed:
            return observed
        plan = self.plan
        props = {"run.exec.cpu_pinning": plan.policy, "run.exec.cpu_set_source": plan.source}
        if plan.policy != "none" or plan.require_verified:
            props["run.exec.pinning_verified"] = "unknown"
        if self.launches:
            props["run.exec.cpu_set"] = plan.union
        return props


def prepare(request, device_numa="", sys_root="/sys", which=None):
    """Prepared for a host.cpu_pinning request (None or {} is policy none). Raises pinning_plan.PinningError when the
    request is inconsistent in itself."""
    req = request or {"policy": "none"}
    notes = []
    topology = None
    if req.get("policy", "none") != "none":
        try:
            topology = pinning_plan.read_topology(sys_root)
        except pinning_plan.PinningError as e:
            notes.append(f"cpu_pinning: host topology unreadable ({e})")
    plan = pinning_plan.resolve(req, topology, device_numa)
    tool = None
    if plan.reason:
        notes.append(f"cpu_pinning {plan.policy}: not applied, {plan.reason}")
    elif plan.apply:
        tool = (
            (which or shutil.which)(plan.launcher) if os.name == "posix" and sys.platform.startswith("linux") else None
        )
        if not tool:
            why = f"{plan.launcher} not found" if sys.platform.startswith("linux") else "needs Linux"
            notes.append(f"cpu_pinning {plan.policy}: not applied, {why}")
            plan = replace(plan, apply=False, source="unresolved", reason=why, union=None, by_device=None)
    return Prepared(plan, tool, notes)
