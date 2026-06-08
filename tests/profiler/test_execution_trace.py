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

import unittest
from unittest.mock import patch
import torch
from typing import Any
from torch.profiler import profile
from torch.testing._internal.common_utils import (
    skipIfHpu,
    skipIfTorchDynamo,
    TestCase,
)
import json
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
import tempfile
import os
from torch.profiler import (
    supported_activities,
    record_function,
    kineto_available,
)


Json = dict[str, Any]


class TestExecutionTrace(TestCase):
    def payload(self, device, use_device=False):
        u = torch.randn(3, 4, 5)
        with torch.no_grad():
            with torch.profiler.record_function("## TEST 1 ##", "1, 2, 3"):
                inf_val = float("inf")
                neg_inf_val = float("-inf")
                nan_val = float("nan")
                rf_handle = _record_function_with_args_enter(
                    "## TEST 2 ##",
                    1,
                    False,
                    2.5,
                    [u, u],
                    (u, u),
                    "hello",
                    u,
                    inf_val,
                    neg_inf_val,
                    nan_val,
                )

                x = torch.randn(10, 10)
                if use_device:
                    x = x.to(device)
                y = torch.randn(10, 10)
                if use_device:
                    y = y.to(device)

                z = x + y + x * y + x * y
                gelu = torch.nn.GELU()
                m = torch.randn(2)
                _ = gelu(m)
                if use_device:
                    z = z.cpu()
                _record_function_with_args_exit(rf_handle)

    def get_execution_trace_rf_ids(self, nodes: list[Json]) -> list[int]:
        """Returns a sorted list of rf_id (record function ids) in execution trace"""

        def get_rf_id(node):
            attrs = node["attrs"]
            for a in attrs:
                if a["name"] == "rf_id":
                    return a["value"]

            return None

        rf_ids = (
            get_rf_id(n)
            for n in nodes
            if n["name"] != "[pytorch|profiler|execution_trace|process]"
            and n["name"] != "[pytorch|profiler|execution_trace|thread]"
        )

        print(rf_ids)
        return sorted(rf_id for rf_id in rf_ids if rf_id is not None)

    def get_kineto_rf_ids(self, events: list[Json]) -> list[int]:
        """Returns a sorted list of Record function IDs for CPU operators and user annotations"""
        ops_and_annoations = (
            e for e in events if e.get("cat", "") in ["cpu_op", "user_annotation"]
        )

        return sorted(
            e.get("args", "{}").get("Record function id", -1)
            for e in ops_and_annoations
        )

    def get_execution_trace_root(self, output_file_name):  # returns List[Json]
        import gzip

        nodes = []
        with (
            gzip.open(output_file_name)
            if output_file_name.endswith(".gz")
            else open(output_file_name)
        ) as f:
            et_graph = json.load(f)
            if "nodes" not in et_graph:
                raise AssertionError(f"Expected 'nodes' in execution trace: {et_graph}")
            nodes = et_graph["nodes"]

        return nodes

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @skipIfHpu
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @patch.dict(
        os.environ,
        {
            "ENABLE_PYTORCH_EXECUTION_TRACE": "1",
            "ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS": "1",
        },
    )
    def test_execution_trace_env_enabled_with_kineto(self, device="spyre"):
        trace_called_num = 0

        def trace_handler(p):
            nonlocal trace_called_num
            trace_called_num += 1

        use_device = device != "cpu"

        with (
            # Kineto Trace (kt_file — .kineto.json) A timeline of hardware + system events, intended for visualization in tools like Chrome useful for timi
            tempfile.NamedTemporaryFile(
                mode="w+t", suffix=".kineto.json", delete=False
            ) as kt,
        ):
            kt_name = kt.name

        # Uncomment for debugging
        # print("Output kineto = ", kt.name)

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=1
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for idx in range(10):
                with record_function(f"## LOOP {idx} ##"):
                    self.payload(device, use_device=use_device)
                p.step()

        p.export_chrome_trace(kt_name)

        et_path = p.execution_trace_observer.get_output_file_path()

        # et_res_path should be an empty directory and for spyre pytorch 2.12 update required verfied
        # et_res_path = p.execution_trace_observer.get_resources_dir(et_path)
        # self.assertTrue(os.path.isdir(et_res_path))
        # self.assertEqual(len(os.listdir(et_res_path)), 0)

        # only one cycle has ran
        self.assertEqual(trace_called_num, 1)
        # the path should be set up due to our env variables
        self.assertTrue(et_path is not None)
        # Compare the collected Execution Trace and Kineto Trace
        # in terms of record func
        nodes = self.get_execution_trace_root(et_path)
        # print(nodes)
        loop_count = 0
        found_root_node = False
        for n in nodes:
            if "name" not in n:
                raise AssertionError(f"Expected node to have 'name': {n}")
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1
        self.assertTrue(found_root_node)
        # Since profiler trace is active for 2 iteration
        self.assertEqual(loop_count, 2)

        # Compare the collected Execution Trace and Kineto Trace
        # in terms of record func ID (rf_id) and External IDs
        # both of these should match for the same trace window

        with open(kt_name) as f:
            kineto = json.load(f)
            events = kineto["traceEvents"]

        os.remove(kt_name)

        rf_ids_et = self.get_execution_trace_rf_ids(nodes)
        rf_ids_kineto = self.get_kineto_rf_ids(events)
        self.assertCountEqual(rf_ids_et, rf_ids_kineto)
        self.assertListEqual(
            rf_ids_et,
            rf_ids_kineto,
            msg=f"ET and kineto rf_id should exactly match\n"
            f"  rf_ids_et = {rf_ids_et}\n"
            f" rf_ids_kineto = {rf_ids_kineto}\n",
        )
