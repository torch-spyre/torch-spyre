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
import torch
import unittest
from torch._environment import is_fbcode
from torch.profiler import ProfilerActivity
from torch.testing._internal.common_utils import (
    TestCase,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests

if is_fbcode():
    import caffe2.test.profiler_test_cpp_thread_lib as cpp  # @manual=//caffe2/test:profiler_test_cpp_thread_lib
else:
    # cpp extensions use relative paths. Those paths are relative to
    # this file, so we'll change the working directory temporarily
    old_working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cpp = torch.utils.cpp_extension.load(
        name="profiler_test_cpp_thread_lib",
        sources=[
            "test_cpp_thread.cpp",
        ],
        verbose=True,
    )

    # return the working directory (see setUp)
    os.chdir(old_working_dir)


KinetoProfiler: torch.profiler.profile | None = None
IterationCount = 5
ActivateIteration = 2
device = "cpu"


def blueprint(text):
    print(f"\33[34m{text}\33[0m")


# onIterationStart() will be called by C++ training engine in cpp_thread_test_lib.cpp
class PythonProfilerEventHandler(cpp.ProfilerEventHandler):
    def onIterationStart(self, iteration: int) -> None:
        global KinetoProfiler, IterationCount
        assert KinetoProfiler is not None
        # it is important to start the profiler on the same thread that step() is called
        # and yes, onIterationStart() will always be called on the same thread
        if iteration == 0:
            # this also means step() starts on iteration 1, not 0
            KinetoProfiler.start()
            blueprint("starting kineto profiler")
        elif iteration == IterationCount - 1:
            KinetoProfiler.stop()
            blueprint("stopping kineto profiler")
        else:
            blueprint("stepping kineto profiler")
            KinetoProfiler.step()

    def emulateTraining(self, iteration: int, thread_id: int) -> None:
        global device
        # blueprint(f"training iteration {iteration} in thread {thread_id}")
        torch_device = getattr(torch, device)
        if not hasattr(torch_device, "synchronize"):
            raise AssertionError(f"Device {device} does not have synchronize method")
        sync_func = torch_device.synchronize

        with torch.autograd.profiler.record_function("user_function"):
            a = torch.ones(1, device=device)
            b = torch.ones(1, device=device)
            torch.add(a, b).cpu()
            sync_func()


@unittest.skipIf(True, "Blocked on profiler infra - see #<issue>")
class CppThreadTest(TestCase):
    ThreadCount = 20  # set to 2 for debugging
    EventHandler = None
    TraceObject: torch.profiler.profile | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.EventHandler = PythonProfilerEventHandler()
        cpp.ProfilerEventHandler.Register(cls.EventHandler)

    @classmethod
    def tearDownClass(cls):
        if not is_fbcode():
            torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

    def setUp(self) -> None:
        super().setUp()
        global device
        device = self.device_type

        # Warmup pass
        self.start_profiler(False)
        cpp.start_threads(1, IterationCount, False)

    def start_profiler(self, profile_memory):
        global KinetoProfiler
        KinetoProfiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=ActivateIteration, repeat=1
            ),
            on_trace_ready=self.set_trace,
            with_stack=True,
            profile_memory=profile_memory,
            record_shapes=True,
        )

    def set_trace(self, trace_obj) -> None:
        type(self).TraceObject = trace_obj

    def assert_text(self, condition, text, msg):
        if condition:
            print(f"\33[32m{text}\33[0m")
        else:
            print(f"\33[31m{text}\33[0m")
        self.assertTrue(condition, msg)

    def check_trace(self, expected, mem=False) -> None:
        blueprint("verifying trace")
        trace = type(self).TraceObject
        assert trace is not None
        event_list = trace.events()
        for key, values in expected.items():
            count = values[0]
            min_count = count * (ActivateIteration - 1)
            dev = values[1]
            filtered = filter(
                lambda ev: ev.name == key
                and str(ev.device_type) == f"DeviceType.{dev}",
                event_list,
            )

            if mem:
                mem_key = f"{self.device_type}_memory_usage"
                actual = 0
                for ev in filtered:
                    sev = str(ev)
                    has_device_memory_usage = (
                        sev.find(f"{mem_key}=0 ") < 0 and sev.find(f"{mem_key}=") > 0
                    )
                    if has_device_memory_usage:
                        actual += 1
                self.assert_text(
                    actual >= min_count,
                    f"{key}: {actual} >= {min_count}",
                    f"not enough event with {mem_key} set",
                )
            else:
                actual = len(list(filtered))
                if count == 1:  # test_without
                    count *= ActivateIteration
                    self.assert_text(
                        actual == count,
                        f"{key}: {actual} == {count}",
                        "baseline event count incorrect",
                    )
                else:
                    self.assert_text(
                        actual >= min_count,
                        f"{key}: {actual} >= {min_count}",
                        "not enough event recorded",
                    )

    def test_with_enable_profiler_in_child_thread(self) -> None:
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
                "user_function": [self.ThreadCount, "PrivateUse1"],
            }
        )

    def test_without_enable_profiler_in_child_thread(self) -> None:
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, False)
        self.check_trace(
            {
                "aten::add": [1, "CPU"],
                "user_function": [1, "PrivateUse1"],
            }
        )

    def test_profile_memory(self) -> None:
        self.start_profiler(True)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
            },
            mem=True,
        )


instantiate_device_type_tests(CppThreadTest, globals(), only_for=("spyre"))
