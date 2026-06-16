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
from torch_spyre._C import launch_kernel, prepare_kernel, launch_jobplan
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre.profiler._ffdc import (
    CATEGORY_RUNTIME_LAUNCH,
    CATEGORY_UNIMPLEMENTED,
    collect as _ffdc_collect,
)

logger = get_inductor_logger("kernel_runner")


class SpyreUnimplementedRunner:
    def __init__(self, name: str, op: str):
        self.kernel_name = name
        self.op = op

    def run(self, *args, **kw_args):
        exc = RuntimeError(
            f"Invoked {self.kernel_name} which contains"
            f" unimplemented operation {self.op}"
        )
        try:
            _ffdc_collect(
                exc,
                failure_category=CATEGORY_UNIMPLEMENTED,
                kernel_name=self.kernel_name,
            )
        except Exception:
            pass
        raise exc


class SpyreSDSCKernelRunner:
    def __init__(self, name: str, code_dir: str):
        self.kernel_name = name
        self.code_dir = code_dir
        self.jobplan = None
        dump_spyre_code = os.environ.get("DUMP_SPYRE_CODE", "1")
        if dump_spyre_code.isdigit() and int(dump_spyre_code) != 0:
            self.jobplan = prepare_kernel(code_dir + "/spyreCodeDir")

    def run(self, *args, **kw_args):
        logger.info("RUN: %s %s", self.kernel_name, self.code_dir)

        with torch.profiler.record_function(f"launch_kernel:{self.kernel_name}"):
            try:
                if self.jobplan:
                    launch_jobplan(self.jobplan, args)
                else:
                    launch_kernel(self.code_dir, args)
            except Exception as exc:
                try:
                    _ffdc_collect(
                        exc,
                        failure_category=CATEGORY_RUNTIME_LAUNCH,
                        kernel_name=self.kernel_name,
                        code_dir=self.code_dir,
                    )
                except Exception:
                    pass
                raise
