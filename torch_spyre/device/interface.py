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

import torch
from torch._dynamo.device_interface import DeviceInterface
from typing import Any
from dataclasses import dataclass

# Recording the device properties in the main process but used in worker process.
caching_worker_device_properties: dict[str, Any] = {}
caching_worker_current_devices: dict[str, int] = {}


@dataclass(frozen=True)
class SpyreDeviceProperties:
    type: str
    index: int
    multi_processor_count: int


class SpyreInterface(DeviceInterface):
    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.spyre.is_available()  # type: ignore[attr-defined]

    @classmethod
    def get_device_properties(
        cls, device: torch.types.Device = None
    ) -> SpyreDeviceProperties:
        return cls.Worker.get_device_properties(device)

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> Any:
        # TODO (tmhoangt): read this from cache
        # as worker process don't get access to device due to driver limitation
        return ""

    class Worker(DeviceInterface.Worker):
        # TODO (yoheiueda) Support non-zero index values when multiple Spyre cards are supported in the future
        @staticmethod
        def set_device(device: int):
            raise NotImplementedError

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def get_device_properties(device: torch.types.Device = None):
            # TODO (tmhoangt): read this from cache
            # as worker process don't get access to device due to driver limitation
            return SpyreDeviceProperties(type="dd2", index=0, multi_processor_count=32)
