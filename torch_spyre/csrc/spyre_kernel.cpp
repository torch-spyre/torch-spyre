/*
 * Copyright 2025 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spyre_kernel.h"

#include <c10/util/Exception.h>

#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "job_plan.h"
#include "logging.h"
#include "spyre_allocator.h"
#include "spyre_stream.h"

namespace spyre {

void launchJobPlan(const JobPlan& job_plan, const std::vector<at::Tensor>& args,
                   const SpyreStream& stream) {
  stream.launch(job_plan, args);
}

void launchJobPlan(const JobPlan& job_plan,
                   const std::vector<at::Tensor>& args) {
  auto stream = getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1, -1));
  launchJobPlan(job_plan, args, stream);
}

}  // namespace spyre
