/*
 * Copyright 2026 The Torch-Spyre Authors.
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
 *
 * Portions derived from libkineto AIU plugin.
 */
#pragma once

#include <cassert>
#include <cstdlib>
#include <deque>
#include <memory>
#include <vector>

#include "AiuptiProfilerMacros.h"

namespace KINETO_NAMESPACE {

class AiuptiActivityBuffer {
 public:
  explicit AiuptiActivityBuffer(size_t size) : size_(size) {
    buf_.reserve(size);
  }
  AiuptiActivityBuffer() = delete;
  AiuptiActivityBuffer& operator=(const AiuptiActivityBuffer&) = delete;
  AiuptiActivityBuffer(AiuptiActivityBuffer&&) = default;
  AiuptiActivityBuffer& operator=(AiuptiActivityBuffer&&) = default;

  size_t size() const {
    return size_;
  }

  void setSize(size_t size) {
    assert(size <= buf_.capacity());
    size_ = size;
  }

  uint8_t* data() {
    return buf_.data();
  }

 private:
  std::vector<uint8_t> buf_;
  size_t size_;
};

using AiuptiActivityBufferDeque =
    std::deque<std::pair<uint8_t*, std::unique_ptr<AiuptiActivityBuffer>>>;
}  // namespace KINETO_NAMESPACE
