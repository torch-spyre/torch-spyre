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

#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <torch/library.h>

int64_t _fused_sdp_choice(const at::Tensor& query, const at::Tensor& key,
                          const at::Tensor& value,
                          const std::optional<at::Tensor>& attn_mask,
                          double dropout_p, bool is_causal,
                          std::optional<double> scale, bool enable_gqa) {
  auto backend = sdp::SDPBackend::overrideable;
  return static_cast<int64_t>(backend);
}

using namespace at::native;  // NOLINT

REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice);

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_fused_sdp_choice", &_fused_sdp_choice);
}
