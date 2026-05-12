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

#include <ATen/autocast_mode.h>
#include <torch/library.h>

// Registers ops for AutocastPrivateUse1 (Spyre) device
// dispatch key to enable automatic mixed precision

// Fallthrough for all ops not explicitly listed below.
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  // ============================================================================
  // Keep in FP16/BF16
  // ============================================================================
  KERNEL_PRIVATEUSEONE(_convolution, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(conv1d, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(conv2d, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(conv3d, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(conv_tbc, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(conv_transpose1d, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(conv_transpose2d, input, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(conv_transpose3d, input, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(convolution, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(prelu, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(addmm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(addmv, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(addr, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(matmul, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(einsum, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(mv, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(linalg_vecdot, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(linear, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(addbmm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(baddbmm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(bmm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(chain_matmul, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(linalg_multi_dot, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(_thnn_fused_lstm_cell, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(_thnn_fused_gru_cell, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(lstm_cell, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(gru_cell, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(rnn_tanh_cell, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(rnn_relu_cell, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(_scaled_dot_product_flash_attention, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(scaled_dot_product_attention, lower_precision_fp)

  // ============================================================================
  // Keep in FP32
  // ============================================================================

  KERNEL_PRIVATEUSEONE(acos, fp32)
  KERNEL_PRIVATEUSEONE(asin, fp32)
  KERNEL_PRIVATEUSEONE(cosh, fp32)
  KERNEL_PRIVATEUSEONE(erfinv, fp32)
  KERNEL_PRIVATEUSEONE(exp, fp32)
  KERNEL_PRIVATEUSEONE(expm1, fp32)
  KERNEL_PRIVATEUSEONE(log, fp32)
  KERNEL_PRIVATEUSEONE(log10, fp32)
  KERNEL_PRIVATEUSEONE(log2, fp32)
  KERNEL_PRIVATEUSEONE(log1p, fp32)
  KERNEL_PRIVATEUSEONE(reciprocal, fp32)
  KERNEL_PRIVATEUSEONE(rsqrt, fp32)
  KERNEL_PRIVATEUSEONE(sinh, fp32)
  KERNEL_PRIVATEUSEONE(tan, fp32)
  KERNEL_PRIVATEUSEONE(pow, Tensor_Scalar, fp32)
  KERNEL_PRIVATEUSEONE(pow, Tensor_Tensor, fp32)
  KERNEL_PRIVATEUSEONE(pow, Scalar, fp32)
  KERNEL_PRIVATEUSEONE(softplus, fp32)
  KERNEL_PRIVATEUSEONE(layer_norm, fp32)
  KERNEL_PRIVATEUSEONE(native_layer_norm, fp32)
  KERNEL_PRIVATEUSEONE(group_norm, fp32)
  KERNEL_PRIVATEUSEONE(cosine_similarity, fp32)
  KERNEL_PRIVATEUSEONE(poisson_nll_loss, fp32)
  KERNEL_PRIVATEUSEONE(cosine_embedding_loss, fp32)
  KERNEL_PRIVATEUSEONE(nll_loss, fp32)
  KERNEL_PRIVATEUSEONE(nll_loss2d, fp32)
  KERNEL_PRIVATEUSEONE(hinge_embedding_loss, fp32)
  KERNEL_PRIVATEUSEONE(kl_div, fp32)
  KERNEL_PRIVATEUSEONE(l1_loss, fp32)
  KERNEL_PRIVATEUSEONE(smooth_l1_loss, fp32)
  KERNEL_PRIVATEUSEONE(huber_loss, fp32)
  KERNEL_PRIVATEUSEONE(mse_loss, fp32)
  KERNEL_PRIVATEUSEONE(margin_ranking_loss, fp32)
  KERNEL_PRIVATEUSEONE(multilabel_margin_loss, fp32)
  KERNEL_PRIVATEUSEONE(soft_margin_loss, fp32)
  KERNEL_PRIVATEUSEONE(triplet_margin_loss, fp32)
  KERNEL_PRIVATEUSEONE(multi_margin_loss, fp32)
  KERNEL_PRIVATEUSEONE(binary_cross_entropy_with_logits, fp32)
  KERNEL_PRIVATEUSEONE(dist, fp32)
  KERNEL_PRIVATEUSEONE(pdist, fp32)
  KERNEL_PRIVATEUSEONE(cdist, fp32)
  KERNEL_PRIVATEUSEONE(renorm, fp32)
  KERNEL_PRIVATEUSEONE(logsumexp, fp32)
  KERNEL_PRIVATEUSEONE(upsample_nearest1d, fp32)
  KERNEL_PRIVATEUSEONE(_upsample_nearest_exact1d, fp32)
  KERNEL_PRIVATEUSEONE(upsample_nearest2d, fp32)
  KERNEL_PRIVATEUSEONE(_upsample_nearest_exact2d, fp32)
  KERNEL_PRIVATEUSEONE(upsample_nearest3d, fp32)
  KERNEL_PRIVATEUSEONE(_upsample_nearest_exact3d, fp32)
  KERNEL_PRIVATEUSEONE(upsample_linear1d, fp32)
  KERNEL_PRIVATEUSEONE(upsample_bilinear2d, fp32)
  KERNEL_PRIVATEUSEONE(_upsample_bilinear2d_aa, fp32)
  KERNEL_PRIVATEUSEONE(upsample_trilinear3d, fp32)
  KERNEL_PRIVATEUSEONE(upsample_bicubic2d, fp32)
  KERNEL_PRIVATEUSEONE(_upsample_bicubic2d_aa, fp32)

  // ============================================================================
  // Keep largest dtype
  // ============================================================================
  KERNEL_PRIVATEUSEONE(addcdiv, promote)
  KERNEL_PRIVATEUSEONE(addcmul, promote)
  KERNEL_PRIVATEUSEONE(atan2, promote)
  KERNEL_PRIVATEUSEONE(bilinear, promote)
  KERNEL_PRIVATEUSEONE(cross, promote)
  KERNEL_PRIVATEUSEONE(dot, promote)
  KERNEL_PRIVATEUSEONE(vdot, promote)
  KERNEL_PRIVATEUSEONE(grid_sampler, promote)
  KERNEL_PRIVATEUSEONE(index_put, promote)
  KERNEL_PRIVATEUSEONE(tensordot, promote)
  KERNEL_PRIVATEUSEONE(scatter_add, promote)
}
