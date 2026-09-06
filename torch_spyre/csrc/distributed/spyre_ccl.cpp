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
 */
#include "spyre_ccl.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_allocator.h"
#include "spyre_composite_address.h"
#include "spyre_stream.h"
#include "types_mapping.h"

namespace c10d {

/***********************************************
 * Wrapper Backend for the Sypre Collective Library
 ***********************************************/
SpyreCCLBackend::SpyreCCLBackend(const c10::intrusive_ptr<::c10d::Store>& store,
                                 int rank, int size)
    : Backend(rank, size), group_context_(nullptr) {
  DEBUGINFO("# [Spyre CCL]: Constructor for ", getBackendName());

  /*
   * Start the communication library
   * Pass it the shared runtime library handle, and default stream.
   */
  spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                  spyre::getDefaultStreamRuntimeHandle());
  group_context_ = spyre_comms::get_world_context();
  if (nullptr == group_context_) {
    std::string _err_msg =
        "[" + getBackendName() + "]: Failed to capture the world context";
    throw std::runtime_error(_err_msg);
  }
}

SpyreCCLBackend::~SpyreCCLBackend() {
  spyre_comms::finalize_library();
}

/* **********************************************
 * Internal support functions
 ********************************************** */

/**
 * @brief Converts PyTorch reduction operation type to Spyre reduction operation
 * type.
 *
 * Maps c10d::ReduceOp enum values to spyre_comms::SpyreReductionOpType enum
 * values. Currently only supports SUM operation; other operations return
 * UNSUPPORTED.
 *
 * @param reduce_op The PyTorch reduction operation type to convert
 * @return The corresponding Spyre reduction operation type, or UNSUPPORTED if
 * not supported
 */
spyre_comms::SpyreReductionOpType convert_reduce_op_type(
    const ReduceOp reduce_op) {
  switch (reduce_op) {
    case ReduceOp::SUM:
      return spyre_comms::SpyreReductionOpType::SUM;
    default:
      return spyre_comms::SpyreReductionOpType::UNSUPPORTED;
  }
}

inline std::pair<spyre_comms::TensorDataTypeEnum,
                 spyre_comms::TensorDataTypeEnum>
convert_string_to_datatype_pair(const std::string& type_name) {
  /* val-1 = type on CPU-side
   * val-2 = type on Spyre-side
   */
  static const std::unordered_map<std::string,
                                  std::pair<spyre_comms::TensorDataTypeEnum,
                                            spyre_comms::TensorDataTypeEnum>>
      type_map = {
          // Boolean and string
          {"bool",
           {spyre_comms::TensorDataTypeEnum::boolean,
            spyre_comms::TensorDataTypeEnum::sen_fp16}},
          {"string",
           {spyre_comms::TensorDataTypeEnum::string,
            spyre_comms::TensorDataTypeEnum::string}},

          // IEEE floats
          {"fp8_143",
           {spyre_comms::TensorDataTypeEnum::float8,
            spyre_comms::TensorDataTypeEnum::sen_fp8}},
          // TODO(tmhoangt): figure out why there is not FP8 variant specific in
          // sen_datatype_enum
          {"fp8_152",
           {spyre_comms::TensorDataTypeEnum::float8,
            spyre_comms::TensorDataTypeEnum::sen_fp8}},
          {"float16",
           {spyre_comms::TensorDataTypeEnum::float16,
            spyre_comms::TensorDataTypeEnum::sen_fp16}},
          {"float32",
           {spyre_comms::TensorDataTypeEnum::float32,
            spyre_comms::TensorDataTypeEnum::float32}},
          {"float64",
           {spyre_comms::TensorDataTypeEnum::float64,
            spyre_comms::TensorDataTypeEnum::float64}},
          {"float128",
           {spyre_comms::TensorDataTypeEnum::float128,
            spyre_comms::TensorDataTypeEnum::float128}},
          {"float256",
           {spyre_comms::TensorDataTypeEnum::float256,
            spyre_comms::TensorDataTypeEnum::float256}},

          // Decimal
          {"decimal32",
           {spyre_comms::TensorDataTypeEnum::decimal32,
            spyre_comms::TensorDataTypeEnum::decimal32}},
          {"decimal64",
           {spyre_comms::TensorDataTypeEnum::decimal64,
            spyre_comms::TensorDataTypeEnum::decimal64}},
          {"decimal128",
           {spyre_comms::TensorDataTypeEnum::decimal128,
            spyre_comms::TensorDataTypeEnum::decimal128}},

          // bfloat
          {"bfloat16",
           {spyre_comms::TensorDataTypeEnum::bfloat16,
            spyre_comms::TensorDataTypeEnum::sen_fp16}},
          {"bfloat16_compute",
           {spyre_comms::TensorDataTypeEnum::bfloat16,
            spyre_comms::TensorDataTypeEnum::float32}},

          // Signed ints
          {"int1",
           {spyre_comms::TensorDataTypeEnum::int1,
            spyre_comms::TensorDataTypeEnum::sen_int1}},
          {"int2",
           {spyre_comms::TensorDataTypeEnum::int2,
            spyre_comms::TensorDataTypeEnum::sen_int2}},
          {"int4",
           {spyre_comms::TensorDataTypeEnum::int4,
            spyre_comms::TensorDataTypeEnum::sen_int4}},
          {"int8",
           {spyre_comms::TensorDataTypeEnum::int8,
            spyre_comms::TensorDataTypeEnum::sen_int8}},
          {"int16",
           {spyre_comms::TensorDataTypeEnum::int16,
            spyre_comms::TensorDataTypeEnum::sen_int16}},
          {"int32",
           {spyre_comms::TensorDataTypeEnum::int32,
            spyre_comms::TensorDataTypeEnum::sen_int32}},
          {"int64",
           {spyre_comms::TensorDataTypeEnum::int64,
            spyre_comms::TensorDataTypeEnum::sen_int32}},

          // Unsigned ints
          {"uint1",
           {spyre_comms::TensorDataTypeEnum::uint1,
            spyre_comms::TensorDataTypeEnum::sen_uint1}},
          {"uint2",
           {spyre_comms::TensorDataTypeEnum::uint2,
            spyre_comms::TensorDataTypeEnum::sen_uint2}},
          {"uint4",
           {spyre_comms::TensorDataTypeEnum::uint4,
            spyre_comms::TensorDataTypeEnum::sen_uint4}},
          {"uint8",
           {spyre_comms::TensorDataTypeEnum::uint8,
            spyre_comms::TensorDataTypeEnum::sen_uint8}},
          {"uint16",
           {spyre_comms::TensorDataTypeEnum::uint16,
            spyre_comms::TensorDataTypeEnum::sen_uint16}},
          {"uint32",
           {spyre_comms::TensorDataTypeEnum::uint32,
            spyre_comms::TensorDataTypeEnum::sen_uint32}},
          {"uint64",
           {spyre_comms::TensorDataTypeEnum::uint64,
            spyre_comms::TensorDataTypeEnum::sen_uint32}},

          // Quantized ints
          {"qint1",
           {spyre_comms::TensorDataTypeEnum::qint1,
            spyre_comms::TensorDataTypeEnum::qint1}},
          {"qint2",
           {spyre_comms::TensorDataTypeEnum::qint2,
            spyre_comms::TensorDataTypeEnum::qint2}},
          {"qint4",
           {spyre_comms::TensorDataTypeEnum::qint4,
            spyre_comms::TensorDataTypeEnum::qint4}},
          {"qint8",
           {spyre_comms::TensorDataTypeEnum::qint8,
            spyre_comms::TensorDataTypeEnum::qint8}},
          {"qint16",
           {spyre_comms::TensorDataTypeEnum::qint16,
            spyre_comms::TensorDataTypeEnum::qint16}},
          {"qint32",
           {spyre_comms::TensorDataTypeEnum::qint32,
            spyre_comms::TensorDataTypeEnum::qint32}},
          {"qint64",
           {spyre_comms::TensorDataTypeEnum::qint64,
            spyre_comms::TensorDataTypeEnum::qint64}},

          {"quint1",
           {spyre_comms::TensorDataTypeEnum::quint1,
            spyre_comms::TensorDataTypeEnum::quint1}},
          {"quint2",
           {spyre_comms::TensorDataTypeEnum::quint2,
            spyre_comms::TensorDataTypeEnum::quint2}},
          {"quint4",
           {spyre_comms::TensorDataTypeEnum::quint4,
            spyre_comms::TensorDataTypeEnum::quint4}},
          {"quint8",
           {spyre_comms::TensorDataTypeEnum::quint8,
            spyre_comms::TensorDataTypeEnum::quint8}},
          {"quint16",
           {spyre_comms::TensorDataTypeEnum::quint16,
            spyre_comms::TensorDataTypeEnum::quint16}},
          {"quint32",
           {spyre_comms::TensorDataTypeEnum::quint32,
            spyre_comms::TensorDataTypeEnum::quint32}},
          {"quint64",
           {spyre_comms::TensorDataTypeEnum::quint64,
            spyre_comms::TensorDataTypeEnum::quint64}},

          // Complex
          {"complex64",
           {spyre_comms::TensorDataTypeEnum::complex64,
            spyre_comms::TensorDataTypeEnum::complex64}},
          {"complex128",
           {spyre_comms::TensorDataTypeEnum::complex128,
            spyre_comms::TensorDataTypeEnum::complex128}},

          // Sentient types
          {"sen_fp8",
           {spyre_comms::TensorDataTypeEnum::sen_fp8,
            spyre_comms::TensorDataTypeEnum::sen_fp8}},
          {"sen_fp16",
           {spyre_comms::TensorDataTypeEnum::sen_fp16,
            spyre_comms::TensorDataTypeEnum::sen_fp16}},
          {"sen_fp8_compute",
           {spyre_comms::TensorDataTypeEnum::sen_fp8,
            spyre_comms::TensorDataTypeEnum::float32}},
          {"sen_fp16_compute",
           {spyre_comms::TensorDataTypeEnum::sen_fp16,
            spyre_comms::TensorDataTypeEnum::float32}},

          {"sen_int1",
           {spyre_comms::TensorDataTypeEnum::sen_int1,
            spyre_comms::TensorDataTypeEnum::sen_int1}},
          {"sen_int2",
           {spyre_comms::TensorDataTypeEnum::sen_int2,
            spyre_comms::TensorDataTypeEnum::sen_int2}},
          {"sen_int4",
           {spyre_comms::TensorDataTypeEnum::sen_int4,
            spyre_comms::TensorDataTypeEnum::sen_int4}},
          {"sen_int8",
           {spyre_comms::TensorDataTypeEnum::sen_int8,
            spyre_comms::TensorDataTypeEnum::sen_int8}},
          {"sen_int16",
           {spyre_comms::TensorDataTypeEnum::sen_int16,
            spyre_comms::TensorDataTypeEnum::sen_int16}},
          {"sen_int24",
           {spyre_comms::TensorDataTypeEnum::sen_int24,
            spyre_comms::TensorDataTypeEnum::sen_int24}},
          {"sen_int32",
           {spyre_comms::TensorDataTypeEnum::sen_int32,
            spyre_comms::TensorDataTypeEnum::sen_int32}},
          {"sen_int4_compute",
           {spyre_comms::TensorDataTypeEnum::sen_int4,
            spyre_comms::TensorDataTypeEnum::int32}},
          {"sen_int8_compute",
           {spyre_comms::TensorDataTypeEnum::sen_int8,
            spyre_comms::TensorDataTypeEnum::int32}},

          {"sen_uint1",
           {spyre_comms::TensorDataTypeEnum::sen_uint1,
            spyre_comms::TensorDataTypeEnum::sen_uint1}},
          {"sen_uint2",
           {spyre_comms::TensorDataTypeEnum::sen_uint2,
            spyre_comms::TensorDataTypeEnum::sen_uint2}},
          {"sen_uint4",
           {spyre_comms::TensorDataTypeEnum::sen_uint4,
            spyre_comms::TensorDataTypeEnum::sen_uint4}},
          {"sen_uint8",
           {spyre_comms::TensorDataTypeEnum::sen_uint8,
            spyre_comms::TensorDataTypeEnum::sen_uint8}},
          {"sen_uint16",
           {spyre_comms::TensorDataTypeEnum::sen_uint16,
            spyre_comms::TensorDataTypeEnum::sen_uint16}},
          {"sen_uint24",
           {spyre_comms::TensorDataTypeEnum::sen_uint24,
            spyre_comms::TensorDataTypeEnum::sen_uint24}},
          {"sen_uint32",
           {spyre_comms::TensorDataTypeEnum::sen_uint32,
            spyre_comms::TensorDataTypeEnum::sen_uint32}},
      };

  auto it = type_map.find(type_name);
  if (it != type_map.end()) {
    return it->second;
  }
  return {spyre_comms::TensorDataTypeEnum::dt_undef,
          spyre_comms::TensorDataTypeEnum::dt_undef};
}

spyre_comms::TensorInfo SpyreCCLBackend::getTensorInfo(
    const at::Tensor& input) {
  std::vector<int64_t> shape;
  if (input.dim() == 0) {
    shape = {1};
  } else {
    for (const int64_t& element : input.sizes()) {
      shape.push_back(element);
    }
  }
  auto str_type = spyre::torchScalarToString[input.scalar_type()];
  const auto [sen_dtype_cpu, sen_dtype_dev] =
      convert_string_to_datatype_pair(str_type);
  spyre_comms::TensorShape t_shape(shape);
  spyre_comms::TensorInfo ti{sen_dtype_cpu, t_shape};
  return ti;
}

/**
 * @brief Prepares a PyTorch tensor for use with Spyre communication operations.
 *
 * Converts a PyTorch tensor to a spyre_comms::Tensor with proper metadata.
 *
 * @param input_tensor The PyTorch tensor to prepare (must be on device)
 * @param output_tensor Pointer to the spyre_comms::Tensor to populate with
 * prepared data
 */
void SpyreCCLBackend::prepare_tensor(const at::Tensor& input_tensor,
                                     spyre_comms::Tensor* output_tensor) {
  spyre_comms::TensorInfo tensor_info = getTensorInfo(input_tensor);
  *output_tensor =
      spyre_comms::Tensor(tensor_info, input_tensor.storage().data_ptr().get());
  // Update the data pointer on the object with a borrowed pointer to the
  // tensor's flex::CompositeAddress; the borrowing setter ensures spyre-comms
  // will never try to delete it.
  output_tensor->SetSpyreDeviceAddressBorrowed(
      spyre::get_composite_address(input_tensor));
}

/**
 * @brief Validates that a single tensor meets requirements for collective
 * operations.
 *
 * Checks that the tensor is contiguous, dense (not sparse), and suitable for
 * Spyre device operations. Throws an error if any validation fails.
 *
 * @param tensor The tensor to validate
 * @throws TORCH_CHECK exception if tensor is not contiguous or is sparse
 */
void SpyreCCLBackend::check_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "The tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "The tensor has to be dense");
  }
  // Add check for spyre device tensor
}

/**
 * @brief Validates a vector of tensors for collective operations.
 *
 * Checks that the number of tensors in the vector is within the specified range
 * [min_allowed, max_allowed], and validates each individual tensor using
 * check_single_tensor(). Throws an error if validation fails.
 *
 * @param tensors The vector of tensors to validate
 * @param min_allowed Minimum number of tensors allowed in the vector
 * @param max_allowed Maximum number of tensors allowed in the vector
 * @throws TORCH_CHECK exception if tensor count is out of range or any tensor
 * is invalid
 */
void SpyreCCLBackend::check_vector_tensor(
    const std::vector<at::Tensor>& tensors, int min_allowed, int max_allowed) {
  if (static_cast<int>(tensors.size()) < min_allowed) {
    std::string _err_msg = "[" + getBackendName() +
                           "]: Too few tensors. Expected at least " +
                           std::to_string(min_allowed) +
                           " Actual: " + std::to_string(tensors.size());
    TORCH_CHECK(false, _err_msg);
  }
  if (static_cast<int>(tensors.size()) > max_allowed) {
    std::string _err_msg = "[" + getBackendName() +
                           "]: Too many tensors. Expected at most " +
                           std::to_string(max_allowed) +
                           " Actual: " + std::to_string(tensors.size());
    TORCH_CHECK(false, _err_msg);
  }
  for (auto& tensor : tensors) {
    check_single_tensor(tensor);
  }
}

/* **********************************************
 * Interface functions
 ********************************************** */
c10::intrusive_ptr<Work> SpyreCCLBackend::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllgatherOptions& opts) {
  if (static_cast<int>(outputTensors.size()) != 1) {
    std::string _err_msg =
        "[" + getBackendName() +
        "]: Too many tensors in the output list. Expected exactly 1" +
        " Actual: " + std::to_string(outputTensors.size());
    TORCH_CHECK(false, _err_msg);
  }
  if (static_cast<int>(outputTensors[0].size()) !=
      static_cast<int>(group_context_->getSize())) {
    std::string _err_msg =
        "[" + getBackendName() +
        "]: Incorrect output list size. The list size should be exactly " +
        std::to_string(group_context_->getSize()) +
        " Actual: " + std::to_string(outputTensors.size());
    TORCH_CHECK(false, _err_msg);
  }
  check_vector_tensor(inputTensors, 1, 1);

  spyre_comms::Tensor input_tensor;
  prepare_tensor(inputTensors[0], &input_tensor);

  std::vector<spyre_comms::Tensor> output_tensors;
  for (auto& outputTensor : outputTensors[0]) {
    spyre_comms::Tensor output_tensor;
    prepare_tensor(outputTensor, &output_tensor);
    output_tensors.push_back(output_tensor);
  }

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::ALLGATHER);
  work->work_schedule_ =
      group_context_->allgather(output_tensors, input_tensor);
  work->work_schedule_->start();
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::_allgather_base(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  // Do not intend to support: It is deprecated
  // https://github.com/pytorch/pytorch/blob/62226611ded023ff1119b103ed3f540f75e38e9d/torch/csrc/distributed/c10d/Backend.hpp#L197-L209
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
  check_vector_tensor(tensors, 1, 1);
  if (opts.reduceOp != ReduceOp::SUM) {
    std::string _err_msg = "[" + getBackendName() +
                           "]: Allreduce only supports SUM operation." +
                           " Actual: " + std::to_string(opts.reduceOp);
    TORCH_CHECK(false, _err_msg);
  }
  // Spyre Comms only supports float16 tensors because of the 'op' path
  // through the compiler. The data transfers do not have this restriction.
  if (spyre::torchScalarToString[tensors[0].scalar_type()] != "float16") {
    std::string _err_msg =
        "[" + getBackendName() + "]: Allreduce only supports float16 tensors." +
        " Actual: " + spyre::torchScalarToString[tensors[0].scalar_type()];
    TORCH_CHECK(false, _err_msg);
  }

  spyre_comms::Tensor tensor;
  prepare_tensor(tensors[0], &tensor);

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::ALLREDUCE);
  work->work_schedule_ =
      group_context_->allreduce(tensor, convert_reduce_op_type(opts.reduceOp));
  work->work_schedule_->start();
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::allreduce_coalesced(
    std::vector<at::Tensor>& tensors, const AllreduceCoalescedOptions& opts) {
  // Do not intend to support: No public interface
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllToAllOptions& opts) {
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::alltoall_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes, const AllToAllOptions& opts) {
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::barrier(const BarrierOptions& opts) {
  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::BARRIER);
  work->work_schedule_ = group_context_->barrier();
  work->work_schedule_->start();
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  check_vector_tensor(tensors, 1, 1);
  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::BROADCAST);

  spyre_comms::Tensor tensor;
  prepare_tensor(tensors[0], &tensor);

  work->work_schedule_ = group_context_->broadcast(tensor, opts.rootRank);
  work->work_schedule_->start();
  if (!opts.asyncOp) {
    work->wait();
  }

  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const GatherOptions& opts) {
  if (opts.rootRank == group_context_->getRank()) {
    if (static_cast<int>(outputTensors.size()) != 1) {
      std::string _err_msg =
          "[" + getBackendName() +
          "]: Too many tensors in the output list. Expected exactly 1" +
          " Actual: " + std::to_string(outputTensors.size());
      TORCH_CHECK(false, _err_msg);
    }
    if (static_cast<int>(outputTensors[0].size()) !=
        static_cast<int>(group_context_->getSize())) {
      std::string _err_msg =
          "[" + getBackendName() +
          "]: Incorrect output list size. The list size should be exactly " +
          std::to_string(group_context_->getSize()) +
          " Actual: " + std::to_string(outputTensors.size());
      TORCH_CHECK(false, _err_msg);
    }
  }
  check_vector_tensor(inputTensors, 1, 1);

  spyre_comms::Tensor input_tensor;
  prepare_tensor(inputTensors[0], &input_tensor);

  std::vector<spyre_comms::Tensor> output_tensors;
  if (opts.rootRank == group_context_->getRank()) {
    for (auto& outputTensor : outputTensors[0]) {
      spyre_comms::Tensor output_tensor;
      prepare_tensor(outputTensor, &output_tensor);
      output_tensors.push_back(output_tensor);
    }
  }

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::GATHER);
  work->work_schedule_ =
      group_context_->gather(output_tensors, input_tensor, opts.rootRank);
  work->work_schedule_->start();
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::reduce(
    std::vector<at::Tensor>& tensors, const ReduceOptions& opts) {
  check_vector_tensor(tensors, 1, 1);
  if (opts.reduceOp != ReduceOp::SUM) {
    std::string _err_msg = "[" + getBackendName() +
                           "]: Reduce only supports SUM operation." +
                           " Actual: " + std::to_string(opts.reduceOp);
    TORCH_CHECK(false, _err_msg);
  }
  // Spyre Comms only supports float16 tensors because of the 'op' path
  // through the compiler. The data transfers do not have this restriction.
  if (spyre::torchScalarToString[tensors[0].scalar_type()] != "float16") {
    std::string _err_msg =
        "[" + getBackendName() + "]: Reduce only supports float16 tensors." +
        " Actual: " + spyre::torchScalarToString[tensors[0].scalar_type()];
    TORCH_CHECK(false, _err_msg);
  }

  spyre_comms::Tensor tensor;
  prepare_tensor(tensors[0], &tensor);

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::REDUCE);
  work->work_schedule_ = group_context_->reduce(
      tensor, convert_reduce_op_type(opts.reduceOp), opts.rootRank);
  work->work_schedule_->start();
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::send(std::vector<at::Tensor>& tensors,
                                               int dstRank, int tag) {
  // NOTE: The c10d::Backend::send() signature carries no opts struct and
  // therefore exposes no asyncOp flag.  We follow the same non-blocking
  // submission pattern used by all collectives and leave completion control
  // to the caller via Work::wait().
  check_vector_tensor(tensors, 1, 1);

  spyre_comms::Tensor tensor;
  prepare_tensor(tensors[0], &tensor);

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::SEND);
  work->work_schedule_ = group_context_->send(tensor, dstRank, tag);
  work->work_schedule_->start();
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::recv(std::vector<at::Tensor>& tensors,
                                               int srcRank, int tag) {
  // NOTE: The c10d::Backend::recv() signature carries no opts struct and
  // therefore exposes no asyncOp flag.  We follow the same non-blocking
  // submission pattern used by all collectives and leave completion control
  // to the caller via Work::wait().
  check_vector_tensor(tensors, 1, 1);

  spyre_comms::Tensor tensor;
  prepare_tensor(tensors[0], &tensor);

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::RECV);
  work->work_schedule_ = group_context_->recv(tensor, srcRank, tag);
  work->work_schedule_->start();
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::recvAnysource(
    std::vector<at::Tensor>& tensors, int tag) {
  // Do not intend to support: Too much protocol overhead, and not commonly used
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Backend> SpyreCCLBackend::createSpyreCCLBackend(
    const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<SpyreCCLBackend>(store, rank, size);
}

/***********************************************
 * Wrapper Work for the Sypre Collective Library
 ***********************************************/
SpyreCCLWork::SpyreCCLWork(OpType opType)
    : Work(-1, opType),
      future_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))) {}

bool SpyreCCLWork::isCompleted() {
  if (work_schedule_ && !completed_) {
    try {
      completed_ = work_schedule_->query();
    }
    catch (...) {
      exception_ = std::current_exception();
      completed_ = true;
      std::rethrow_exception(exception_);
    }
    // Resolve the future as soon as completion is first detected so that
    // callers who poll isCompleted() rather than calling wait() still see
    // a resolved future — matching the c10d::Work contract.
    if (completed_ && !future_->completed()) {
      future_->markCompleted(c10::IValue(std::vector<at::Tensor>{}));
    }
  }
  return completed_;
}

bool SpyreCCLWork::isSuccess() const {
  return exception_ == nullptr;
}

bool SpyreCCLWork::wait(std::chrono::milliseconds timeout) {
  if (!work_schedule_ || completed_) {
    // Already done; resolve the future if it hasn't been yet.
    if (!future_->completed()) {
      future_->markCompleted(c10::IValue(std::vector<at::Tensor>{}));
    }
    return true;
  }

  if (timeout == kUnsetTimeout || timeout == kNoTimeout) {
    // No deadline: block until the device signals completion.
    try {
      work_schedule_->wait();
    }
    catch (...) {
      exception_ = std::current_exception();
      std::rethrow_exception(exception_);
    }
  } else {
    // Timed wait: poll query() until completion or deadline, then throw on
    // timeout to match the c10d::Work base-class contract.
    auto deadline = std::chrono::steady_clock::now() + timeout;
    try {
      while (!work_schedule_->query()) {
        if (std::chrono::steady_clock::now() >= deadline) {
          TORCH_CHECK(false, "[SpyreCCL]: Operation timed out!");
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
    catch (...) {
      exception_ = std::current_exception();
      std::rethrow_exception(exception_);
    }
  }

  completed_ = true;

  // Resolve the future so any getFuture() chain-continuations are unblocked.
  if (!future_->completed()) {
    future_->markCompleted(c10::IValue(std::vector<at::Tensor>{}));
  }

  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> SpyreCCLWork::getFuture() {
  return future_;
}

}  // namespace c10d
