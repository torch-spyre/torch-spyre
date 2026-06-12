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
#pragma once

#include <pybind11/chrono.h>
#include <torch/python.h>

#include <spyre_comms.hpp>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include "module.h"

namespace c10d {

/***********************************************
 * Custom Exception Classes
 ***********************************************/
/**
 * @brief Exception thrown when an operation is not supported by the Spyre CCL
 * backend.
 *
 * This exception is used to indicate that a particular collective operation
 * or feature is not implemented or supported by the Spyre communication
 * library.
 */
class SpyreCCLNotSupportedException : public std::runtime_error {
 public:
  /**
   * @brief Constructs a not supported exception with backend name and
   * operation.
   *
   * @param backend_name Name of the backend (e.g., "SpyreCCL")
   * @param operation_name Name of the unsupported operation
   */
  SpyreCCLNotSupportedException(const std::string& backend_name,
                                const std::string& operation_name)
      : std::runtime_error("[" + backend_name + "]: The \"" + operation_name +
                           "\" operation is not supported.") {}
};

/***********************************************
 * Wrapper torch.distributed Backend for the Sypre Collective Library
 ***********************************************/
class SpyreCCLBackend : public c10d::Backend {
 public:
  SpyreCCLBackend(const c10::intrusive_ptr<::c10d::Store>& store, int rank,
                  int size);

  ~SpyreCCLBackend();

  /*
   * Informative
   */
  [[nodiscard]] const std::string getBackendName() const override {
    return std::string("SpyreCCL");
  }

  /*
   * Allgather
   */
  [[nodiscard]] c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  [[nodiscard]] c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer, at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  /*
   * Allreduce
   */
  [[nodiscard]] c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  [[nodiscard]] c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  /*
   * Alltoall
   */
  [[nodiscard]] c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  [[nodiscard]] c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  /*
   * Barrier
   */
  [[nodiscard]] c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  /*
   * Broadcast
   */
  [[nodiscard]] c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  /*
   * Gather
   */
  [[nodiscard]] c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  /*
   * Reduce
   */
  [[nodiscard]] c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  /*
   * Reduce-Scatter
   */
  [[nodiscard]] c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  /*
   * Scatter
   */
  [[nodiscard]] c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  /*
   * Point-to-Point
   */
  [[nodiscard]] c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors,
                                              int dstRank, int tag) override;

  [[nodiscard]] c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors,
                                              int srcRank, int tag) override;

  [[nodiscard]] c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors, int tag) override;

  /*
   * Shutdown
   */
  void abort() {};
  void shutdown() {};

  /*
   * Backend registration
   */
  [[nodiscard]] static c10::intrusive_ptr<Backend> createSpyreCCLBackend(
      const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
      const std::chrono::duration<float>& timeout);

 private:
  std::shared_ptr<spyre_comms::Context> group_context_;

  [[nodiscard]] spyre_comms::TensorInfo getTensorInfo(const at::Tensor& input);
  void prepare_tensor(const at::Tensor& input_tensor,
                      spyre_comms::Tensor* output_tensor);
  void check_single_tensor(const at::Tensor& tensor);
  void check_vector_tensor(const std::vector<at::Tensor>& tensors,
                           int min_allowed = 1, int max_allowed = 1);
};

/***********************************************
 * Wrapper backend for the Sypre Collective Library - Work
 ***********************************************/
class SpyreCCLWork : public Work {
  friend class SpyreCCLBackend;

 public:
  SpyreCCLWork(OpType opType);
  [[nodiscard]] bool isCompleted() override;
  [[nodiscard]] bool isSuccess() const override;
  [[nodiscard]] bool wait(
      std::chrono::milliseconds timeout = kUnsetTimeout) override;
  [[nodiscard]] virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture()
      override;

 private:
  c10::intrusive_ptr<c10::ivalue::Future> future_;
  std::shared_ptr<spyre_comms::WorkSchedule> work_schedule_;
};

}  // namespace c10d
