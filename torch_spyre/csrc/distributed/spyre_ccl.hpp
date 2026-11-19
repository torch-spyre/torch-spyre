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

#include <chrono>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <spyre_comms.hpp>
#include <string>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <vector>

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

  // Implements the new name; its alias _allgather_base forwards here.
  [[nodiscard]] c10::intrusive_ptr<Work> all_gather_single(
      at::Tensor& outputBuffer, at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  [[nodiscard]] c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  // Its alias allgather_into_tensor_coalesced forwards here.
  [[nodiscard]] c10::intrusive_ptr<Work> all_gather_single_coalesced(
      std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
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

  // Its alias alltoall_base forwards here.
  [[nodiscard]] c10::intrusive_ptr<Work> all_to_all_single(
      at::Tensor& outputBuffer, at::Tensor& inputBuffer,
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

  // Its alias gather_into_tensor forwards here.
  [[nodiscard]] c10::intrusive_ptr<Work> gather_single(
      at::Tensor& outputBuffer, at::Tensor& inputBuffer,
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

  // Its alias _reduce_scatter_base forwards here.
  [[nodiscard]] c10::intrusive_ptr<Work> reduce_scatter_single(
      at::Tensor& outputBuffer, at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  // Its alias reduce_scatter_tensor_coalesced forwards here.
  [[nodiscard]] c10::intrusive_ptr<Work> reduce_scatter_single_coalesced(
      std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
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
   *
   * shutdown() is a silent no-op: spyre_comms has no graceful-shutdown API,
   * and the silent no-op matches the behaviour of the Gloo, MPI, and UCC
   * backends. Finalization is handled by the SpyreCCLBackend destructor.
   *
   * abort() is not supported by the underlying spyre_comms library and will
   * throw SpyreCCLNotSupportedException.
   */
  void abort() override {
    throw SpyreCCLNotSupportedException(getBackendName(), __func__);
  }
  void shutdown() override {}

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
  // SUM only, on float16 or bfloat16 (DL16 on device).
  void check_reduction(const at::Tensor& tensor, const std::string& op_name,
                       const ReduceOp& reduce_op);
  // `whole` is group-size copies of `part` concatenated along dim 0.
  void check_dim0_concat(const at::Tensor& whole, const at::Tensor& part,
                         const std::string& op_name);
  // One tensor per rank with `like`'s sizes and device layout.
  std::vector<at::Tensor> make_rank_buffers(const at::Tensor& like);
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
  // Start `schedule` and track it; the work completes once all are done.
  void add_schedule(std::unique_ptr<spyre_comms::WorkSchedule> schedule);
  // Take over the schedules, kept-alive tensors and completion steps of
  // another (not yet waited) work. Used to build the coalesced ops.
  void absorb(SpyreCCLWork& other);
  // Run the completion steps once, then resolve the future.
  void finish();

  c10::intrusive_ptr<c10::ivalue::Future> future_;
  // getFuture() starts one thread that waits for the work and completes the
  // future: callers such as dist.all_gather_coalesced(async_op=True) return
  // only the future, and nothing else would drive it to completion.
  std::once_flag future_waiter_started_;
  // Serializes wait(), isCompleted() and finish() between that thread and the
  // caller.
  std::mutex mutex_;
  // Tensors whose device addresses spyre-comms borrows but the caller does
  // not own: materialized inputs (allgather's cloned views), staged copies
  // (reduce_scatter), per-rank buffers and packed tensors. Released once the
  // work completes.
  std::vector<at::Tensor> keep_alive_;
  std::vector<std::shared_ptr<spyre_comms::WorkSchedule>> work_schedules_;
  // Copies into caller-supplied outputs that can only run after the device
  // work completes (for example assembling all_gather_single's output from
  // per-rank buffers). Run by the first wait() or isCompleted() that sees
  // completion, so an async caller gets the result from work.wait().
  std::vector<std::function<void()>> on_complete_;
  bool completed_ = false;
  std::exception_ptr exception_ = nullptr;
};

}  // namespace c10d
