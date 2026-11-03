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
#include <libaiupti/aiupti_runtime_cbid.h>

#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AiuptiActivityProfiler.h"
#include "kernel_provenance_registry.h"

namespace KINETO_NAMESPACE {

namespace {

// GenericTraceActivity stores a 32-bit flow ID, but ITraceActivity and Kineto's
// loggers support 64 bits. Give submission flows their own ID range without
// changing the batch IDs used by hardware-launch-to-device flows.
class RuntimeTraceActivity : public libkineto::GenericTraceActivity {
 public:
  using GenericTraceActivity::GenericTraceActivity;

  int64_t flowId() const override {
    return submissionFlowId ? submissionFlowId : GenericTraceActivity::flowId();
  }

  void log(libkineto::ActivityLogger& logger) const override {
    // handleGenericActivity copies a GenericTraceActivity and would slice away
    // the 64-bit flow ID. The session transfers ownership in its trace buffer.
    logger.handleActivity(*this);
  }

  int64_t submissionFlowId{0};
};

}  // namespace

// =========== Session Private Methods ============= //
const libkineto::ITraceActivity*
AiuptiActivityProfilerSession::linkedCpuActivity(uint64_t externalId) {
  // Kineto's lookup callback currently accepts int32_t. Preserve wider IDs in
  // metadata, but never truncate one into a different CPU activity's identity.
  if (!cpuActivity_ || externalId == 0 ||
      externalId > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return nullptr;
  }
  const auto* linked = cpuActivity_(static_cast<int32_t>(externalId));
  return linked && linked->correlationId() == static_cast<int64_t>(externalId)
             ? linked
             : nullptr;
}

const libkineto::ITraceActivity* AiuptiActivityProfilerSession::linkedActivity(
    int32_t correlationId,
    const std::unordered_map<int64_t, int64_t>& correlationMap) {
  const auto& it = correlationMap.find(correlationId);
  if (it != correlationMap.end()) return linkedCpuActivity(it->second);
  return nullptr;
}

template <class ze_handle_type>
inline std::string handleToHexString(ze_handle_type handle) {
  return fmt::format("0x{:016x}", reinterpret_cast<uintptr_t>(handle));
}

// Raw device timestamps TS1..TS5 as a compact JSON array.
template <class activity_type>
inline std::string cyclesTsJson(const activity_type* activity) {
  return fmt::format("[{},{},{},{},{}]", activity->cycles_ts1,
                     activity->cycles_ts2, activity->cycles_ts3,
                     activity->cycles_ts4, activity->cycles_ts5);
}

template <class activity_type>
inline bool hasCyclesTs(const activity_type* activity) {
  return activity->cycles_ts1 != 0 || activity->cycles_ts2 != 0 ||
         activity->cycles_ts3 != 0 || activity->cycles_ts4 != 0 ||
         activity->cycles_ts5 != 0;
}

inline std::string runtimeCbidName(AIUpti_runtime_api_trace_cbid cbid) {
  switch (cbid) {
    case AIUPTI_RUNTIME_TRACE_CBID_INVALID:
      return "aiuINVALID";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB:
      return "aiuLaunchControlBlocks";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_CMPT:
      return "aiuLaunchControlBlocks";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_DMI:
      return "aiuLaunchDMIControlBlocks";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_DMO:
      return "aiuLaunchDMOControlBlocks";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CMPT_STREAM:
      return "aiuLaunchComputeStream";
    case AIUPTI_RUNTIME_TRACE_CBID_INIT_GRAPH:
      return "aiuInitGraph";
    case AIUPTI_RUNTIME_TRACE_CBID_MALLOC:
      return "aiuMalloc";
    case AIUPTI_RUNTIME_TRACE_CBID_RESIZE_TENSOR_ALLOCATION:
      return "aiuResizeTensorAllocation";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_SUPER_NODE:
      return "aiuLaunchSuperNode";
    case AIUPTI_RUNTIME_TRACE_CBID_SUPER_NODE_EXECUTE:
      return "aiuSuperNodeExecution";
    case AIUPTI_RUNTIME_TRACE_CBID_GRAPH_EXECUTE:
      return "aiuGraphExecution";
    case AIUPTI_RUNTIME_TRACE_CBID_NODE_COMPUTE:
      return "aiuNodeCompute";
    case AIUPTI_RUNTIME_TRACE_CBID_DATA_CONVERT:
      return "aiuDataConvert";
    case AIUPTI_RUNTIME_TRACE_CBID_INIT_SCHEDULER:
      return "aiuInitScheduler";
    case AIUPTI_RUNTIME_TRACE_CBID_CREATE_VIRTUAL_ADDRESSES:
      return "aiuCreateVirtualAddresses";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_SCHEDULE_COMPUTE:
      return "aiuLaunchScheduleCompute";
    case AIUPTI_RUNTIME_TRACE_CBID_SCHEDULE_WAIT:
      return "aiuScheduleWait";
    case AIUPTI_RUNTIME_TRACE_CBID_PREPARE_DMAS:
      return "aiuPrepareDMAs";
    case AIUPTI_RUNTIME_TRACE_CBID_PREPARE_SYNC_RDMA:
      return "aiuPrepareAndSyncRDMA";
    case AIUPTI_RUNTIME_TRACE_CBID_CLEAR_CACHE:
      return "aiuClearCache";
    case AIUPTI_RUNTIME_TRACE_CBID_PRELOAD_CACHE:
      return "aiuPreloadCache";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_COMPUTE_STREAM:
      return "aiuLaunchComputeStream";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_BARRIER_1:
      return "aiuRDMABarrier1";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_POST_KEYS:
      return "aiuPostRDMAKeys";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_BARRIER_2:
      return "aiuRDMABarrier2";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_FETCH_KEYS:
      return "aiuFetchRDMAKeys";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_UPDATE_CBS:
      return "aiuUpdateRDMACBs";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_BARRIER_3:
      return "aiuRDMABarrier3";
    case AIUPTI_RUNTIME_TRACE_CBID_RDMA_DEADLOCK_CHECK:
      return "aiuCheckRDMADeadlock";
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_DTOF:
      return "aiuFileTransferDtoF";
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_MTOF:
      return "aiuFileTransferMtoF";
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_FTOD:
      return "aiuFileTransferFtoD";
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_FTOM:
      return "aiuFileTransferFtoM";
    case AIUPTI_RUNTIME_TRACE_CBID_DATA_TRANSFER_DTOH:
      return "aiuDataTransferDtoH";
    case AIUPTI_RUNTIME_TRACE_CBID_DATA_TRANSFER_HTOD:
      return "aiuDataTransferHtoD";
    case AIUPTI_RUNTIME_TRACE_CBID_CLOCK_CALIBRATION:
      return "aiuClockCalibration";
    case AIUPTI_RUNTIME_TRACE_CBID_COMPILE_GRAPH:
      return "aiuCompileGraph";
    case AIUPTI_RUNTIME_TRACE_CBID_PARSE_RESPONSE:
      return "aiuParseResponse";
    case AIUPTI_RUNTIME_TRACE_CBID_ISSUE_CALLBACK:
      return "aiuIssueCallback";
    case AIUPTI_RUNTIME_TRACE_CBID_VERIFY_ASYC_MSGS:
      return "aiuVerifyAsyncMsgs";
    // Host-side spans emitted by spyre-comms (RFC 2973), cbids 41-77. These are
    // what make collectives visible in a PyTorch trace: AIUPTI records carry no
    // name, so the label shown to the user comes from here. A cbid missing from
    // this switch renders as "Unknown CBID <n>" rather than failing.
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_ALLREDUCE:
      return "aiuCommsCollAllreduce";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_ALLGATHER:
      return "aiuCommsCollAllgather";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_GATHER:
      return "aiuCommsCollGather";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_GATHERV:
      return "aiuCommsCollGatherv";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_REDUCE:
      return "aiuCommsCollReduce";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_BROADCAST:
      return "aiuCommsCollBroadcast";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_BARRIER:
      return "aiuCommsCollBarrier";
    case AIUPTI_RUNTIME_TRACE_CBID_WORK_SCHED_START:
      return "aiuCommsWorkSchedStart";
    case AIUPTI_RUNTIME_TRACE_CBID_WORK_SCHED_WAIT:
      return "aiuCommsWorkSchedWait";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_ALLREDUCE_SETUP:
      return "aiuCommsCollAllreduceSetup";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_ALLGATHER_SETUP:
      return "aiuCommsCollAllgatherSetup";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_GATHER_SETUP:
      return "aiuCommsCollGatherSetup";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_GATHERV_SETUP:
      return "aiuCommsCollGathervSetup";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_REDUCE_SETUP:
      return "aiuCommsCollReduceSetup";
    case AIUPTI_RUNTIME_TRACE_CBID_COLL_BROADCAST_SETUP:
      return "aiuCommsCollBroadcastSetup";
    case AIUPTI_RUNTIME_TRACE_CBID_WORK_SCHED_PHASE_RECORDING:
      return "aiuCommsWorkSchedPhaseRecord";
    case AIUPTI_RUNTIME_TRACE_CBID_WORK_SCHED_PHASE_WIREUP:
      return "aiuCommsWorkSchedPhaseWireup";
    case AIUPTI_RUNTIME_TRACE_CBID_WORK_SCHED_MSG_INFO_MGR:
      return "aiuCommsWorkSchedMsgInfoMgr";
    case AIUPTI_RUNTIME_TRACE_CBID_WORK_SCHED_HDMA_RT_MGR:
      return "aiuCommsWorkSchedHdmaRtMgr";
    case AIUPTI_RUNTIME_TRACE_CBID_WORK_SCHED_PHASE_LAUNCH:
      return "aiuCommsWorkSchedPhaseLaunch";
    case AIUPTI_RUNTIME_TRACE_CBID_OP_SYNCHRONIZE:
      return "aiuCommsOpSync";
    case AIUPTI_RUNTIME_TRACE_CBID_OP_D2H_TRANSFER:
      return "aiuCommsOpD2HXfer";
    case AIUPTI_RUNTIME_TRACE_CBID_OP_H2D_TRANSFER:
      return "aiuCommsOpH2DXfer";
    case AIUPTI_RUNTIME_TRACE_CBID_OP_SEND_DATA:
      return "aiuCommsOpSendData";
    case AIUPTI_RUNTIME_TRACE_CBID_OP_MULTICAST_SEND_DATA:
      return "aiuCommsOpMcastSendData";
    case AIUPTI_RUNTIME_TRACE_CBID_OP_RECV_DATA:
      return "aiuCommsOpRecvData";
    case AIUPTI_RUNTIME_TRACE_CBID_OP_BROADCAST_DATA:
      return "aiuCommsOpBcastData";
    case AIUPTI_RUNTIME_TRACE_CBID_OP_GATHER_DATA:
      return "aiuCommsOpGatherData";
    case AIUPTI_RUNTIME_TRACE_CBID_OP_ALLGATHER_DATA:
      return "aiuCommsOpAllgatherData";
    case AIUPTI_RUNTIME_TRACE_CBID_OP_BINARY_COMPUTE:
      return "aiuCommsOpBinaryComp";
    case AIUPTI_RUNTIME_TRACE_CBID_OP_COPY:
      return "aiuCommsOpCopy";
    case AIUPTI_RUNTIME_TRACE_CBID_BUNDLE_GENERATE:
      return "aiuCommsBundleGen";
    case AIUPTI_RUNTIME_TRACE_CBID_ARTIFACT_LOAD_TO_DEVICE:
      return "aiuCommsArtifactLoad";
    case AIUPTI_RUNTIME_TRACE_CBID_ARTIFACT_LAUNCH_COMPUTE_OP:
      return "aiuCommsArtifactLaunchOp";
    case AIUPTI_RUNTIME_TRACE_CBID_COST_ESTIMATE:
      return "aiuCommsCostEst";
    case AIUPTI_RUNTIME_TRACE_CBID_BENCH_PAIRWISE:
      return "aiuCommsBenchPairwise";
    case AIUPTI_RUNTIME_TRACE_CBID_BENCH_ALLREDUCE:
      return "aiuCommsBenchAllreduce";
    case AIUPTI_RUNTIME_TRACE_CBID_SUBMIT_TO_HARDWARE:
      return "aiuSubmitToHardware";
    case AIUPTI_RUNTIME_TRACE_CBID_FLEX_ROUNDTRIP:
      return "flexRoundTrip";
    case AIUPTI_RUNTIME_TRACE_CBID_AIU_ROUNDTRIP:
      return "aiuRoundtrip";
    case AIUPTI_RUNTIME_TRACE_CBID_WORK_SCHED_QUEUE_WAIT:
      return "aiuCommsWorkSchedQueueWait";
    case AIUPTI_RUNTIME_TRACE_CBID_WORK_SCHED_PREPARE:
      return "aiuCommsWorkSchedPrepare";
    case AIUPTI_RUNTIME_TRACE_CBID_WORK_SCHED_PRE_RECORD_RESET:
      return "aiuCommsWorkSchedPreRecordReset";
    case AIUPTI_RUNTIME_TRACE_CBID_OOB_BARRIER:
      return "aiuCommsOobBarrier";
    case AIUPTI_RUNTIME_TRACE_CBID_HDMA_RESET:
      return "aiuCommsHdmaReset";
    case AIUPTI_RUNTIME_TRACE_CBID_HDMA_FIRMWARE_RESET:
      return "aiuCommsHdmaFirmwareReset";
    case AIUPTI_RUNTIME_TRACE_CBID_WAIT_FOR_QUEUE_CAPACITY:
      return "aiuWaitForQueueCapacity";

    // ResponseWorker CBIDS (79-82)
    case AIUPTI_RUNTIME_TRACE_CBID_FETCH_RESPONSE_BLOCKS:
      return "aiuFetchResponseBlocks";
    case AIUPTI_RUNTIME_TRACE_CBID_RESPONSE_COMPLETION_THREAD:
      return "aiuResponseCompletionThread";
    case AIUPTI_RUNTIME_TRACE_CBID_PARSE_RESPONSE_BLOCKS:
      return "aiuParseResponseBlocks";
    case AIUPTI_RUNTIME_TRACE_CBID_ITERATION_DURATION:
      return "aiuIterationDuration";

    // CBIDS 83 to 84 are reserved
    // P2P RDMA CBIDS (85 - 91)
    case AIUPTI_RUNTIME_TRACE_CBID_P2P_RDMA_MULTICAST:
      return "aiuP2PRdmaMulticast";
    case AIUPTI_RUNTIME_TRACE_CBID_P2P_RDMA_WAIT:
      return "aiuP2PRdmaWait";
    case AIUPTI_RUNTIME_TRACE_CBID_SCHEDULE_P2P_RDMA_SEND:
      return "aiuScheduleP2PRdmaSend";
    case AIUPTI_RUNTIME_TRACE_CBID_SCHEDULE_P2P_RDMA_WAIT:
      return "aiuScheduleP2PRdmaWait";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_P2P_RDMA_SEND:
      return "aiuLaunchP2PRdmaSend";
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_P2P_RDMA_WAIT:
      return "aiuLaunchP2PRdmaWait";
    case AIUPTI_RUNTIME_TRACE_CBID_INITIALIZE_P2P:
      return "aiuInitializeP2P";

    // Scheduler submit/barrier CBIDS (102-105)
    case AIUPTI_RUNTIME_TRACE_CBID_SUBMIT_DMA:
      return "aiuSubmitDMA";
    case AIUPTI_RUNTIME_TRACE_CBID_SUBMIT_COMPUTE:
      return "aiuSubmitCompute";
    case AIUPTI_RUNTIME_TRACE_CBID_SUBMIT_FILL:
      return "aiuSubmitFill";
    case AIUPTI_RUNTIME_TRACE_CBID_ISSUE_BARRIER:
      return "aiuIssueBarrier";
    default:
      break;
  }
  return "Unknown CBID " + std::to_string(cbid);
}

template <size_t N>
inline std::string fixedCharField(const char (&field)[N]) {
  return std::string(field, strnlen(field, N));
}

// Collective-communication metadata, when the producer set any. Emitted as
// trace args only, so activity names and grouping are unaffected. The first
// byte is checked before building a string because most records carry none.
//
// Producer contract (libaiupti AIUpti_ActivityCompute/_ActivityMemcpy): the
// names are fixed char[AIUPTI_COLL_NAME_SIZE] arrays, empty when not set, and
// coll_bytes is 0 both for a zero-byte transfer and when unknown. flex fills
// coll_algo only from a structured "[Coll,Algo,Bytes]" label, whose fields are
// all required, so a non-empty coll_algo means coll_bytes was really parsed.
// On collective records coll_bytes is therefore the number when known
// (including a real 0) and "" when the label carried no size.

template <class T, class = void>
struct HasCollFields : std::false_type {};
template <class T>
struct HasCollFields<T, std::void_t<decltype(std::declval<T&>().coll_group),
                                    decltype(std::declval<T&>().coll_algo),
                                    decltype(std::declval<T&>().coll_bytes)>>
    : std::true_type {};

template <class trace_activity_type, class aiupti_activity_type>
inline void addCollMetadata(trace_activity_type& trace_activity,
                            const aiupti_activity_type* activity) {
  if constexpr (HasCollFields<aiupti_activity_type>::value) {
    if (activity->coll_group[0] != '\0') {
      trace_activity->addMetadataQuoted("coll_group",
                                        fixedCharField(activity->coll_group));
    }
    if (activity->coll_algo[0] != '\0') {
      trace_activity->addMetadataQuoted("coll_algo",
                                        fixedCharField(activity->coll_algo));
    }
    if (activity->coll_bytes != 0 || activity->coll_algo[0] != '\0') {
      trace_activity->addMetadata("coll_bytes", activity->coll_bytes);
    } else if (activity->coll_group[0] != '\0') {
      trace_activity->addMetadataQuoted("coll_bytes", "");
    }
  }
}

void AiuptiActivityProfilerSession::handleRuntimeActivity(
    const AIUpti_ActivityAPI* activity, libkineto::ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const libkineto::ITraceActivity* linked =
      linkedCpuActivity(activity->external_id);
  auto cbIDName =
      runtimeCbidName((AIUpti_runtime_api_trace_cbid)activity->cbid);
  auto runtime = std::make_unique<RuntimeTraceActivity>(
      traceBuffer_.span, libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
      cbIDName);
  auto* runtime_activity = runtime.get();
  traceBuffer_.activities.push_back(std::move(runtime));
  runtime_activity->startTime = activity->start;
  runtime_activity->endTime = activity->end;
  runtime_activity->id = activity->correlation_id;
  runtime_activity->device = activity->process_id;
  const bool calibration = activity->calibration_id != 0;
  // One explicitly logical lane per calibrator, retaining physical host TIDs
  // in metadata. Measurement helper threads do not create extra trace tracks.
  runtime_activity->resource =
      calibration ? 1000000000 + activity->calibration_id : activity->thread_id;
  runtime_activity->threadId = activity->thread_id;
  if (!hasDeviceResource(runtime_activity->device,
                         runtime_activity->resource)) {
    std::string name;
    if (calibration) {
      name = fmt::format("Spyre calibration (device {})",
                         activity->calibration_id - 1);
    } else {
      const auto found =
          threadNames_.find({activity->process_id, activity->thread_id});
      if (found != threadNames_.end() && !found->second.empty()) {
        name = fmt::format("{} ({})", found->second, activity->thread_id);
      }
    }
    // Missing metadata only affects the label; numeric identity and flows
    // remain.
    recordThreadStream(runtime_activity->device, runtime_activity->resource,
                       name);
  }
  if (calibration) {
    runtime_activity->addMetadata("calibration_device",
                                  activity->calibration_id - 1);
    runtime_activity->addMetadata("host_thread", activity->thread_id);
  }
  if (activity->external_id)
    runtime_activity->addMetadata("external_correlation",
                                  activity->external_id);
  if (activity->user_external_id)
    runtime_activity->addMetadata("user_external_correlation",
                                  activity->user_external_id);
  const bool isLaunch =
      activity->cbid >= AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB &&
      activity->cbid <= AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_DMO;
  const bool flowStart = !calibration && isLaunch &&
                         deviceActivityIds_.count(activity->correlation_id);
  runtime_activity->flow.id = flowStart ? activity->correlation_id : 0;
  runtime_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  runtime_activity->flow.start = flowStart;
  runtime_activity->linked = linked;
  runtime_activity->addMetadata("correlation", activity->correlation_id);

  if (activity->cbid == AIUPTI_RUNTIME_TRACE_CBID_SUBMIT_TO_HARDWARE &&
      !calibration && linked && linked->deviceId() == activity->process_id &&
      linked->timestamp() <= runtime_activity->startTime) {
    const auto flowId = ++nextSubmissionFlowId_;
    runtime_activity->submissionFlowId = flowId;

    // A CPU op can submit several batches, while each Kineto activity carries
    // only one flow endpoint. Add a zero-duration marker per batch at the CPU
    // origin instead of copying or mutating the original CPU activity. Keep it
    // in the trace buffer so deferred exports retain the endpoint's lifetime.
    auto source = std::make_unique<RuntimeTraceActivity>(
        traceBuffer_.span, libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
        "aiuCpuLaunch");
    source->startTime = source->endTime = linked->timestamp();
    source->device = linked->deviceId();
    source->resource = linked->resourceId();
    source->threadId = linked->getThreadId();
    source->id = activity->correlation_id;
    source->linked = linked;
    source->submissionFlowId = flowId;
    source->flow.type = libkineto::kLinkAsyncCpuGpu;
    source->flow.start = true;
    source->addMetadata("external_correlation", activity->external_id);
    source->addMetadata("correlation", activity->correlation_id);
    traceBuffer_.activities.push_back(std::move(source));
    traceBuffer_.activities.back()->log(*logger);
  }

  switch ((AIUpti_runtime_api_trace_cbid)activity->cbid) {
    case AIUPTI_RUNTIME_TRACE_CBID_SUBMIT_TO_HARDWARE:
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB:
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_CMPT:
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_DMI:
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_DMO:
      runtime_activity->addMetadata("num_cbs", activity->data);
      break;

    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_DTOF:
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_MTOF:
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_FTOD:
    case AIUPTI_RUNTIME_TRACE_CBID_FILE_TRANSFER_FTOM:
    case AIUPTI_RUNTIME_TRACE_CBID_DATA_TRANSFER_DTOH:
    case AIUPTI_RUNTIME_TRACE_CBID_DATA_TRANSFER_HTOD:
      runtime_activity->addMetadata("bytes", activity->data);
      break;

    case AIUPTI_RUNTIME_TRACE_CBID_ISSUE_BARRIER:
      runtime_activity->addMetadata("fence_size", activity->data);
      break;

    case AIUPTI_RUNTIME_TRACE_CBID_P2P_RDMA_MULTICAST:
      runtime_activity->addMetadata("bytes", activity->data);
      break;

    case AIUPTI_RUNTIME_TRACE_CBID_P2P_RDMA_WAIT:
    case AIUPTI_RUNTIME_TRACE_CBID_SCHEDULE_P2P_RDMA_WAIT:
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_P2P_RDMA_WAIT:
      runtime_activity->addMetadata("num_sids", activity->data);
      break;

    case AIUPTI_RUNTIME_TRACE_CBID_SCHEDULE_P2P_RDMA_SEND:
    case AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_P2P_RDMA_SEND:
      runtime_activity->addMetadata("num_dest_ranks", activity->data);
      break;

    case AIUPTI_RUNTIME_TRACE_CBID_INITIALIZE_P2P:
      runtime_activity->addMetadata("world_size", activity->data);
      break;

    default:
      break;
  }
  runtime_activity->log(*logger);
}

void AiuptiActivityProfilerSession::handleKernelActivity(
    const AIUpti_ActivityCompute* activity, libkineto::ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const libkineto::ITraceActivity* linked =
      linkedActivity(activity->correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(traceBuffer_.span,
                                libkineto::ActivityType::CONCURRENT_KERNEL,
                                activity->name);
  auto& kernel_activity = traceBuffer_.activities.back();
  kernel_activity->startTime = activity->start;
  kernel_activity->endTime = activity->end;
  kernel_activity->addMetadataQuoted(
      "timestamp_source", activity->timestamps_estimated
                              ? "estimated (host completion, 1 ns/tick)"
                              : "calibrated");
  kernel_activity->id = activity->correlation_id;
  kernel_activity->device = activity->device_id;
  kernel_activity->resource = activity->stream_id;
  kernel_activity->threadId = activity->stream_id;
  kernel_activity->flow.id = launchActivityIds_.count(activity->correlation_id)
                                 ? activity->correlation_id
                                 : 0;
  kernel_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  kernel_activity->flow.start = 0;
  kernel_activity->linked = linked;
  kernel_activity->addMetadata("queued", activity->queued);
  kernel_activity->addMetadata("submitted", activity->submitted);
  kernel_activity->addMetadata("device", kernel_activity->deviceId());
  kernel_activity->addMetadata("stream", activity->stream_id);
  kernel_activity->addMetadataQuoted("context",
                                     std::to_string(activity->context_id));
  kernel_activity->addMetadata("correlation", activity->correlation_id);
  addCollMetadata(kernel_activity, activity);
  if (const auto key = spyre::extractKernelProvenanceKey(activity->name)) {
    kernel_activity->addMetadataQuoted("provenance_key", *key);
    if (const auto ids = spyre::lookupKernelProvenance(*key)) {
      kernel_activity->addMetadata("debug_handles",
                                   nlohmann::json(*ids).dump());
    }
  }
  // Compute records always carry counters, so emit unconditionally: an
  // all-zero array here signals a counter failure rather than absence.
  kernel_activity->addMetadata("cycles_ts", cyclesTsJson(activity));

  recordStream(kernel_activity->device, kernel_activity->resource);
  kernel_activity->log(*logger);
}

template <class memory_activity_type>
inline std::string bandwidth(memory_activity_type* activity) {
  auto duration = activity->end - activity->start;
  auto bytes = activity->bytes;
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

inline std::string memoryCopyOperationName(uint8_t kind) {
  switch (kind) {
    case (uint8_t)AIUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case (uint8_t)AIUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case (uint8_t)AIUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "PtoP";
    default:
      break;
  }
  return "<unknown>";
}

inline uint32_t getBaseResourceId(const AIUpti_ActivityMemcpy* activity) {
  return activity->copy_kind * 100;
}

inline uint32_t getBaseResourceId(const AIUpti_ActivityMemory* activity) {
  return 400;
}

inline uint32_t getBaseResourceId(const AIUpti_ActivityMemset* activity) {
  return 400;  // put memset and memory release on the same PID
}

template <class memory_activity_type>
uint32_t AiuptiActivityProfilerSession::getResourceId(
    memory_activity_type* activity) {
  uint32_t overlap_offset = 0;
  uint32_t base_resource_id = getBaseResourceId(activity);
  const auto it =
      activeThreadMap_.find({activity->device_id, base_resource_id});
  if (it != activeThreadMap_.end()) {
    std::vector<int64_t>& last_active_times = (*it).second;
    bool found_useable_thread = false;
    for (auto i = 0; i < last_active_times.size(); ++i) {
      if (last_active_times[i] <= activity->start) {
        found_useable_thread = true;
        overlap_offset = i;
        last_active_times[i] = activity->end;
        break;
      }
    }
    if (!found_useable_thread) {
      last_active_times.push_back(activity->end);
      overlap_offset = last_active_times.size() - 1;
    }
    activeThreadMap_[{activity->device_id, base_resource_id}][overlap_offset] =
        activity->end;
  } else {
    activeThreadMap_[{activity->device_id, base_resource_id}] = {activity->end};
  }
  return base_resource_id + overlap_offset;
}

template uint32_t AiuptiActivityProfilerSession::getResourceId<
    AIUpti_ActivityMemcpy>(AIUpti_ActivityMemcpy* activity);
template uint32_t AiuptiActivityProfilerSession::getResourceId<
    AIUpti_ActivityMemory>(AIUpti_ActivityMemory* activity);
template uint32_t AiuptiActivityProfilerSession::getResourceId<
    AIUpti_ActivityMemset>(AIUpti_ActivityMemset* activity);

void AiuptiActivityProfilerSession::handleMemcpyActivity(
    const AIUpti_ActivityMemcpy* activity, libkineto::ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const libkineto::ITraceActivity* linked =
      linkedActivity(activity->correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(
      traceBuffer_.span, libkineto::ActivityType::GPU_MEMCPY,
      fmt::format("Memcpy ({})", memoryCopyOperationName(activity->copy_kind)));
  auto& memcpy_activity = traceBuffer_.activities.back();
  memcpy_activity->startTime = activity->start;
  memcpy_activity->endTime = activity->end;
  memcpy_activity->addMetadataQuoted(
      "timestamp_source", activity->timestamps_estimated
                              ? "estimated (host completion, 1 ns/tick)"
                              : "calibrated");
  memcpy_activity->id = activity->correlation_id;
  memcpy_activity->device = activity->device_id;
  memcpy_activity->resource = getResourceId(activity);
  memcpy_activity->threadId = activity->stream_id;
  memcpy_activity->flow.id = launchActivityIds_.count(activity->correlation_id)
                                 ? activity->correlation_id
                                 : 0;
  memcpy_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  memcpy_activity->flow.start = 0;
  memcpy_activity->linked = linked;
  memcpy_activity->addMetadataQuoted(
      "call", memoryCopyOperationName(activity->copy_kind));
  memcpy_activity->addMetadata("device", memcpy_activity->deviceId());
  memcpy_activity->addMetadataQuoted("context",
                                     std::to_string(activity->context_id));
  memcpy_activity->addMetadata("correlation", activity->correlation_id);
  memcpy_activity->addMetadata("memory operation id", activity->copy_kind);
  memcpy_activity->addMetadata("bytes", activity->bytes);
  memcpy_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));
  memcpy_activity->addMetadata("stream", activity->stream_id);
  // AIUpti_ActivityMemcpy does not say whether the record came from a
  // command buffer (real counters) or a standalone flex MEMCPY (always 0;
  // today flex reports every DMA this way), so omit the key when all slots
  // are 0. Gate on record kind instead once libaiupti marks which records
  // carry counters.
  if (hasCyclesTs(activity)) {
    memcpy_activity->addMetadata("cycles_ts", cyclesTsJson(activity));
  }
  addCollMetadata(memcpy_activity, activity);

  if (memcpy_activity->resource == getBaseResourceId(activity)) {
    recordMemoryStream(memcpy_activity->device, memcpy_activity->resource,
                       fmt::format("Memcpy ({}):", memoryCopyOperationName(
                                                       activity->copy_kind)));
  } else {
    recordMemoryStream(memcpy_activity->device, memcpy_activity->resource, " ");
  }
  memcpy_activity->log(*logger);
}

inline std::string memoryOperationName(uint8_t kind) {
  switch (kind) {
    case (uint8_t)AIUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION:
      return "Allocation";
    case (uint8_t)AIUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE:
      return "Release";
    default:
      break;
  }
  return "<unknown>";
}

void AiuptiActivityProfilerSession::handleMemoryActivity(
    const AIUpti_ActivityMemory* activity, libkineto::ActivityLogger* logger) {
  // do not track memory allocation events because they are the same as memset
  if (activity->memory_operation_type ==
      (uint8_t)AIUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE) {
    traceBuffer_.span.opCount += 1;
    traceBuffer_.gpuOpCount += 1;
    const libkineto::ITraceActivity* linked =
        nullptr;  // Allocation IDs are separate from submission batch IDs.
    traceBuffer_.emplace_activity(
        traceBuffer_.span, libkineto::ActivityType::PRIVATEUSE1_DRIVER,
        fmt::format("Memory ({})",
                    memoryOperationName(activity->memory_operation_type)));
    auto& mem_activity = traceBuffer_.activities.back();
    mem_activity->startTime = activity->start;
    mem_activity->endTime = activity->end;
    // Memory management uses host timestamps, not calibrated device ticks.
    mem_activity->addMetadataQuoted("timestamp_source", "host");
    mem_activity->id = activity->correlation_id;
    mem_activity->device = activity->device_id;
    mem_activity->resource = getResourceId(activity);
    mem_activity->threadId = activity->stream_id;
    mem_activity->flow.id = 0;
    mem_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
    mem_activity->flow.start = 0;
    mem_activity->linked = linked;
    mem_activity->addMetadataQuoted(
        "call", memoryOperationName(activity->memory_operation_type));
    mem_activity->addMetadata("device", mem_activity->deviceId());
    mem_activity->addMetadataQuoted("context",
                                    std::to_string(activity->process_id));
    mem_activity->addMetadata("correlation", activity->correlation_id);
    mem_activity->addMetadata("memory operation id",
                              activity->memory_operation_type);
    mem_activity->addMetadata("bytes", activity->bytes);
    mem_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));
    mem_activity->addMetadata("stream", activity->stream_id);

    if (mem_activity->resource == getBaseResourceId(activity)) {
      recordMemoryStream(mem_activity->device, mem_activity->resource,
                         "Memory management:");
    } else {
      recordMemoryStream(mem_activity->device, mem_activity->resource, " ");
    }
    mem_activity->log(*logger);
  }

  // Create event for AIU memory view
  traceBuffer_.span.opCount += 1;
  traceBuffer_.emplace_activity(traceBuffer_.span,
                                libkineto::ActivityType::CPU_INSTANT_EVENT,
                                "[memory]");
  auto& memory_event = traceBuffer_.activities.back();

  memory_event->startTime = activity->start;
  memory_event->addMetadataQuoted("timestamp_source", "host");

  // Following convention where all memory events are put on the
  // CPU thread. "Device Type" will denote CPU vs. AIU memory events
  // 0 (CPU), 1 (AIU)
  memory_event->device = libkineto::systemThreadId();
  memory_event->resource = libkineto::systemThreadId();

  int64_t bytes = static_cast<int64_t>(activity->bytes);
  if (activity->memory_operation_type ==
      AIUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE) {
    bytes *= -1;
  }
  totalAllocatedBytes_ += bytes;
  memory_event->addMetadata("Total Reserved", 0);
  memory_event->addMetadata("Total Allocated", totalAllocatedBytes_);
  memory_event->addMetadata("Bytes", bytes);
  memory_event->addMetadata("Addr", activity->address);
  memory_event->addMetadata("Device Id", activity->device_id);
  memory_event->addMetadata("Device Type", 1);

  memory_event->log(*logger);
}

void AiuptiActivityProfilerSession::handleMemsetActivity(
    const AIUpti_ActivityMemset* activity, libkineto::ActivityLogger* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const libkineto::ITraceActivity* linked =
      nullptr;  // Allocation IDs are separate from submission batch IDs.
  traceBuffer_.emplace_activity(traceBuffer_.span,
                                libkineto::ActivityType::GPU_MEMSET,
                                "Memset (Device)");
  auto& memset_activity = traceBuffer_.activities.back();
  memset_activity->startTime = activity->start;
  memset_activity->endTime = activity->end;
  // Libaiupti represents host-timed allocations as MEMSET records.
  memset_activity->addMetadataQuoted("timestamp_source", "host");
  memset_activity->id = activity->correlation_id;
  memset_activity->device = activity->device_id;
  // TODO(mcalman): investigate why memset activities are being processed out
  // of order This prevents us from using getResourceId which handles overlap
  memset_activity->resource = getBaseResourceId(activity);
  memset_activity->threadId = activity->stream_id;
  memset_activity->flow.id = 0;
  memset_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  memset_activity->flow.start = 0;
  memset_activity->linked = linked;
  memset_activity->addMetadataQuoted("call", "Memset");
  memset_activity->addMetadata("device", memset_activity->deviceId());
  memset_activity->addMetadataQuoted("context",
                                     std::to_string(activity->context_id));
  memset_activity->addMetadata("correlation", activity->correlation_id);
  memset_activity->addMetadata("bytes", activity->bytes);
  memset_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));
  memset_activity->addMetadata("stream", activity->stream_id);

  if (memset_activity->resource == getBaseResourceId(activity)) {
    recordMemoryStream(memset_activity->device, memset_activity->resource,
                       "Memory management:");
  } else {
    recordMemoryStream(memset_activity->device, memset_activity->resource, " ");
  }
  memset_activity->log(*logger);
}

void AiuptiActivityProfilerSession::handlePtiActivity(
    const AIUpti_Activity* record, libkineto::ActivityLogger* logger) {
  switch (record->kind) {
    case (uint8_t)AIUPTI_ACTIVITY_KIND_THREAD:
      // Names were collected by the prepass; metadata is not a timeline span.
      break;
    case (uint8_t)AIUPTI_ACTIVITY_KIND_RUNTIME:
      handleRuntimeActivity(reinterpret_cast<const AIUpti_ActivityAPI*>(record),
                            logger);
      break;
    case (uint8_t)AIUPTI_ACTIVITY_KIND_CMPT:
      handleKernelActivity(
          reinterpret_cast<const AIUpti_ActivityCompute*>(record), logger);
      break;
    case (uint8_t)AIUPTI_ACTIVITY_KIND_MEMCPY:
      handleMemcpyActivity(
          reinterpret_cast<const AIUpti_ActivityMemcpy*>(record), logger);
      break;
    case (uint8_t)AIUPTI_ACTIVITY_KIND_MEMSET:
      handleMemsetActivity(
          reinterpret_cast<const AIUpti_ActivityMemset*>(record), logger);
      break;
    case (uint8_t)AIUPTI_ACTIVITY_KIND_MEMORY:
      handleMemoryActivity(
          reinterpret_cast<const AIUpti_ActivityMemory*>(record), logger);
      break;
    default:
      errors_.push_back("Unexpected activity type: " +
                        std::to_string(record->kind));
      break;
  }
}

}  // namespace KINETO_NAMESPACE
