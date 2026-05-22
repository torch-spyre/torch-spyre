#!/bin/bash

mkdir -p logs

CONFIGS=(
  "torch_spyre_tests/test_device_enum_config.yaml"
  "torch_spyre_tests/test_fallbacks_config.yaml"
  "torch_spyre_tests/test_modules_config.yaml"
  "torch_spyre_tests/test_regex_config.yaml"
  "torch_spyre_tests/test_spyre_config.yaml"
  "torch_spyre_tests/test_spyre_lazy_silent_config.yaml"
  "torch_spyre_tests/test_stream_config.yaml"
  "torch_spyre_tests/test_spyre_profiler_config.yaml"
  "torch_spyre_tests/test_prepare_kernel_config.yaml"

  "torch_spyre_tests/inductor/test_building_blocks_config.yaml"
  "torch_spyre_tests/inductor/test_codegen_config.yaml"
  "torch_spyre_tests/inductor/test_decomp_config.yaml"
  "torch_spyre_tests/inductor/test_inductor_fx_passes_config.yaml"
  "torch_spyre_tests/inductor/test_normalization_scalars_config.yaml"
  "torch_spyre_tests/inductor/test_inductor_ops_config.yaml"
  "torch_spyre_tests/inductor/test_ops_lx_planning_config.yaml"
  "torch_spyre_tests/inductor/test_inductor_scalar_config.yaml"
  "torch_spyre_tests/inductor/test_logging_config.yaml"
  "torch_spyre_tests/inductor/test_dedup_constants_config.yaml"
  "torch_spyre_tests/inductor/test_padding_config.yaml"
  "torch_spyre_tests/inductor/test_overwrite.yaml"
  "torch_spyre_tests/inductor/test_restickify_config.yaml"
  "torch_spyre_tests/inductor/test_scratchpad_patterns_config.yaml"
  "torch_spyre_tests/inductor/test_scratchpad_use_config.yaml"
  "torch_spyre_tests/inductor/test_dtype_scalars_config.yaml"
  "torch_spyre_tests/inductor/test_cache_config.yaml"

  "torch_spyre_tests/tensors/test_coordinates_config.yaml"
  "torch_spyre_tests/tensors/test_it_space_splits_config.yaml"
  "torch_spyre_tests/tensors/test_tensor_layout_config.yaml"
)

printf "\n%-60s %-15s %-15s\n" "TEST NAME" "REAL TIME" "MAX_MEM(MB)"
printf "%-60s %-15s %-15s\n" "---------" "---------" "-----------"

for CONFIG in "${CONFIGS[@]}"; do

    TEST_NAME=$(basename "$CONFIG" .yaml)

    (
        time bash tests/run_test.sh "tests/configs/${CONFIG}" -v
    ) > "logs/${TEST_NAME}.log" 2>&1 &

    PID=$!

    MAX=0

    while kill -0 $PID 2>/dev/null; do

        MEM=$(ps -eo rss= --ppid $PID | awk '{sum+=$1} END {print sum+0}')

        if [ "$MEM" -gt "$MAX" ]; then
            MAX=$MEM
        fi

        sleep 0.2
    done

    wait $PID

    REAL_TIME=$(grep "^real" "logs/${TEST_NAME}.log" | awk '{print $2}')

    MAX_MB=$((MAX / 1024))

    printf "%-60s %-15s %-15s\n" \
        "$TEST_NAME" \
        "$REAL_TIME" \
        "$MAX_MB"

done
