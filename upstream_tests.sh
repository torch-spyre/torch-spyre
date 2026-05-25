#!/bin/bash

mkdir -p logs

CONFIGS=(
  "test_view_ops_config.yaml"
  "test_profiler_config.yaml"
)

printf "\n%-40s %-15s %-15s\n" \
  "TEST NAME" "REAL TIME" "MAX_MEM(MB)"

printf "%-40s %-15s %-15s\n" \
  "---------" "---------" "-----------"

for CONFIG in "${CONFIGS[@]}"; do

    TEST_NAME=$(basename "$CONFIG" .yaml)

    (
        time bash tests/run_test.sh \
            "tests/configs/upstream_tests/${CONFIG}" \
            --capture=sys \
            -v
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

    printf "%-40s %-15s %-15s\n" \
        "$TEST_NAME" \
        "$REAL_TIME" \
        "$MAX_MB"

done
