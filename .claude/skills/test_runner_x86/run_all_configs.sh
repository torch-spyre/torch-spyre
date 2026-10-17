#!/bin/bash
# Run all tests from a specified config directory

set -e

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <namespace> <pod_name> <config_directory_path_in_pod>"
    echo ""
    echo "Example:"
    echo "  $0 cicd-project my-pod /home/senuser/torch-spyre/tests/configs/upstream_tests_beta"
    exit 1
fi

NAMESPACE="$1"
POD_NAME="$2"
BASE_PATH="$3"

# Discover all YAML config files in the specified directory
echo "Discovering test configs in: $BASE_PATH"
TEST_CONFIGS=()
while IFS= read -r line; do
    TEST_CONFIGS+=("$line")
done < <(oc exec "$POD_NAME" -n "$NAMESPACE" -- bash -c "ls -1 $BASE_PATH/*.yaml 2>/dev/null | xargs -n1 basename" || echo "")

if [ ${#TEST_CONFIGS[@]} -eq 0 ] || [ -z "${TEST_CONFIGS[0]}" ]; then
    echo "Error: No YAML config files found in $BASE_PATH"
    exit 1
fi

echo "=========================================="
echo "Running all configs from: $BASE_PATH"
echo "Namespace: $NAMESPACE"
echo "Pod: $POD_NAME"
echo "Total configs: ${#TEST_CONFIGS[@]}"
echo "=========================================="
echo ""

PASSED=0
FAILED=0

for config in "${TEST_CONFIGS[@]}"; do
    echo "----------------------------------------"
    echo "Running: $config"
    echo "----------------------------------------"
    
    if ./run_tests.sh "$NAMESPACE" "$POD_NAME" "$BASE_PATH/$config"; then
        echo "✅ $config - PASSED"
        ((PASSED++))
    else
        echo "❌ $config - FAILED"
        ((FAILED++))
    fi
    echo ""
done

echo "=========================================="
echo "Summary:"
echo "  Total: ${#TEST_CONFIGS[@]}"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo "=========================================="

exit 0
