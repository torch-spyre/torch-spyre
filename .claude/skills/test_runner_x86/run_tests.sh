#!/bin/bash
# Shell wrapper for test runner - executes PyTorch tests on Spyre/OpenShift pods
# Tests are run directly from pod using pod's internal test configs

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_tests.py"

# Default values
TIMEOUT=3600
STREAM_LOGS=false
FRAMEWORK_PATH="/home/senuser/torch-spyre/tests"
OUTPUT_DIR="./test_results"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 <namespace> <pod_name> <test_config_path_in_pod> [options]

Arguments:
  namespace                OpenShift namespace where the pod is running
  pod_name                Name of the running pod
  test_config_path_in_pod Path to test YAML config file inside the pod

Options:
  --timeout <seconds>        Test execution timeout (default: 3600)
  --stream                   Stream logs in real-time during execution
  --framework-path <path>    Path to test framework in pod (default: /home/senuser/torch-spyre/tests)
  --output-dir <path>        Local directory for test results (default: ./test_results)
  -h, --help                 Display this help message

Examples:
  # Run a single test config from pod
  $0 cicd-project anjali-torch-spyre-may14 /home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml

  # Run with streaming logs
  $0 cicd-project anjali-torch-spyre-may14 /home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml --stream

  # Run with custom timeout
  $0 cicd-project anjali-torch-spyre-may14 /home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml --timeout 7200

  # Run with custom output directory
  $0 cicd-project anjali-torch-spyre-may14 /home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml --output-dir ./my_results

EOF
    exit 0
}

# Check if help is requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
fi

# Check minimum arguments
if [ $# -lt 3 ]; then
    print_error "Insufficient arguments provided"
    echo ""
    usage
fi

# Parse required arguments
NAMESPACE="$1"
POD_NAME="$2"
TEST_CONFIG_PATH="$3"
shift 3

# Parse optional arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout)
            TIMEOUT="$2"
            EXTRA_ARGS+=("--timeout" "$2")
            shift 2
            ;;
        --stream)
            STREAM_LOGS=true
            EXTRA_ARGS+=("--stream")
            shift
            ;;
        --framework-path)
            FRAMEWORK_PATH="$2"
            EXTRA_ARGS+=("--framework-path" "$2")
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            EXTRA_ARGS+=("--output-dir" "$2")
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if oc CLI is available
if ! command -v oc &> /dev/null; then
    print_error "OpenShift CLI (oc) not found. Please install it first."
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install it first."
    exit 1
fi

# Display configuration
print_info "Test Runner Configuration:"
echo "  Namespace:        $NAMESPACE"
echo "  Pod Name:         $POD_NAME"
echo "  Test Config:      $TEST_CONFIG_PATH (in pod)"
echo "  Timeout:          ${TIMEOUT}s"
echo "  Stream Logs:      $STREAM_LOGS"
echo "  Framework Path:   $FRAMEWORK_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Execute the Python script
print_info "Starting test execution..."
echo ""

if python3 "$PYTHON_SCRIPT" "$NAMESPACE" "$POD_NAME" "$TEST_CONFIG_PATH" "${EXTRA_ARGS[@]}"; then
    echo ""
    print_success "Test execution completed successfully!"
    
    # Display summary if available
    if [ -f "${OUTPUT_DIR}/summary.json" ]; then
        echo ""
        print_info "Test Summary:"
        if command -v jq &> /dev/null; then
            jq -r '
                "  Total Tests:  \(.total_tests)",
                "  Passed:       \(.passed)",
                "  Failed:       \(.failed)",
                "  XFailed:      \(.xfailed)",
                "  Skipped:      \(.skipped)",
                "  Duration:     \(.duration_seconds)s"
            ' "${OUTPUT_DIR}/summary.json"
        else
            cat "${OUTPUT_DIR}/summary.json"
        fi
        echo ""
        print_info "Full results available in: ${OUTPUT_DIR}/"
    fi
    
    exit 0
else
    EXIT_CODE=$?
    echo ""
    print_error "Test execution failed with exit code: $EXIT_CODE"
    
    # Check for common issues
    if [ -f "${OUTPUT_DIR}/errors.log" ]; then
        print_warning "Check ${OUTPUT_DIR}/errors.log for details"
    fi
    
    echo ""
    print_info "Troubleshooting tips:"
    echo "  1. Verify pod is running: oc get pod $POD_NAME -n $NAMESPACE"
    echo "  2. Check pod logs: oc logs $POD_NAME -n $NAMESPACE"
    echo "  3. Verify test config exists: oc exec $POD_NAME -n $NAMESPACE -- ls -la $TEST_CONFIG_PATH"
    echo "  4. Check test framework: oc exec $POD_NAME -n $NAMESPACE -- ls -la $FRAMEWORK_PATH"
    
    exit $EXIT_CODE
fi