#!/usr/bin/env bash
# run_test.sh -- Single-entry-point test runner for torch-spyre OOT tests.
#
# Usage:
#   bash run_test.sh /path/to/test_suite_config.yaml [extra pytest args...]
#
# For each test file, any TestCase subclass that is NOT already passed to
# instantiate_device_type_tests() is automatically wrapped: a temporary
# wrapper script is generated that imports the original file and appends
# the missing instantiate_device_type_tests() calls so the OOT framework
# can control those classes via the YAML config.  The wrapper is deleted
# after the run.  No upstream files are modified.


set -euo pipefail


if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path/to/test_suite_config.yaml> [extra pytest args...]" >&2
    exit 1
fi

YAML_CONFIG="$(realpath "$1")"
shift
EXTRA_PYTEST_ARGS=("$@")

if [[ ! -f "$YAML_CONFIG" ]]; then
    echo "ERROR: YAML config not found: $YAML_CONFIG" >&2
    exit 1
fi

echo "[spyre_run] Using YAML config: $YAML_CONFIG"
YAML_DIR="$(dirname "$YAML_CONFIG")"

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

_walk_up_for_sentinel() {
    local dir sentinel
    dir="$(realpath "$1")"
    sentinel="$2"
    for _ in $(seq 1 12); do
        if [[ -e "$dir/$sentinel" ]]; then
            echo "$dir"
            return 0
        fi
        [[ "$dir" == "/" ]] && break
        dir="$(dirname "$dir")"
    done
    return 1
}

_find_sibling_with_sentinel() {
    local dir sentinel
    dir="$(realpath "$1")"
    sentinel="$2"
    for _ in $(seq 1 6); do
        dir="$(dirname "$dir")"
        [[ "$dir" == "/" ]] && break
        for sibling in "$dir"/*/; do
            [[ -f "${sibling}${sentinel}" ]] && { echo "${sibling%/}"; return 0; }
        done
    done
    return 1
}

# ---------------------------------------------------------------------------
# 2. Resolve and export TORCH_ROOT
# ---------------------------------------------------------------------------
echo "[spyre_run] Resolving TORCH_ROOT..."
if [[ -n "${TORCH_ROOT:-}" && -d "$TORCH_ROOT" ]]; then
    echo "[spyre_run]   already set: $TORCH_ROOT"
else
    TORCH_ROOT=""

    _found=$(python3 -c "
import torch, os
candidate = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
if os.path.isfile(os.path.join(candidate, 'test', 'test_binary_ufuncs.py')):
    print(candidate)
" 2>/dev/null) || true
    [[ -n "$_found" ]] && TORCH_ROOT="$_found"

    if [[ -z "$TORCH_ROOT" ]]; then
        TORCH_ROOT=$(_find_sibling_with_sentinel "$YAML_DIR" "test/test_binary_ufuncs.py" 2>/dev/null) || true
    fi

    if [[ -z "$TORCH_ROOT" ]]; then
        echo "ERROR: Could not locate PyTorch source root." >&2
        echo "       Expected pytorch/ as a sibling of your torch-spyre repo, or" >&2
        echo "       an editable install (pip install -e .)." >&2
        echo "       Set TORCH_ROOT explicitly if the layout differs." >&2
        exit 1
    fi
fi
export TORCH_ROOT
export PYTORCH_ROOT="$TORCH_ROOT"
echo "[spyre_run]   TORCH_ROOT=$TORCH_ROOT"

# ---------------------------------------------------------------------------
# 3. Resolve and export TORCH_DEVICE_ROOT
# ---------------------------------------------------------------------------
echo "[spyre_run] Resolving TORCH_DEVICE_ROOT..."
if [[ -n "${TORCH_DEVICE_ROOT:-}" && -d "$TORCH_DEVICE_ROOT" ]]; then
    echo "[spyre_run]   already set: $TORCH_DEVICE_ROOT"
else
    TORCH_DEVICE_ROOT=""

    _found=$(python3 -c "
import importlib.metadata, json, os
try:
    dist = importlib.metadata.distribution('torch_spyre')
    direct_url = os.path.join(str(dist._path), 'direct_url.json')
    if os.path.isfile(direct_url):
        data = json.load(open(direct_url))
        url = data.get('url', '')
        if url.startswith('file://'):
            candidate = url[len('file://'):]
            if os.path.isfile(os.path.join(candidate, 'tests', 'spyre_test_base_common.py')):
                print(candidate)
except Exception:
    pass
" 2>/dev/null) || true
    [[ -n "$_found" ]] && TORCH_DEVICE_ROOT="$_found"

    if [[ -z "$TORCH_DEVICE_ROOT" ]]; then
        _found=$(python3 -c "
import importlib.util, os
spec = importlib.util.find_spec('spyre_test_base_common')
if spec:
    print(os.path.dirname(os.path.dirname(os.path.abspath(spec.origin))))
" 2>/dev/null) || true
        [[ -n "$_found" ]] && TORCH_DEVICE_ROOT="$_found"
    fi

    if [[ -z "$TORCH_DEVICE_ROOT" ]]; then
        TORCH_DEVICE_ROOT=$(_walk_up_for_sentinel "$YAML_DIR" "tests/spyre_test_base_common.py" 2>/dev/null) || true
    fi

    if [[ -z "$TORCH_DEVICE_ROOT" ]]; then
        echo "ERROR: Could not locate torch-spyre source root." >&2
        echo "       Expected torch_spyre to be installed as an editable install" >&2
        echo "       (pip install -e .), or the repo adjacent to your YAML." >&2
        echo "       Set TORCH_DEVICE_ROOT explicitly if the layout differs." >&2
        exit 1
    fi
fi
export TORCH_DEVICE_ROOT
export TORCH_SPYRE_ROOT="$TORCH_DEVICE_ROOT"
echo "[spyre_run]   TORCH_DEVICE_ROOT=$TORCH_DEVICE_ROOT"

# ---------------------------------------------------------------------------
# 4. Export all framework environment variables
# ---------------------------------------------------------------------------
export PYTORCH_TESTING_DEVICE_ONLY_FOR="privateuse1"
export TORCH_TEST_DEVICES="${TORCH_DEVICE_ROOT}/tests/spyre_test_base_common.py"
export PYTORCH_TEST_CONFIG="$YAML_CONFIG"

_spyre_tests_path="${TORCH_DEVICE_ROOT}/tests"
case ":${PYTHONPATH:-}:" in
    *":$_spyre_tests_path:"*) ;;
    *) export PYTHONPATH="$_spyre_tests_path:${PYTHONPATH:-}" ;;
esac

echo ""
echo "[spyre_run] Environment set:"
echo "  TORCH_ROOT                      = $TORCH_ROOT"
echo "  TORCH_DEVICE_ROOT               = $TORCH_DEVICE_ROOT"
echo "  PYTORCH_TESTING_DEVICE_ONLY_FOR = $PYTORCH_TESTING_DEVICE_ONLY_FOR"
echo "  TORCH_TEST_DEVICES              = $TORCH_TEST_DEVICES"
echo "  PYTORCH_TEST_CONFIG             = $PYTORCH_TEST_CONFIG"
echo "  PYTHONPATH                      = $PYTHONPATH"
echo ""

# ---------------------------------------------------------------------------
# 5. Extract raw file paths from YAML
# ---------------------------------------------------------------------------
_extract_file_paths_from_yaml() {
    grep -E '^\s*(- )?path:\s' "$1" \
        | sed 's/.*path:[[:space:]]*//' \
        | sed 's/[[:space:]]*#.*//' \
        | sed '/^[[:space:]]*$/d'
}

echo "[spyre_run] Parsing YAML for test file paths..."
RAW_PATHS=()
while IFS= read -r line; do
    RAW_PATHS+=("$line")
done < <(_extract_file_paths_from_yaml "$YAML_CONFIG")

if [[ ${#RAW_PATHS[@]} -eq 0 ]]; then
    echo "ERROR: No file paths found in YAML config." >&2
    exit 1
fi

echo "[spyre_run] Found ${#RAW_PATHS[@]} path entry(s):"
for p in "${RAW_PATHS[@]}"; do
    echo "  $p"
done

# ---------------------------------------------------------------------------
# 6. Token expansion
# ---------------------------------------------------------------------------
_expand_path() {
    local p="$1"
    p="${p//\$\{TORCH_ROOT\}/$TORCH_ROOT}"
    p="${p//\$\{TORCH_DEVICE_ROOT\}/$TORCH_DEVICE_ROOT}"
    if command -v envsubst &>/dev/null; then
        p=$(echo "$p" | envsubst)
    fi
    echo "$p"
}

# ---------------------------------------------------------------------------
# 7. Expand globs and collect resolved test files
# ---------------------------------------------------------------------------
shopt -s globstar nullglob 2>/dev/null || true

TEST_FILES=()
for raw in "${RAW_PATHS[@]}"; do
    expanded=$(_expand_path "$raw")
    if [[ "$expanded" == *'*'* || "$expanded" == *'?'* ]]; then
        matched=( $expanded )
        if [[ ${#matched[@]} -eq 0 ]]; then
            echo "WARNING: Glob pattern matched no files: $expanded" >&2
        fi
        for f in "${matched[@]}"; do
            [[ -f "$f" ]] && TEST_FILES+=("$f")
        done
    else
        if [[ -f "$expanded" ]]; then
            TEST_FILES+=("$expanded")
        else
            echo "WARNING: Resolved path does not exist, skipping: $expanded" >&2
        fi
    fi
done

if [[ ${#TEST_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No test files resolved from YAML paths." >&2
    exit 1
fi

echo ""
echo "[spyre_run] Resolved test file(s):"
for f in "${TEST_FILES[@]}"; do
    echo "  $f"
done
echo ""

# ---------------------------------------------------------------------------
# 8. AST analyzer
#    Returns JSON: {all, device_type, parametrized, uncontrolled, plain_no_device}
#
#    "uncontrolled" = ALL TestCase subclasses not yet passed to
#    instantiate_device_type_tests(), regardless of whether their test
#    methods take a `device` arg or not.  ALL of these get wrapper injection.
#
#   
#      When the YAML says mode:skip or unlisted_test_mode:skip, TorchTestBase
#      replaces the test method with a unittest.SkipTest wrapper BEFORE the
#      test body ever executes -- the `device` arg is never passed to the
#      original method.  The framework therefore controls all such tests via
#      the YAML, including plain TestCase classes.
# ---------------------------------------------------------------------------
_ANALYZER_PY='
import ast, sys, json
from pathlib import Path

def class_methods_info(classdef):
    """Return (has_device_method, [all_test_method_names]) for a ClassDef."""
    methods = []
    has_device = False
    for node in ast.walk(classdef):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            if any(a.arg == "device" for a in node.args.args):
                has_device = True
            methods.append(node.name)
    return has_device, methods

def analyze(path):
    try:
        source = Path(path).read_text()
    except OSError as e:
        print(json.dumps({"error": str(e)})); return
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        print(json.dumps({"error": f"SyntaxError: {e}"})); return

    # ALL TestCase subclasses in this file
    all_classes = {}   # name -> has_device_method
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):        base_name = base.id
                elif isinstance(base, ast.Attribute): base_name = base.attr
                if "TestCase" in base_name or base_name.endswith("TestBase"):
                    has_device, _ = class_methods_info(node)
                    all_classes[node.name] = has_device
                    break

    device_type_instantiated = set()
    parametrized_instantiated = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call): continue
        func = node.func
        fname = ""
        if isinstance(func, ast.Name):        fname = func.id
        elif isinstance(func, ast.Attribute): fname = func.attr
        if fname == "instantiate_device_type_tests" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name): device_type_instantiated.add(arg.id)
        elif fname == "instantiate_parametrized_tests" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name): parametrized_instantiated.add(arg.id)

    already_handled = device_type_instantiated | parametrized_instantiated
    uncontrolled = sorted(set(all_classes) - already_handled)

    # Plain classes (no device arg in any test method) within uncontrolled
    plain_no_device = sorted(
        cls for cls in uncontrolled if not all_classes[cls]
    )

    print(json.dumps({
        "all":                  sorted(all_classes),
        "device_type":          sorted(device_type_instantiated),
        "parametrized":         sorted(parametrized_instantiated),
        "uncontrolled":         uncontrolled,
        "plain_no_device":      plain_no_device,
    }))

analyze(sys.argv[1])
'

# ---------------------------------------------------------------------------
# 9. Wrapper generator
#
#    For any test file that has uncontrolled classes (of ANY kind), generate
#    a temporary wrapper .py placed beside the original so that conftest.py
#    discovery, relative imports, and sys.path all work identically.
#
#    The wrapper star-imports the original module (picking up all existing
#    instantiate_* calls) then appends one instantiate_device_type_tests()
#    call per uncontrolled class.
#
#    Safety for plain (no-device) classes:
#      TorchTestBase._should_run() replaces test methods with a SkipTest
#      wrapper BEFORE the test body executes.  The `device` arg injected
#      by instantiate_device_type_tests() is therefore never seen by the
#      original method body when YAML mode is skip.  If a plain test is
#      listed as mandatory_success/xfail a warning is emitted (it would
#      fail at runtime with a device-arg TypeError).
#
#    Naming:  <original_stem>__oot_wrapper.py  (deleted by EXIT trap)
# ---------------------------------------------------------------------------

WRAPPER_FILES=()

_cleanup_wrappers() {
    for wf in "${WRAPPER_FILES[@]+"${WRAPPER_FILES[@]}"}"; do
        [[ -f "$wf" ]] && rm -f "$wf" && \
            echo "[spyre_run] Cleaned up wrapper: $wf"
    done
}
trap _cleanup_wrappers EXIT

# generate_wrapper_if_needed <test_file>
# Sets global _RUN_FILE to the path pytest should actually run.
_RUN_FILE=""
generate_wrapper_if_needed() {
    local test_file="$1"
    _RUN_FILE="$test_file"

    local result
    if ! result=$(python3 -c "$_ANALYZER_PY" "$test_file" 2>/dev/null); then
        echo "[spyre_run] WARNING: could not analyze $test_file -- running as-is" >&2
        return 0
    fi

    local err
    err=$(echo "$result" | python3 -c "
import json,sys; d=json.load(sys.stdin); print(d.get('error',''))
" 2>/dev/null) || true
    if [[ -n "$err" ]]; then
        echo "[spyre_run] WARNING: parse error in $test_file: $err -- running as-is" >&2
        return 0
    fi

    local uncontrolled_str plain_str
    uncontrolled_str=$(echo "$result" | python3 -c "
import json,sys; d=json.load(sys.stdin); print(' '.join(d['uncontrolled']))
")
    plain_str=$(echo "$result" | python3 -c "
import json,sys; d=json.load(sys.stdin); print(' '.join(d['plain_no_device']))
")

    if [[ -z "$uncontrolled_str" ]]; then
        return 0   # all classes already framework-controlled
    fi

    read -r -a UNCONTROLLED_CLASSES <<< "$uncontrolled_str"
    local -a PLAIN_CLASSES=()
    [[ -n "$plain_str" ]] && read -r -a PLAIN_CLASSES <<< "$plain_str"

    # Warn about plain classes -- they are safe only when YAML skips them.
    # If the user listed any as mandatory_success/xfail they will hit a
    # device-arg TypeError at runtime.
    if [[ ${#PLAIN_CLASSES[@]} -gt 0 ]]; then
        echo "[spyre_run] NOTE: the following classes have no 'device' arg in their"
        echo "[spyre_run]       test methods. They are safe under mode:skip but will"
        echo "[spyre_run]       fail at runtime if listed as mandatory_success/xfail:"
        for cls in "${PLAIN_CLASSES[@]}"; do
            echo "[spyre_run]         $cls"
        done
    fi

    local original_dir original_stem module_name wrapper_path
    original_dir="$(dirname "$test_file")"
    original_stem="$(basename "$test_file" .py)"
    module_name="$original_stem"
    wrapper_path="${original_dir}/${original_stem}__oot_wrapper.py"

    echo "[spyre_run] Injecting instantiate_device_type_tests for uncontrolled classes in: $(basename "$test_file")"
    for cls in "${UNCONTROLLED_CLASSES[@]}"; do
        echo "[spyre_run]   -> $cls"
    done
    echo "[spyre_run] Generating wrapper: $(basename "$wrapper_path")"

    {
        echo "# Auto-generated by run_test.sh -- DO NOT EDIT -- deleted after run"
        echo "# Wrapper for: $test_file"
        echo "#"
        echo "# Injects instantiate_device_type_tests() for ALL TestCase subclasses"
        echo "# not already registered, so TorchTestBase controls them via YAML."
        echo "# Classes with no 'device' arg are safe when YAML mode is 'skip'."
        echo ""
        echo "import sys as _sys, os as _os"
        echo "_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))"
        echo ""
        echo "# Star-import brings in all classes, helpers, and existing"
        echo "# instantiate_device_type_tests / instantiate_parametrized_tests calls."
        echo "from ${module_name} import *  # noqa: F401,F403"
        echo ""
        echo "import inspect as _inspect"
        echo "from torch.testing._internal.common_device_type import ("
        echo "    instantiate_device_type_tests as _instantiate,"
        echo ")"
        echo ""
        echo "# ---------------------------------------------------------------------------"
        echo "# @staticmethod preservation"
        echo "#"
        echo "# instantiate_device_type_tests copies non-test class members via"
        echo "# getattr(), which UNWRAPS @staticmethod descriptors into plain functions."
        echo "# When test methods subsequently call self.static_helper(), Python treats"
        echo "# the plain function as an instance method and injects self as the first"
        echo "# positional arg, causing:"
        echo "#   TypeError: Cls.method() takes 0 positional arguments but 1 was given"
        echo "#"
        echo "# _restore_staticmethods() uses inspect.getattr_static (which does NOT"
        echo "# unwrap descriptors) to find all @staticmethod members on the original"
        echo "# class, then re-applies them on every generated device-specific subclass."
        echo "# ---------------------------------------------------------------------------"
        echo "def _restore_staticmethods(original_cls, scope):"
        echo "    prefix = original_cls.__name__"
        echo "    for name, obj in list(scope.items()):"
        echo "        if (isinstance(obj, type)"
        echo "                and name.startswith(prefix)"
        echo "                and name != prefix):"
        echo "            for attr in dir(original_cls):"
        echo "                desc = _inspect.getattr_static(original_cls, attr, None)"
        echo "                if isinstance(desc, staticmethod):"
        echo "                    setattr(obj, attr, desc)"
        echo ""
        echo "# Inject instantiate_device_type_tests for every uncontrolled class."
        echo "# TorchTestBase replaces skipped tests before their body executes so"
        echo "# the injected device arg never reaches plain-test method bodies."
        echo "# After each injection, restore @staticmethod descriptors that"
        echo "# instantiate_device_type_tests unwrapped during member copying."
        for cls in "${UNCONTROLLED_CLASSES[@]}"; do
            # Save the class reference before _instantiate deletes it from globals().
            # instantiate_device_type_tests() calls del scope[class_name] at the end,
            # so the name is no longer resolvable after the call.
            echo "_cls_${cls} = ${cls}"
            echo "_instantiate(_cls_${cls}, globals())"
            echo "_restore_staticmethods(_cls_${cls}, globals())"
        done
    } > "$wrapper_path"

    WRAPPER_FILES+=("$wrapper_path")
    _RUN_FILE="$wrapper_path"
}

# ---------------------------------------------------------------------------
# 10. Clean up any stale wrappers from previous crashed/interrupted runs
#     before generating new ones, so pytest never picks up an old wrapper.
# ---------------------------------------------------------------------------
echo "[spyre_run] Cleaning up any stale OOT wrappers from previous runs..."
for test_file in "${TEST_FILES[@]}"; do
    original_dir="$(dirname "$test_file")"
    original_stem="$(basename "$test_file" .py)"
    stale_wrapper="${original_dir}/${original_stem}__oot_wrapper.py"
    if [[ -f "$stale_wrapper" ]]; then
        echo "[spyre_run]   Removing stale wrapper: $stale_wrapper"
        rm -f "$stale_wrapper"
    fi
done

# ---------------------------------------------------------------------------
# 11. Build the final run list (original or wrapper per file)
# ---------------------------------------------------------------------------
echo "[spyre_run] Checking for uncontrolled TestCase classes..."
echo ""

RUN_FILES=()
for test_file in "${TEST_FILES[@]}"; do
    generate_wrapper_if_needed "$test_file"
    RUN_FILES+=("$_RUN_FILE")
done

echo ""

# ---------------------------------------------------------------------------
# 12. Run pytest for each file (original or wrapper)
# ---------------------------------------------------------------------------
OVERALL_EXIT=0

for i in "${!RUN_FILES[@]}"; do
    run_file="${RUN_FILES[$i]}"
    original_file="${TEST_FILES[$i]}"
    run_dir="$(dirname "$run_file")"
    run_basename="$(basename "$run_file")"

    echo "========================================================================"
    if [[ "$run_file" != "$original_file" ]]; then
        echo "[spyre_run] Running (via OOT wrapper): $original_file"
    else
        echo "[spyre_run] Running: $run_file"
    fi
    echo "========================================================================"

    (
        cd "$run_dir"
        python3 -m pytest "$run_basename" "${EXTRA_PYTEST_ARGS[@]}" || true
    )
    _exit=$?
    if [[ $_exit -ne 0 ]]; then
        echo "[spyre_run] WARNING: pytest exited with code $_exit for $original_file" >&2
        OVERALL_EXIT=$_exit
    fi
done

echo ""
echo "[spyre_run] Done. Overall exit code: $OVERALL_EXIT"
exit $OVERALL_EXIT