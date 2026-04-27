#!/usr/bin/env bash
# run_test.sh -- Single-entry-point test runner for torch-spyre OOT tests.
#
# Usage (single config):
#   bash run_test.sh /path/to/yaml/config [extra pytest args...]
#
# Usage (multiple configs -- merged at runtime, temp file cleaned up on exit):
#   bash run_test.sh config_a.yaml config_b.yaml [config_c.yaml ...] [extra pytest args...]
#   bash run_test.sh config_a.yaml config_b.yaml -- [extra pytest args...]
#
# When more than one YAML file is supplied the configs are merged in order via
# spyre_test_utilities.py
#   - `files` entries with the same path are combined (tests deduplicated).
#   - `global` list keys form a superset; identical items are deduplicated.
#   - Conflicting scalar globals raise an error.
# The merged temp file is removed by the EXIT trap at the end of the run.
#
# For each test file, any TestCase subclass that is NOT already passed to
# instantiate_device_type_tests() is automatically wrapped: a temporary
# wrapper script is generated that imports the original file and appends
# the missing instantiate_device_type_tests() calls so the OOT framework
# can control those classes via the YAML config.  The wrapper is deleted
# after the run.  No upstream files are modified.
#
# Additionally, classes that ARE passed to instantiate_device_type_tests()
# upstream but with an `only_for` kwarg restricting them to specific devices
# (e.g. only_for=DEVICE_LIST_SUPPORT_PROFILING_TEST) are re-injected into
# the wrapper without `only_for`, so the spyre/privateuse1 device is included.

# YAML tag --> JUnit XML <properties>
# ------------------------------------
# Tags defined under yaml tags in the YAML are
# injected as <properties> elements directly into the JUnit XML after pytest
# finishes writing it.  This post-processing approach is used because pytest
# does not emit marker <properties> for unittest.TestCase items even with
# junit_family=xunit2 -- which seems to be a pytest limitation.

set -euo pipefail


if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path/to/test_suite_config.yaml> [extra pytest args...]" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Multi-config support
#
# Collect all leading *.yaml / *.yml arguments that resolve to real files as
# YAML configs.  The first non-YAML argument (or anything after "--") is the
# start of extra pytest args.  A single YAML argument is the original
# backward-compatible path and behaves exactly as before.
# ---------------------------------------------------------------------------
YAML_CONFIGS=()
EXTRA_PYTEST_ARGS=()
_parsing_yamls=1

for _arg in "$@"; do
    if [[ "$_arg" == "--" ]]; then
        _parsing_yamls=0
        continue
    fi
    if [[ $_parsing_yamls -eq 1 && ( "$_arg" == *.yaml || "$_arg" == *.yml ) && -f "$_arg" ]]; then
        YAML_CONFIGS+=("$(realpath "$_arg")")
    else
        _parsing_yamls=0
        EXTRA_PYTEST_ARGS+=("$_arg")
    fi
done

if [[ ${#YAML_CONFIGS[@]} -eq 0 ]]; then
    echo "ERROR: No YAML config file(s) found in the arguments." >&2
    echo "Usage: $0 <path/to/test_suite_config.yaml> [extra pytest args...]" >&2
    exit 1
fi

# MERGED_CONFIG_IS_TEMP=1 means we created the file and must delete it on EXIT.
MERGED_CONFIG_IS_TEMP=0

if [[ ${#YAML_CONFIGS[@]} -eq 1 ]]; then
    # -----------------------------------------------------------------------
    # Single-config path
    # -----------------------------------------------------------------------
    YAML_CONFIG="${YAML_CONFIGS[0]}"

    if [[ ! -f "$YAML_CONFIG" ]]; then
        echo "ERROR: YAML config not found: $YAML_CONFIG" >&2
        exit 1
    fi

    echo "[spyre_run] Using YAML config: $YAML_CONFIG"
else
    # -----------------------------------------------------------------------
    # Multi-config path -- merge via spyre_test_utilities.py
    # -----------------------------------------------------------------------
    echo "[spyre_run] Merging ${#YAML_CONFIGS[@]} YAML config(s):"
    for _c in "${YAML_CONFIGS[@]}"; do
        echo "[spyre_run]   $_c"
    done

    _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    _UTILITIES_PY=""
    if [[ -f "${_script_dir}/spyre_test_utilities.py" ]]; then
        _UTILITIES_PY="${_script_dir}/spyre_test_utilities.py"
    else
        _first_config_dir="$(dirname "${YAML_CONFIGS[0]}")"
        if [[ -f "${_first_config_dir}/spyre_test_utilities.py" ]]; then
            _UTILITIES_PY="${_first_config_dir}/spyre_test_utilities.py"
        fi
    fi

    if [[ -z "$_UTILITIES_PY" ]]; then
        echo "ERROR: spyre_test_utilities.py not found beside run_test.sh or beside the first config." >&2
        echo "       Place spyre_test_utilities.py in the same directory as run_test.sh." >&2
        exit 1
    fi

    YAML_CONFIG=$(python3 "$_UTILITIES_PY" "${YAML_CONFIGS[@]}") || {
        echo "ERROR: Failed to merge YAML configs." >&2
        exit 1
    }
    MERGED_CONFIG_IS_TEMP=1
    echo "[spyre_run] Merged config written to: $YAML_CONFIG"
fi

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
#    Returns JSON with these keys:
#      all                   - ALL TestCase subclass names found in the file
#      device_type           - classes passed to instantiate_device_type_tests()
#                              WITHOUT an only_for kwarg (fully open; already
#                              handled for all devices including spyre)
#      device_type_restricted
#                            - classes passed to instantiate_device_type_tests()
#                              WITH an only_for kwarg (restricted to specific
#                              devices; privateuse1/spyre is typically excluded,
#                              so these classes need re-injection without only_for)
#      parametrized          - classes passed to instantiate_parametrized_tests()
#      uncontrolled          - TestCase subclasses not passed to ANY instantiate_*
#                              call at all (need fresh injection)
#      needs_injection       - union of uncontrolled + device_type_restricted
#                              (all classes that require a wrapper injection)
#      plain_no_device       - subset of needs_injection whose test methods have
#                              no `device` parameter
#
#   Re-inject restricted classes
#     When upstream calls:
#     instantiate_device_type_tests(Cls, globals(), only_for=SOME_LIST)
#     the framework only generates device-specific subclasses for the devices
#     listed in SOME_LIST.  If "privateuse1"/"spyre" is absent from that list,
#     no spyre variant is ever created and TorchTestBase never sees the class.
#     Re-injecting via the wrapper (without only_for) lets TorchTestBase
#     control the class through the YAML config like any other test class.
#     Example: TestMemoryProfilerTimeline is called with
#       only_for=DEVICE_LIST_SUPPORT_PROFILING_TEST
#     which typically excludes privateuse1.
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

def _call_has_only_for_kwarg(call_node):
    """Return True if the Call node has an `only_for` keyword argument."""
    return any(kw.arg == "only_for" for kw in call_node.keywords)

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
    # Classify instantiate_device_type_tests() calls:
    #   without only_for  -> fully open, framework already controls all devices
    #   with    only_for  -> restricted; spyre/privateuse1 likely excluded
    device_type_open       = set()   # no only_for kwarg
    device_type_restricted = set()   # has only_for kwarg
    parametrized_instantiated = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        fname = ""
        if isinstance(func, ast.Name):        fname = func.id
        elif isinstance(func, ast.Attribute): fname = func.attr

        if fname == "instantiate_device_type_tests" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                cls_name = arg.id
                if _call_has_only_for_kwarg(node):
                    device_type_restricted.add(cls_name)
                else:
                    device_type_open.add(cls_name)
        elif fname == "instantiate_parametrized_tests" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                parametrized_instantiated.add(arg.id)
    # A class that appears in BOTH open and restricted sets (e.g. the file
    # calls instantiate_device_type_tests twice for the same class, once with
    # only_for and once without) is treated as open: the open call already
    # covers all devices including spyre.
    device_type_restricted -= device_type_open
    
    # "Fully handled" = open device_type + parametrized
    # (restricted is NOT fully handled for spyre)
    fully_handled = device_type_open | parametrized_instantiated

    # uncontrolled: never passed to any instantiate_* call
    uncontrolled = sorted(set(all_classes) - fully_handled - device_type_restricted)

    # needs_injection: everything the wrapper must re-inject
    needs_injection = sorted(set(uncontrolled) | device_type_restricted)

    # Plain classes (no device arg in any test method) within needs_injection
    plain_no_device = sorted(
        cls for cls in needs_injection if not all_classes.get(cls, False)
    )

    print(json.dumps({
        "all":                    sorted(all_classes),
        "device_type":            sorted(device_type_open),
        "device_type_restricted": sorted(device_type_restricted),
        "parametrized":           sorted(parametrized_instantiated),
        "uncontrolled":           uncontrolled,
        "needs_injection":        needs_injection,
        "plain_no_device":        plain_no_device,
    }))

analyze(sys.argv[1])
'

# ---------------------------------------------------------------------------
# 9. Wrapper generator
#
#    For any test file that has classes needing injection (uncontrolled OR
#    device_type_restricted), generate a temporary wrapper .py placed beside
#    the original so that conftest.py discovery, relative imports, and
#    sys.path all work identically.
#
#    The wrapper star-imports the original module (picking up all existing
#    instantiate_* calls) then appends one instantiate_device_type_tests()
#    call per class that needs injection.
#
#    Two categories of injected classes:
#
#    1. uncontrolled  -- classes never passed to any instantiate_* call.
#       Fresh injection; TorchTestBase sees them for the first time.
#
#    2. device_type_restricted -- classes already called upstream with
#       only_for=..., but that only_for list excludes privateuse1/spyre.
#       The upstream call produced zero spyre-device subclasses.  The wrapper
#       re-calls instantiate_device_type_tests() without only_for so
#       TorchTestBase generates the spyre variant and can apply YAML config.
#       The upstream subclasses for other devices (cuda, cpu, etc.) remain
#       untouched in the global scope from the star-import; our call adds
#       the spyre variant alongside them.
#
#    Safety for plain (no-device) classes:
#      TorchTestBase._should_run() replaces test methods with a SkipTest
#      wrapper BEFORE the test body executes.  The `device` arg injected
#      by instantiate_device_type_tests() is therefore never seen by the
#      original method body when YAML mode is skip.  If a plain test is
#      listed as mandatory_success/xfail a warning is emitted (it would
#      fail at runtime with a device-arg TypeError).
#
#    Naming:  <original_stem>__oot_wrapper.py  (cleaned up by EXIT trap)
# ---------------------------------------------------------------------------

WRAPPER_FILES=()

_cleanup_wrappers() {
    for wf in "${WRAPPER_FILES[@]+"${WRAPPER_FILES[@]}"}"; do
        [[ -f "$wf" ]] && rm -f "$wf" && \
            echo "[spyre_run] Cleaned up wrapper: $wf"
    done
    # Remove merged config temp file (only if we created it)
    if [[ $MERGED_CONFIG_IS_TEMP -eq 1 && -n "${YAML_CONFIG:-}" && -f "$YAML_CONFIG" ]]; then
        rm -f "$YAML_CONFIG"
        echo "[spyre_run] Removed merged temp config: $YAML_CONFIG"
    fi
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

    local needs_injection_str plain_str restricted_str uncontrolled_str
    needs_injection_str=$(echo "$result" | python3 -c "
import json,sys; d=json.load(sys.stdin); print(' '.join(d['needs_injection']))
")
    plain_str=$(echo "$result" | python3 -c "
import json,sys; d=json.load(sys.stdin); print(' '.join(d['plain_no_device']))
")
    restricted_str=$(echo "$result" | python3 -c "
import json,sys; d=json.load(sys.stdin); print(' '.join(d['device_type_restricted']))
")
    uncontrolled_str=$(echo "$result" | python3 -c "
import json,sys; d=json.load(sys.stdin); print(' '.join(d['uncontrolled']))
")

    if [[ -z "$needs_injection_str" ]]; then
        return 0   # all classes already fully framework-controlled for spyre
    fi

    read -r -a NEEDS_INJECTION_CLASSES <<< "$needs_injection_str"
    local -a PLAIN_CLASSES=()
    [[ -n "$plain_str" ]] && read -r -a PLAIN_CLASSES <<< "$plain_str"
    local -a RESTRICTED_CLASSES=()
    [[ -n "$restricted_str" ]] && read -r -a RESTRICTED_CLASSES <<< "$restricted_str"
    local -a UNCONTROLLED_CLASSES=()
    [[ -n "$uncontrolled_str" ]] && read -r -a UNCONTROLLED_CLASSES <<< "$uncontrolled_str"
    # Warn about plain classes -- they are safe only when YAML skips them.
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
    # Report uncontrolled classes (never instantiated upstream at all)
    if [[ ${#UNCONTROLLED_CLASSES[@]} -gt 0 ]]; then
        echo "[spyre_run] Injecting instantiate_device_type_tests for uncontrolled classes in: $(basename "$test_file")"
        for cls in "${UNCONTROLLED_CLASSES[@]}"; do
            echo "[spyre_run]   -> $cls"
        done
    fi
    # Report restricted classes (instantiated upstream with only_for, excluding spyre)
    if [[ ${#RESTRICTED_CLASSES[@]} -gt 0 ]]; then
        echo "[spyre_run] Re-injecting instantiate_device_type_tests (dropping only_for) for restricted classes in: $(basename "$test_file")"
        for cls in "${RESTRICTED_CLASSES[@]}"; do
            echo "[spyre_run]   -> $cls  (upstream: only_for=... excluded privateuse1)"
        done
    fi

    local conftest_path
    conftest_path="${original_dir}/__oot_conftest_${original_stem}.py"

    echo "[spyre_run] Generating wrapper: $(basename "$wrapper_path")"
    # Build the per-class injection block first (pure bash, no heredoc nesting issue).
    # All classes use _pre_import_classes which is populated before the star-import.
    local injection_block=""
    for cls in "${NEEDS_INJECTION_CLASSES[@]}"; do
        injection_block+="
_cls_${cls} = _pre_import_classes.get('${cls}')
if _cls_${cls} is None:
    raise RuntimeError('Could not find original class ${cls} in pre-import of module ${module_name}')
globals().setdefault('${cls}', _cls_${cls})
_instantiate(_cls_${cls}, globals())
_restore_staticmethods(_cls_${cls}, globals())
"
    done
    # Separate quoted lists for restricted vs uncontrolled classes --
    # each is retrieved differently from the private module.
    local quoted_restricted_list=""
    for cls in "${RESTRICTED_CLASSES[@]}"; do
        quoted_restricted_list+="'${cls}', "
    done
    local quoted_uncontrolled_list=""
    for cls in "${UNCONTROLLED_CLASSES[@]}"; do
        quoted_uncontrolled_list+="'${cls}', "
    done
    # Write the wrapper using a heredoc — no quoting issues with embedded text.
    # WRAPPER_EOF is unquoted so shell variables ($module_name, $test_file,
    # $quoted_class_list, $injection_block) expand; all other Python lines
    # contain no $ and are passed through literally.
    cat > "$wrapper_path" <<WRAPPER_EOF
# Auto-generated by run_test.sh -- DO NOT EDIT -- deleted after run
# Wrapper for: $test_file
#
# Injects instantiate_device_type_tests() for:
#   - uncontrolled classes: never passed to any instantiate_* upstream.
#   - restricted classes: passed upstream with only_for=... that excluded
#     privateuse1/spyre. Re-injected here WITHOUT only_for so TorchTestBase
#     can generate the spyre variant and control it via the YAML config.
# Classes with no 'device' arg are safe when YAML mode is 'skip'.
 
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

# Ensure the spyre tests directory is on sys.path before any torch imports.
# torch.testing._internal.common_device_type uses runpy to load
# TORCH_TEST_DEVICES (spyre_test_base_common.py), which imports spyre_*
# modules.  Those modules must be findable on sys.path at that point.
# PYTHONPATH is set by run_test.sh but Python only applies it at interpreter
# startup -- subsequent imports don't re-read it, so we inject it explicitly.
for _p in reversed(_os.environ.get('PYTHONPATH', '').split(_os.pathsep)):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------
# Pre-import: capture original class objects BEFORE the star-import runs the
# upstream instantiate_device_type_tests() calls that delete them from scope.
#
# Two categories require different retrieval strategies:
#
# RESTRICTED classes (had only_for=... upstream):
#   instantiate_device_type_tests deletes the class name from the module
#   dict at the end of its run, so by the time exec_module() returns the
#   name is gone.
#
#   The module does "from torch.testing._internal.common_device_type import
#   instantiate_device_type_tests" at the top, binding the real function
#   directly into the module's namespace at import time.  Patching
#   _pre_mod.instantiate_device_type_tests after module_from_spec() has no
#   effect because the binding is resolved during exec_module(), not looked
#   up dynamically.
#
#   Solution: patch the function on the SOURCE MODULE
#   (torch.testing._internal.common_device_type) before exec_module() runs,
#   then restore it immediately after.  This intercepts every call site
#   regardless of how the function was imported.
#
# UNCONTROLLED classes (never passed to instantiate_device_type_tests):
#   Nothing deletes them, so they are still present on _pre_mod after
#   exec_module() completes.  A plain getattr suffices.
# ---------------------------------------------------------------------------

import importlib.util as _ilu

_pre_import_classes = {}
_restricted_names = set([${quoted_restricted_list}])

def _do_pre_import():
    """Capture original class objects before the star-import deletes them."""
    import torch.testing._internal.common_device_type as _cdtype
    
    real_fn = _cdtype.instantiate_device_type_tests

    def _capturing_instantiate(cls, *args, **kwargs):
        if cls.__name__ in _restricted_names:
            # Save the class AND copies of all its test methods BEFORE
            # the real call deletes them via delattr(generic_test_class, name).
            # instantiate_device_type_tests mutates the original class by
            # removing all test methods from it at the end of its run.
            # We must copy them now so our later re-injection has methods to work with.
            import copy as _copy
            _pre_import_classes[cls.__name__] = cls
            _saved_methods = cls.__name__ + '__saved_tests'
            _pre_import_classes[_saved_methods] = {
                name: getattr(cls, name)
                for name in list(cls.__dict__.keys())
                if name.startswith('test')
            }
        return real_fn(cls, *args, **kwargs)
    # Patch at source so every from-import of instantiate_device_type_tests
    # picks up the shim during exec_module.
    _cdtype.instantiate_device_type_tests = _capturing_instantiate
    try:
        _private_spec = _ilu.spec_from_file_location(
            '_oot_pre_${module_name}',
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '${module_name}.py'),
        )
        _pre_mod = _ilu.module_from_spec(_private_spec)
        _private_spec.loader.exec_module(_pre_mod)
    finally:
        _cdtype.instantiate_device_type_tests = real_fn
    # Restricted classes captured via shim; uncontrolled still on _pre_mod.
    for _name in [${quoted_uncontrolled_list}]:
        if hasattr(_pre_mod, _name):
            _pre_import_classes[_name] = getattr(_pre_mod, _name)

_do_pre_import()

# Restore test methods that the real instantiate_device_type_tests deleted
# from restricted classes via delattr(). Without this, our re-injection call
# finds an empty class and generates no test variants.
for _rname in _restricted_names:
    _saved_key = _rname + '__saved_tests'
    _saved = _pre_import_classes.get(_saved_key, {})
    _cls = _pre_import_classes.get(_rname)
    if _cls is not None and _saved:
        for _mname, _mfn in _saved.items():
            if not hasattr(_cls, _mname):
                setattr(_cls, _mname, _mfn)

# ---------------------------------------------------------------------------
# Star-import: executes the original module in THIS file's global scope,
# running all upstream instantiate_* calls (including restricted only_for
# ones).  After this line, restricted class names are deleted from globals()
# by the upstream instantiate_device_type_tests(), but _pre_import_classes
# still holds the original class objects captured above.
# ---------------------------------------------------------------------------
from ${module_name} import *  # noqa: F401,F403

import inspect as _inspect
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests as _instantiate,
)

# ---------------------------------------------------------------------------
# @staticmethod preservation
#
# instantiate_device_type_tests copies non-test class members via
# getattr(), which UNWRAPS @staticmethod descriptors into plain functions.
# When test methods subsequently call self.static_helper(), Python treats
# the plain function as an instance method and injects self as the first
# positional arg, causing:
#   TypeError: Cls.method() takes 0 positional arguments but 1 was given
#
# _restore_staticmethods() uses inspect.getattr_static (which does NOT
# unwrap descriptors) to find all @staticmethod members on the original
# class, then re-applies them on every generated device-specific subclass.
# ---------------------------------------------------------------------------
def _restore_staticmethods(original_cls, scope):
    prefix = original_cls.__name__
    for name, obj in list(scope.items()):
        if (isinstance(obj, type)
                and name.startswith(prefix)
                and name != prefix):
            for attr in dir(original_cls):
                desc = _inspect.getattr_static(original_cls, attr, None)
                if isinstance(desc, staticmethod):
                    setattr(obj, attr, desc)

# ---------------------------------------------------------------------------
# Inject instantiate_device_type_tests for all classes needing injection,
# using pre-captured class objects from _pre_import_classes.
#
# For *restricted* classes: the upstream only_for call produced no spyre
# subclass. We call again WITHOUT only_for so TorchTestBase generates the
# privateuse1/spyre variant alongside the existing cuda/cpu ones.
#
# For *uncontrolled* classes: first-time injection; TorchTestBase sees them
# for the first time here.
#
# After each injection, restore @staticmethod descriptors that
# instantiate_device_type_tests unwrapped during member copying.
#
${injection_block}

WRAPPER_EOF

    # Generate a conftest.py that patches DEVICE_LIST_SUPPORT_PROFILING_TEST
    # before pytest collects any tests. This is the only reliable point to
    # patch it: conftest.py runs before module import during collection, so
    # the from-import in test_memory_profiler.py has not yet bound the
    # original tuple. By patching here, every subsequent from-import binds
    # the list that includes 'privateuse1'.
    cat > "$conftest_path" <<CONFTEST_EOF
# Auto-generated by run_test.sh -- DO NOT EDIT -- deleted after run
import torch.testing._internal.common_utils as _cu
_dl = getattr(_cu, 'DEVICE_LIST_SUPPORT_PROFILING_TEST', None)
if _dl is not None and 'privateuse1' not in _dl:
    _cu.DEVICE_LIST_SUPPORT_PROFILING_TEST = list(_dl) + ['privateuse1']
CONFTEST_EOF

    WRAPPER_FILES+=("$wrapper_path" "$conftest_path")
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
    stale_conftest="${original_dir}/__oot_conftest_${original_stem}.py"
    if [[ -f "$stale_wrapper" ]]; then
        echo "[spyre_run]   Removing stale wrapper: $stale_wrapper"
        rm -f "$stale_wrapper"
    fi
    if [[ -f "$stale_conftest" ]]; then
        echo "[spyre_run]   Removing stale conftest: $stale_conftest"
        rm -f "$stale_conftest"
    fi
done

# ---------------------------------------------------------------------------
# 11. Build the final run list (original or wrapper per file)
# ---------------------------------------------------------------------------
echo "[spyre_run] Checking for uncontrolled/restricted TestCase classes..."
echo ""

RUN_FILES=()
for test_file in "${TEST_FILES[@]}"; do
    generate_wrapper_if_needed "$test_file"
    RUN_FILES+=("$_RUN_FILE")
done

echo ""

# ---------------------------------------------------------------------------
# 12. Run pytest for each file - original / wrapper depending on TestClass
#
# After pytest writes the JUnit XML, a Python post-processor injects YAML
# tags as <properties> elements directly into the XML.
#
# Two regex fixes make matching robust:
#   1. (?<![a-z])name="..."  avoids matching 'name' inside 'classname="..."'
#   2. yaml_class in classname  handles dotted XML classnames like
#      "test.test_binary_ufuncs.TestBinaryUfuncsPRIVATEUSE1"
# ---------------------------------------------------------------------------

_XML_INJECT_PY='
import sys, re
from pathlib import Path
try:
    import yaml
except ImportError:
    sys.exit(0)

xml_path, yaml_path = sys.argv[1], sys.argv[2]

data = yaml.safe_load(open(yaml_path)) or {}
tag_map = {}
for fe in data.get("test_suite_config", {}).get("files", []):
    for te in fe.get("tests", []):
        tags = sorted(set(te.get("tags", []) or []))
        if not tags:
            continue
        for name in te.get("names", []):
            name = name.strip()
            if name:
                tag_map.setdefault(name, set()).update(tags)

def match_tags(classname, testname):
    matched = set()
    for yaml_name, tags in tag_map.items():
        if "::" in yaml_name:
            yaml_class, yaml_method = yaml_name.split("::", 1)
        else:
            yaml_class, yaml_method = "", yaml_name
        if ((yaml_class and yaml_method
                and yaml_class in classname
                and testname.startswith(yaml_method))
                or (yaml_method and not yaml_class
                    and testname.startswith(yaml_method))):
            matched.update(tags)
    return sorted(matched)

def build_props(tags):
    return "<properties>" + "".join(
        f"<property name=\"tag\" value=\"{t}\"/>" for t in tags
    ) + "</properties>"

def inject_full(m):
    attrs, content = m.group(1), m.group(2)
    if "<properties>" in content:
        return m.group(0)
    cn = re.search(r"classname=\"([^\"]*)\"", attrs)
    tn = re.search(r"(?<![a-z])name=\"([^\"]*)\"", attrs)
    if not cn or not tn:
        return m.group(0)
    tags = match_tags(cn.group(1), tn.group(1))
    if not tags:
        return m.group(0)
    return f"<testcase{attrs}>{build_props(tags)}{content}</testcase>"

def inject_self_closing(m):
    attrs = m.group(1)
    cn = re.search(r"classname=\"([^\"]*)\"", attrs)
    tn = re.search(r"(?<![a-z])name=\"([^\"]*)\"", attrs)
    if not cn or not tn:
        return m.group(0)
    tags = match_tags(cn.group(1), tn.group(1))
    if not tags:
        return m.group(0)
    return f"<testcase{attrs}>{build_props(tags)}</testcase>"

xml = Path(xml_path).read_text()
xml = re.sub(r"<testcase([^>]*)>(.*?)</testcase>", inject_full,        xml, flags=re.DOTALL)
xml = re.sub(r"<testcase([^>]*?)/>",               inject_self_closing, xml)
Path(xml_path).write_text(xml)
print(f"[spyre_run] Tags injected into XML: {xml_path}", flush=True)
'

OVERALL_EXIT=0

# ---------------------------------------------------------------------------
# Extract the caller-supplied --junit-xml destination BEFORE the
# per-file loop so we can:
#   1. Route each per-file pytest run to a shard XML (e.g. report__shard_0.xml)
#      instead of the shared final path, preventing later runs from overwriting
#      the XML and the already-injected <properties> tags of earlier runs.
#   2. Merge all shards into the original --junit-xml path after all runs finish.
# ---------------------------------------------------------------------------
_FINAL_XML_PATH=""
_prev_arg=""
for _arg in "${EXTRA_PYTEST_ARGS[@]+"${EXTRA_PYTEST_ARGS[@]}"}"; do
    case "$_arg" in
        --junit-xml=*) _FINAL_XML_PATH="${_arg#--junit-xml=}" ;;
        --junit-xml)   : ;;
        *)  [[ "$_prev_arg" == "--junit-xml" ]] && _FINAL_XML_PATH="$_arg" ;;
    esac
    _prev_arg="$_arg"
done

# Build a copy of EXTRA_PYTEST_ARGS with --junit-xml stripped out.
# Each per-file run injects its own shard path instead.
_EXTRA_NO_XML=()
_skip_next=0
for _arg in "${EXTRA_PYTEST_ARGS[@]+"${EXTRA_PYTEST_ARGS[@]}"}"; do
    if [[ $_skip_next -eq 1 ]]; then
        _skip_next=0
        continue
    fi
    case "$_arg" in
        --junit-xml=*) ;;                  # drop combined form
        --junit-xml)   _skip_next=1 ;;     # drop flag; value dropped next iteration
        *)             _EXTRA_NO_XML+=("$_arg") ;;
    esac
done

# Accumulate shard paths for the final merge step.
_XML_SHARDS=()

# ---------------------------------------------------------------------------
# XML shard merger: combines N JUnit XML files (each produced by a separate
# pytest run) into one, summing suite-level counters and concatenating all
# <testcase> elements (which already carry their injected <properties> tags).
# ---------------------------------------------------------------------------
_XML_MERGE_PY='
import sys, re
from pathlib import Path

out_path    = sys.argv[1]
shard_paths = sys.argv[2:]

all_cases   = []
total_tests = 0
total_err   = 0
total_fail  = 0
total_skip  = 0
total_time  = 0.0

def _attr(xml, name, default="0"):
    m = re.search(rf"{name}=\"([^\"]*)\"", xml)
    return m.group(1) if m else default

for sp in shard_paths:
    txt = Path(sp).read_text()
    suite_m = re.search(r"<testsuite([^>]*)>", txt)
    if suite_m:
        attrs = suite_m.group(1)
        total_tests += int(_attr(attrs, "tests"))
        total_err   += int(_attr(attrs, "errors"))
        total_fail  += int(_attr(attrs, "failures"))
        total_skip  += int(_attr(attrs, "skipped"))
        try:
            total_time += float(_attr(attrs, "time", "0"))
        except ValueError:
            pass
    # Collect full and self-closing <testcase> blocks.
    blocks  = re.findall(r"<testcase[^>]*>.*?</testcase>", txt, re.DOTALL)
    blocks += re.findall(r"<testcase[^>]*/>",              txt)
    all_cases.extend(blocks)

merged = (
    "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
    "<testsuites>"
    f"<testsuite name=\"pytest\" tests=\"{total_tests}\" "
    f"errors=\"{total_err}\" failures=\"{total_fail}\" "
    f"skipped=\"{total_skip}\" time=\"{total_time:.3f}\">"
    + "\n".join(all_cases)
    + "</testsuite></testsuites>"
)
Path(out_path).write_text(merged)
print(f"[spyre_run] Merged {len(shard_paths)} XML shard(s) -> {out_path}", flush=True)
'

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

    # Derive a shard XML path for this file's pytest run.
    _SHARD_XML=""
    if [[ -n "$_FINAL_XML_PATH" ]]; then
        _SHARD_XML="${_FINAL_XML_PATH%.xml}__shard_${i}.xml"
        _XML_SHARDS+=("$_SHARD_XML")
    fi

    # Build per-file args: base args (--junit-xml stripped) + shard path.
    if [[ -n "$_SHARD_XML" ]]; then
        _FILE_PYTEST_ARGS=("${_EXTRA_NO_XML[@]+"${_EXTRA_NO_XML[@]}"}" "--junit-xml=${_SHARD_XML}")
    else
        _FILE_PYTEST_ARGS=("${_EXTRA_NO_XML[@]+"${_EXTRA_NO_XML[@]}"}")
    fi

    # ---------------------------------------------------------------------------
    # -m marker pre-flight
    #
    # When a -m MARKEXPR is present, probe whether this specific file has any
    # tests that match it before running.  The probe uses --collect-only which
    # is fast as no test execution happens and runs from the file's own directory
    # so conftest.py files are discovered correctly.
    #
    # If the probe finds 0 matching tests (exit code 5) the -m flag is stripped
    # from _FILE_PYTEST_ARGS so the file's tests all run normally. 
    # the marker filter applies to files that USE that marker
    # family; files that don't use it are unaffected.
    #
    # ---------------------------------------------------------------------------
    _HAS_M=0
    for _a in "${_EXTRA_NO_XML[@]+"${_EXTRA_NO_XML[@]}"}"; do
        [[ "$_a" == "-m" ]] && { _HAS_M=1; break; }
    done

    if [[ $_HAS_M -eq 1 ]]; then
        # Extract just the -m args for the probe (no --junit-xml, no -v, etc.)
        _PROBE_ARGS=()
        _take_next=0
        for _a in "${_EXTRA_NO_XML[@]+"${_EXTRA_NO_XML[@]}"}"; do
            if [[ $_take_next -eq 1 ]]; then
                _PROBE_ARGS+=("$_a")
                _take_next=0
                continue
            fi
            if [[ "$_a" == "-m" ]]; then
                _PROBE_ARGS+=("$_a")
                _take_next=1
            fi
        done

        _probe_exit=0
        (cd "$run_dir" && python3 -m pytest "$run_basename" \
            "${_PROBE_ARGS[@]}" --collect-only -q 2>/dev/null)
        _probe_exit=$?

        if [[ $_probe_exit -eq 5 ]]; then
            # 0 tests match this marker in this file — strip -m from args.
            echo "[spyre_run] -m filter matched 0 tests in $(basename "$original_file"), running without -m" >&2
            _ARGS_NO_M=()
            _skip_m=0
            for _a in "${_FILE_PYTEST_ARGS[@]+"${_FILE_PYTEST_ARGS[@]}"}"; do
                if [[ $_skip_m -eq 1 ]]; then _skip_m=0; continue; fi
                if [[ "$_a" == "-m" ]]; then _skip_m=1; continue; fi
                _ARGS_NO_M+=("$_a")
            done
            _FILE_PYTEST_ARGS=("${_ARGS_NO_M[@]}")
        fi
    fi

    (
        cd "$run_dir"
        python3 -m pytest "$run_basename" "${_FILE_PYTEST_ARGS[@]}" || true
    )

    _exit=$?

    # Post-process XML to inject YAML tags as <properties>.
    if [[ -n "$_SHARD_XML" && -f "$_SHARD_XML" ]]; then
        python3 -c "$_XML_INJECT_PY" "$_SHARD_XML" "$YAML_CONFIG" || true
    fi

    if [[ $_exit -ne 0 ]]; then
        echo "[spyre_run] WARNING: pytest exited with code $_exit for $original_file" >&2
        OVERALL_EXIT=$_exit
    fi
done

# ---------------------------------------------------------------------------
# Merge all XML shards into the final output path requested by the caller.
# ---------------------------------------------------------------------------
if [[ -n "$_FINAL_XML_PATH" && ${#_XML_SHARDS[@]} -gt 0 ]]; then
    _existing_shards=()
    for _s in "${_XML_SHARDS[@]}"; do
        [[ -f "$_s" ]] && _existing_shards+=("$_s")
    done

    if [[ ${#_existing_shards[@]} -eq 1 ]]; then
        # Single file run: just rename the shard, no merge needed.
        mv "${_existing_shards[0]}" "$_FINAL_XML_PATH"
        echo "[spyre_run] Single XML shard moved to: $_FINAL_XML_PATH"
    elif [[ ${#_existing_shards[@]} -gt 1 ]]; then
        python3 -c "$_XML_MERGE_PY" "$_FINAL_XML_PATH" "${_existing_shards[@]}" || true
        # Clean up shards after successful merge.
        for _s in "${_existing_shards[@]}"; do
            rm -f "$_s"
        done
    else
        echo "[spyre_run] WARNING: No XML shards found to merge." >&2
    fi
fi

echo ""
echo "[spyre_run] Done. Overall exit code: $OVERALL_EXIT"
exit $OVERALL_EXIT