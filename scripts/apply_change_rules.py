#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set


def get_repo_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def load_json_mapping(file_path: Path) -> Dict:
    if not file_path.exists():
        return {}
    with open(file_path, "r") as f:
        return json.load(f)


def config_to_suite_object(config_path: str) -> dict:
    filename = Path(config_path).stem
    name = filename[:-7] if filename.endswith("_config") else filename
    return {"name": name, "config": config_path}


def parse_yaml_matrix_file(matrix_file_path: Path) -> list:
    configs = []
    if not matrix_file_path.exists():
        return configs
    try:
        import yaml

        with open(matrix_file_path, "r") as f:
            data = yaml.safe_load(f)
        suite_list = (
            data.get("jobs", {})
            .get("test", {})
            .get("strategy", {})
            .get("matrix", {})
            .get("suite", [])
        )
        for item in suite_list:
            if isinstance(item, dict) and "config" in item:
                configs.append(item["config"])
    except Exception:
        content = matrix_file_path.read_text(encoding="utf-8")
        import re

        configs = sorted(
            list(set(re.findall(r"config:\s*(torch_spyre_tests/[^\s\n]+)", content)))
        )
    return configs


def main():
    repo_root = get_repo_root()
    mappings_dir = repo_root / "mappings"
    matrix_file = repo_root / ".github" / "workflows" / "_test_matrix.yaml"

    source_to_suite = load_json_mapping(mappings_dir / "source_to_suite.json")
    test_to_suite_data = load_json_mapping(mappings_dir / "test_to_suite.json")
    test_to_suite = test_to_suite_data.get("test_file_to_configs", {})
    all_suites = parse_yaml_matrix_file(matrix_file)

    # 1. Read clean input variables passed directly from the environment shell
    changed_files_str = os.environ.get("CHANGED_FILES", "")
    added_files_str = os.environ.get("ADDED_FILES", "")

    if not changed_files_str:
        print("No file changes discovered or provided.")
        sys.exit(0)

    changed_files = changed_files_str.strip().split()
    added_files = set(added_files_str.strip().split()) if added_files_str else set()

    suites_to_run: Set[str] = set()
    reasons: List[str] = []
    has_new_source_file = False

    for file_path in changed_files:
        is_added = file_path in added_files

        # Rule Category A: Config Mutations
        if (
            file_path.startswith("tests/configs/")
            and file_path.endswith(".yaml")
            and "example" not in file_path.lower()
        ):
            suite_rel_path = file_path.replace("tests/configs/", "")
            suites_to_run.add(suite_rel_path)
            tag = "[NEW CONFIG]" if is_added else "[MODIFIED CONFIG]"
            reasons.append(f"{tag} {file_path} -> targeted execution")

        # Rule Category B: Test Script Mutations
        elif file_path.startswith("tests/") and file_path.endswith(".py"):
            if file_path in test_to_suite:
                suites_to_run.update(test_to_suite[file_path])
                tag = "[NEW TEST]" if is_added else "[MODIFIED TEST]"
                reasons.append(
                    f"{tag} {file_path} -> runs mapped suites: {test_to_suite[file_path]}"
                )

        # Rule Category C: Core Framework Source Mutations
        elif file_path.startswith("torch_spyre/") and file_path.endswith(".py"):
            if is_added:
                has_new_source_file = True
                reasons.append(
                    f"[NEW SOURCE FILE DETECTED] {file_path} -> triggering ALL matrix suites"
                )
            else:
                mapped_suite_names = source_to_suite.get(file_path, [])
                for log_suite_name in mapped_suite_names:
                    for real_suite_path in all_suites:
                        if Path(real_suite_path).stem == log_suite_name:
                            suites_to_run.add(real_suite_path)
                if mapped_suite_names:
                    reasons.append(
                        f"[MODIFIED SOURCE] {file_path} -> runs coverage mapped suites: {mapped_suite_names}"
                    )

    if has_new_source_file:
        suites_to_run.update(all_suites)

    # Intersection guarantees we never attempt running a missing/out-of-scope config
    final_filtered_suites = suites_to_run.intersection(set(all_suites))
    suite_objects = [
        config_to_suite_object(s) for s in sorted(list(final_filtered_suites))
    ]

    with open(mappings_dir / "selected_suites_by_rules.json", "w") as f:
        json.dump(
            {
                "suites": sorted(list(final_filtered_suites)),
                "count": len(final_filtered_suites),
                "reasons": reasons,
            },
            f,
            indent=2,
        )

    print(
        "================================================================================"
    )
    print("INTELLIGENT SELECTION REPORT")
    print(
        "================================================================================"
    )
    for reason in reasons:
        print(reason)
    print("\nFinal Selected Matrix Suites:")
    for suite in suite_objects:
        print(f"  - Name: {suite['name']} | Config: {suite['config']}")


if __name__ == "__main__":
    main()
