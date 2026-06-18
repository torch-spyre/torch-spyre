#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def config_to_suite_object(config_path: str) -> dict:
    filename = Path(config_path).stem
    name = filename[:-7] if filename.endswith("_config") else filename
    return {"name": name, "config": config_path}


def parse_yaml_matrix_file(matrix_file_path: Path) -> list:
    """Parses the workflow matrix file to extract test suite config definitions."""
    configs = []
    if not matrix_file_path.exists():
        print(f"Warning: Matrix workflow file {matrix_file_path} not found.")
        return configs

    try:
        import yaml

        with open(matrix_file_path, "r") as f:
            data = yaml.safe_load(f)
        # Parse only the standard 'test' job matrix array
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
    except ImportError:
        # Fallback regex parsing if pyyaml is missing from the environment
        content = matrix_file_path.read_text(encoding="utf-8")
        # Extract matches following config lines
        matches = re.findall(r"config:\s*(torch_spyre_tests/[^\s\n]+)", content)
        configs = sorted(list(set(matches)))

    return configs


def main():
    repo_root = get_repo_root()
    matrix_file = repo_root / ".github" / "workflows" / "_test_matrix.yaml"
    mappings_dir = repo_root / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse configs from target reusable workflow file
    suite_configs = sorted(parse_yaml_matrix_file(matrix_file))

    # 2. Convert to targeted dictionary matrix objects
    suite_objects = [config_to_suite_object(cfg) for cfg in suite_configs]

    # 3. Save directly within repository space
    output_file = mappings_dir / "selected_suites_by_rules.json"
    reasons = [
        f"[FIRST RUN / NO COVERAGE] Running all {len(suite_configs)} configurations extracted from matrix workflow."
    ]

    with open(output_file, "w") as f:
        json.dump(
            {"suites": suite_configs, "count": len(suite_configs), "reasons": reasons},
            f,
            indent=2,
        )

    print(
        "================================================================================"
    )
    print("FIRST RUN DETERMINED: NO LOCAL COVERAGE DIRECTORY FOUND")
    print(
        "================================================================================"
    )
    print(
        f"Parsed and selected all {len(suite_objects)} suites from {matrix_file.name}:"
    )
    for suite in suite_objects:
        print(f"  - {suite['name']} ({suite['config']})")
    print(f"Saved selections successfully to: {output_file}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write("all_suites<<EOF\n")
            f.write(f"{json.dumps(suite_objects)}\n")
            f.write("EOF\n")
            f.write(f"suite_count={len(suite_objects)}\n")


if __name__ == "__main__":
    main()
