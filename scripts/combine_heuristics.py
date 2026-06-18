#!/usr/bin/env python3
import json
import os
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def main():
    repo_root = get_repo_root()
    mappings_dir = repo_root / "mappings"

    rules_file = mappings_dir / "selected_suites_by_rules.json"
    if not rules_file.exists():
        print("Error: selected_suites_by_rules.json could not be located.")
        return

    with open(rules_file, "r") as f:
        rules_data = json.load(f)

    combined_configs = set(rules_data.get("suites", []))

    final_suite_objects = []
    for config_path in sorted(list(combined_configs)):
        filename = Path(config_path).stem
        name = filename[:-7] if filename.endswith("_config") else filename
        final_suite_objects.append({"name": name, "config": config_path})

    print(
        "================================================================================"
    )
    print("FINAL DETERMINISTIC TEST SELECTION MATRIX LOG")
    print(
        "================================================================================"
    )
    print(f"Total Unique Suites Assigned: {len(final_suite_objects)}")
    for obj in final_suite_objects:
        print(f"  -> Name: {obj['name']} | Config: {obj['config']}")
    print(
        "================================================================================"
    )

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write("final_suites<<EOF\n")
            f.write(f"{json.dumps(final_suite_objects)}\n")
            f.write("EOF\n")
            f.write(f"has_final_suites={'true' if final_suite_objects else 'false'}\n")
            f.write(f"suite_count={len(final_suite_objects)}\n")


if __name__ == "__main__":
    main()
