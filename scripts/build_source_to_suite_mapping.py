#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, List, Set


def get_repo_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def get_all_source_files(source_path: Path, base_dir: Path) -> Set[str]:
    source_files = set()
    for py_file in source_path.rglob("*.py"):
        try:
            rel_path = py_file.relative_to(base_dir)
            source_files.add(str(rel_path))
        except ValueError:
            continue
    return source_files


def main():
    repo_root = get_repo_root()
    source_dir = repo_root / "torch_spyre"
    coverage_dir = repo_root / "coverage"
    output_dir = repo_root / "mappings"
    output_file = output_dir / "source_to_suite.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = get_all_source_files(source_dir, repo_root)
    reverse_map: Dict[str, List[str]] = {
        file_path: [] for file_path in sorted(source_files)
    }

    if not coverage_dir.exists():
        print(
            "Warning: Coverage root directory missing. Creating empty mapping structure."
        )
        with open(output_file, "w") as f:
            json.dump(reverse_map, f, indent=2)
        return

    suites = [d for d in os.listdir(coverage_dir) if (coverage_dir / d).is_dir()]
    print(f"Scanning coverage logs across {len(suites)} suite directory metrics...")

    for suite in sorted(suites):
        json_file = coverage_dir / suite / "coverage.json"
        if not json_file.exists():
            continue

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            files_dict = data.get("files", {})
            for abs_path, file_data in files_dict.items():
                summary = file_data.get("summary", {})
                covered_lines = summary.get("covered_lines", 0)

                if covered_lines > 0 and "torch_spyre/" in abs_path:
                    rel_path = "torch_spyre/" + abs_path.split("torch_spyre/", 1)[1]
                    if rel_path in reverse_map:
                        reverse_map[rel_path].append(suite)

        except (json.JSONDecodeError, FileNotFoundError):
            continue

    with open(output_file, "w") as f:
        json.dump(reverse_map, f, indent=2)

    print(f"✓ Success! Source-to-suite mappings written to: {output_file}")


if __name__ == "__main__":
    main()
