#!/usr/bin/env python3
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def get_repo_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def normalize_config_path_value(path_value: str) -> str:
    path_value = re.sub(r"\$\{[^}]+\}/?", "", path_value)
    path_value = path_value.replace("\\", "/")
    return path_value.lstrip("/")


def parse_suite_yaml_for_test_files(yaml_path: Path) -> List[str]:
    if not yaml_path.exists():
        return []
    text = yaml_path.read_text(encoding="utf-8")
    raw_paths: List[str] = []

    if YAML_AVAILABLE:
        try:
            data = yaml.safe_load(text)

            def extract(d):
                paths = []
                if isinstance(d, dict):
                    if "path" in d:
                        paths.append(str(d["path"]))
                    for v in d.values():
                        paths.extend(extract(v))
                elif isinstance(d, list):
                    for item in d:
                        paths.extend(extract(item))
                return paths

            raw_paths = extract(data)
        except Exception:
            pass

    if not raw_paths:
        for line in text.splitlines():
            content = line.split("#", 1)[0].strip()
            if "path:" in content:
                content = content.lstrip("-").strip()
                if content.startswith("path:"):
                    val = content[len("path:") :].strip().strip('"').strip("'")
                    if val:
                        raw_paths.append(val)

    return [normalize_config_path_value(p) for p in raw_paths]


def main():
    repo_root = get_repo_root()
    configs_root = repo_root / "tests" / "configs"
    output_file = repo_root / "mappings" / "test_to_suite.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # MODIFIED: Filters paths so that only configurations within torch_spyre_tests/ are evaluated
    suite_configs = sorted(
        [
            p
            for p in configs_root.rglob("*.yaml")
            if "example" not in p.name.lower()
            and "torch_spyre_tests/" in str(p.relative_to(configs_root))
        ]
    )

    config_to_files: Dict[str, List[str]] = {}
    file_to_configs = defaultdict(list)

    for cfg in suite_configs:
        cfg_rel = str(cfg.relative_to(configs_root))
        test_files = parse_suite_yaml_for_test_files(cfg)
        config_to_files[cfg_rel] = test_files
        for tf in test_files:
            file_to_configs[tf].append(cfg_rel)

    output_data = {
        "config_to_test_files": config_to_files,
        "test_file_to_configs": {
            k: sorted(v) for k, v in sorted(file_to_configs.items())
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(
        f"✓ Test-to-suite mapping config (torch_spyre_tests/ only) saved to: {output_file}"
    )


if __name__ == "__main__":
    main()
