# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class LoadedCase:
    model: str
    defaults: Dict[str, Any]
    case: Dict[str, Any]
    source_path: Path


def models_dir(pytest_root: Path) -> Path:
    # adjust if needed; this is robust regardless of current working directory
    return pytest_root / "tests" / "resource" / "models"


def freeze(x: Any) -> Any:
    if isinstance(x, dict):
        return tuple(sorted((k, freeze(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(freeze(v) for v in x)
    if isinstance(x, str) and "torch." not in x:
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


def case_key(case: Dict[str, Any], defaults: Dict[str, Any]) -> tuple:
    # Used for dedupe; intentionally does NOT include model name.
    op = case["op"]
    dtype = case.get("dtype", defaults.get("dtype", "float16"))
    seed = case.get("seed", defaults.get("seed", None))
    attrs = freeze(case.get("attrs", {}))
    kwmap = freeze(case.get("kwmap", {}))

    inputs_sig: List[tuple[Any, ...]] = []
    for inp in case.get("inputs", []):
        if "tensor" in inp:
            t = inp["tensor"]
            inputs_sig.append(
                (
                    "tensor",
                    tuple(t["shape"]),
                    tuple(t["stride"]),
                    t.get("storage_offset", 0),
                    t.get("dtype", dtype),
                    t.get("device", "cpu"),
                    t.get("init", "rand"),
                    freeze(t.get("init_args", {})),
                )
            )
        elif "tensor_list" in inp:
            lst = []
            for t in inp["tensor_list"]:
                lst.append(
                    (
                        tuple(t["shape"]),
                        tuple(t["stride"]),
                        t.get("storage_offset", 0),
                        t.get("dtype", dtype),
                        t.get("device", "cpu"),
                        t.get("init", "rand"),
                        freeze(t.get("init_args", {})),
                    )
                )
            inputs_sig.append(("tensor_list", tuple(lst)))
        elif "value" in inp:
            inputs_sig.append(("value", freeze(inp["value"])))
        elif "py" in inp:
            inputs_sig.append(("py", freeze(inp["py"])))
        else:
            raise ValueError(f"Unknown input entry: {inp}")

    return (op, dtype, seed, attrs, kwmap, tuple(inputs_sig))


def load_all_cases(pytest_root: Path) -> List[LoadedCase]:
    items: List[LoadedCase] = []
    for p in sorted(models_dir(pytest_root).glob("*.yaml")):
        print(p)
        if p.name.endswith("template.yaml"):  # skip template.yaml file
            continue
        try:
            spec = yaml.safe_load(p.read_text())
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file: {p}") from e
        model = spec.get("model", p.stem)
        defaults = dict(spec.get("defaults", {}))
        for case in spec.get("cases", []):
            # allow "cases" to omit "op" if the YAML provides a top-level op (optional)
            if "op" not in case and "op" in spec:
                case = dict(case)
                case["op"] = spec["op"]
            items.append(
                LoadedCase(model=model, defaults=defaults, case=case, source_path=p)
            )
    return items
