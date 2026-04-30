"""
Automatic module configuration generator using forward hooks.

This script automatically generates YAML configuration for all unique modules
in a model by:
1. Loading the model
2. Registering forward hooks on all modules
3. Running a forward pass to capture module inputs
4. Analyzing captured data to generate YAML config

Usage:
    python auto_generate_module_config.py --model_path ibm-granite/granite-3.3-8b-instruct --batch_size 2 --seq_len 128
"""

import torch
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from transformers import AutoModel, AutoTokenizer
from torch.utils._pytree import tree_flatten


class PrettyDumper(yaml.SafeDumper):
    """Custom YAML dumper with consistent 2-space indentation."""

    def increase_indent(self, flow=False, indentless=False):
        """Ensure consistent indentation (no indentless sequences)."""
        return super().increase_indent(flow, False)

    def represent_data(self, data):
        """Override to handle shape lists specially."""
        # Check if this is a list that should be inline (shape values)
        if isinstance(data, list) and len(data) > 0:
            # Check if all elements are integers (shape lists are all ints)
            if all(isinstance(x, int) for x in data):
                # This is likely a shape list - use flow style
                return self.represent_sequence(
                    "tag:yaml.org,2002:seq", data, flow_style=True
                )

        # For everything else, use default representation
        return super().represent_data(data)


def _is_special_tensor(name: str) -> bool:
    """Check if tensor name indicates it should not be random."""
    return any(keyword in name.lower() for keyword in ["position", "mask", "ids"])


def _extract_tensor_info(tensor: torch.Tensor, name: str) -> Dict[str, Any]:
    """Extract information from a single tensor."""
    return {
        "type": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "is_random": not _is_special_tensor(name),
        "requires_grad": tensor.requires_grad,
    }


def _process_pytree_structure(value: Any, name: str) -> Dict[str, Any] | None:
    """
    Process a pytree structure (nested tensors/lists/tuples/dicts) and extract info.

    Uses PyTorch's tree_flatten to handle arbitrary nesting uniformly.
    """
    # Check if this is a tensor or contains tensors
    if isinstance(value, torch.Tensor):
        # Single tensor - simple case
        return {"name": name, **_extract_tensor_info(value, name)}

    # Use tree_flatten to extract all tensor leaves regardless of nesting.
    # We intentionally do not reconstruct the original structure since only
    # tensor metadata is needed for config generation.
    flat_values, _ = tree_flatten(value)

    # Extract info from all tensors in the flattened structure
    # Single source of truth: pytree handles all container types uniformly
    tensor_infos = []
    for item in flat_values:
        if isinstance(item, torch.Tensor):
            tensor_infos.append(_extract_tensor_info(item, name))

    # Post-process: enrich dict tensors with their keys
    if isinstance(value, dict) and tensor_infos:
        dict_keys = [k for k, v in value.items() if isinstance(v, torch.Tensor)]
        for i, key in enumerate(dict_keys):
            if i < len(tensor_infos):
                tensor_infos[i]["dict_key"] = key

    # If we found tensors, return with structure info
    if tensor_infos:
        # Determine container type from the original value
        if isinstance(value, tuple):
            container_type = "tuple"
        elif isinstance(value, list):
            container_type = "list"
        elif isinstance(value, dict):
            container_type = "dict"
        else:
            container_type = "pytree"

        return {
            "name": name,
            "type": container_type,
            "items": tensor_infos,
        }

    return None


class ModuleInfoCapture:
    """Captures module information during forward pass using hooks."""

    def __init__(self):
        self.module_data: Dict[str, Dict[str, Any]] = {}
        self.seen_module_types: Set[str] = set()

    def capture_constructor_info(
        self, module, module_name: str, module_type: str
    ) -> Dict[str, Any]:
        """
        Capture constructor information from an instantiated module.

        This inspects the module to infer what constructor args were used.
        For Transformers modules, we look for config objects and layer_idx.
        """
        constructor_args = []
        constructor_kwargs = {}

        # Special handling for decoder layers that don't expose config attribute
        # but require it as constructor arg (e.g., GraniteDecoderLayer)
        if "decoder" in module_type.lower() and "layer" in module_type.lower():
            # Try to get config from parent model or infer from module structure
            # For now, we'll look for self_attn or mlp submodules that might have config
            if hasattr(module, "self_attn") and hasattr(module.self_attn, "config"):
                config = module.self_attn.config
            elif hasattr(module, "mlp") and hasattr(module.mlp, "config"):
                config = module.mlp.config
            else:
                config = None

            if config is not None:
                config_class = type(config).__name__
                config_module = type(config).__module__

                # Extract key config parameters
                config_kwargs = {}
                for attr in [
                    "hidden_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "intermediate_size",
                    "max_position_embeddings",
                    "_attn_implementation",
                ]:
                    if hasattr(config, attr):
                        config_kwargs[attr] = getattr(config, attr)

                constructor_args.append(
                    {
                        "type": "config",
                        "config_path": f"{config_module}.{config_class}",
                        "config_kwargs": config_kwargs,
                    }
                )

                # Decoder layers typically need layer_idx as kwarg
                # Always add it for decoder layers, even if not found as attribute
                layer_idx_value = 0  # Default to 0
                if hasattr(module, "layer_idx") and module.layer_idx is not None:
                    layer_idx_value = module.layer_idx
                constructor_kwargs["layer_idx"] = {
                    "type": "int",
                    "value": layer_idx_value,
                }
        # Check if module has a config attribute (common in Transformers)
        elif hasattr(module, "config"):
            config = module.config
            config_class = type(config).__name__
            config_module = type(config).__module__

            # Extract key config parameters
            config_kwargs = {}
            for attr in [
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "intermediate_size",
                "max_position_embeddings",
                "_attn_implementation",
            ]:
                if hasattr(config, attr):
                    config_kwargs[attr] = getattr(config, attr)

            constructor_args.append(
                {
                    "type": "config",
                    "config_path": f"{config_module}.{config_class}",
                    "config_kwargs": config_kwargs,
                }
            )

            # Check for layer_idx (common in decoder layers with config)
            # Note: layer_idx can be 0, so check for attribute existence, not truthiness
            if hasattr(module, "layer_idx"):
                layer_idx_value = (
                    module.layer_idx if module.layer_idx is not None else 0
                )
                constructor_kwargs["layer_idx"] = {
                    "type": "int",
                    "value": layer_idx_value,
                }
        else:
            # No config - check for direct constructor parameters
            # RMSNorm: hidden_size or dim
            if hasattr(module, "weight") and hasattr(module.weight, "shape"):
                # Normalization layers typically have weight with shape (hidden_size,)
                if len(module.weight.shape) == 1:
                    hidden_size = module.weight.shape[0]
                    constructor_args.append({"type": "int", "value": hidden_size})
            elif hasattr(module, "normalized_shape"):
                # LayerNorm-style
                if isinstance(module.normalized_shape, tuple):
                    hidden_size = module.normalized_shape[0]
                else:
                    hidden_size = module.normalized_shape
                constructor_args.append({"type": "int", "value": hidden_size})

        return {
            "constructor_args": constructor_args,
            "constructor_kwargs": constructor_kwargs,
        }

    def create_hook(self, module_name: str, module_type: str, module_instance):
        """Create a forward hook that captures module input information."""

        def hook(module, args, kwargs):
            # Only capture first occurrence of each module type
            if module_type in self.seen_module_types:
                return

            self.seen_module_types.add(module_type)

            # Capture constructor information
            constructor_info = self.capture_constructor_info(
                module, module_name, module_type
            )

            # Capture module information
            module_info = {
                "name": module_type,
                "module_path": f"{module.__class__.__module__}.{module.__class__.__name__}",
                "example_instance": module_name,
                "constructor_args": constructor_info["constructor_args"],
                "constructor_kwargs": constructor_info["constructor_kwargs"],
                "inputs": [],
            }

            # Analyze positional arguments using pytree
            for i, arg in enumerate(args):
                input_info = _process_pytree_structure(arg, f"arg_{i}")
                if input_info:
                    module_info["inputs"].append(input_info)

            # Analyze keyword arguments using pytree
            for key, value in kwargs.items():
                input_info = _process_pytree_structure(value, key)
                if input_info:
                    module_info["inputs"].append(input_info)

            # Store the captured information
            self.module_data[module_type] = module_info

        return hook

    def get_captured_modules(self) -> List[Dict[str, Any]]:
        """Return list of captured module information."""
        return list(self.module_data.values())


def get_unique_modules(model) -> Dict[str, Tuple[str, Any]]:
    """
    Get unique module types from the model.

    Returns:
        Dict mapping module_type -> (module_name, module_instance)
    """
    unique = {}

    # Get existing modules from PyTorch's module_db to avoid duplicates
    try:
        from torch.testing._internal.common_modules import module_db

        # Extract just the class name from module_db names (e.g., "nn.Linear" -> "Linear")
        existing_modules = set()
        for m in module_db:
            # module_db names are like "nn.Linear", "nn.Conv2d", etc.
            if "." in m.name:
                class_name = m.name.split(".")[-1]
                existing_modules.add(class_name)
            else:
                existing_modules.add(m.name)
        print(f"Found {len(existing_modules)} existing modules in PyTorch's module_db")
    except ImportError:
        existing_modules = set()
        print("Warning: Could not import module_db, will not filter duplicates")

    for name, module in model.named_modules():
        if name == "":  # Skip root
            continue

        module_type = type(module).__name__

        # Skip if already in upstream module_db
        if module_type in existing_modules:
            print(f"  Skipping {module_type} (already in module_db)")
            continue

        # Only keep first occurrence of each type
        if module_type not in unique:
            unique[module_type] = (name, module)

    return unique


def _convert_constructor_arg_to_sample_input(
    arg_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert constructor arg spec to sample_inputs_func format."""
    if arg_spec["type"] == "config":
        # Config objects become a special marker - will be handled by test code
        return {"value": f"<config:{arg_spec['config_path']}>"}
    elif arg_spec["type"] == "int":
        return {"value": arg_spec["value"]}
    elif arg_spec["type"] == "float":
        return {"value": arg_spec["value"]}
    elif arg_spec["type"] == "str":
        return {"value": arg_spec["value"]}
    elif arg_spec["type"] == "bool":
        return {"value": arg_spec["value"]}
    else:
        return {"value": None}


def _tensor_info_to_spec(tensor_info: Dict[str, Any], name: str) -> Dict[str, Any]:
    """
    Convert a single tensor info dict to sample_inputs tensor spec format.

    This function can be used with tree_map to transform entire structures.
    """
    dtype = tensor_info["dtype"]
    if not dtype.startswith("torch."):
        dtype = f"torch.{dtype}"

    # Determine init strategy based on tensor characteristics
    is_random = tensor_info.get("is_random", True)
    init = "randn" if is_random else "zeros"
    init_args = {}

    # Special handling for position/id tensors
    if _is_special_tensor(name):
        init = "randint"
        init_args = {"high": 10000}

    tensor_spec = {
        "shape": tensor_info["shape"],
        "stride": None,  # Let PyTorch compute default stride
        "storage_offset": 0,
        "dtype": dtype,
        "device": "spyre",
        "init": init,
    }

    if init_args:
        tensor_spec["init_args"] = init_args

    return tensor_spec


def _convert_captured_input_to_sample_input(inp_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert captured input spec to sample_inputs_func format.

    Uses pytree utilities to handle single tensors and nested collections uniformly.
    The key insight: pytree lets us treat single tensors and collections the same way.
    """
    inp_name = inp_spec["name"]
    inp_type = inp_spec["type"]

    if inp_type == "tensor":
        # Single tensor - wrap in standard format
        return {"tensor": _tensor_info_to_spec(inp_spec, inp_name)}

    elif inp_type in ("tuple", "list", "dict", "pytree"):
        # Collection of tensors - pytree handles all container types uniformly
        # Convert each tensor in the flattened structure
        tensor_list = [
            _tensor_info_to_spec(item, inp_name) for item in inp_spec.get("items", [])
        ]

        return {"tensor_list": tensor_list}

    else:
        return {"value": None}


def _build_module_entry_dict(module_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a module entry dictionary for YAML generation.

    Args:
        module_info: Captured module information

    Returns:
        Dictionary representing a module entry for YAML
    """
    # Build constructor_inputs
    constructor_args = []
    constructor_kwargs = {}

    for arg_spec in module_info.get("constructor_args", []):
        constructor_args.append(_convert_constructor_arg_to_sample_input(arg_spec))

    for key, kwarg_spec in module_info.get("constructor_kwargs", {}).items():
        if kwarg_spec["type"] == "int":
            constructor_kwargs[key] = kwarg_spec["value"]

    # Build forward_inputs
    forward_args = []
    forward_kwargs = {}

    for inp_spec in module_info.get("inputs", []):
        inp_name = inp_spec["name"]
        converted = _convert_captured_input_to_sample_input(inp_spec)

        if inp_name.startswith("arg_"):
            forward_args.append(converted)
        else:
            forward_kwargs[inp_name] = converted

    # Build module entry
    entry = {
        "name": module_info["name"],
        "module_path": module_info["module_path"],
        "description": f"Module: {module_info['module_path']}",
        "constructor_inputs": {
            "args": constructor_args if constructor_args else [],
            "kwargs": constructor_kwargs if constructor_kwargs else {},
        },
        "forward_inputs": {
            "args": forward_args if forward_args else [],
            "kwargs": forward_kwargs if forward_kwargs else {},
        },
    }

    return entry


def generate_unified_yaml_config(
    captured_modules: List[Dict[str, Any]], model_name: str
) -> str:
    """Generate unified YAML configuration using yaml.dump().

    This creates a single YAML file with edits.modules.include that contains:
    - Module name and path
    - constructor_inputs: Args/kwargs for module.__init__()
    - forward_inputs: Args/kwargs for module.forward()
    """
    # Build module entries
    module_entries = [_build_module_entry_dict(m) for m in captured_modules]

    # Build the complete configuration dictionary
    config = {
        "test_suite_config": {
            "files": [
                {
                    "path": "${TORCH_ROOT}/test/test_modules.py",
                    "unlisted_test_mode": "skip",
                    "tests": [
                        {
                            "names": ["*TestModule*::test_forward"],
                            "mode": "mandatory_success",
                            "tags": [f"model__{model_name}"],
                            "edits": {"modules": {"include": module_entries}},
                        }
                    ],
                },
                {
                    "path": "${TORCH_DEVICE_ROOT}/tests/test_modules_custom.py",
                    "unlisted_test_mode": "skip",
                    "tests": [
                        {
                            "names": [
                                "*TestModuleCustom*::test_eager_vs_compile",
                                "*TestModuleCustom*::test_layout",
                                "*TestModuleCustom*::test_stride",
                            ],
                            "mode": "mandatory_success",
                            "tags": [f"model__{model_name}", "custom_tests"],
                            "edits": {"modules": {"include": module_entries}},
                        }
                    ],
                },
            ],
            "global": {
                "supported_dtypes": [
                    {"name": "float16", "precision": {"atol": 0.005, "rtol": 0.005}},
                    {"name": "float32", "precision": {"atol": 0.001, "rtol": 0.001}},
                ],
                "input_config": {"seed": 123},
            },
        }
    }

    # Generate YAML string with header comments and consistent 2-space indentation
    header = f"""# Auto-generated unified test configuration for {model_name}
# Generated by auto_generate_module_config.py
# Format compatible with PyTorch's test_modules.py (using edits.modules.include)

"""

    # Use custom Dumper with 2-space indentation for consistency
    yaml_str = header + yaml.dump(
        config,
        Dumper=PrettyDumper,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
        width=float("inf"),  # Prevent line wrapping
    )
    return yaml_str


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate module configuration YAML using forward hooks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model path (e.g., ibm-granite/granite-3.3-8b-instruct)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for forward pass (default: 2)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length for forward pass (default: 128)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML file path (default: ./configs/<model>_spyre.yaml)",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).eval()

    print("Analyzing model structure...")
    unique_modules = get_unique_modules(model)
    print(f"Found {len(unique_modules)} unique module types")

    # Create capture object
    capture = ModuleInfoCapture()

    # Register hooks on all unique modules
    handles = []
    for module_type, (module_name, module_instance) in unique_modules.items():
        hook = capture.create_hook(module_name, module_type, module_instance)
        handle = module_instance.register_forward_pre_hook(hook, with_kwargs=True)
        handles.append(handle)

    print("Running forward pass to capture module inputs...")
    # Create dummy input with specified sequence length
    # Generate enough text to reach desired seq_len
    text = "This is a test input for capturing module information. " * (
        args.seq_len // 10 + 1
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=args.seq_len,
        truncation=True,
        padding="max_length",
    )
    print(f"  Input shape: {inputs['input_ids'].shape}")

    # Run forward pass (this triggers all hooks)
    with torch.no_grad():
        model(**inputs)

    with torch.no_grad():
        decode_inputs = {
            "input_ids": torch.cat(
                [
                    inputs["input_ids"],
                    torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long),
                ],
                dim=1,
            ),
            "attention_mask": torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones((inputs["input_ids"].shape[0], 1), dtype=torch.long),
                ],
                dim=1,
            ),
        }

        _ = model(**decode_inputs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    print(f"Captured information for {len(capture.get_captured_modules())} modules")

    # Generate YAML
    # Extract model name from path (handle both local paths and HuggingFace paths)
    model_path_parts = args.model_path.rstrip("/").split("/")
    model_name = model_path_parts[
        -1
    ]  # e.g., "granite-3.3-8b-instruct" or "granite-3.0-2b-instruct"

    # For the YAML content, use underscores for the model_name field
    model_name_normalized = model_name.replace("-", "_").replace(".", "_")

    print(f"Model name: {model_name} (normalized: {model_name_normalized})")

    # Generate unified YAML config (new format)
    unified_yaml_content = generate_unified_yaml_config(
        capture.get_captured_modules(), model_name_normalized
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Use tests/configs directory for unified format
        output_path = f"./tests/configs/{model_name_normalized}_spyre.yaml"

    # Write unified YAML file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(unified_yaml_content)

    print(f"\n✓ Generated unified configuration: {output_file}")

    # Print module summary
    total_modules = len(capture.get_captured_modules())
    print("\n  Module Summary:")
    print(f"    Total modules captured: {total_modules}")
    for module_info in capture.get_captured_modules():
        print(f"      - {module_info['name']}")

    print("\nNext steps:")
    print(f"1. Review the generated YAML file: {output_file}")
    print("2. Run tests:")
    print(f"   export PYTORCH_TEST_CONFIG={output_file.absolute()}")
    print("   cd $PYTORCH/test")
    print(f"   python test_model_ops_v2.py -k {model_name_normalized}")


if __name__ == "__main__":
    main()
