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
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from transformers import AutoModel, AutoTokenizer


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

            # Analyze positional arguments
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    input_info = {
                        "name": f"arg_{i}",
                        "type": "tensor",
                        "shape": list(arg.shape),
                        "dtype": str(arg.dtype).replace("torch.", ""),
                        "is_random": True,  # Default assumption
                        "requires_grad": arg.requires_grad,
                    }
                    module_info["inputs"].append(input_info)
                elif isinstance(arg, (tuple, list)):
                    # Handle tuple/list of tensors (e.g., position_embeddings)
                    for j, item in enumerate(arg):
                        if isinstance(item, torch.Tensor):
                            input_info = {
                                "name": f"arg_{i}_item_{j}",
                                "type": "tensor",
                                "shape": list(item.shape),
                                "dtype": str(item.dtype).replace("torch.", ""),
                                "is_random": True,
                                "requires_grad": item.requires_grad,
                            }
                            module_info["inputs"].append(input_info)

            # Analyze keyword arguments
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    # Detect special non-random tensors
                    is_random = True
                    if (
                        "position" in key.lower()
                        or "mask" in key.lower()
                        or "ids" in key.lower()
                    ):
                        is_random = False

                    input_info = {
                        "name": key,
                        "type": "tensor",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype).replace("torch.", ""),
                        "is_random": is_random,
                        "requires_grad": value.requires_grad,
                    }
                    module_info["inputs"].append(input_info)
                elif isinstance(value, (tuple, list)):
                    # Handle tuple/list (e.g., position_embeddings = (cos, sin))
                    is_random = "position" not in key.lower()
                    tuple_items = []
                    for j, item in enumerate(value):
                        if isinstance(item, torch.Tensor):
                            tuple_items.append(
                                {
                                    "type": "tensor",
                                    "shape": list(item.shape),
                                    "dtype": str(item.dtype).replace("torch.", ""),
                                    "is_random": is_random,
                                    "requires_grad": item.requires_grad,
                                }
                            )

                    # Add as a single tuple input
                    if tuple_items:
                        input_info = {
                            "name": key,
                            "type": "tuple",
                            "items": tuple_items,
                        }
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


def map_to_generic_generator(module_info: Dict[str, Any]) -> List[str]:
    """
    Map module to appropriate input generator(s).

    Prefers model-specific generators over generic ones when available.
    Returns list of generator function names that could work for this module.
    """
    module_name = module_info["name"].lower()
    module_path = module_info["module_path"].lower()

    generators = []

    # Check for model-specific generators first
    # Granite modules
    if "granite" in module_path:
        if "attention" in module_name or "attn" in module_name:
            generators.append(
                "model_tests.module_input_generators.generate_granite_attention_inputs"
            )
        elif "mlp" in module_name:
            generators.append(
                "model_tests.module_input_generators.generate_granite_mlp_inputs"
            )
        elif "rmsnorm" in module_name:
            generators.append(
                "model_tests.module_input_generators.generate_granite_rmsnorm_inputs"
            )
        elif "decoder" in module_name and "layer" in module_name:
            generators.append(
                "model_tests.module_input_generators.generate_granite_decoder_layer_inputs"
            )
        elif "rotary" in module_name or "rope" in module_name:
            generators.append(
                "model_tests.module_input_generators.generate_granite_rotary_embedding_inputs"
            )

    # SiLU activation (transformers-specific)
    if "siluactivation" in module_name.replace("_", ""):
        generators.append(
            "model_tests.module_input_generators.generate_silu_activation_inputs"
        )

    # If no model-specific generator found, fall back to generic ones
    if not generators:
        # Attention modules
        if "attention" in module_name or "attn" in module_name:
            generators.append(
                "model_tests.generic_input_generators.generate_attention_inputs"
            )

        # MLP/FFN modules
        elif (
            "mlp" in module_name or "ffn" in module_name or "feedforward" in module_name
        ):
            generators.append(
                "model_tests.generic_input_generators.generate_mlp_inputs"
            )

        # Normalization modules
        elif (
            "norm" in module_name
            or "layernorm" in module_name
            or "rmsnorm" in module_name
        ):
            generators.append(
                "model_tests.generic_input_generators.generate_norm_inputs"
            )

        # Decoder layer
        elif "decoder" in module_name and "layer" in module_name:
            generators.append(
                "model_tests.generic_input_generators.generate_decoder_layer_inputs"
            )

        # Rotary embedding
        elif "rotary" in module_name or "rope" in module_name:
            generators.append(
                "model_tests.generic_input_generators.generate_rotary_embedding_inputs"
            )

        # Activation functions
        elif (
            "activation" in module_name
            or "silu" in module_name
            or "gelu" in module_name
        ):
            generators.append(
                "model_tests.generic_input_generators.generate_activation_inputs"
            )

        # Linear layers
        elif "linear" in module_name:
            generators.append(
                "model_tests.generic_input_generators.generate_linear_inputs"
            )

        # Default: generic single tensor input
        else:
            generators.append(
                "model_tests.generic_input_generators.generate_generic_inputs"
            )

    return generators


def generate_yaml_config(
    captured_modules: List[Dict[str, Any]], model_name: str
) -> str:
    """Generate YAML configuration from captured module information."""

    config = {"model_name": model_name, "modules": []}

    for module_info in captured_modules:
        module_name = module_info["name"].lower()

        # Determine if train/eval modes differ (for dropout, etc.)
        train_and_eval_differ = "attention" in module_name or "decoder" in module_name

        module_config = {
            "name": module_info["name"],
            "module_path": module_info["module_path"],
            "constructor_args": module_info.get("constructor_args", []),
            "constructor_kwargs": module_info.get("constructor_kwargs", {}),
            "captured_inputs": module_info["inputs"],
            "dtypes": ["float16", "float32"],
            "skips": [],  # No default skips - let tests fail naturally
            "decorators": [],
            "train_and_eval_differ": train_and_eval_differ,
        }

        config["modules"].append(module_config)

    # Convert to YAML with custom formatting
    yaml_str = f"# Auto-generated module configuration for {model_name}\n"
    yaml_str += "# Generated by auto_generate_module_config.py\n"
    yaml_str += "# This config uses programmatic input generators\n\n"
    yaml_str += f"model_name: {model_name}\n\n"
    yaml_str += "modules:\n"

    for module in config["modules"]:
        yaml_str += f'\n  - name: "{module["name"]}"\n'
        yaml_str += f'    module_path: "{module["module_path"]}"\n'
        yaml_str += '    allow_missing_modules: "true"\n'

        # Constructor args
        if module["constructor_args"]:
            yaml_str += "    constructor_args:\n"
            for arg in module["constructor_args"]:
                yaml_str += f"      - type: {arg['type']}\n"
                if arg["type"] == "config":
                    yaml_str += f'        config_path: "{arg["config_path"]}"\n'
                    yaml_str += "        config_kwargs:\n"
                    for k, v in arg["config_kwargs"].items():
                        yaml_str += f"          {k}: {v}\n"
                else:
                    yaml_str += f"        value: {arg['value']}\n"
        else:
            yaml_str += "    constructor_args: []\n"

        # Constructor kwargs
        if module["constructor_kwargs"]:
            yaml_str += "    constructor_kwargs:\n"
            for key, kwarg in module["constructor_kwargs"].items():
                yaml_str += f"      {key}:\n"
                yaml_str += f"        type: {kwarg['type']}\n"
                yaml_str += f"        value: {kwarg['value']}\n"
        else:
            yaml_str += "    constructor_kwargs: {}\n"

        # Captured inputs (forward pass)
        yaml_str += "    captured_inputs:\n"
        for inp in module["captured_inputs"]:
            yaml_str += f"      - name: {inp['name']}\n"
            yaml_str += f"        type: {inp['type']}\n"
            if inp["type"] == "tuple":
                yaml_str += "        items:\n"
                for item in inp["items"]:
                    yaml_str += f"          - type: {item['type']}\n"
                    yaml_str += f"            shape: {item['shape']}\n"
                    yaml_str += f"            dtype: {item['dtype']}\n"
                    yaml_str += f"            is_random: {item['is_random']}\n"
            else:
                yaml_str += f"        shape: {inp['shape']}\n"
                yaml_str += f"        dtype: {inp['dtype']}\n"
                yaml_str += f"        is_random: {inp['is_random']}\n"

        yaml_str += "    dtypes:\n"
        for dtype in module["dtypes"]:
            yaml_str += f"      - {dtype}\n"

        # Skips
        if module["skips"]:
            yaml_str += "    skips:\n"
            for skip in module["skips"]:
                yaml_str += f"      - {skip}\n"
        else:
            yaml_str += "    skips: []\n"

        yaml_str += "    decorators: []\n"

        # Train and eval differ flag
        if module.get("train_and_eval_differ", False):
            yaml_str += "    train_and_eval_differ: true\n"

    return yaml_str


def classify_module_complexity(module_info: Dict[str, Any]) -> str:
    """Classify module as 'simple' or 'complex' based on characteristics.

    Simple modules:
    - Normalization layers (RMSNorm, LayerNorm)
    - Activation functions (SiLU, GELU)
    - Simple linear transformations

    Complex modules:
    - Attention mechanisms (require masks, position embeddings)
    - Decoder layers (stateful, caching, dropout)
    - Modules with training/eval differences

    Returns:
        'simple' or 'complex'
    """
    module_name = module_info["name"].lower()

    # Simple modules
    if any(
        keyword in module_name
        for keyword in ["norm", "activation", "silu", "gelu", "relu"]
    ):
        return "simple"

    # Check for simple structure (no complex inputs)
    has_complex_inputs = False
    for inp in module_info.get("inputs", []):
        # Tuples (like position_embeddings) indicate complexity
        if inp.get("type") == "tuple":
            has_complex_inputs = True
        # Multiple tensor inputs indicate complexity
        if (
            inp.get("name", "").startswith("arg_")
            and len(module_info.get("inputs", [])) > 2
        ):
            has_complex_inputs = True

    # Complex modules
    if any(keyword in module_name for keyword in ["attention", "attn", "decoder"]):
        return "complex"

    if has_complex_inputs:
        return "complex"

    # Default to simple for basic modules (MLP, Linear, etc.)
    return "simple"


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


def _convert_captured_input_to_sample_input(inp_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert captured input spec to sample_inputs_func format."""
    if inp_spec["type"] == "tensor":
        # Map dtype string to torch.dtype format
        dtype = inp_spec["dtype"]
        if not dtype.startswith("torch."):
            dtype = f"torch.{dtype}"

        # Determine init strategy
        init = "randn" if inp_spec.get("is_random", True) else "zeros"
        if "position" in inp_spec["name"].lower() or "ids" in inp_spec["name"].lower():
            init = "randint"
            init_args = {"high": 10000}
        else:
            init_args = {}

        tensor_spec = {
            "tensor": {
                "shape": inp_spec["shape"],
                "stride": None,  # Let PyTorch compute default stride
                "storage_offset": 0,
                "dtype": dtype,
                "device": "spyre",
                "init": init,
            }
        }

        if init_args:
            tensor_spec["tensor"]["init_args"] = init_args

        return tensor_spec

    elif inp_spec["type"] == "tuple":
        # Handle tuples (e.g., position_embeddings)
        tensor_list = []
        for item in inp_spec.get("items", []):
            dtype = item["dtype"]
            if not dtype.startswith("torch."):
                dtype = f"torch.{dtype}"

            init = "randn" if item.get("is_random", True) else "zeros"

            tensor_list.append(
                {
                    "shape": item["shape"],
                    "stride": None,
                    "storage_offset": 0,
                    "dtype": dtype,
                    "device": "spyre",
                    "init": init,
                }
            )

        return {"tensor_list": tensor_list}

    else:
        return {"value": None}


def generate_unified_yaml_config(
    captured_modules: List[Dict[str, Any]], model_name: str
) -> str:
    """Generate unified YAML configuration in test_model_ops_v2.py format.

    This creates a single YAML file with edits.modules.include that contains:
    - Module name and path
    - constructor_inputs: Args/kwargs for module.__init__()
    - forward_inputs: Args/kwargs for module.forward()
    - Separated format for proper ModuleInput generation
    """
    # Classify modules
    simple_modules = []
    complex_modules = []
    all_module_entries = []

    for module_info in captured_modules:
        complexity = classify_module_complexity(module_info)

        # Build constructor_inputs (separate from forward inputs)
        constructor_args = []
        constructor_kwargs = {}

        # Add constructor args
        for arg_spec in module_info.get("constructor_args", []):
            constructor_args.append(_convert_constructor_arg_to_sample_input(arg_spec))

        # Add constructor kwargs
        for key, kwarg_spec in module_info.get("constructor_kwargs", {}).items():
            if kwarg_spec["type"] == "int":
                constructor_kwargs[key] = kwarg_spec["value"]

        # Build forward_inputs (captured during forward pass)
        forward_args = []
        forward_kwargs = {}

        # Add forward inputs
        for inp_spec in module_info.get("inputs", []):
            inp_name = inp_spec["name"]
            # Positional args (arg_0, arg_1, etc.)
            if inp_name.startswith("arg_"):
                forward_args.append(_convert_captured_input_to_sample_input(inp_spec))
            else:
                # Named parameters go to kwargs
                converted = _convert_captured_input_to_sample_input(inp_spec)
                if "tensor" in converted:
                    forward_kwargs[inp_name] = converted
                elif "tensor_list" in converted:
                    forward_kwargs[inp_name] = converted

        module_entry = {
            "name": module_info["name"],
            "module_path": module_info["module_path"],
            "constructor_inputs": {
                "args": constructor_args,
                "kwargs": constructor_kwargs,
            },
            "forward_inputs": {
                "args": forward_args,
                "kwargs": forward_kwargs,
            },
            "complexity": complexity,
        }

        all_module_entries.append(module_entry)

        if complexity == "simple":
            simple_modules.append(module_info["name"])
        else:
            complex_modules.append(module_info["name"])

    # Generate YAML
    yaml_str = f"# Auto-generated unified test configuration for {model_name}\n"
    yaml_str += "# Generated by auto_generate_module_config.py\n"
    yaml_str += "# Format compatible with PyTorch's test_modules.py (using edits.modules.include)\n\n"
    yaml_str += "test_suite_config:\n"
    yaml_str += "  files:\n"
    yaml_str += "    - path: ${TORCH_ROOT}/test/test_modules.py\n"
    yaml_str += "      unlisted_test_mode: skip\n"
    yaml_str += "      tests:\n"
    yaml_str += "        - names:\n"
    yaml_str += "            - '*TestModule*::test_forward'\n"
    yaml_str += "          mode: mandatory_success\n"
    yaml_str += "          tags:\n"
    yaml_str += f"            - model__{model_name}\n"
    yaml_str += "          seed: 123\n"
    yaml_str += "          edits:\n"
    yaml_str += "            modules:\n"
    yaml_str += "              include:\n"

    # Add all modules with their constructor_inputs and forward_inputs
    for entry in all_module_entries:
        yaml_str += f"                - name: {entry['name']}\n"
        yaml_str += f"                  module_path: {entry['module_path']}\n"
        yaml_str += f"                  description: 'Module: {entry['module_path']}'\n"

        # Add tags based on complexity
        if entry["complexity"] == "complex":
            yaml_str += "                  tags: [complex]\n"

        # Constructor inputs
        yaml_str += "                  constructor_inputs:\n"
        if entry["constructor_inputs"]["args"]:
            yaml_str += "                    args:\n"
            for arg in entry["constructor_inputs"]["args"]:
                if "tensor" in arg:
                    yaml_str += "                      - tensor:\n"
                    tensor = arg["tensor"]
                    yaml_str += f"                          shape: {tensor['shape']}\n"
                    if tensor.get("stride"):
                        yaml_str += (
                            f"                          stride: {tensor['stride']}\n"
                        )
                    yaml_str += f"                          storage_offset: {tensor['storage_offset']}\n"
                    yaml_str += f"                          dtype: {tensor['dtype']}\n"
                    yaml_str += (
                        f"                          device: {tensor['device']}\n"
                    )
                    yaml_str += f"                          init: {tensor['init']}\n"
                    if tensor.get("init_args"):
                        yaml_str += "                          init_args:\n"
                        for k, v in tensor["init_args"].items():
                            yaml_str += f"                            {k}: {v}\n"
                elif "tensor_list" in arg:
                    yaml_str += "                      - tensor_list:\n"
                    for tensor in arg["tensor_list"]:
                        yaml_str += (
                            f"                          - shape: {tensor['shape']}\n"
                        )
                        yaml_str += (
                            f"                            dtype: {tensor['dtype']}\n"
                        )
                        yaml_str += (
                            f"                            device: {tensor['device']}\n"
                        )
                        yaml_str += (
                            f"                            init: {tensor['init']}\n"
                        )
                elif "value" in arg:
                    yaml_str += f"                      - value: {arg['value']}\n"
        else:
            yaml_str += "                    args: []\n"

        if entry["constructor_inputs"]["kwargs"]:
            yaml_str += "                    kwargs:\n"
            for key, value in entry["constructor_inputs"]["kwargs"].items():
                yaml_str += f"                      {key}: {value}\n"
        else:
            yaml_str += "                    kwargs: {}\n"

        # Forward inputs
        yaml_str += "                  forward_inputs:\n"
        if entry["forward_inputs"]["args"]:
            yaml_str += "                    args:\n"
            for arg in entry["forward_inputs"]["args"]:
                if "tensor" in arg:
                    yaml_str += "                      - tensor:\n"
                    tensor = arg["tensor"]
                    yaml_str += f"                          shape: {tensor['shape']}\n"
                    if tensor.get("stride"):
                        yaml_str += (
                            f"                          stride: {tensor['stride']}\n"
                        )
                    yaml_str += f"                          storage_offset: {tensor['storage_offset']}\n"
                    yaml_str += f"                          dtype: {tensor['dtype']}\n"
                    yaml_str += (
                        f"                          device: {tensor['device']}\n"
                    )
                    yaml_str += f"                          init: {tensor['init']}\n"
                    if tensor.get("init_args"):
                        yaml_str += "                          init_args:\n"
                        for k, v in tensor["init_args"].items():
                            yaml_str += f"                            {k}: {v}\n"
                elif "tensor_list" in arg:
                    yaml_str += "                      - tensor_list:\n"
                    for tensor in arg["tensor_list"]:
                        yaml_str += (
                            f"                          - shape: {tensor['shape']}\n"
                        )
                        yaml_str += (
                            f"                            dtype: {tensor['dtype']}\n"
                        )
                        yaml_str += (
                            f"                            device: {tensor['device']}\n"
                        )
                        yaml_str += (
                            f"                            init: {tensor['init']}\n"
                        )
                elif "value" in arg:
                    yaml_str += f"                      - value: {arg['value']}\n"

        else:
            yaml_str += "                    args: []\n"

        if entry["forward_inputs"]["kwargs"]:
            yaml_str += "                    kwargs:\n"
            for key, value in entry["forward_inputs"]["kwargs"].items():
                if isinstance(value, dict) and "tensor" in value:
                    yaml_str += f"                      {key}:\n"
                    yaml_str += "                        tensor:\n"
                    tensor = value["tensor"]
                    yaml_str += f"                          shape: {tensor['shape']}\n"
                    yaml_str += f"                          dtype: {tensor['dtype']}\n"
                    yaml_str += (
                        f"                          device: {tensor['device']}\n"
                    )
                    yaml_str += f"                          init: {tensor['init']}\n"
                    if tensor.get("init_args"):
                        yaml_str += "                          init_args:\n"
                        for k, v in tensor["init_args"].items():
                            yaml_str += f"                            {k}: {v}\n"
                elif isinstance(value, dict) and "tensor_list" in value:
                    yaml_str += f"                      {key}:\n"
                    yaml_str += "                        tensor_list:\n"
                    for tensor in value["tensor_list"]:
                        yaml_str += (
                            f"                          - shape: {tensor['shape']}\n"
                        )
                        yaml_str += (
                            f"                            dtype: {tensor['dtype']}\n"
                        )
                        yaml_str += (
                            f"                            device: {tensor['device']}\n"
                        )
                        yaml_str += (
                            f"                            init: {tensor['init']}\n"
                        )
                else:
                    yaml_str += f"                      {key}: {value}\n"
        else:
            yaml_str += "                    kwargs: {}\n"

        yaml_str += "\n"

    # Add custom tests section for eager/compile, layout, and stride validation
    yaml_str += "    # Custom tests for eager/compile, layout, and stride validation\n"
    yaml_str += "    - path: ${TORCH_DEVICE_ROOT}/tests/test_modules_custom.py\n"
    yaml_str += "      unlisted_test_mode: skip\n"
    yaml_str += "      tests:\n"
    yaml_str += "        - names:\n"
    yaml_str += "            - '*TestModuleCustom*::test_eager_vs_compile'\n"
    yaml_str += "            - '*TestModuleCustom*::test_layout'\n"
    yaml_str += "            - '*TestModuleCustom*::test_stride'\n"
    yaml_str += "          mode: mandatory_success\n"
    yaml_str += "          tags:\n"
    yaml_str += f"            - model__{model_name}\n"
    yaml_str += "            - custom_tests\n"
    yaml_str += "          seed: 123\n"
    yaml_str += "          edits:\n"
    yaml_str += "            modules:\n"
    yaml_str += "              include:\n"

    # Add all modules again for custom tests (without complexity tags)
    for entry in all_module_entries:
        yaml_str += f"                - name: {entry['name']}\n"
        yaml_str += f"                  module_path: {entry['module_path']}\n"
        yaml_str += f"                  description: 'Module: {entry['module_path']}'\n"

        # Constructor inputs
        yaml_str += "                  constructor_inputs:\n"
        if entry["constructor_inputs"]["args"]:
            yaml_str += "                    args:\n"
            for arg in entry["constructor_inputs"]["args"]:
                if "tensor" in arg:
                    yaml_str += "                      - tensor:\n"
                    tensor = arg["tensor"]
                    yaml_str += f"                          shape: {tensor['shape']}\n"
                    if tensor.get("stride"):
                        yaml_str += (
                            f"                          stride: {tensor['stride']}\n"
                        )
                    yaml_str += f"                          storage_offset: {tensor['storage_offset']}\n"
                    yaml_str += f"                          dtype: {tensor['dtype']}\n"
                    yaml_str += (
                        f"                          device: {tensor['device']}\n"
                    )
                    yaml_str += f"                          init: {tensor['init']}\n"
                    if tensor.get("init_args"):
                        yaml_str += "                          init_args:\n"
                        for k, v in tensor["init_args"].items():
                            yaml_str += f"                            {k}: {v}\n"
                elif "tensor_list" in arg:
                    yaml_str += "                      - tensor_list:\n"
                    for tensor in arg["tensor_list"]:
                        yaml_str += (
                            f"                          - shape: {tensor['shape']}\n"
                        )
                        yaml_str += (
                            f"                            dtype: {tensor['dtype']}\n"
                        )
                        yaml_str += (
                            f"                            device: {tensor['device']}\n"
                        )
                        yaml_str += (
                            f"                            init: {tensor['init']}\n"
                        )
                elif "value" in arg:
                    yaml_str += f"                      - value: {arg['value']}\n"
        else:
            yaml_str += "                    args: []\n"

        if entry["constructor_inputs"]["kwargs"]:
            yaml_str += "                    kwargs:\n"
            for key, value in entry["constructor_inputs"]["kwargs"].items():
                yaml_str += f"                      {key}: {value}\n"
        else:
            yaml_str += "                    kwargs: {}\n"

        # Forward inputs
        yaml_str += "                  forward_inputs:\n"
        if entry["forward_inputs"]["args"]:
            yaml_str += "                    args:\n"
            for arg in entry["forward_inputs"]["args"]:
                if "tensor" in arg:
                    yaml_str += "                      - tensor:\n"
                    tensor = arg["tensor"]
                    yaml_str += f"                          shape: {tensor['shape']}\n"
                    if tensor.get("stride"):
                        yaml_str += (
                            f"                          stride: {tensor['stride']}\n"
                        )
                    yaml_str += f"                          storage_offset: {tensor['storage_offset']}\n"
                    yaml_str += f"                          dtype: {tensor['dtype']}\n"
                    yaml_str += (
                        f"                          device: {tensor['device']}\n"
                    )
                    yaml_str += f"                          init: {tensor['init']}\n"
                    if tensor.get("init_args"):
                        yaml_str += "                          init_args:\n"
                        for k, v in tensor["init_args"].items():
                            yaml_str += f"                            {k}: {v}\n"
                elif "tensor_list" in arg:
                    yaml_str += "                      - tensor_list:\n"
                    for tensor in arg["tensor_list"]:
                        yaml_str += (
                            f"                          - shape: {tensor['shape']}\n"
                        )
                        yaml_str += (
                            f"                            dtype: {tensor['dtype']}\n"
                        )
                        yaml_str += (
                            f"                            device: {tensor['device']}\n"
                        )
                        yaml_str += (
                            f"                            init: {tensor['init']}\n"
                        )
                elif "value" in arg:
                    yaml_str += f"                      - value: {arg['value']}\n"
        else:
            yaml_str += "                    args: []\n"

        if entry["forward_inputs"]["kwargs"]:
            yaml_str += "                    kwargs:\n"
            for key, value in entry["forward_inputs"]["kwargs"].items():
                if isinstance(value, dict) and "tensor" in value:
                    yaml_str += f"                      {key}:\n"
                    yaml_str += "                        tensor:\n"
                    tensor = value["tensor"]
                    yaml_str += f"                          shape: {tensor['shape']}\n"
                    yaml_str += f"                          dtype: {tensor['dtype']}\n"
                    yaml_str += (
                        f"                          device: {tensor['device']}\n"
                    )
                    yaml_str += f"                          init: {tensor['init']}\n"
                    if tensor.get("init_args"):
                        yaml_str += "                          init_args:\n"
                        for k, v in tensor["init_args"].items():
                            yaml_str += f"                            {k}: {v}\n"
                elif isinstance(value, dict) and "tensor_list" in value:
                    yaml_str += f"                      {key}:\n"
                    yaml_str += "                        tensor_list:\n"
                    for tensor in value["tensor_list"]:
                        yaml_str += (
                            f"                          - shape: {tensor['shape']}\n"
                        )
                        yaml_str += (
                            f"                            dtype: {tensor['dtype']}\n"
                        )
                        yaml_str += (
                            f"                            device: {tensor['device']}\n"
                        )
                        yaml_str += (
                            f"                            init: {tensor['init']}\n"
                        )
                else:
                    yaml_str += f"                      {key}: {value}\n"
        else:
            yaml_str += "                    kwargs: {}\n"

        yaml_str += "\n"

    # Global configuration
    yaml_str += "  global:\n"
    yaml_str += "    supported_dtypes:\n"
    yaml_str += "      - name: float16\n"
    yaml_str += "        precision:\n"
    yaml_str += "          atol: 0.005\n"
    yaml_str += "          rtol: 0.005\n"
    yaml_str += "      - name: float32\n"
    yaml_str += "        precision:\n"
    yaml_str += "          atol: 0.001\n"
    yaml_str += "          rtol: 0.001\n"

    # Set supported_modules to empty list to filter out ALL built-in PyTorch modules.
    # The _OOTModuleListPatcher logic (line 348) keeps modules if they're in
    # supported_modules OR included_modules. With supported_modules=[], only
    # modules from edits.modules.include will be tested.
    yaml_str += "\n    supported_modules: []\n"

    return yaml_str


def generate_test_config_yaml(
    captured_modules: List[Dict[str, Any]], model_name: str
) -> str:
    """Generate test configuration YAML for test suite.

    DEPRECATED: Use generate_unified_yaml_config() instead.
    This function is kept for backward compatibility.
    """
    return generate_unified_yaml_config(captured_modules, model_name)


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
        help="Output YAML file path (default: ../tests/configs/<model>_spyre.yaml)",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.float16).eval()

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
    simple_count = sum(
        1
        for m in capture.get_captured_modules()
        if classify_module_complexity(m) == "simple"
    )
    complex_count = len(capture.get_captured_modules()) - simple_count

    print("\n  Module Summary:")
    print(f"    Simple modules: {simple_count}")
    for module_info in capture.get_captured_modules():
        if classify_module_complexity(module_info) == "simple":
            print(f"      - {module_info['name']}")

    print(f"\n    Complex modules: {complex_count}")
    for module_info in capture.get_captured_modules():
        if classify_module_complexity(module_info) == "complex":
            print(f"      - {module_info['name']}")

    print("\nNext steps:")
    print(f"1. Review the generated YAML file: {output_file}")
    print("2. Run tests:")
    print(f"   export PYTORCH_TEST_CONFIG={output_file.absolute()}")
    print("   cd $PYTORCH/test")
    print(f"   python test_model_ops_v2.py -k {model_name_normalized}")


if __name__ == "__main__":
    main()
