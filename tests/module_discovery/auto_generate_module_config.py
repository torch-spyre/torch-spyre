"""
Automatic module configuration generator using forward hooks.

This script automatically generates YAML configuration for all unique modules
in a model by:
1. Loading the model
2. Registering forward hooks on all modules
3. Running a forward pass to capture module inputs
4. Analyzing captured data to generate YAML config

Usage:
    python auto_generate_module_config.py --model_path ibm-granite/granite-3.3-8b-instruct --seq_len 128
"""

import torch
import argparse
import yaml
import json
from pathlib import Path
import hashlib
from typing import Dict, List, Any, Tuple, Set
from transformers import AutoModel, AutoTokenizer
from torch.utils._pytree import tree_flatten
import logging

logger = logging.getLogger(__name__)


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
    logger.info(
        f"Found {len(existing_modules)} existing modules in PyTorch's module_db"
    )
except ImportError:
    existing_modules = set()
    logger.warning("could not import module_db, will not filter duplicates")


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
        self.seen_module_configs: Set[str] = (
            set()
        )  # Track unique configs, not just types
        # Track model-level context (KV cache, execution mode)
        self.current_model_context: Dict[str, Any] = {}

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

    def create_model_hook(self):
        """Create a model-level hook to detect execution mode (prefill vs decode)..

        This hook runs BEFORE module-level hooks and sets context that module hooks can use.
        """

        def model_hook(model, args, kwargs):
            # Capture model-level context
            past_key_values = kwargs.get("past_key_values", None)
            attention_mask = kwargs.get("attention_mask", None)

            # Detect execution mode
            if past_key_values is None:
                mode = "prefill"
            else:
                mode = "decode"

            # Store context for module hooks to access
            self.current_model_context = {
                "mode": mode,
                "attention_mask": attention_mask,
            }

        return model_hook

    def create_hook(self, module_name: str, module_type: str, module_instance):
        """Create a forward hook that captures module input information.

        This hook captures unique invocations of the module, deduplicating by input pattern.
        This allows testing with multiple input configurations (e.g., prefill + decode)
        without storing redundant identical invocations.
        """

        def hook(module, args, kwargs):
            # Capture constructor information to create unique config identifier
            constructor_info = self.capture_constructor_info(
                module, module_name, module_type
            )

            # Create a unique identifier based on module type + constructor args
            # This allows us to capture multiple variants of the same module type
            config_signature = self._create_config_signature(
                module_type, constructor_info
            )

            # Create unique module name for this variant
            unique_module_name = self._create_unique_module_name(
                module_type, constructor_info, config_signature
            )

            # Initialize module_info if this is the first invocation
            if unique_module_name not in self.module_data:
                self.seen_module_configs.add(config_signature)

                self.module_data[unique_module_name] = {
                    "name": unique_module_name,
                    "module_type": module_type,
                    "module_path": f"{module.__class__.__module__}.{module.__class__.__name__}",
                    "example_instance": module_name,
                    "constructor_args": constructor_info["constructor_args"],
                    "constructor_kwargs": constructor_info["constructor_kwargs"],
                    "invocations": [],  # List of unique invocations
                    "invocation_signatures": set(),  # Track seen invocation patterns
                }

            # Capture this invocation's inputs
            invocation_inputs = []

            # Analyze positional arguments using pytree
            for i, arg in enumerate(args):
                input_info = _process_pytree_structure(arg, f"arg_{i}")
                if input_info:
                    invocation_inputs.append(input_info)

            # Analyze keyword arguments using pytree
            for key, value in kwargs.items():
                if key in ("past_key_values", "past_key_value"):
                    continue  # Skip - not needed for module-level tests
                input_info = _process_pytree_structure(value, key)
                if input_info:
                    invocation_inputs.append(input_info)

            # Create signature for this invocation to detect duplicates
            invocation_sig = self._create_invocation_signature(invocation_inputs)

            # Only add if this is a new unique invocation pattern
            if (
                invocation_sig
                not in self.module_data[unique_module_name]["invocation_signatures"]
            ):
                self.module_data[unique_module_name]["invocation_signatures"].add(
                    invocation_sig
                )
                self.module_data[unique_module_name]["invocations"].append(
                    invocation_inputs
                )

        return hook

    def _create_config_signature(
        self, module_type: str, constructor_info: Dict[str, Any]
    ) -> str:
        """Create a unique signature for a module configuration.

        This signature is used to detect duplicate configurations.
        layer_idx is EXCLUDED because we only need one representative layer.
        """
        # Build signature from constructor args
        sig_parts = [module_type]

        for arg in constructor_info.get("constructor_args", []):
            if arg["type"] == "int":
                sig_parts.append(f"int_{arg['value']}")
            elif arg["type"] == "config":
                sig_parts.append(f"config_{arg['config_path']}")
            else:
                sig_parts.append(f"{arg['type']}")

        # IMPORTANT: Exclude layer_idx from signature
        # We only need one representative layer, not all 40 decoder layers
        for key, kwarg in constructor_info.get("constructor_kwargs", {}).items():
            if key == "layer_idx":
                continue  # Skip layer_idx - treat all layers as same config
            if kwarg["type"] == "int":
                sig_parts.append(f"{key}_{kwarg['value']}")

        return "__".join(sig_parts)

    def _create_unique_module_name(
        self, module_type: str, constructor_info: Dict[str, Any], config_signature: str
    ) -> str:
        """Create a unique, human-readable name for a module variant.

        Names are based on the config signature (which excludes layer_idx),
        ensuring that modules with identical configs get the same name and
        their invocations are grouped together.

        Examples:
            MyRMSNorm with dim=4096 -> MyRMSNorm_4096
            MyRMSNorm with dim=2048 -> MyRMSNorm_2048
            GraniteDecoderLayer (all layers same config) -> GraniteDecoderLayer_layer0
        """
        # Check if there's a simple int arg (common for norm layers)
        args = constructor_info.get("constructor_args", [])
        if len(args) == 1 and args[0]["type"] == "int":
            return f"{module_type}_{args[0]['value']}"

        # For modules with layer_idx, use "layer0" as representative name
        # since all layers have the same config (layer_idx excluded from signature)
        kwargs = constructor_info.get("constructor_kwargs", {})
        if "layer_idx" in kwargs:
            # Use layer0 as the canonical name for all layers
            return f"{module_type}_layer0"

        # If no simple identifier, use a hash of the config signature
        # This ensures uniqueness while keeping names readable
        sig_hash = hashlib.sha256(config_signature.encode()).hexdigest()[:8]
        return f"{module_type}_{sig_hash}"

    def _create_invocation_signature(
        self, invocation_inputs: List[Dict[str, Any]]
    ) -> str:
        """Create a signature for an invocation based on input patterns.

        This signature captures the structure of inputs (shapes, dtypes, types)
        but not the actual values, allowing us to deduplicate identical invocations.

        Args:
            invocation_inputs: List of input info dicts from _process_pytree_structure

        Returns:
            A string signature representing this invocation pattern
        """

        def _extract_pattern(input_info: Dict[str, Any]) -> Dict[str, Any]:
            """Extract the pattern from an input, removing variable data.

            input_info structure from _process_pytree_structure:
            - Single tensor: {"name": "arg_0", "shape": [...], "dtype": ..., ...}
            - Container: {"name": "arg_0", "type": "list/tuple/dict/pytree", "items": [...]}
            """
            # Check if this is a container with items
            if "type" in input_info and "items" in input_info:
                # Container (list, tuple, dict, pytree)
                pattern = {
                    "type": input_info["type"],
                    "items": [
                        {
                            "shape": item.get("shape"),
                            "dtype": str(item.get("dtype")),
                            "init": item.get("init"),
                        }
                        for item in input_info["items"]
                    ],
                }
                return pattern
            elif "shape" in input_info:
                # Single tensor
                return {
                    "type": "tensor",
                    "shape": input_info.get("shape"),
                    "dtype": str(input_info.get("dtype")),
                    "init": input_info.get("init"),
                }
            else:
                # Unknown structure
                return {"type": "unknown"}

        # Build pattern for all inputs
        patterns = []
        for input_info in invocation_inputs:
            # input_info is already a dict with structure like:
            # {"name": "arg_0", "tensor": {...}} or {"name": "x", "type": "list", "items": [...]}
            # We want to extract the pattern from the whole input_info
            patterns.append(_extract_pattern(input_info))

        # Convert to JSON for consistent string representation
        pattern_str = json.dumps(patterns, sort_keys=True)
        return hashlib.sha256(pattern_str.encode()).hexdigest()

    def get_captured_modules(self) -> List[Dict[str, Any]]:
        """Return list of captured module information."""
        # Remove invocation_signatures before returning (internal tracking only)
        result = []
        for module_data in self.module_data.values():
            module_copy = module_data.copy()
            module_copy.pop("invocation_signatures", None)
            result.append(module_copy)
        return result


def get_all_custom_modules(model) -> List[Tuple[str, str, Any]]:
    """
    Get ALL custom module instances from the model (not just unique types).

    Returns:
        List of (module_name, module_type, module_instance) tuples
    """
    custom_modules = []
    for name, module in model.named_modules():
        if name == "":  # Skip root
            continue

        module_type = type(module).__name__

        # Skip if already in upstream module_db
        if module_type in existing_modules:
            continue

        # Keep ALL instances (not just first of each type)
        custom_modules.append((name, module_type, module))

    return custom_modules


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
        module_info: Captured module information with multiple invocations

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

    # Build forward_inputs from all invocations
    # NEW: Handle multiple invocations - each invocation becomes a separate input set
    invocations = module_info.get("invocations", [])

    if not invocations:
        # Fallback for old format (backward compatibility)
        invocations = [module_info.get("inputs", [])]

    # Process each invocation
    forward_inputs_list = []
    for invocation_inputs in invocations:
        forward_args = []
        forward_kwargs = {}

        for inp_spec in invocation_inputs:
            # Validate inp_spec has required fields
            if "name" not in inp_spec:
                logger.error(f"inp_spec missing 'name' field: {inp_spec}")
                continue  # Skip malformed entries

            inp_name = inp_spec["name"]
            converted = _convert_captured_input_to_sample_input(inp_spec)

            if inp_name.startswith("arg_"):
                forward_args.append(converted)
            else:
                forward_kwargs[inp_name] = converted

        forward_inputs_list.append(
            {
                "args": forward_args if forward_args else [],
                "kwargs": forward_kwargs if forward_kwargs else {},
            }
        )

    forward_inputs = forward_inputs_list

    # Build module entry
    entry = {
        "name": module_info["name"],
        "module_path": module_info["module_path"],
        "description": f"Module: {module_info['module_path']}",
        "constructor_inputs": {
            "args": constructor_args if constructor_args else [],
            "kwargs": constructor_kwargs if constructor_kwargs else {},
        },
        "forward_inputs": forward_inputs,
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
                                "*TestModuleCustom*::test_layout_stride",
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

    logger.info(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).eval()
    all_custom_modules = get_all_custom_modules(model)
    logger.info(f"Found {len(all_custom_modules)} custom module instances")

    # Create capture object
    capture = ModuleInfoCapture()

    # This hook sets context that module-level hooks will read
    model_hook = capture.create_model_hook()
    model_handle = model.register_forward_pre_hook(model_hook, with_kwargs=True)
    handles = [model_handle]

    # Register hooks on ALL custom module instances (not just unique types)
    for module_name, module_type, module_instance in all_custom_modules:
        hook = capture.create_hook(module_name, module_type, module_instance)
        handle = module_instance.register_forward_pre_hook(hook, with_kwargs=True)
        handles.append(handle)

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
    logger.info(f"  Input shape: {inputs['input_ids'].shape}")
    outputs = None
    try:
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
    except Exception as e:
        logger.exception(f"  ERROR during prefill: {e}")

    if (
        outputs is not None
        and hasattr(outputs, "past_key_values")
        and outputs.past_key_values is not None
    ):
        try:
            with torch.no_grad():
                # Single new token for decode
                next_token = torch.zeros(
                    (inputs["input_ids"].shape[0], 1), dtype=torch.long
                )
                decode_inputs = {
                    "input_ids": next_token,  # Shape: [B, 1]
                    "attention_mask": torch.cat(
                        [
                            inputs["attention_mask"],
                            torch.ones(
                                (inputs["input_ids"].shape[0], 1), dtype=torch.long
                            ),
                        ],
                        dim=1,
                    ),
                    "past_key_values": outputs.past_key_values,  # Use cached KV
                    "use_cache": True,
                }
                logger.info(
                    f"Decode input_ids shape: {decode_inputs['input_ids'].shape}"
                )
                logger.info(
                    f"Decode attention_mask shape: {decode_inputs['attention_mask'].shape}"
                )
                logger.info(
                    f"Decode past_key_values layers: {len(decode_inputs['past_key_values'])}"
                )

                decode_outputs = model(**decode_inputs)
                logger.info(
                    f"Decode complete. Output shape: {decode_outputs.logits.shape if hasattr(decode_outputs, 'logits') else 'N/A'}"
                )
        except Exception:
            logger.exception("ERROR during decode")
    else:
        logger.info("\n  Skipping decode pass - no KV cache available")

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Generate YAML
    # Extract model name from path (handle both local paths and HuggingFace paths)
    model_path_parts = args.model_path.rstrip("/").split("/")
    model_name = model_path_parts[
        -1
    ]  # e.g., "granite-3.3-8b-instruct" or "granite-3.0-2b-instruct"

    # For the YAML content, use underscores for the model_name field
    model_name_normalized = model_name.replace("-", "_").replace(".", "_")

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

    logger.info(f"\n✓ Generated unified configuration: {output_file}")

    # Print module summary
    total_modules = len(capture.get_captured_modules())
    logger.info("\n  Module Summary:")
    logger.info(f"    Total modules captured: {total_modules}")
    for module_info in capture.get_captured_modules():
        logger.info(f"      - {module_info['name']}")


if __name__ == "__main__":
    main()
