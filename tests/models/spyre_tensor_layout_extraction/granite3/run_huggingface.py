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

import logging
import argparse
from collections import Counter, defaultdict

import torch
import torch._inductor.config as inductor_config
import torch._dynamo as dynamo
from transformers import AutoModelForCausalLM

from ..extract_spyre_tensor_layout import (
    make_capture_fns,
    process_ir_graphs,
    collect_primitive_ops,
    deduplicate_ops,
    MODULE_CLASSNAME_FALLBACK,
)
from ..utils import write_yaml
from typing import Any
from transformers.cache_utils import DynamicCache

log = logging.getLogger(__name__)

# ── Inductor config ───────────────────────────────────────────────────────
inductor_config.trace.provenance_tracking_level = 1
inductor_config.fx_graph_cache = False
torch._functorch.config.enable_autograd_cache = False

# ── Model identity ────────────────────────────────────────────────────────
MODEL_ID = "ibm-granite/granite-3.3-8b-instruct"
MODULE_CLASSNAME_FALLBACK["GraniteRMSNorm"] = "torch.nn.functional.rms_norm"

_pre_node_info: dict[Any, Any] = {}
_pre_node_order: dict[Any, int] = {}
_order_counter: list[int] = [0]
_ir_graphs: list[dict[str, Any]] = []
_cur_pre_graph_id: list[int | None] = [None]

_dynamo_backend, _install_patches, _remove_patches = make_capture_fns(
    _pre_node_info,
    _pre_node_order,
    _order_counter,
    _ir_graphs,
    _cur_pre_graph_id,
)


# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (Granite-specific)
# ══════════════════════════════════════════════════════════════════════════


def load_model():
    print(f"Loading {MODEL_ID} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        use_cache=True,
    ).eval()
    print(f"{type(model).__name__} loaded.")
    return model


def make_prefill_inputs(
    model, batch_size: int = 1, seq_length: int = 41, past_key_values=None
) -> dict:
    V = model.config.vocab_size
    return {
        "input_ids": torch.randint(0, V, (batch_size, seq_length), dtype=torch.long),
        "position_ids": torch.arange(seq_length, dtype=torch.long).unsqueeze(0),
        "past_key_values": past_key_values
        if past_key_values is not None
        else DynamicCache(),
        "use_cache": True,
    }


def make_decode_inputs(
    model, batch_size: int = 1, prefill_length: int = 41, past_key_values=None
) -> dict:
    V = model.config.vocab_size
    return {
        "input_ids": torch.randint(0, V, (batch_size, 1), dtype=torch.long),
        "position_ids": torch.tensor([[prefill_length]], dtype=torch.long),
        "past_key_values": past_key_values,
        "use_cache": True,
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description=f"Capture Spyre layouts for {MODEL_ID}"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=41)
    parser.add_argument(
        "--output",
        default=None,
        help="Output YAML (default: <model>-with-device-layout.yaml)",
    )
    args = parser.parse_args()

    model_name = MODEL_ID.split("/")[-1]
    output_path = args.output or f"{model_name}-with-device-layout.yaml"

    dynamo.reset()
    model = load_model()

    try:
        _install_patches()

        compiled = torch.compile(
            model, backend=_dynamo_backend, fullgraph=False, dynamic=False
        )

        print("Running prefill pass ...")
        cache = DynamicCache()
        with torch.no_grad():
            try:
                _ = compiled(
                    **make_prefill_inputs(
                        model, args.batch_size, args.seq_length, past_key_values=cache
                    )
                )
                print("Prefill done.")
            except (RuntimeError, ValueError) as exc:
                print(f"Prefill error: {exc}")
                log.exception("Prefill failed")

        print("Running decode pass ...")
        with torch.no_grad():
            try:
                _ = compiled(
                    **make_decode_inputs(
                        model, args.batch_size, args.seq_length, past_key_values=cache
                    )
                )
                print("Decode done.")
            except (RuntimeError, ValueError) as exc:
                print(f"Decode error: {exc}")
                log.exception("Decode failed")

    finally:
        _remove_patches()

    if not _ir_graphs:
        print("\nERROR: No IR graphs captured.")
        print("  Set TORCHINDUCTOR_FX_GRAPH_CACHE=0 and retry, or delete")
        print("  /tmp/torchinductor_$USER and ~/.triton/cache.")
        return

    print(
        f"Captured {len(_ir_graphs)} IR graph(s), "
        f"{len(_pre_node_info)} pre-decomp nodes. Processing ..."
    )

    all_layouts, via_map, via_walk = process_ir_graphs(
        _ir_graphs, _pre_node_info, _pre_node_order
    )

    captured_keys = {(e["pre_node"], e["pre_node_graph_id"]) for e in all_layouts}
    primitives = collect_primitive_ops(captured_keys, _pre_node_info, _pre_node_order)

    layouts_by_key = defaultdict(list)
    primitives_by_key = defaultdict(list)
    for e in all_layouts:
        layouts_by_key[(e["pre_node"], e["pre_node_graph_id"])].append(e)
    for e in primitives:
        primitives_by_key[(e["pre_node"], e["pre_node_graph_id"])].append(e)

    all_merged = []
    for pre_key in sorted(_pre_node_order, key=lambda k: _pre_node_order[k]):
        order = _pre_node_order[pre_key]
        for e in layouts_by_key.get(pre_key, []):
            e["execution_order"] = order
            all_merged.append(e)
        for e in primitives_by_key.get(pre_key, []):
            e["execution_order"] = order
            all_merged.append(e)

    all_layouts = deduplicate_ops(all_merged)
    comp_ops = [e for e in all_layouts if not e.get("primitive", False)]
    prim_ops = [e for e in all_layouts if e.get("primitive", False)]

    print(f"\n{'=' * 70}")
    print(
        f"Total ops  : {len(all_layouts)}  "
        f"(compiled: {len(comp_ops)}, primitive: {len(prim_ops)})"
    )
    print(f"Resolution : mapping={via_map}, walk={via_walk}")

    print(f"\n── Compiled ({len(comp_ops)}) ─────────────────────────────────────")
    for op, cnt in sorted(
        Counter(e["op"] for e in comp_ops).items(), key=lambda x: -x[1]
    ):
        print(f"  {cnt:4d}  {op}")

    print(f"\n── Primitive ({len(prim_ops)}) — fused/view/int ops ───────────────")
    for op, cnt in sorted(
        Counter(e["op"] for e in prim_ops).items(), key=lambda x: -x[1]
    ):
        print(f"  {cnt:4d}  {op}")

    print(f"\nWriting YAML → {output_path} ...")
    write_yaml(all_layouts, output_path, model_name)
    print(f"Saved → {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
