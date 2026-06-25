# Copyright 2025 The Torch-Spyre Authors. All Rights Reserved.
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
"""Sphinx extension: generate the knowledge graph JSON at build time."""

import json
import logging
from pathlib import Path

from extract_graph import build_graph

logger = logging.getLogger(__name__)


def _generate_graph(app):
    """Run the AST extraction and write graph.json into _static/js/."""
    ext_dir = Path(__file__).resolve().parent
    torch_spyre_root = ext_dir.parents[2] / "torch_spyre"

    graph_data = build_graph(str(torch_spyre_root))

    output_dir = ext_dir.parent / "_static" / "js"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "graph.json"
    with open(output_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    node_count = len(graph_data["nodes"])
    edge_count = len(graph_data["edges"])
    logger.info(
        "[knowledge_graph] Generated graph.json: %d nodes, %d edges",
        node_count,
        edge_count,
    )


def setup(app):
    app.connect("builder-inited", _generate_graph)
    return {"version": "0.1", "parallel_read_safe": True}
