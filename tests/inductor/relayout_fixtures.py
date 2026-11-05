# Copyright 2026 The Torch-Spyre Authors.
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

"""#3440's grouped gather and broadcast graphs, shared by the committed-path
tests (``test_work_division_hint.py``) and the solver-path tests
(``test_solver_relayout_e2e.py``) so both paths are judged on one fixture. The
backend payload check that goes with them lives in ``utils_inductor``.

Not a test module (no ``test_`` prefix).
"""

import torch

import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
from torch_spyre._inductor import spyre_hint


def grouped_relayout_graph(kind: str):
    """#3440's device fixture: ``neg(value)`` feeding ``bmm(attention, hidden)``
    with work-division hints that force a grouped movement on 32 cores.

    gather:    neg {H:4, Lk:8} -> bmm {H:4, Lq:8}. The consumer indexes no Lk,
               so its view of ``hidden`` has 4 owners (one per head), each slice
               assembled from the 8 Lk fragments and held by the 8 Lq cores.
    broadcast: neg {D:2} on 2 cores -> bmm {Lq:16, D:2} on 32. Each of the two
               complete column slices is sent to sixteen consumers.

    Returns ``(fn, device_args, reference, expect)`` where ``expect`` holds the
    source core count, the destination core count, and the distinct owner
    counts of the two views.
    """
    if kind == "gather":
        batch, query, key, width = 4, 8, 128, 64
        producer, consumer = {"H": batch, "Lk": 8}, {"H": batch, "Lq": query}
        expect = dict(
            source_cores=32, destination_cores=32, src_owners=32, dst_owners=4
        )
    elif kind == "broadcast":
        batch, query, key, width = 1, 16, 64, 128
        producer, consumer = {"D": 2}, {"Lq": query, "D": 2}
        expect = dict(source_cores=2, destination_cores=32, src_owners=2, dst_owners=2)
    else:
        raise ValueError(kind)
    torch.manual_seed(0)
    value = torch.randn(batch, key, width, dtype=torch.float16)
    attention = torch.randn(batch, query, key, dtype=torch.float16)
    for name, size in (("H", batch), ("Lk", key), ("Lq", query), ("D", width)):
        _pnd.declare_tensor_dim(name, size)

    def fn(value, attention):
        with spyre_hint(work_div=producer):
            hidden = torch.neg(value)
        with spyre_hint(work_div=consumer):
            return torch.bmm(attention, hidden)

    device_args = (
        _pnd.name_tensor_dims(value.to("spyre"), ["H", "Lk", "D"]),
        _pnd.name_tensor_dims(attention.to("spyre"), ["H", "Lq", "Lk"]),
    )
    reference = torch.bmm(attention.float(), torch.neg(value.float()))
    return fn, device_args, reference, expect
