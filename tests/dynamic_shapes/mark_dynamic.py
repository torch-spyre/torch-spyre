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


# Compiled via torch.compile with an explicit dynamic dimension registered
# through torch._dynamo.mark_dynamic.  Dim 0 is annotated with min=1, max=576,
# so the ShapeEnv records a finite upper bound for that symbol. This is planned to
# exercise the path where compute_max_size can return the
# ShapeEnv bound directly rather than falling back to size_hint.
# Right now this is a work in progress- the bound is not yet propagated into
# the SDSC.


import torch
import torch._dynamo as dynamo
import torch.nn.functional as F

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)


def gelu_fn(a):
    return F.gelu(a)


x = torch.rand(512, 1024, dtype=torch.float16)

# Mark dim 0 as dynamic with an explicit upper bound.
x_device = x.to(DEVICE)
dynamo.mark_dynamic(x_device, 0, min=1, max=576)
compiled_fn = torch.compile(gelu_fn)
cpu_result = gelu_fn(x)

# EXPECTED FAILURE: dispatch currently fails here. bundle.py emits an extra
# input_arg parameter for the mark_dynamic bound, but the runtime (.run() /
# launch_jobplan) isn't yet wired to supply that extra argument, so Deeptools
# raises a "Number of inputs mismatches." DtException. Tracked in #2434.
# Kept as a reproducer for that issue rather than removed or silently fixed
# up here.
_EXPECTED_MSG_SUBSTRINGS = ("Number of inputs mismatches",)

try:
    compiled_result = compiled_fn(x_device).cpu()
except RuntimeError as e:
    if not any(s in str(e) for s in _EXPECTED_MSG_SUBSTRINGS):
        raise
    print(f"Expected dispatch failure (see #2434): {e}")
else:
    # Compare results
    print(f"CPU result\n{cpu_result}")
    print(f"Spyre Compiled result\n{compiled_result}")
    cpu_delta = torch.abs(compiled_result - cpu_result).max()
    print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
