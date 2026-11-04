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
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    batch_size = 2
    input_dim = 128
    hidden_dim = 256
    output_dim = 128

    model = SimpleMLP(input_dim, hidden_dim, output_dim).half().to("spyre")
    model.eval()

    x = torch.randn(batch_size, input_dim, dtype=torch.float16, device="spyre")

    compiled_mlp = torch.compile(model)

    with torch.no_grad():
        output = compiled_mlp(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
