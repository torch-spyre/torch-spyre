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
"""Running a HuggingFace causal language model on Spyre.

This script loads a Granite instruct checkpoint with
AutoSpyreModelForCausalLM, tokenizes a prompt, generates a short continuation
on the device, and decodes it. It requires a Spyre device and the hf-adapters
package to run.
"""

from hf_adapters import AutoSpyreModelForCausalLM
from transformers import AutoTokenizer

model_id = "ibm-granite/granite-3.3-8b-instruct"

# AutoSpyreModelForCausalLM selects the matching adapter, prepares the model
# for Spyre, and moves it to the device. The tokenizer is the stock HF one.
model = AutoSpyreModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Tokenize a prompt on CPU
inputs = tokenizer(["What is 2+2?"], return_tensors="pt", padding=True)

# Generate on the device. generate() takes the tokenized input_ids and returns
# the input prefix followed by the newly generated tokens.
sequences = model.generate(**inputs, max_new_tokens=32)

# Decode only the generated continuation
outputs = tokenizer.batch_decode(
    sequences[:, inputs["input_ids"].shape[1] :],
    skip_special_tokens=True,
)
print(outputs[0])
