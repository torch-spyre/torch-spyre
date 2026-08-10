# Running HuggingFace models on Spyre

Most models people want to run on Spyre already exist as stock
[HuggingFace Transformers](https://github.com/huggingface/transformers)
checkpoints. The
[hf-adapters](https://github.com/torch-spyre/hf-adapters) project lets you
run those checkpoints on the Spyre device without forking the model or
writing a custom class. Each adapter monkey-patches the standard HF model at
load time and replaces only the operations Spyre cannot execute natively:
RoPE precomputation, RMSNorm, LM-head padding, KV cache management, and the
generation loop. Weights, tokenizer, and config come straight from
`transformers`.

hf-adapters is a separate Apache-2.0 project in the same
[torch-spyre](https://github.com/torch-spyre) GitHub organization. It depends
on `torch_spyre` for the Spyre device, so install and build Torch-Spyre first
(see [Installation](../getting_started/installation.md)).

![How hf-adapters runs a stock HuggingFace checkpoint on Spyre: the loader selects an adapter by config type, keeps weights, tokenizer, embeddings, projections, and the MLP from transformers, and replaces only the operations Spyre cannot run natively.](../_static/images/hf-adapters/fig-hf-adapters-approach.svg)

The loader reads the checkpoint's config, picks the adapter for that model
family, and patches one live HF model instance. Everything Spyre executes
natively stays as it is in `transformers`. Only the operations Spyre cannot
run natively are swapped: RoPE becomes a precomputed rotation matmul because
Spyre has no `sin`/`cos`, RMSNorm stays in fp16 because Spyre does not convert
dtype on-device, the LM head is padded to a stick-aligned vocab so work
division fits the 256 MB per-core span limit, the decoder blocks become
compiled `block_forward` functions with raw-tensor KV caches, the attention
mask is built on CPU in fp16, and `generate()` is a 64-block padded decode
loop. The per-operation rationale is documented in the project's
[ARCHITECTURE.md](https://github.com/torch-spyre/hf-adapters/blob/main/ARCHITECTURE.md#how-the-adapters-work).

Coverage spans four kinds of model: generative causal-LMs, embedding models
through sentence-transformers, vision-language models that take an image and
produce text, and speculative-decoding drafters. That is Llama, Qwen,
Granite, Mistral, Phi, Gemma, OLMo, and GPT decoders; BERT, XLM-RoBERTa,
MPNet, and ModernBERT encoders; and the Granite Vision, Mistral3 Vision, and
Gemma 4 multimodal models. The canonical per-adapter list of verified
checkpoints is in the project's
[ARCHITECTURE.md](https://github.com/torch-spyre/hf-adapters/blob/main/ARCHITECTURE.md#verified-checkpoints).

## Install

hf-adapters uses [uv](https://docs.astral.sh/uv/) for dependency management.
Clone it alongside your Torch-Spyre checkout and sync the `spyre` dependency
group, which pulls in `torch_spyre`:

```bash
git clone https://github.com/torch-spyre/hf-adapters.git
cd hf-adapters
uv sync --group spyre
```

The `spyre` group is only needed on a host with Spyre hardware. The CPU
accuracy tests, which compare an adapter against stock HF, run from a plain
`uv sync` with no accelerator present.

## Generative models

Load a causal-LM with `AutoSpyreModelForCausalLM`. It reads the checkpoint's
config, selects the matching adapter, prepares the model for Spyre, and moves
it to the device. The tokenizer is the stock HF one.

```python
from hf_adapters import AutoSpyreModelForCausalLM
from transformers import AutoTokenizer

model = AutoSpyreModelForCausalLM.from_pretrained(
    "ibm-granite/granite-3.3-8b-instruct"
)
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-8b-instruct")

outputs = model.generate(tokenizer, ["What is 2+2?"], max_new_tokens=128)
print(outputs[0])
```

The `generate` method attached to the model is not the stock HF one. It runs a
64-block padded decode loop and takes the tokenizer and a list of prompts
rather than pre-tokenized `input_ids`. The signature is
`generate(tokenizer, prompts, max_new_tokens=128, do_sample=False,
temperature=1.0, top_k=50)`. See
[docs/generate_vs_stock_hf.md](https://github.com/torch-spyre/hf-adapters/blob/main/docs/generate_vs_stock_hf.md)
in the project for how it differs from `transformers.generate`.

## Embedding models

For embedding models, use `sentence-transformers` with `backend="spyre"`.
Importing `hf_adapters.st_backend` registers the backend, after which
`SentenceTransformer` applies the right Spyre adapter when it loads the model.

```python
import hf_adapters.st_backend  # registers the Spyre backend
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", backend="spyre")
embeddings = model.encode(["hello world", "how are you"])
```

The standard `SentenceTransformer` methods, `encode()`, `similarity()`, and
the rest, work unchanged.

## A note on numerical accuracy

Spyre has known numerical accuracy limits on the single-token decode path.
Greedy token mismatches between a CPU run and a Spyre run are expected on some
models until the relevant `torch_spyre` fixes merge. The hf-adapters test
suite accounts for this: the multimodal tests, for example, assert a per-step
logit cosine floor rather than exact token equality. If you are comparing
Spyre output against a CPU reference and see divergence on later tokens, this
is the reason. The per-adapter notes in
[ARCHITECTURE.md](https://github.com/torch-spyre/hf-adapters/blob/main/ARCHITECTURE.md#verified-checkpoints)
record which checkpoints are affected.

## Learn more about the approach

The hf-adapters project documents its design in detail:

- [ARCHITECTURE.md](https://github.com/torch-spyre/hf-adapters/blob/main/ARCHITECTURE.md)
  covers how the adapters work, the per-operation deviations from stock
  HuggingFace, per-model adaptations, and the verified checkpoint list.
- [generate_vs_stock_hf.md](https://github.com/torch-spyre/hf-adapters/blob/main/docs/generate_vs_stock_hf.md)
  explains how the Spyre `generate()` differs from `transformers.generate`.
- [spyre-numerical-findings.md](https://github.com/torch-spyre/hf-adapters/blob/main/docs/spyre-numerical-findings.md)
  records the numerical accuracy findings behind the caveat above.
- [fms_comparison.md](https://github.com/torch-spyre/hf-adapters/blob/main/docs/fms_comparison.md)
  compares the adapter path with the Foundation Model Stack.

## See Also

- [Running Models](running_models.md) for compiling your own models with
  `torch.compile`
- [Supported Operations](supported_operations.md)
- [hf-adapters on GitHub](https://github.com/torch-spyre/hf-adapters)
