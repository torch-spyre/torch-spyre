# Examples

The
[docs/source/user_guide/examples/](https://github.com/torch-spyre/torch-spyre/tree/main/docs/source/user_guide/examples)
directory in this repository contains self-contained scripts demonstrating
common Torch-Spyre use cases.

## Available Examples

| Script | Description |
|--------|-------------|
| `tensor_allocate.py` | Creating and allocating tensors on the Spyre device |
| `softmax.py` | Computing softmax on Spyre |
| `gelu.py` | Computing GELU activation on Spyre |
| `mean.py` | Computing mean reduction on Spyre |
| `mul.py` | Element-wise multiplication on Spyre |
| `softplus.py` | Computing softplus activation on Spyre |
| `for_each_tile.py` | Tiling a computation into a loop with `for_each_tile` (a map over M, a split-K carry) |
| `profile_ops.py` | Measuring one operation's device time with the PyTorch profiler (see [the cost model](../../compiler/cost_model.md)) |
| `run_cost_model_sweep.py` | Re-measuring every configuration in the cost-model database |
| `attention.py` | Scaled dot-product attention from matmul and softmax (compile-only) |
| `mlp.py` | SwiGLU feed-forward block on Spyre |
| `rms_norm.py` | RMS normalization along the reduction dimension |
| `embedding.py` | Embedding lookup with a zeroed padding row |
| `fp8.py` | FP8 channel-wise quantization roundtrip |
| `tensor_layout.py` | Moving a tensor to Spyre with an explicit on-device layout |
| `custom_op.py` | Registering a custom op with a Spyre lowering |
| `flash_attention.py` | Block-wise flash attention with online softmax |
| `hf_model.py` | Running a HuggingFace causal LM through hf-adapters |

## Distributed Examples

| Script | Description |
|--------|-------------|
| `distributed/allgather.py` | AllGather collective on Spyre |
| `distributed/allreduce.py` | AllReduce collective on Spyre |
| `distributed/barrier.py` | Barrier synchronization on Spyre |
| `distributed/broadcast.py` | Broadcast collective on Spyre |
| `distributed/gather.py` | Gather collective on Spyre |
| `distributed/reduce.py` | Reduce collective on Spyre |
| `distributed/compiled/broadcast_demo_multirank.py` | Multi-rank broadcast walkthrough with pre- and post-broadcast computation |
| `distributed/compiled/all_gather_demo_multirank.py` | Multi-rank all-gather walkthrough |
| `distributed/compiled/all_reduce_demo_multirank.py` | Multi-rank allreduce via `torch.compile`'s plan/run collective ops |
| `distributed/compiled/all_reduce_demo_multicalls_multirank.py` | Multi-rank allreduce with multiple compiled calls sharing one plan |

## Scratchpad Planning Examples

These scripts model the LX scratchpad layout solver in isolation and plot the
resulting buffer layouts. They require `matplotlib` and `numpy`.

| Script | Description |
|--------|-------------|
| `scratchpad/toy_layout.py` | Plot the layout for a fixed ordering of four buffers, with no annealing |
| `scratchpad/random_buffers.py` | Compare first-fit against simulated-annealing quality on a set of random buffers |
| `scratchpad/inplace_annealing.py` | Convergence study on an 18-buffer workload with in-place reuse |

## Provenance Audit

The provenance audit traces which source-to-kernel provenance fields survive
each compilation stage. It is an observability tool, so it now lives under
[Debugging](../debugging/index.md): see
[how to run it](../debugging/provenance/README.md) and the
[example audit of `SimpleMLP`](../debugging/provenance/provenance_audit.md).

## Running an Example

```bash
python docs/source/user_guide/examples/tensor_allocate.py
python docs/source/user_guide/examples/softmax.py
```

## Writing Your Own Example

A minimal Torch-Spyre script follows this pattern:

```python
import torch

DEVICE = torch.device("spyre")

# Move data to device
x = torch.rand(512, 1024, dtype=torch.float16).to(DEVICE)

# Run computation (optionally with torch.compile)
output = torch.some_op(x)

# Move result back to CPU for inspection
print(output.cpu())
```

## See Also

- [Quickstart](../../getting_started/quickstart.md)
- [Running Models](../running_models.md)
