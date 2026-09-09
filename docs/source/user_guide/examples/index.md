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
| `spyre_hints.py` | Using Spyre compiler hints to control tiling |
| `profile_ops.py` | Measuring one operation's device time with the PyTorch profiler (see [the cost model](../../compiler/cost_model.md)) |
| `run_cost_model_sweep.py` | Re-measuring every configuration in the cost-model database |

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
| `scratchpad/profile_native_packer.py` | Paired A/B benchmark of the C++ packer against the Python one, with dispersion statistics (needs neither `matplotlib` nor `numpy`; see [results](../../compiler/native_packer_performance.md)) |

## Provenance Audit

A multi-stage audit that traces a model through the compilation pipeline and
records, at each stage, which source-to-kernel provenance fields are carried or
dropped ([issue #2574](https://github.com/torch-spyre/torch-spyre/issues/2574)).
The **README** explains how to run the audit; the **example** is one generated
artifact from auditing `SimpleMLP`.

```{toctree}
:maxdepth: 1

provenance/README
provenance/provenance_audit
```

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
