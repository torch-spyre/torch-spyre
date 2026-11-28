# spyre-cli examples

Example kernels for trying `spyre launch`. They are compiled on the machine
that runs them rather than checked in, because a compiled kernel only loads on
the toolchain that built it.

On a machine with a Spyre card, build them first, from this directory:

```bash
python3 examples.py            # all examples, into build/
python3 examples.py add mm     # or just some
```

`examples.py` compiles each op with `torch.compile(fn, backend="inductor")`,
checks the result against CPU, and copies the one kernel directory it produces
to `build/<name>`. That directory holds the SDSC bundle (`bundle.mlir`,
`sdsc_*.json`) and `spyreCodeDir/`, which is what `spyre launch` loads.

It rejects an example that `spyre launch` could not run correctly: one that
compiles to more than one kernel, since `spyre launch` runs one; one whose
kernel takes its inputs in a different order than the op does; or one whose
kernel takes arguments besides its inputs and output. The last two are read
from the compiled wrapper's kernel call, and are described under Limitations.

Tensors are fp16 unless noted:

| Example | Op | Command |
|---|---|---|
| `add` | `a + b` | `spyre launch -i 512x1024@fp16 -i 512x1024@fp16 -o 512x1024@fp16 build/add` |
| `sub` | `a - b` | `spyre launch -i 512x1024@fp16 -i 512x1024@fp16 -o 512x1024@fp16 build/sub` |
| `mul` | `a * b` | `spyre launch -i 512x1024@fp16 -i 512x1024@fp16 -o 512x1024@fp16 build/mul` |
| `div` | `a / b` (positive inputs) | `spyre launch -i 256x2048@fp16 -i 256x2048@fp16 -o 256x2048@fp16 build/div` |
| `relu` | `torch.relu(a)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 build/relu` |
| `exp` | `torch.exp(a)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 build/exp` |
| `sigmoid` | `torch.sigmoid(a)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 build/sigmoid` |
| `gelu` | `F.gelu(a)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 build/gelu` |
| `fma` | `a * b + c` | `spyre launch -i 512x1024@fp16 -i 512x1024@fp16 -i 512x1024@fp16 -o 512x1024@fp16 build/fma` |
| `swiglu` | `F.silu(a) * b` | `spyre launch -i 128x4096@fp16 -i 128x4096@fp16 -o 128x4096@fp16 build/swiglu` |
| `add_fp32` | `a + b`, fp32 | `spyre launch -i 512x1024@fp32 -i 512x1024@fp32 -o 512x1024@fp32 build/add_fp32` |
| `mul_bf16` | `a * b`, bf16 | `spyre launch -i 512x1024@bf16 -i 512x1024@bf16 -o 512x1024@bf16 build/mul_bf16` |
| `add_3d` | `a + b`, 3-D | `spyre launch -i 4x256x512@fp16 -i 4x256x512@fp16 -o 4x256x512@fp16 build/add_3d` |
| `softmax` | `torch.softmax(a, dim=-1)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 build/softmax` |
| `mean_keepdim` | `a.mean(dim=-1, keepdim=True)` | `spyre launch -i 512x1024@fp16 -o 512x1@fp16 build/mean_keepdim` |
| `mm` | `a @ b` | `spyre launch -i 512x1024@fp16 -i 1024x256@fp16 -o 512x256@fp16 build/mm` |
| `bmm` | `torch.bmm(a, b)` | `spyre launch -i 8x128x256@fp16 -i 8x256x128@fp16 -o 8x128x128@fp16 build/bmm` |

## Fused PE + SFP

Each of these compiles to one kernel in which a matmul on the PE array
(`pt`) feeds elementwise or reduction steps on the SFP (`sfp`), and
`examples.py` checks that it uses exactly those two units. `x` is
`512x1024`, `w` `1024x256` and `r` `512x256`:

| Example | Op | Command |
|---|---|---|
| `mm_relu` | `relu(x @ w)` | `spyre launch -i 512x1024@fp16 -i 1024x256@fp16 -o 512x256@fp16 build/mm_relu` |
| `mm_gelu` | `F.gelu(x @ w)` | `spyre launch -i 512x1024@fp16 -i 1024x256@fp16 -o 512x256@fp16 build/mm_gelu` |
| `mm_silu` | `F.silu(x @ w)` | `spyre launch -i 512x1024@fp16 -i 1024x256@fp16 -o 512x256@fp16 build/mm_silu` |
| `mm_residual` | `x @ w + r` | `spyre launch -i 512x1024@fp16 -i 1024x256@fp16 -i 512x256@fp16 -o 512x256@fp16 build/mm_residual` |
| `mm_scale` | `(x @ w) * s`, `s` a 1-element tensor | `spyre launch -i 512x1024@fp16 -i 1024x256@fp16 -i 1@fp16 -o 512x256@fp16 build/mm_scale` |
| `swiglu_mlp` | `F.silu(x @ w1) * (x @ w3)` | `spyre launch -i 512x1024@fp16 -i 1024x256@fp16 -i 1024x256@fp16 -o 512x256@fp16 build/swiglu_mlp` |
| `attn_scores` | `torch.softmax(q @ k.T, dim=-1)` | `spyre launch -i 512x128@fp16 -i 512x128@fp16 -o 512x512@fp16 build/attn_scores` |

The CLI fills every input with ones, so e.g. `add` prints a tensor of 2s.

## Testing

`examples.py` holds the registry of examples (shapes, dtype and tolerance,
the latter taken from the matching test in `tests/inductor/test_inductor_ops.py`)
and the op behind each one. `check.py <name> [dir]` runs one built example
through both pathways: `spyre launch` must exit 0 with no device error, and
the SDK, given random inputs, must match the CPU reference.
`tests/test_examples.py` builds every example once into a temporary directory,
then runs `check.py` for each, every step in a fresh process:

```bash
python3 -m pytest tests/test_examples.py -v
```

The tests skip themselves when no Spyre device is present. To add an example,
add an entry to `EXAMPLES` and `references()` in `examples.py`.

## Limitations

- spyre-cli allocates every tensor with the default device layout. A kernel
  compiled for any other layout gives wrong results with no error. For example
  `a.sum(dim=-1)` writes a `(512,)` output with `device_size=[1, 512, 64]`,
  where the default is `[8, 64]`, so it is not included here.
- Scalar constants in the op become extra kernel arguments that only the
  compiled wrapper knows about. `a * 0.5 + 1.0` compiles to a kernel taking
  `(a, 0.5, 1.0, out)`, so spyre launch cannot supply them and the runtime
  rejects the launch ("Number of inputs provided (2) does not match number of
  inputs expected (4)"). Ops like `torch.where(a > 0, a, b)` and an RMSNorm
  with an epsilon hit the same limit.
  A constant passed as a tensor input instead works: `mm_scale` multiplies by
  a 1-element tensor, and runs where the same op with a Python scalar does
  not.
- A kernel can take its inputs in a different order than the op, so `spyre
  launch`, which binds positionally, runs it with the wrong tensors. The
  kernel for `torch.addmm(b, x, w)` takes `(x, w, b, out)`. Reordering the op's
  parameters does not help, because the kernel reorders by position: with the
  inputs given as `(x, w, b)` it takes `(w, b, x, out)`. Launched in the op's
  order it runs and gives wrong results, with no error. `F.linear` behaves the
  same way, so neither is included here, and `examples.py` rejects any example
  whose kernel reorders its inputs.
