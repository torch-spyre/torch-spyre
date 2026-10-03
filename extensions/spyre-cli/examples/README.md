# spyre-cli examples

Precompiled kernel directories you can launch with `spyre launch` without
compiling anything first. Each directory is the unmodified output of
`torch.compile(fn, backend="inductor")` for one op: `bundle.mlir` and
`sdsc_*.json` are the SDSC bundle, `spyreCodeDir/` is what `spyre launch`
actually loads.

Tensors are fp16 unless noted. Run from this directory on a machine with a Spyre card:

| Example | Op | Command |
|---|---|---|
| `add` | `a + b` | `spyre launch -i 512x1024@fp16 -i 512x1024@fp16 -o 512x1024@fp16 add` |
| `sub` | `a - b` | `spyre launch -i 512x1024@fp16 -i 512x1024@fp16 -o 512x1024@fp16 sub` |
| `mul` | `a * b` | `spyre launch -i 512x1024@fp16 -i 512x1024@fp16 -o 512x1024@fp16 mul` |
| `div` | `a / b` (positive inputs) | `spyre launch -i 256x2048@fp16 -i 256x2048@fp16 -o 256x2048@fp16 div` |
| `relu` | `torch.relu(a)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 relu` |
| `exp` | `torch.exp(a)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 exp` |
| `sigmoid` | `torch.sigmoid(a)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 sigmoid` |
| `gelu` | `F.gelu(a)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 gelu` |
| `fma` | `a * b + c` | `spyre launch -i 512x1024@fp16 -i 512x1024@fp16 -i 512x1024@fp16 -o 512x1024@fp16 fma` |
| `swiglu` | `F.silu(a) * b` | `spyre launch -i 128x4096@fp16 -i 128x4096@fp16 -o 128x4096@fp16 swiglu` |
| `add_fp32` | `a + b`, fp32 | `spyre launch -i 512x1024@fp32 -i 512x1024@fp32 -o 512x1024@fp32 add_fp32` |
| `mul_bf16` | `a * b`, bf16 | `spyre launch -i 512x1024@bf16 -i 512x1024@bf16 -o 512x1024@bf16 mul_bf16` |
| `add_3d` | `a + b`, 3-D | `spyre launch -i 4x256x512@fp16 -i 4x256x512@fp16 -o 4x256x512@fp16 add_3d` |
| `softmax` | `torch.softmax(a, dim=-1)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 softmax` |
| `mean_keepdim` | `a.mean(dim=-1, keepdim=True)` | `spyre launch -i 512x1024@fp16 -o 512x1@fp16 mean_keepdim` |
| `mm` | `a @ b` | `spyre launch -i 512x1024@fp16 -i 1024x256@fp16 -o 512x256@fp16 mm` |
| `bmm` | `torch.bmm(a, b)` | `spyre launch -i 8x128x256@fp16 -i 8x256x128@fp16 -o 8x128x128@fp16 bmm` |

The CLI fills every input with ones, so e.g. `add` prints a tensor of 2s.

## Caveats

- The kernels are tied to the toolchain that built them (torch-spyre
  `e2028e39`, image `icr.io/ai_sw_accel/2.0/torch-spyre:latest` as of
  2026-09-30). If the runtime stops loading them, regenerate them on a Spyre
  machine by compiling the same op with `torch.compile` and copying the one
  kernel directory it writes under `$TORCHINDUCTOR_CACHE_DIR/inductor-spyre/`.
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
