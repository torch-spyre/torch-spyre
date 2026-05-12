# Back-End Compiler (DeepTools)

The back-end compiler is a proprietary component called **DeepTools**,
developed by IBM. It takes the SuperDSC JSON specifications produced by
the Torch-Spyre front-end and generates optimized Spyre program binaries.

## Responsibilities

The back-end compiler is responsible for:

- **Dataflow mapping** — mapping SuperDSC operations to optimized Spyre
  dataflows and execution patterns
- **Core scheduling** — determining the precise execution order and
  timing of operations across cores
- **Binary generation** — producing the executable program binaries
  loaded onto the Spyre device at runtime

## Front-End Artifacts

For every `torch.compile`d function, the front-end emits two kinds of
artifacts:

| Artifact | Consumer | Purpose |
|----------|----------|---------|
| **SuperDSC JSON** | DeepTools | Per-kernel operation specification: tensor layouts, work division, OpFunc selection. DeepTools turns each SuperDSC into a device binary that includes both compute and the load/store sequence that stages tiles from LPDDR5 into the LX scratchpad. |
| **DCI (Data Conversion Information)** | Runtime (`copyAsync`) | The `DataConversionInfo` struct, built by `generate_dci()` in `spyre_mem.cpp` from a tensor's `SpyreTensorLayout`. It carries loop ranges, host and device strides, and dtype info, and drives the host ↔ LPDDR5 DMA transfer for each graph input and output. |

## SuperDSC Format

SuperDSC (Super Design Space Config) is a JSON-based intermediate representation that describes the full tile-level compute graph for all 32 Spyre cores. Each artifact is self-contained: it carries everything the hardware needs to execute one scheduled operation deterministically across every core.

### Top-level structure

| Field | Purpose |
|---|---|
| `coreFoldProp_` | How the iteration space is divided across cores (for example `{"factor_": 2}` for a 2-core split). |
| `numWkSlicesPerDim_` | Number of work slices per iteration dimension. `{"c0": 2, "c1": 1}` says dim `c0` is split two ways and dim `c1` is not split. |
| `coreIdToWkSlice_` | Maps each core ID to the slice indices it owns. |
| `dscs_` | Array of `DesignSpaceConfig` entries, one per compute configuration. |

Each `dscs_` entry is a complete description of one compute configuration:

| Field | Purpose |
|---|---|
| `N_` | Full iteration-space extents. `{"c0_": 4, "c1_": 64}` for a 4×64 op. |
| `dataStageParam_` | Per-core dimension sizes for the steady-state (`ss_`) and epilogue (`el_`) passes. Tells the runtime how to partition data for transfer into scratchpad. |
| `primaryDsInfo_` | Tiling information per logical role (`INPUT`, `KERNEL`, `OUTPUT`, `KERNEL_IDX`): `layoutDimOrder_`, `stickDimOrder_`, `stickSize_`. |
| `labeledDs_` | Tensor descriptors. Each entry pairs a tensor argument with its `dsType_` (tiling layout class), `dataFormat_` (for example `SEN169_FP16`), and `memOrg_` (HBM or LX residency). The `layoutDimOrder_` of each entry is independent: two arguments of the same op can pick different dim orders. |
| `scheduleTree_` | Allocate nodes, one per tensor, with memory placement (HBM or LX scratchpad), dimension ordering, per-core start addresses via fold mappings, and coordinate information. |
| `computeOp_` | One entry per operation, encoding the execution unit (`PT` or `SFP`), op name, data format, fidelity, and input/output tensor references. |

### Folding and affine transforms

SuperDSC stays compact through *folding*. A single parameterized artifact can describe behavior across cores, corelets, rows, and time steps without repeating itself. Fold properties use affine transforms of the form `alpha * index + beta` to compute per-core coordinates and addresses:

```json
{"Affine": {"alpha_": 64, "beta_": 0}}
```

The result is that one JSON file describes the behavior of all 32 cores.

:::{note}
The `hbm` field name appearing throughout the SuperDSC IR is a legacy label that refers to device memory in general. Spyre's actual device memory is LPDDR5, not HBM.
:::

### Codegen pipeline (front-end to SuperDSC)

Three components in the front-end collaborate to produce a SuperDSC artifact for each scheduled node:

1. **`SpyreKernel`** ([`spyre_kernel.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/spyre_kernel.py)) collects the iteration space from the scheduler and builds an RValue AST that represents the computation. Node types include `TensorAccess`, `PointwiseOp`, `ReductionOp`, and `Constant`. Leaves are tensor reads or constants; internal nodes are operations.
2. **`OpSpec`** ([`op_spec.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/op_spec.py)) wraps the kernel's output in a structured descriptor: the operation name, the iteration space encoded as [SymPy](https://www.sympy.org/) symbolic expressions, tensor arguments annotated with device coordinates (tile index and intra-stick offset), plus any auxiliary information.
3. **`generate_sdsc()`** ([`codegen/compute_ops.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/codegen/compute_ops.py)) takes the `OpSpec` and emits the final JSON IR. Symbolic expressions are resolved to concrete loop bounds, tiling parameters are expanded, and the `scheduleTree_` is assembled. The output is written as JSON (for example `sdsc_0.json`), which DeepTools then consumes to produce the device binary.

### Example: an `add` OpSpec

The `device_coordinates` on each `TensorArg` are SymPy expressions over
the iteration variables, not plain integer offsets. Here is the
artifact for an `add` between two tensors that share an iteration space
with three loop variables: `c0` of extent 10 with unit stride, `z0` of
extent 50 walking the iteration space at stride 25 (the second value in
each `iteration_space` entry — for example `(sympify('50'), 25)`), and
`c1` of extent 200 with unit stride:

```python
OpSpec(
    op='add',
    is_reduction=False,
    iteration_space={sympify('c0'): (sympify('10'), 1),
                     sympify('z0'): (sympify('50'), 25),
                     sympify('c1'): (sympify('200'), 1)},
    op_info={},
    args=[
        TensorArg(
            is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
            device_size=[10, 4, 50, 64],
            device_coordinates=[sympify('c0'), sympify('floor(c1/64)'),
                                sympify('z0'), sympify('Mod(c1, 64)')],
            allocation={},
        ),
        TensorArg(
            is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
            device_size=[4, 50, 10, 64],
            device_coordinates=[sympify('floor(c1/64)'), sympify('z0'),
                                sympify('c0'), sympify('Mod(c1, 64)')],
            allocation={},
        ),
        TensorArg(
            is_input=False, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
            device_size=[4, 50, 10, 64],
            device_coordinates=[sympify('floor(c1/64)'), sympify('z0'),
                                sympify('c0'), sympify('Mod(c1, 64)')],
            allocation={},
        ),
    ],
)
```

A few things are worth pulling out from this. First, the same iteration
variables (`c0`, `z0`, `c1`) thread through every argument, but each
argument resolves them differently because each tensor sits in a
different device shape. Second, the stick dimension shows up as a pair
of expressions (`floor(c1/64)` and `Mod(c1, 64)`), one for the tile
index and one for the intra-stick offset. Third, the two input tensors
end up with different dim orders (`[c0, floor(c1/64), z0, Mod(c1, 64)]`
vs `[floor(c1/64), z0, c0, Mod(c1, 64)]`), and that is fine: per-argument
`layoutDimOrder_` in the SuperDSC `labeledDs_` is independent.

### Why JSON

SuperDSC artifacts have to be diffable and inspectable during development, which is why JSON is the wire format. When an op gives wrong results on a particular core layout, opening the artifact in a text editor and reading that core's address mapping is usually the fastest path to a diagnosis. JSON also slots cleanly into `torch.compile`'s artifact cache.

### From SuperDSC to KTIR

SuperDSC was designed to get Torch-Spyre running quickly with an IR that closely matches the hardware model. The team is now transitioning to KernelTile IR (KTIR), an MLIR-based representation that generalizes the concepts SuperDSC introduced (compute tiles, scratchpad staging, compile-time core partitioning) into a community specification aimed at any dataflow accelerator. See [RFC 0682 - KTIR Spec](https://github.com/torch-spyre/rfcs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md).

## Invocation

The front-end compiler invokes DeepTools programmatically as part of
the `torch.compile` pipeline. The binary artifacts are cached by
Inductor's standard compilation cache.

## Further Reading

- [Inductor Front-End](inductor_frontend.md) — how the front-end
  generates SuperDSC
- [Dataflow Architecture](../architecture/dataflow_architecture.md) — the
  hardware model that DeepTools targets
