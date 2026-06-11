# KTIR (Kernel Tile IR)

KTIR is an MLIR dialect for tiled, multi-core accelerator kernels. It
extends torch-spyre's existing SuperDSC IR into a community
specification for any dataflow accelerator with scratchpad memory and
compile-time core partitioning. The dialect is the planned successor
to SuperDSC in the torch-spyre compilation pipeline.

:::{admonition} Status
:class: note

The specification is published as
[RFC 0682](https://github.com/torch-spyre/rfcs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md)
(merged March 2026). Two open-source companion projects implement the
dialect today, both Apache-2.0:

- [torch-spyre/ktir-cpu](https://github.com/torch-spyre/ktir-cpu) — CPU
  interpreter and validator. The README describes it as an experimental
  research prototype that implements a subset of RFC 0682.
- [torch-spyre/ktir-mlir-frontend](https://github.com/torch-spyre/ktir-mlir-frontend) —
  MLIR parser and Python bindings (`mlir_ktdp`).

The torch-spyre production path still goes through SuperDSC. KTIR
adoption is incremental: the spec is stable, the reference interpreter
is up, and the backend lowering path is in development.
:::

## Role in the compilation pipeline

KTIR sits between the torch-spyre Inductor front-end and the backend
compiler:

```text
PyTorch model
    │
    ▼  torch.compile, Spyre Inductor backend
    │
FX graph → ATen IR → LoopLevel IR
    │
    ▼  emit KTIR-shaped kernels
    │
KTIR  (this dialect)
    │
    ▼  backend lowering
    │
hardware binaries
```

For the current production path see [Inductor Frontend](inductor_frontend.md)
(emits SuperDSC) and [Backend Compiler](backend.md) (consumes
SuperDSC).

## Three-step memory access pattern

The defining design choice in KTIR is decoupling memory access into
three explicit steps. Each step is a separate op in the KTDP dialect:

| Step | Op | What it does |
|---|---|---|
| 1. Describe layout | `ktdp.construct_memory_view` | Names a memory region with sizes, strides, coordinate set, memory space |
| 2. Address a tile | `ktdp.construct_access_tile` | Selects which slice of the view this core touches |
| 3. Move data | `ktdp.load`, `ktdp.store` | Transfers between the tile and a tensor SSA value |

The separation lets the compiler reason about memory layout, work
division, and data movement independently. Spyre's hardware exposes
HBM and per-core LX scratchpad as distinct memory spaces, so each
`construct_memory_view` carries an explicit
`#ktdp.spyre_memory_space<HBM>` or `<LX>` attribute. A
`construct_distributed_memory_view` variant covers the case where a
tensor is split across many per-core scratchpad slices instead of
sitting in a single HBM region.

## Worked example: 1D element-wise add

A 1024-element vector add over 32 cores looks like this in KTDP. Each
core picks up a 32-element slice based on its grid coordinate:

```
func.func @add(%A: index, %B: index, %Out: index)
    attributes {grid = [32]} {
  %c32 = arith.constant 32 : index
  %id  = ktdp.get_compute_tile_id : index
  %off = arith.muli %id, %c32 : index

  %A_view = ktdp.construct_memory_view %A, sizes:[1024], strides:[1] {
    coordinate_set = affine_set<(d0): (0 <= d0, d0 <= 1023)>,
    memory_space   = #ktdp.spyre_memory_space<HBM>
  } : memref<1024xf16>

  %A_tile = ktdp.construct_access_tile %A_view[%off] {
    access_tile_set   = affine_set<(d0): (0 <= d0, d0 <= 31)>,
    access_tile_order = affine_map<(d0) -> (d0)>
  } -> !ktdp.access_tile<32xindex>

  %a = ktdp.load %A_tile : !ktdp.access_tile<32xindex> -> tensor<32xf16>
  // ... construct B_view and B_tile, then:
  // %s = arith.addf %a, %b : tensor<32xf16>
  // ktdp.store %s, %Out_tile : tensor<32xf16>, !ktdp.access_tile<32xindex>
  return
}
```

The 32 cores execute the same function body in parallel.
`get_compute_tile_id` returns each core's grid coordinate, and
`construct_access_tile` uses that coordinate to select the per-core
slice of the view. Partitioning is fixed at compile time. There is no
runtime block dispatcher.

## ktir-cpu reference interpreter

[ktir-cpu](https://github.com/torch-spyre/ktir-cpu) parses KTDP MLIR,
executes it with NumPy on a simulated multi-core grid, and produces
correctness output plus optional roofline latency estimates. Two
parser frontends are available:

- **Regex parser** for rapid iteration without LLVM dependencies.
- **MLIR frontend** through `mlir_ktdp` (from
  [ktir-mlir-frontend](https://github.com/torch-spyre/ktir-mlir-frontend))
  for strict LLVM 22 conformance.

Both feed the same interpreter, so a kernel that runs through one
runs through the other.

The interpreter targets RFC 0682 but does not yet implement every
KTDP op. Conformance gaps are tracked as `xfail(strict=True)` tests
under `tests/test_spec_gaps.py`. An unexpected pass on one of those
tests signals that a gap has been closed and the marker should be
promoted to a regular test. The full gap analysis is at
`docs/gap_analysis.md` in the ktir-cpu repository.

ktir-cpu also supports AI-driven compiler development: a frontend
pass can emit candidate kernels, run them through the interpreter,
and use correctness output and the latency report to score them.
Determinism and CPU-only execution make this feedback loop practical.

## Why an MLIR dialect

The constraints that shape KTIR's design, drawn from RFC 0682:

- **Tiled, persistent cores.** Spyre cores are persistent and
  partitioned at compile time. The dialect models this with a fixed
  `grid` attribute and a per-core access tile. There is no GPU-style
  thread-block dispatch.
- **Explicit scratchpad.** Per-core LX is small, and the compiler
  manages its allocation (there is no hardware cache). KTIR describes
  staged transfers explicitly through the three-step access pattern
  instead of relying on an implicit cache hierarchy.
- **Cross-stack reuse.** MLIR provides existing dialects (`arith`,
  `math`, `linalg`, `scf`) for the inner kernel body. KTDP only adds
  the Spyre-specific access primitives.
- **Multiple frontends.** A community spec lets multiple compilers
  target the same dialect. The torch-spyre Inductor backend is the
  primary consumer today.

## See also

- [KTIR Specification (RFC 0682)](https://github.com/torch-spyre/rfcs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md)
- [torch-spyre/ktir-cpu](https://github.com/torch-spyre/ktir-cpu) — CPU interpreter and validator
- [torch-spyre/ktir-mlir-frontend](https://github.com/torch-spyre/ktir-mlir-frontend) — MLIR parser and Python bindings
- [Inductor Frontend](inductor_frontend.md) — current source of compiled kernels
- [Backend Compiler](backend.md) — current target of compiled kernels
