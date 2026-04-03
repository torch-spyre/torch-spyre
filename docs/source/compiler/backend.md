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

## Interface with the Front-End

The front-end compiler communicates with the back-end through two
artifact types:

| Artifact | Description |
|----------|-------------|
| **SuperDSC JSON** | High-level operation specification per kernel, including tensor layouts, work division, and OpFunc selection |
| **DCI (Data Copy Instructions)** | Explicit DMA transfer specifications for staging data between DDR and scratchpad |

## SuperDSC Format

> **TODO:** Document the SuperDSC JSON schema and fields.

## Invocation

The front-end compiler invokes DeepTools programmatically as part of
the `torch.compile` pipeline. The binary artifacts are cached by
Inductor's standard compilation cache.

## Further Reading

- [Inductor Front-End](inductor_frontend.md) — how the front-end
  generates SuperDSC
- [Dataflow Architecture](../architecture/dataflow_architecture.md) — the
  hardware model that DeepTools targets
