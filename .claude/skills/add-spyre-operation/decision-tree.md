# Decision Tree: Choosing the Right Pattern

Use this flowchart to decide which pattern to follow when adding a new
operation to the Spyre backend.

```
Is there an existing ATen op for this?
│
├─ YES
│   │
│   ├─ Can it map directly to a single Spyre OpFunc?
│   │   │
│   │   ├─ YES → Pattern 1: Direct ATen → OpFunc Mapping
│   │   │         File: spyre_kernel.py (add @staticmethod to SpyreOpFuncs)
│   │   │
│   │   └─ NO
│   │       │
│   │       └─ Can it be rewritten using ops Spyre already supports?
│   │           │
│   │           ├─ YES → Pattern 2: Spyre-Specific Decomposition
│   │           │         File: decompositions.py (@register_spyre_decomposition)
│   │           │
│   │           └─ NO → Pattern 3: Custom Op + Lowering
│   │                     Files: customops.py + lowering.py + spyre_kernel.py
│   │
│   └─ Does Inductor have a default decomposition that breaks it?
│       │
│       ├─ YES, and the decomposition is fine → Do nothing (it already works)
│       │
│       └─ YES, but you want the direct mapping instead
│             → Pattern 1 overrides the default decomposition
│
└─ NO (Spyre-specific fused/specialized op)
    │
    └─ Pattern 3: Custom Op + Lowering
        │
        ├─ If it can be decomposed into existing ops before lowering:
        │   → Define custom op in customops.py
        │   → Register decomposition in decompositions.py
        │   (Example: spyre.compact)
        │
        └─ If it needs its own lowering and OpFunc:
            → Define custom op in customops.py
            → Register lowering in lowering.py
            → Add OpFunc in spyre_kernel.py
            (Example: spyre.clamp, spyre.gelu)
```

## Additional Considerations

- **SuperDSC codegen needed?** If the op introduces a new operation type at
  the SuperDSC level, also modify `codegen/compute_ops.py` or
  `codegen/data_ops.py`.
- **Multi-file ops?** Some ops (like `layer_norm`) use a decomposition that
  introduces custom ops, which then each need their own lowerings. Follow
  the chain: decomposition → custom ops → lowerings → OpFuncs.
- **Fallback?** If the op cannot run on Spyre in some configurations,
  register a fallback in `fallbacks.py`.
