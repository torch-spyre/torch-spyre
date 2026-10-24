# GraphLowering Invariants

The `graph_validation` module
(`torch_spyre/_inductor/graph_validation.py`) validates structural
invariants of the `GraphLowering` IR after each compiler pass in the
pre-scheduling pipeline. Validation is gated by the config flag
`validate_graph_invariants` (env var `SPYRE_VALIDATE_GRAPH`, default
enabled).

## Design Contract: `graph.buffers` vs `name_to_buffer`

Understanding the relationship between these two data structures is
essential for writing correct passes:

- **`graph.buffers`** is an append-only list. Buffers are never removed
  from it — even dead buffers remain in the list, with their names added
  to `graph.removed_buffers`.
- **`graph.name_to_buffer`** is the authoritative name → buffer mapping.
  After `replace_computed_buffer_body()`, `name_to_buffer` holds the
  *replacement* buffer object while `graph.buffers` retains the
  *original*. This intentional divergence is why the validator checks
  name consistency (does `get_name()` match the key?) rather than object
  identity (`is`).
- **`graph.removed_buffers`** is the set of buffer names that have been
  logically removed. Passes should also pop removed names from
  `name_to_buffer`, but some passes (`deadcode_elimination`,
  `propagate_layouts`) do not yet do this — hence the validator
  downgrades stale-entry detection to a debug log rather than an error.

## Invariant Reference

### INV-1: Buffer Name Uniqueness

> Buffer names in `graph.buffers` must be unique.

**Fields:** `graph.buffers`

Each buffer's `get_name()` must return a distinct value. Duplicates
indicate a registration bug — likely a buffer appended without going
through `register_buffer()`.

### INV-2: Buffer List Immutability

> Buffers must never be removed from `graph.buffers`.

**Fields:** `graph.buffers`

The buffer list is append-only. Passes that remove a buffer must add
its name to `graph.removed_buffers` rather than deleting from the list.
This invariant compares `len(graph.buffers)` before and after a pass.

### INV-3: `name_to_buffer` Consistency

> `name_to_buffer` must be consistent with `buffers` and
> `removed_buffers`.

**Fields:** `graph.name_to_buffer`, `graph.buffers`,
`graph.removed_buffers`

Two sub-checks:

1. **Forward check:** For every `(name, buf)` in `name_to_buffer`,
   `buf.get_name()` must equal `name`.
2. **Reverse check:** For every buffer in `graph.buffers` whose name is
   not in `removed_buffers`, `name_to_buffer` must contain an entry for
   that name.

Stale entries (a removed name still present in `name_to_buffer`) are
currently logged at debug level, not raised as errors.

### INV-4: `name_to_op` Consistency

> `name_to_op` must be consistent with `operations`.

**Fields:** `graph.name_to_op`, `graph.operations`

For every operation in `graph.operations`:

- `get_operation_name()` must not be `None`.
- The name must exist as a key in `graph.name_to_op`.
- `graph.name_to_op[name]` must be the same object (`is`) as the
  operation in `graph.operations`.

### INV-5: Reads From Defined Buffers

> Operations may only read from defined buffers.

**Fields:** `graph.name_to_buffer`, `graph.graph_inputs`,
`graph.constants`, `graph.torchbind_constants`

For every `MemoryDep` in an operation's read set, the dependency name
must appear in `name_to_buffer`, `graph_inputs`, `constants`, or
`torchbind_constants`. `StarDep` dependencies (ordering constraints)
are ignored.

### INV-6: Graph Outputs Valid

> Graph outputs must reference valid buffers.

**Fields:** `graph.graph_outputs`, `graph.name_to_buffer`,
`graph.graph_inputs`, `graph.constants`

Every entry in `graph_outputs` (except `NoneAsConstantBuffer` and
`ShapeAsConstantBuffer`) must have a name that appears in the defined
names set.

### INV-7: `name_to_users` Consistency

> `name_to_users` keys must reference defined or removed names.

**Fields:** `graph.name_to_users`, `graph.name_to_buffer`,
`graph.graph_inputs`, `graph.constants`, `graph.removed_buffers`

Every key in `name_to_users` must appear in the defined names set or in
`removed_buffers`. An orphaned key indicates a bookkeeping bug in a
pass.

### INV-8: No Reads From Removed Buffers

> No live operation may read from a removed buffer.

**Fields:** `graph.operations`, `graph.removed_buffers`

For every `MemoryDep` in an operation's read set, the dependency name
must not appear in `removed_buffers`. This catches operations that were
not properly updated when their input was removed by dead-code
elimination or another pass.

## Planned Future Work

### INV-9: Synthetic Index Bounds Checking

> Validate that operations do not read past the end of a tensor.

This would perform symbolic range evaluation on each operation's index
expressions against the buffer's declared sizes. It requires
substantially more infrastructure than the current structural checks and
is tracked as a follow-up.

## Cross-Graph Buffer Name Uniqueness

INV-1 currently checks uniqueness within a single `GraphLowering`
instance. Cross-graph uniqueness (across subgraphs in a composed kernel)
is enforced by the `qualify_name()` mechanism, which prefixes buffer
names with the graph's name. A formal cross-graph invariant check is
tracked as a follow-up.

## Common Pass Patterns

### Replacing a buffer body

Use `replace_computed_buffer_body()` from `pass_utils.py`. This updates
`name_to_buffer` to point to the new `ComputedBuffer` while leaving
`graph.buffers` unchanged. The validator's INV-3 forward check tolerates
this because it compares names, not object identity.

### Removing a buffer

Add the buffer's name to `graph.removed_buffers` **and** pop it from
`graph.name_to_buffer`:

```python
graph.removed_buffers.add(buf_name)
graph.name_to_buffer.pop(buf_name, None)
```

Do **not** remove the buffer from `graph.buffers` (INV-2 enforces this).

### Adding an operation

Register the buffer, then register the operation:

```python
name = graph.register_buffer(buf, set_name=True)
graph.register_operation(buf)
```

This ensures both `name_to_buffer` and `name_to_op` are updated.

### Setting `operation_name` manually

If a pass sets `buf.operation_name` directly (rather than going through
`register_operation()`), it **must** also update `name_to_op`:

```python
buf.operation_name = new_name
V.graph.name_to_op[new_name] = buf
```

Failing to do so violates INV-4.

## Error Format

When a single invariant is violated, the error message takes the form:

```
[after pass_name] GraphLowering validation failed: INV-N: description. details
```

When multiple invariants are violated in the same pass, they are
collected into a single exception:

```
[after pass_name] GraphLowering validation failed: multiple violations.
3 invariant violations detected after pass_name:
  1. INV-3: name_to_buffer missing entry for live buffer: Buffer 'buf7' ...
  2. INV-5: operation reads from undefined buffer: Operation 'buf12' ...
  3. INV-6: graph output references undefined buffer: graph_outputs[0] ...
```

The exception's `.violations` attribute provides programmatic access to
the individual `GraphValidationError` instances.
