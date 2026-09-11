---
name: debug-graph-validation
description: "Debugging GraphValidationError exceptions from the graph invariant validator. Covers how to read the error, identify the responsible pass, and common root causes per INV code."
---

# Debugging GraphValidationError

When `config.validate_graph_invariants` is enabled (default), the graph
validator runs after every compiler pass in `CustomPreSchedulingPasses`.
A `GraphValidationError` means a pass left the `GraphLowering` IR in an
inconsistent state.

## Reading the Error

Every `GraphValidationError` has three parts:

```
[after split_multi_ops] GraphLowering validation failed: INV-3: ... . detail
       ^                                                  ^            ^
       pass that just ran                                 invariant    what went wrong
```

- **`pass_name`**: the pass that just finished — the invariant held
  before it ran and broke after, so start investigating there.
- **`invariant`**: the INV code (INV-1 through INV-8).
- **`detail`**: the specific buffer, operation, or name involved.

For multiple violations, the error collects them all:

```
3 invariant violations detected after split_multi_ops:
  1. INV-3: ...
  2. INV-5: ...
```

Access individual violations programmatically via `err.violations`.

## Controlling Validation

```bash
# Disable validation entirely
SPYRE_VALIDATE_GRAPH=0 python3 my_script.py

# Enable (default)
SPYRE_VALIDATE_GRAPH=1 python3 my_script.py
```

Or in code:

```python
from torch_spyre._inductor import config
config.validate_graph_invariants = False
```

## Common Root Causes by INV Code

### INV-1: Duplicate buffer names

**Symptom:** "Buffer name 'bufN' appears at indices X and Y"

**Cause:** A pass appended a buffer to `graph.buffers` without using
`graph.register_buffer()`, or manually set `buf.name` to collision.

**Fix:** Use `graph.register_buffer(buf, set_name=True)` which assigns
a unique name automatically.

### INV-2: Buffers removed from list

**Symptom:** "graph.buffers had N entries before pass but now has M"

**Cause:** A pass popped, deleted, or otherwise shortened
`graph.buffers`.

**Fix:** Never remove from `graph.buffers`. To mark a buffer dead, add
its name to `graph.removed_buffers` and pop it from
`graph.name_to_buffer`.

### INV-3: name_to_buffer inconsistency

**Symptom:** "name_to_buffer key does not match buffer name" or
"name_to_buffer missing entry for live buffer"

**Cause (key mismatch):** `name_to_buffer[X]` holds a buffer whose
`get_name()` returns Y != X. Usually a buffer object was placed under
the wrong key.

**Cause (missing entry):** A buffer exists in `graph.buffers` (not
removed) but has no entry in `name_to_buffer`. The buffer was appended
to the list without a corresponding `name_to_buffer` entry.

**Fix:** Use `graph.register_buffer()` or, when replacing a buffer body,
use `replace_computed_buffer_body()` from `pass_utils.py`.

### INV-4: name_to_op inconsistency

**Symptom:** "name_to_op missing entry", "operation_name=None", or
"name_to_op points to wrong object"

**Cause (missing):** An operation was added to `graph.operations` via
`register_operation()` but its name is missing from `name_to_op`.

**Cause (wrong object):** A pass set `buf.operation_name = X` without
updating `graph.name_to_op[X] = buf`, so the dict points to a stale
object.

**Fix:** After setting `operation_name` manually, always do:
`V.graph.name_to_op[new_name] = buf`.

### INV-5: Read from undefined buffer

**Symptom:** "reads buffer 'X' which is not in name_to_buffer,
graph_inputs, or constants"

**Cause:** An operation's `get_read_writes()` returns a `MemoryDep` for
a name that does not appear in `name_to_buffer`, `graph_inputs`, or
`constants`. The producer buffer may have been removed without updating
the consumer.

**Fix:** Ensure all producer buffers are registered before the consumer
is validated, or update the consumer's reads when a producer is removed.

### INV-6: Invalid graph output

**Symptom:** "graph_outputs[N] has name 'X' which is not in ..."

**Cause:** A graph output references a buffer that was removed or never
registered.

**Fix:** Update `graph.graph_outputs` when removing or replacing output
buffers.

### INV-7: Orphaned name_to_users entry

**Symptom:** "name_to_users contains 'X' which is not in ..."

**Cause:** `name_to_users` has a key for a buffer name that is not in
`name_to_buffer`, `graph_inputs`, `constants`, or `removed_buffers`.

**Fix:** Clean up `name_to_users` when removing a buffer entirely (not
just marking it removed).

### INV-8: Read from removed buffer

**Symptom:** "reads buffer 'X' which is in removed_buffers"

**Cause:** A live operation still reads from a buffer that was marked
removed. The dead-code pass should have removed the consumer too, or the
consumer's reads should have been updated.

**Fix:** Ensure that when a buffer is removed, all consumers are either
also removed or updated to read from a replacement.

## Debugging Workflow

1. **Read the pass name** from the error to identify which pass broke
   the invariant.
2. **Run with logging** to see the IR before and after the pass:
   ```bash
   SPYRE_LOG_PASSES=the_pass_name python3 my_script.py
   ```
3. **Check the specific buffer/operation** named in the error detail.
4. **Look at the pass code** to find where it modifies
   `graph.buffers`, `name_to_buffer`, `name_to_op`, or
   `removed_buffers` and check for missing bookkeeping.
5. **Use `config.patch`** to toggle validation around a specific pass
   in a test:
   ```python
   from torch_spyre._inductor import config
   with config.patch({"validate_graph_invariants": True}):
       torch.compile(model, backend="spyre")(input)
   ```

## Reference

- Invariant specification: `docs/source/compiler/graph_invariants.md`
- Validator source: `torch_spyre/_inductor/graph_validation.py`
- Call site: `torch_spyre/_inductor/passes.py` (in
  `CustomPreSchedulingPasses.__call__`)
- Unit tests: `tests/inductor/test_graph_validation.py`
