# Layout and stride semantics

For the full tiled-tensor layout specification, see
[`docs/source/user_guide/tensors_and_layouts.md`](../../../../../../docs/source/user_guide/tensors_and_layouts.md).
This file covers the `stride_map` concept specifically, which trips people
up because it's easy to conflate with `device_stride`.

## `stride_map` vs `device_stride` vs `pytorch_stride`

- `pytorch_stride` — the tensor's ordinary PyTorch logical-memory strides
  (row-major unless it's a transposed view).
- `device_size` / `device_stride` — the shape/strides of the *device
  buffer*, one entry per physical device dimension. `device_stride` is
  always row-major-by-construction over `device_size`
  (`device_stride[i] = prod(device_size[i+1:])`) — this holds for **any**
  `device_size` array; it is a definitional consequence of row-major
  layout, not a sortedness property of `device_size`'s raw values. A
  `device_size` like `[256, 1, 1, 8, 64]` is not "corrupted" just because
  the numbers aren't monotonically decreasing.
- `stride_map[j]` — how far stepping +1 along **device dim j** moves you in
  **PyTorch logical memory**. This is a completely different axis from
  `device_stride`, which measures distance in the device buffer. If the
  host tensor's strides aren't row-major (e.g. a transposed view),
  `stride_map` won't be decreasing either — its order is a property of the
  host tensor's strides, independent of the device buffer's own
  (always-row-major) physical layout.

General rule for computing `stride_map[j]` from a device dim's coordinate
expression over pytorch dim `p`:

| Device dim role | Coordinate expr example | `stride_map[j]` |
|---|---|---|
| inner stick | `Mod(p, 64)` | `pytorch_stride[p]` |
| outer stick tile | `floor(p / 64)` | `64 * pytorch_stride[p]` |
| non-stick | `floor(p)` | `pytorch_stride[p]` |

### Worked example: `[128, 256]` row-major, cols = stick (before restickify)

```
pytorch_size:    [128, 256]
pytorch_stride:  [256, 1]
device_size:     [4, 128, 64]
device_stride:   [8192, 64, 1]
stride_map:      [64, 256, 1]
```

| dim | coord expr | size | dev_stride | stride_map |
|---|---|---|---|---|
| 0 | `floor(p1/64)` | 4 | 8192 | 64 |
| 1 | `floor(p0)` | 128 | 64 | 256 |
| 2 | `Mod(p1, 64)` | 64 | 1 | 1 |

The stick is `Mod(p1, 64)` — the pytorch column, since `pytorch_stride[1] = 1`.

### Worked example: same tensor, rows = stick (after restickify)

`pytorch_size`/`pytorch_stride` are unchanged — only the device-side
tiling choice changes:

```
device_size:     [2, 256, 64]
device_stride:   [16384, 64, 1]
stride_map:      [16384, 1, 256]
```

| dim | coord expr | size | dev_stride | stride_map |
|---|---|---|---|---|
| 0 | `floor(p0/64)` | 2 | 16384 | 16384 |
| 1 | `floor(p1)` | 256 | 64 | 1 |
| 2 | `Mod(p0, 64)` | 64 | 1 | 256 |

The stick is now `Mod(p0, 64)` — the pytorch row.

## Gotcha: layouts are per-op, not kernel-wide

Different ops in the same compiled kernel can legitimately commit to
different, mutually inconsistent device layouts for the *same logical
tensor dimension* — e.g. dim H can be device-outermost for one op's input
and device-near-innermost for another op's output/copy in the same fused
kernel. **This is valid, not a bug.** Don't assume a single canonical
layout holds kernel-wide when you're investigating a
`device_tile_advance_expr` (or any device-layout-derived value) mismatch
between two ops — check whether they've actually committed to the same
layout before treating a discrepancy as a bug.

To recover "where is logical dim X in this op's device layout" for a given
generated `OpSpec`, look at `device_coordinates` and find the position
whose coordinate expression equals the bare iteration-space symbol for X
— don't assume a fixed dim ordering across ops.
