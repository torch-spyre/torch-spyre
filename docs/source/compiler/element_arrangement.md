Element arrangement
===

In the Spyre device, tensors are packed into 128 Byte sticks that hold 64
elements at 16-bit precision or 32 elements at 32-bit (FP32) precision.

When a tensor precision is widened from 16-bit (`DL16`/`BF16`) to 32-bit
(`FP32`) on the device, the elements cannot remain in both their original
physical location and logical order. The reverse type conversion is also same.
To avoid redistributing data across multiple sticks, the conversion leaves the
elements **staggered** within the sticks.

But due to requirements to support new models there is a need to restore
or apply different element arrangements on Spyre. This document describes the
element arrangement and its awareness through the model compilation graph.

Restore EA
---

Restore STANDARD element arrangement in a D2H or H2D copy of a tensor whose
SpyreTensorLayout has a staggered element arrangement
(`DL16_TO_FP32` or `FP32_TO_DL16`).

The restore does not move any data on the device. It rewrites the data
conversion stride info (DCSI) of the copy so that the existing copy engine
reads every staggered element from its source location and writes it to its
standard destination column.

* **D2H:** The device holds staggered data; the DCSI is rewritten so the host
  receives a STANDARD element arrangement.
* **H2D:** The host holds STANDARD data; the DCSI is rewritten so the device
  receives the staggered element arrangement.

### How the DCSIs are rewritten

A DCSI describes one copy as a loop nest. Every loop has a size, a source
stride and a destination stride, listed innermost first. In a D2H copy the
source is the device and the destination is the host.

`get_device_stride_infos` first builds the DCSIs as if the layout were in
STANDARD arrangement. For a staggered layout those DCSIs would copy the
elements to the wrong host columns. When the copy is D2H and the layout is
`DL16_TO_FP32` or `FP32_TO_DL16`, `get_device_stride_infos` then calls
`ea_restore`, which replaces the DCSIs with equivalent ones that undo the
stagger. `generate_dci` needs no special handling: `get_device_stride_infos`
already returns restored DCSIs for these layouts.

```text
generate_dci
  -> get_device_stride_infos(..., host2device = false)
       1. build standard DCSIs (whole sticks + tail DCSI for a partial stick)
       2. record how the stick dim is tiled (see below)
       3. ea_restore(dcsis, stl, stick facts)
            -> ea_restore_fp16tofp32   (DL16_TO_FP32)
            -> ea_restore_fp32tofp16   (FP32_TO_DL16)
                 -> replace_dim        (split one DCSI dim into new loops)
```

`replace_dim(dcsi, dim, loops)` returns a copy of a DCSI in which dimension
`dim` is replaced by a list of loops, each given as
`{size, device stride, host stride}` (innermost first). All other dimensions
are kept, so batch and row loops are untouched.

#### Stick tiling facts

The restore does not guess the stick loop from the DCSI strides. The tile map
that `get_device_stride_infos` already computes says how each host dim is
split over the device dims, and the following facts are taken from it as local
variables and passed to `ea_restore` as plain arguments:

| Fact | Meaning |
| :--- | :--- |
| `stick_host_extent` | Host extent (in elements) of the stick dim, after folding leading host dims that are not in the tile map. |
| `stick_count_dim` | Index in the DCSI of the device dim that counts sticks (innermost first), or -1 if there is none. Device dims of size 1 count no sticks and are ignored. |
| `allocated_sticks` | Number of sticks allocated along the stick dim (`device_size` of the stick-count dim, 1 if none). |
| `is_stick_tail` | For each DCSI, whether it is the remainder DCSI for the partial last stick of the stick dim. |

The stick dim is the host dim that is tiled into the innermost device dim (the
elements of a stick). If more than one non-trivial device dim counts its
sticks, the stick dim is ambiguous and `DL16_TO_FP32` raises
`Unable to determine the stick dimension for DL16_TO_FP32 D2H`.

Because the stick-count dim comes from the tile map, it can be at any position
of the device layout. For example a `(2, 4, 64)` fp16 tensor converted to FP32
has device size `[8, 1, 2, 32]`, where the stick-count dim is the third device
dim and the rows are folded in front of it.

#### Partial last stick

For a stick dim that is not a multiple of the stick size,
`get_device_stride_infos` produces the whole sticks in one DCSI and the
remainder in a tail DCSI, which starts at the next stick. The tail DCSI is cut
at one stick of the **device** dtype (32 FP32 elements or 64 16-bit elements).
`is_stick_tail` flags it; each element arrangement handles it differently (see
below).

### Restoring EA DL16_TO_FP32 to STANDARD

When a 16-bit stick containing 16 element groups ($G_0$ to $G_{15}$) is upcast
to FP32, the groups spill into two FP32 sticks. The distribution is
alternating:

* **Even groups** ($G_0, G_2, \dots$) are stored in the first stick (`stick0`).
* **Odd groups** ($G_1, G_3, \dots$) are stored in the second stick (`stick1`).

* **Element Group:** A set of 4 consecutive host elements. These groups always
  stay together during conversion.
* **Stick Pair:** Two adjacent device sticks that collectively hold the data
  for 64 host columns (2 sticks $\times$ 32 elements).
* **Unit:** 8 host columns, made of one group from each stick of a pair. A
  stick pair holds 8 units.

#### Visual Representation

Consider one host row spanning one stick pair (64 columns):

**Host Logical Order:**

```text
Columns:  [ 0..3 ] [ 4..7 ] [ 8..11 ] [ 12..15 ] ... [ 60..63 ]
Groups:   [  G0  ] [  G1  ] [  G2   ] [  G3   ] ... [  G15  ]
```

**Device Physical Storage (Same Stick Pair):**

*Stick 0 (Even Groups)*

```text
Pos:      [ 0..3 ] [ 4..7 ] [ 8..11 ] [ 12..15 ] ... [ 28..31 ]
Groups:   [  G0  ] [  G2  ] [  G4   ] [  G6   ] ... [  G14  ]
```

*Stick 1 (Odd Groups)*

```text
Pos:      [ 0..3 ] [ 4..7 ] [ 8..11 ] [ 12..15 ] ... [ 28..31 ]
Groups:   [  G1  ] [  G3  ] [  G5   ] [  G7   ] ... [  G15  ]
```

To reconstruct the host row from the device, one must read $G_0$ from `stick0`
pos 0..3, $G_1$ from `stick1` pos 0..3, $G_2$ from `stick0` pos 4..7, and so
on.

#### Index Mapping

To restore the standard arrangement, we map every host column to its specific
location on the device. A host column index can be decomposed into four
components:

```
host_col = (stick_pair * 64) + (group_in_stick * 8) + (stick_in_pair * 4) + elem_in_group
```

| Component | Range | Description |
| :--- | :--- | :--- |
| `stick_pair` | $[0, N/2)$ | Identifies the pair of sticks. |
| `group_in_stick` | $[0, 8)$ | Identifies the group within a single stick (the unit within the pair). |
| `stick_in_pair` | $[0, 2)$ | Identifies which of the two sticks in the pair. |
| `elem_in_group` | $[0, 4)$ | Identifies the specific element within the group. |

The corresponding device location is calculated as:

* **Device Stick:** `2 * stick_pair + stick_in_pair`
* **Device Position:** `group_in_stick * 4 + elem_in_group`

#### Solution: split the DCSI loops pair by pair

The standard whole-sticks DCSI (`ea_restore_fp16tofp32`) is rewritten by
splitting two of its dimensions into smaller loops with different source
(device) and destination (host) strides.

Loop Splitting Table:

| Original Dimension | New Loop | Size | Src Step (Device) | Dst Step (Host) | Logic |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Dim 0**<br>(32 elements) | `elem_in_group` | 4 | 1 elem | 1 col | Accesses individual elements. |
| | `group_in_stick` | 8 | 4 elems | 8 cols | Skips the 4 columns belonging to the other stick. |
| **Stick-count dim**<br>(N sticks) | `stick_in_pair` | 2 | 1 stick | 4 cols | Switches between the two sticks in a pair. |
| | `stick_pair` | N/2 | 2 sticks | 64 cols | Moves to the next pair of sticks. |

How it works:

1. **`group_in_stick` Loop:** Advancing this loop moves the device pointer by
   4 elements (the size of one group) but jumps 8 columns on the host. This
   jump accounts for the 4 columns of the current group plus the 4 columns of
   the interleaved group from the other stick.
2. **`stick_in_pair` Loop:** Advancing this loop moves the device pointer by
   an entire stick (32 elements per row) but only advances 4 columns on the
   host. This allows the copy engine to pick up the interleaved groups from
   the second stick.
3. **`stick_pair` Loop:** Advances both pointers by the full width of a stick
   pair (64 columns / 2 sticks).

All other DCSI dimensions (rows, batch) are copied unchanged. The device
strides come from the DCSI itself, so a device layout that puts rows between
the sticks (for example `[2, 4, 32]`) works without special cases.

#### Stick dims that are not a whole number of pairs

The tail DCSI that `get_device_stride_infos` produces for a partial last stick
is cut at 32 columns, but the unit of this layout is the stick **pair**
(64 columns), so the tail DCSI does not describe a region of the staggered
layout and cannot be restored on its own. Instead the restore discards it and
rebuilds the whole stick dim from `stick_host_extent`, pair by pair:

| Piece | Host columns | What it restores |
| :--- | :--- | :--- |
| 1. Whole pairs | `0 .. 64 * P - 1` where `P = extent / 64` | Every full pair, with the loops of the table above. |
| 2. Whole units of the last pair | the next `8 * U` columns where `U = (extent % 64) / 8` | The same loops with `group_in_stick = U` and a single pair. These units use both sticks of the last pair. |
| 3. First stick of the last unit | the next `min(r, 4)` columns where `r = extent % 8` | One group (or part of it) from `stick0`. |
| 4. Second stick of the last unit | the next `max(r - 4, 0)` columns | The remaining elements from `stick1`, offset by one stick. |

Empty pieces are skipped. Each piece starts at a host column, from which the
host offset (`column * host element stride`) and the device offset
(`(pair * 2 + stick_in_pair) * stick stride + group_in_stick * 4 * element
stride`) are derived.

Note that the data of a partial last pair is spread over **both** sticks of
the pair, because units alternate between them. For example the 32 columns
after one whole pair occupy the first 16 positions of both sticks of the next
pair.

#### Worked example: `(4, 64)` fp16 to fp32

With an illustrative FP32 device layout of `[2, 4, 32]` (2 sticks, 4 rows,
32 elements), the DCSI for the D2H copy, innermost first, is:

| Loop | Size | Src (device) | Dst (host) |
| :--- | :---: | :---: | :---: |
| elements | 32 | 1 | 1 |
| rows | 4 | 32 | 64 |
| sticks | 2 | 128 | 32 |

Copied as is, device stick 1 would land at host columns 32..63, which is the
standard order but not the staggered one. After the restore:

| Loop | Size | Src (device) | Dst (host) |
| :--- | :---: | :---: | :---: |
| `elem_in_group` | 4 | 1 | 1 |
| `group_in_stick` | 8 | 4 | 8 |
| rows | 4 | 32 | 64 |
| `stick_in_pair` | 2 | 128 | 4 |
| `stick_pair` | 1 | 256 | 64 |

For example `stick_in_pair = 1`, everything else 0, reads device element 128
(stick 1, position 0, which holds $G_1$) and writes host column 4. This is
the mapping of the Index Mapping section.

#### Worked examples: partial pairs, fp16 to fp32

The stick dim is the last dim; `extent` is its size.

| Shape (stick dim) | Pairs needed / sticks allocated | Pieces (elems x units x sticks x pairs at host column) |
| :--- | :---: | :--- |
| `64` | 1 / 2 | `4x8x2x1 @0` |
| `128` | 2 / 4 | `4x8x2x2 @0` |
| `96` (1 pair + 32 columns) | 2 / 4 | `4x8x2x1 @0`, `4x4x2x1 @64` |
| `160` (2 pairs + 32 columns) | 3 / 6 | `4x8x2x2 @0`, `4x4x2x1 @128` |
| `100` (1 pair + 36 columns) | 2 / 4 | `4x8x2x1 @0`, `4x4x2x1 @64`, `4x1x1x1 @96` |
| `68` (1 pair + 4 columns) | 2 / 4 | `4x8x2x1 @0`, `4x1x1x1 @64` |

For `100`, the last piece starts at pair 1, unit 4, `stick0`: it reads device
positions 16..19 of the first stick of the second pair and writes host columns
96..99.

#### Constraints

* **Stick pairs must be allocated:** the stick dim must be allocated as whole
  stick pairs, and the last (possibly partial) pair of the host extent must
  exist on the device. The check compares the sticks needed,
  `ceil(extent / 64) * 2`, with the sticks that are **allocated**, not with the
  sticks the host extent happens to use. A stick dim of 96 uses 3 FP32 sticks
  of data but 4 are allocated, so it is restored.
* **Stick dims narrower than 64:** a 16-bit stick dim narrower than 64 elements
  (for example 16, 32 or 44) is converted to a single FP32 stick, so half of
  each pair has nowhere to go. These tensors are rejected instead of being read
  back wrong:

  ```text
  DL16_TO_FP32 D2H requires whole stick pairs along the stick dimension:
  44 elements need 2 sticks, but 1 are allocated
  ```

* **Stick dimension:** the stick dim must be identified unambiguously from the
  tile map, otherwise `Unable to determine the stick dimension for
  DL16_TO_FP32 D2H` is raised.
* **Stick size:** the elements per stick must be a multiple of 8 (2 sticks
  $\times$ 4 elements per group).
* **Copies start at a pair boundary:** see *Known limitations*.

### Restoring EA FP32_TO_DL16 to STANDARD

When two FP32 sticks (32 elements each) are narrowed to 16-bit, their data is
packed into a single 16-bit stick of 64 elements. The element groups of the
two FP32 sticks are interleaved, which is the inverse of the `DL16_TO_FP32`
distribution:

* Groups of the first FP32 stick (`half0`) go to the **even** group slots.
* Groups of the second FP32 stick (`half1`) go to the **odd** group slots.

Unlike `DL16_TO_FP32`, the staggering stays inside a **single** 16-bit stick,
so there is no stick pairing.

* **Element Group:** A set of 4 consecutive host elements.
* **Half:** 32 consecutive host columns within one 64-column stick, matching
  one source FP32 stick.

#### Visual Representation

Consider one host row spanning one 16-bit stick (64 columns):

**Host Logical Order:**

```text
Columns:  [ 0..3 ] [ 4..7 ] ... [ 28..31 ] [ 32..35 ] ... [ 60..63 ]
Groups:   [  G0  ] [  G1  ] ... [  G7   ] [  G8   ] ... [  G15  ]
          |<------ half 0 ------>|<------- half 1 ------->|
```

**Device Physical Storage (one 16-bit stick):**

```text
Pos:      [ 0..3 ] [ 4..7 ] [ 8..11 ] [ 12..15 ] ... [ 56..59 ] [ 60..63 ]
Groups:   [  G0  ] [  G8  ] [  G1   ] [  G9    ] ... [  G7   ] [  G15  ]
```

To reconstruct the host row, one must read $G_0$ from pos 0..3, $G_1$ from
pos 8..11, ..., $G_7$ from pos 56..59, then $G_8$ from pos 4..7, and so on.

#### Index Mapping

A host column index within a stick is decomposed into three components:

```
host_col = (stick * 64) + (half_in_stick * 32) + (group_in_half * 4) + elem_in_group
```

| Component | Range | Description |
| :--- | :--- | :--- |
| `stick` | $[0, N)$ | Identifies the 16-bit stick. |
| `half_in_stick` | $[0, 2)$ | Identifies which half of the stick's host columns. |
| `group_in_half` | $[0, 8)$ | Identifies the group within a half. |
| `elem_in_group` | $[0, 4)$ | Identifies the specific element within the group. |

The corresponding device location is calculated as:

* **Device Stick:** stick
* **Device Position:** group_in_half *8 + half_in_stick* 4 + elem_in_group

#### Solution: split the DCSI loops

Only dim 0 is split (`ea_restore_fp32tofp16`). All other dimensions, including
the stick dimension, are copied unchanged.

| Original Dimension | New Loop | Size | Src Step (Device) | Dst Step (Host) | Logic |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Dim 0**<br>(64 elements) | `elem_in_group` | 4 | 1 elem | 1 col | Accesses individual elements. |
| | `half_in_stick` | 2 | 4 elems | 32 cols | Switches between the two interleaved halves. |
| | `group_in_half` | 8 | 8 elems | 4 cols | Moves to the next group of the same half. |

How it works:

1. **`half_in_stick` Loop:** Advancing this loop moves the device pointer by
   one group (4 elements) but jumps half a stick (32 columns) on the host.
2. **`group_in_half` Loop:** Advancing this loop skips the group of the other
   half on the device (8 elements) but advances only one group (4 columns) on
   the host.

The device side is walked contiguously (steps 1, 4, 8 elements).

#### Partial sticks

One device stick holds the same 64 host columns as in standard order, so the
tail DCSI that `get_device_stride_infos` produces for a partial last stick
already covers the right region and is restored **on its own**, with no
stick-count or stick-dim information. Its dim 0 holds `c` host columns
(1 to 64) and is split into up to three pieces:

| Piece | Host columns | Loops (elems x halves x groups) |
| :--- | :--- | :--- |
| 1. Whole halves | `0 .. 32 * H - 1` where `H = c / 32` | `4 x H x 8` |
| 2. Whole groups of the next half | the next `4 * G` columns where `G = (c % 32) / 4` | `4 x 1 x G` |
| 3. Remaining elements | the next `c % 4` columns | `c%4 x 1 x 1` |

Each piece starts at a host column; its device offset is
`(half + group * 2) * 4` elements from the start of the stick, where `half`
and `group` are the half and group of that first column.

#### Worked examples, fp32 to fp16

Only the elements loop changes, so the rows are shown as an unchanged loop.

`(4, 64)`, one whole stick per row. Before the restore, innermost first:

| Loop | Size | Src (device) | Dst (host) |
| :--- | :---: | :---: | :---: |
| elements | 64 | 1 | 1 |
| rows | 4 | 64 | 64 |

After the restore:

| Loop | Size | Src (device) | Dst (host) |
| :--- | :---: | :---: | :---: |
| `elem_in_group` | 4 | 1 | 1 |
| `half_in_stick` | 2 | 4 | 32 |
| `group_in_half` | 8 | 8 | 4 |
| rows | 4 | 64 | 64 |

For example `half_in_stick = 1`, everything else 0, reads device position 4
(which holds $G_8$) and writes host column 32.

`(4, 100)`: a whole stick (64 columns) and a tail DCSI of 36 columns that
starts at the second stick.

| DCSI | Columns `c` | Pieces |
| :--- | :---: | :--- |
| whole stick | 64 | `4x2x8 @0` |
| tail | 36 | `4x1x8 @0` (columns 0..31), `4x1x1 @32` (columns 32..35, device position 4, which holds $G_8$) |

`(4, 44)`: a single partial stick (44 columns): `4x1x8 @0` for columns 0..31,
then `4x1x3 @32` for columns 32..43, which reads device positions 4..7, 12..15
and 20..23.

#### Constraints

* **Stick size:** the stick size must be divisible by 8 (2 halves $\times$ 4
  elements per group).
* **Stick extent:** a DCSI must hold between 1 and one stick of elements in
  dim 0, otherwise `FP32_TO_DL16 D2H: unexpected stick extent` is raised.
* No even-stick-count requirement and no stick dimension identification are
  needed, since the stagger never crosses a stick boundary.
* **Compile-time restriction:** the conversion itself halves the stick count,
  so `rescale_stl_for_dtype` only accepts an FP32 layout whose 32-element
  sticks along the stick dim are a whole number of 64-element output sticks. A
  stick dim of 68, 96 or 160 (3, 3 and 5 sticks) or of 32 (1 stick) is
  rejected at compile time with `cannot rescale device layout`, before any D2H
  copy. The D2H restore does not see these tensors.

### Supported inputs

| Conversion | Stick dim | Result |
| :--- | :--- | :--- |
| fp16 to fp32 | multiple of 64 (64, 128, ...) | Restored. |
| | not a multiple of 64 but at least 64 (68, 96, 100, 160, ...) | Restored. The partial last pair is rebuilt from the host extent. |
| | narrower than 64 (16, 32, 44, ...) | Rejected on D2H: `requires whole stick pairs`. |
| fp32 to fp16 | multiple of 64 (64, 128, ...) | Restored. |
| | a whole stick plus a partial stick (100) | Restored. The tail DCSI is restored on its own. |
| | a single partial stick (16, 44) | Restored. |
| | odd number of 32-element FP32 sticks (32, 68, 96, 160) | Rejected at compile time. |

All of the above apply to any rank: the stick dim is the last dim and the other
dims only add outer loops that are copied unchanged.

### Known limitations

* **Slices of a staggered tensor:** the restore assumes that the copy starts at
  the beginning of a stick pair. A D2H copy of a slice that starts on a stick
  but not on a pair boundary (for example `y[:, 32:96]` of a `(4, 128)`
  `DL16_TO_FP32` result) computes its device offset in standard order, so the
  restore treats sticks 1 and 2 as a pair instead of sticks (0, 1) and (2, 3)
  and returns wrong data without an error. The planned fix is to copy the whole
  allocation to the host in `spyre_copy_from` and slice it on the CPU, so the
  restore only sees copies that start at stick 0.
* **Padded stick dims and work division:** for stick dims that are not
  stick-aligned, work division does not take the padding into account yet, so
  with some divisions the conversion itself leaves the device values wrong.
  This is independent of the D2H restore. A `work_div` hint that gives every
  core a single row avoids it.

### Testing

`tests/inductor/test_ea_bidirectional_conversion.py::test_ea_restore_d2h`
converts a tensor of small integers on the device, reads it back with
`.cpu()` and compares it with the CPU result, for every 16-bit type. Inputs
are converted with a `work_div` hint that gives each core one row, so the
device values are correct for padded shapes. The shape lists are:

* **Both conversions:** 1D to 4D shapes with a stick dim of 64 or 128
  (for example `(64,)`, `(4, 64)`, `(7, 128)`, `(8, 16, 128)`,
  `(2, 4, 8, 64)`).
* **fp16 to fp32, partial pairs:** stick dims 68, 96, 100 and 160, in 1D, 2D
  and 3D.
* **fp32 to fp16, partial sticks:** stick dims 100 and 44, and `(4, 16)`.
* **fp16 to fp32, narrower than 64:** `(44,)`, `(4, 16)`, `(4, 32)`, `(4, 44)`,
  `(7, 44)`, `(2, 4, 44)` and `(2, 3, 4, 44)` must raise
  `requires whole stick pairs` (checked with `pytest.raises`). Every other
  shape is compared by value, so wrong data fails the test instead of being
  hidden.

fp32 to fp16 shapes that the compiler rejects (see the compile-time
restriction above) are not part of this test.
