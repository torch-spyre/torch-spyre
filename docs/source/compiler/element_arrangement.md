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

EA shuffle
---

**Shuffle:** Apply or restore STANDARD element arrangement in a D2H or H2D copy
of a tensor whose SpyreTensorLayout has a staggered element arrangement
(`DL16_TO_FP32` or `FP32_TO_DL16`).

| Copy | Host | Device | Operation |
| :--- | :--- | :--- | :--- |
| **D2H** | receives STANDARD | holds staggered | **ea_restore**: staggered to STANDARD |
| **H2D** | holds STANDARD | receives staggered | **ea_apply**: STANDARD to staggered |

Neither operation moves data on the device. The copy engine does the shuffle
while it copies, driven by the per-dim element arrangement of the data
conversion info (DCI): `input_dimwise_ea_` for a D2H copy and
`output_dimwise_ea_` for a H2D copy.

> [!NOTE]
>
> Updating DCSI to handle staggered tensor transfers between device and host in
> STANDARD format is complex and out of sync with Spyre. Instead, using
> `dimwise_ea` fields in DCI simplifies the process, allowing the backend
> compiler to handle element shuffling.

### How the DCI is updated

In a D2H copy the source is the device and the destination is the host; in an
H2D copy it is the other way round.

`get_device_stride_infos` builds the DCSIs as if the layout were in STANDARD
arrangement and, for a staggered layout, also returns a `dimwise_ea` list. This
holds for both directions. `generate_dci` stores the list on the device side
of the copy.

```text
generate_dci
  -> get_device_stride_infos(..., host2device, dimwise_ea)
       1. build standard DCSIs
       2. record how the stick dim is tiled (see below)
       3. dimwise_ea[i] = identity_dim_ea(...)   for every host dim
          ea_shuffle(...)                        overwrites the stick dim
            -> ea_shuffle_fp16tofp32   (DL16_TO_FP32)
            -> ea_shuffle_fp32tofp16   (FP32_TO_DL16)
  -> reverse dimwise_ea (the DCI shapes are innermost first)
  -> host2device ? dci.output_dimwise_ea_ : dci.input_dimwise_ea_
```

`ea_shuffle` creates identical entries for D2H and H2D, storing them in
different DCI fields.

| Field | Meaning |
| :--- | :--- |
| `dimShape_` | Extent of the dim's coordinate space, over the allocated (padded) device extent. |
| `cumElemsBefore_` | For each device dim the host dim is tiled into (innermost first), the number of coordinates before it. |
| `cumOffset_` | The device stride of each of those device dims. |
| `subElems_` | Mixed radix split of the standard coordinate `c`, innermost first. |
| `subStride_` | Stride of each piece of that split in the device column space. |

The staggered coordinate is `p(c) = sum(digit_m(c) * subStride_[m])`, where
`digit_m(c)` are the digits of `c` in the radix `subElems_`. `cumElemsBefore_`
and `cumOffset_` then turn `p(c)` into a device offset.

| Direction | Field | Engine computes (`DataConvertOpFuncs::GENERAL_SHUFFLE`) |
| :--- | :--- | :--- |
| D2H (restore) | `input_dimwise_ea_` | `out[layout(c)] = in[layout(p(c))]` |
| H2D (apply) | `output_dimwise_ea_` | `out[layout(p(c))] = in[layout(c)]` |

All other host dims get `identity_dim_ea`: the coordinate is unchanged and only
laid out over its device dims. So batch and row dims need no special handling.

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

#### Padding and partial sticks

The permutation covers the **allocated** extent of the stick dim
(`allocated_sticks * elems_per_stick`), not the host extent. A partial last
stick, or a partial last stick pair, therefore needs no special case: the
columns that exist on the host are mapped by the same `p(c)`, and the padding
columns are never read or written.

### EA DL16_TO_FP32

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

A STANDARD (host) column index decomposes into four components:

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

#### Shuffle parameters

`ea_shuffle_fp16tofp32` writes the mixed radix split of `host_col` above,
innermost first. The device column space has `elems_per_stick` (32) columns
per stick:

| Piece | `subElems_` | `subStride_` | Meaning of the stride |
| :--- | :---: | :---: | :--- |
| `elem_in_group` | 4 | 1 | Next element in the stick. |
| `stick_in_pair` | 2 | 32 | Next stick (one stick of device columns). |
| `group_in_stick` | 8 | 4 | Next group inside a stick. |
| `stick_pair` | N/2 | 64 | Next pair of sticks. |

`cumElemsBefore_ = {1, 32}` and
`cumOffset_ = {device element stride, device stick stride}` turn the device
column into a device offset.

#### Example: `(4, 64)` fp16 to fp32

With an illustrative FP32 device layout of `[2, 4, 32]` (2 sticks, 4 rows,
32 elements; strides `[128, 32, 1]`) the host dims get these entries, listed
innermost first after the reverse in `generate_dci`:

| Host dim | `dimShape_` | `cumElemsBefore_` | `cumOffset_` | `subElems_` | `subStride_` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| columns (stick dim) | 64 | `{1, 32}` | `{1, 128}` | `{4, 2, 8, 1}` | `{1, 32, 4, 64}` |
| rows | 4 | `{1}` | `{32}` | `{4}` | `{1}` |

Example: host column 4 has `stick_in_pair = 1` and every other digit 0, so
`p(4) = 32`. That is device stick 1, position 0, device element 128, which
holds $G_1$.

* D2H (EA Restore): the engine reads device element 128 and writes host
  column 4.
* H2D (EA Apply): the engine reads host column 4 and writes device element 128.

#### Partial pairs

For `extent = 100` (1 pair + 36 columns), 4 FP32 sticks (2 pairs) are
allocated and `dimShape_ = 128`. Host column 96 is in the second pair
(`stick_pair = 1`) with `group_in_stick = 4`, `stick_in_pair = 0`, so
`p(96) = 64 + 16 = 80`: device stick 2, positions 16..19 for columns
96..99. No tail handling is needed.

#### Constraints

* **Stick pairs must be allocated:** the stick dim must be allocated as whole
  stick pairs (`allocated_sticks` is even). A stick dim of 96 uses 3 FP32
  sticks of data but 4 are allocated, so it is supported.
* **Stick dims narrower than 64:** a 16-bit stick dim narrower than 64 elements
  (for example 16, 32 or 44) is converted to a single FP32 stick, so half of
  each pair has nowhere to go. These tensors are rejected instead of being
  read or written wrong:

```text
  DL16_TO_FP32 element arrangement requires whole stick pairs along the stick
  dimension, but 1 sticks are allocated
```

* **Stick dimension:** the stick dim must be identified unambiguously from the
  tile map, otherwise `Unable to determine the stick dimension for
  DL16_TO_FP32` is raised.
* **Stick size:** the elements per stick must be a multiple of 8 (2 sticks * 4
  elements per group).
* **Copies start at a pair boundary:** see *Known limitations*.

### EA FP32_TO_DL16

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

To go between the two, $G_0$ is at pos 0..3, $G_1$ at pos 8..11, ..., $G_7$ at
pos 56..59, then $G_8$ at pos 4..7, and so on.

#### Index Mapping

A standard (host) column index within a stick decomposes into three
components:

```
host_col = (stick * 64) + (half_in_stick * 32) + (group_in_half * 4) + elem_in_group
```

| Component | Range | Description |
| :--- | :--- | :--- |
| `stick` | $[0, N)$ | Identifies the 16-bit stick. |
| `half_in_stick` | $[0, 2)$ | Identifies which half of the stick's host columns. |
| `group_in_half` | $[0, 8)$ | Identifies the group within a half. |
| `elem_in_group` | $[0, 4)$ | Identifies the specific element within the group. |

The corresponding device location is:

* **Device Stick:** `stick`
* **Device Position:** `group_in_half * 8 + half_in_stick * 4 + elem_in_group`
#### Shuffle parameters

`ea_shuffle_fp32tofp16` writes the mixed radix split of `host_col` above.
The pieces must be listed in order of significance in `host_col` (elements,
then groups, then halves, then sticks), because `subElems_` is a mixed radix
split of it:

| Piece | `subElems_` | `subStride_` | Meaning of the stride |
| :--- | :---: | :---: | :--- |
| `elem_in_group` | 4 | 1 | Next element in the stick. |
| `group_in_half` | 8 | 8 | Skips the group of the other half. |
| `half_in_stick` | 2 | 4 | Moves by one group. |
| `stick` | N | 64 | Next 16-bit stick. |

`cumElemsBefore_ = {1, 64}` and `cumOffset_ = {device element stride,
device stick stride}` turn the device column into a device offset. When the
stick dim has no stick-count device dim, the stick stride is 0 and
`allocated_sticks` is 1.

#### Worked example: `(4, 64)` fp32 to fp16

With a device layout of `[4, 64]` (4 rows, 1 stick of 64 elements):

| Host dim | `dimShape_` | `cumElemsBefore_` | `cumOffset_` | `subElems_` | `subStride_` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| columns (stick dim) | 64 | `{1, 64}` | `{1, 0}` | `{4, 8, 2, 1}` | `{1, 8, 4, 64}` |
| rows | 4 | `{1}` | `{64}` | `{4}` | `{1}` |

Example: host column 32 has `half_in_stick = 1` and every other digit 0, so
`p(32) = 4`: device position 4, which holds $G_8$.

* D2H (restore): the engine reads device position 4 and writes host column 32.
* H2D (apply): the engine reads host column 32 and writes device position 4.
#### Partial sticks

The stagger never crosses a stick boundary, so a partial last stick is handled
by the same `p(c)` over the allocated extent. For `(4, 44)` (a single partial
stick, 44 columns) host columns 32..43 are mapped to device positions 4..7,
12..15 and 20..23.

#### Constraints

* **Stick size:** the stick size must be divisible by 8 (2 halves $\times$ 4
  elements per group).
* No even-stick-count requirement is needed, since the stagger never crosses
  a stick boundary.
* **Compile-time restriction:** the conversion itself halves the stick count,
  so `rescale_stl_for_dtype` only accepts an FP32 layout whose 32-element
  sticks along the stick dim are a whole number of 64-element output sticks. A
  stick dim of 68, 96 or 160 (3, 3 and 5 sticks) or of 32 (1 stick) is
  rejected at compile time with `cannot rescale device layout`, before any
  copy. The shuffle does not see these tensors.

### Supported inputs

| Conversion | Stick dim | Result |
| :--- | :--- | :--- |
| fp16 to fp32 | multiple of 64 (64, 128, ...) | Supported. |
| | not a multiple of 64 but at least 64 (68, 96, 100, 160, ...) | Supported. The permutation covers the allocated pairs. |
| | narrower than 64 (16, 32, 44, ...) | Rejected: `requires whole stick pairs`. |
| fp32 to fp16 | multiple of 64 (64, 128, ...) | Supported. |
| | a whole stick plus a partial stick (100) | Supported. |
| | a single partial stick (16, 44) | Supported. |
| | odd number of 32-element FP32 sticks (32, 68, 96, 160) | Rejected at compile time. |

All of the above apply to any rank, and to both directions (restore on D2H,
apply on H2D): the stick dim is the last dim and the other dims get an
identity entry.

### Known limitations

* **Slices of a staggered tensor:** the shuffle assumes that the copy starts at
  the beginning of a stick pair. A copy of a slice that starts on a stick
  but not on a pair boundary (for example `y[:, 32:96]` of a `(4, 128)`
  `DL16_TO_FP32` result) computes its device offset in standard order, so the
  shuffle treats sticks 1 and 2 as a pair instead of sticks (0, 1) and (2, 3)
  and produces wrong data without an error. The planned fix for D2H is to copy
  the whole allocation to the host in `spyre_copy_from` and slice it on the
  CPU, so the shuffle only sees copies that start at stick 0.
* **Padded stick dims and work division:** for stick dims that are not
  stick-aligned, work division does not take the padding into account yet, so
  with some divisions the conversion itself leaves the device values wrong.
  This is independent of the shuffle. A `work_div` hint that gives every
  core a single row avoids it.
