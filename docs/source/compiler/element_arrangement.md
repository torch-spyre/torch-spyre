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

Restore EA D2H
---

Restore STANDARD element arrangement in a D2H copy of a tensor whose
SpyreTensorLayout has a staggered element arrangement.

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

$$
\text{host\_col} = (\text{stick\_pair} \times 64) + (\text{group\_in\_stick} \times 8) + (\text{stick\_in\_pair} \times 4) + \text{elem\_in\_group}
$$

| Component | Range | Description |
| :--- | :--- | :--- |
| `stick_pair` | $[0, N/2)$ | Identifies the pair of sticks. |
| `group_in_stick` | $[0, 8)$ | Identifies the group within a single stick. |
| `stick_in_pair` | $[0, 2)$ | Identifies which of the two sticks in the pair. |
| `elem_in_group` | $[0, 4)$ | Identifies the specific element within the group. |

The corresponding device location is calculated as:

* **Device Stick:** $2 \times \text{stick\_pair} + \text{stick\_in\_pair}$
* **Device Position:** $\text{group\_in\_stick} \times 4 + \text{elem\_in\_group}$

#### Solution Approach 1: Split the DCSI loops

One way to restore the standard element arrangement during a D2H copy is by
rewriting the copy loop nest by splitting the original dimensions into smaller
loops with different source (device) and destination (host) strides.

Loop Splitting Table:

| Original Dimension | New Loop | Size | Src Step (Device) | Dst Step (Host) | Logic |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Dim 0**<br>(32 elements) | `elem_in_group` | 4 | 1 elem | 1 col | Accesses individual elements. |
| | `group_in_stick` | 8 | 4 elems | 8 cols | Skips the 4 columns belonging to the other stick. |
| **Stick Dim**<br>(N sticks) | `stick_in_pair` | 2 | 1 stick | 4 cols | Switches between the two sticks in a pair. |
| | `stick_pair` | $N/2$ | 2 sticks | 64 cols | Moves to the next pair of sticks. |

How it works:

1. **`group_in_stick` Loop:** Advancing this loop moves the device pointer by
   4 elements (the size of one group) but jumps 8 columns on the host. This
   jump accounts for the 4 columns of the current group plus the 4 columns of
   the interleaved group from the other stick.
2. **`stick_in_pair` Loop:** Advancing this loop moves the device pointer by
   an entire stick (32 elements) but only advances 4 columns on the host. This
   allows the copy engine to pick up the interleaved groups from the second
   stick.
3. **`stick_pair` Loop:** Advances both pointers by the full width of a stick
   pair (64 columns / 2 sticks).

 Constraints

* **Full Sticks Only:** The restoration logic requires that sticks are fully
  populated.
* **Even Number of Sticks:** The stick dimension must have an even number of
  sticks to form complete pairs.
* **Host Stride Identification:** The stick dimension is identified by its
  **HOST stride** (one stick's worth of host columns), not its device stride,
  as the device may use different internal ordering (e.g., stick-major).

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

$$
\text{host\_col} = (\text{stick} \times 64) + (\text{half\_in\_stick} \times 32) + (\text{group\_in\_half} \times 4) + \text{elem\_in\_group}
$$

| Component | Range | Description |
| :--- | :--- | :--- |
| `stick` | $[0, N)$ | Identifies the 16-bit stick. |
| `half_in_stick` | $[0, 2)$ | Identifies which half of the stick's host columns. |
| `group_in_half` | $[0, 8)$ | Identifies the group within a half. |
| `elem_in_group` | $[0, 4)$ | Identifies the specific element within the group. |

The corresponding device location is calculated as:

* **Device Stick:** $\text{stick}$
* **Device Position:** $\text{group\_in\_half} \times 8 + \text{half\_in\_stick} \times 4 + \text{elem\_in\_group}$

#### Solution Approach 1: Split the DCSI loops

Only dim 0 is split. All other dimensions, including the stick dimension, are
copied unchanged.

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

Constraints:

* **Full Sticks Only:** Dim 0 must cover a complete 16-bit stick.
* **Stick Size:** The stick size must be divisible by 8 (2 halves $\times$ 4
  elements per group).
* No even-stick-count requirement and no stick dimension identification are
  needed, since the stagger never crosses a stick boundary.
