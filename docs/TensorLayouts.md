# Tensor Layouts

In this document we discuss the rationale for Spyre tensor layouts, the
specifics, and their relationship with PyTorch tensor layouts.

## PyTorch Tensor Layouts

A PyTorch tensor has an integer _rank_ also referred to as a number of
dimensions. More precisely, the _dimensions_ of a PyTorch tensors are the
integers in the range `range(rank)`.

A tensor layout consists of a _size_ vector with rank elements and a _stride_
vector with rank elements. Elements of the size and stride vectors are often
informally referred to as sizes and strides as a shorthand for per-dimension
sizes and strides.

The stride vector makes it possible to map a tuple of rank coordinates to an
offset, hence to order the tensor elements in a 1d contiguous memory space.

```
offset = lambda coordinates : np.dot(coordinates, stride)
```

## Motivation for Spyre Tensor Layouts

PyTorch tensors have a single stride per dimension, hence cannot represent tiled
tensors. Because of this limitation we introduce Spyre tensor layouts with
higher ranks than their PyTorch counterparts. Intuitively by breaking PyTorch
dimensions into pieces, we can build tiles and tensors from these tiles.

While strides make it possible to express padding in PyTorch tensor layouts,
because Spyre tensor layouts have more dimensions, we need more dimensions of
padding. Therefore, we introduce padded sizes in Spyre tensor layouts maintained
separately from Pytorch sizes. Since PyTorch already maintains sizes, we only
include padded sizes in a Spyre tensor layout. While we could work with strides
instead, we find it easier to reason about padded sizes and order of dimensions
separately rather than combining them into strides.

PyTorch eliminates tensor dimensions with per-dimension size 1 whenever
possible, for instance replacing layout `(size=[512, 1, 256], stride=[256, 256,
1])` with layout `(size=[512, 256], stride=[256, 1])`. After careful
consideration we concluded that dimensions of size 1 must not contribute to the
Spyre layout of a tensor. For this reason, we say a PyTorch tensor layout is in
_canonical form_ if it has no dimension of size 1 and canonicalize PyTorch
tensor layouts before reasoning about them. To be clear, this does not preclude
selecting a different layout on Spyre for a tensor of size `[512, 1]` vs. a
tensor of size `[512]` but this will require explicitly specifying the desired
Spyre layout as the default is the same for both.

A number of operations on Spyre produce _sparse_ tensors, i.e., tensors with a
single element per 128-byte _stick_ of tensor data. In order to describe sparse
tensor layouts we permit Spyre tensor layouts to optionally include a single
synthetic dimension that does not correspond to any dimension of the PyTorch
layout. This synthetic inner dimension associated with a size equal to the
maximal number of elements per stick for the tensor data type will ensure that
the sparse tensor has a single element of the corresponding PyTorch tensor per
stick.

## Spyre Tensor Layouts

A Spyre tensor has a Spyre tensor layout in addition to a PyTorch tensor layout.
While the PyTorch layout of a tensor may or may not be described in canonical
form, in the sequel we always assume it is, implicitly canonicalizing the
PyTorch layout if necessary.

A Spyre tensor layout consists of a _device\_size_ vector, a _dim\_map_ vector
with the same number of elements called _device\_rank_.

The device_rank is always greater than or equal to the rank of the
(canonicalized) PyTorch tensor layout.

In combination with a PyTorch tensor layout, a Spyre tensor layout makes it
possible to represent tiled tensors, sparse tensors, and padded tensors
altogether.

In contrast with a PyTorch tensor layout, a Spyre tensor has no explicit stride
vector. A Spyre tensor layout is always in row-major format, i.e., the strides
in the implicit stride vector are always decreasing obtained by formula:

```
stride[i] = math.prod(size[i+1:device_rank])
```

For now a Spyre tensor layout has a unique _stick dimension_, which is always
dimension device_rank-1. Elements in an 128-byte-aligned 128-byte _stick_ of
tensor data (in a 128-byte-aligned tensor) share the same coordinates for
dimensions 0 to device_rank-2. The device_size of the stick dimension is always
the maximal number of element per stick for the tensor data type.

The dim_map vector maps the dimensions in the Spyre tensor layout back to the
dimensions in the PyTorch tensor layout. The elements of this vector are
integers in the range `range(-1, rank)` where elements in range `range(rank)`
represent dimensions of the PyTorch tensor layout and `-1` if present
represents a synthetic dimension that does not exist in the PyTorch tensor
layout. dim_map elements in `range(rank)` must occur at least once. dim_map
elements may be repeated.

Repeated dimensions in dim_map encode tiling. For example, for a 3d PyTorch
tensor of size `[128, 256, 512]`, a dim_map `[1, 2, 0, 2]` and device_size
`[256, 8, 128, 64]` specifies that dimension 2 of the PyTorch tensor is tiled
with dimension 0, whereas dimension 1 of the PyTorch tensor becomes the
outermost dimension of the Spyre tensor layout. In this example, the element
with coordinates `(a, b, c, d)` in the Spyre tensor corresponds to the PyTorch
element `(c, a, b*64 + d)`. The coordinates of a tiled dimension are always
combined into a PyTorch coordinate with strides increasing right-to-left akin to
the implicit strides of the whole Spyre tensor layout.

The stride of the PyTorch layout does not play a role when mapping Spyre
coordinates to PyTorch coordinates but of course it matters to mapping the
PyTorch coordinates to an offset from the base address of the PyTorch tensor.

Dimensions in device_size may be padded. For example the previous Spyre tensor
layout with dim_map `[1, 2, 0, 2]` and device_size `[256, 8, 128, 64]` may also
be used for a PyTorch tensor of size `[100, 200, 500]` in which case coordinates
in the Spyre tensor layout that do not map to valid coordinates in the PyTorch
tensor layout represent padding.

## Access patterns

- Dividing tensor access across cores

## Default Layouts and Layout Compatibility

- Default layouts for input tensors
- Operation validation and layouts for computed tensors

## Generating DCIs and SuperDSCs

TODO

## Future Extensions

- Gaps
- Multiple stick dimensions
- Multiple memory spaces
- RoPE
