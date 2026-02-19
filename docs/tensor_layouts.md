# Tensor Layouts

In this document we discuss the rationale for Spyre tensor layouts, the
specifics, and their relationship with PyTorch tensor layouts. This document
complements the [Tiled Tensor RFC](../RFCs/0047-TiledTensors/0047-TiledTensorsRFC.md)
by describing the specific device memory layouts and related APIs used for Tensors in Torch-Spyre.

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

A number of compute operations on Spyre produce _sparse_ tensors, i.e., tensors with a
single element per 128-byte _stick_ of tensor data. In order to describe sparse
tensor layouts we permit Spyre tensor layouts to optionally include a single
synthetic dimension that does not correspond to any dimension of the PyTorch
layout. This synthetic inner dimension associated with a size equal to the
maximal number of elements per stick for the tensor data type will ensure that
the sparse tensor has a single element of the corresponding PyTorch tensor per
stick.

## Spyre Tensor Layouts

A Spyre tensor has a Spyre tensor layout in addition to a PyTorch tensor layout.

A Spyre tensor layout consists of a _device\_size_ vector, a _dim\_map_ vector
with the same number of elements called _device\_rank_.

The device_rank is always greater than or equal to the rank of the
(canonicalized) PyTorch tensor layout.

In combination with a PyTorch tensor layout, a Spyre tensor layout makes it
possible to represent tiled tensors, sparse tensors, and padded tensors.

In contrast with a PyTorch tensor layout, a Spyre tensor has no explicit stride
vector. A Spyre tensor layout is always in row-major format, i.e., the strides
in the implicit stride vector are always decreasing obtained by formula:

```
stride[i] = math.prod(size[i+1:device_rank])
```

For now, a Spyre tensor layout has a unique _stick dimension_, which is always
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

## Default Layouts and Controlling Layouts

Spyre tensors are created using two fundamental PyTorch APIs.  
- The `to()` method is used to transfer all elements of an existing
   (host) tensor to a newly allocated device tensor; the result of `to`
   is the device tensor object.
- The `new_empty()`, `new_empty_strided()`, etc. methods are used
   to create an uninitialized device tensor; the result of the method
   is the device tensor object.
Both of these APIs can be invoked either with or without providing an
explicit `SpyreTensorLayout`.  When a `SpyreTensorLayout` is provided, it
specifies precisely how the device tensor will be laid out. When the APIs are
invoked without providing a `SpyreTensorLayout` the device tensor
is created using a default layout. Conceptually the default layout
(a) designates the last dimension as the stick dimension, (b) tiles
along the first dimension, and (c) pads the size of the stick dimension
to make it evenly divisible into sticks.

### Default Layout Example
The layout metadata is encoded by the runtime C++ class `SpyreTensorLayout` (see [spyre_tensor_impl.h](../torch_spyre/csrc/spyre_tensor_impl.h)).
An instance of this class is embedded as a field in the `SpyreTensorImpl` class.
It can be accessed in Python via an added Tensor method `device_tensor_layout()`.
The key elements of metadata are:
- `device_size`: analagous to PyTorch's `size` but with padded values and extra dimensions for tiling.
- `dim_map`: a vector of the same length as `device_size` giving the index in the PyTorch `size` array for each element of `device_size`.
- `device_dtype`: the datatype of the Tensor.

As a concrete example, run the following program:

```
import torch
x = torch.rand(5, 100, 150, dtype=torch.float16)
y = x.to("spyre")
stl = y.device_tensor_layout()
print(stl)
```

You should see something like:

```
SpyreTensorLayout(device_size=[100, 3, 5, 64], dim_map =[1, 2, 0, 2], device_dtype=DataFormats.SEN169_FP16)
```

The 3-D tensor has a 4-D `device_size`.  
A float16 is two bytes, therefore each stick contains 64 data values.
The stick dimension of `150` has been padded to `192` and broken into two device dimensions of (`3` and `64`).

### Specifiying Alternate Layouts

The minimal constructor for a `SpyreTensorLayout` takes a `size` and `dtype` and
builds a instance that encodes the default layout.  This constructor
is what is used behind the scenes when the user does not specify a layout.

As an example, we can explictly request the default layout in a `to` by doing:

```
import torch
from torch_spyre._C import SpyreTensorLayout
x = torch.rand(5, 100, 150, dtype=torch.float16)
stl = SpyreTensorLayout((5, 100, 150), torch.float16)
y = x.to("spyre",device_layout=stl)
print(y.device_tensor_layout())
```

You should see exactly the same output as before:

```
SpyreTensorLayout(device_size=[100, 3, 5, 64], dim_map =[1, 2, 0, 2], device_dtype=DataFormats.SEN169_FP16)
```

A second constructor of `SpyreTensorLayout` enables finer-grained control.
It takes an additional `dim_order` allowing the programmer
to fine-tune the layout based on their knowledge of how the Tensor will be used
in computation.

For example, changing the constructor in the above program to

```
stl = SpyreTensorLayout((5, 100, 150), torch.float16, [1,0,2])
```

yields a tensor with the tiling inverted:

```
SpyreTensorLayout(device_size=[5, 3, 100, 64], dim_map =[0, 2, 1, 2], device_dtype=DataFormats.SEN169_FP16)
```

## Layout Compatibility

- Operation validation and layouts for computed tensors

## Generating DCIs and SuperDSCs

TODO

## Future Extensions

- Gaps
- Multiple stick dimensions
- Multiple memory spaces
- RoPE
