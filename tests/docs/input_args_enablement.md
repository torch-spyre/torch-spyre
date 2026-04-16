# Customization of Input Arguments for Tests -- Tensors, Keyword Arguments

**Authors:**

- Anubhav Jana (IBM Research, India)
- Kazuaki Ishizaki (IBM Research, Tokyo, Japan)
- Tuan Hoang Trong (IBM Research, Yorktown Heights, US)
- Umamaheswari Devi (IBM Research, India)
- Ashok Pon Kumar Sree Prakash (IBM Research, India)

This document describes the `edits.inputs` field, which will let us specify exact input arguments for any test in the config. It is part of the `edits` block inside a test entry.

---

## Motivation

Currently upstream pytorch test framework generates inputs automatically by default without any explicit control. This works for most cases, but sometimes we need explicit control, for e.g. to generate shapes to handle spyre specific tensor layouts:

- A module test might need a specific tensor shape (e.g. `BatchNorm2d` might need NCHW input)
- An op test might need a non-contiguous memory layout to exercise a specific code path
- A test takes scalar, `None`, or slice arguments alongside tensors
- We want to pin inputs to shapes observed during model tracing

When `edits.inputs` is absent, the framework uses its default input generation. When present, our input spec takes over.

---

## Where it goes in the config

`edits.inputs` is a field inside a test entry's `edits` block, at the same level as `edits.ops`, `edits.dtypes`, and `edits.modules`:

```yaml
tests:
  - names:
      - TestModule::test_forward
    mode: mandatory_success
    edits:
      inputs:           # <- here
        args:
          - tensor:
              shape: [4, 16, 32, 32]
              dtype: float16
              device: spyre
              init: rand
        kwargs:
          training: false
```

---

## Structure

```
edits.inputs
├── args      ordered list of positional arguments
└── kwargs    flat dict of keyword arguments
```

Both are optional. You can specify only `args`, only `kwargs`, or both.

---

## `args` — positional arguments

`args` is an ordered list. Each element is **exactly one** of four kinds.

### 1. `tensor` — a single tensor

```yaml
args:
  - tensor:
      shape: [1, 11, 2880]
      dtype: float16
      device: spyre
      init: rand
```

#### Tensor fields

| Field | Required | Default | Description |
|---|---|---|---|
| `shape` | Yes | — | List of positive ints. Use `[]` for a scalar tensor |
| `dtype` | Yes | — | PyTorch dtype name — see [Dtype names](#dtype-names) |
| `device` | Yes | — | Device string: `spyre`, `cpu`, `cuda:0`, etc. |
| `init` | Yes | — | How to populate values — see [Init strategies](#init-strategies) |
| `init_args` | No | — | Extra arguments for the init strategy |
| `stride` | No | contiguous | List of ints, same length as `shape`. Omit for default C-contiguous layout |
| `storage_offset` | No | `0` | Storage offset in number of elements |

When `stride` is specified, the framework allocates an appropriately-sized backing storage and might construct the tensor with `torch.as_strided`.

#### Init strategies

| `init` | PyTorch call | Notes |
|---|---|---|
| `rand` | `torch.rand(shape, ...)` | Uniform [0, 1). Floating-point dtypes only |
| `randn` | `torch.randn(shape, ...)` | Standard normal. Floating-point dtypes only |
| `zeros` | `torch.zeros(shape, ...)` | All zeros |
| `ones` | `torch.ones(shape, ...)` | All ones |
| `randint` | `torch.randint(low, high, shape, ...)` | Requires `init_args.high`. `init_args.low` defaults to `0` |
| `arange` | `torch.arange(shape[0], ...)` | Shape must be 1-D |
| `eye` | `torch.eye(shape[0], ...)` | Shape must be 2-D with equal dimensions |
| `full` | `torch.full(shape, fill_value, ...)` | Requires `init_args.fill_value` |
| `file` | `torch.load(path)` or format-equivalent | Load tensor from a file on disk. Requires `init_args.path`. Supported formats: `.pt`, `.npy`, `.safetensors` |

`init_args` sub-fields:

```yaml
init_args:
  low: 0                            # randint only -- lower bound (default 0)
  high: 1000                        # randint only -- upper bound (required)
  fill_value: 3.14                  # full only -- scalar fill value (required)
  path: /path/to/tensor.pt          # file only -- path to tensor file (required)
  key: weight                       # file only -- key to load from file (optional, for safetensors / dict-based .pt)
```

`path` accepts `.pt`, `.npy`, and `.safetensors` files. Use this when a tensor is too large or too specific to reconstruct from a simple init strategy — for example, a weight matrix captured during model tracing, or a reference input required for numerical comparison. The `shape` and `dtype` fields are still required and are validated against the loaded tensor at config load time; a mismatch raises an error.

```yaml
# Load a tensor saved from a model trace
- tensor:
    shape: [201088, 2880]
    dtype: float16
    device: spyre
    init: file
    init_args:
      path: ${TORCH_DEVICE_ROOT}/tests/fixtures/embed_weight.pt

# Load a specific key from a safetensors checkpoint
- tensor:
    shape: [4096, 2880]
    dtype: float16
    device: spyre
    init: file
    init_args:
      path: ${TORCH_DEVICE_ROOT}/tests/fixtures/model.safetensors
      key: model.layers.0.self_attn.q_proj.weight

# Load a numpy array saved with np.save
- tensor:
    shape: [1, 11, 2880]
    dtype: float16
    device: spyre
    init: file
    init_args:
      path: ${TORCH_DEVICE_ROOT}/tests/fixtures/hidden_states.npy
```

`fill_value` accepts any number and populates every element of the tensor with that constant -- we may use it when an op's behaviour is sensitive to a specific input value, such as a padding sentinel, a fixed denominator, or a known mask value:

```yaml
- tensor:
    shape: [4, 128]
    dtype: int64
    device: spyre
    init: full
    init_args:
      fill_value: -1      # e.g. padding index sentinel
 
- tensor:
    shape: [1, 16, 32, 32]
    dtype: float16
    device: spyre
    init: full
    init_args:
      fill_value: 0.5     # e.g. fixed denominator or scale factor
```

#### Dtype names

`float16`, `float32`, `float64`, `bfloat16`, `int8`, `int16`, `int32`, `int64`, `uint8`, `bool`, `complex64`, `complex128`

---

### 2. `tensor_list` — a list of tensors as one positional argument

Use this for ops that take a sequence of tensors as a single positional argument, such as `torch.cat` or `torch.stack`.

```yaml
args:
  - tensor_list:
      - shape: [1, 8, 11, 32]
        dtype: float16
        device: spyre
        init: rand
      - shape: [1, 8, 11, 32]
        dtype: float16
        device: spyre
        init: rand
```

Each item in the list is a tensor spec with the same fields as `tensor` above.

---

### 3. `value` — a scalar or None

Use this for plain Python scalars: numbers, `null` (`None`), `true`/`false`.

```yaml
args:
  - value: 2          # int
  - value: 1.5        # float
  - value: null       # None
  - value: false      # False
```

`value` covers the non-tensor positional arguments that appear in most op call signatures -- things like a `dim` axis, a `repeats` count, an `alpha` scaling factor, or a `None` where an optional argument is not used.

Consider `torch.transpose(input, dim0, dim1)`. In Python you would call it as:

```python
torch.transpose(tensor, 1, 2)
```

The equivalent config — one tensor followed by two integer `value` entries:

```yaml
args:
  - tensor:
      shape: [1, 32, 11]
      dtype: float16
      device: spyre
      init: rand
  - value: 1          # dim0
  - value: 2          # dim1
```

Or `torch.clamp(input, min, max)` where one bound is disabled:

```python
torch.clamp(tensor, min=None, max=7.0)
```

```yaml
args:
  - tensor:
      shape: [44, 2880]
      dtype: float16
      device: spyre
      init: rand
  - value: null       # min -- disabled
  - value: 7.0        # max
```

---

### 4. `py` — a Python literal expression

Use this for arguments that cannot be expressed as a simple YAML scalar: slices, tuples, `Ellipsis`. The string is evaluated with `ast.literal_eval` -- no arbitrary code execution.

```yaml
args:
  - py: "(Ellipsis, slice(None, -1, None))"
  - py: "(slice(None), None, slice(None))"
  - py: "(1, 2, 3)"
```

`py` covers index expressions and other structured arguments that cannot be written as a plain YAML scalar. The most common use is fancy indexing -- anywhere you would write `tensor[...]` in Python.

Consider `scores[..., :-1]` -- dropping the last element along the final dimension:

```python
# Python
output = scores[(Ellipsis, slice(None, -1, None))]
```

```yaml
args:
  - tensor:
      shape: [1, 64, 11, 129]
      dtype: float16
      device: spyre
      init: rand
  - py: "(Ellipsis, slice(None, -1, None))"
```

Or `tensor[:, None, :]` -- inserting a new axis in the middle:

```python
# Python
output = tensor[(slice(None), None, slice(None))]
```

```yaml
args:
  - tensor:
      shape: [1, 11]
      dtype: int64
      device: spyre
      init: randint
      init_args:
        high: 1000
  - py: "(slice(None), None, slice(None))"
```

Use `value` instead when the argument is a plain scalar or `None` on its own. `py` is only needed when the argument is a composite — a tuple, a slice, or `Ellipsis`.

Supported Python literals: strings, bytes, numbers, tuples, lists, dicts, sets, booleans, `None`, `Ellipsis`, `slice(start, stop, step)`.

---

## `kwargs` — keyword arguments

A flat key-value mapping passed as keyword arguments to the test or module forward call.

```yaml
kwargs:
  dim: -1
  keepdim: true
  training: false
  dtype: float16     # resolved to torch.float16
  max_norm: null     # resolved to None
```

`kwargs` covers the named arguments in any op or module call. Consider `torch.mean(input, dim, keepdim)`:

```python
# Python
output = torch.mean(tensor, dim=-1, keepdim=True)
```

```yaml
args:
  - tensor:
      shape: [1, 11, 2880]
      dtype: float16
      device: spyre
      init: rand
kwargs:
  dim: -1
  keepdim: true
```

Or `torch.nn.functional.softmax(input, dim, dtype)` where the dtype kwarg is resolved to a `torch.*` type automatically:

```python
# Python
output = F.softmax(tensor, dim=-1, dtype=torch.float16)
```

```yaml
args:
  - tensor:
      shape: [1, 64, 11, 128]
      dtype: float16
      device: spyre
      init: rand
kwargs:
  dim: -1
  dtype: float16     # resolved to torch.float16 by our framework
```

Type resolution applied automatically:

| YAML value | Python value |
|---|---|
| String matching a dtype name | `torch.<dtype>` object |
| `null` | `None` |
| `true` / `false` | `True` / `False` |
| Number | int or float as-is |
| Any other string | plain Python string |

---

## Interaction with `@ops` dtype parametrization

For tests decorated with `@ops`, the framework generates one variant per `(op, dtype)` combination. When `edits.inputs` is present:

- The **shape, layout, and init strategy** come from our spec.
- The **dtype** of each tensor is cast to match the active variant dtype if there is a mismatch. A warning is emitted when this happens.

For module tests and plain tests without `@ops`, the tensor spec dtype is used as-is with no override.

---

## Examples

### Module test with specific input shape

`BatchNorm2d` requires a 4-D NCHW tensor. The default input generator may not produce the right shape:

```yaml
- names:
    - TestModule::test_forward
  mode: mandatory_success
  edits:
    modules:
      include:
        - name: torch.nn.BatchNorm2d
    inputs:
      args:
        - tensor:
            shape: [4, 16, 32, 32]
            dtype: float16
            device: spyre
            init: rand
      kwargs:
        training: false
```

---

### Non-contiguous layout

Stress the transpose code path with a column-major tensor:

```yaml
- names:
    - TestBinaryUfuncs::test_contig_vs_transposed
  mode: mandatory_success
  edits:
    inputs:
      args:
        - tensor:
            shape: [64, 64]
            dtype: float16
            device: spyre
            init: randn
            stride: [1, 64]       # column-major
        - tensor:
            shape: [64, 64]
            dtype: float16
            device: spyre
            init: randn
```

---

### Mixed tensors, scalars, and None

Matching the call signature of `torch.nn.functional.embedding`:

```python
# Python
output = torch.nn.functional.embedding(
    input,          # shape [1, 11], int64
    weight,         # shape [201088, 2880], float16
    padding_idx=199999,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
)
```

```yaml
- names:
    - TestNN::test_embedding
  mode: mandatory_success
  edits:
    inputs:
      args:
        - tensor:
            shape: [1, 11]
            dtype: int64
            device: spyre
            init: randint
            init_args:
              high: 1000
        - tensor:
            shape: [201088, 2880]
            dtype: float16
            device: spyre
            init: rand
        - value: 199999      # padding_idx
        - value: null        # max_norm
        - value: 2.0         # norm_type
        - value: false       # scale_grad_by_freq
        - value: false       # sparse
```

---

### Tensor list (torch.cat style)

```python
# Python
output = torch.cat(
    [tensor_a,   # shape [1, 8, 11, 32], float16
     tensor_b],  # shape [1, 8, 11, 32], float16
    dim=-1,
)
```

```yaml
- names:
    - TestBinaryUfuncs::test_cat
  mode: mandatory_success
  edits:
    inputs:
      args:
        - tensor_list:
            - shape: [1, 8, 11, 32]
              dtype: float16
              device: spyre
              init: rand
            - shape: [1, 8, 11, 32]
              dtype: float16
              device: spyre
              init: rand
      kwargs:
        dim: -1
```

---

### Slice / Ellipsis index argument

```python
# Python
output = tensor[..., :-1]   # tensor shape [1, 64, 11, 129], float16
```

```yaml
- names:
    - TestViewOps::test_getitem_slice
  mode: mandatory_success
  edits:
    inputs:
      args:
        - tensor:
            shape: [1, 64, 11, 129]
            dtype: float16
            device: spyre
            init: rand
        - py: "(Ellipsis, slice(None, -1, None))"
```

---

### randint with explicit bounds

```python
# Python
output = tensor.index_copy_(
    2,       # dim
    index,   # shape [11], int64, values in [0, 128)
    source,  # shape [1, 8, 11, 64], float16
)
# tensor shape [1, 8, 128, 64], float16
```

```yaml
- names:
    - TestOps::test_index_copy
  mode: mandatory_success
  edits:
    inputs:
      args:
        - tensor:
            shape: [1, 8, 128, 64]
            dtype: float16
            device: spyre
            init: rand
        - value: 2
        - tensor:
            shape: [11]
            dtype: int64
            device: spyre
            init: randint
            init_args:
              low: 0
              high: 128
        - tensor:
            shape: [1, 8, 11, 64]
            dtype: float16
            device: spyre
            init: rand
```

---

### Large matmul shapes from model tracing

Pin inputs to shapes observed in a GPT model trace:

```python
# Python
output = torch.bmm(
    input,   # shape [44, 1, 2880], float16
    mat2,    # shape [44, 2880, 5760], float16
)
# output shape [44, 1, 5760]
```

```yaml
- names:
    - TestLinalg::test_bmm
  mode: mandatory_success
  tags:
    - gpt_oss_20b
  edits:
    inputs:
      args:
        - tensor:
            shape: [44, 1, 2880]
            dtype: float16
            device: spyre
            init: rand
        - tensor:
            shape: [44, 2880, 5760]
            dtype: float16
            device: spyre
            init: rand
```

---

### dtype kwarg resolved to torch type

Useful when an op accepts a `dtype` argument to control the computation or output type -- without this resolution, passing the string `"float16"` directly would cause a type error since PyTorch expects a `torch.dtype` object, not a string.

```python
# Python
output = torch.nn.functional.softmax(
    tensor,              # shape [1, 64, 11, 128], float16
    dim=-1,
    dtype=torch.float16, # controls compute dtype -- must be a torch.dtype, not a string
)
```

```yaml
- names:
    - TestNN::test_softmax
  mode: mandatory_success
  edits:
    inputs:
      args:
        - tensor:
            shape: [1, 64, 11, 128]
            dtype: float16
            device: spyre
            init: rand
      kwargs:
        dim: -1
        dtype: float16       # -> torch.float16
```

---

## Validation rules

| Rule | Detail |
|---|---|
| Each `args` element must contain exactly one of `tensor`, `tensor_list`, `value`, `py` | Mixing keys within one element is an error |
| `tensor.shape` | Non-empty list of positive ints. `[]` allowed for scalar tensors |
| `tensor.dtype` | Must be a valid PyTorch dtype name |
| `tensor.device` | Must be a valid device string |
| `tensor.init` | Must be one of the eight supported strategies |
| `init_args.high` | Required when `init: randint` |
| `init_args.fill_value` | Required when `init: full` |
| `tensor.stride` | Must have same length as `tensor.shape` if specified |
| `tensor.storage_offset` | Must be a non-negative integer if specified |
| `arange` | Shape must be 1-D |
| `eye` | Shape must be 2-D with equal dimensions |
| `py` values | Must be parseable by `ast.literal_eval`. Arbitrary code execution is not permitted |
| dtype mismatch with `@ops` variant | Warning emitted; variant dtype takes precedence and the tensor is cast |
| `init_args.path` | Required when `init: file`. Must resolve to an existing file with a supported extension (`.pt`, `.npy`, `.safetensors`) |
| `init_args.key` | Optional when `init: file`. Required when the `.pt` file contains a `dict` or the `.safetensors` file holds multiple tensors and no default can be inferred |
| `tensor.shape` when `init: file` | Must match the shape of the loaded tensor exactly. Mismatch raises a validation error at config load time |
| `tensor.dtype` when `init: file` | Must match the dtype of the loaded tensor. Mismatch raises a validation error at config load time |

---

## Field reference at a glance

```
edits.inputs
│
├── args (list)
│   ├── - tensor:
│   │       shape:           [int, ...]        required
│   │       dtype:           string            required
│   │       device:          string            required
│   │       init:            string            required
│   │       init_args:
│   │           low:         int               optional (randint)
│   │           high:        int               required for randint
│   │           fill_value:  number            required for full
│   │           path:        string            required for file          
│   │           key:         string            optional for file     
│   │       stride:          [int, ...]        optional
│   │       storage_offset:  int               optional
│   │
│   ├── - tensor_list:
│   │       - shape / dtype / device / init /  (same fields as tensor)
│   │         init_args / stride / storage_offset
│   │
│   ├── - value: <scalar | null | true | false>
│   │
│   └── - py: "<ast.literal_eval-compatible string>"
│
└── kwargs
        <key>: <value>      dtype strings auto-resolved to torch.* objects
```
