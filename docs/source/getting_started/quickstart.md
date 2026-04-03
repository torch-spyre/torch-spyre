# Run PyTorch on Spyre device

Using Torch-Spyre, you can run PyTorch on `spyre` device as further described in this document.

## Creating a Tensor

The Torch-Spyre adds the `spyre` device type to PyTorch. This device type works similarly to other PyTorch device types. The example below shows an example of creating a Torch-Spyre tensor:

``` python3
import torch

x = [[1, 2], [3, 4]]
x = torch.tensor(x, dtype=torch.float16, device="spyre")
print(x)
print(x.device)
```

## Running Tensor Operations

Torch-Spyre supported operations can be performed on `spyre` device in a similar way to using other devices.

For example, you can add `spyre` tensors together as below:

``` python3
import torch

DEVICE = torch.device("spyre")
x = torch.rand(512, 1024, dtype=torch.float16).to(DEVICE)
y = torch.rand(512, 1024, dtype=torch.float16).to(DEVICE)

output = x + y # or torch.add(x, y)
print(output)
```

You can do matrix multiplication in a various ways as below:

``` python3
import torch

DEVICE = torch.device("spyre")
x = torch.rand(512, 1024, dtype=torch.float16).to(DEVICE)
y = torch.rand(1024, 512, dtype=torch.float16).to(DEVICE)

output = torch.matmul(x, y)
print(f"Output of torch.matmul\n: {output}")

output = torch.mm(x, y)
print(f"Output of torch.mm\n: {output}")

output = x @ y
print(f"Output of matmul with @ operator\n: {output}")
```

And here is an example of using `torch.compile`:

``` python3
import torch

DEVICE = torch.device("spyre")
x = torch.rand(512, 1024, dtype=torch.float16).to(DEVICE)
y = torch.rand(1024, 512, dtype=torch.float16).to(DEVICE)
c_matmul = torch.compile(torch.matmul)
output = c_matmul(x, y)
print(f"Output of matmul with torch.compile\n: {output}")
```

# More Examples
Refer to the [examples](https://github.com/torch-spyre/torch-spyre/tree/main/examples) directory in this repository, which provides more examples of using PyTorch on `spyre` device.
