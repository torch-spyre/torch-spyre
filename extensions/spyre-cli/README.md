# spyre-cli

## Install

Needs a working Spyre stack environment.

```
git clone ...
uv pip install .
```

## CLI

```
spyre launch --path <folder with spyreCode> <path to io file>
```

## SDK

Example, in a folder with the "spyreCode" for a torch.add operation:

```
import torch
import spyre_cli

a = torch.ones([512, 1024], device="spyre", dtype=torch.float16)
b = torch.ones([512, 1024], device="spyre", dtype=torch.float16)
c = torch.empty([512, 1024], device="spyre", dtype=torch.float16)

spyre_cli.launch(a, b, c)

print(c)
```

However, you have the following constraints:
1. You need to pass the right input and output list.
2. If the shapes don't match, there is no error - the code will silently work.
