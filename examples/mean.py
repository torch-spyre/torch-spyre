import torch


DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

x = torch.rand(128, 64, dtype=torch.float16)

cpu_result = torch.mean(x, dim=0)

x_device = x.to(DEVICE)

compiled = torch.compile(lambda a: torch.mean(a, dim=0))
compiled_result = compiled(x_device).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
cpu_delta = torch.abs(compiled_result - cpu_result).max()

print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
