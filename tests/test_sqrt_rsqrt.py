"""Simple test for sqrt and rsqrt operations comparing Spyre vs CPU."""

import torch

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

print("=" * 60)
print("Testing sqrt operation (Spyre vs CPU)")
print("=" * 60)

# Test sqrt with positive values
x = torch.abs(torch.rand(128, 256, dtype=torch.float32)) + 1e-3

# Compute on CPU
cpu_result_sqrt = torch.sqrt(x)

# Compute on Spyre device
x_device = x.to(DEVICE)
compiled_sqrt = torch.compile(lambda a: torch.sqrt(a))
spyre_result_sqrt = compiled_sqrt(x_device).cpu()

# Compare results
print(f"CPU result (sqrt):\n{cpu_result_sqrt}")
print(f"Spyre result (sqrt):\n{spyre_result_sqrt}")
delta_sqrt = torch.abs(spyre_result_sqrt - cpu_result_sqrt).max()
print(f"Max delta (sqrt) Spyre vs. CPU: {delta_sqrt}")

print("\n" + "=" * 60)
print("Testing rsqrt operation (Spyre vs CPU)")
print("=" * 60)

# Test rsqrt with positive values (avoid zero)
y = torch.abs(torch.rand(128, 256, dtype=torch.float16)) + 1e-3

# Compute on CPU
cpu_result_rsqrt = torch.rsqrt(y)

# Compute on Spyre device
y_device = y.to(DEVICE)
compiled_rsqrt = torch.compile(lambda a: torch.rsqrt(a))
spyre_result_rsqrt = compiled_rsqrt(y_device).cpu()

# Compare results
print(f"CPU result (rsqrt):\n{cpu_result_rsqrt}")
print(f"Spyre result (rsqrt):\n{spyre_result_rsqrt}")
delta_rsqrt = torch.abs(spyre_result_rsqrt - cpu_result_rsqrt).max()
print(f"Max delta (rsqrt) Spyre vs. CPU: {delta_rsqrt}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"sqrt  max delta: {delta_sqrt}")
print(f"rsqrt max delta: {delta_rsqrt}")

# Check if deltas are within acceptable range for FP16
tolerance = 1e-2  # FP16 has limited precision
if delta_sqrt < tolerance and delta_rsqrt < tolerance:
    print("\n✓ All tests passed!")
else:
    print(f"\n✗ Tests failed! Deltas exceed tolerance ({tolerance})")

