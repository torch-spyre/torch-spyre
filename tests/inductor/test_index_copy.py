import torch

def store(out, index, src):
    out.index_copy_(0, index, src)

for rows in (1, 2):
    out = torch.zeros(rows, 8, 128, dtype=torch.float16, device="spyre")
    src = torch.randn(rows, 8, 128, dtype=torch.float16).to("spyre")
    idx = torch.arange(rows, dtype=torch.int64).to("spyre")
    torch.compile(store, dynamic=False)(out, idx, src)
    print(f"rows={rows}: destination still all zero? {bool(out.cpu().eq(0).all())}")
