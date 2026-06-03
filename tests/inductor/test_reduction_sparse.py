import torch

import pytest

DEVICE = torch.device("spyre")


def _make_2d_tensor(s1, s2):
    # A (cpu tensor), B(spyre tensor): shape [s1, s2];
    A = torch.randn((s1, s2), dtype=torch.float16)
    B = A.to(device="spyre")
    return A, B


SIZES_2D_FULL = [
    (256, 128),
    (128, 256),
    (128, 128),
    (64, 128),
    (128, 64),
]


@pytest.fixture(params=SIZES_2D_FULL, ids=lambda p: f"{p[0]}x{p[1]}")
def tensors_2arg(request):
    s1, s2 = request.param
    return _make_2d_tensor(s1, s2)


def _test_reduction_result(tensors_2arg, reduction_fn):
    torch._dynamo.reset_code_caches()
    torch._inductor.codecache.FxGraphCache.clear()

    compiled_reduction_fn = torch.compile(reduction_fn)

    cpu_tensor, spyre_tensor = tensors_2arg
    res_cpu = compiled_reduction_fn(cpu_tensor)
    res_spyre = compiled_reduction_fn(spyre_tensor)

    cpu_layout = res_cpu.to(device="spyre").device_tensor_layout()
    spyre_layout = res_spyre.device_tensor_layout()
    assert cpu_layout == spyre_layout
    torch.allclose(res_cpu, res_spyre.to("cpu"), atol=0.1)


def test_sum_result_sparsity(tensors_2arg):
    def sum_fn(a):
        return torch.sum(a, dim=-1)

    _test_reduction_result(tensors_2arg, sum_fn)


def test_max_result_sparsity(tensors_2arg):
    def max_fn(a):
        return torch.amax(dim=-1)

    _test_reduction_result(tensors_2arg, max_fn)
