from nos.test.utils import benchmark, requires_torch_cuda


@requires_torch_cuda
def test_torch_cuda():
    """This test is skipped when torch.cuda.is_available() is False."""
    import torch

    assert torch.cuda.is_available()


@benchmark
def test_benchmark():
    """This test is demarkated as a benchmark (slow), and will only run when NOS_TEST_BENCHMARK=1."""
    assert True