import pytest
import torch

from nos.common.system import (
    docker_compose_command,
    get_nvidia_smi,
    get_system_info,
    get_torch_cuda_info,
    get_torch_info,
    get_torch_mps_info,
    has_docker,
    has_docker_compose,
    has_gpu,
    has_nvidia_docker,
    has_nvidia_docker_runtime_enabled,
    is_apple,
    is_apple_silicon,
    is_inside_docker,
)
from nos.test.utils import skip_if_no_torch_cuda


def test_system_info():
    info = get_system_info(docker=True, gpu=False)
    assert "system" in info
    assert "cpu" in info
    assert "memory" in info
    assert "docker" in info
    assert "gpu" not in info

    # if not is_inside_docker():
    #     assert info["docker"]["version"] is not None
    #     assert info["docker"]["sdk_version"] is not None
    #     assert info["docker"]["compose_version"] is not None

    _ = has_gpu()
    _ = has_docker()
    _ = has_docker_compose
    _ = is_inside_docker()
    _ = is_apple()
    _ = is_apple_silicon()
    _ = has_nvidia_docker()
    _ = has_nvidia_docker_runtime_enabled()
    assert docker_compose_command() in (None, "docker-compose", "docker compose")


@skip_if_no_torch_cuda
def test_system_info_with_gpu():
    info = get_system_info(docker=True, gpu=True)
    assert "docker" in info
    assert "gpu" in info
    assert len(info["gpu"]["devices"]) > 0


@pytest.mark.skipif(torch.cuda.is_available(), reason="Skipping CPU-only tests when GPU is available.")
def test_system_utilities_cpu():
    # Check if within docker with psutil
    # if not is_inside_docker():
    #     assert has_docker(), "Docker not installed."
    #     assert get_docker_info() is not None

    assert get_torch_info() is not None, "torch unavailable."
    assert get_torch_cuda_info() is None, "No GPU detected via torch.cuda."
    assert get_torch_mps_info() is None


@skip_if_no_torch_cuda
def test_system_utilities_gpu():
    # if not is_inside_docker():
    #     assert has_docker(), "Docker not installed."
    #     assert get_docker_info() is not None
    #     assert has_nvidia_docker(), "NVIDIA Docker not installed."
    #     assert has_nvidia_docker_runtime_enabled(), "No GPU detected within NVIDIA Docker."

    assert has_gpu(), "No GPU detected."
    assert get_nvidia_smi() is not None

    assert get_torch_info() is not None, "torch unavailable."
    assert get_torch_cuda_info() is not None, "No GPU detected via torch.cuda."
