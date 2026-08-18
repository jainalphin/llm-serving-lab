import pytest

from main import calculate_cuda_kv_cache_budget


def test_cuda_kv_budget_respects_utilization_target():
    assert calculate_cuda_kv_cache_budget(
        free_memory=800,
        total_memory=1000,
        memory_utilization=0.8,
        safety_memory=100,
    ) == 600


def test_cuda_kv_budget_respects_activation_safety_margin():
    assert calculate_cuda_kv_cache_budget(
        free_memory=800,
        total_memory=1000,
        memory_utilization=0.9,
        safety_memory=150,
    ) == 650


@pytest.mark.parametrize(
    "kwargs",
    (
        {"free_memory": 800, "total_memory": 1000, "memory_utilization": 0},
        {"free_memory": 800, "total_memory": 1000, "memory_utilization": 1.1},
        {"free_memory": 0, "total_memory": 1000},
        {"free_memory": 1100, "total_memory": 1000},
        {"free_memory": 800, "total_memory": 1000, "safety_memory": -1},
    ),
)
def test_cuda_kv_budget_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        calculate_cuda_kv_cache_budget(**kwargs)


def test_cuda_kv_budget_rejects_exhausted_memory():
    with pytest.raises(RuntimeError):
        calculate_cuda_kv_cache_budget(
            free_memory=100,
            total_memory=1000,
            safety_memory=200,
        )
