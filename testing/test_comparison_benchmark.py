import asyncio
import math
import time
from types import SimpleNamespace

from comparison_benchmark import (
    RequestRecord,
    arrival_offsets,
    consume_vllm_request,
    create_monitor,
    deterministic_prompts,
    request_metrics,
    summarize_scenario,
)


class DummyTokenizer:
    all_special_ids = [0, 1]

    def __len__(self):
        return 32


def test_deterministic_prompt_tokens_are_exact_and_repeatable():
    tokenizer = DummyTokenizer()
    first = deterministic_prompts(tokenizer, 9, 4, seed=123)
    second = deterministic_prompts(tokenizer, 9, 4, seed=123)
    assert first == second
    assert all(len(prompt) == 9 for prompt in first)
    assert all(token not in tokenizer.all_special_ids for prompt in first for token in prompt)
    assert len({tuple(prompt) for prompt in first}) == 4


def test_arrival_offsets_support_fixed_rate_and_burst():
    assert arrival_offsets(2.0, 4) == [0.0, 0.5, 1.0, 1.5]
    assert arrival_offsets(math.inf, 4) == [0.0, 0.0, 0.0, 0.0]


def test_common_request_metrics_and_goodput():
    records = [
        RequestRecord(
            request_index=0,
            scheduled_arrival=0.0,
            token_times=[0.1, 0.2, 0.3],
            token_ids=[4, 5, 6],
        ),
        RequestRecord(
            request_index=1,
            scheduled_arrival=0.5,
            token_times=[0.7, 0.9, 1.1],
            token_ids=[7, 8, 9],
        ),
    ]
    first = request_metrics(records[0])
    assert math.isclose(first["ttft"], 0.1)
    assert math.isclose(first["tpot"], 0.1)
    assert math.isclose(first["e2e"], 0.3)

    summary = summarize_scenario(
        engine="test",
        request_rate=2.0,
        records=records,
        duration=1.1,
        output_length=3,
        telemetry=None,
        ttft_slo_ms=150,
        tpot_slo_ms=150,
        e2e_slo_ms=500,
    )
    assert summary["successful_requests"] == 2
    assert not summary["failed_requests"]
    assert math.isclose(summary["goodput_requests_per_second"], 1 / 1.1)
    assert summary["generated_tokens"] == 6


def test_vllm_delta_stream_records_each_generated_token_once():
    class FakeVLLMEngine:
        async def generate(self, **kwargs):
            assert kwargs["prompt"] == {"prompt_token_ids": [10, 11]}
            for token_ids in ([20], [21, 22]):
                yield SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=token_ids)]
                )

    record = RequestRecord(request_index=0, scheduled_arrival=0.0)
    asyncio.run(
        consume_vllm_request(
            FakeVLLMEngine(),
            sampling_params=object(),
            prompt=[10, 11],
            record=record,
            benchmark_start=time.perf_counter(),
        )
    )
    assert record.error is None
    assert record.token_ids == [20, 21, 22]
    assert len(record.token_times) == 3


def test_gpu_monitor_targets_only_the_first_cuda_visible_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setattr("comparison_benchmark.torch.cuda.is_available", lambda: True)
    monitor = create_monitor(SimpleNamespace(telemetry_interval_ms=200))
    assert monitor.gpu_id == "1"
