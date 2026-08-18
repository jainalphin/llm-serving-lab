import argparse

import torch

from src.model.gpt2 import (
    DEFAULT_DISTILGPT2_MODEL_ID,
    DEFAULT_GPT2_MODEL_ID,
    load_gpt2_pretrained,
)
from src.model.kv_manager import KVCacheManager
from src.model.paged_attention import PagedAttention
from src.model.paged_decoder import PagedDecoderLM, TransformerConfig
from src.model.tokenizer import ByteTokenizer
from src.scheduler.orca_scheduler import (
    ORCA_STRATEGY,
    SUPPORTED_SCHEDULING_STRATEGIES,
    ContinuousBatchScheduler,
)


REFERENCE_MODEL = "reference"
DISTILGPT2_MODEL = "distilgpt2"
GPT2_MODEL = "gpt2"
SUPPORTED_MODELS = (REFERENCE_MODEL, DISTILGPT2_MODEL, GPT2_MODEL)
SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")


def _resolve_execution_dtype(execution_dtype, device):
    if execution_dtype not in SUPPORTED_DTYPES:
        choices = ", ".join(SUPPORTED_DTYPES)
        raise ValueError(f"Unknown dtype '{execution_dtype}'. Choose one of: {choices}")

    resolved = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[execution_dtype]
    if device.type == "cpu" and resolved == torch.float16:
        raise ValueError("float16 execution requires a CUDA device")
    if (
        device.type == "cuda"
        and resolved == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("This CUDA device does not support bfloat16 execution")
    return resolved


def _load_model(
    model_name,
    device,
    execution_dtype,
    gpt2_model_id,
    distilgpt2_model_id,
):
    if model_name == REFERENCE_MODEL:
        tokenizer = ByteTokenizer()
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            mlp_hidden_size=256,
            max_sequence_length=128,
        )
        model = PagedDecoderLM(config)
        default_kv_cache_memory = 32 * 1024 * 1024
    elif model_name in (DISTILGPT2_MODEL, GPT2_MODEL):
        model_id = (
            distilgpt2_model_id
            if model_name == DISTILGPT2_MODEL
            else gpt2_model_id
        )
        model, tokenizer = load_gpt2_pretrained(model_id)
        config = model.config
        default_kv_cache_memory = (
            1024 * 1024 * 1024 if device.type == "cuda" else 256 * 1024 * 1024
        )
    else:
        choices = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {choices}")

    model = model.to(device=device, dtype=execution_dtype).eval()
    return model, tokenizer, config, default_kv_cache_memory


def build_scheduler(
    model_name=REFERENCE_MODEL,
    gpt2_model_id=DEFAULT_GPT2_MODEL_ID,
    distilgpt2_model_id=DEFAULT_DISTILGPT2_MODEL_ID,
    scheduling_strategy=ORCA_STRATEGY,
    prefill_chunk_size=16,
    max_batch_size=None,
    kv_cache_memory_mb=None,
    execution_dtype="float32",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_dtype = _resolve_execution_dtype(execution_dtype, device)
    model, tokenizer, config, default_kv_cache_memory = _load_model(
        model_name,
        device,
        resolved_dtype,
        gpt2_model_id,
        distilgpt2_model_id,
    )
    if max_batch_size is None:
        max_batch_size = 32 if device.type == "cuda" else 4
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if kv_cache_memory_mb is None:
        kv_cache_memory = default_kv_cache_memory
    else:
        if kv_cache_memory_mb <= 0:
            raise ValueError("kv_cache_memory_mb must be positive")
        kv_cache_memory = kv_cache_memory_mb * 1024 * 1024

    kv_manager = KVCacheManager(
        block_size=16,
        total_memory=kv_cache_memory,
        tensor_dtype=next(model.parameters()).dtype,
        device=device,
        num_layers=config.num_layers,
        num_kv_heads=config.num_heads,
        head_dim=config.head_dim,
    )
    paged_attention = PagedAttention(
        kv_manager=kv_manager,
    )
    return ContinuousBatchScheduler(
        model_engine=model,
        max_batch_size=max_batch_size,
        tokenizer=tokenizer,
        kv_manager=kv_manager,
        paged_attn_manager=paged_attention,
        scheduling_strategy=scheduling_strategy,
        prefill_chunk_size=prefill_chunk_size,
    )


def main():
    parser = argparse.ArgumentParser(description="Run the PagedServe inference engine")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=REFERENCE_MODEL,
        help="model backend to run",
    )
    parser.add_argument(
        "--strategy",
        choices=SUPPORTED_SCHEDULING_STRATEGIES,
        default=ORCA_STRATEGY,
        help="request scheduling strategy",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=16,
        help="prompt tokens per Sarathi prefill chunk",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        help="maximum requests per iteration (default: 32 on CUDA, 4 on CPU)",
    )
    parser.add_argument(
        "--kv-cache-memory-mb",
        type=int,
        help="KV-cache memory budget in MiB",
    )
    parser.add_argument(
        "--dtype",
        choices=SUPPORTED_DTYPES,
        default="float32",
        help="execution dtype; float16/bfloat16 are explicit optimized runs",
    )
    args = parser.parse_args()

    scheduler = build_scheduler(
        args.model,
        scheduling_strategy=args.strategy,
        prefill_chunk_size=args.prefill_chunk_size,
        max_batch_size=args.max_batch_size,
        kv_cache_memory_mb=args.kv_cache_memory_mb,
        execution_dtype=args.dtype,
    )
    first_request = scheduler.add_request("Paged attention", max_new_tokens=8)
    second_request = scheduler.add_request("Orca", max_new_tokens=8)
    results = scheduler.run_until_complete()

    print(f"request {first_request}: {results[first_request]!r}")
    print(f"request {second_request}: {results[second_request]!r}")


if __name__ == "__main__":
    main()
