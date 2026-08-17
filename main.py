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


def _load_model(model_name, device, gpt2_model_id, distilgpt2_model_id):
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
        kv_cache_memory = 32 * 1024 * 1024
    elif model_name in (DISTILGPT2_MODEL, GPT2_MODEL):
        model_id = (
            distilgpt2_model_id
            if model_name == DISTILGPT2_MODEL
            else gpt2_model_id
        )
        model, tokenizer = load_gpt2_pretrained(model_id)
        config = model.config
        kv_cache_memory = 256 * 1024 * 1024
    else:
        choices = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {choices}")

    return model.to(device).eval(), tokenizer, config, kv_cache_memory


def build_scheduler(
    model_name=REFERENCE_MODEL,
    gpt2_model_id=DEFAULT_GPT2_MODEL_ID,
    distilgpt2_model_id=DEFAULT_DISTILGPT2_MODEL_ID,
    scheduling_strategy=ORCA_STRATEGY,
    prefill_chunk_size=16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config, kv_cache_memory = _load_model(
        model_name,
        device,
        gpt2_model_id,
        distilgpt2_model_id,
    )
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
        max_batch_size=4,
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
    args = parser.parse_args()

    scheduler = build_scheduler(
        args.model,
        scheduling_strategy=args.strategy,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    first_request = scheduler.add_request("Paged attention", max_new_tokens=8)
    second_request = scheduler.add_request("Orca", max_new_tokens=8)
    results = scheduler.run_until_complete()

    print(f"request {first_request}: {results[first_request]!r}")
    print(f"request {second_request}: {results[second_request]!r}")


if __name__ == "__main__":
    main()
