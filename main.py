import torch

from src.model.kv_manager import KVCacheManager
from src.model.paged_attention import PagedAttention
from src.model.paged_decoder import PagedDecoderLM, TransformerConfig
from src.model.tokenizer import ByteTokenizer
from src.scheduler.orca_scheduler import ContinuousBatchScheduler


def build_scheduler():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = PagedDecoderLM(config).to(device).eval()
    kv_manager = KVCacheManager(
        block_size=16,
        total_memory=32 * 1024 * 1024,
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
    )


def main():
    scheduler = build_scheduler()
    first_request = scheduler.add_request("Paged attention", max_new_tokens=8)
    second_request = scheduler.add_request("Orca", max_new_tokens=8)
    results = scheduler.run_until_complete()

    print(f"request {first_request}: {results[first_request]!r}")
    print(f"request {second_request}: {results[second_request]!r}")


if __name__ == "__main__":
    main()
