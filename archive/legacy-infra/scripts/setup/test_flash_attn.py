#!/usr/bin/env python3
"""GPU smoke test for FlashAttention2 in rllm/.venv."""
from __future__ import annotations

import sys

import torch


def main() -> int:
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("abi", torch._C._GLIBCXX_USE_CXX11_ABI)
    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 2

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print("gpu", props.name, "capability", f"{props.major}.{props.minor}")

    import flash_attn
    import flash_attn_2_cuda
    from flash_attn import flash_attn_func

    print("flash_attn", flash_attn.__version__, flash_attn.__file__)
    print("cuda_ext", flash_attn_2_cuda.__file__)

    q = torch.randn(2, 32, 8, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 32, 8, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn(2, 32, 8, 64, device=device, dtype=torch.bfloat16)
    out = flash_attn_func(q, k, v, causal=True)
    if out.shape != q.shape:
        print("unexpected FA2 output shape", tuple(out.shape), file=sys.stderr)
        return 3
    if not torch.isfinite(out).all():
        print("FA2 output has non-finite values", file=sys.stderr)
        return 4
    print("flash_attn_func ok", tuple(out.shape), float(out.float().abs().mean()))

    from transformers.utils import is_flash_attn_2_available

    if not is_flash_attn_2_available():
        print("transformers is_flash_attn_2_available is False", file=sys.stderr)
        return 5
    print("is_flash_attn_2_available True")

    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM._from_config(
        config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to(device)
    attn_impl = getattr(model.config, "_attn_implementation", None)
    print("model attn", attn_impl)
    if attn_impl != "flash_attention_2":
        print("model did not enable flash_attention_2", file=sys.stderr)
        return 6
    tokens = torch.randint(0, config.vocab_size, (2, 16), device=device)
    logits = model(tokens).logits
    if not torch.isfinite(logits).all():
        print("model forward has non-finite logits", file=sys.stderr)
        return 7
    print("qwen2 fa2 forward ok", tuple(logits.shape))
    print("FLASH_ATTN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
