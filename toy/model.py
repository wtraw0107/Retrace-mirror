"""Toy GPT: the Pythia architecture (GPT-NeoX), scaled down.

Using GPTNeoXConfig rather than a hand-written Transformer means the same
attribution code can later load real Pythia checkpoints (Section 7). Every
field below except the size knobs is left at the Pythia default.

Parameter count, per layer, with d = hidden_size and intermediate = 4d:
    attention  qkv: d*3d + 3d     out proj: d*d + d
    mlp        up:  d*4d + 4d     down:     4d*d + d
    2 layernorms:   4d
    total:          12 d^2 + 13 d
plus embed_in (V*d), embed_out (V*d, untied), and a final layernorm (2d).
Rotary position embeddings have no parameters.
"""

import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM


def build_config(vocab_size, bos_token_id, eos_token_id,
                 n_layers=4, hidden_size=256, n_heads=4, context_length=64):
    return GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        intermediate_size=4 * hidden_size,
        max_position_embeddings=context_length,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        # No dropout: we want to memorize, and TRAK needs deterministic gradients.
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=False,
    )


def build_model(config, seed):
    torch.manual_seed(seed)  # init depends only on the seed, not on data loading
    # Eager attention: speed is irrelevant at this size, the math stays explicit,
    # and it composes cleanly with torch.func (vmap/grad) for per-example gradients.
    return GPTNeoXForCausalLM._from_config(config, attn_implementation="eager")


def param_breakdown(model):
    """Return {group: n_params} and check the 12d^2 + 13d formula against the real model."""
    cfg = model.config
    d, L = cfg.hidden_size, cfg.num_hidden_layers
    counts = {
        "embed_in": model.gpt_neox.embed_in.weight.numel(),
        "layers": sum(p.numel() for p in model.gpt_neox.layers.parameters()),
        "final_layer_norm": sum(p.numel() for p in model.gpt_neox.final_layer_norm.parameters()),
        # Pythia calls this embed_out; transformers v5 renamed it lm_head. This accessor works for both.
        "embed_out": model.get_output_embeddings().weight.numel(),
    }
    counts["total"] = sum(p.numel() for p in model.parameters())
    counts["non_embedding"] = counts["layers"] + counts["final_layer_norm"]
    assert counts["layers"] == L * (12 * d * d + 13 * d), counts
    assert counts["total"] == sum(v for k, v in counts.items() if k not in ("total", "non_embedding"))
    return counts
