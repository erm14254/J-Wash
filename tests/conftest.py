import pytest
import torch

@pytest.fixture(scope="module")
def tiny():
    import transformers
    from packaging.version import Version
    assert Version("5.5") <= Version(transformers.__version__) < Version("5.15"), (
        "Qwen3.5-MoE integration requires transformers>=5.5,<5.15; installed " + transformers.__version__)
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=67, hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        linear_conv_kernel_dim=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, moe_intermediate_size=16,
        shared_expert_intermediate_size=12, num_experts=4, num_experts_per_tok=2,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1],
                         "mrope_interleaved": True})
    cfg._attn_implementation = "eager"; cfg._experts_implementation = "eager"
    torch.manual_seed(7)
    model = Qwen3_5MoeForCausalLM(cfg).float().eval()
    return model
