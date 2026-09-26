"""Focused regression check for exact layer/key selection in the bridge graft."""
from stage6b_generation_bridge import select_graft_keys


def test_selects_only_seven_matrix_weights_in_layers_8_through_19():
    expected = [f"model.layers.{i}.{attn}.{name}.weight"
                for i in range(8, 20)
                for attn, names in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                                    ("mlp", ("gate_proj", "up_proj", "down_proj")))
                for name in names]
    keys = expected + [
        "model.layers.7.self_attn.q_proj.weight",
        "model.layers.20.self_attn.q_proj.weight",
        "model.layers.10.self_attn.rotary_emb.weight",
        "model.layers.18.self_attn.q_proj.bias",
        "model.layers.1.self_attn.q_proj.weight",
    ]
    selected = select_graft_keys(keys)
    assert len(selected) == 84
    assert set(selected) == set(expected)


if __name__ == "__main__":
    test_selects_only_seven_matrix_weights_in_layers_8_through_19()
    print("exact graft-key selection check passed (84 keys)")
