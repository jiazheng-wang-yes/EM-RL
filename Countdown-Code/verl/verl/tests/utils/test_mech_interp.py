import torch
from torch import nn

from verl.utils.mech_interp import (
    EMVectorRuntime,
    ResidualVectorHookContext,
    normalize_vector,
    project_onto_vector,
    remove_projection,
)


class TinyBlock(nn.Module):
    def forward(self, hidden):
        return hidden + 1


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, hidden):
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


def test_normalize_and_remove_projection():
    vector = normalize_vector(torch.tensor([3.0, 4.0]))
    assert torch.allclose(vector.norm(), torch.tensor(1.0))

    hidden = torch.tensor([[[3.0, 4.0], [4.0, -3.0]]])
    removed = remove_projection(hidden, vector)
    projection = project_onto_vector(removed, vector)
    assert torch.allclose(projection, torch.zeros_like(projection), atol=1e-6)


def test_hook_records_and_cleans_up():
    model = TinyModel()
    runtime = EMVectorRuntime(vectors={0: normalize_vector(torch.tensor([1.0, 0.0]))}, layers=[0])
    layer = model.model.layers[0]
    assert len(layer._forward_hooks) == 0
    with ResidualVectorHookContext(model, runtime) as ctx:
        model(torch.zeros(2, 3, 2))
        assert 0 in ctx.projections
        assert ctx.projections[0].shape == (2, 3)
        assert len(layer._forward_hooks) == 1
    assert len(layer._forward_hooks) == 0


def test_projection_summary_shapes_and_masks():
    runtime = EMVectorRuntime(vectors={0: normalize_vector(torch.tensor([1.0, 0.0]))}, layers=[0])
    projections = {0: torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 4.0, 6.0]])}
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    summary = runtime.summarize_full_sequence(projections, response_length=2, attention_mask=attention_mask)
    assert set(summary) == {"em_projection_l0_prompt_last", "em_projection_l0_response_mean"}
    assert torch.allclose(summary["em_projection_l0_prompt_last"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(summary["em_projection_l0_response_mean"], torch.tensor([2.5, 5.0]))


def test_add_and_ablate_intervention_modes():
    hidden = torch.zeros(1, 1, 2)
    model = TinyModel()
    add_runtime = EMVectorRuntime(
        vectors={0: normalize_vector(torch.tensor([1.0, 0.0]))},
        layers=[0],
        intervention_mode="add",
        scale=2.0,
    )
    with add_runtime.hook_context(model):
        out = model(hidden)
    assert torch.allclose(out[..., 0], torch.full((1, 1), 4.0))

    ablate_runtime = EMVectorRuntime(
        vectors={0: normalize_vector(torch.tensor([1.0, 0.0]))},
        layers=[0],
        intervention_mode="ablate",
    )
    with ablate_runtime.hook_context(model):
        out = model(torch.ones(1, 1, 2))
    assert torch.allclose(out[..., 0], torch.ones(1, 1))
