"""CPU tests for the in-memory CRD snapshot used during training."""
from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch import nn

import stage6b_train as trainer


class _Layer(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, hidden):
        return hidden * self.weight


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(1.0) for _ in range(21)])


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Block()
        self.model.layers[8].weight.data.fill_(2.0)
        self.model.layers[20].weight.data.fill_(1.5)


class _NoopHooks:
    def __init__(self, _model):
        pass

    def resid(self, *_args, **_kwargs):
        pass

    def clear(self):
        pass


class RouteSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.model = _ToyModel()
        self.initial = {name: value.detach().clone()
                        for name, value in self.model.named_parameters()}
        self.control = {
            name: torch.tensor(0.5 if name == "model.layers.20.weight" else 0.0)
            for name, _ in self.model.named_parameters()
        }
        self.sequences = [
            {"kind": "align", "example_id": "p1"},
            {"kind": "mis", "example_id": "p1"},
        ]
        self.batch = {
            "seqs": self.sequences,
            "masks": {"resp": torch.ones((2, 1), dtype=torch.bool)},
        }

    def _run_with_patches(self, *, score_batch=None):
        def score(model, batch):
            hidden = torch.ones((len(batch["seqs"]), 1, 2))
            model.model.layers[20](hidden)  # trigger the carrier-layer hook
            return {"direct_weight": float(model.model.layers[8].weight.detach())}

        def per_sequence(_batch, scored, _refs):
            return [
                {"kind": "align", "example_id": "p1", "lp_mean": 0.0},
                {"kind": "mis", "example_id": "p1", "lp_mean": scored["direct_weight"]},
            ]

        def persona_scalars(hidden, _batch, _carrier, _direction):
            value = float(hidden.mean())
            return {("align", "p1"): value, ("mis", "p1"): value}

        stack = [
            patch.object(trainer, "load_sequences", return_value=self.sequences),
            patch.object(trainer, "make_batches", return_value=[self.batch]),
            patch.object(trainer, "load_state_dict_cpu", return_value=self.control),
            patch.object(trainer, "score_batch", side_effect=score_batch or score),
            patch.object(trainer, "per_sequence", side_effect=per_sequence),
            patch.object(trainer, "Hooks", _NoopHooks),
            patch.object(trainer, "strict_ids", return_value={"p1"}),
            patch("stage6b_route_check.persona_scalars", side_effect=persona_scalars),
            patch.object(torch.cuda, "empty_cache"),
        ]
        for item in stack:
            item.start()
        return stack

    def _restore_patches(self, stack):
        for item in reversed(stack):
            item.stop()

    def test_crd_snapshot_reports_region_effect_and_restores_training_weights(self):
        stack = self._run_with_patches()
        try:
            result = trainer.direct_route_snapshot_assay(
                self.model, tok=SimpleNamespace(pad_token_id=0), model_key="qwen2_5_7b", device="cpu",
                direct_names={"model.layers.8.weight"},
                carrier=torch.tensor([[1.0], [0.0]]), direction=torch.tensor([1.0]),
                persona_m={"p1": 3.0},
            )
        finally:
            self._restore_patches(stack)

        self.assertEqual(result["S_C"], 0.0)
        self.assertEqual(result["S_C_per_prompt"], {"p1": 0.0})
        self.assertEqual(result["strict50_S_C_per_prompt"], {"p1": 0.0})
        self.assertEqual(result["TE"]["point"], 2.0)
        self.assertEqual(result["DE"]["point"], 2.0)
        self.assertEqual(result["absolute_mediated_effect"], 0.0)
        self.assertEqual(result["clamp_identity_max_abs"], 0.0)
        self.assertTrue(result["MF"] is None or result["MF"] == 0.0)
        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(value.detach(), self.initial[name]), name)

    def test_crd_snapshot_restores_training_weights_after_scoring_error(self):
        def fail(*_args, **_kwargs):
            raise RuntimeError("synthetic scorer failure")

        stack = self._run_with_patches(score_batch=fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "synthetic scorer failure"):
                trainer.direct_route_snapshot_assay(
                    self.model, tok=SimpleNamespace(pad_token_id=0), model_key="qwen2_5_7b", device="cpu",
                    direct_names={"model.layers.8.weight"},
                    carrier=torch.tensor([[1.0], [0.0]]), direction=torch.tensor([1.0]),
                    persona_m={"p1": 3.0},
                )
        finally:
            self._restore_patches(stack)

        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(value.detach(), self.initial[name]), name)


if __name__ == "__main__":
    unittest.main()
