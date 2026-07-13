"""Stage 4 (conditional): name behavior components via a sparse dictionary.

Trains a lightweight TopK sparse autoencoder on activations from a
``disentangle_activations_v1`` artifact, then decomposes one or more
``em_residual_vector_v1`` component directions against the learned decoder
rows. This is the unsupervised naming fallback when supervised labels cannot
isolate a component.

The trainer uses BatchTopK-style sparsity (fixed number of active latents per
example, no L1 shrinkage on decoder norms) to avoid the complete-shrinkage
artifacts documented in Minder et al. (NeurIPS 2025).

Inputs:
  --activations          disentangle_activations_v1 artifact
  --component-vectors    one or more em_residual_vector_v1 paths (comma list)
  --position / --layer   activation site to train on

Outputs:
  --output-dir/sae.pt              trained dictionary (decoder rows = features)
  --output-dir/decomposition.json  cosine + least-squares attribution per component
  --output-dir/top_examples.jsonl  highest-activating train examples per feature

Example:
  python decompose_sae_components.py \\
    --activations outputs/disentangle/em_finance_acts.pt \\
    --component-vectors outputs/disentangle/erase/leace_mean_diff.pt,outputs/disentangle/cross/shared_component.pt \\
    --position answer_mean --layer 25 \\
    --output-dir outputs/disentangle/sae_em_l25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "model-organisms-for-EM") not in sys.path:
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))

from em_organism_dir.mech_interp.disentangle_common import load_activations  # noqa: E402
from em_organism_dir.mech_interp.vector_ops import EM_VECTOR_FORMAT, normalize_vector  # noqa: E402

SAE_FORMAT = "disentangle_topk_sae_v1"


class TopKSAE(nn.Module):
    """Minimal BatchTopK SAE: fixed-k active latents, unit-norm decoder rows."""

    def __init__(self, d_model: int, dict_size: int, top_k: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.top_k = top_k
        self.encoder = nn.Linear(d_model, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x)
        values, indices = torch.topk(pre, k=min(self.top_k, pre.shape[1]), dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, indices, F.relu(values))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--component-vectors", required=True, help="Comma-separated em_residual_vector_v1 paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--position", default="answer_mean")
    parser.add_argument("--layer", type=int, default=None, help="Layer index; default = middle layer in artifact")
    parser.add_argument("--dict-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--top-features", type=int, default=16, help="Features to report per component direction")
    parser.add_argument("--top-examples", type=int, default=8, help="Examples per reported feature")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    artifact = load_activations(args.activations)
    if args.position not in artifact["acts"]:
        raise SystemExit(f"Position {args.position!r} missing; available: {sorted(artifact['acts'])}")
    layers = sorted(artifact["acts"][args.position])
    layer = int(args.layer if args.layer is not None else layers[len(layers) // 2])
    if layer not in artifact["acts"][args.position]:
        raise SystemExit(f"Layer {layer} missing; available: {layers}")

    x = artifact["acts"][args.position][layer].float()
    d_model = x.shape[1]
    dict_size = min(args.dict_size, max(64, d_model * 8))
    sae = TopKSAE(d_model, dict_size, args.top_k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

    n = x.shape[0]
    order = torch.randperm(n)
    for epoch in range(args.epochs):
        sae.train()
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            batch = x[idx]
            recon, _ = sae(batch)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae._normalize_decoder()
            epoch_loss += float(loss.item())
            steps += 1
        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == 0:
            print(json.dumps({"epoch": epoch + 1, "mse": epoch_loss / max(1, steps)}), flush=True)

    sae.eval()
    with torch.no_grad():
        _, z_all = sae(x)
    decoder_cols = F.normalize(sae.decoder.weight.detach(), dim=0)  # (d_model, dict_size)

    component_paths = [part.strip() for part in args.component_vectors.split(",") if part.strip()]
    decomposition: dict[str, Any] = {"layer": layer, "position": args.position, "components": {}}

    for path in component_paths:
        comp_artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
        if comp_artifact.get("format") != EM_VECTOR_FORMAT:
            raise ValueError(f"{path} is not {EM_VECTOR_FORMAT}")
        vector = normalize_vector(comp_artifact["layers"][layer].float())
        cosines = decoder_cols.T @ vector  # (dict_size,)
        topk = torch.topk(cosines.abs(), k=min(args.top_features, dict_size))
        selected = topk.indices
        dec_sub = decoder_cols[:, selected]  # (d, k)
        weights = torch.linalg.lstsq(dec_sub, vector.unsqueeze(1)).solution.squeeze(1)
        feature_rows = []
        for rank, (feat_idx, cos_val) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
            activation = z_all[:, feat_idx]
            top_ex_idx = torch.topk(activation, k=min(args.top_examples, n)).indices.tolist()
            feature_rows.append(
                {
                    "rank": rank + 1,
                    "feature_id": int(feat_idx),
                    "cosine": float(cos_val),
                    "lstsq_weight": float(weights[rank].item()),
                    "mean_activation": float(activation.mean().item()),
                    "top_example_indices": top_ex_idx,
                    "top_example_prompts": [artifact["texts"]["prompt"][i][:200] for i in top_ex_idx],
                    "top_example_responses": [artifact["texts"]["response"][i][:200] for i in top_ex_idx],
                }
            )
        decomposition["components"][Path(path).stem] = {
            "path": str(path),
            "top_features": feature_rows,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sae_path = output_dir / "sae.pt"
    torch.save(
        {
            "format": SAE_FORMAT,
            "d_model": d_model,
            "dict_size": dict_size,
            "top_k": args.top_k,
            "layer": layer,
            "position": args.position,
            "activations": str(Path(args.activations)),
            "state_dict": sae.state_dict(),
        },
        sae_path,
    )
    report_path = output_dir / "decomposition.json"
    report_path.write_text(json.dumps(decomposition, indent=2), encoding="utf-8")

    examples_path = output_dir / "top_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as handle:
        for comp_name, comp_data in decomposition["components"].items():
            for feat in comp_data["top_features"]:
                handle.write(
                    json.dumps(
                        {
                            "component": comp_name,
                            "feature_id": feat["feature_id"],
                            "cosine": feat["cosine"],
                            "examples": list(
                                zip(feat["top_example_prompts"], feat["top_example_responses"], strict=True)
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(
        json.dumps(
            {
                "sae": str(sae_path),
                "decomposition": str(report_path),
                "top_examples": str(examples_path),
                "components": list(decomposition["components"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
