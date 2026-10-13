"""Opt-in exact-batch capture for bounded, single-update experiments."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from verl import DataProto


def fingerprint(batch):
    h = hashlib.sha256()
    for key, value in sorted(batch.batch.items()):
        value = value.detach().cpu().contiguous()
        h.update(str((key, str(value.dtype), tuple(value.shape))).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    h.update(json.dumps(batch.non_tensor_batch.get("uid", []).tolist()
                        if isinstance(batch.non_tensor_batch.get("uid"), np.ndarray)
                        else batch.non_tensor_batch.get("uid", []), sort_keys=True).encode())
    return h.hexdigest()


def prepare(batch, config, step):
    probe = config.get("probe", {})
    if not probe.get("capture_update_batches", False):
        return batch, None
    if step != 1 or int(config.trainer.total_training_steps) != 2:
        raise ValueError("Exact replay diagnostic requires exactly one optimizer update")
    root = Path(config.trainer.default_local_dir) / "diagnostic_batches"
    root.mkdir(parents=True, exist_ok=True)
    replay = probe.get("replay_batch", "")
    if replay:
        # This file is a trusted, locally captured DataProto, never user/model text.
        source = Path(replay).resolve()
        allowed = Path(config.trainer.default_local_dir).resolve().parent
        if not source.is_relative_to(allowed) or source.name != "before.pkl":
            raise ValueError("Replay must reference a captured batch in this experiment's rollout area")
        batch = DataProto.load_from_disk(source)
        metadata = json.loads(source.with_name("before.json").read_text())
        if metadata["model"] != str(config.actor_rollout_ref.model.path):
            raise ValueError("Replay initialization differs from the capture model")
        if fingerprint(batch) != metadata["batch_sha256"]:
            raise ValueError("Replay tensor fingerprint changed")
    snapshot = copy.deepcopy(batch).to("cpu")
    path = root / "before.pkl"
    if path.exists():
        raise FileExistsError(path)
    snapshot.save_to_disk(path)
    metadata = {
        "model": str(config.actor_rollout_ref.model.path),
        "batch_sha256": fingerprint(snapshot), "replay_source": str(replay),
        "tensor_shapes": {k: list(v.shape) for k, v in snapshot.batch.items()},
        "tensor_dtypes": {k: str(v.dtype) for k, v in snapshot.batch.items()},
        "lr": float(config.actor_rollout_ref.actor.optim.lr),
        "step": step,
    }
    (root / "before.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return batch, root


def finish(batch, post, root, baseline=None):
    """Save actual actor log probabilities before rollout-engine synchronization."""
    copy.deepcopy(post).to("cpu").save_to_disk(root / "after_actor.pkl")
    mask = batch.batch["response_mask"].bool()
    reference = baseline if baseline is not None else batch
    delta = (post.batch["old_log_probs"] - reference.batch["old_log_probs"])[mask]
    values = {
        "response_tokens": int(mask.sum()),
        "mean_logprob_change": float(delta.mean()),
        "max_abs_logprob_change": float(delta.abs().max()),
        "mean_squared_logprob_change": float((delta ** 2).mean()),
        "finite": bool(torch.isfinite(delta).all()),
        "baseline": "fresh_actor_after_batch_preparation" if baseline is not None else "stored_old_log_probs",
        "interpretation": "Fixed-prefix actor diagnostic, not full-distribution KL or a rollout-engine equivalence check.",
    }
    (root / "after_actor.json").write_text(json.dumps(values, indent=2) + "\n")
