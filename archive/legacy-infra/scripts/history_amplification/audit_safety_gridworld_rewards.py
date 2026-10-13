#!/usr/bin/env python3
"""Audit observed/hidden reward separation in a pinned Safety Gridworld source.

This is an environment fixture, not a language-model or training experiment.
It retains every sampled action and step reward for reproducibility.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


def rollout(gym, env_id, seed, actions, numpy_seed=None):
    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    env = gym.make(env_id, render_mode="ansi")
    env.reset(seed=seed)
    steps = []
    observed_total = 0.0
    hidden_total = 0.0
    hidden_defined = True
    for index, action in enumerate(actions):
        _, reward, terminated, truncated, info = env.step(action)
        assert reward == info["observed_reward"]
        observed_total += float(reward)
        hidden = info["hidden_reward"]
        if hidden is None:
            hidden_defined = False
        else:
            hidden_total += float(hidden)
        steps.append({
            "step": index,
            "action": action,
            "observed_reward": float(reward),
            "hidden_reward": None if hidden is None else float(hidden),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        if terminated or truncated:
            break
    env.close()
    return {
        "seed": seed,
        "actions": actions,
        "steps": steps,
        "observed_total": observed_total,
        "hidden_total": hidden_total if hidden_defined else None,
        "diverged": hidden_defined and abs(observed_total - hidden_total) > 1e-9,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=39017)
    args = parser.parse_args()

    package = args.source / "agent_system/environments/env_package/safe_gridworlds/safe-grid-gym"
    sys.path[:0] = [str(package), str(package / "ai-safety-gridworlds"),
                    str(package / "ai-safety-gridworlds/pycolab")]
    import gymnasium as gym
    import safe_grid_gym  # noqa: F401; registers environments

    env_ids = ["BoatRace-v0", "IslandNavigation-v0", "TomatoWatering-v0",
               "AbsentSupervisor-v0", "RocksDiamonds-v0"]
    rng = random.Random(args.seed)
    results = {}
    for env_id in env_ids:
        episodes = []
        for episode in range(args.episodes):
            actions = [rng.randrange(4) for _ in range(args.steps)]
            episode_seed = args.seed + episode
            episodes.append(rollout(gym, env_id, episode_seed, actions,
                                    numpy_seed=episode_seed))
        controlled_replay = rollout(gym, env_id, episodes[0]["seed"],
                                    episodes[0]["actions"], numpy_seed=episodes[0]["seed"])
        assert controlled_replay == episodes[0], env_id
        # Audit the public reset(seed=...) contract without controlling global NumPy.
        native_first = rollout(gym, env_id, args.seed, episodes[0]["actions"])
        native_replay = rollout(gym, env_id, args.seed, episodes[0]["actions"])
        divergent = [row for row in episodes if row["diverged"]]
        results[env_id] = {
            "episodes": episodes,
            "divergent_count": len(divergent),
            "divergent_rate": len(divergent) / len(episodes),
            "gym_seed_replay_exact": native_first == native_replay,
            "gym_seed_first": native_first,
            "gym_seed_replay": native_replay,
            "explicit_numpy_seed_replay_exact": True,
        }

    report = {
        "source_commit": subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "scope": (
            "Fixed-seed random environment fixture. Demonstrates reachable reward-channel "
            "divergence only; no model behavior, causal strategy label, or training effect."
        ),
        "sampling": {"episodes_per_environment": args.episodes,
                     "maximum_steps": args.steps, "action_rng_seed": args.seed},
        "environments": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {"divergent_count": value["divergent_count"],
                             "divergent_rate": value["divergent_rate"]}
                      for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
