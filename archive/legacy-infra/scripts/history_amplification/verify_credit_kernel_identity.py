"""Numerically verify equal policies can have different one-step credit allocation."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "eval_runs/history_amplification/credit_kernel_identity.json"


def logit(theta, x):
    u, v = theta[:2], theta[2:]
    return float(np.dot(u * v, x))


def reward_gradient(theta, x):
    u, v = theta[:2], theta[2:]
    probability = 1 / (1 + np.exp(-logit(theta, x)))
    return (1 - probability) * np.concatenate([v * x, u * x])


def kernel(theta, left, right):
    u, v = theta[:2], theta[2:]
    return float(np.dot((u * u + v * v) * left, right))


def main():
    reward_state = np.array([1.0, 1.0])
    intended_state = np.array([1.0, 0.0])
    proxy_state = np.array([0.0, 1.0])
    rows = []
    for scale in [1.0, 2.0, 4.0, 8.0]:
        policies = {
            "intended_sensitive": np.array([scale, 1.0, 1 / scale, 1.0]),
            "proxy_sensitive": np.array([1.0, scale, 1.0, 1 / scale]),
        }
        for name, theta in policies.items():
            initial = {
                "reward": logit(theta, reward_state),
                "intended": logit(theta, intended_state),
                "proxy": logit(theta, proxy_state),
            }
            gradient = reward_gradient(theta, reward_state)
            probability = 1 / (1 + np.exp(-initial["reward"]))
            for learning_rate in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                updated = theta + learning_rate * gradient
                actual = {
                    "intended": logit(updated, intended_state) - initial["intended"],
                    "proxy": logit(updated, proxy_state) - initial["proxy"],
                }
                predicted = {
                    "intended": learning_rate * (1 - probability)
                    * kernel(theta, intended_state, reward_state),
                    "proxy": learning_rate * (1 - probability)
                    * kernel(theta, proxy_state, reward_state),
                }
                rows.append({
                    "scale": scale, "parameterization": name,
                    "learning_rate": learning_rate, "initial_logits": initial,
                    "actual_change": actual, "first_order_prediction": predicted,
                    "actual_proxy_minus_intended": actual["proxy"] - actual["intended"],
                    "predicted_proxy_minus_intended": predicted["proxy"] - predicted["intended"],
                    "maximum_absolute_error": max(abs(actual[key] - predicted[key]) for key in actual),
                })
    for scale in [1.0, 2.0, 4.0, 8.0]:
        initial_logits = [row["initial_logits"] for row in rows
                          if row["scale"] == scale and row["learning_rate"] == 1e-6]
        if initial_logits[0] != initial_logits[1]:
            raise AssertionError("Constructed policies are not behavior-identical")
    result = {
        "model": "Bernoulli policy with logit sum_j u_j v_j x_j",
        "update": "One gradient-ascent step on log probability of rewarded action at x=(1,1)",
        "behavioral_equivalence": "Both parameterizations have u_j v_j=1 for both features, so logits are identical for every x before the update.",
        "interpretation": "The construction verifies the local identity and feature allocation effect. It is a mathematical control, not evidence of broad empirical prevalence or a training-history mechanism.",
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "rows": len(rows),
        "max_error_lr_1e-6": max(row["maximum_absolute_error"] for row in rows
                                  if row["learning_rate"] == 1e-6),
        "scale_8_lr_1e-6": [row for row in rows if row["scale"] == 8
                              and row["learning_rate"] == 1e-6],
    }, indent=2))


if __name__ == "__main__":
    main()
