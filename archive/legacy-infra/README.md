# Legacy infrastructure

Code and outputs from closed research lines. The persona-control line does not use any of it. Moved here unchanged on 2026-10-12. To reuse a part, move it back to the repository root, because scripts refer to paths relative to the root.

| Folder | What it is | Line |
|---|---|---|
| `Countdown-Code/` | Countdown coding task with a hackable reward (verl) | reward hacking after SFT |
| `HarmBench/`, `strong_reject/`, `BeHonest/`, `agentic-misalignment/` | Third-party safety and honesty benchmarks | reward hacking after SFT, EM causal |
| `lm-evaluation-harness/` | Third-party capability evaluation (own git checkout) | reward hacking after SFT |
| `data_generation/` | SFT data distillation (DeepSeek, reward-hack descriptions) | reward hacking after SFT |
| `scripts/` | All former `scripts/` subfolders except `scripts/paper/`: training, RL, evaluation, expansion matrix, EM causal, targeted transfer, setup | all closed lines |
| `outputs/`, `runs/`, `wandb/`, `agent_outputs/` | Old run outputs, history-amplification runs, W&B logs, agentic eval configs | closed lines |
| `materialize.py`, `equation.txt`, `output.txt`, `out.txt` | Loose files from the root | — |

Findings for these lines are in `docs/archive/`. Paths written there, such as `scripts/countdown_code/...`, now live under `archive/legacy-infra/`.

Kept at the root because current code uses them: `rllm/` (its `.venv` runs the vLLM judges), `benchmarks/` (IFBench, IHEval, MMLU), `model-organisms-for-EM/` (training and evaluation data), `scripts/paper/`.
