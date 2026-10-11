# shared

Modules that the other folders import. `activation_route.py` and `side_path.py` also run on their own as analyses (see their docstrings).

| Module | Contents |
|---|---|
| `common.py` | Model settings (`MODEL_SPECS`), repository and data paths, chat rendering, hooks, batching and pair scoring |
| `eval_definitions.py` | Judge models and revisions, judge prompts, prompt rendering |
| `factors.py` | Low-rank factors of the weight update and frozen-assay bookkeeping |
| `side_path.py` | Float32 side paths: low-rank edits added next to the unchanged bf16 weights, and a rank sweep of such edits |
| `activation_route.py` | Activation-route analysis of the weight update; its `PROTECTED_SUFFIXES` (the seven edited matrix types) is used by training |
| `judge_runtime.py` | Judge model specifications and the pinned judge runtime |
| `plot_palette.py` | Plot colors |
| `_paths.py` | Puts every code folder on `sys.path` (the same file is in each code folder) |
