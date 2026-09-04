# figures/

Generated plots, one subdirectory per area:

```
figures/<area>/<figure-set>/<name>.{png,pdf}
```

Regenerable by definition -- every figure must be reproducible by rerunning the
script that made it, so nothing here is precious. If a figure cannot be
regenerated from a script plus data in `logs/` or `eval_runs/`, that is a bug in
the script, not a reason to protect the file.

Name a figure set for what it shows. Append a date only when you need to keep an
older version alongside a new one (`format_reward_ablation_20260903`).

## In progress elsewhere

`logs/countdown_code/plots/` still holds `expansion_matrix/` and `model_families/`.
Another agent's expansion experiments write there via the default `--out-dir` of
`plot_expansion_matrix.py` and `plot_model_family_signals.py`, so those two were
deliberately left in place rather than moved here. Move them, and update the two
script defaults, once that work is finished.
