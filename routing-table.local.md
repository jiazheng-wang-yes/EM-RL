# Local Routing Extensions

These rows survive bootstrap. The shipped `.claude/skills/my-router/references/routing-table.md` is overwritten on refresh.

### Keyword-based routing

| Keywords in prompt | Skill | Source |
|---|---|---|
| "space quota", "disk quota", "scratch quota", "clean checkpoints", "optimizer states", "HF instances", "hf instances", "disk is full", "quota reminder" | `scratch-quota-cleanup` | `~/.cursor/skills/scratch-quota-cleanup` and `~/.codex/skills/scratch-quota-cleanup` |
