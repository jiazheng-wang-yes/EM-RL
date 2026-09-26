# Stage 6B Calibration Status

## Current Qwen2.5-7B ACL calibration

Calibration job `1865759` and its `afterany` monitor `1865781` both completed
with exit 0. The validated 16-step report is ready for review. Selected
`lambda_D = 0.75` (0.966 retained harmful-task learning) and
`lambda_P = 0.0001455827` (46.42% persona-drift reduction); both meet the
prespecified calibration constraints. No broad EM was evaluated and no
184-step training was submitted.

The eight subruns have complete step 0–16 metrics and matching permanent
training-metrics journals; all nine source hashes match. The slowdown row's
zero benign training-loss field is a placeholder because that condition does
not compute a benign pass; it is reported as not measured. See the [full
calibration report](../../eval_runs/persona_control_acl/calibration/qwen2_5_7b/seed_61791/workflows/pcacl_qwen2_5_7b_seed61791_20260923T140323342306186_202668/calibration_attempt_a1_job1865781.md)
and [run record](../plans/persona-control-acl-run-record.md). The calibration
check is now waiting for review before any 184-step training.

## Earlier non-ACL pilot result

- Calibration job: `1835133`
- Terminal state: `FAILED`
- Exit code: `0:53`
- Observed at: `2026-09-22T16:04:33.572312+00:00`

## Failure diagnostic

The dependent monitor ran after a non-complete terminal state. No retry or full experiment was submitted.

```text
[missing log: /net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/pc6b_calibrate_1835133.err]
```

No 184-step experiment was launched by this monitor.
