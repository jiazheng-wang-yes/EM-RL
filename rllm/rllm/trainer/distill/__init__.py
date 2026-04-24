"""Shared distillation utilities for cross-tokenizer teacher-student alignment."""

from rllm.trainer.distill.advantage import (
    compute_distill_reverse_kl,
    compute_step_distill_advantage,
    discounted_future_sum,
)
from rllm.trainer.distill.alignment import align_teacher_logprobs, visualize_alignment
from rllm.trainer.distill.gkd import build_gkd_target_ids, replace_batch_row_with_gkd_target

__all__ = [
    "align_teacher_logprobs",
    "build_gkd_target_ids",
    "compute_distill_reverse_kl",
    "compute_step_distill_advantage",
    "discounted_future_sum",
    "replace_batch_row_with_gkd_target",
    "visualize_alignment",
]
