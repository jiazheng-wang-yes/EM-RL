"""Compatibility shims loaded by Python worker processes for this probe."""

try:
    from transformers import PreTrainedTokenizerBase
except Exception:
    PreTrainedTokenizerBase = None


if PreTrainedTokenizerBase is not None:
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):

        @property
        def all_special_tokens_extended(self):
            return list(getattr(self, "all_special_tokens", []))

        PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended

    if not hasattr(PreTrainedTokenizerBase, "special_tokens_map_extended"):

        @property
        def special_tokens_map_extended(self):
            return dict(getattr(self, "special_tokens_map", {}))

        PreTrainedTokenizerBase.special_tokens_map_extended = special_tokens_map_extended
