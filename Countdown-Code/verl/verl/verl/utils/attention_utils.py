# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _torch_unpad_input(hidden_states, attention_mask):
    """Pure-torch unpad compatible with flash_attn.bert_padding.unpad_input."""
    import torch

    # hidden_states: (batch, seqlen, ...)
    assert hidden_states.dim() >= 2
    batch, seqlen = hidden_states.shape[:2]
    flat_mask = attention_mask.bool().reshape(-1)
    indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
    hidden_states = hidden_states.reshape(batch * seqlen, *hidden_states.shape[2:])[indices]
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    return hidden_states, indices, cu_seqlens, max_seqlen


def _flash_attn_unpad_is_sane():
    import torch
    try:
        from flash_attn.bert_padding import unpad_input as _fa_unpad
        ids = torch.arange(10).view(2, 5)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        out, *_ = _fa_unpad(ids.unsqueeze(-1), mask)
        return out.dim() >= 2 and out.shape[0] == 5 and out.shape[-1] == 1
    except Exception:
        return False


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_cuda_available, is_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    index_first_axis = pad_input = rearrange = unpad_input = None
    if is_cuda_available:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
            if not _flash_attn_unpad_is_sane():
                unpad_input = _torch_unpad_input
        except Exception:
            index_first_axis = pad_input = rearrange = unpad_input = None
    if unpad_input is None and is_npu_available:
        from verl.utils.npu_utils import index_first_axis, pad_input, rearrange, unpad_input
    if unpad_input is None:
        # CPU / broken-flash fallback: use einops rearrange + torch unpad; pad/index from flash if present else minimal stubs.
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange
        except Exception:
            from einops import rearrange as _einops_rearrange

            def rearrange(*args, **kwargs):
                return _einops_rearrange(*args, **kwargs)

            def index_first_axis(tensor, indices):
                return tensor[indices]

            def pad_input(hidden_states, indices, batch, seqlen):
                import torch
                output = hidden_states.new_zeros(batch * seqlen, *hidden_states.shape[1:])
                output[indices] = hidden_states
                return output.view(batch, seqlen, *hidden_states.shape[1:])

        unpad_input = _torch_unpad_input

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    """
    Unified entry point for `index_first_axis` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.index_first_axis`
      - On NPU: `transformers.integrations.npu_flash_attention.index_first_axis`
        (falls back to `transformers.modeling_flash_attention_utils._index_first_axis`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    """
    Unified entry point for `pad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.pad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.pad_input`
        (falls back to `transformers.modeling_flash_attention_utils._pad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    """
    Unified entry point for `rearrange` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.rearrange`
      - On NPU: `transformers.integrations.npu_flash_attention.rearrange`
        (falls back to `einops.rearrange` if no dedicated NPU implementation exists).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    """
    Unified entry point for `unpad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.unpad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.unpad_input`
        (falls back to `transformers.modeling_flash_attention_utils._unpad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
