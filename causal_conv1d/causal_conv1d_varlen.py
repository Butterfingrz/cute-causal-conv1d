import torch
from torch import Tensor

from causal_conv1d.cutedsl_kernels import causal_conv1d_varlen_states_cutedsl


def causal_conv1d_varlen_states(
    x: Tensor, cu_seqlens: Tensor, state_len: int
) -> Tensor:
    """Gather the last ``state_len`` tokens of each packed sequence with CuTe DSL."""
    return causal_conv1d_varlen_states_cutedsl(x, cu_seqlens.contiguous(), state_len)


def causal_conv1d_varlen_states_ref(
    x: Tensor, cu_seqlens: Tensor, state_len: int
) -> Tensor:
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    states = torch.zeros(
        batch, state_len, dim, dtype=x.dtype, device=x.device
    ).transpose(1, 2)
    for i in range(batch):
        end_idx = cu_seqlens[i + 1]
        start_idx = torch.maximum(cu_seqlens[i], end_idx - state_len)
        states[i, :, -(end_idx - start_idx) :] = x[start_idx:end_idx].T
    return states
