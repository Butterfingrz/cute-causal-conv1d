"""Python dispatch for the compilation-free CuTe DSL implementation.

The common channel-last forward path runs a CuTe DSL kernel.  The less common
stateful/layout paths temporarily use PyTorch operators while their dedicated
CuTe DSL kernels are brought up; importantly, this module never imports or
builds ``causal_conv1d_cuda``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from causal_conv1d.cutedsl_kernels import (
    causal_conv1d_fwd_cutedsl,
    causal_conv1d_bwd_cutedsl,
    causal_conv1d_update_cutedsl,
)


def _causal_conv1d_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    silu_activation: bool,
) -> torch.Tensor:
    """Stride-preserving reference used for uncommon forward specializations."""
    dtype_in = x.dtype
    width = weight.shape[1]
    x_compute = x.to(weight.dtype)
    if initial_states is not None:
        x_padded = torch.cat((initial_states.to(weight.dtype), x_compute), dim=-1)
    else:
        x_padded = F.pad(x_compute, (width - 1, 0))

    out = torch.zeros_like(x_compute)
    if bias is not None:
        out = out + bias[:, None]
    for w in range(width):
        term = x_padded[..., w : w + x.shape[-1]] * weight[:, w][None, :, None]
        if seq_idx is not None:
            current = seq_idx
            previous = F.pad(seq_idx, (width - 1, 0), value=-1)[
                ..., w : w + x.shape[-1]
            ]
            term = term * ((current >= 0) & (previous == current))[:, None, :]
        out = out + term
    if seq_idx is not None:
        out = out * (seq_idx >= 0)[:, None, :]
    if silu_activation:
        out = F.silu(out)
    return out.to(dtype_in)


def _last_history(history: torch.Tensor, length: int) -> torch.Tensor:
    """Return the last ``length`` entries, left-padding short histories."""
    if history.shape[-1] < length:
        return F.pad(history, (length - history.shape[-1], 0))
    return history[..., -length:]


def causal_conv1d_fwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    final_states_out: torch.Tensor | None,
    silu_activation: bool,
) -> torch.Tensor:
    if x.stride(1) == 1 and seq_idx is None and initial_states is None:
        out = causal_conv1d_fwd_cutedsl(
            x, weight, bias, "silu" if silu_activation else None
        )
    else:
        out = _causal_conv1d_torch(
            x, weight, bias, seq_idx, initial_states, silu_activation
        )

    if final_states_out is not None:
        width = weight.shape[1]
        history = (
            x if initial_states is None else torch.cat((initial_states, x), dim=-1)
        )
        final_states = _last_history(history, width - 1)
        final_states_out.copy_(final_states)
    return out


def causal_conv1d_bwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
    dx: torch.Tensor | None,
    return_dinitial_states: bool,
    silu_activation: bool,
):
    if seq_idx is None and initial_states is None and dfinal_states is None:
        computed_dx, dweight, dbias = causal_conv1d_bwd_cutedsl(
            x, weight, bias, dout, silu_activation
        )
        if dx is None:
            dx = computed_dx
        else:
            dx.copy_(computed_dx)
        return dx, dweight, dbias, None

    # Stateful/segmented fallback while those backward specializations are brought up.
    with torch.enable_grad():
        x_var = x.detach().requires_grad_(True)
        weight_var = weight.detach().requires_grad_(True)
        bias_var = bias.detach().requires_grad_(True) if bias is not None else None
        initial_var = (
            initial_states.detach().requires_grad_(True)
            if initial_states is not None
            else None
        )
        out = _causal_conv1d_torch(
            x_var, weight_var, bias_var, seq_idx, initial_var, silu_activation
        )
        outputs = [out]
        grad_outputs = [dout]
        if dfinal_states is not None:
            width = weight.shape[1]
            history = (
                x_var
                if initial_var is None
                else torch.cat((initial_var, x_var), dim=-1)
            )
            final = _last_history(history, width - 1)
            outputs.append(final)
            grad_outputs.append(dfinal_states)
        inputs = [x_var, weight_var]
        if bias_var is not None:
            inputs.append(bias_var)
        if initial_var is not None:
            inputs.append(initial_var)
        grads = torch.autograd.grad(outputs, inputs, grad_outputs, allow_unused=True)

    grad_idx = 0
    computed_dx = grads[grad_idx]
    grad_idx += 1
    if dx is None:
        dx = computed_dx
    elif computed_dx is not None:
        dx.copy_(computed_dx)
    dweight = grads[grad_idx].to(weight.dtype) if grads[grad_idx] is not None else None
    grad_idx += 1
    dbias = None
    if bias is not None:
        dbias = grads[grad_idx].to(bias.dtype) if grads[grad_idx] is not None else None
        grad_idx += 1
    dinitial_states = None
    if initial_states is not None and return_dinitial_states:
        dinitial_states = (
            grads[grad_idx].to(initial_states.dtype)
            if grads[grad_idx] is not None
            else None
        )
    return dx, dweight, dbias, dinitial_states


def causal_conv1d_update_function(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    silu_activation: bool,
    cache_seqlens: torch.Tensor | None,
    conv_state_indices: torch.Tensor | None,
) -> torch.Tensor:
    if x.stride(1) == 1 and conv_state.stride(1) == 1:
        return causal_conv1d_update_cutedsl(
            x,
            conv_state,
            weight,
            bias,
            "silu" if silu_activation else None,
            cache_seqlens,
            conv_state_indices,
        )

    # Layout fallback while the channel-first update specialization is brought up.
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    out = torch.zeros_like(x)

    if conv_state_indices is None:
        x_active = x
        cache_seqlens_active = cache_seqlens
        selected = conv_state
    else:
        valid = conv_state_indices >= 0
        if not valid.any():
            return out
        x_active = x[valid]
        indices_active = conv_state_indices[valid].long()
        cache_seqlens_active = (
            cache_seqlens[valid] if cache_seqlens is not None else None
        )
        selected = conv_state[indices_active]

    if cache_seqlens_active is None:
        x_new = torch.cat((selected, x_active), dim=-1).to(weight.dtype)
        selected.copy_(x_new[..., -state_len:].to(selected.dtype))
    else:
        history_idx = (
            torch.arange(-(width - 1), 0, device=x.device)[None, :]
            + cache_seqlens_active[:, None]
        ).remainder(state_len)
        history_idx = history_idx[:, None, :].expand(-1, dim, -1)
        x_new = torch.cat((selected.gather(2, history_idx), x_active), dim=-1).to(
            weight.dtype
        )
        copy_idx = (
            torch.arange(seqlen, device=x.device)[None, :]
            + cache_seqlens_active[:, None]
        ).remainder(state_len)
        copy_idx = copy_idx[:, None, :].expand(-1, dim, -1)
        selected.scatter_(2, copy_idx, x_active)

    out_active = F.conv1d(x_new, weight[:, None], bias, groups=dim)[..., -seqlen:]
    if silu_activation:
        out_active = F.silu(out_active)

    if conv_state_indices is not None:
        conv_state[indices_active] = selected
        out[valid] = out_active.to(x.dtype)
        return out

    return out_active.to(x.dtype)
