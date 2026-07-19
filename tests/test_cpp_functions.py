import pytest
import torch

from causal_conv1d.causal_conv1d_interface import (
    causal_conv1d_update,
    causal_conv1d_update_ref,
)


@pytest.mark.parametrize("has_cache_seqlens", [False, True])
def test_update_channel_first_masks_negative_state_indices(has_cache_seqlens):
    """Negative gather indices remain masked in the channel-first fallback."""
    device = "cuda"
    batch, dim, seqlen, width, state_len = 4, 8, 2, 3, 4
    x = torch.randn(batch, dim, seqlen, device=device)
    conv_state = torch.randn(6, dim, state_len, device=device)
    conv_state_original = conv_state.clone()
    conv_state_indices = torch.tensor([1, -1, 4, -1], device=device, dtype=torch.int32)
    cache_seqlens = (
        torch.tensor([3, 7, 5, 9], device=device, dtype=torch.int32)
        if has_cache_seqlens
        else None
    )
    weight = torch.randn(dim, width, device=device)
    bias = torch.randn(dim, device=device)

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation="silu",
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
    )

    valid = conv_state_indices >= 0
    indices_valid = conv_state_indices[valid].long()
    state_valid = conv_state_original[indices_valid].clone()
    out_valid = causal_conv1d_update_ref(
        x[valid],
        state_valid,
        weight,
        bias,
        activation="silu",
        cache_seqlens=cache_seqlens[valid] if cache_seqlens is not None else None,
    )
    expected_out = torch.zeros_like(out)
    expected_out[valid] = out_valid
    expected_state = conv_state_original.clone()
    expected_state[indices_valid] = state_valid

    assert torch.equal(out, expected_out)
    assert torch.equal(conv_state, expected_state)
