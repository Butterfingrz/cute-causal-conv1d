import torch

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from causal_conv1d.causal_conv1d_interface import (
    causal_conv1d_ref,
    causal_conv1d_update_ref,
)


def check_forward(channel_last: bool, initial: bool, final: bool) -> None:
    torch.manual_seed(10 + channel_last + 2 * initial + 4 * final)
    batch, dim, seqlen, width = 2, 64, 65, 4
    if channel_last:
        x = torch.randn(
            batch, seqlen, dim, device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
    else:
        x = torch.randn(batch, dim, seqlen, device="cuda", dtype=torch.bfloat16)
    x.requires_grad_()
    weight = torch.randn(dim, width, device="cuda", requires_grad=True)
    bias = torch.randn(dim, device="cuda", requires_grad=True)
    states = None
    if initial:
        states = torch.randn(
            batch, width - 1, dim, device="cuda", dtype=x.dtype
        ).transpose(1, 2)
        states.requires_grad_()

    actual = causal_conv1d_fn(
        x,
        weight,
        bias,
        initial_states=states,
        return_final_states=final,
        activation="silu",
    )
    x_ref = x.detach().clone().requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    b_ref = bias.detach().clone().requires_grad_()
    s_ref = states.detach().clone().requires_grad_() if states is not None else None
    expected = causal_conv1d_ref(
        x_ref,
        w_ref,
        b_ref,
        initial_states=s_ref,
        return_final_states=final,
        activation="silu",
    )
    actual_out, expected_out = (actual[0], expected[0]) if final else (actual, expected)
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-2, atol=5e-2)
    if final:
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        actual_out = actual_out + actual[1].sum()
        expected_out = expected_out + expected[1].sum()
    grad = torch.randn_like(actual_out)
    actual_out.backward(grad)
    expected_out.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-2, atol=5e-2)
    torch.testing.assert_close(weight.grad, w_ref.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(bias.grad, b_ref.grad, rtol=2e-3, atol=2e-3)
    if states is not None:
        torch.testing.assert_close(states.grad, s_ref.grad, rtol=1e-2, atol=5e-2)


def check_update() -> None:
    torch.manual_seed(20)
    batch, dim, seqlen, width, state_len = 4, 64, 3, 4, 7
    x = torch.randn(batch, seqlen, dim, device="cuda", dtype=torch.bfloat16).transpose(
        1, 2
    )
    state = torch.randn(batch, state_len, dim, device="cuda", dtype=x.dtype).transpose(
        1, 2
    )
    state_ref = state.clone()
    weight = torch.randn(dim, width, device="cuda")
    bias = torch.randn(dim, device="cuda")
    actual = causal_conv1d_update(x, state, weight, bias, activation="silu")
    expected = causal_conv1d_update_ref(x, state_ref, weight, bias, activation="silu")
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)
    torch.testing.assert_close(state, state_ref, rtol=0, atol=0)


def main() -> None:
    check_forward(True, False, False)
    check_forward(True, True, True)
    check_forward(False, False, False)
    check_update()
    print("Public API forward/backward/state/update smoke: PASS")


if __name__ == "__main__":
    main()
