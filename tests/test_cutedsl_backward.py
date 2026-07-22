from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import causal_conv1d.cutedsl_kernels as cutedsl_kernels
from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref


FORWARD_BENCHMARK_SHAPES = (
    pytest.param(1, 768, 128, id="b1-d768-l128"),
    pytest.param(1, 768, 2048, id="b1-d768-l2048"),
    pytest.param(1, 2048, 512, id="b1-d2048-l512"),
    pytest.param(1, 2048, 2048, id="b1-d2048-l2048"),
    pytest.param(1, 4096, 128, id="b1-d4096-l128"),
    pytest.param(1, 4096, 2048, id="b1-d4096-l2048"),
    pytest.param(2, 2048, 2048, id="b2-d2048-l2048"),
    pytest.param(8, 2048, 512, id="b8-d2048-l512"),
)

FUSED_INPUT_DTYPES = (
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float32, id="fp32"),
)

REDUCTION_MODES = (
    pytest.param(False, id="atomic"),
    pytest.param(True, id="deterministic"),
)

FOCUSED_FUSED_CASES = (
    pytest.param(
        torch.bfloat16,
        torch.float32,
        2,
        False,
        False,
        1,
        8,
        63,
        False,
        id="bf16-fp32-w2-l63-atomic",
    ),
    pytest.param(
        torch.float32,
        torch.float32,
        3,
        True,
        True,
        2,
        40,
        64,
        True,
        id="fp32-fp32-w3-l64-deterministic",
    ),
    pytest.param(
        torch.bfloat16,
        torch.bfloat16,
        4,
        True,
        False,
        2,
        72,
        65,
        False,
        id="bf16-bf16-w4-l65-atomic",
    ),
    pytest.param(
        torch.float32,
        torch.float32,
        2,
        False,
        True,
        1,
        40,
        127,
        True,
        id="fp32-fp32-w2-l127-deterministic",
    ),
    pytest.param(
        torch.bfloat16,
        torch.bfloat16,
        3,
        True,
        False,
        2,
        64,
        128,
        False,
        id="bf16-bf16-w3-l128-atomic",
    ),
    pytest.param(
        torch.float32,
        torch.float32,
        4,
        False,
        False,
        2,
        40,
        129,
        True,
        id="fp32-fp32-w4-l129-deterministic",
    ),
    pytest.param(
        torch.float32,
        torch.float16,
        4,
        True,
        False,
        2,
        40,
        129,
        True,
        id="fp32-fp16-w4-l129-deterministic",
    ),
)


def _fused_backward_supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0]
        >= cutedsl_kernels._FusedChannelLastBackward.MIN_COMPUTE_CAPABILITY_MAJOR
    )


_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
_requires_fused_backward = pytest.mark.skipif(
    not _fused_backward_supported(),
    reason="fused CuTe backward requires SM80+ CUDA",
)


def _set_torch_deterministic(enabled: bool) -> bool:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(enabled)
    return previous


def _reference_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    silu_activation: bool,
) -> tuple[torch.Tensor, ...]:
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None

    if weight.dtype == torch.float32:
        out_ref = causal_conv1d_ref(
            x_ref,
            weight_ref,
            bias_ref,
            activation="silu" if silu_activation else None,
        )
    else:
        # The fused kernel always accumulates in FP32, including mixed-dtype
        # parameter specializations.
        preact = F.conv1d(
            x_ref.float(),
            weight_ref.float().unsqueeze(1),
            bias_ref.float() if bias_ref is not None else None,
            padding=weight.shape[1] - 1,
            groups=x.shape[1],
        )[..., : x.shape[2]]
        out_ref = F.silu(preact) if silu_activation else preact
        out_ref = out_ref.to(x.dtype)

    inputs = (x_ref, weight_ref) + ((bias_ref,) if bias_ref is not None else ())
    return torch.autograd.grad(out_ref, inputs, dout)


def _run_fused_backward_case(
    *,
    batch: int,
    dim: int,
    seqlen: int,
    width: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    has_bias: bool,
    silu_activation: bool,
    deterministic: bool,
    monkeypatch: pytest.MonkeyPatch,
    gradient_tolerance: tuple[float, float] | None = None,
) -> None:
    dtype_seed = {
        torch.bfloat16: 0,
        torch.float16: 1,
        torch.float32: 2,
    }
    seed = (
        batch * 1_000_000
        + dim * 100
        + seqlen
        + width * 10_000_000
        + dtype_seed[input_dtype] * 100_000_000
        + dtype_seed[weight_dtype] * 200_000_000
        + int(has_bias) * 300_000_000
        + int(silu_activation) * 400_000_000
    )
    torch.manual_seed(seed)
    x = torch.randn(
        batch,
        seqlen,
        dim,
        device="cuda",
        dtype=input_dtype,
    ).transpose(1, 2)
    weight = torch.randn(dim, width, device="cuda", dtype=weight_dtype)
    bias = (
        torch.randn(dim, device="cuda", dtype=weight_dtype)
        if has_bias
        else None
    )
    dout = torch.randn_like(x)

    assert x.stride(1) == 1
    assert cutedsl_kernels._can_use_fused_channel_last_backward(
        x,
        dout,
        torch.empty_like(x),
    )
    expected = _reference_backward(
        x,
        weight,
        bias,
        dout,
        silu_activation,
    )

    fused_compile = cutedsl_kernels._compile_fused_channel_last_backward
    reduce_compile = cutedsl_kernels._compile_fused_reduce_gradients
    fused_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    reduce_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_fused_compile(*args, **kwargs):
        fused_calls.append((args, kwargs))
        return fused_compile(*args, **kwargs)

    def record_reduce_compile(*args, **kwargs):
        reduce_calls.append((args, kwargs))
        return reduce_compile(*args, **kwargs)

    monkeypatch.setattr(
        cutedsl_kernels,
        "_compile_fused_channel_last_backward",
        record_fused_compile,
    )
    monkeypatch.setattr(
        cutedsl_kernels,
        "_compile_fused_reduce_gradients",
        record_reduce_compile,
    )
    monkeypatch.setattr(
        cutedsl_kernels,
        "_compile_backward",
        lambda *args, **kwargs: pytest.fail(
            "generic compiler called for an aligned fused case"
        ),
    )
    monkeypatch.setenv(
        "CAUSAL_CONV1D_DETERMINISTIC",
        "1" if deterministic else "0",
    )

    previous_deterministic = _set_torch_deterministic(not deterministic)
    try:
        actual = cutedsl_kernels.causal_conv1d_bwd_cutedsl(
            x,
            weight,
            bias,
            dout,
            silu_activation,
        )
        torch.cuda.synchronize()
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    tile_l = 64 if seqlen <= 128 else 128
    assert len(fused_calls) == 1
    assert fused_calls[0][0] == (
        input_dtype,
        weight_dtype,
        width,
        has_bias,
        silu_activation,
        tile_l,
        deterministic,
    )
    assert len(reduce_calls) == int(deterministic)
    if deterministic:
        assert reduce_calls[0][0] == (width, has_bias)

    dx_rtol, dx_atol = (
        (3e-4, 1e-3) if input_dtype == torch.float32 else (1e-2, 5e-2)
    )
    if gradient_tolerance is None:
        gradient_tolerance = (
            (2e-3, 2e-2)
            if input_dtype == torch.float32
            else (1e-2, 5e-2)
        )
    grad_rtol, grad_atol = gradient_tolerance

    torch.testing.assert_close(
        actual[0],
        expected[0],
        rtol=dx_rtol,
        atol=dx_atol,
    )
    torch.testing.assert_close(
        actual[1],
        expected[1],
        rtol=grad_rtol,
        atol=grad_atol,
    )
    if has_bias:
        torch.testing.assert_close(
            actual[2],
            expected[2],
            rtol=grad_rtol,
            atol=grad_atol,
        )
    else:
        assert actual[2] is None


@pytest.mark.parametrize(
    ("env_value", "torch_deterministic", "expected"),
    [
        (None, False, False),
        (None, True, True),
        ("1", False, True),
        ("0", True, False),
        ("invalid", False, False),
        ("invalid", True, True),
    ],
    ids=[
        "global-default",
        "global-deterministic",
        "env-enables",
        "env-disables",
        "invalid-env-default",
        "invalid-env-deterministic",
    ],
)
def test_cutedsl_backward_deterministic_selector(
    env_value: str | None,
    torch_deterministic: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if env_value is None:
        monkeypatch.delenv("CAUSAL_CONV1D_DETERMINISTIC", raising=False)
    else:
        monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", env_value)
    previous = _set_torch_deterministic(torch_deterministic)
    try:
        assert cutedsl_kernels._use_deterministic_mode() is expected
    finally:
        torch.use_deterministic_algorithms(previous)


@_requires_fused_backward
@pytest.mark.parametrize(
    (
        "input_dtype",
        "weight_dtype",
        "width",
        "has_bias",
        "silu_activation",
        "batch",
        "dim",
        "seqlen",
        "deterministic",
    ),
    FOCUSED_FUSED_CASES,
)
def test_cutedsl_backward_fused_specializations(
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    width: int,
    has_bias: bool,
    silu_activation: bool,
    batch: int,
    dim: int,
    seqlen: int,
    deterministic: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gradient_tolerance = (
        (1e-3, 1e-3)
        if weight_dtype == torch.float32
        else (1e-2, 5e-2)
    )
    _run_fused_backward_case(
        batch=batch,
        dim=dim,
        seqlen=seqlen,
        width=width,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        has_bias=has_bias,
        silu_activation=silu_activation,
        deterministic=deterministic,
        monkeypatch=monkeypatch,
        gradient_tolerance=gradient_tolerance,
    )


@_requires_fused_backward
def test_cutedsl_backward_uses_current_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TVM-FFI environment stream must preserve PyTorch ordering."""
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    torch.manual_seed(1165)
    batch, dim, seqlen, width = 2, 72, 65, 4
    stream = torch.cuda.Stream()
    done = torch.cuda.Event()
    with torch.cuda.stream(stream):
        x = torch.randn(
            batch,
            seqlen,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).transpose(1, 2)
        weight = torch.randn(dim, width, device="cuda", dtype=torch.float32)
        bias = torch.randn(dim, device="cuda", dtype=torch.float32)
        dout = torch.randn_like(x)
        actual = cutedsl_kernels.causal_conv1d_bwd_cutedsl(
            x,
            weight,
            bias,
            dout,
            True,
        )
        done.record(stream)

    torch.cuda.current_stream().wait_event(done)
    expected = _reference_backward(x, weight, bias, dout, True)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=5e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-3, atol=1e-3)


@_requires_cuda
@pytest.mark.parametrize(
    "fallback_reason",
    [
        "fp16",
        "channel_first",
        "channel_first_l1",
        "dim_not_multiple_of_8",
        "misaligned",
        "padded_dout",
        "pre_sm80",
    ],
)
def test_cutedsl_backward_fused_falls_back(
    fallback_reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(2000)
    batch, dim, seqlen, width = 1, 64, 17, 4
    input_dtype = torch.bfloat16
    if fallback_reason == "fp16":
        input_dtype = torch.float16
        x = torch.randn(
            batch,
            seqlen,
            dim,
            device="cuda",
            dtype=input_dtype,
        ).transpose(1, 2)
    elif fallback_reason == "channel_first":
        x = torch.randn(
            batch,
            dim,
            seqlen,
            device="cuda",
            dtype=input_dtype,
        )
    elif fallback_reason == "channel_first_l1":
        seqlen = 1
        x = torch.randn(
            batch,
            dim,
            seqlen,
            device="cuda",
            dtype=input_dtype,
        )
    elif fallback_reason == "dim_not_multiple_of_8":
        dim = 12
        input_dtype = torch.float32
        x = torch.randn(
            batch,
            seqlen,
            dim,
            device="cuda",
            dtype=input_dtype,
        ).transpose(1, 2)
    elif fallback_reason == "misaligned":
        backing = torch.randn(
            batch,
            seqlen,
            dim + 8,
            device="cuda",
            dtype=input_dtype,
        )
        x = backing[:, :, 1 : dim + 1].transpose(1, 2)
        assert x.data_ptr() % 16 != 0
        assert x.stride(2) % 8 == 0
    else:
        x = torch.randn(
            batch,
            seqlen,
            dim,
            device="cuda",
            dtype=input_dtype,
        ).transpose(1, 2)

    if fallback_reason == "pre_sm80":
        monkeypatch.setattr(
            torch.cuda,
            "get_device_capability",
            lambda *args, **kwargs: (7, 5),
        )

    if fallback_reason == "padded_dout":
        dout_backing = torch.randn(
            batch,
            seqlen,
            dim + 1,
            device="cuda",
            dtype=input_dtype,
        )
        dout = dout_backing[:, :, :dim].transpose(1, 2)
        assert dout.data_ptr() % 16 == 0
        assert dout.stride(2) % 8 != 0
    else:
        dout = torch.randn_like(x)

    weight = torch.randn(dim, width, device="cuda", dtype=torch.float32)
    bias = torch.randn(dim, device="cuda", dtype=torch.float32)
    assert not cutedsl_kernels._can_use_fused_channel_last_backward(
        x,
        dout,
        torch.empty_like(x),
    )

    generic_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_compile_backward(*args, **kwargs):
        generic_calls.append((args, kwargs))

        def fake_dx_kernel(x, weight, bias, dout, dx):
            dx.copy_(dout)

        def fake_weight_kernel(x, weight, bias, dout, dweight, dbias):
            dweight.fill_(0.25)
            dbias.fill_(0.5)

        return fake_dx_kernel, fake_weight_kernel

    def fail_fused_compile(*args, **kwargs):
        pytest.fail(
            f"fused compiler called for fallback case {fallback_reason}"
        )

    monkeypatch.setattr(
        cutedsl_kernels,
        "_compile_backward",
        fake_compile_backward,
    )
    monkeypatch.setattr(
        cutedsl_kernels,
        "_compile_fused_channel_last_backward",
        fail_fused_compile,
    )
    monkeypatch.setattr(
        cutedsl_kernels,
        "_compile_fused_reduce_gradients",
        fail_fused_compile,
    )
    dx, dweight, dbias = cutedsl_kernels.causal_conv1d_bwd_cutedsl(
        x,
        weight,
        bias,
        dout,
        False,
    )

    assert len(generic_calls) == 1
    assert torch.equal(dx, dout)
    assert torch.all(dweight == 0.25)
    assert torch.all(dbias == 0.5)


@_requires_cuda
def test_cutedsl_backward_generic_channel_last_numeric_fallback() -> None:
    """Exercise the real generic executors, not only dispatch spies."""
    torch.manual_seed(2017)
    batch, dim, seqlen, width = 1, 64, 17, 4
    x = torch.randn(
        batch,
        seqlen,
        dim,
        device="cuda",
        dtype=torch.float16,
    ).transpose(1, 2)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.float32)
    bias = torch.randn(dim, device="cuda", dtype=torch.float32)
    dout = torch.randn_like(x)

    expected = _reference_backward(x, weight, bias, dout, True)
    actual = cutedsl_kernels.causal_conv1d_bwd_cutedsl(
        x,
        weight,
        bias,
        dout,
        True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual[0], expected[0], rtol=3e-3, atol=5e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-3, atol=1e-3)


@_requires_fused_backward
@pytest.mark.parametrize("deterministic", REDUCTION_MODES)
@pytest.mark.parametrize("input_dtype", FUSED_INPUT_DTYPES)
@pytest.mark.parametrize("width", (2, 3, 4), ids=lambda width: f"w{width}")
@pytest.mark.parametrize(
    ("batch", "dim", "seqlen"),
    FORWARD_BENCHMARK_SHAPES,
)
def test_cutedsl_backward_forward_benchmark_shape_matrix(
    batch: int,
    dim: int,
    seqlen: int,
    width: int,
    input_dtype: torch.dtype,
    deterministic: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_fused_backward_case(
        batch=batch,
        dim=dim,
        seqlen=seqlen,
        width=width,
        input_dtype=input_dtype,
        weight_dtype=torch.float32,
        has_bias=True,
        silu_activation=True,
        deterministic=deterministic,
        monkeypatch=monkeypatch,
    )


@_requires_fused_backward
@pytest.mark.parametrize("deterministic", REDUCTION_MODES)
@pytest.mark.parametrize(
    "silu_activation",
    (False, True),
    ids=("linear", "silu"),
)
@pytest.mark.parametrize(
    "has_bias",
    (False, True),
    ids=("no-bias", "bias"),
)
@pytest.mark.parametrize("input_dtype", FUSED_INPUT_DTYPES)
@pytest.mark.parametrize("width", (2, 3, 4), ids=lambda width: f"w{width}")
def test_cutedsl_backward_feature_matrix(
    width: int,
    input_dtype: torch.dtype,
    has_bias: bool,
    silu_activation: bool,
    deterministic: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_fused_backward_case(
        batch=2,
        dim=72,
        seqlen=129,
        width=width,
        input_dtype=input_dtype,
        weight_dtype=torch.float32,
        has_bias=has_bias,
        silu_activation=silu_activation,
        deterministic=deterministic,
        monkeypatch=monkeypatch,
    )
