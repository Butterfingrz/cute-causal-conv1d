"""CuTe DSL kernels for causal depthwise convolution.

The kernels in this module are compiled lazily by the CuTe DSL runtime.  There is
no C++ or CUDA extension build step.
"""

from __future__ import annotations

import operator
import os
from functools import cache

import torch


def _cutedsl_imports():
    try:
        import cutlass
        from cuda.bindings.driver import CUstream
        from cutlass import BFloat16, Float16, Float32, Int32, Int64, cute
        from cutlass._mlir.dialects import llvm
        from cutlass.cutlass_dsl import T, dsl_user_op
    except ImportError as exc:  # pragma: no cover - depends on the CUDA environment
        raise ImportError(
            "causal-conv1d requires nvidia-cutlass-dsl; install the package "
            "for your CUDA toolkit"
        ) from exc
    return (
        cutlass,
        CUstream,
        BFloat16,
        Float16,
        Float32,
        Int32,
        Int64,
        cute,
        llvm,
        T,
        dsl_user_op,
    )


(
    cutlass,
    CUstream,
    BFloat16,
    Float16,
    Float32,
    Int32,
    Int64,
    cute,
    llvm,
    T,
    dsl_user_op,
) = _cutedsl_imports()


@dsl_user_op
def _tanh_approx(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


_TORCH_TO_CUTE = {
    torch.float32: Float32,
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
}


def _fake_tensor(dtype, shape, stride, assumed_align):
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=assumed_align
    )


class _ChannelLastForward:
    """Forward kernel for tensors whose channel dimension has unit stride."""

    TILE_L = 64
    TILE_C = 64
    THREADS = 128

    def __init__(
        self,
        width: int,
        has_bias: bool,
        silu: bool,
        vector_elems: int,
        even_dim: bool,
        even_seqlen: bool,
    ) -> None:
        self.width = width
        self.has_bias = has_bias
        self.silu = silu
        self.vector_elems = vector_elems
        self.even_dim = even_dim
        self.even_seqlen = even_seqlen
        # Narrow filters need more independent blocks to hide activation
        # latency; width 4 amortizes its halo better with the 64-token tile.
        self.tile_l = 32 if width < 4 else self.TILE_L

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor | None,
        out: cute.Tensor,
        stream: CUstream,
    ):
        batch, dim, seqlen = x.shape
        self.kernel(x, weight, bias, out).launch(
            grid=(
                batch,
                cute.ceil_div(seqlen, self.tile_l),
                cute.ceil_div(dim, self.TILE_C),
            ),
            block=(self.THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor | None,
        out: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        batch_idx, tile_l_idx, tile_c_idx = cute.arch.block_idx()
        dim = x.shape[1]
        seqlen = x.shape[2]
        l_base = tile_l_idx * self.tile_l
        c_base = tile_c_idx * self.TILE_C

        smem = cutlass.utils.SmemAllocator()
        sx = smem.allocate_tensor(
            x.element_type,
            cute.make_layout(
                (self.tile_l + self.width - 1, self.TILE_C),
                stride=(self.TILE_C + self.vector_elems, 1),
            ),
            byte_alignment=16,
        )

        # Each participating thread moves one aligned 16-byte channel vector,
        # matching the transaction shape of the original CUDA kernel.
        vectors_per_row = self.TILE_C // self.vector_elems
        rows_per_round = self.THREADS // vectors_per_row
        row_in_round = tid // vectors_per_row
        vector_in_row = tid % vectors_per_row
        vector_global_c = c_base + vector_in_row * self.vector_elems
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            x.element_type,
            num_bits_per_copy=128,
        )
        for row_base in cutlass.range_constexpr(0, self.tile_l, rows_per_round):
            local_l = self.width - 1 + row_base + row_in_round
            global_l = l_base + row_base + row_in_round
            if cutlass.const_expr(self.even_dim and self.even_seqlen) or (
                global_l < seqlen and vector_global_c < dim
            ):
                src = cute.local_tile(
                    x[batch_idx, None, global_l],
                    (self.vector_elems,),
                    (c_base // self.vector_elems + vector_in_row,),
                )
                dst = cute.local_tile(
                    sx[local_l, None],
                    (self.vector_elems,),
                    (vector_in_row,),
                )
                cute.copy(copy_atom, src, dst)
        # The first width - 1 rows are the causal halo.
        if row_in_round < self.width - 1:
            global_l = l_base + row_in_round - (self.width - 1)
            if global_l >= 0 and (
                cutlass.const_expr(self.even_dim) or vector_global_c < dim
            ):
                src = cute.local_tile(
                    x[batch_idx, None, global_l],
                    (self.vector_elems,),
                    (c_base // self.vector_elems + vector_in_row,),
                )
                dst = cute.local_tile(
                    sx[row_in_round, None],
                    (self.vector_elems,),
                    (vector_in_row,),
                )
                cute.copy(copy_atom, src, dst)
            else:
                for i in cutlass.range_constexpr(self.vector_elems):
                    sx[row_in_round, vector_in_row * self.vector_elems + i] = (
                        x.element_type(0.0)
                    )

        cute.arch.sync_threads()

        outputs_per_thread = self.tile_l * self.TILE_C // self.THREADS
        thread_out = cute.make_rmem_tensor((outputs_per_thread,), Float32)
        threads_per_channel = self.THREADS // self.TILE_C
        thread_local_c = tid // threads_per_channel
        thread_l_start = (tid % threads_per_channel) * outputs_per_thread
        thread_global_c = c_base + thread_local_c
        thread_bias = Float32(0.0)
        thread_weight = cute.make_rmem_tensor((self.width,), Float32)
        thread_x = cute.make_rmem_tensor(
            (outputs_per_thread + self.width - 1,), Float32
        )
        if cutlass.const_expr(self.even_dim) or thread_global_c < dim:
            if cutlass.const_expr(self.has_bias):
                thread_bias = Float32(bias[thread_global_c])
            for w in cutlass.range_constexpr(self.width):
                thread_weight[w] = Float32(weight[thread_global_c, w])
            for i in cutlass.range_constexpr(outputs_per_thread + self.width - 1):
                thread_x[i] = Float32(sx[thread_l_start + i, thread_local_c])
        for i in cutlass.range_constexpr(outputs_per_thread):
            local_l = thread_l_start + i
            global_l = l_base + local_l
            if cutlass.const_expr(self.even_dim and self.even_seqlen) or (
                global_l < seqlen and thread_global_c < dim
            ):
                acc = thread_bias
                for w in cutlass.range_constexpr(self.width):
                    acc += thread_weight[w] * thread_x[i + w]
                thread_out[i] = acc
            else:
                thread_out[i] = Float32(0.0)

        # All input reads must complete before recycling shared memory for the
        # coalesced/vectorized output exchange. Width 2 schedules this barrier
        # after SiLU; widths 3-4 do better when preactivations meet here first.
        if cutlass.const_expr(not self.silu or self.width >= 3):
            cute.arch.sync_threads()

        # Keep activation separate from convolution so all independent values
        # are available to the scheduler.
        if cutlass.const_expr(self.silu):
            for i in cutlass.range_constexpr(outputs_per_thread):
                half = Float32(0.5) * thread_out[i]
                thread_out[i] = half * _tanh_approx(half) + half

        if cutlass.const_expr(self.silu and self.width == 2):
            cute.arch.sync_threads()

        for i in cutlass.range_constexpr(outputs_per_thread):
            local_l = thread_l_start + i
            sx[self.width - 1 + local_l, thread_local_c] = thread_out[i].to(
                x.element_type
            )
        cute.arch.sync_threads()

        for row_base in cutlass.range_constexpr(0, self.tile_l, rows_per_round):
            local_l = row_base + row_in_round
            global_l = l_base + local_l
            if cutlass.const_expr(self.even_dim and self.even_seqlen) or (
                global_l < seqlen and vector_global_c < dim
            ):
                src = cute.local_tile(
                    sx[self.width - 1 + local_l, None],
                    (self.vector_elems,),
                    (vector_in_row,),
                )
                dst = cute.local_tile(
                    out[batch_idx, None, global_l],
                    (self.vector_elems,),
                    (c_base // self.vector_elems + vector_in_row,),
                )
                cute.copy(copy_atom, src, dst)


@cache
def _compile_channel_last_forward(
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    width: int,
    has_bias: bool,
    silu: bool,
    even_dim: bool,
    even_seqlen: bool,
):
    x_dtype = _TORCH_TO_CUTE[input_dtype]
    w_dtype = _TORCH_TO_CUTE[weight_dtype]
    vector_elems = 128 // x_dtype.width
    batch = cute.sym_int()
    dim = cute.sym_int()
    seqlen = cute.sym_int()
    batch_stride = cute.sym_int64(divisibility=vector_elems)
    seqlen_stride = cute.sym_int64(divisibility=vector_elems)
    x = _fake_tensor(
        x_dtype,
        (batch, dim, seqlen),
        (batch_stride, 1, seqlen_stride),
        16,
    )
    weight = _fake_tensor(w_dtype, (dim, width), (width, 1), 16)
    bias = _fake_tensor(w_dtype, (dim,), (1,), 16) if has_bias else None
    out_batch_stride = cute.sym_int64(divisibility=vector_elems)
    out_seqlen_stride = cute.sym_int64(divisibility=vector_elems)
    out = _fake_tensor(
        x_dtype,
        (batch, dim, seqlen),
        (out_batch_stride, 1, out_seqlen_stride),
        16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _ChannelLastForward(
            width,
            has_bias,
            silu,
            vector_elems,
            even_dim,
            even_seqlen,
        ),
        x,
        weight,
        bias,
        out,
        stream,
        options="--enable-tvm-ffi",
    )


def causal_conv1d_fwd_cutedsl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Run causal depthwise conv1d with the CuTe DSL channel-last kernel."""
    if x.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "x and weight must have shapes (batch, dim, seqlen) and (dim, width)"
        )
    if x.stride(1) != 1:
        raise NotImplementedError("the initial CuTe DSL kernel requires channel-last x")
    if weight.shape != (x.shape[1], weight.shape[1]) or weight.shape[1] not in (
        2,
        3,
        4,
    ):
        raise ValueError("weight must have shape (dim, width), with width in {2, 3, 4}")
    if bias is not None and bias.shape != (x.shape[1],):
        raise ValueError("bias must have shape (dim,)")
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.dtype not in _TORCH_TO_CUTE or weight.dtype not in _TORCH_TO_CUTE:
        raise TypeError("x and weight must be float32, float16, or bfloat16")
    vector_elems = 128 // _TORCH_TO_CUTE[x.dtype].width
    if x.shape[1] % vector_elems != 0:
        raise ValueError(f"channel-last dim must be divisible by {vector_elems}")
    out = torch.empty_like(x)
    tile_l = 32 if weight.shape[1] < 4 else _ChannelLastForward.TILE_L
    kernel = _compile_channel_last_forward(
        x.dtype,
        weight.dtype,
        weight.shape[1],
        bias is not None,
        activation in ("silu", "swish"),
        x.shape[1] % _ChannelLastForward.TILE_C == 0,
        x.shape[2] % tile_l == 0,
    )
    kernel(x, weight, bias, out)
    return out


class _UpdateKernel:
    """Fused convolution-state update for decoding and short speculative steps."""

    THREADS = 128

    def __init__(
        self,
        width: int,
        has_bias: bool,
        silu: bool,
        circular: bool,
        indexed: bool,
    ) -> None:
        self.width = width
        self.has_bias = has_bias
        self.silu = silu
        self.circular = circular
        self.indexed = indexed

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        conv_state: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor | None,
        out: cute.Tensor,
        cache_seqlens: cute.Tensor | None,
        conv_state_indices: cute.Tensor | None,
        stream: CUstream,
    ):
        self.kernel(
            x,
            conv_state,
            weight,
            bias,
            out,
            cache_seqlens,
            conv_state_indices,
        ).launch(
            grid=(x.shape[0], cute.ceil_div(x.shape[1], self.THREADS), 1),
            block=(self.THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        conv_state: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor | None,
        out: cute.Tensor,
        cache_seqlens: cute.Tensor | None,
        conv_state_indices: cute.Tensor | None,
    ):
        tid, _, _ = cute.arch.thread_idx()
        batch_idx, channel_tile, _ = cute.arch.block_idx()
        channel = channel_tile * self.THREADS + tid
        dim = x.shape[1]
        seqlen = x.shape[2]
        state_len = conv_state.shape[2]
        state_batch = batch_idx
        if cutlass.const_expr(self.indexed):
            state_batch = conv_state_indices[batch_idx]

        if channel < dim:
            if state_batch >= 0:
                cache_pos = Int32(0)
                if cutlass.const_expr(self.circular):
                    cache_pos = cache_seqlens[batch_idx]
                for seq_pos in range(seqlen):
                    acc = Float32(0.0)
                    if cutlass.const_expr(self.has_bias):
                        acc = Float32(bias[channel])
                    for w in cutlass.range_constexpr(self.width):
                        input_pos = seq_pos + w - (self.width - 1)
                        value = x.element_type(0.0)
                        if input_pos >= 0:
                            value = x[batch_idx, channel, input_pos]
                        else:
                            state_pos = state_len + input_pos
                            if cutlass.const_expr(self.circular):
                                state_pos = (
                                    cache_pos + input_pos + state_len
                                ) % state_len
                            value = conv_state[state_batch, channel, state_pos]
                        acc += Float32(weight[channel, w]) * Float32(value)
                    if cutlass.const_expr(self.silu):
                        acc *= cute.arch.rcp_approx(
                            Float32(1.0) + cute.math.exp(-acc, fastmath=True)
                        )
                    out[batch_idx, channel, seq_pos] = acc.to(out.element_type)

                # Update state only after every output has consumed the old state.
                if cutlass.const_expr(self.circular):
                    for seq_pos in range(seqlen):
                        state_pos = (cache_pos + seq_pos) % state_len
                        conv_state[state_batch, channel, state_pos] = x[
                            batch_idx, channel, seq_pos
                        ]
                else:
                    for state_pos in range(state_len):
                        source_pos = seqlen + state_pos
                        if source_pos < state_len:
                            conv_state[state_batch, channel, state_pos] = conv_state[
                                state_batch, channel, source_pos
                            ]
                        else:
                            conv_state[state_batch, channel, state_pos] = x[
                                batch_idx, channel, source_pos - state_len
                            ]
            else:
                for seq_pos in range(seqlen):
                    out[batch_idx, channel, seq_pos] = x.element_type(0.0)


@cache
def _compile_update(
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    width: int,
    has_bias: bool,
    silu: bool,
    circular: bool,
    indexed: bool,
):
    x_dtype = _TORCH_TO_CUTE[input_dtype]
    w_dtype = _TORCH_TO_CUTE[weight_dtype]
    batch = cute.sym_int()
    state_batch = cute.sym_int()
    dim = cute.sym_int()
    seqlen = cute.sym_int()
    state_len = cute.sym_int()
    x_batch_stride = cute.sym_int64(divisibility=1)
    x_l_stride = cute.sym_int64(divisibility=1)
    state_batch_stride = cute.sym_int64(divisibility=1)
    state_l_stride = cute.sym_int64(divisibility=1)
    x = _fake_tensor(
        x_dtype,
        (batch, dim, seqlen),
        (x_batch_stride, 1, x_l_stride),
        x_dtype.width // 8,
    )
    conv_state = _fake_tensor(
        x_dtype,
        (state_batch, dim, state_len),
        (state_batch_stride, 1, state_l_stride),
        x_dtype.width // 8,
    )
    weight = _fake_tensor(w_dtype, (dim, width), (width, 1), w_dtype.width // 8)
    bias = _fake_tensor(w_dtype, (dim,), (1,), w_dtype.width // 8) if has_bias else None
    out = _fake_tensor(
        x_dtype,
        (batch, dim, seqlen),
        (x_batch_stride, 1, x_l_stride),
        x_dtype.width // 8,
    )
    cache_seqlens = _fake_tensor(Int32, (batch,), (1,), 4) if circular else None
    conv_state_indices = _fake_tensor(Int32, (batch,), (1,), 4) if indexed else None
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _UpdateKernel(width, has_bias, silu, circular, indexed),
        x,
        conv_state,
        weight,
        bias,
        out,
        cache_seqlens,
        conv_state_indices,
        stream,
        options="--enable-tvm-ffi",
    )


def causal_conv1d_update_cutedsl(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cache_seqlens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.stride(1) != 1 or conv_state.stride(1) != 1:
        raise ValueError("x and conv_state must use channel-last storage")
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    if weight.shape[1] not in (2, 3, 4):
        raise ValueError("width must be 2, 3, or 4")
    if conv_state.shape[2] < weight.shape[1] - 1:
        raise ValueError("conv_state is shorter than the convolution history")
    out = torch.empty_like(x)
    kernel = _compile_update(
        x.dtype,
        weight.dtype,
        weight.shape[1],
        bias is not None,
        activation in ("silu", "swish"),
        cache_seqlens is not None,
        conv_state_indices is not None,
    )
    kernel(x, conv_state, weight, bias, out, cache_seqlens, conv_state_indices)
    return out


class _VarlenStatesKernel:
    THREADS = 128

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        cu_seqlens: cute.Tensor,
        states: cute.Tensor,
        stream: CUstream,
    ):
        self.kernel(x, cu_seqlens, states).launch(
            grid=(states.shape[0], cute.ceil_div(states.shape[1], self.THREADS), 1),
            block=(self.THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        cu_seqlens: cute.Tensor,
        states: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        batch_idx, channel_tile, _ = cute.arch.block_idx()
        channel = channel_tile * self.THREADS + tid
        if channel < states.shape[1]:
            begin = cu_seqlens[batch_idx]
            end = cu_seqlens[batch_idx + 1]
            state_len = states.shape[2]
            for state_pos in range(state_len):
                token = end - state_len + state_pos
                value = x.element_type(0.0)
                if token >= begin:
                    value = x[token, channel]
                states[batch_idx, channel, state_pos] = value


@cache
def _compile_varlen_states(input_dtype: torch.dtype, index_dtype: torch.dtype):
    x_dtype = _TORCH_TO_CUTE[input_dtype]
    idx_dtype = Int32 if index_dtype == torch.int32 else Int64
    tokens = cute.sym_int()
    batch_plus_one = cute.sym_int()
    batch = cute.sym_int()
    dim = cute.sym_int()
    state_len = cute.sym_int()
    x_token_stride = cute.sym_int64(divisibility=1)
    states_batch_stride = cute.sym_int64(divisibility=1)
    states_l_stride = cute.sym_int64(divisibility=1)
    x = _fake_tensor(
        x_dtype,
        (tokens, dim),
        (x_token_stride, 1),
        x_dtype.width // 8,
    )
    cu_seqlens = _fake_tensor(idx_dtype, (batch_plus_one,), (1,), idx_dtype.width // 8)
    states = _fake_tensor(
        x_dtype,
        (batch, dim, state_len),
        (states_batch_stride, 1, states_l_stride),
        x_dtype.width // 8,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _VarlenStatesKernel(),
        x,
        cu_seqlens,
        states,
        stream,
        options="--enable-tvm-ffi",
    )


def causal_conv1d_varlen_states_cutedsl(
    x: torch.Tensor, cu_seqlens: torch.Tensor, state_len: int
) -> torch.Tensor:
    if x.ndim != 2 or x.stride(1) != 1:
        raise ValueError(
            "x must have contiguous channels and shape (total_tokens, dim)"
        )
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must be int32 or int64")
    batch = cu_seqlens.numel() - 1
    states = torch.empty(
        batch, state_len, x.shape[1], device=x.device, dtype=x.dtype
    ).transpose(1, 2)
    _compile_varlen_states(x.dtype, cu_seqlens.dtype)(x, cu_seqlens, states)
    return states


class _BackwardDxKernel:
    """Layout-generic stateless input-gradient kernel."""

    THREADS = 256

    def __init__(self, width: int, has_bias: bool, silu: bool) -> None:
        self.width = width
        self.has_bias = has_bias
        self.silu = silu

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor | None,
        dout: cute.Tensor,
        dx: cute.Tensor,
        stream: CUstream,
    ):
        total = x.shape[0] * x.shape[1] * x.shape[2]
        self.kernel(x, weight, bias, dout, dx).launch(
            grid=(cute.ceil_div(total, self.THREADS), 1, 1),
            block=(self.THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor | None,
        dout: cute.Tensor,
        dx: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        seqlen = x.shape[2]
        dim = x.shape[1]
        linear = block * self.THREADS + tid
        total = x.shape[0] * dim * seqlen
        if linear < total:
            batch_idx = linear // (dim * seqlen)
            rem = linear % (dim * seqlen)
            channel = rem // seqlen
            pos = rem % seqlen
            grad = Float32(0.0)
            for future in cutlass.range_constexpr(self.width):
                out_pos = pos + future
                if out_pos < seqlen:
                    preact = Float32(0.0)
                    if cutlass.const_expr(self.has_bias):
                        preact = Float32(bias[channel])
                    for w in cutlass.range_constexpr(self.width):
                        input_pos = out_pos + w - (self.width - 1)
                        if input_pos >= 0:
                            preact += Float32(weight[channel, w]) * Float32(
                                x[batch_idx, channel, input_pos]
                            )
                    dpreact = Float32(dout[batch_idx, channel, out_pos])
                    if cutlass.const_expr(self.silu):
                        sigmoid = cute.arch.rcp_approx(
                            Float32(1.0) + cute.math.exp(-preact, fastmath=True)
                        )
                        dpreact *= sigmoid * (
                            Float32(1.0) + preact * (Float32(1.0) - sigmoid)
                        )
                    grad += dpreact * Float32(weight[channel, self.width - 1 - future])
            dx[batch_idx, channel, pos] = grad.to(dx.element_type)


class _BackwardWeightKernel:
    """Layout-generic stateless parameter-gradient kernel."""

    THREADS = 128
    WARPS = 4

    def __init__(self, width: int, has_bias: bool, silu: bool) -> None:
        self.width = width
        self.has_bias = has_bias
        self.silu = silu

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor | None,
        dout: cute.Tensor,
        dweight: cute.Tensor,
        dbias: cute.Tensor | None,
        stream: CUstream,
    ):
        self.kernel(x, weight, bias, dout, dweight, dbias).launch(
            grid=(x.shape[1], 1, 1),
            block=(self.THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor | None,
        dout: cute.Tensor,
        dweight: cute.Tensor,
        dbias: cute.Tensor | None,
    ):
        tid, _, _ = cute.arch.thread_idx()
        channel, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        seqlen = x.shape[2]
        count = x.shape[0] * seqlen
        partial = cute.make_rmem_tensor((self.width,), Float32)
        partial.fill(Float32(0.0))
        partial_bias = Float32(0.0)

        for linear in range(tid, count, self.THREADS):
            batch_idx = linear // seqlen
            out_pos = linear % seqlen
            preact = Float32(0.0)
            if cutlass.const_expr(self.has_bias):
                preact = Float32(bias[channel])
            for w in cutlass.range_constexpr(self.width):
                input_pos = out_pos + w - (self.width - 1)
                if input_pos >= 0:
                    preact += Float32(weight[channel, w]) * Float32(
                        x[batch_idx, channel, input_pos]
                    )
            dpreact = Float32(dout[batch_idx, channel, out_pos])
            if cutlass.const_expr(self.silu):
                sigmoid = cute.arch.rcp_approx(
                    Float32(1.0) + cute.math.exp(-preact, fastmath=True)
                )
                dpreact *= sigmoid * (Float32(1.0) + preact * (Float32(1.0) - sigmoid))
            partial_bias += dpreact
            for w in cutlass.range_constexpr(self.width):
                input_pos = out_pos + w - (self.width - 1)
                if input_pos >= 0:
                    partial[w] += dpreact * Float32(x[batch_idx, channel, input_pos])

        smem = cutlass.utils.SmemAllocator()
        reduce_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.width + 1, self.WARPS), stride=(self.WARPS, 1)),
            byte_alignment=16,
        )
        for w in cutlass.range_constexpr(self.width):
            warp_sum = cute.arch.warp_reduction(partial[w], operator.add)
            if lane == 0:
                reduce_buffer[w, warp] = warp_sum
        if cutlass.const_expr(self.has_bias):
            warp_sum = cute.arch.warp_reduction(partial_bias, operator.add)
            if lane == 0:
                reduce_buffer[self.width, warp] = warp_sum
        cute.arch.sync_threads()

        if warp == 0:
            for w in cutlass.range_constexpr(self.width):
                value = Float32(0.0)
                if lane < self.WARPS:
                    value = reduce_buffer[w, lane]
                value = cute.arch.warp_reduction(value, operator.add)
                if lane == 0:
                    dweight[channel, w] = value
            if cutlass.const_expr(self.has_bias):
                value = Float32(0.0)
                if lane < self.WARPS:
                    value = reduce_buffer[self.width, lane]
                value = cute.arch.warp_reduction(value, operator.add)
                if lane == 0:
                    dbias[channel] = value


class _FusedChannelLastBackward:
    """Fused stateless backward for aligned channel-last tensors."""

    ALIGNMENT_BYTES = 16
    CHANNEL_ALIGNMENT = 8
    CHANNEL_TILE_BY_DTYPE = {
        torch.bfloat16: 64,
        torch.float32: 32,
    }
    COPY_BITS = 128
    LONG_TILE_L = 128
    MIN_COMPUTE_CAPABILITY_MAJOR = 8
    SHORT_SEQUENCE_MAX = 128
    SHORT_TILE_L = 64
    THREADS = 128

    @classmethod
    def tile_l_for_seqlen(cls, seqlen: int) -> int:
        if seqlen <= cls.SHORT_SEQUENCE_MAX:
            return cls.SHORT_TILE_L
        return cls.LONG_TILE_L

    def __init__(
        self,
        width: int,
        has_bias: bool,
        silu: bool,
        tile_l: int,
        channel_tile: int,
        channel_vec: int,
        deterministic: bool,
    ) -> None:
        self.width = width
        self.has_bias = has_bias
        self.silu = silu
        self.tile_l = tile_l
        self.channel_tile = channel_tile
        self.channel_vec = channel_vec
        self.channel_group = self.THREADS // channel_tile
        self.deterministic = deterministic
        self.values_per_thread = tile_l // self.channel_group
        vectors_per_row = channel_tile // channel_vec
        rows_per_round = self.THREADS // vectors_per_row
        if (
            width not in (2, 3, 4)
            or self.THREADS % channel_tile != 0
            or channel_tile % channel_vec != 0
            or self.THREADS % vectors_per_row != 0
            or self.channel_group > 32
            or 32 % self.channel_group != 0
            or self.channel_group & (self.channel_group - 1) != 0
            or tile_l % self.channel_group != 0
            or tile_l % rows_per_round != 0
            or width - 1 > rows_per_round
        ):
            raise ValueError("invalid fused backward tile configuration")

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor | None,
        dout: cute.Tensor,
        dx: cute.Tensor,
        dweight_workspace: cute.Tensor,
        dbias_workspace: cute.Tensor | None,
        stream: CUstream,
    ):
        batch, dim, seqlen = x.shape
        self.kernel(
            x,
            weight,
            bias,
            dout,
            dx,
            dweight_workspace,
            dbias_workspace,
        ).launch(
            grid=(
                batch,
                cute.ceil_div(seqlen, self.tile_l),
                cute.ceil_div(dim, self.channel_tile),
            ),
            block=(self.THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor | None,
        dout: cute.Tensor,
        dx: cute.Tensor,
        dweight_workspace: cute.Tensor,
        dbias_workspace: cute.Tensor | None,
    ):
        tid, _, _ = cute.arch.thread_idx()
        batch_idx, tile_l_idx, tile_c_idx = cute.arch.block_idx()
        dim = x.shape[1]
        seqlen = x.shape[2]
        halo = self.width - 1
        l_base = tile_l_idx * self.tile_l
        c_base = tile_c_idx * self.channel_tile

        smem = cutlass.utils.SmemAllocator()
        x_rows = self.tile_l + 2 * halo
        dout_rows = self.tile_l + halo
        sx = smem.allocate_tensor(
            x.element_type,
            cute.make_layout(
                (x_rows, self.channel_tile),
                stride=(self.channel_tile + self.channel_vec, 1),
            ),
            byte_alignment=self.ALIGNMENT_BYTES,
        )
        sdout = smem.allocate_tensor(
            x.element_type,
            cute.make_layout(
                (dout_rows, self.channel_tile),
                stride=(self.channel_tile + self.channel_vec, 1),
            ),
            byte_alignment=self.ALIGNMENT_BYTES,
        )

        vectors_per_row = self.channel_tile // self.channel_vec
        rows_per_round = self.THREADS // vectors_per_row
        row_in_round = tid // vectors_per_row
        vector_in_row = tid % vectors_per_row
        vector_global_c = c_base + vector_in_row * self.channel_vec
        source_vector_idx = c_base // self.channel_vec + vector_in_row
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            x.element_type,
            num_bits_per_copy=self.COPY_BITS,
        )

        # Invalid causal halo and residue positions contribute zero. Clear the
        # shared tiles before issuing only the valid 16-byte async copies.
        for idx in range(tid, x_rows * self.channel_tile, self.THREADS):
            sx[idx // self.channel_tile, idx % self.channel_tile] = x.element_type(
                0.0
            )
        for idx in range(tid, dout_rows * self.channel_tile, self.THREADS):
            sdout[idx // self.channel_tile, idx % self.channel_tile] = (
                x.element_type(0.0)
            )
        cute.arch.sync_threads()

        for row_base in cutlass.range_constexpr(
            0, self.tile_l, rows_per_round
        ):
            local_l = row_base + row_in_round
            global_l = l_base + local_l
            if global_l < seqlen and vector_global_c < dim:
                cute.copy(
                    copy_atom,
                    cute.local_tile(
                        x[batch_idx, None, global_l],
                        (self.channel_vec,),
                        (source_vector_idx,),
                    ),
                    cute.local_tile(
                        sx[halo + local_l, None],
                        (self.channel_vec,),
                        (vector_in_row,),
                    ),
                )
                cute.copy(
                    copy_atom,
                    cute.local_tile(
                        dout[batch_idx, None, global_l],
                        (self.channel_vec,),
                        (source_vector_idx,),
                    ),
                    cute.local_tile(
                        sdout[local_l, None],
                        (self.channel_vec,),
                        (vector_in_row,),
                    ),
                )

        if row_in_round < halo:
            left_l = l_base + row_in_round - halo
            if left_l >= 0 and vector_global_c < dim:
                cute.copy(
                    copy_atom,
                    cute.local_tile(
                        x[batch_idx, None, left_l],
                        (self.channel_vec,),
                        (source_vector_idx,),
                    ),
                    cute.local_tile(
                        sx[row_in_round, None],
                        (self.channel_vec,),
                        (vector_in_row,),
                    ),
                )

            right_l = l_base + self.tile_l + row_in_round
            if right_l < seqlen and vector_global_c < dim:
                cute.copy(
                    copy_atom,
                    cute.local_tile(
                        dout[batch_idx, None, right_l],
                        (self.channel_vec,),
                        (source_vector_idx,),
                    ),
                    cute.local_tile(
                        sdout[self.tile_l + row_in_round, None],
                        (self.channel_vec,),
                        (vector_in_row,),
                    ),
                )
                if cutlass.const_expr(self.silu):
                    cute.copy(
                        copy_atom,
                        cute.local_tile(
                            x[batch_idx, None, right_l],
                            (self.channel_vec,),
                            (source_vector_idx,),
                        ),
                        cute.local_tile(
                            sx[halo + self.tile_l + row_in_round, None],
                            (self.channel_vec,),
                            (vector_in_row,),
                        ),
                    )

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        channel_in_tile = tid // self.channel_group
        sequence_part = tid % self.channel_group
        channel = c_base + channel_in_tile
        sequence_base = sequence_part * self.values_per_thread

        weight_values = cute.make_rmem_tensor((self.width,), Float32)
        for w in cutlass.range_constexpr(self.width):
            weight_values[w] = Float32(0.0)
        bias_value = Float32(0.0)
        if channel < dim:
            for w in cutlass.range_constexpr(self.width):
                weight_values[w] = Float32(weight[channel, w])
            if cutlass.const_expr(self.has_bias):
                bias_value = Float32(bias[channel])

        dout_values = cute.make_rmem_tensor(
            (halo + self.values_per_thread,), Float32
        )
        x_values = cute.make_rmem_tensor(
            (2 * halo + self.values_per_thread,), Float32
        )
        for i in cutlass.range_constexpr(halo + self.values_per_thread):
            dout_values[i] = Float32(sdout[sequence_base + i, channel_in_tile])
            x_values[i] = Float32(sx[sequence_base + i, channel_in_tile])

        if cutlass.const_expr(self.silu):
            for i in cutlass.range_constexpr(
                halo + self.values_per_thread,
                2 * halo + self.values_per_thread,
            ):
                x_values[i] = Float32(sx[sequence_base + i, channel_in_tile])
            for i in cutlass.range_constexpr(self.values_per_thread + halo):
                preact = bias_value
                for w in cutlass.range_constexpr(self.width):
                    preact += weight_values[w] * x_values[i + w]
                sigmoid = cute.arch.rcp_approx(
                    Float32(1.0) + cute.math.exp(-preact, fastmath=True)
                )
                dout_values[i] *= sigmoid * (
                    Float32(1.0) + preact * (Float32(1.0) - sigmoid)
                )

        dweight_partial = cute.make_rmem_tensor((self.width,), Float32)
        for w in cutlass.range_constexpr(self.width):
            dweight_partial[w] = Float32(0.0)
        dbias_partial = Float32(0.0)
        for i in cutlass.range_constexpr(self.values_per_thread):
            grad = dout_values[i]
            dbias_partial += grad
            for w in cutlass.range_constexpr(self.width):
                dweight_partial[w] += x_values[i + w] * grad

        for w in cutlass.range_constexpr(self.width):
            dweight_partial[w] = cute.arch.warp_reduction(
                dweight_partial[w],
                operator.add,
                threads_in_group=self.channel_group,
            )
        if cutlass.const_expr(self.has_bias):
            dbias_partial = cute.arch.warp_reduction(
                dbias_partial,
                operator.add,
                threads_in_group=self.channel_group,
            )

        if sequence_part == 0 and channel < dim:
            if cutlass.const_expr(self.deterministic):
                for w in cutlass.range_constexpr(self.width):
                    dweight_workspace[batch_idx, tile_l_idx, channel, w] = (
                        dweight_partial[w]
                    )
                if cutlass.const_expr(self.has_bias):
                    dbias_workspace[batch_idx, tile_l_idx, channel] = dbias_partial
            else:
                for w in cutlass.range_constexpr(self.width):
                    cute.arch.atomic_add(
                        dweight_workspace.iterator
                        + dweight_workspace.layout((channel, w)),
                        dweight_partial[w],
                        scope="gpu",
                    )
                if cutlass.const_expr(self.has_bias):
                    cute.arch.atomic_add(
                        dbias_workspace.iterator + dbias_workspace.layout(channel),
                        dbias_partial,
                        scope="gpu",
                    )

        dx_values = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        for i in cutlass.range_constexpr(self.values_per_thread):
            grad = Float32(0.0)
            for w in cutlass.range_constexpr(self.width):
                grad += (
                    weight_values[self.width - 1 - w] * dout_values[i + w]
                )
            dx_values[i] = grad

        # Reuse sx only after every thread has finished reading the input tile.
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(self.values_per_thread):
            sx[halo + sequence_base + i, channel_in_tile] = dx_values[i].to(
                x.element_type
            )
        cute.arch.sync_threads()

        egress_c = c_base + tid % self.channel_tile
        first_egress_row = tid // self.channel_tile
        for row in range(
            first_egress_row,
            self.tile_l,
            self.THREADS // self.channel_tile,
        ):
            global_l = l_base + row
            if global_l < seqlen and egress_c < dim:
                dx[batch_idx, egress_c, global_l] = sx[
                    halo + row, tid % self.channel_tile
                ]


class _FusedReduceGradients:
    """Reduce deterministic fused-backward workspaces in a fixed order."""

    THREADS = 128

    def __init__(self, width: int, has_bias: bool) -> None:
        self.width = width
        self.has_bias = has_bias

    @cute.jit
    def __call__(
        self,
        dweight_workspace: cute.Tensor,
        dbias_workspace: cute.Tensor | None,
        dweight: cute.Tensor,
        dbias: cute.Tensor | None,
        stream: CUstream,
    ):
        dim = dweight.shape[0]
        self.kernel(dweight_workspace, dbias_workspace, dweight, dbias).launch(
            grid=(cute.ceil_div(dim, self.THREADS), 1, 1),
            block=(self.THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        dweight_workspace: cute.Tensor,
        dbias_workspace: cute.Tensor | None,
        dweight: cute.Tensor,
        dbias: cute.Tensor | None,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        channel = block * self.THREADS + tid
        dim = dweight.shape[0]
        batch = dweight_workspace.shape[0]
        sequence_tiles = dweight_workspace.shape[1]
        if channel < dim:
            for w in cutlass.range_constexpr(self.width):
                value = Float32(0.0)
                for batch_idx in range(batch):
                    for tile_idx in range(sequence_tiles):
                        value += dweight_workspace[
                            batch_idx, tile_idx, channel, w
                        ]
                dweight[channel, w] = value
            if cutlass.const_expr(self.has_bias):
                value = Float32(0.0)
                for batch_idx in range(batch):
                    for tile_idx in range(sequence_tiles):
                        value += dbias_workspace[batch_idx, tile_idx, channel]
                dbias[channel] = value


@cache
def _compile_backward(
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    width: int,
    has_bias: bool,
    silu: bool,
):
    x_dtype = _TORCH_TO_CUTE[input_dtype]
    w_dtype = _TORCH_TO_CUTE[weight_dtype]
    batch, dim, seqlen = cute.sym_int(), cute.sym_int(), cute.sym_int()
    x_strides = tuple(cute.sym_int64(divisibility=1) for _ in range(3))
    dout_strides = tuple(cute.sym_int64(divisibility=1) for _ in range(3))
    dx_strides = tuple(cute.sym_int64(divisibility=1) for _ in range(3))
    x = _fake_tensor(x_dtype, (batch, dim, seqlen), x_strides, x_dtype.width // 8)
    dout = _fake_tensor(x_dtype, (batch, dim, seqlen), dout_strides, x_dtype.width // 8)
    dx = _fake_tensor(x_dtype, (batch, dim, seqlen), dx_strides, x_dtype.width // 8)
    weight = _fake_tensor(w_dtype, (dim, width), (width, 1), w_dtype.width // 8)
    bias = _fake_tensor(w_dtype, (dim,), (1,), w_dtype.width // 8) if has_bias else None
    dweight = _fake_tensor(Float32, (dim, width), (width, 1), 4)
    dbias = _fake_tensor(Float32, (dim,), (1,), 4) if has_bias else None
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    dx_kernel = cute.compile(
        _BackwardDxKernel(width, has_bias, silu),
        x,
        weight,
        bias,
        dout,
        dx,
        stream,
        options="--enable-tvm-ffi",
    )
    weight_kernel = cute.compile(
        _BackwardWeightKernel(width, has_bias, silu),
        x,
        weight,
        bias,
        dout,
        dweight,
        dbias,
        stream,
        options="--enable-tvm-ffi",
    )
    return dx_kernel, weight_kernel


def _fake_deterministic_backward_workspaces(
    batch,
    sequence_tiles,
    dim,
    width: int,
    has_bias: bool,
):
    """Build dynamic fake tensors shared by fused compile entry points."""
    dweight_workspace = _fake_tensor(
        Float32,
        (batch, sequence_tiles, dim, width),
        tuple(cute.sym_int64(divisibility=1) for _ in range(4)),
        4,
    )
    dbias_workspace = (
        _fake_tensor(
            Float32,
            (batch, sequence_tiles, dim),
            tuple(cute.sym_int64(divisibility=1) for _ in range(3)),
            4,
        )
        if has_bias
        else None
    )
    return dweight_workspace, dbias_workspace


@cache
def _compile_fused_channel_last_backward(
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    width: int,
    has_bias: bool,
    silu: bool,
    tile_l: int,
    deterministic: bool,
):
    x_dtype = _TORCH_TO_CUTE[input_dtype]
    weight_dtype_cute = _TORCH_TO_CUTE[weight_dtype]
    if input_dtype not in _FusedChannelLastBackward.CHANNEL_TILE_BY_DTYPE:
        raise ValueError("fused backward only supports bfloat16 and float32 inputs")
    channel_tile = _FusedChannelLastBackward.CHANNEL_TILE_BY_DTYPE[input_dtype]
    channel_vec = _FusedChannelLastBackward.COPY_BITS // x_dtype.width
    if channel_vec * x_dtype.width != _FusedChannelLastBackward.COPY_BITS:
        raise ValueError("fused backward requires exact 128-bit input vectors")

    batch, dim, seqlen = cute.sym_int(), cute.sym_int(), cute.sym_int()
    x_batch_stride = cute.sym_int64(divisibility=channel_vec)
    x_seqlen_stride = cute.sym_int64(divisibility=channel_vec)
    x = _fake_tensor(
        x_dtype,
        (batch, dim, seqlen),
        (x_batch_stride, 1, x_seqlen_stride),
        _FusedChannelLastBackward.ALIGNMENT_BYTES,
    )
    dout_batch_stride = cute.sym_int64(divisibility=channel_vec)
    dout_seqlen_stride = cute.sym_int64(divisibility=channel_vec)
    dout = _fake_tensor(
        x_dtype,
        (batch, dim, seqlen),
        (dout_batch_stride, 1, dout_seqlen_stride),
        _FusedChannelLastBackward.ALIGNMENT_BYTES,
    )
    dx_batch_stride = cute.sym_int64(divisibility=channel_vec)
    dx_seqlen_stride = cute.sym_int64(divisibility=channel_vec)
    dx = _fake_tensor(
        x_dtype,
        (batch, dim, seqlen),
        (dx_batch_stride, 1, dx_seqlen_stride),
        _FusedChannelLastBackward.ALIGNMENT_BYTES,
    )
    weight = _fake_tensor(
        weight_dtype_cute,
        (dim, width),
        (width, 1),
        weight_dtype_cute.width // 8,
    )
    bias = (
        _fake_tensor(
            weight_dtype_cute,
            (dim,),
            (1,),
            weight_dtype_cute.width // 8,
        )
        if has_bias
        else None
    )

    if deterministic:
        sequence_tiles = cute.sym_int()
        dweight_workspace, dbias_workspace = (
            _fake_deterministic_backward_workspaces(
                batch,
                sequence_tiles,
                dim,
                width,
                has_bias,
            )
        )
    else:
        dweight_workspace = _fake_tensor(
            Float32, (dim, width), (width, 1), 4
        )
        dbias_workspace = (
            _fake_tensor(Float32, (dim,), (1,), 4) if has_bias else None
        )

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _FusedChannelLastBackward(
            width,
            has_bias,
            silu,
            tile_l,
            channel_tile,
            channel_vec,
            deterministic,
        ),
        x,
        weight,
        bias,
        dout,
        dx,
        dweight_workspace,
        dbias_workspace,
        stream,
        options="--enable-tvm-ffi",
    )


@cache
def _compile_fused_reduce_gradients(width: int, has_bias: bool):
    batch, sequence_tiles, dim = cute.sym_int(), cute.sym_int(), cute.sym_int()
    dweight_workspace, dbias_workspace = (
        _fake_deterministic_backward_workspaces(
            batch,
            sequence_tiles,
            dim,
            width,
            has_bias,
        )
    )
    dweight = _fake_tensor(Float32, (dim, width), (width, 1), 4)
    dbias = _fake_tensor(Float32, (dim,), (1,), 4) if has_bias else None
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _FusedReduceGradients(width, has_bias),
        dweight_workspace,
        dbias_workspace,
        dweight,
        dbias,
        stream,
        options="--enable-tvm-ffi",
    )


def _use_deterministic_mode() -> bool:
    """Match the native CUDA environment/global deterministic selector."""
    value = os.environ.get("CAUSAL_CONV1D_DETERMINISTIC")
    if value:
        if value[0] == "1":
            return True
        if value[0] == "0":
            return False
    return torch.are_deterministic_algorithms_enabled()


def _can_use_fused_channel_last_backward(
    x: torch.Tensor,
    dout: torch.Tensor,
    dx: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy every fused 128-bit copy contract."""
    tensors = (x, dout, dx)
    if (
        x.ndim != 3
        or x.dtype not in (torch.bfloat16, torch.float32)
        or any(t.ndim != 3 for t in tensors)
        or any(t.shape != x.shape for t in tensors)
        or any(t.dtype != x.dtype for t in tensors)
        or any(not t.is_cuda or t.device != x.device for t in tensors)
        or x.shape[1] % _FusedChannelLastBackward.CHANNEL_ALIGNMENT != 0
        or any(t.stride(1) != 1 for t in tensors)
    ):
        return False

    if (
        torch.cuda.get_device_capability(x.device)[0]
        < _FusedChannelLastBackward.MIN_COMPUTE_CAPABILITY_MAJOR
    ):
        return False

    alignment = _FusedChannelLastBackward.ALIGNMENT_BYTES
    vector_elems = alignment // x.element_size()
    return all(
        t.data_ptr() % alignment == 0
        and t.stride(0) % vector_elems == 0
        and t.stride(2) % vector_elems == 0
        for t in tensors
    )


def _launch_fused_channel_last_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    dx: torch.Tensor,
    silu_activation: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Allocate gradients and launch the fused channel-last specialization."""
    width = weight.shape[1]
    has_bias = bias is not None
    tile_l = _FusedChannelLastBackward.tile_l_for_seqlen(x.shape[2])
    deterministic = _use_deterministic_mode()
    fused_kernel = _compile_fused_channel_last_backward(
        x.dtype,
        weight.dtype,
        width,
        has_bias,
        silu_activation,
        tile_l,
        deterministic,
    )

    if deterministic:
        sequence_tiles = (x.shape[2] + tile_l - 1) // tile_l
        dweight_workspace = torch.empty(
            (x.shape[0], sequence_tiles, x.shape[1], width),
            device=x.device,
            dtype=torch.float32,
        )
        dbias_workspace = (
            torch.empty(
                (x.shape[0], sequence_tiles, x.shape[1]),
                device=x.device,
                dtype=torch.float32,
            )
            if has_bias
            else None
        )
        dweight = torch.empty(
            (x.shape[1], width),
            device=x.device,
            dtype=torch.float32,
        )
        dbias = (
            torch.empty(x.shape[1], device=x.device, dtype=torch.float32)
            if has_bias
            else None
        )
        fused_kernel(
            x,
            weight,
            bias,
            dout,
            dx,
            dweight_workspace,
            dbias_workspace,
        )
        reduce_kernel = _compile_fused_reduce_gradients(width, has_bias)
        reduce_kernel(
            dweight_workspace,
            dbias_workspace,
            dweight,
            dbias,
        )
        return dweight, dbias

    # The fast reducer uses atomics, so its destinations must start at zero.
    dweight = torch.zeros(
        (x.shape[1], width),
        device=x.device,
        dtype=torch.float32,
    )
    dbias = (
        torch.zeros(x.shape[1], device=x.device, dtype=torch.float32)
        if has_bias
        else None
    )
    fused_kernel(x, weight, bias, dout, dx, dweight, dbias)
    return dweight, dbias


def _launch_generic_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    dx: torch.Tensor,
    silu_activation: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch the layout-generic two-kernel backward fallback."""
    has_bias = bias is not None
    dweight = torch.empty_like(weight, dtype=torch.float32)
    dbias = torch.empty_like(bias, dtype=torch.float32) if has_bias else None
    dx_kernel, weight_kernel = _compile_backward(
        x.dtype,
        weight.dtype,
        weight.shape[1],
        has_bias,
        silu_activation,
    )
    dx_kernel(x, weight, bias, dout, dx)
    weight_kernel(x, weight, bias, dout, dweight, dbias)
    return dweight, dbias


def causal_conv1d_bwd_cutedsl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    silu_activation: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run stateless backward through the fused path or generic fallback."""
    dx = torch.empty_like(x)
    if _can_use_fused_channel_last_backward(x, dout, dx):
        dweight, dbias = _launch_fused_channel_last_backward(
            x,
            weight,
            bias,
            dout,
            dx,
            silu_activation,
        )
    else:
        dweight, dbias = _launch_generic_backward(
            x,
            weight,
            bias,
            dout,
            dx,
            silu_activation,
        )
    return (
        dx,
        dweight.to(weight.dtype),
        (dbias.to(bias.dtype) if dbias is not None else None),
    )
