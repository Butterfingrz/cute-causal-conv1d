"""CuTe DSL kernels for causal depthwise convolution.

The kernels in this module are compiled lazily by the CuTe DSL runtime.  There is
no C++ or CUDA extension build step.
"""

from __future__ import annotations

from functools import cache
import operator

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


def causal_conv1d_bwd_cutedsl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    silu_activation: bool,
):
    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight, dtype=torch.float32)
    dbias = torch.empty_like(bias, dtype=torch.float32) if bias is not None else None
    dx_kernel, weight_kernel = _compile_backward(
        x.dtype, weight.dtype, weight.shape[1], bias is not None, silu_activation
    )
    dx_kernel(x, weight, bias, dout, dx)
    weight_kernel(x, weight, bias, dout, dweight, dbias)
    return (
        dx,
        dweight.to(weight.dtype),
        (dbias.to(bias.dtype) if dbias is not None else None),
    )
