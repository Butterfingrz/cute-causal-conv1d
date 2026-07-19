# causal-conv1d: CuTe DSL rewrite

This repository is an API-compatible rewrite of `causal-conv1d` using
[NVIDIA CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).
It replaces the CUDA/C++ extension with Python-authored GPU kernels while
preserving the public PyTorch interface, tensor layouts, state handling, and
autograd behavior.

The rewrite supports FP32, FP16, and BF16 inputs and kernel widths 2, 3, and 4.
Existing code using `causal_conv1d_fn`, `causal_conv1d_update`, or
`causal_conv1d_varlen_states` can keep the same call signatures and tensor
formats.

## Installation

```bash
pip install .
```

Installation produces a pure-Python wheel and does not invoke a C++ compiler,
`nvcc`, Ninja, or a custom CUDA extension build. CuTe DSL specializes and caches
GPU kernels on first use for the active CUDA environment.

Requirements:

- PyTorch with CUDA support
- `nvidia-cutlass-dsl`
- NVIDIA GPU supported by the installed CuTe DSL release

## API compatibility

```python
from causal_conv1d import causal_conv1d_fn

out = causal_conv1d_fn(x, weight, bias, activation="silu")
```

- `x`: `(batch, dim, seqlen)`
- `weight`: `(dim, width)`
- `bias`: optional `(dim,)`
- `activation`: `None`, `"silu"`, or `"swish"`

The operation is equivalent to:

```python
import torch.nn.functional as F

out = F.conv1d(
    x,
    weight.unsqueeze(1),
    bias,
    padding=weight.shape[1] - 1,
    groups=x.shape[1],
)[..., : x.shape[-1]]
```

The package also provides `causal_conv1d_update` for convolution-state updates
and `causal_conv1d_varlen_states` for packed variable-length state extraction.

## Validation

The numerical and performance harnesses are in `tests/`:

```bash
pytest -q tests/test_causal_conv1d.py
python tests/benchmark_cutedsl_forward.py \
  --cuda-graph --warmup 100 --iterations 1000 --repeats 5
```

The benchmark intentionally imports an installed `causal_conv1d_cuda` extension
as the direct baseline; it is not a runtime dependency of this package. CUDA
Graph replay removes Python and TVM-FFI launch overhead so this command compares
the generated GPU kernels. Each result is the median of alternating-order runs.

### H100 results

On NVIDIA H100, the CUDA Graph benchmark covered FP16 and BF16, kernel widths
2-4, and eight representative tensor shapes (48 cases total). The CuTe DSL
kernels matched or exceeded the installed CUDA extension in every case, with
measured speedups from 1.008x to 1.199x.

Nsight Compute measurements also covered both a latency-oriented case and a
larger bandwidth-oriented case:

| dtype | width | shape `(B, D, L)` | CUDA extension | CuTe DSL | speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| FP16 | 3 | `(1, 2048, 2048)` | 9.70 us | 9.28 us | 1.045x |
| BF16 | 4 | `(1, 4096, 2048)` | 15.74 us | 15.58 us | 1.010x |

Both profiled CuTe DSL kernels had zero register spills. The dense numerical
suite completed with 6,484 passing cases, 3,888 expected skips, and 2 expected
xfails; dedicated update and packed variable-length suites added 1,620 and
1,296 passing cases, respectively.
