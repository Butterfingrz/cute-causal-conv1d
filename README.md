# Causal depthwise conv1d in CuTe DSL

This package implements causal depthwise 1D convolution for PyTorch with
[NVIDIA CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).
It supports FP32, FP16, and BF16 inputs and kernel widths 2, 3, and 4.

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

## API

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
