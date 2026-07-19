import argparse

import torch

import causal_conv1d_cuda
from causal_conv1d.cutedsl_kernels import causal_conv1d_fwd_cutedsl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("implementation", choices=("original", "cutedsl"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--width", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(
        args.batch,
        args.seqlen,
        args.dim,
        device="cuda",
        dtype=getattr(torch, args.dtype),
    ).transpose(1, 2)
    weight = torch.randn(args.dim, args.width, device="cuda", dtype=torch.float32)
    bias = torch.randn(args.dim, device="cuda", dtype=torch.float32)

    if args.implementation == "original":

        def fn():
            out = torch.empty_like(x)
            causal_conv1d_cuda.causal_conv1d_fwd(
                x, weight, bias, None, None, out, None, True
            )
            return out
    else:

        def fn():
            return causal_conv1d_fwd_cutedsl(x, weight, bias, "silu")

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iterations):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
