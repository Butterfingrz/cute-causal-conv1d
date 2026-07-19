import argparse
import csv
import statistics
import sys

import torch

import causal_conv1d_cuda
from causal_conv1d.cutedsl_kernels import causal_conv1d_fwd_cutedsl


SHAPES = (
    (1, 768, 128),
    (1, 768, 2048),
    (1, 2048, 512),
    (1, 2048, 2048),
    (1, 4096, 128),
    (1, 4096, 2048),
    (2, 2048, 2048),
    (8, 2048, 512),
)


def time_cuda(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--width", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--no-silu", action="store_true")
    parser.add_argument("--csv")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--shape",
        action="append",
        nargs=3,
        type=int,
        metavar=("BATCH", "DIM", "SEQLEN"),
        help="benchmark only this shape; may be specified more than once",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    activation = None if args.no_silu else "silu"
    rows = []
    shapes = tuple(map(tuple, args.shape)) if args.shape else SHAPES
    for batch, dim, seqlen in shapes:
        torch.manual_seed(batch * 1_000_000 + dim * 100 + seqlen)
        x = torch.randn(batch, seqlen, dim, device="cuda", dtype=dtype).transpose(1, 2)
        weight = torch.randn(dim, args.width, device="cuda", dtype=torch.float32)
        bias = torch.randn(dim, device="cuda", dtype=torch.float32)

        def original():
            out = torch.empty_like(x)
            causal_conv1d_cuda.causal_conv1d_fwd(
                x, weight, bias, None, None, out, None, activation is not None
            )
            return out

        def cutedsl():
            return causal_conv1d_fwd_cutedsl(x, weight, bias, activation)

        actual = cutedsl()
        expected = original()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)
        original_timed, cutedsl_timed = original, cutedsl
        if args.cuda_graph:
            original_graph = torch.cuda.CUDAGraph()
            cutedsl_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(original_graph):
                original()
            with torch.cuda.graph(cutedsl_graph):
                cutedsl()
            original_timed = original_graph.replay
            cutedsl_timed = cutedsl_graph.replay
        original_samples = []
        cutedsl_samples = []
        # Alternate measurement order so thermal/clock drift cannot
        # systematically favor either implementation.
        for repeat in range(args.repeats):
            if repeat % 2 == 0:
                original_samples.append(
                    time_cuda(original_timed, args.warmup, args.iterations)
                )
                cutedsl_samples.append(
                    time_cuda(cutedsl_timed, args.warmup, args.iterations)
                )
            else:
                cutedsl_samples.append(
                    time_cuda(cutedsl_timed, args.warmup, args.iterations)
                )
                original_samples.append(
                    time_cuda(original_timed, args.warmup, args.iterations)
                )
        original_us = statistics.median(original_samples)
        cutedsl_us = statistics.median(cutedsl_samples)
        row = {
            "batch": batch,
            "dim": dim,
            "seqlen": seqlen,
            "dtype": args.dtype,
            "width": args.width,
            "activation": activation or "none",
            "original_us": original_us,
            "cutedsl_us": cutedsl_us,
            "original_min_us": min(original_samples),
            "original_max_us": max(original_samples),
            "cutedsl_min_us": min(cutedsl_samples),
            "cutedsl_max_us": max(cutedsl_samples),
            "speedup": original_us / cutedsl_us,
        }
        rows.append(row)
        print(
            f"B={batch:<2} D={dim:<4} L={seqlen:<4} "
            f"original={original_us:8.3f} us cutedsl={cutedsl_us:8.3f} us "
            f"speedup={original_us / cutedsl_us:.3f}x",
            flush=True,
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    if any(row["speedup"] < 1.0 for row in rows):
        sys.exit(2)


if __name__ == "__main__":
    main()
