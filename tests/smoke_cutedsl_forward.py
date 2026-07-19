import torch

from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref
from causal_conv1d.cutedsl_kernels import causal_conv1d_fwd_cutedsl


def main():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for width in (2, 3, 4):
            for seqlen in (1, 63, 64, 65, 129):
                x = torch.randn(2, seqlen, 96, device="cuda", dtype=dtype).transpose(
                    1, 2
                )
                weight = torch.randn(96, width, device="cuda", dtype=torch.float32)
                bias = torch.randn(96, device="cuda", dtype=torch.float32)
                for activation in (None, "silu"):
                    out = causal_conv1d_fwd_cutedsl(x, weight, bias, activation)
                    ref = causal_conv1d_ref(x, weight, bias, activation=activation)
                    torch.testing.assert_close(
                        out,
                        ref,
                        rtol=3e-4 if dtype == torch.float32 else 1e-2,
                        atol=1e-3 if dtype == torch.float32 else 5e-2,
                    )
    print("CuTe DSL forward smoke: PASS")


if __name__ == "__main__":
    main()
