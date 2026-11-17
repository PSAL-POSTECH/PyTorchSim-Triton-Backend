"""PyTorchSim's _depthwise_kernel (stride 2, pad 1): the tap left of column 0 reads x[-1], masked off, and with x
placed at the start of DRAM that element is below DRAM; a masked-off element must not be read (Spike: User load
segfault @ 0x7ffffffc). tests/ops/conv/test_conv_decomposition.py in PyTorchSim, at its shape."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, C, H, W, K, S, P = 2, 16, 15, 15, 3, 2, 1
HO = WO = (H + 2 * P - K) // S + 1
BC, BW = 128, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_depthwise_pad_read_before_the_tensor_kernel(
            x_ptr, w_ptr, b_ptr, o_ptr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, HO: tl.constexpr,
            WO: tl.constexpr, K: tl.constexpr, S: tl.constexpr, P: tl.constexpr, BC: tl.constexpr,
            BW: tl.constexpr):
        pid = tl.program_id(0)
        oh = tl.program_id(1)
        cblocks = tl.cdiv(C, BC)
        n = pid // cblocks
        cc = (pid % cblocks) * BC + tl.arange(0, BC)
        ow = tl.arange(0, BW)
        live_c = cc < C
        live_w = ow < WO
        acc = tl.zeros((BC, BW), dtype=tl.float32)
        for i in range(K):
            ih = oh * S + i - P
            for j in range(K):
                iw = ow * S + j - P
                ok = live_c[:, None] & (live_w & (iw >= 0) & (iw < W))[None, :] & (ih >= 0) & (ih < H)
                xv = tl.load(x_ptr + (n * C + cc[:, None]) * (H * W) + ih * W + iw[None, :], mask=ok, other=0.0)
                wv = tl.load(w_ptr + cc * (K * K) + i * K + j, mask=live_c, other=0.0)
                acc += xv * wv[:, None]
        acc += tl.load(b_ptr + cc, mask=live_c, other=0.0)[:, None]
        tl.store(o_ptr + (n * C + cc[:, None]) * (HO * WO) + oh * WO + ow[None, :], acc,
                 mask=live_c[:, None] & live_w[None, :])

    return mask_depthwise_pad_read_before_the_tensor_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N * C * H * W, generator=g, dtype=torch.float32),
            "w": torch.randn(C * K * K, generator=g, dtype=torch.float32),
            "b": torch.randn(C, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(N, C, H, W)
    w = inputs["w"].reshape(C, 1, K, K)
    o = torch.nn.functional.conv2d(x, w, inputs["b"], stride=S, padding=P, groups=C)
    return {"o": o.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="mask_depthwise_pad_read_before_the_tensor_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "C": "constexpr", "H": "constexpr", "W": "constexpr", "HO": "constexpr", "WO": "constexpr",
               "K": "constexpr", "S": "constexpr", "P": "constexpr", "BC": "constexpr", "BW": "constexpr"},
    constexprs={"C": C, "H": H, "W": W, "HO": HO, "WO": WO, "K": K, "S": S, "P": P, "BC": BC, "BW": BW},
    args=[Arg("x", IN, "float32", (N * C * H * W,)), Arg("w", IN, "float32", (C * K * K,)),
          Arg("b", IN, "float32", (C,)), Arg("o", OUT, "float32", (N * C * HO * WO,))],
    grid=(N, HO, 1),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
