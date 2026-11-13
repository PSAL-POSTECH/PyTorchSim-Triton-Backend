"""BERT's attention output projection: a[m, k] = attn[((k + K*m) // D) % H, m, k % D] under a K tile of 1024 > K = H*D,
so the head axis (16 slots) wraps at H = 12 only past the mask; counting laps over 16 made a cut that was refused onto
the dot's tile of digits (DigitRefused, since PTB #398). A head-merge read past its mask: that wrap is no cut."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, H, D, N, BK = 32, 12, 64, 128, 1024     # rows, heads, head width, out columns, K tile (> H*D)
K = H * D


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_div_rem_wrap_past_mask_into_dot_kernel(a_ptr, w_ptr, o_ptr,
                                                   M: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                                                   N: tl.constexpr, BK: tl.constexpr):
        m = tl.arange(0, M)
        k = tl.arange(0, BK)
        n = tl.arange(0, N)
        x = k[None, :] + H * D * m[:, None]
        a = tl.load(a_ptr + x % D + D * m[:, None] + M * D * ((x // D) % H),
                    mask=k[None, :] < H * D, other=0.0)
        w = tl.load(w_ptr + k[:, None] * N + n[None, :], mask=k[:, None] < H * D, other=0.0)
        tl.store(o_ptr + m[:, None] * N + n[None, :], tl.dot(a, w))

    return ptr_div_rem_wrap_past_mask_into_dot_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(H * M * D, generator=g, dtype=torch.float32),
            "w": torch.randn(K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    a = inputs["a"].reshape(H, M, D).permute(1, 0, 2).reshape(M, K)
    return {"o": (a @ inputs["w"].reshape(K, N)).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="ptr_div_rem_wrap_past_mask_into_dot_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "H": "constexpr", "D": "constexpr", "N": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "H": H, "D": D, "N": N, "BK": BK},
    args=[Arg("a", IN, "float32", (H * M * D,)), Arg("w", IN, "float32", (K * N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
