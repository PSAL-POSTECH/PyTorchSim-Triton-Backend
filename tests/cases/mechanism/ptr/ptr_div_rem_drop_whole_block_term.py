"""`(k + H*D*m) // D % H` and `(k + H*D*m) % D`: the row term H*D*m is whole laps of both moduli, so the
address is the affine [m, h, d] walk of a [H, M, D] tensor read as [M, H*D] (Inductor's transpose fused
into an mm A load). Breaks if visitOperandDiv refuses a second axis whose stride the divisor divides."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, H, D, BM = 16, 4, 32, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_div_rem_drop_whole_block_term_kernel(x_ptr, o_ptr, M: tl.constexpr, H: tl.constexpr,
                                                  D: tl.constexpr, BM: tl.constexpr):
        m = tl.program_id(0) * BM + tl.arange(0, BM)
        k = tl.arange(0, H * D)
        xindex = k[None, :] + (H * D) * m[:, None]
        x10 = (xindex // D) % H
        x9 = xindex % D
        v = tl.load(x_ptr + (x9 + D * m[:, None] + (M * D) * x10))
        tl.store(o_ptr + xindex, v)

    return ptr_div_rem_drop_whole_block_term_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(H * M * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].reshape(H, M, D).permute(1, 0, 2).reshape(-1).clone()}


SPEC = KernelSpec(
    name="ptr_div_rem_drop_whole_block_term_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "H": "constexpr",
               "D": "constexpr", "BM": "constexpr"},
    constexprs={"M": M, "H": H, "D": D, "BM": BM},
    args=[Arg("x", IN, "float32", (H * M * D,)), Arg("o", OUT, "float32", (M * H * D,))],
    grid=(M // BM,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0}, "09-custom.mlir": {"pytorchsim.runtime_attribute": 0}},
)
