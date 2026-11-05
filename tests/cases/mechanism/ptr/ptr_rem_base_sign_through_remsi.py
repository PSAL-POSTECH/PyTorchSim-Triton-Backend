"""`(k + H*D*m) // D % H` and `(k + H*D*m) % D` on the row base `(pid % n_prog) * BM` need the base >= 0;
a remsi takes its dividend's sign, so `pid % n_prog` is >= 0 for any n_prog. Breaks if lowerBoundOf stops
reading a remsi's sign from its dividend alone (the % becomes a sum wrap and the address a gather)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, H, D, BM = 16, 4, 32, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_rem_base_sign_through_remsi_kernel(x_ptr, o_ptr, n_prog, M: tl.constexpr, H: tl.constexpr,
                                                  D: tl.constexpr, BM: tl.constexpr):
        m = (tl.program_id(0) % n_prog) * BM + tl.arange(0, BM)
        k = tl.arange(0, H * D)
        xindex = k[None, :] + (H * D) * m[:, None]
        x10 = (xindex // D) % H
        x9 = xindex % D
        v = tl.load(x_ptr + (x9 + D * m[:, None] + (M * D) * x10))
        tl.store(o_ptr + xindex, v)

    return ptr_rem_base_sign_through_remsi_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(H * M * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].reshape(H, M, D).permute(1, 0, 2).reshape(-1).clone()}


SPEC = KernelSpec(
    name="ptr_rem_base_sign_through_remsi_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n_prog": "i32", "M": "constexpr", "H": "constexpr",
               "D": "constexpr", "BM": "constexpr"},
    constexprs={"M": M, "H": H, "D": D, "BM": BM},
    args=[Arg("x", IN, "float32", (H * M * D,)), Arg("o", OUT, "float32", (M * H * D,))],
    grid=(M // BM,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0}},
    extra={"scalar_args": [("n_prog", "int32_t")], "scalar_values": {"n_prog": M // BM}},
)
