"""vcix_vexp_lane_layout with its layout named as a gluon DistributedLinearLayout: the lanes on axis 1
of an 8x128 tile, each lane's 8 rows in its registers, spelled as bases. The linear layout crosses
into the plugin, the anchor takes it as #ttg.linear and the encoding passes through. torch: exp(x)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 128


def kernel():
    import triton
    import triton.language as tl
    from triton.experimental.gluon.language._layouts import DistributedLinearLayout
    from triton.language.extra import torchsim

    LANES_ON_COLUMNS: tl.constexpr = DistributedLinearLayout(
        reg_bases=[[1 << i, 0] for i in range(M.bit_length() - 1)],
        lane_bases=[[0, 1 << i] for i in range(N.bit_length() - 1)],
        warp_bases=[], block_bases=[], shape=[M, N])

    @triton.jit
    def vcix_vexp_linear_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        y = torchsim.vcix("v.iv", opcode=3, imm=0, args=[tl.load(x_ptr + offs)],
                          layout=LANES_ON_COLUMNS)
        tl.store(o_ptr + offs, y)

    return vcix_vexp_linear_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.exp(inputs["x"])}


SPEC = KernelSpec(
    name="vcix_vexp_linear_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-5,
)
