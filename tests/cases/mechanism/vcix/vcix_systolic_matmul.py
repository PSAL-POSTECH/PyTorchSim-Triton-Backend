"""C = A @ B on the systolic array driven by hand with the vcix primitive alone (no tl.dot), M = K = N
= 256, one array pass. Its author knows the tpu model's meanings; the compiler does not: vcix("iv",
opcode=1) is a weight push (lane n keeps its last 256 weights, so B with lanes on N gives lane n
column n), vcix("iv", opcode=0) an input push (element i of every lane is one input vector: A with
lanes on K), vcix("v.i", opcode=2) the pop (lane n, depth m: C[m, n]). Pins: 09-custom has one loop each of
weight push, input push and pop, 16 trips at vl 16; no torchsim.dot, no compiler crossing.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 256, 256, 256


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import torchsim

    @triton.jit
    def vcix_systolic_matmul_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr,
                                    N: tl.constexpr):
        a = tl.load(a_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
        b = tl.load(b_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
        torchsim.vcix("iv", opcode=1, imm=0, args=[b], layouts=[torchsim.lane_layout(1)])
        torchsim.vcix("iv", opcode=0, imm=0, args=[a], layouts=[torchsim.lane_layout(1)])
        c = torchsim.vcix("v.i", opcode=2, imm=0, args=[], layouts=[], out=(M, N),
                          dtype=tl.float32, out_layout=torchsim.lane_layout(1))
        tl.store(c_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], c)

    return vcix_systolic_matmul_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g, dtype=torch.float32),
            "b": torch.randn(K, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"], inputs["b"])}


SPEC = KernelSpec(
    name="vcix_systolic_matmul_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "constexpr",
               "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-3,
    expect_ir={"01-ttir.mlir": {"torchsim.vcix": 3, "tt.dot": 0},
               "08-transformed.mlir": {"torchsim.dot": 0, "torchsim.crosslane_move": 0},
               "10-lowered.mlir": {"vcix.iv": 2, "vcix.v.i": 1, "torchsim.crosslane_move": 0}},
)
