"""Two programs of 256 rows over 512: `pid * 256 + arange(256) < 512` is true only because the
program id is inside the grid the kernel was compiled for, and its select must be gone by 08.
What breaks it: the launch grid not reaching the passes as the range of the program ids."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K = 512, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def program_id_inside_grid_kernel(a_ptr, b_ptr, c_ptr):
        xnumel = 512
        r0_numel = 64
        xoffset = tl.program_id(0) * 256
        xindex = xoffset + tl.arange(0, 256)[:, None]
        xmask = xindex < xnumel
        r0_index = tl.arange(0, 64)[None, :]
        r0_mask = r0_index < r0_numel
        tmp0 = tl.load(a_ptr + (r0_index), r0_mask, other=0.0)
        tmp1 = tl.load(b_ptr + (xindex + 512 * r0_index), r0_mask & xmask, other=0.0)
        tmp3 = tl.broadcast_to(tmp0 * tmp1, [256, 64])
        tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
        tmp4 = tl.sum(tmp5, 1)[:, None]
        tl.store(c_ptr + (xindex), tmp4, xmask)

    return program_id_inside_grid_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(K, generator=g, dtype=torch.float32),
            "b": torch.randn(K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"c": inputs["a"] @ inputs["b"].reshape(K, N)}


SPEC = KernelSpec(
    name="program_id_inside_grid_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32"},
    constexprs={},
    args=[Arg("a", IN, "float32", (K,)), Arg("b", IN, "float32", (K * N,)),
          Arg("c", OUT, "float32", (N,))],
    grid=(2,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"08-transformed.mlir": {"arith.cmpi": 0, "arith.select": 0,
                                       "torchsim.vlane_idx": 0}},
)
