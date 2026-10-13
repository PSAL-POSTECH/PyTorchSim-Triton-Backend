"""Neither extent divides: 400 of 2 x 256 rows and 100 of 4 x 32 columns are live, and everything
past them is +inf in memory the unmasked loads do read -- so both compares must stay and act.
What breaks it: folding either compare; the +inf then reaches a sum and the answer is inf or nan."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K = 400, 100
N_HELD, K_HELD = 512, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def needed_select_kernel(a_ptr, b_ptr, c_ptr):
        xnumel = 400
        r0_numel = 100
        xoffset = tl.program_id(0) * 256
        xindex = xoffset + tl.arange(0, 256)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, 32)[None, :]
        _tmp4 = tl.full([256, 32], 0, tl.float32)
        for r0_offset in range(0, r0_numel, 32):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            tmp0 = tl.load(a_ptr + (r0_index))
            tmp1 = tl.load(b_ptr + (xindex + 512 * r0_index))
            tmp5 = _tmp4 + tmp0 * tmp1
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tl.store(c_ptr + (xindex), tmp4)

    return needed_select_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.full((K_HELD,), float("inf"))
    b = torch.full((K_HELD, N_HELD), float("inf"))
    a[:K] = torch.randn(K, generator=g, dtype=torch.float32)
    b[:K, :N] = torch.randn(K, N, generator=g, dtype=torch.float32)
    return {"a": a, "b": b.reshape(-1)}


def reference(inputs):
    import torch
    c = torch.zeros(N_HELD)
    c[:N] = inputs["a"][:K] @ inputs["b"].reshape(K_HELD, N_HELD)[:K, :N]
    return {"c": c}


SPEC = KernelSpec(
    name="needed_select_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32"},
    constexprs={},
    args=[Arg("a", IN, "float32", (K_HELD,)), Arg("b", IN, "float32", (K_HELD * N_HELD,)),
          Arg("c", OUT, "float32", (N_HELD,))],
    grid=(2,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"08-transformed.mlir": {"arith.cmpi": 2, "arith.select": 1}},
)
