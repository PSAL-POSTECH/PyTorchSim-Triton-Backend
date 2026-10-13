"""An Inductor looped reduction whose tile lies inside the tensor on both axes: `r0_mask & xmask` is
always true, so the accumulator's select and the index vector feeding it must be gone by 08.
What breaks it: an induction variable read as [lb, ub - 1], or a linalg.index with no range."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K = 256, 96


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def always_true_select_kernel(a_ptr, b_ptr, c_ptr):
        xnumel = 256
        r0_numel = 96
        xoffset = tl.program_id(0) * 256
        xindex = xoffset + tl.arange(0, 256)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, 32)[None, :]
        _tmp4 = tl.full([256, 32], 0, tl.float32)
        for r0_offset in range(0, r0_numel, 32):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            tmp0 = tl.load(a_ptr + (r0_index), r0_mask, other=0.0)
            tmp1 = tl.load(b_ptr + (xindex + 256 * r0_index), r0_mask & xmask, other=0.0)
            tmp5 = _tmp4 + tmp0 * tmp1
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tl.store(c_ptr + (xindex), tmp4, xmask)

    return always_true_select_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(K, generator=g, dtype=torch.float32),
            "b": torch.randn(K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"c": inputs["a"] @ inputs["b"].reshape(K, N)}


SPEC = KernelSpec(
    name="always_true_select_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32"},
    constexprs={},
    args=[Arg("a", IN, "float32", (K,)), Arg("b", IN, "float32", (K * N,)),
          Arg("c", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"08-transformed.mlir": {"arith.cmpi": 0, "arith.select": 0,
                                       "torchsim.vlane_idx": 0}},
)
