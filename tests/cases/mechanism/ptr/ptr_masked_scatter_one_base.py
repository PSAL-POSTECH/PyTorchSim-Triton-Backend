"""A masked store through pointers that all lie in `o`, as one scatter over all of memory
(dram_arg -2) that parks its masked-off writes below 0. Breaks if the bound narrows to one
argument that is not `o` -- `a` -- : the DMA then drops the kept writes and `o` keeps its -7s."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, INOUT

N, M = 128, 100


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_masked_scatter_one_base_kernel(o_ptr, a_ptr, N: tl.constexpr, M: tl.constexpr):
        offs = tl.arange(0, N)
        q = tl.where((offs % 2) == 0, o_ptr + offs + 1, o_ptr + offs - 1)
        tl.store(q, tl.load(a_ptr + offs), mask=offs < M)

    return ptr_masked_scatter_one_base_kernel


def make_inputs(torch, seed=0):
    return {"o": torch.full((N,), -7.0, dtype=torch.float32),
            "a": torch.arange(N, dtype=torch.float32) + 1}


def reference(inputs):
    o = inputs["o"].clone()
    i = __import__("torch").arange(M)
    o[i + 1 - 2 * (i % 2)] = inputs["a"][:M]
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_masked_scatter_one_base_kernel", kernel=kernel,
    signature={"o_ptr": "*fp32", "a_ptr": "*fp32", "N": "constexpr", "M": "constexpr"},
    constexprs={"N": N, "M": M},
    args=[Arg("o", INOUT, "float32", (N,)), Arg("a", IN, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"04-transformed.mlir": {"dram_arg = -2": 1}},
)
