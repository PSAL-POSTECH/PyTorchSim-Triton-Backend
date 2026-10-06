"""An accumulator built by a PREDICATED add, then read twice: bufferize leaves
a copy and the storage walk has to place it. Breaks it: dropping the second loop
or the late store of acc, either of which removes the copy.

POSITIVE. THE BRANCH IS FOLDED BY HAND, and with it this kernel's old subject: it used to write `if flag > 0.0:` and pin the scf.if spelling of bufferize's storage walk (before the fix, on ec59cf0: adapt FAIL, '_Fatal: memref.copy between storage this pass cannot place (scf.if and memref.alloc)'). This route REFUSES control flow rather than lowering it -- the four other kernels that reach stage 4 with an scf.if are all refused by design -- and a branch that reaches the ELF is a branch the NPU model has to take a side of, so the flag rides as a `tl.where` and the load is unconditional. What is left pinned is the copy an accumulator read twice leaves behind. loop_16_mask_while still pins the scf.while spelling.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 64
POLISH = 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop20_if_inplace_accum_kernel(x_ptr, f_ptr, o1_ptr, o2_ptr,
                                       POLISH: tl.constexpr,
                                       BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        flag = tl.load(f_ptr)
        acc = acc + tl.where(flag > 0.0, tl.load(x_ptr + r), 0.0)
        b = acc
        for j in range(0, POLISH):
            b = b * 0.5 + 1.0
        tl.store(o1_ptr + r, b)
        tl.store(o2_ptr + r, acc * 2.0)

    return loop20_if_inplace_accum_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BLOCK, generator=g, dtype=torch.float32),
            "f": torch.ones(1, dtype=torch.float32)}


def reference(inputs):
    acc = inputs["x"] if float(inputs["f"][0]) > 0.0 else inputs["x"] * 0.0
    b = acc
    for _ in range(POLISH):
        b = b * 0.5 + 1.0
    return {"o1": b, "o2": acc * 2.0}


SPEC = KernelSpec(
    name="loop20_if_inplace_accum_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*fp32", "o1_ptr": "*fp32",
               "o2_ptr": "*fp32", "POLISH": "constexpr", "BLOCK": "constexpr"},
    constexprs={"POLISH": POLISH, "BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (BLOCK,)),
        Arg("f", IN, "float32", (1,)),
        Arg("o1", OUT, "float32", (BLOCK,)),
        Arg("o2", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
