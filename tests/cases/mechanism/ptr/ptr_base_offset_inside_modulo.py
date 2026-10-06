"""A splat scalar written INSIDE a modulo whose expression is already a gather.

    idx = (pid * STEP + r // 2) % M

`r // 2` is not modelled, so the state is unstructured before the `%` is seen;
visitOperandRem then takes its !isStructured() early return, which parks the
WHOLE `(pid*STEP + r//2) % M` as the gather index -- and `pid*STEP` is already
inside that value while the state still holds it in PtrState::baseOffset.

foldBaseIntoWrap, which is what moves an inside-the-wrap term onto its axis, is
below that early return and never runs here.

PID 0 HIDES IT. With pid = 0 the base term is zero and the address is right, so
the grid has to be >1 for the second program to read at base+STEP too many.

A base offset written inside a modulo whose expression is already unstructured: visitOperandRem's !isStructured() early return parks the whole value as the gather index while PtrState::baseOffset still holds the same term.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, M, STEP, GRID = 8, 16, 8, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def base_offset_inside_modulo(a_ptr, o_ptr, N: tl.constexpr,
                                  M: tl.constexpr, STEP: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, N)
        idx = (pid * STEP + r // 2) % M
        v = tl.load(a_ptr + idx)
        tl.store(o_ptr + pid * N + r, v)

    return base_offset_inside_modulo


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a = inputs["a"]
    out = torch.empty(GRID * N, dtype=torch.float32)
    for pid in range(GRID):
        r = torch.arange(N)
        idx = (pid * STEP + r // 2) % M
        out[pid * N:(pid + 1) * N] = a[idx]
    return {"o": out.contiguous()}


SPEC = KernelSpec(
    name="base_offset_inside_modulo", kernel=kernel,
    signature={"a_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "M": "constexpr", "STEP": "constexpr"},
    constexprs={"N": N, "M": M, "STEP": STEP},
    args=[Arg("a", IN, "float32", (M,)),
          Arg("o", OUT, "float32", (GRID * N,))],
    grid=(GRID,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
