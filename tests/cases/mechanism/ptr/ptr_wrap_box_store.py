"""A wrapped store with a BOX mask, which was the one empty cell of the grid.

    rows = r % M                       wrap on axis 0
    tl.store(o + rows*C + c, v, mask=c < pid+4)     box mask on axis 1, runtime

The load side of this cell is occupied (ptr_modulo_wrap_shapes and others); the
store side was not, so nothing pinned that a wrapped make_tptr and a dynamic
mask_dims survive together on the way out.

The store half of `wrapped pointer + box mask`, which the grid had empty while the load half was occupied.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, M, GRID = 8, 8, 3, 2
OSIZE = M * C


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wrap_box_store(a_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr,
                       M: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, R)[:, None]
        c = tl.arange(0, C)[None, :]
        v = tl.load(a_ptr + (r * C + c))
        tl.store(o_ptr + ((r % M) * C + c), v, mask=c < pid + 4)

    return wrap_box_store


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(R * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a = inputs["a"].reshape(R, C)
    out = torch.zeros(OSIZE, dtype=torch.float32)
    r = torch.arange(R).reshape(R, 1)
    c = torch.arange(C).reshape(1, C)
    for pid in range(GRID):
        keep = (c < pid + 4).expand(R, C)
        dst = ((r % M) * C + c).expand(R, C)
        for i in range(R):
            for j in range(C):
                if keep[i, j]:
                    out[dst[i, j]] = a[i, j]
    return {"o": out.contiguous()}


SPEC = KernelSpec(
    name="wrap_box_store", kernel=kernel,
    signature={"a_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr",
               "C": "constexpr", "M": "constexpr"},
    constexprs={"R": R, "C": C, "M": M},
    args=[Arg("a", IN, "float32", (R * C,)),
          Arg("o", OUT, "float32", (OSIZE,))],
    grid=(GRID,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
