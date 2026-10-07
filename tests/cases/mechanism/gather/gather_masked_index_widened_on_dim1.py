"""A masked index load widened for a gather on dim 1: its mask must follow it onto the gathered axis."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

TD = 16; BM = 8; BN = 16; NT = 64 * TD; NC = 12


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_masked_index_widened_on_dim1_kernel(tbl_ptr, c_ptr, o_ptr, TD: tl.constexpr,
                                    BM: tl.constexpr, BN: tl.constexpr, NC: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        cidx = tl.load(c_ptr + rn, mask=rn < NC, other=3)
        v = tl.load(tbl_ptr + rm[:, None] * TD + cidx[None, :])
        rq = tl.arange(0, 4)
        w = tl.load(tbl_ptr + rq[:, None] * TD + cidx[None, :])
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], v)
        tl.store(o_ptr + BM * BN + rq[:, None] * BN + rn[None, :], w)

    return gather_masked_index_widened_on_dim1_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"tbl": torch.randn(NT, generator=g, dtype=torch.float32),
            "c": torch.randint(0, TD, (BN,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    tbl, c = inputs["tbl"], inputs["c"].clone()
    c[NC:] = 3
    rm = torch.arange(BM)
    v = tbl[rm[:, None] * TD + c.long()[None, :]]
    w = v[:4]
    return {"o": torch.cat([v.reshape(-1), w.reshape(-1)])}


SPEC = KernelSpec(
    name="gather_masked_index_widened_on_dim1_kernel", kernel=kernel,
    signature={"tbl_ptr": "*fp32", "c_ptr": "*i32", "o_ptr": "*fp32",
               "TD": "constexpr", "BM": "constexpr", "BN": "constexpr", "NC": "constexpr"},
    constexprs={"TD": TD, "BM": BM, "BN": BN, "NC": NC},
    args=[Arg("tbl", IN, "float32", (NT,)), Arg("c", IN, "int32", (BN,)),
          Arg("o", OUT, "float32", (BM * BN + 4 * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
