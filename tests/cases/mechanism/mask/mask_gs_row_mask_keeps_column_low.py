"""A row-masked gather with a column lower bound: p01's mask rebuild must keep static_mask_lows."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

TD = 16; BM = 8; BN = 16; NT = 64 * TD; NI = 16; LO = 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_gs_row_mask_keeps_column_low_kernel(tbl_ptr, i32_ptr, s_ptr, o_ptr, TD: tl.constexpr,
                              BM: tl.constexpr, BN: tl.constexpr, LO: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        nkeep = tl.load(s_ptr)
        i32 = tl.load(i32_ptr + rm)
        m = (i32 < nkeep)[:, None] & (rn[None, :] >= LO)
        v = tl.load(tbl_ptr + i32[:, None] * TD + rn[None, :], mask=m, other=-7.0)
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], v)

    return mask_gs_row_mask_keeps_column_low_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.zeros(NI, dtype=torch.int32)
    idx[:BM] = torch.randperm(NT // TD, generator=g)[:BM].to(torch.int32)
    return {"tbl": torch.randn(NT, generator=g, dtype=torch.float32), "i32": idx,
            "s": torch.tensor([40], dtype=torch.int32)}


def reference(inputs):
    import torch
    tbl, i32 = inputs["tbl"], inputs["i32"]
    nkeep = int(inputs["s"][0])
    a = i32[:BM].long()[:, None]
    rn = torch.arange(BN)
    m = (i32[:BM] < nkeep)[:, None] & (rn[None, :] >= LO)
    v = torch.where(m, tbl[a * TD + rn[None, :]], torch.full((), -7.0))
    return {"o": v.reshape(-1)}


SPEC = KernelSpec(
    name="mask_gs_row_mask_keeps_column_low_kernel", kernel=kernel,
    signature={"tbl_ptr": "*fp32", "i32_ptr": "*i32", "s_ptr": "*i32", "o_ptr": "*fp32",
               "TD": "constexpr", "BM": "constexpr", "BN": "constexpr", "LO": "constexpr"},
    constexprs={"TD": TD, "BM": BM, "BN": BN, "LO": LO},
    args=[Arg("tbl", IN, "float32", (NT,)), Arg("i32", IN, "int32", (NI,)),
          Arg("s", IN, "int32", (1,)), Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
