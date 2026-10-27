"""The gather dim's MASK ENTRY, all four values, on one pointer shape.

A tts.make_gather_scatter_tptr carries one mask entry for the gathered dim, and
that single value is what decides how the access is addressed. Four loads and
four stores in ONE kernel on purpose: the arms are only legible as a
comparison, and everything else about the eight accesses is identical. Each
store gets its OWN destination buffer so the four can be told apart in
08-transformed.mlir by which dram_arg they write.

THERE IS NO LOOP ANY MORE, and that is the headline. StructuredToMemref turned
this into an `scf.for` over the gather offsets with one memref.copy per row,
and its four arms were four TRIP COUNTS (8 / 4 / min(8,nrow) / 8-plus-an-if).
That pass is deleted (rule 21). b08 hands the whole thing to one indirect
transfer per access and the DMA walks the rows: 08-transformed.mlir has ZERO
scf.for and ten pytorchsim.transfer ops.

COVERS (5)   convert_tts_to_pytorchsim.py, _emit_gather_scatter

  gs-no-mask            s1/store 1. No mask entry -> a plain indirect transfer,
      no masked_axes at all.
          {indirect, offset_stride = 32, dram_stride = [0, 1]}
  gs-mask-const         s2/store 2. A constant mask dim -> masked_axes = [0, 1]
      with both clamps folded: operands ..., %c0, %c4, %c0, %c12.
  gs-mask-dynamic       s3/store 3. A runtime mask dim -> the same shape with
      the high end an SSA value: ..., %c0, %21, %c0, %c12. Same arm as
      ptr_memref_masked_fill.py's two dims, reached through the gather.
  gs-mask-per-element   s4/store 4. MaskAnalysis cannot parse `gidx != 0`, so
      it arrives as a per-element carrier rather than a box. No masked_axes;
      instead `indirect_dim = 0` names the axis the index walks, and
      _empty_read_mask_carriers has already turned the mask into a select.
  gs-scatter-dram-bound store 4 only. A scatter whose masked-off addresses have
      nowhere safe to land carries pytorchsim.dram_bound, and the DMA drops anything
      outside [base, base+bytes). This is the parking lot, and it is the last
      transfer in the file:
          {..., indirect_dim = 0, pytorchsim.dram_bound}

SIBLING: tests/cases/mechanism/gather/ptr_gather_2d_row.py builds the same
pointer and stops at 01-structured.mlir; this file is about what b08 does with
it.

the gather dim's mask entry, all four values, read and written; one buffer per scatter so they can be told apart.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NT = 512            # table / destination footprint
NI = 64
TD = 32             # table row stride; > BN so the non-gather dim is real
BM = 8              # rows gathered
BN = 16             # columns per row
CBASE = 4           # column offset, to keep the non-gather term non-zero
KCONST = 4          # the constant row bound
KCOL = 12           # the column bound (also constant)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gs_memref_trips_kernel(tbl_ptr, idx_ptr, o1_ptr, o2_ptr, o3_ptr,
                                   o4_ptr, nrow, TD: tl.constexpr,
                                   BM: tl.constexpr, BN: tl.constexpr,
                                   CBASE: tl.constexpr, KCONST: tl.constexpr,
                                   KCOL: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        cols = CBASE + rn
        gidx = tl.load(idx_ptr + rm)                  # the rows to touch
        sgp = tbl_ptr + gidx[:, None] * TD + cols[None, :]

        m_const = (rm < KCONST)[:, None] & (rn < KCOL)[None, :]
        m_dyn = (rm < nrow)[:, None] & (rn < KCOL)[None, :]
        m_uns = (gidx != 0)[:, None]                  # MaskAnalysis cannot parse

        s1 = tl.load(sgp)                             # no mask   -> trip 8
        s2 = tl.load(sgp, mask=m_const, other=0.0)    # const 4   -> trip 4
        s3 = tl.load(sgp, mask=m_dyn, other=0.0)      # runtime   -> minsi
        s4 = tl.load(sgp, mask=m_uns, other=0.0)      # per-elem  -> trip 8 + if

        tl.store(o1_ptr + gidx[:, None] * TD + cols[None, :], s1)
        tl.store(o2_ptr + gidx[:, None] * TD + cols[None, :], s2, mask=m_const)
        tl.store(o3_ptr + gidx[:, None] * TD + cols[None, :], s3, mask=m_dyn)
        tl.store(o4_ptr + gidx[:, None] * TD + cols[None, :], s4, mask=m_uns)

    return ptr_gs_memref_trips_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # duplicate-free rows: the four scatters must be well-defined permutations
    idx = torch.zeros(NI, dtype=torch.int32)
    idx[:BM] = torch.randperm(NT // TD, generator=g)[:BM].to(torch.int32)
    return {"tbl": torch.randn(NT, generator=g, dtype=torch.float32),
            "idx": idx}


NROW = 6


def reference(inputs):
    import torch
    tbl, idx = inputs["tbl"], inputs["idx"].long()[:BM]
    rows = torch.stack([tbl[i * TD + CBASE:i * TD + CBASE + BN] for i in idx])
    rm = torch.arange(BM)[:, None]
    rn = torch.arange(BN)[None, :]
    masks = [torch.ones(BM, BN, dtype=torch.bool),
             (rm < KCONST) & (rn < KCOL),
             (rm < NROW) & (rn < KCOL),
             (idx != 0)[:, None].expand(BM, BN)]
    outs = []
    for m in masks:
        o = torch.zeros(NT, dtype=torch.float32)
        vals = torch.where(m, rows, torch.zeros(()))
        for k, i in enumerate(idx):
            sel = m[k]
            base = i * TD + CBASE
            o[base:base + BN] = torch.where(sel, vals[k], o[base:base + BN])
        outs.append(o)
    return {"o1": outs[0], "o2": outs[1], "o3": outs[2], "o4": outs[3]}


SPEC = KernelSpec(
    name="ptr_gs_memref_trips_kernel", kernel=kernel,
    signature={"tbl_ptr": "*fp32", "idx_ptr": "*i32", "o1_ptr": "*fp32",
               "o2_ptr": "*fp32", "o3_ptr": "*fp32", "o4_ptr": "*fp32",
               "nrow": "i32", "TD": "constexpr", "BM": "constexpr",
               "BN": "constexpr", "CBASE": "constexpr", "KCONST": "constexpr",
               "KCOL": "constexpr"},
    constexprs={"TD": TD, "BM": BM, "BN": BN, "CBASE": CBASE,
                "KCONST": KCONST, "KCOL": KCOL},
    args=[Arg("tbl", IN, "float32", (NT,)), Arg("idx", IN, "int32", (NI,)),
          Arg("o1", OUT, "float32", (NT,)), Arg("o2", OUT, "float32", (NT,)),
          Arg("o3", OUT, "float32", (NT,)), Arg("o4", OUT, "float32", (NT,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("nrow", "int32_t")],
           "scalar_values": {"nrow": NROW}},
)
