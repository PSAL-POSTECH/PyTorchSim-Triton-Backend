"""A cumsum of a BOOL, which torch promotes to int64 -- 64 bits for a count of 512.

This is the shape every `masked_scatter` arrives in, and the one Kimi K2.5's
vision tower stops on: `torch.cumsum` of a bool is int64 by promotion rule, not
because any value in it needs 64 bits, and Inductor carries that dtype down.
The scan then costs FOUR TIMES the vector registers it has to: at VLEN 512 a
`<64 x i64>` is 8 registers of 32 and a `<64 x i32>` is 4.

`open` BECAUSE THE PASS THAT FIXED IT IS GONE. truncate_accumulation read the
bound off the op -- axis extent x element bound -- and rebuilt the scan in i32,
truncating in and extending back out. It was deleted because it rewrote exactly
one accumulation in the whole suite, which is this one, and rule 5 calls that
shape out: a rule that only ever fires on one kernel is that kernel's answer
written down. `git show 98cba84~3` is the pass, and its reasoning is in the
commit above it.

GREEN SINCE 2026-09-05, AND NOT BY TAKING THE PRESSURE DOWN. llc still spills:
one `vs2r.v`, two registers, in a 96-byte frame. What changed is that stage 6
stopped reading a spill as fatal. spike remaps an sp-relative vector access to
`spad_base + vu_sram_byte - (kernel_sb - addr)`, and what can miss that window
is a FRAME deeper than one lane's scratchpad, not a store of any particular
width -- 96 bytes of 524288 here. The values were always exact; what was
missing was an ELF, and the refusal was what withheld it.

The i64 promotion is still four times the registers the bound needs, and
truncate_accumulation is still gone for firing on one kernel. That is a cost
question now rather than a correctness one.

a bool cumsum, int64 by torch promotion, over a tile wide enough that the unnecessary width spills a vector register group. POSITIVE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 8
N = 512
NB = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_cumsum_bool_promoted_kernel(m_ptr, o_ptr, M: tl.constexpr,
                                        N: tl.constexpr, NB: tl.constexpr):
        cols = tl.arange(0, N)
        rows = tl.arange(0, M)[:, None]
        # A SCAN LONGER THAN ONE TILE IS A LOOP WITH A CARRY, which is the shape
        # Inductor writes.  One scan on its own peaks at 17 of 32 registers and
        # llc fits it (measured), so a single-tile version would pin nothing.
        acc = tl.zeros([M, 1], tl.int64)
        for b in range(0, NB):
            off = b * N
            # THE `.to(tl.int64)` IS THE POINT and is not incidental: it is what
            # torch's promotion rule writes for `cumsum` of a bool, so dropping
            # it would test a kernel nobody generates.
            m = tl.load(m_ptr + off + cols).to(tl.int1).to(tl.int64)
            v = tl.broadcast_to(m[None, :], [M, N])
            c = tl.cumsum(v, axis=1)
            # THE TILE TOTAL, READ THE WAY INDUCTOR READS IT -- `select_one`,
            # a masked multiply over the whole tile rather than an index.  This
            # is the piece that tips llc over; see the module docstring.
            sel = tl.broadcast_to((cols == N - 1)[None, :].to(tl.int64), [M, N])
            t = tl.sum(c * sel, axis=1)[:, None]
            # The first tile has no carry yet, so the carry arrives as a select
            # rather than an add -- again what Inductor writes.
            tl.store(o_ptr + rows * (N * NB) + off + cols[None, :],
                     tl.where(off > 0, acc + c, c))
            acc = tl.where(off > 0, acc + t, t)

    return ops_cumsum_bool_promoted_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"m": (torch.rand(N * NB, generator=g) < 0.6)}


def reference(inputs):
    import torch
    # The carry makes the per-tile scans one scan over the whole thing.
    c = inputs["m"].to(torch.int64).cumsum(0)
    return {"o": c.unsqueeze(0).expand(M, N * NB).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="ops_cumsum_bool_promoted_kernel",
    kernel=kernel,
    signature={"m_ptr": "*i1", "o_ptr": "*i64", "M": "constexpr",
               "N": "constexpr", "NB": "constexpr"},
    constexprs={"M": M, "N": N, "NB": NB},
    args=[Arg("m", IN, "bool", (N * NB,)),
          Arg("o", OUT, "int64", (M * N * NB,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # The [512] row handed to every lane is twice the machine's lanes: a gather cannot
    # take it, the crossbar's permute can, in laps -- no DRAM trip.
    expect_ir={"04-transformed.mlir": {"dram_arg = -1": 0}},
)
