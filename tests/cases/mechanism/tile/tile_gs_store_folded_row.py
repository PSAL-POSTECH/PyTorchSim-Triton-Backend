"""A masked store through a gather TILE POINTER whose row index is not injective.

    idx_w = rm % W ; nh = rm // W ; idx_h = nh % H ; idx_n = nh // H
    keep  = idx_n < 1                             true exactly on rm < W*H
    o[idx_w + W*idx_h + ROWS*rn] = x[rm, rn]      stored under keep

rm covers BM = 64 rows, only ROWS = W*H = 49 of which are real. The row index
`idx_w + 7*idx_h` folds rm = 49..63 back onto 0..14, and `keep` is false exactly
there -- the same shape resnet18's SEVENTEENTH conv stores its 7x7x512 output
in, with idx_n/idx_y_h/idx_y_w/idx_y_c spelled the way Inductor spells them.

WHAT BREAKS IT. _empty_read_mask_carriers used to turn a masked gather-pointer store
into read-the-destination / select / write-it-back through the store's own
pointer:

    %cur = tts.load(bare_ptr)
    %new = select(broadcast(%keep), %v, %cur)
    tts.store(bare_ptr, %new)

and argued that the read touches no address the write was not going to touch
anyway. True, and not enough: the claim it needs is that each masked element is
handed back ITS OWN value, and that holds only while the gathered index is
INJECTIVE. Here it is not. Rows 49..63 read what rows 0..14 held BEFORE the
store and write it back over what rows 0..14 had just computed. The fix leaves
the mask on the descriptor and lets convert_accesses_to_transfer park the masked-off
positions in the buffer's padding, which cannot collide by construction.

MEASURED, N = 32, on the tree before the fix (294839c):

    480 of 1568 elements wrong = 15 rows x 32 channels, every one a row that
    had been right. max_abs_err 1.99984, max_rel_err 1: each is the
    destination's pre-store content (0.0) in place of a live value.

    o[:, 0..14] == 0.0                 want x[0..14, :].T, all >= 1.0
    o[:, 15..48] exact

After the fix (2338932): max_abs_err 0.

x is `rand + 1.0`, so every value that should land is at least 1.0 and the
clobber is unmistakably the zeroed output rather than an arithmetic slip.

WHY THE ROW ARITHMETIC IS DONE AT RANK 2, which is the whole reason an earlier
attempt at this kernel pinned nothing. Writing the index rank-1 first and
expanding after --

    row = (rm % W) + W * ((rm // W) % H)
    tl.store(o_ptr + row[:, None] + ROWS * rn[None, :], ...)

-- hits PtrAnalysis::visitOperandAdd's first guard, `lhsState.getRank() == 1 &&
lhsState.hasModulo()` ("do not support this pattern: a + arange(0, K) % M"),
the whole store falls out of the structured world, and what comes out of stage 3
is a `tts.scatter` with a 2-D index and a 2-D mask -- a path that was ALREADY
fixed and never reaches a masked store at all. Expanding each term
first, exactly as Inductor emits it, keeps the rank at 2 where `hasModulo() &&
!rhs.isStructured()` sends it to rebuildAsGatherScatter, and stage 3 gives the
tts.make_gather_scatter_tptr this file is about:

    tts.make_gather_scatter_tptr(%o, %idx, %c49, %keep)
      gather_scatter_dim = 0, sizes = [64, 32],
      static_strides = [1, dynamic]
    tts.store(%ptr, %v) static_mask_dims = [0, 32]

which is resnet's descriptor with 128 channels changed to 32.

WHY NOT 128, resnet's own BLOCK_N. It reproduces, but on the pre-fix tree it
stops one stage earlier: the read-back select widens to <128 x i32>, 4096 bits
against a VLEN 256 x LMUL 8 = 2048 bit budget, and stage 6 refuses to build.
A build refusal pins the fix far more weakly than wrong numbers do, so this
file uses the largest N that still gets all the way to Spike on both trees.

A masked store through a gather tile pointer whose row index folds the masked rows onto live ones: reading the destination back is not a no-op unless that index is injective.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM, N, W, H = 64, 32, 7, 7
ROWS = W * H            # 49 real rows in a 64-row tile


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gs_store_folded_row_kernel(x_ptr, o_ptr, BM: tl.constexpr,
                                   N: tl.constexpr, W: tl.constexpr,
                                   H: tl.constexpr, ROWS: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, N)
        # Inductor's own decomposition of a flat nhw index, verbatim.
        idx_w = rm % W
        nh = rm // W
        idx_h = nh % H
        idx_n = nh // H
        # The load is a plain structured one on purpose: the STORE is the only
        # thing here that has to become a gather tile pointer.
        v = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        # EXPAND FIRST, THEN COMBINE -- see the docstring. Rank-1 arithmetic
        # here turns the store into a tts.scatter and tests the wrong path.
        iw = idx_w[:, None]
        ih = idx_h[:, None]
        ic = rn[None, :]
        tl.store(o_ptr + tl.broadcast_to(iw + W * ih + ROWS * ic, [BM, N]), v,
                 (idx_n < 1)[:, None])

    return gs_store_folded_row_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Strictly >= 1.0 so that a value replaced by the zeroed destination is a
    # collision and cannot be mistaken for a value that happened to be zero.
    return {"x": torch.rand(BM * N, generator=g, dtype=torch.float32) + 1.0}


def reference(inputs):
    # row(m) = m % 7 + 7 * ((m // 7) % 7) is the identity for m < 49, so the
    # store is a transpose of the first ROWS rows into a column-major buffer.
    # Rows 49..63 are masked off and must leave nothing behind.
    return {"o": inputs["x"].reshape(BM, N)[:ROWS].t().reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="gs_store_folded_row_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BM": "constexpr",
               "N": "constexpr", "W": "constexpr", "H": "constexpr",
               "ROWS": "constexpr"},
    constexprs={"BM": BM, "N": N, "W": W, "H": H, "ROWS": ROWS},
    args=[Arg("x", IN, "float32", (BM * N,)),
          Arg("o", OUT, "float32", (ROWS * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
