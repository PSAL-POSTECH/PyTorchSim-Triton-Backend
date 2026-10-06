"""The fused embedding backward -- scatter-ADD of gradient rows into a
[vocab, hidden] table, where several tokens in the batch hit the SAME row.

    grad_table = 0
    for t in 0 .. T-1:
        if tok[t] != PAD:
            grad_table[tok[t], :] += grad_out[t, :]

WHY THE COLLISIONS ARE THE ENTIRE KERNEL. The embedding forward
(`embedding.py`) is a gather: one index per lane, every lane reads its own table
row, and two lanes reading the same row is harmless. Reverse the arrows and that
same duplication becomes a WRITE HAZARD -- "the" and "," appear dozens of times
in a batch, so dozens of gradient rows land on one table row and every one of
them has to be counted. This is `nn.Embedding.backward` and Liger's fused
`embedding` backward, and it runs once per step for every model that has an
embedding table, which is all of them.

NOTHING ELSE IN THIS SUITE ACCUMULATES INTO A LARGE TABLE. `moe_combine.py` is
the nearest thing and it is deliberately not this: its destination index is a
PERMUTATION within each pass, built that way on the host precisely so no two
source rows in one tile share a destination, and the duplicates that do exist are
separated by the sequential loop over ranks. Here the duplicates are inside a
single batch, they are data, and no host-side permutation can remove them --
which tokens repeat is what the text says.

  `ptr_scatter_roundtrip.py` writes each address once. `kv_cache_write.py` writes
  each slot once. This is the only kernel in the three suites where a destination
  is genuinely written more than once by one launch.

NO ATOMICS, AND THAT IS THE PRODUCTION CHOICE TOO. An atomic float add is
nondeterministic in ORDER, so the same batch gives a slightly different embedding
table on a rerun and a training run stops being reproducible; on top of that,
"the" appearing 200 times in a batch serialises 200 atomics on one address. The
real implementations avoid it two ways:

  SORT AND SEGMENT      sort the token ids, so equal ids become contiguous runs,
                        then reduce each run and write once. This is what
                        `torch.embedding_dense_backward` does on CUDA. It costs a
                        sort and needs a segmented reduction over a run whose
                        length is not known until run time.
  MASK AND REDUCE       tile the VOCABULARY instead of the tokens. A program owns
                        a block of table rows and sweeps the whole batch,
                        accumulating `where(row == tok[t], acc + g[t], acc)`.
                        Every destination is owned by exactly one lane for the
                        whole sweep, so the collision resolves as a repeated
                        read-modify-write of a REGISTER, never of memory.

This kernel is the second, because it is the one that fits a machine with no
lane-crossing primitive: the sort form needs a segmented scan along the token
axis, which is a fold across lanes, while the mask form's only cross-lane
operation is a broadcast of one scalar token id against the lane-resident row
ids. The vocabulary tile is what makes it work and also what bounds it -- with
V = 128k the vocabulary is blocked and each program sweeps the batch for its own
block, contributing zero for every token outside it. The code is identical; only
the row offset changes.

  The cost is honest and worth seeing in the IR: the batch is read once per
  vocabulary block rather than once, so this trades a sort for bandwidth. At
  V = 128k and a 2048-token batch, blocked 64 rows at a time, that is 2000
  sweeps -- which is why the real kernel makes the block as tall as it can and
  why the accumulator staying in registers matters so much.

WHAT THE BACKEND SEES. A [V, D] accumulator that is loop-carried across the whole
token loop and updated by a `tl.where` whose predicate is `rv[:, None] == tok_t`
-- one runtime SCALAR compared against a lane-resident index vector, so the
predicate varies along the lane axis and is constant in-lane, the mirror of
`causal_conv1d.py`'s (which varies in-lane and is constant across lanes). The
gradient row `g` is rank-1 and broadcasts along the lanes. No address anywhere in
the kernel depends on the data: the indirection has been turned into a
predicate, which is the whole trick.

THE TOKENS ARE BUILT TO COLLIDE. Of the 32 tokens, thirteen are the same id, six
more share another, two are padding, and the rest are spread out -- so one table
row accumulates thirteen gradient rows, one accumulates six, 51 of the 64 rows
stay exactly zero, and PAD's row must stay zero even though two tokens point at
it. A kernel that dropped all but the last hit, or that ignored padding_idx,
still produces a table of plausible-looking numbers; only the rows with a run in
them are wrong.

  ONE OF THE THIRTEEN IS NOT ADJACENT TO THE OTHER TWELVE -- it sits at position
  26, past the padding and past six unrelated ids. Real text is like that: a
  frequent token recurs throughout the batch rather than in a block. It matters
  because the run is then invisible to anything that only compares NEIGHBOURING
  tokens, which is the shortcut a sort-free segmented reduction is tempted into.
  The mask form does not care where in the sweep a hit falls, and that is its
  actual advantage over sort-and-segment, not just its simplicity.

PAD IS NOT A MASK ON THE OUTPUT, IT IS A MASK ON THE CONTRIBUTION. `padding_idx`
means that row never receives gradient, so the skip has to happen inside the
accumulation -- zeroing the row afterwards happens to give the same answer here
but stops doing so the moment the table is an INOUT that already holds something.

The only kernel here that accumulates into a large table with genuinely colliding destinations. No atomics and no sort: the vocabulary is on the lanes, the batch is swept, and the indirection becomes a predicate `row == tok`, so the [V, D] accumulator never leaves registers. Rows 37 and 5 carry the runs; row 0 must stay zero.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

V = 64                  # vocabulary rows held by this program -- one block
D = 16                  # hidden size, in-lane
T = 32                  # tokens in the batch
PAD = 0                 # padding_idx: this row never receives gradient


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def embedding_grad_scatter_add_kernel(tok_ptr, go_ptr, gt_ptr,
                                          V: tl.constexpr, D: tl.constexpr,
                                          T: tl.constexpr,
                                          PAD: tl.constexpr):
        rv = tl.arange(0, V)                 # table rows -- on the lanes
        rd = tl.arange(0, D)                 # hidden -- in-lane

        # The accumulator IS the table block. It is loop-carried for the whole
        # sweep, so a row hit twelve times is twelve register updates and not
        # one memory address touched twelve times.
        acc = tl.zeros((V, D), dtype=tl.float32)

        for t in tl.static_range(T):
            tok = tl.load(tok_ptr + t)                  # one runtime scalar
            g = tl.load(go_ptr + t * D + rd)            # [D], this token's grad

            # THE COLLISION, RESOLVED AS A PREDICATE. Exactly one lane matches,
            # every other lane keeps what it had, and padding contributes to no
            # lane at all. No address here depends on `tok`.
            hit = (rv[:, None] == tok) & (tok != PAD)
            acc = tl.where(hit, acc + g[None, :], acc)

        tl.store(gt_ptr + rv[:, None] * D + rd[None, :], acc)

    return embedding_grad_scatter_add_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Deliberate, documented collisions: a 12-deep run, a 6-deep run, two
    # padding tokens, eleven scattered singletons -- and one FAR-AWAY repeat of
    # the run's id, so the thirteen hits on row 37 are not contiguous.
    tok = torch.randint(0, V, (T,), generator=g, dtype=torch.int32)
    tok[0:12] = 37          # a run of twelve on one row
    tok[12:18] = 5          # another accumulates six
    tok[18:20] = PAD        # and this one must stay exactly zero
    tok[26] = 37            # the thirteenth hit, nowhere near the other twelve
    return {"tok": tok,
            "go": torch.randn(T * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    go = inputs["go"].reshape(T, D)
    tok = inputs["tok"].long()
    keep = tok != PAD
    gt = torch.zeros(V, D, dtype=torch.float32)
    gt.index_add_(0, tok[keep], go[keep])
    return {"gt": gt.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="embedding_grad_scatter_add_kernel", kernel=kernel,
    signature={"tok_ptr": "*i32", "go_ptr": "*fp32", "gt_ptr": "*fp32",
               "V": "constexpr", "D": "constexpr", "T": "constexpr",
               "PAD": "constexpr"},
    constexprs={"V": V, "D": D, "T": T, "PAD": PAD},
    args=[Arg("tok", IN, "int32", (T,)),
          Arg("go", IN, "float32", (T * D,)),
          Arg("gt", OUT, "float32", (V * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
)
