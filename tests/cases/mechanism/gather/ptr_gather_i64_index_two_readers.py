"""Two gathers off ONE index tile that is already i64, with tiles written between.

SIBLING: tests/cases/mechanism/gather/ptr_gather_1d_arith.py is the i32 half of this.
Its four gathers share two index tiles and that is what first showed a transfer
reading another transfer's offsets; the repair was to give every index tile a
`memref.alloc` of convert_accesses_to_transfer's own. This file is the case that repair did not cover.

WHY AN i64 TILE WAS THE HOLE. `Ctx.index_buffer` exists to widen an index tile
to the i64 the DMA's index field needs, and the private alloc was a BY-PRODUCT
of that widening: the widened tile is a new tensor, so it gets
materialize_in_destination into an alloc, and the alloc's unregistered user is
what keeps one-shot-bufferize off the buffer. A tile that arrives ALREADY i64
has nothing to widen, so it returned a bare `bufferization.to_buffer` and got no
alloc. The safeguard was riding on the type conversion instead of standing on
its own. convert_accesses_to_transfer's own docstring recorded the hole ("it is still exposed to that
same reuse ... no kernel in the suite does today"); this is that kernel.

WHY THE BUFFER IS THEN REUSED. `torchsim.transfer` WAS UNREGISTERED, so bufferize
could not see that operand [8] is read by it. It is registered now and its
BufferizableOpInterface says so; the copies below are `alloc_tensor copy(...)`. The index tensor looks dead at its
`to_buffer` and the buffer is handed to the next tile that needs scratch.

WHAT THIS KERNEL LINES UP (all three are needed -- drop any one and it passes)
  1. the index tile is i64 from the start -- `i_ptr` is *i64, so `c + 64*i`
     never goes near arith.extsi
  2. TWO gathers read that one tile -- `t_ptr + idx` and `u_ptr + idx` are the
     same expression, so cse merges them into one tensor with two readers
  3. index tiles for OTHER gathers are computed between the two readers --
     `j*3` and `j*5`, whose range (up to NX) is far past the tables' NT, so the
     corruption is loud instead of landing back inside t/u

WHAT PROVES IT (04-passes/04-torchsim-convert-accesses-to-transfer.mlir)
  Before: one `bufferization.to_buffer ... tensor<128xi64> to memref<128xi64>`
  and three allocs; after place_in_spad both the t and the u transfer take the SAME
  buffer as operand [8], and the j*3 / j*5 tiles are stored into it in between.
  So the u gather indexes a 2048-element table with numbers meant for an
  8192-element one.
  After: four indirect transfers on FOUR different buffers, which is the whole
  contract -- the counts are pinned below and the identity is what they stand
  for. Three of the four are copies; the fourth index arrives in a tile of its
  own already, and the first transfer to want it takes it as it is, because the
  slot is read and never written.

MEASURED: FAIL on develop f6232e1, 128/128 elements outside tolerance; PASS
with the alloc made unconditional. The e2e that found it is DeepSeek-V3's RoPE
kernel, where position_ids are int64 (condition 1), cos and sin are indexed by
the same `x0 + 64*tmp5` (condition 2), and the q/k row indices are computed
between them (condition 3). There it read a 2048-element sin table with indices
up to 18431 and ran off the end of pk's heap: `User load segfault @
0xd103963c`. A crash was the lucky outcome -- inside the heap it is silently
wrong numbers, which is what this kernel reproduces.

torch equivalent: t[idx] * x[j*3] + x[j*5] * u[idx], idx = arange%64 + 64*i.

one i64 index tile read by two gathers with other index tiles written between them.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NT = 2048           # the two small tables the SHARED i64 index reads
NX = 8192           # the big buffer the two INTERVENING gathers read
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_i64_index_two_readers_kernel(
            t_ptr, u_ptr, x_ptr, i_ptr, j_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        c = r % 64
        i = tl.load(i_ptr + r)              # i64 in, so the index tile is i64
        idx = c + 64 * i                    # < NT. ONE tile, TWO readers.

        a = tl.load(t_ptr + idx)            # reader 1

        # Two gathers whose index tiles are i32 and range far past NT. They are
        # what gets computed between the two readers.
        j = tl.load(j_ptr + r)
        b = tl.load(x_ptr + j * 3)
        d = tl.load(x_ptr + j * 5)

        e = tl.load(u_ptr + idx)            # reader 2, same idx -> cse merges

        tl.store(o_ptr + r, a * b + d * e)

    return ptr_gather_i64_index_two_readers_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"t": torch.randn(NT, generator=g, dtype=torch.float32),
            "u": torch.randn(NT, generator=g, dtype=torch.float32),
            "x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, NT // 64, (BLOCK,), generator=g,
                               dtype=torch.int64),
            "j": torch.randint(0, NX // 5, (BLOCK,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    t, u, x = inputs["t"], inputs["u"], inputs["x"]
    import torch
    i, j = inputs["i"].long(), inputs["j"].long()
    idx = torch.arange(BLOCK) % 64 + 64 * i
    return {"o": t[idx] * x[j * 3] + x[j * 5] * u[idx]}


SPEC = KernelSpec(
    name="ptr_gather_i64_index_two_readers_kernel", kernel=kernel,
    signature={"t_ptr": "*fp32", "u_ptr": "*fp32", "x_ptr": "*fp32",
               "i_ptr": "*i64", "j_ptr": "*i32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("t", IN, "float32", (NT,)), Arg("u", IN, "float32", (NT,)),
          Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int64", (BLOCK,)),
          Arg("j", IN, "int32", (BLOCK,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # ONE FEWER COPY THAN TRANSFERS. Both numbers, or neither says anything: 4
    # alone allows a shared buffer and 3 alone allows three transfers.
    expect_ir={"04-passes/03-torchsim-convert-accesses-to-transfer.mlir": {
        "offset_stride": 4,
        "bufferization.alloc_tensor() copy(": 3}},
)
