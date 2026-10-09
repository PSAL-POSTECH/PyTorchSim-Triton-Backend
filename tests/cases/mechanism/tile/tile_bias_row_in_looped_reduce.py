"""a REPLICATING ROW read inside a loop-carried reduction.

PASSES. It did not when it was written, and what it pins is the fix:
lib_axis_graph gained the `scf.for` edge at develop-e2e-sd15 27fd9e6.
Measured on this tree, mn 1.49012e-08 and vr 2.38419e-07.

WHAT IT USED TO DO. The row is indexed by the reduction index alone -- `tl.load(bias_ptr + r)` --
so it broadcasts across the lanes with a DRAM row stride of 0. That is what
Inductor emits for an addmm epilogue's bias, and it is correct on its own. Read
it INSIDE a loop whose accumulator crosses the back edge more than once and the
reduction comes out wrong.

TWO INGREDIENTS, AND ONLY TWO. Measured on this file, changing one thing at a
time; everything else here is the same kernel:

    the replicating row, loop runs TWICE   (R0_BLOCK 512)   FAIL  0.803176
    the replicating row, loop runs ONCE    (R0_BLOCK 1024)  PASS  2.79397e-08
    no replicating row, loop runs twice    (R0_BLOCK 512)   PASS  1.55903e-06
    no replicating row, loop runs 3x       (R0_BLOCK 256)   PASS  3.50177e-07

NOT the ingredients, each measured and each ruled out:

  * the BLOCK DEPTH. It fails at R0_BLOCK 128, 256 and 512 alike, so this is
    not the lane-count limit that `gather/gather_masked_deep_in_loop.py` pins.
  * WELFORD. The same kernel with Welford's three-way carry and with two plain
    sums fail the same way (1.00416 and 0.803176). Welford is what the model
    happens to use, not what breaks.
  * the PARTIAL TAIL. 768 in blocks of 256 divides exactly and still fails once
    the row is there; 768 in blocks of 512 leaves a 256-wide masked tail and
    passes once the row is gone.

REDUCED FROM GPT-2 (tests/models/GPT2/test_gpt2.py --preset small), kernel 12
`triton_npu_fused_add_addmm_native_layer_norm_view_12`, where the row is the
addmm bias `tl.load(in_ptr1 + r0_1, r0_mask)` summed into the LayerNorm's
input. Per-kernel verify stops there: buf26, `aten.add.Tensor`, 24570/24576
elements over tolerance, max abs diff 2.60108, 11 buffers verified OK first.

WHAT THE FAILURE LOOKS LIKE, and it is worth knowing because it is not obvious
from the output: given the kernel's own recorded raws, its result is an EXACT
affine transform of the true input -- linear residual 1.155e-07 -- so the
elementwise half is perfect and only the reduced statistics are wrong. The mean
is off and the variance is 4.76x to 8.29x too large. A wrong reduction that
still normalises cleanly reads as "the numbers are a bit off" and is in fact a
different sum.

WHY IT APPEARED WHEN IT DID. GPT-2 --preset small passed end to end at
2.0862e-06 before PyTorchSim declined persistence for reductions this backend
would block differently (e2e-bert 08ba6cd). Inductor used to bake R0_BLOCK =
1024 into this kernel, which made the loop body run ONCE and hid this. That
change is right -- it is what stops kernel 9 blowing the scratchpad, and the
retry it restores is the only correction the block size has -- so this defect
was always here and is only now reachable.

WHERE IT WENT WRONG, and what the fix turned out to be

  08-passes/10-psto-place-lane-banks.mlir, on the broken tree -- the two ends
  of the loop's
  back edge sat in different axis classes:

      %alloc_4  memref<32x512xf32>  torchsim.lane_axis = 1     <- alone on axis 1
      %alloc_5  memref<32x512xf32>  torchsim.lane_axis = 0
      %alloc_6  memref<32x512xf32>  torchsim.lane_axis = 0

  `%alloc_5`/`%alloc_6` are the accumulators' INITS, outside the loop;
  `%alloc_4` and `%alloc_14` are what the loop BODY computes. An iter_arg
  went in on axis 0 and came out on axis 1, and nothing refused it.

  THE CAUSE was one level further down and is not this pass: `_maps_of` in
  lib_axis_graph had no arm for `scf.for` or `scf.yield`, so both fell
  through to its `return None` -- which does not mean "no rule here" but
  RELATES NO AXES. For a back edge that is a false statement rather than a
  missing one, so the four spellings of one buffer (init, block argument,
  result, yielded value) stayed in different classes. Fixed by giving the
  graph that edge (develop-e2e-sd15 27fd9e6), merged here. The module
  docstring had named the edge since the file was written; `scf` appeared in
  the file exactly once, in that sentence.

  It showed downstream as the banking: the broken tree emitted
  `buf7/8/9_spad_512lane` for the 32x512 tiles and puts the row in
  `buf6_spad_512lane : memref<1x512xf32>`, so each lane sees 4 of the row's
  512 values. The passing tree MVINs the row with `dram_stride = [0, 1]` --
  the replicating transfer, the same row into every lane's bank -- into a
  full tile banked like everything else. The replicating MVIN is what this
  kernel does not get.

WHAT BREAKS THIS KERNEL AGAIN
  Anything that lets the two sides of an scf.for back edge take different lane
  axes. The failure is silent -- it compiles, nothing warns, and the result is
  an exact affine transform of the right answer with the wrong statistics.

  Related: `tile/tile_bias_row_deeper_than_one_per_lane.py` pins the same
  REPLICATING row in an addmm epilogue, unlooped, failing on depth past the
  lane count -- fixed at 2007619. This is the loop's spelling of it and that
  fix does not reach it.

A rank-1 row broadcast over the lanes, read inside a reduction loop that runs more than once. Two ingredients only: remove the row, or make the loop run once, and it passes. Not the block depth, not Welford, not the masked tail -- all three measured and ruled out. Reduced from GPT-2 --preset small kernel 12.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS = 32           # xnumel, and XBLOCK is clamped to it
XBLOCK = 32
N = 768             # r0_numel -- GPT-2 small's n_embd
R0_BLOCK = 512      # so the loop body runs TWICE; at 1024 it ran once and hid this


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_bias_row_in_looped_reduce_kernel(x_ptr, bias_ptr, mn_ptr, vr_ptr,
                                              ROWS: tl.constexpr,
                                              N: tl.constexpr,
                                              R0_BLOCK: tl.constexpr,
                                              XBLOCK: tl.constexpr):
        x = tl.arange(0, XBLOCK)[:, None]
        xmask = x < ROWS
        r0_base = tl.arange(0, R0_BLOCK)[None, :]

        acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        acc2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        for r0_offset in range(0, N, R0_BLOCK):
            r0 = r0_offset + r0_base
            r0_mask = r0 < N
            v0 = tl.load(x_ptr + (r0 + N * x), r0_mask & xmask, other=0.0)
            # THE REPLICATING ROW: indexed by the reduction index alone, so it
            # broadcasts over the lanes with a DRAM row stride of 0. Remove it
            # -- reduce v0 by itself -- and this kernel passes.
            v1 = tl.load(bias_ptr + r0, r0_mask, other=0.0)
            v = tl.broadcast_to(v0 + v1, [XBLOCK, R0_BLOCK])
            acc += v
            acc2 += v * v

        s1 = tl.sum(acc, 1)[:, None]
        s2 = tl.sum(acc2, 1)[:, None]
        mean = s1 / N
        tl.store(mn_ptr + x, mean, xmask)
        tl.store(vr_ptr + x, s2 / N - mean * mean, xmask)

    return tile_bias_row_in_looped_reduce_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS, N, generator=g, dtype=torch.float32),
            "bias": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"] + inputs["bias"]
    return {"mn": x.mean(1), "vr": x.var(1, unbiased=False)}


SPEC = KernelSpec(
    name="tile_bias_row_in_looped_reduce_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "bias_ptr": "*fp32", "mn_ptr": "*fp32",
               "vr_ptr": "*fp32", "ROWS": "constexpr", "N": "constexpr",
               "R0_BLOCK": "constexpr", "XBLOCK": "constexpr"},
    constexprs={"ROWS": ROWS, "N": N, "R0_BLOCK": R0_BLOCK, "XBLOCK": XBLOCK},
    args=[Arg("x", IN, "float32", (ROWS, N)),
          Arg("bias", IN, "float32", (N,)),
          Arg("mn", OUT, "float32", (ROWS,)),
          Arg("vr", OUT, "float32", (ROWS,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
