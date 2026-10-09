"""bitsandbytes NF4 with double quantisation -- dequant is a TABLE LOOKUP, twice.

    absmax[b, n] = code2[aq[b, n]] * s2[n] + off[n]     # pass 2: the SCALE's own
    w[k, n]      = code1[idx[k, n]] * absmax[k//G, n]   # pass 1: the WEIGHT's

`code1` is 16 fixed fp32 constants, `code2` is the second-level table. Neither
level is `value * scale`; both are `table[value] * scale`.

WHY A CODEBOOK AND NOT AN AFFINE GRID
  Affine quantisation puts its 16 levels at equal spacing, which is optimal only
  if the weights are uniformly distributed. They are not -- a trained weight
  matrix is very close to zero-mean Gaussian. NF4 ("4-bit NormalFloat") instead
  places the 16 levels at the quantiles of a standard normal, so each level
  claims an equal PROBABILITY MASS rather than an equal interval: dense near
  zero where the weights are, sparse in the tails. The levels are irrational
  numbers with no arithmetic relationship to each other, so there is no formula
  -- the table IS the definition. This is what QLoRA quantises to and what every
  `load_in_4bit=True` checkpoint holds.

  DOUBLE QUANTISATION is the second half. One absmax per 64 weights, stored fp32,
  costs 0.5 bits per weight -- an eighth of the payload. So bitsandbytes
  quantises the absmax array too, with its own blockwise code and its own fp32
  scale plus a mean offset, and gets that down to about 0.127 bits per weight.
  The result is a scale you cannot read: you have to dequantise it first.

WHAT IT ASKS OF THE BACKEND, AND WHY NOTHING ELSE HERE ASKS IT
  1. A GATHER FROM A TINY TABLE BY A FULL TILE OF INDICES. `code_ptr + idx` where
     `idx` is [K, N] is a 512-way gather into a 16-entry array. Every existing
     quant kernel in this suite (quant_int4_matmul, int8_dequant_matmul,
     mxfp4_shift_scale_matmul) reconstructs its value ARITHMETICALLY from the
     stored bits -- shift, mask, subtract, multiply. None of them indexes a
     table, and a table is not an arithmetic function of the index. On a lane
     machine this is either a real vector gather or a 16-way select chain, and
     which one it becomes is the interesting answer. Note the contrast with
     gptq_actorder_gather.py, whose gather is a PERMUTATION -- source and
     destination the same size, each element read once. This one is many-to-one
     off a table that fits in a register file.
  2. THE TWO PASSES ARE CHAINED, not parallel. Pass 2's output is pass 1's scale,
     so a second gather's RESULT feeds the address-independent half of a first
     gather's arithmetic. The dependence is data, not layout, and it cannot be
     reassociated away.
  3. THE COARSE-AXIS STRETCH LANDS ON AN INDEX. quant_int4_matmul.py stretches a
     float scale over its group; here `rk // G` selects the absmax block and the
     second gather is then issued at the full [K, N] shape, so what gets
     broadcast is an integer that is about to become an address. Address
     arithmetic derived from an integer division of the iteration index, feeding
     a load whose result is another load's index.

FP32-ONLY, AND WHY THAT COSTS NOTHING HERE. Real NF4 stores the codebook in the
compute dtype and the payload as packed uint8. This backend has no sub-byte
type, so the payload is packed nibbles in int32 -- the same unpack
quant_int4_matmul.py does, minus the `- 8`, because an NF4 index is UNSIGNED: the
sign lives in the table, not in the bit pattern. That asymmetry is worth keeping;
it is a real difference between a codebook and an affine grid.

The 16 `code1` values below are bitsandbytes' actual NF4 table. `code2` stands in
for its 256-entry dynamic-exponent absmax code with a 16-entry geometric table --
the same shape (positive, unequally spaced, no closed form) at a readable width.

IT SEGFAULTED THE COMPILER, AND THE TWO BUGS IT FOUND. It reached the suite as
`KILLED by signal 11 (SIGSEGV) -- no diagnostic was printed`; with faulthandler
armed in run.py it named `widen_broadcast_loads.py:51`.

  GONE WITH THE PASS -- ONE ROW, TWO SHAPES. `code1` is a 16-element codebook
  row and this kernel broadcasts it to a 4x16 tile AND to a 32x16 tile, both
  from the same `tensor.expand_shape`. widen_broadcast_loads rewrote the LOAD's
  TYPE to make the DMA replicate it, which is one shape for every reader -- so
  it widened for the first, erased the expand_shape the second still had a use
  on, and the next walk over a dangling use list took the process down. It
  learned to enumerate the readers and decline unless they agreed, and then the
  pass was deleted outright: retyping a value shared by two readers is a hazard
  that only exists because the rewrite was on the load rather than on the
  operand. fit_layout_to_lanes._clone_loaded_row_wide MINTS a second buffer and
  re-points ONE operand, so two readers wanting two shapes is not a case it can
  reach -- "the hazards that come with retyping a sub-DAG are simply not
  reachable here" is its own docstring, and this kernel is why that sentence is
  in it.

  STILL RED -- fit_layout_to_lanes materialises the row at the WRONG shape. With
  the widening declined, the 1x16 row survives to stage 4's layout passes, and
  fit_layout_to_lanes relayouts it into a 4x16 buffer for BOTH consumers:

      error: 'linalg.generic' op inferred input/output operand #2 has shape's
             dimension #0 to be 32, but found 4
      ... : (memref<32x16xf32,1>, memref<32x16xf32,1>, memref<4x16xf32,1>,
             memref<4x16xf32,1>, memref<32x16xf32,1>) -> ()

  Rule 4's half-rewrite, one pass lower: the relayout picked a shape and the op
  reading it was not retyped with it. 08-passes/07-torchsim-bufferize.mlir has the
  generic reading a 1x16; 08-passes/10-psto-place-lane-banks.mlir has it reading a
  4x16. NOT FIXED HERE -- p08_fit_layout_to_lanes is owned by another agent this
  session, so this is reported rather than edited.

NF4: dequant by 16-entry codebook lookup, and the block scale is itself a lookup -- two chained many-to-one gathers where every other quant kernel here reconstructs its value arithmetically.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, N, G = 32, 16, 8          # in features, out features, weights per absmax
NB = K // G                  # absmax blocks per column

# bitsandbytes' NF4 codebook: the 16 quantiles of a standard normal, rescaled so
# the extremes are exactly -1 and +1. Index 7 is exactly zero on purpose.
NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
       -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
       0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
       0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
       0.7229568362236023, 1.0)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def nf4_codebook_dequant_kernel(wq_ptr, aq_ptr, c1_ptr, c2_ptr, s2_ptr,
                                    off_ptr, am_ptr, w_ptr, K: tl.constexpr,
                                    N: tl.constexpr, G: tl.constexpr,
                                    NB: tl.constexpr):
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        rb = tl.arange(0, NB)

        s2 = tl.load(s2_ptr + rn)
        off = tl.load(off_ptr + rn)

        # PASS 2 -- the absmax array is itself quantised. A [NB, N] tile of
        # 4-bit indices, gathered through the second-level table, then rescaled
        # affinely. Stored so the intermediate is checkable on its own.
        aq_b = tl.load(aq_ptr + rb[:, None] * N + rn[None, :])        # [NB, N]
        am_b = tl.load(c2_ptr + aq_b) * s2[None, :] + off[None, :]
        tl.store(am_ptr + rb[:, None] * N + rn[None, :], am_b)

        # The same lookup again at the granularity pass 1 needs. THE COARSE-AXIS
        # STRETCH IS ON THE INDEX, not on the dequantised value: `rk // G` picks
        # the block, and the gather is then issued at [K, N]. Doing it the other
        # way would need a rank-3 broadcast of am_b, and it would not be what a
        # real kernel does -- the index is a byte, the value is a float.
        aq_k = tl.load(aq_ptr + (rk[:, None] // G) * N + rn[None, :])  # [K, N]
        am_k = tl.load(c2_ptr + aq_k) * s2[None, :] + off[None, :]

        # PASS 1 -- the weights. Two nibbles per container, and the index is
        # UNSIGNED: unlike an affine int4 there is no bias by 8, because the
        # sign is a property of the table entry and not of the bit pattern.
        byte = tl.load(wq_ptr + (rk[:, None] // 2) * N + rn[None, :])
        idx = (byte >> ((rk[:, None] % 2) * 4)) & 0xF                 # [K, N]

        # A 512-element gather into a 16-entry table. Not an arithmetic
        # reconstruction of the value -- there is no formula for these levels.
        # Its scale is the OTHER gather's result: the chain is data, not layout.
        lvl = tl.load(c1_ptr + idx)                                   # [K, N]
        tl.store(w_ptr + rk[:, None] * N + rn[None, :], lvl * am_k)

    return nf4_codebook_dequant_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # The second-level table: positive, unequally spaced, no closed form --
    # standing in for bitsandbytes' dynamic-exponent absmax code.
    c2 = torch.tensor([0.02 * (1.35 ** i) for i in range(16)],
                      dtype=torch.float32)
    return {"wq": torch.randint(0, 256, (K // 2, N), generator=g,
                                dtype=torch.int32),
            "aq": torch.randint(0, 16, (NB, N), generator=g, dtype=torch.int32),
            "c1": torch.tensor(NF4, dtype=torch.float32),
            "c2": c2,
            "s2": torch.rand(N, generator=g, dtype=torch.float32) * 0.5 + 0.25,
            "off": torch.rand(N, generator=g, dtype=torch.float32) * 0.05}


def reference(inputs):
    import torch
    aq = inputs["aq"].to(torch.int64)
    am = inputs["c2"][aq] * inputs["s2"][None, :] + inputs["off"][None, :]

    byte = inputs["wq"]
    lo = byte & 0xF
    hi = (byte >> 4) & 0xF
    idx = torch.stack([lo, hi], dim=1).reshape(K, N).to(torch.int64)
    lvl = inputs["c1"][idx]

    w = lvl * am.repeat_interleave(G, dim=0)
    return {"am": am.contiguous(), "w": w.contiguous()}


SPEC = KernelSpec(
    name="nf4_codebook_dequant_kernel", kernel=kernel,
    signature={"wq_ptr": "*i32", "aq_ptr": "*i32", "c1_ptr": "*fp32",
               "c2_ptr": "*fp32", "s2_ptr": "*fp32", "off_ptr": "*fp32",
               "am_ptr": "*fp32", "w_ptr": "*fp32", "K": "constexpr",
               "N": "constexpr", "G": "constexpr", "NB": "constexpr"},
    constexprs={"K": K, "N": N, "G": G, "NB": NB},
    args=[Arg("wq", IN, "int32", (K // 2, N)),
          Arg("aq", IN, "int32", (NB, N)),
          Arg("c1", IN, "float32", (16,)),
          Arg("c2", IN, "float32", (16,)),
          Arg("s2", IN, "float32", (N,)),
          Arg("off", IN, "float32", (N,)),
          Arg("am", OUT, "float32", (NB, N)),
          Arg("w", OUT, "float32", (K, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-6,
)
