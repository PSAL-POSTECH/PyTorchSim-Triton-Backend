"""GPT-2's tied lm_head -- one table, read as rows going in and transposed coming out.

    x      = wte[tokens] + wpe[positions]        the input embedding
    logits = x @ wte.T                           the output head, SAME wte

WEIGHT TYING IS ONE TABLE DOING TWO JOBS. Press & Wolf (2016) and Inan et al.
(2016) noticed that the input embedding and the output projection are the same
map read in opposite directions -- one turns a token id into a vector, the other
scores a vector against every token -- so making them one matrix costs nothing in
expressiveness and saves V*D parameters. For GPT-2 small that is 38M of 124M, a
third of the model, and it regularises besides: a token's input and output
representations can no longer drift apart. Every GPT-2 checkpoint ships tied,
and so do most decoders since.

WHY IT IS INTERESTING HERE AND NOT ONLY IN THE PARAMETER COUNT. This backend
gives a buffer a LANE AXIS -- which of its axes is spread across the banks -- and
the whole of `pytorchsim_triton_compiler/elf/passes/lib_lane_axis.py` exists to make one answer per buffer.
Tying asks the one question that construction is least comfortable with:

    the gather        reads wte a ROW at a time, indexed by token id. Rows are
                      what banks hold, so the scattered axis and the lane axis
                      line up and it is one index per lane (this is exactly what
                      `embedding.py` establishes).
    the head          reads wte as a MATRIX, transposed -- logits[t, v] needs
                      wte[v, :], so v runs along the matmul's free axis and d,
                      the contraction axis, runs down what the buffer stores
                      contiguously.

Same buffer, two orientations, one kernel. `rmsnorm` already makes two lane-axis
answers for one buffer meet -- this does it for the largest table in the model
and with a gather on one side, which is the version that actually appears in a
language model.

THE TRANSPOSED READ IS THE COST. wte is [V, D] row-major, so wte.T[d, v] sits at
v*D + d: the tile's fast axis has stride D and its slow axis has stride 1. That
is a strided read of the vector axis, the thing `rope.py` chose the rotate_half
form specifically to avoid. Whether it is free, a staged DMA, or a refusal is the
question this file is here to answer -- and note that a real implementation has
no way around it either, because the transpose is what tying MEANS.

WHY THE POSITION TABLE IS ADDED HERE. GPT-2 uses LEARNED positional embeddings
-- a second table wpe[T, D] added to the token embedding, no RoPE, no ALiBi. It
is a contiguous slice rather than a gather (positions are 0..T-1), so it costs
nothing, and it is included because leaving it out would make this the input
layer of a model that does not exist.

torch equivalent: x = wte[tok] + wpe[:T]; logits = x @ wte.T

GPT-2: weight tying -- one table read as a row gather going in and as a transposed matrix coming out, both in one kernel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, D, V = 128, 64, 128           # tokens (== lane count), model width, vocab
BD = 32                          # contraction step; the dot needs a K loop


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def lm_head_tied_kernel(tok_ptr, wte_ptr, wpe_ptr, o_ptr,
                            T: tl.constexpr, D: tl.constexpr,
                            V: tl.constexpr, BD: tl.constexpr):
        t = tl.arange(0, T)
        vv = tl.arange(0, V)

        logits = tl.zeros((T, V), dtype=tl.float32)
        tok = tl.load(tok_ptr + t)
        for k in range(0, D, BD):
            d = k + tl.arange(0, BD)
            # GOING IN: a row of wte per token, plus the learned position row.
            x = tl.load(wte_ptr + tok[:, None] * D + d[None, :])
            x += tl.load(wpe_ptr + t[:, None] * D + d[None, :])
            # COMING OUT: the same table, transposed -- v along the free axis,
            # d down the axis the buffer stores contiguously.
            wt = tl.load(wte_ptr + vv[None, :] * D + d[:, None])
            logits += tl.dot(x, wt)

        tl.store(o_ptr + t[:, None] * V + vv[None, :], logits)

    return lm_head_tied_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"tok": torch.randint(0, V, (T,), generator=g, dtype=torch.int32),
            "wte": torch.randn(V * D, generator=g, dtype=torch.float32) * 0.1,
            "wpe": torch.randn(T * D, generator=g, dtype=torch.float32) * 0.1}


def reference(inputs):
    import torch
    wte = inputs["wte"].reshape(V, D)
    x = wte[inputs["tok"].long()] + inputs["wpe"].reshape(T, D)
    return {"o": (x @ wte.t()).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="lm_head_tied_kernel",
    kernel=kernel,
    signature={"tok_ptr": "*i32", "wte_ptr": "*fp32", "wpe_ptr": "*fp32",
               "o_ptr": "*fp32", "T": "constexpr", "D": "constexpr",
               "V": "constexpr", "BD": "constexpr"},
    constexprs={"T": T, "D": D, "V": V, "BD": BD},
    args=[Arg("tok", IN, "int32", (T,)),
          Arg("wte", IN, "float32", (V * D,)),
          Arg("wpe", IN, "float32", (T * D,)),
          Arg("o", OUT, "float32", (T * V,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
