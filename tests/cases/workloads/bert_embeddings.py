"""BERT's input embedding -- three lookups added, then normalised.

    e = W_tok[ids] + W_pos[0..T) + W_type[types]
    out = LayerNorm(e)

THREE TABLES, ONE TILE. GPT-2 adds two (token and position); BERT adds a third
because it is trained on PAIRS of sentences -- "is B the sentence that followed
A?" -- and the model has to be able to tell which half a token is in. The
segment id is 0 or 1, so `W_type` is a table with two rows.

  The next-sentence objective it was added for turned out not to matter much
  (RoBERTa drops it and does better), but the embedding stayed: it is what lets
  one sequence carry a question and a passage, which is how BERT is used for
  extraction.

BIDIRECTIONAL IS THE ABSENCE OF SOMETHING. BERT's attention has no causal mask
at all -- that is `softmax.py`, and there is nothing to add for it. What makes
BERT BERT is the masked-token objective, which is a loss, not an operation.

WHAT IT ASKS OF THIS MACHINE, and why three is more interesting than one.
`embedding.py` already has a row gather feeding a broadcast. Here THREE
index-dependent reads land in one tile and are summed, and only one of the three
has a data-dependent index:

    W_tok[ids]      ids come from DRAM        -- a real gather
    W_pos[i]        i is the row index        -- affine, so a strided read
    W_type[types]   types come from DRAM      -- a gather of a 2-row table

The middle one is the interesting one: it looks like a gather in the source and
is not one after PtrAnalysis, so the three reads reaching the same adder arrive
by two different paths. Then the LayerNorm folds the feature axis on top.

torch equivalent: LayerNorm(w_tok[ids] + w_pos[:T] + w_type[types])

BERT: three index-dependent reads into one tile -- two gathers and one strided read that only looks like a gather -- then LayerNorm.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, V, C, TY = 64, 128, 32, 2       # tokens, vocab, hidden, segment types
EPS = 1e-5


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def bert_embeddings_kernel(ids_ptr, ty_ptr, wt_ptr, wp_ptr, wy_ptr,
                               g_ptr, b_ptr, o_ptr,
                               T: tl.constexpr, V: tl.constexpr,
                               C: tl.constexpr, TY: tl.constexpr,
                               EPS: tl.constexpr):
        t = tl.arange(0, T)
        c = tl.arange(0, C)

        ids = tl.load(ids_ptr + t)
        typ = tl.load(ty_ptr + t)

        # Three reads into one tile: two gathers and one strided read.
        etok = tl.load(wt_ptr + ids[:, None] * C + c[None, :])
        epos = tl.load(wp_ptr + t[:, None] * C + c[None, :])
        etyp = tl.load(wy_ptr + typ[:, None] * C + c[None, :])
        e = etok + epos + etyp

        # LayerNorm over the feature axis.
        mu = tl.sum(e, axis=1) / C
        d = e - mu[:, None]
        var = tl.sum(d * d, axis=1) / C
        n = d * tl.exp(-0.5 * tl.log(var + EPS))[:, None]
        out = n * tl.load(g_ptr + c)[None, :] + tl.load(b_ptr + c)[None, :]
        tl.store(o_ptr + t[:, None] * C + c[None, :], out)

    return bert_embeddings_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"ids": torch.randint(0, V, (T,), generator=g, dtype=torch.int32),
            "ty": torch.randint(0, TY, (T,), generator=g, dtype=torch.int32),
            "wt": torch.randn(V * C, generator=g, dtype=torch.float32),
            "wp": torch.randn(T * C, generator=g, dtype=torch.float32),
            "wy": torch.randn(TY * C, generator=g, dtype=torch.float32),
            "g": torch.randn(C, generator=g, dtype=torch.float32),
            "b": torch.randn(C, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    e = (inputs["wt"].reshape(V, C)[inputs["ids"].long()]
         + inputs["wp"].reshape(T, C)
         + inputs["wy"].reshape(TY, C)[inputs["ty"].long()])
    mu = e.mean(1, keepdim=True)
    d = e - mu
    var = (d * d).mean(1, keepdim=True)
    out = d * torch.rsqrt(var + EPS) * inputs["g"][None, :] + inputs["b"][None, :]
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="bert_embeddings_kernel",
    kernel=kernel,
    signature={"ids_ptr": "*i32", "ty_ptr": "*i32", "wt_ptr": "*fp32",
               "wp_ptr": "*fp32", "wy_ptr": "*fp32", "g_ptr": "*fp32",
               "b_ptr": "*fp32", "o_ptr": "*fp32",
               "T": "constexpr", "V": "constexpr", "C": "constexpr",
               "TY": "constexpr", "EPS": "constexpr"},
    constexprs={"T": T, "V": V, "C": C, "TY": TY, "EPS": EPS},
    args=[Arg("ids", IN, "int32", (T,)), Arg("ty", IN, "int32", (T,)),
          Arg("wt", IN, "float32", (V * C,)),
          Arg("wp", IN, "float32", (T * C,)),
          Arg("wy", IN, "float32", (TY * C,)),
          Arg("g", IN, "float32", (C,)), Arg("b", IN, "float32", (C,)),
          Arg("o", OUT, "float32", (T * C,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
