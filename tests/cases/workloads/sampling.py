"""One decoding step -- temperature, greedy argmax, top-k, and a cumsum.

    p       = softmax(logits / temperature)
    greedy  = argmax(p)
    thr     = the K-th largest of each row          top-k, no sort
    keep    = p >= thr,  renormalised
    cum     = cumsum(keep)                          what top-p thresholds against

THE SAMPLER IS WHERE A MODEL STOPS BEING A FUNCTION. Everything else in this
suite is deterministic arithmetic; this is the step that turns a distribution
into a token, and the three knobs are three different ways of deciding how much
of the tail to believe.

    temperature   divides the LOGITS, so it changes the distribution's shape.
                  T -> 0 approaches argmax, T > 1 flattens. Dividing the
                  probabilities instead would only rescale them.
    top-k         keep the k most likely, renormalise. A fixed count, so a flat
                  distribution keeps k plausible options and a peaked one keeps
                  k-1 implausible ones.
    top-p         keep the smallest prefix whose probability sums past p. A
                  fixed MASS, which adapts to the shape -- and it needs the
                  cumulative sum, which is why the scan is here.

TOP-K WITHOUT A SORT, and it is the same trick `moe_router` uses for k = 2: take
the max, erase it, repeat. K passes over V elements beats sorting V of them when
K is small, and K is always small. What comes out is the THRESHOLD rather than
the indices, which is what the mask wants anyway.

A SORT IS WHAT THIS KERNEL DOES NOT HAVE, and top-p properly needs one -- the
prefix has to be taken in descending order. The cumsum here runs over the row as
it stands, which exercises the scan without claiming to be top-p. Sorting on a
lane-banked machine is a rearrangement across banks, so it would be a DRAM round
trip per stage of whatever network was used; that is a real design question and
not one this kernel settles.

THE SCAN IS THE NEW OP. Every reduction in this suite collapses an axis; a scan
keeps it and makes each output depend on every earlier input. `_unroll_scan`
unrolls it (`_unroll_scan`) because there is no scan instruction -- the vector
unit can fold and it can splat, but a running total is neither.

torch equivalent:
    p = softmax(logits / temp, dim=-1)
    thr = p.topk(K, dim=-1).values[:, -1:]
    keep = torch.where(p >= thr, p, 0);  keep = keep / keep.sum(-1, keepdim=True)
    cum = keep.cumsum(-1)

A decoding step: temperature on the logits, greedy argmax, top-k by repeated max, and a CUMSUM -- the one scan in this suite.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, V, K = 8, 64, 4
TEMP = 0.7


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sampling_kernel(l_ptr, p_ptr, g_ptr, c_ptr,
                        B: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
                        TEMP: tl.constexpr):
        b = tl.arange(0, B)
        v = tl.arange(0, V)
        idx = b[:, None] * V + v[None, :]

        # TEMPERATURE divides the LOGITS -- before the softmax, so the shape of
        # the distribution changes rather than its scale.
        z = tl.load(l_ptr + idx) / TEMP
        m = tl.max(z, axis=1)
        ex = tl.exp(z - m[:, None])
        p = ex / tl.sum(ex, axis=1)[:, None]

        # GREEDY: the argmax, stored as the token id it is.
        tl.store(g_ptr + b, tl.argmax(p, axis=1))

        # TOP-K BY REPEATED MAX. After K rounds `thr` holds the K-th largest;
        # each round erases the current max so the next one finds the next.
        q = p
        thr = tl.max(q, axis=1)
        for _ in tl.static_range(K - 1):
            q = tl.where(q >= thr[:, None], -1.0, q)
            thr = tl.max(q, axis=1)

        keep = tl.where(p >= thr[:, None], p, 0.0)
        keep = keep / tl.sum(keep, axis=1)[:, None]
        tl.store(p_ptr + idx, keep)

        # THE SCAN. Each output depends on every earlier input, which is what
        # neither a fold nor a splat can express.
        tl.store(c_ptr + idx, tl.cumsum(keep, axis=1))

    return sampling_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"l": torch.randn(B * V, generator=g, dtype=torch.float32) * 2.0}


def reference(inputs):
    import torch
    z = inputs["l"].reshape(B, V) / TEMP
    p = torch.softmax(z, dim=-1)
    greedy = p.argmax(dim=-1).to(torch.int32)
    thr = p.topk(K, dim=-1).values[:, -1:]
    keep = torch.where(p >= thr, p, torch.zeros_like(p))
    keep = keep / keep.sum(-1, keepdim=True)
    return {"p": keep.reshape(-1).contiguous(),
            "g": greedy.contiguous(),
            "c": keep.cumsum(-1).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="sampling_kernel",
    kernel=kernel,
    signature={"l_ptr": "*fp32", "p_ptr": "*fp32", "g_ptr": "*i32",
               "c_ptr": "*fp32", "B": "constexpr", "V": "constexpr",
               "K": "constexpr", "TEMP": "constexpr"},
    constexprs={"B": B, "V": V, "K": K, "TEMP": TEMP},
    args=[Arg("l", IN, "float32", (B * V,)),
          Arg("p", OUT, "float32", (B * V,)),
          Arg("g", OUT, "int32", (B,)),
          Arg("c", OUT, "float32", (B * V,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
