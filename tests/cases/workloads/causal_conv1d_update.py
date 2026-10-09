"""Mamba's conv1d at DECODE -- one token, and a ring buffer that is shifted in place.

    window  = concat(state[:, 0:W-1], x)            # W taps ending at this token
    conv    = sum_k w[:, k] * window[:, k] + b      # per channel, depthwise
    y       = conv * sigmoid(conv)                  # SiLU, same as prefill
    state  <- window[:, 1:W]                        # drop the oldest, append x

WHY THIS EXISTS AT ALL AS A SEPARATE KERNEL. `causal_conv1d.py` is the PREFILL
form: the whole sequence is resident, so tap k is just a shifted read of the same
buffer and the left pad is a mask. At decode there is no sequence. One token
arrives, the previous W-1 tokens exist ONLY as a per-request cache tensor, and
the same convolution has to be re-expressed as a read-modify-write of that cache.
mamba-ssm ships the two as `causal_conv1d_fn` and `causal_conv1d_update` for
exactly this reason, and every serving stack calls the second one once per
generated token -- so it runs T times more often than the first.

WHAT IS STRUCTURALLY DIFFERENT FROM causal_conv1d.py
  * IT HAS STATE. That kernel is pure: inputs in, outputs out, nothing carried.
    This one's `state` is INOUT -- the same buffer is read, then overwritten, in
    one launch. Every load of the old ring must be issued before the store of the
    new one, which is a hazard the pure form cannot express; a backend that sinks
    a load past the store gets a wrong answer, not a compile error.
  * THE SEQUENCE AXIS IS GONE. There is no L. The whole tile IS the cache,
    W-1 = 3 columns wide, so the shift is not a sweep along a long axis but a
    one-column rotation of a tile barely wider than its element.
  * THE MASK MOVED. Prefill masks the LEFT edge of the sequence. Here the
    boundary is the RIGHT end of the ring: the newest column does not come from a
    shifted read, it comes from `x`. The same `tl.where` that was a zero-pad is
    now a splice of two different sources.

THE CACHE IS 3 WIDE AND THE TILE IS 4, WHICH IS THE OTHER HALF OF THE POINT.
`tl.arange` refuses a non-power-of-two extent, and W-1 is 3 for every Mamba
config in existence. So the tile is rounded up to W and EVERY touch of the cache
carries a predicate: the load is masked at k >= W-1, the splice fills that column
with x, and the store is masked back down again. The padded column is not waste
-- it is exactly where the new token goes, which is why the convolution comes out
as one clean reduction over all W taps instead of W-1 taps plus a fixup.

WHY THE SHIFT IS A RELOAD AND NOT A REGISTER ROTATE. The tap index is the in-lane
axis, and a rotate along it is a lane-local permute this target has no primitive
for. Reading the cache a second time at offset +1 turns the permute into address
arithmetic -- the same trade `causal_conv1d.py` makes for its taps.

W=4 because `d_conv=4` is what every Mamba and Mamba-2 config ships. `conv` is
stored beside `y` so a bad tap can be told from a bad SiLU, and `state` is
checked so a bad tap can be told from a bad rotation.

IT USED TO ABORT THE COMPILER, AND THAT IS WHAT IT PINS. Before vectorize's
`_mask_tail` was fixed, this kernel reached the suite as `KILLED by signal 6
(SIGABRT) -- no diagnostic was printed`. The W=4 tap axis is SHORTER than
MIN_VEC, so the accumulator is widened past it and the tail has to be masked
with the reduction's identity -- and that mask was built by handing
`DenseElementsAttr.get` a list of Python bools, which it has no path for:

    RuntimeError: Invalid attribute when attempting to create an
    ArrayAttribute (std::bad_cast)

The raise then escaped `_nest`, which was driven by a bare `__enter__`, so the
insertion points were never left: the exception was replaced by `Unbalanced
Location enter/exit` from transform.run's own `with`, and the abandoned generator
was finalised at interpreter shutdown against a dead context -- `Fatal Python
error: PyThreadState_Get`, i.e. SIGABRT, AFTER the driver had already reported
the failure and returned 1.

So W=4 is not decoration here. Any reduction whose axis is under MIN_VEC walks
that path, and this is the smallest kernel in the suite that does.

Mamba decode conv: an INOUT ring buffer read then overwritten in one launch, on a 3-wide cache carried in a 4-wide tile. causal_conv1d.py is the same filter with no state.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT, INOUT

D = 64                          # channels (d_inner)
W = 4                           # d_conv -- the tile width, a power of two
WM1 = W - 1                     # the cache is one tap short of the filter, and 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def causal_conv1d_update_kernel(x_ptr, w_ptr, b_ptr, state_ptr,
                                    c_ptr, y_ptr,
                                    D: tl.constexpr, W: tl.constexpr,
                                    WM1: tl.constexpr):
        d = tl.arange(0, D)[:, None]
        k = tl.arange(0, W)[None, :]            # tap index; the cache holds k<W-1
        rd = tl.arange(0, D)

        x = tl.load(x_ptr + rd)                 # the one new token, [D]
        # The ring, widened to a power of two. Column W-1 is not in memory.
        st = tl.load(state_ptr + d * WM1 + k,
                     mask=tl.broadcast_to(k < WM1, (D, W)), other=0.0)
        xb = tl.broadcast_to(x[:, None], (D, W))
        win = tl.where(k == WM1, xb, st)        # the full W-tap window

        # ONE reduction on the inner axis, all W taps. Depthwise: no channel mix.
        acc = tl.sum(win * tl.load(w_ptr + d * W + k), axis=1) + tl.load(b_ptr + rd)
        tl.store(c_ptr + rd, acc)
        tl.store(y_ptr + rd, acc * tl.sigmoid(acc))

        # THE RING WRITE: new[:, k] = win[:, k+1]. Re-read rather than permute;
        # the k+1 read leaves the buffer on the last channel, so it is predicated
        # even though the splice replaces that column with x.
        nxt = tl.load(state_ptr + d * WM1 + (k + 1),
                      mask=tl.broadcast_to(k + 1 < WM1, (D, W)), other=0.0)
        tl.store(state_ptr + d * WM1 + k, tl.where(k + 1 == WM1, xb, nxt),
                 mask=tl.broadcast_to(k < WM1, (D, W)))

    return causal_conv1d_update_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(D, generator=g, dtype=torch.float32),
            "w": torch.randn(D * W, generator=g, dtype=torch.float32) * 0.5,
            "b": torch.randn(D, generator=g, dtype=torch.float32) * 0.1,
            "state": torch.randn(D * WM1, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    w = inputs["w"].reshape(D, W)
    st = inputs["state"].reshape(D, WM1)
    window = torch.cat([st, x[:, None]], dim=1)             # [D, W]
    conv = (window * w).sum(dim=1) + inputs["b"]
    return {"state": window[:, 1:].reshape(-1).contiguous(),
            "c": conv.contiguous(),
            "y": torch.nn.functional.silu(conv).contiguous()}


SPEC = KernelSpec(
    name="causal_conv1d_update_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "b_ptr": "*fp32",
               "state_ptr": "*fp32", "c_ptr": "*fp32", "y_ptr": "*fp32",
               "D": "constexpr", "W": "constexpr", "WM1": "constexpr"},
    constexprs={"D": D, "W": W, "WM1": WM1},
    args=[Arg("x", IN, "float32", (D,)),
          Arg("w", IN, "float32", (D * W,)),
          Arg("b", IN, "float32", (D,)),
          Arg("state", INOUT, "float32", (D * WM1,)),
          Arg("c", OUT, "float32", (D,)),
          Arg("y", OUT, "float32", (D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
