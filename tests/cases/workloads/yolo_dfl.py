"""YOLOv8's Distribution Focal Loss decode -- a softmax whose output is a number.

    d_e   = sum over bins c of  softmax(logits[a, e, :])[c] * c     for e in ltrb
    x1y1  = anchor - (d_l, d_t)
    x2y2  = anchor + (d_r, d_b)
    box   = (centre, size) * stride[a]

torch equivalent: `Detect.forward` -- `self.dfl(x)` (a Conv2d whose weights are
frozen at `arange(reg_max)`) then `dist2bbox(..., xywh=True) * strides`.

A BOX EDGE IS PREDICTED AS A HISTOGRAM. A direct regression head outputs one
float per edge and has no way to say "I am unsure, it is somewhere between 4 and
7". Girshick-style DFL outputs C=16 LOGITS per edge instead, softmaxes them into
a distribution over integer distances, and takes the EXPECTATION. The network
can then be genuinely bimodal on an occluded boundary, and the loss can push
mass toward the two integers bracketing the label rather than pulling a single
scalar. YOLOv8, YOLOv11, PP-YOLOE and RTMDet all ship this head.

THE EXPECTATION IS A SOFTMAX FOLLOWED BY A SECOND FOLD OVER THE SAME AXIS, and
that is what makes it different from every softmax already here. softmax.py,
softmax_causal.py and sampling.py all END at the distribution -- the reduction's
result is the output. Here the distribution is an intermediate: it is
immediately contracted against `arange(C)`, a CONSTANT VECTOR, collapsing the
bin axis away entirely. So the axis is folded three times in a row (max, sum,
then the weighted sum) and nothing downstream has a bin axis at all.

  Ultralytics implements that contraction as a 1x1 Conv2d with frozen weights,
  which is a matmul against a constant. Here it is `tl.sum(p * c, axis=1)` --
  the constant is `tl.arange`, so it never touches memory, and the whole decode
  is one pass with no second kernel and no weight tensor.

FOUR EDGES, FOUR SEPARATE TILES, AND ON PURPOSE. The natural layout is
[A, 4, C], which is rank 3; flattening it to [A*4, C] gives the right softmax
but then the four distances arrive as one vector of length A*4 that the box
decode needs as [A, 4] -- and reshaping a COMPUTED tensor is exactly what this
backend will not do. So the bin axis is folded four times, once per edge, with
the edge index a literal in the address (`e * C`), and each fold hands back a
clean [A] vector that the decode can use directly. The @triton.jit helper takes
`e` as a tl.constexpr and is called with 0, 1, 2, 3.

THE DECODE IS AN ANCHOR-RELATIVE, STRIDE-SCALED AFFINE MAP, and the stride is
PER ANCHOR because the anchors come from different pyramid levels concatenated
into one tensor -- so `* stride` is a per-row broadcast out of memory, not the
constant it looks like.

Distances are non-negative by construction (the bins are 0..C-1 and the weights
are a probability simplex), which is the other reason the head is written this
way: no clamp is needed to keep the box from inverting.

The only softmax in the suite that is not the answer: it is immediately contracted against arange(C) to an expectation, collapsing the bin axis, then decoded against per-anchor strides.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

A = 32                          # anchor points (8x4 here, two pyramid levels)
C = 16                          # reg_max -- YOLOv8's actual value
GW = 8                          # grid width used to lay the anchors out


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _expect(p_ptr, E: tl.constexpr, A: tl.constexpr, C: tl.constexpr):
        """softmax over the C bins of one edge, then the dot with arange(C)."""
        a = tl.arange(0, A)
        c = tl.arange(0, C)
        z = tl.load(p_ptr + a[:, None] * (4 * C) + E * C + c[None, :])
        z = z - tl.max(z, axis=1)[:, None]
        e = tl.exp(z)
        p = e / tl.sum(e, axis=1)[:, None]
        # The frozen arange(C) "convolution", as a constant that never loads.
        return tl.sum(p * c.to(tl.float32)[None, :], axis=1)

    @triton.jit
    def yolo_dfl_kernel(p_ptr, anc_ptr, str_ptr, o_ptr,
                        A: tl.constexpr, C: tl.constexpr):
        a = tl.arange(0, A)

        # ltrb, each its own fold over the bin axis. Four calls with literal
        # edge indices, because [A, 4, C] would be rank 3 and [A*4, C] would
        # need the result reshaped.
        dl = _expect(p_ptr, 0, A, C)
        dt = _expect(p_ptr, 1, A, C)
        dr = _expect(p_ptr, 2, A, C)
        db = _expect(p_ptr, 3, A, C)

        ax = tl.load(anc_ptr + a * 2 + 0)
        ay = tl.load(anc_ptr + a * 2 + 1)
        st = tl.load(str_ptr + a)          # per anchor: pyramid levels differ

        x1 = ax - dl
        y1 = ay - dt
        x2 = ax + dr
        y2 = ay + db

        # dist2bbox(xywh=True): centre and size, then out of grid units.
        tl.store(o_ptr + a * 4 + 0, (x1 + x2) * 0.5 * st)
        tl.store(o_ptr + a * 4 + 1, (y1 + y2) * 0.5 * st)
        tl.store(o_ptr + a * 4 + 2, (x2 - x1) * st)
        tl.store(o_ptr + a * 4 + 3, (y2 - y1) * st)

    return yolo_dfl_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    i = torch.arange(A, dtype=torch.float32)
    # A real anchor grid: cell centres at +0.5, in grid units.
    anc = torch.stack([i % GW + 0.5, torch.div(i, GW, rounding_mode="floor")
                       + 0.5], dim=1)
    # Two pyramid levels concatenated, which is why stride is per anchor.
    st = torch.where(i < A // 2, 8.0, 16.0)
    return {"p": torch.randn(A * 4 * C, generator=g, dtype=torch.float32) * 2.0,
            "anc": anc.reshape(-1).contiguous(),
            "st": st}


def reference(inputs):
    import torch
    p = torch.softmax(inputs["p"].reshape(A, 4, C), dim=-1)
    d = (p * torch.arange(C, dtype=torch.float32)).sum(-1)      # (A, 4) ltrb
    anc = inputs["anc"].reshape(A, 2)
    x1y1 = anc - d[:, :2]
    x2y2 = anc + d[:, 2:]
    box = torch.cat([(x1y1 + x2y2) * 0.5, x2y2 - x1y1], dim=1)
    return {"o": (box * inputs["st"][:, None]).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="yolo_dfl_kernel", kernel=kernel,
    signature={"p_ptr": "*fp32", "anc_ptr": "*fp32", "str_ptr": "*fp32",
               "o_ptr": "*fp32", "A": "constexpr", "C": "constexpr"},
    constexprs={"A": A, "C": C},
    args=[Arg("p", IN, "float32", (A * 4 * C,)),
          Arg("anc", IN, "float32", (A * 2,)),
          Arg("st", IN, "float32", (A,)),
          Arg("o", OUT, "float32", (A * 4,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
