"""A wrap on the ROW axis, three times, refused three different ways.

Split out of the old tests/cases/ptr_structured.py section E. When PtrAnalysis
leaves a non-zero entry in `shape:` (see ptr_modulo_wrap_shapes.py for how it
gets there), b11 tries to cut the axis into RUNS -- one strided piece per lap of
the wrap, which lower_to_dma emits as one descriptor each. _cut_facts refuses in
three cases and hands the access to _emit_per_element instead, and the three
loads here are one of each. That is what this file is for.

    j(p) = (o + p*st) // W is the lap; within one lap the address is
    o + p*st - j*W, linear in p. Runs can only be counted when o, st, W and n
    are all known HERE.

COVERS (3)   a02_convert_tts_to_pytorchsim.py, _cut_facts / _wrap_cut

  wrap-offset-runtime     load a. `rows = pid * BM + rm`, so `o` is an SSA
      value and _as_int gives None -- the lap boundaries are not known at
      compile time. WM itself is a constexpr; only the START is runtime.
  wrap-extent-runtime     load b. `rows % wmod` with wmod an i32 kernel
      ARGUMENT, so `W` is the unknown one. Same refusal, other operand -- the
      two ids exist because a fix for one need not fix the other.
      (This is the open "runtime-geometry wrap" class.)
  wrap-axis-also-clamped  load c. The mask bounds the same axis the wrap cuts.
      Outside the box a clamp wants the FILL and a piece wants a SKIP, and one
      axis has one box to say it with -- the explicit `_narrowed_axes & wdims`
      guard, not an _as_int failure. Contrast ptr_wrap_side_by_side.py, where
      the clamp is on the OTHER axis and the guard correctly stays quiet.

WM is 12 and the block is 16 rows, so the seam is inside the tile for program
0: no launch geometry is needed to hit it. ptr_wrap_stacked_run.py is the same
shape with a STATIC start and no mask, and it does get its runs.

THE SEPARATE BASE ARGUMENT (wsm_ptr) IS NOW INERT. It was here because
StructuredToMemref's rewriteMaskedLoad erased a two-operand
unrealized_conversion_cast that rewriteStructuredLoad still needed, so two wrap
loads sharing one tts.make_tptr crashed it. That pass is deleted (rule 21) and
b11 lowers each access on its own; the second base costs nothing and is kept so
the three loads stay independently readable.

Row-modulo wrap, refused three ways: runtime start, runtime extent, and a clamp on the wrapped axis. Runtime i32 scalar wmod=12. Numerically runnable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

WM = 12          # wrap extent in ROWS -- smaller than BM, so the seam is inside
BM = 16
BN = 32
NW = WM * BN
NO = BM * BN
COLKEEP = 16     # the masked load's column bound


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_stacked_kernel(ws_ptr, wsm_ptr, o_ptr, wmod,
                                WM: tl.constexpr, BM: tl.constexpr,
                                BN: tl.constexpr):
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        rows = pid * BM + rm
        wr = rows % WM

        # 1 unmasked, compile-time wrap extent
        a = tl.load(ws_ptr + wr[:, None] * BN + rn[None, :])
        # 2 unmasked, RUNTIME wrap extent (same base: different shape, no CSE)
        b = tl.load(ws_ptr + (rows % wmod)[:, None] * BN + rn[None, :])
        # 3 masked -- its OWN base, see the docstring
        c = tl.load(wsm_ptr + wr[:, None] * BN + rn[None, :],
                    mask=(rm[:, None] < BM) & (rn[None, :] < 16), other=-1.0)

        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], a + b + c)

    return ptr_wrap_stacked_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"ws": torch.randn(NW, generator=g, dtype=torch.float32),
            "wsm": torch.randn(NW, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    ws = inputs["ws"].reshape(WM, BN)
    wsm = inputs["wsm"].reshape(WM, BN)
    rows = torch.arange(BM) % WM            # pid == 0
    a = ws[rows]
    c = torch.where(torch.arange(BN).reshape(1, BN) < COLKEEP,
                    wsm[rows], torch.tensor(-1.0))
    return {"o": (a + a + c).reshape(NO)}


SPEC = KernelSpec(
    name="ptr_wrap_stacked_kernel", kernel=kernel,
    signature={"ws_ptr": "*fp32", "wsm_ptr": "*fp32", "o_ptr": "*fp32",
               "wmod": "i32", "WM": "constexpr", "BM": "constexpr",
               "BN": "constexpr"},
    constexprs={"WM": WM, "BM": BM, "BN": BN},
    args=[Arg("ws", IN, "float32", (NW,)), Arg("wsm", IN, "float32", (NW,)),
          Arg("o", OUT, "float32", (NO,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"wmod": WM}},
)
