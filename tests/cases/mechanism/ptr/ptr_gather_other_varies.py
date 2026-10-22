"""an unstructured load whose `other=` VARIES per element -- still refused.

The other half of ptr_bitcast_bool_tile.py, and the reason that one is allowed
through.  Both take the unstructured (per-element) load path, by the same
route: a `*i1` pointee makes triton bitcast the POINTER TENSOR, PtrAnalysis
cannot walk that, so the access never becomes a `tts.load`.

The difference is the fill, and it is the whole point:

    ptr_bitcast_bool_tile   other = 0            linalg.fill  -> uniform
    this file               other = (r > 50)     linalg.generic -> varies

`tts.gather`'s `other` slot holds ONE value, which _empty_read_mask_carriers splats across
the result.  A uniform tile says the same thing as that scalar, so convert_unsupported_pointers reduces
it (`uniform_scalar`) and wires it.  A tile that varies does not: there is no
single value to put in the slot, and picking one of its elements -- or dropping
it and letting the masked-off positions read zero -- would be a WRONG ANSWER
that nothing downstream could detect, because the addresses and the types are
all still correct.  So convert_unsupported_pointers refuses it by name.

WHAT BREAKS THIS KERNEL
  It starts passing if the gather ever learns a tile-valued fill, or if someone
  "generalises" uniform_scalar into returning an arbitrary element.  The first
  is a real feature and this file should then flip to expect="pass" with the
  reference below, which is already the correct math.  The second is the bug
  this is here to catch.

  Change `other=(r > 50)` to `other=0` and it compiles and runs -- that is
  ptr_bitcast_bool_tile.py.  The two files differ in exactly that expression.

The refusing half of ptr_bitcast_bool_tile.py. Same unstructured load path, non-uniform `other` tile.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

N = 128
BLOCK = 128
LIVE = 100          # < BLOCK, so the mask really masks and `other` is reachable
SPLIT = 50          # where the `other` tile changes value, so it is not uniform


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_other_varies_kernel(m_ptr, x_ptr, o_ptr, LIVE: tl.constexpr,
                                       SPLIT: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        ok = r < LIVE
        # `other` is a TILE that varies with r. The gather's fill is one value,
        # so there is nothing correct to put in the slot -- convert_unsupported_pointers must refuse.
        m = tl.load(m_ptr + r, mask=ok, other=(r > SPLIT))
        x = tl.load(x_ptr + r, mask=ok, other=0.0)
        tl.store(o_ptr + r, tl.where(m, x, 0.0))

    return ptr_gather_other_varies_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"m": (torch.rand(N, generator=g) > 0.5),
            "x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(N)
    live = r < LIVE
    # masked-off positions take `other`, which here is (r > SPLIT)
    m = torch.where(live, inputs["m"], r > SPLIT)
    x = torch.where(live, inputs["x"], torch.zeros_like(inputs["x"]))
    return {"o": torch.where(m, x, torch.zeros_like(x))}


SPEC = KernelSpec(
    name="ptr_gather_other_varies_kernel",
    kernel=kernel,
    signature={"m_ptr": "*i1", "x_ptr": "*fp32", "o_ptr": "*fp32",
               "LIVE": "constexpr", "SPLIT": "constexpr", "BLOCK": "constexpr"},
    constexprs={"LIVE": LIVE, "SPLIT": SPLIT, "BLOCK": BLOCK},
    args=[Arg("m", IN, "bool", (N,)),
          Arg("x", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect="refuse",
    expect_stop=Stop("transform", Reason.FILL_VARIES),
    expect_why="the gather's `other` is a single value that _empty_read_mask_carriers "
               "splats; a per-element fill has no correct scalar to reduce to, "
               "and choosing one would be silently wrong.",
)
