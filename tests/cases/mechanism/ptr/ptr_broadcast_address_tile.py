"""A BROADCAST of a tile of pointers -- the one shape `_address_move` rewrites.

A generic whose body only yields its input moves addresses rather than
computing one, so convert_unsupported_pointers rebuilds it over i64.  Getting one to exist takes two
things at once, which is why the suite had none:

  1. the pointer chain must stay UNSTRUCTURED, or the linalg lowering folds the whole
     thing into a tts.make_tptr and convert_unsupported_pointers never sees a pointer tensor.  `tl.where`
     between two bases does that (see ptr_select_between_two_bases.py).
  2. something must then BROADCAST the pointer tile, which is what produces a
     linalg.generic with an empty body -- ins map (d0,d1)->(d0), outs map
     (d0,d1)->(d0,d1), and a bare linalg.yield between them.

MEASURED BEFORE THIS FILE EXISTED: across 394 kernels, 1110 generics had an
empty body and NOT ONE of them was over pointers, so `_address_move` was
entered 1110 times and rewrote nothing.  Entering is not adopting -- rule 15a.

WHAT BREAKS THIS KERNEL
Dropping `like=op` from the `elementwise_generic` call in `_address_move`.
The maps here are NOT identity, and forging identity ones instead gives

    'linalg.generic' op expected operand #0 rank (1) to match the result rank
    of indexing_map (2)

which is the whole reason lib grew a `like=` at all: the helper's default
forges identity maps, and a broadcast is exactly the case that survives only
if the source op's own maps are carried over.

The four rows are identical on purpose -- a broadcast that dropped a row would
still agree with the reference on row 0.  The alternating sign across the
columns comes from the `tl.where`, so a select that ignored its condition
would not match either.

Coverage: the first kernel in this suite to reach the rewriting half of `_address_move`, and the one that pins `like=` -- the broadcast's maps are not identity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, R = 64, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def broadcast_address_tile_kernel(a_ptr, b_ptr, o_ptr,
                                      N: tl.constexpr, R: tl.constexpr):
        c = tl.arange(0, N)
        r = tl.arange(0, R)
        # The where leaves the chain unstructured; the broadcast then makes a
        # generic over POINTERS whose body is a bare yield.
        p = tl.where((c % 2) == 0, a_ptr + c, b_ptr + c)
        p2 = tl.broadcast_to(p[None, :], (R, N))
        tl.store(o_ptr + r[:, None] * N + c[None, :], tl.load(p2))

    return broadcast_address_tile_kernel


def make_inputs(torch, seed=0):
    return {"a": torch.arange(N, dtype=torch.float32),
            "b": -torch.arange(N, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(N, dtype=torch.float32)
    row = torch.where(torch.arange(N) % 2 == 0, i, -i)
    return {"o": row.unsqueeze(0).expand(R, N).contiguous()}


SPEC = KernelSpec(
    name="broadcast_address_tile_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr", "R": "constexpr"},
    constexprs={"N": N, "R": R},
    args=[Arg("a", IN, "float32", (N,)), Arg("b", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (R, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
