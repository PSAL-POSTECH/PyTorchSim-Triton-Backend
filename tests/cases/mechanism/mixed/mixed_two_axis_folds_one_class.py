"""Folds of ONE buffer on two axes, UNBALANCED, whose results get tied together.

    c = tl.dot(a, b)
    r1 = tl.sum(c, 1);  max(c, 1);  min(c, 1)      three fold axis 1
    r0 = tl.sum(c, 0)                              one folds axis 0

THE IMBALANCE IS LOAD-BEARING and was found by trying without it: three-against-
one makes fit_layout_to_lanes TURN the group (see mixed_reduce_three_to_one, which
exists for that turn), and the turn is what pulls two differently-folded results
toward one class. One fold against one does not turn, and does not crash.

WHAT WOULD TIE THEM, and what this file is here to keep from arriving: an
identity `linalg.generic` reading one and writing the other. That is what
`bank_copies` makes of a `memref.copy`, deliberately -- its whole argument is
that identity maps put axis d of both operands in ONE axis class so
fit_layout_to_lanes must agree or report a conflict. Here agreeing is impossible:
one buffer's axis comes from a fold of axis 1 and the other's from a fold of
axis 0, and `axis_of` walks `_folds` to derive each.

MEASURED, 2026-09-02, by turning `_refill_copies` off so the copy in this shape
became a generic instead of a second fill:

    Fatal Python error: Segmentation fault
      lib_mlir_util.py:152 in rank_of          ShapedType(v.type) on a dead Value
      lib_axis_graph.py:1724 in axis_of
      lib_axis_graph.py:1167 in _derived
      lib_axis_graph.py:1782 in axis_of
      p08_fit_layout_to_lanes.py:5437 in _emit_buffer_stamps

`_derived` guards recursion with `if value.__hash__() in self._deriving`, and
the walk is over (value, where) PAIRS -- a different `where` re-enters the same
value, the guard does not fire, and the stack runs out inside MLIR's C++.

WHOSE IT IS. fit_layout_to_lanes's, not bufferize's: an identity generic is the
ordinary output of `_bank_copies` 2731 times over this suite, and crashing on
one of them is a guard that is keyed on too little. bufferize's `_refill_copies`
happens to keep this shape from arising -- 334 copies become second fills rather
than generics -- which is why nothing has ever hit it. That is a mask, not a
fix, and this kernel is here so the next thing to emit an identity generic over
two differently-folded buffers finds a test rather than a segfault.

torch equivalent: two reductions of one matmul, on the two axes.

Two folds of one tile on two axes. Records that tying their results into one axis class segfaults fit_layout_to_lanes's derivation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = N = K = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_two_axis_folds_one_class_kernel(a_ptr, b_ptr, r1_ptr, r0_ptr,
                                              M: tl.constexpr, N: tl.constexpr,
                                              K: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + n[None, :])
        c = tl.dot(a, b)
        tl.store(r1_ptr + m, tl.sum(c, axis=1))
        tl.store(r1_ptr + M + m, tl.max(c, axis=1))
        tl.store(r1_ptr + 2 * M + m, tl.min(c, axis=1))
        tl.store(r0_ptr + n, tl.sum(c, axis=0))

    return mixed_two_axis_folds_one_class_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M * K, generator=g, dtype=torch.float32),
            "b": torch.randn(K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    c = inputs["a"].view(M, K) @ inputs["b"].view(K, N)
    import torch
    return {"r1": torch.cat([c.sum(dim=1), c.amax(dim=1), c.amin(dim=1)]),
            "r0": c.sum(dim=0)}


SPEC = KernelSpec(
    name="mixed_two_axis_folds_one_class_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "r1_ptr": "*fp32",
               "r0_ptr": "*fp32", "M": "constexpr", "N": "constexpr",
               "K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K},
    args=[Arg("a", IN, "float32", (M * K,)), Arg("b", IN, "float32", (K * N,)),
          Arg("r1", OUT, "float32", (3 * M,)), Arg("r0", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
