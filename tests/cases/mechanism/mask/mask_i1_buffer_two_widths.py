"""An i1 mask written by one loop and read by another that issues a different width.

    m      = x > 0                               cheap body  -> stays wide
    o      = where(m, a*b + c - a*c, 0)          crowded body -> must narrow

The mask is a real scratchpad buffer between the two -- `memref<R x C x i1, 1>`
-- because the compare and the select end up in separate banked loops. That is
ordinary. What is not ordinary is that the two loops need not agree on how many
elements one instruction moves, and for a SUB-BYTE element type they have to.

WHY. An `i1` memref is ADDRESSED one byte per element (element (i, j) is at
`base + i*C + j`) but a `vector<c x i1>` is MOVED PACKED -- `vsm.v`/`vlm.v`
transfer ceil(c/8) bytes. So the bits for elements j..j+c-1 land at BYTE OFFSET
j, and where element k lives depends on c. Measured on alibi's 06-kernel.s when
the two widths first came apart:

    producer, 64 wide    vmsgt.vi v16, v8, -1 ; vsm.v v16, (a2)
                         64 bits -> bytes 0..7 of the row
    consumer, 32 wide    vlm.v v0, (a0) ; addi a0, a0, 32
                         reads bytes 32..35 -- nothing was ever written there

and 6078 of 16384 outputs came back wrong, with no diagnostic anywhere: the IR
verifies, the assembly is legal, and the bytes are simply not the ones the other
loop wrote.

WHAT BREAKS THIS. legalize picking each loop's element count on its own.
It has to give every loop over a sub-byte buffer the same count -- the smallest
of the ones it chose, since those came out of a register-file search and
lowering one is always still legal. A buffer of i8 or wider needs no such
coupling: one element is at least one addressable unit, so its address does not
depend on how many are moved.

The crowded arm exists to make the consumer narrow; without it both loops pick
the same width by accident and this kernel pins nothing.

TORCH: o = where(x > 0, a*b + c - a*c, 0)

A packed i1 mask buffer whose producer and consumer loops would otherwise pick different vector widths: a sub-byte element's byte address depends on how many are moved at a time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, K = 64, 64, 2
N = R * C
KN = K * N


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_i1_buffer_two_widths_kernel(x_ptr, a_ptr, b_ptr, c_ptr, o_ptr,
                                         R: tl.constexpr, C: tl.constexpr):
        i = tl.arange(0, R)
        j = tl.arange(0, C)
        off = i[:, None] * C + j[None, :]
        # Producer: two values live, so this loop keeps the widest count. The
        # static_range below is what keeps the mask a BUFFER rather than letting
        # fuse fold the compare into its one consumer.
        m = tl.load(x_ptr + off) > 0.0
        for k in tl.static_range(2):
            # Consumer: a, b, c and two products live at once, so the register
            # file forces a narrower count here than the compare above chose.
            a = tl.load(a_ptr + k * R * C + off)
            b = tl.load(b_ptr + k * R * C + off)
            c = tl.load(c_ptr + k * R * C + off)
            tl.store(o_ptr + k * R * C + off,
                     tl.where(m, a * b + c - a * c, 0.0))

    return mask_i1_buffer_two_widths_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            **{n: torch.randn(KN, generator=g, dtype=torch.float32)
               for n in ("a", "b", "c")}}


def reference(inputs):
    import torch
    x, a, b, c = (inputs[n] for n in ("x", "a", "b", "c"))
    m = (x > 0.0).repeat(K)
    o = torch.where(m, a * b + c - a * c, torch.zeros_like(a))
    return {"o": o.contiguous()}


SPEC = KernelSpec(
    name="mask_i1_buffer_two_widths_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32",
               "c_ptr": "*fp32", "o_ptr": "*fp32",
               "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (N,)), Arg("a", IN, "float32", (KN,)),
          Arg("b", IN, "float32", (KN,)), Arg("c", IN, "float32", (KN,)),
          Arg("o", OUT, "float32", (KN,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
