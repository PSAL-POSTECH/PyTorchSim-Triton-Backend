"""A SCALAR store with a mask -- the one cell of convert_accesses_to_transfer's grid that had no answer.

WHAT IT PINS. `tl.store(p + i, v, mask=c)` on a SCALAR offset becomes
`tts.scatter(%ptr, %i32, %val, %i1)`, which `_emit_scalar` lowers to one
`memref.store` on the flat DRAM view. The descriptor has no per-element
write-enable and a scalar has no descriptor at all, so the mask goes where this
pass sends every write's mask: the ADDRESS. `_park_offset` names the first
element past what emit_wrapper_c rounded the buffer up to -- inside the doubled
allocation, outside anything the kernel addresses -- and an `arith.select` on
the index picks it when the mask is false.

BOTH ARMS ARE LIVE. `o[5]` is written under a mask that is true and `o[7]`
under one that is false, and `n` is a RUNTIME scalar argument so neither folds.
A kernel with only the true arm passes whether or not the mask is honoured.

WHAT IT FAILED WITH BEFORE:

    Fatal: mask on a scalar tts.scatter not wired (it would be silently
    dropped): "tts.scatter"(%0, %c5_i32, %8, %4)

WHAT WOULD BREAK IT. Parking a scalar store into an `in` argument: only `out`
and `inout` are doubled (emit_wrapper_c), and an `in` is merely 512-byte
aligned, so there is nothing behind it to park in.

The scalar arm of 'a write's mask is an address'. Both mask arms are exercised, so dropping the mask fails on o[7].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_scalar_masked_store_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        tl.store(o_ptr + r, tl.load(x_ptr + r))
        v = tl.load(x_ptr + 3, mask=n > 0, other=0.0)
        tl.store(o_ptr + 5, v, mask=n > 0)      # taken
        tl.store(o_ptr + 7, v, mask=n < 0)      # NOT taken: o[7] keeps x[7]

    return ptr_scalar_masked_store_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    o = inputs["x"].clone()
    o[5] = inputs["x"][3]
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_scalar_masked_store_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t")], "scalar_values": {"n": N}},
)
