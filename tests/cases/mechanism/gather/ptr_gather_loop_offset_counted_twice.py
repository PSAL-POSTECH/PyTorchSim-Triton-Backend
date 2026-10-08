"""A gather whose index is non-affine in the LOOP index: the offset is added twice.

SEGFAULT, NOT A WRONG NUMBER. The address the DMA is handed leaves the buffer
by a factor of the modulus, so this file is red at stage 7 with

    User load segfault @ 0x...

and never reaches verify. The first trip is fine -- its base term is zero --
so the read that dies is the SECOND one.

WHAT THE LINALG LOWERING BUILDS (03-linalg.mlir), for the load below:

    %80  = fill(%arg20) + arange(RBLOCK)     r0 = r0_offset + r
    %111 = (%80 / 2) + (%80 % 2) * HALF      the gather's index tensor
    %105 = index_cast(%arg20) * HALF         <- the loop term, again
    %107 = %105 + pid_x * R
    tts.make_gather_scatter_tptr(base, %111, R, %107)

`%111` is computed from `%80`, which ALREADY carries `%arg20` -- the loop
offset is inside the index tensor. `%105` adds it a second time, and with the
wrong coefficient on top: `r0 % 2` does not vary with `r0_offset` at all (the
step is a multiple of the modulus, so `(r0_offset + r) % 2 == r % 2`), and
`r0 // 2` advances by `r0_offset / 2`, which is the part the index tensor
already holds. The loop's correct contribution to the dim-0 offset is ZERO.

MEASURED on the numbers below: at `r0_offset = RBLOCK`, the base term is
`RBLOCK * HALF` = 131072 elements into a buffer of 32768 -- four times past
its end, and eight times past in the model kernel this came from.

WHERE IT CAME FROM. Stable Diffusion v1.5's UNet,
`triton_npu_fused_add_addmm_clone_convolution_div_native_group_norm_permute_silu_view_83`
-- a GroupNorm Welford over a PERMUTED tile, `r0_1 = r0 % 2`, `r0_2 = r0 // 2`,
reading `in_ptr2 + (r0_2 + 1024*r0_1 + 2048*x0)`. Under SPIKE_DEBUG the second
trip asks for `in_ptr2 + 0x200000` on a buffer of 0x40000 bytes, and the
indirect index the DMA reads at that same element is 256 -- which is
`r0_offset / 2`, the loop offset, proving it is in the index tensor too.

WHY THE WEIGHT IS THERE. `(r0 // 2) + HALF * (r0 % 2)` is a PERMUTATION of
[0, R): every element of the row is read exactly once, so a plain sum returns
the same answer whatever order the gather delivers them in, and the kernel
would go green on a fix that got the order wrong. The `* (r0 + 1)` weight
makes the answer depend on WHICH element landed WHERE.

torch equivalent, per row i:  sum_r x[i, (r // 2) + HALF * (r % 2)] * (r + 1).

gather index non-affine in the loop index: PtrAnalysis adds the loop offset to the dim-0 base as well, and the second trip reads off the end of the buffer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X = 32
R = 1024
HALF = R // 2
XBLOCK = 32
RBLOCK = 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_loop_offset_counted_twice_kernel(
            x_ptr, o_ptr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr,
            R: tl.constexpr, HALF: tl.constexpr):
        xi = tl.arange(0, XBLOCK)[:, None]
        acc = tl.zeros([XBLOCK, RBLOCK], tl.float32)
        for r0_offset in range(0, R, RBLOCK):
            r0 = r0_offset + tl.arange(0, RBLOCK)[None, :]
            idx = (r0 // 2) + HALF * (r0 % 2) + R * xi
            acc += tl.load(x_ptr + idx) * (r0 + 1).to(tl.float32)
        tl.store(o_ptr + tl.arange(0, XBLOCK), tl.sum(acc, 1))

    return ptr_gather_loop_offset_counted_twice_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(X * R, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(X, R)
    r = torch.arange(R)
    idx = (r // 2) + HALF * (r % 2)
    w = (r + 1).to(torch.float32)
    return {"o": (x[:, idx] * w).sum(dim=1)}


SPEC = KernelSpec(
    name="ptr_gather_loop_offset_counted_twice_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "XBLOCK": "constexpr",
               "RBLOCK": "constexpr", "R": "constexpr", "HALF": "constexpr"},
    constexprs={"XBLOCK": XBLOCK, "RBLOCK": RBLOCK, "R": R, "HALF": HALF},
    args=[Arg("x", IN, "float32", (X * R,)), Arg("o", OUT, "float32", (X,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
