"""A bottleneck epilogue: a matmul, then a residual READ at a modulo address.

    x5    = xindex % P
    resid = tl.load(r_ptr + (idx_c + C * x5), mask)
    out   = relu((acc - mean) * scale + resid)

WHERE IT COMES FROM. resnet50's bottleneck blocks, whose Inductor kernel is
`_native_batch_norm_legit_no_training_add_convolution_relu_*`. Every one of them
ends this way, and the `add` is the skip connection: the accumulator comes out of
`tl.dot`, and the tensor added to it is read back with a MODULO in its address,
because the epilogue indexes the output's flattened `xindex` while the residual
is laid out per (channel, spatial) position.

WHAT IS NEW ABOUT IT, measured against resnet18 rather than assumed. resnet18's
convolutions build the same `x5 = xindex % P` and use it in the STORE address
only -- tests/cases/mechanism/tile/tile_gs_store_folded_row.py and
mask/mask_conv_bounds_row_over_lanes.py are that side. Reading a whole tile back
through one is the bottleneck's, and it is a different question for this backend:
a store may be parked or masked per element, while a load has to produce a tile
the lane axis can be chosen for.

AND THE TILE IS AS WIDE AS THE MACHINE. `BLOCK_N` is 128 in every resnet50
bottleneck, which is exactly `vpu_num_lanes`, so the N axis maps one column per
lane with nothing left over. The suite's other tiles are 8 to 64 wide and a tail
is what they exercise; this is the case with no tail at all.

torch equivalent: relu((a @ b - mean) * scale + resid[gathered]).

resnet50's bottleneck epilogue: a residual tile READ at a modulo address, on a tile exactly as wide as the lane count.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM = 8
BN = 128          # the lane count exactly
BK = 16
P = 5             # the modulo, deliberately not a power of two
NR = P * BN       # the residual, laid out (position, channel)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_bottleneck_modulo_residual_kernel(a_ptr, b_ptr, r_ptr, m_ptr,
                                               s_ptr, o_ptr,
                                               BM: tl.constexpr,
                                               BN: tl.constexpr,
                                               BK: tl.constexpr,
                                               P: tl.constexpr):
        om = tl.arange(0, BM)
        on = tl.arange(0, BN)
        a = tl.load(a_ptr + om[:, None] * BK + tl.arange(0, BK)[None, :])
        b = tl.load(b_ptr + tl.arange(0, BK)[:, None] * BN + on[None, :])
        acc = tl.dot(a, b, allow_tf32=True)

        # The residual's own address: a modulo of the output's flat index, times
        # the channel stride. This is the line the file is about.
        x5 = om % P
        resid = tl.load(r_ptr + (on[None, :] + BN * x5[:, None]))

        mean = tl.load(m_ptr + on)
        scale = tl.load(s_ptr + on)
        v = (acc - mean[None, :]) * scale[None, :] + resid
        tl.store(o_ptr + om[:, None] * BN + on[None, :],
                 tl.maximum(v, 0.0))

    return tile_bottleneck_modulo_residual_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(BM * BK, generator=g, dtype=torch.float32),
            "b": torch.randn(BK * BN, generator=g, dtype=torch.float32),
            "r": torch.randn(NR, generator=g, dtype=torch.float32),
            "m": torch.randn(BN, generator=g, dtype=torch.float32),
            "s": torch.randn(BN, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a = inputs["a"].reshape(BM, BK)
    b = inputs["b"].reshape(BK, BN)
    r = inputs["r"].reshape(P, BN)
    m = inputs["m"].reshape(1, BN)
    s = inputs["s"].reshape(1, BN)
    x5 = torch.arange(BM) % P
    acc = a @ b
    v = (acc - m) * s + r[x5]
    return {"o": torch.clamp(v, min=0.0).reshape(BM * BN)}


SPEC = KernelSpec(
    name="tile_bottleneck_modulo_residual_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "r_ptr": "*fp32",
               "m_ptr": "*fp32", "s_ptr": "*fp32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr",
               "P": "constexpr"},
    constexprs={"BM": BM, "BN": BN, "BK": BK, "P": P},
    args=[Arg("a", IN, "float32", (BM * BK,)), Arg("b", IN, "float32", (BK * BN,)),
          Arg("r", IN, "float32", (NR,)), Arg("m", IN, "float32", (BN,)),
          Arg("s", IN, "float32", (BN,)),
          Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
