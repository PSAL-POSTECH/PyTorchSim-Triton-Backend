"""Transposed convolution, stride 2 -- U-Net's learned upsample.

    scatter form   out[2y + kh, 2x + kw] += w[kh, kw] * in[y, x]
    polyphase      out[2y + py, 2x + px] = sum_j w[2j + py, 2i + px] in[y-j, x-i]

WHY IT IS CALLED TRANSPOSED. A convolution is a linear map, so it has a matrix;
this op is that matrix's TRANSPOSE, which is why it takes the output shape of a
convolution back to its input shape. It is NOT the inverse -- nothing is
recovered, the weights are learned separately -- and "deconvolution" is a
misnomer that stuck anyway.

THE SCATTER FORM IS THE DEFINITION AND THE GATHER FORM IS THE IMPLEMENTATION.
A scatter needs read-modify-write into a buffer several inputs share, and with
the output on the lanes two lanes can want the same slot. The gather form has
each output read what it needs and owns its own slot, which is the only form
this machine can run -- and the parity test `(oy - kh) % 2 == 0` is what makes
them the same map: an output only sees taps that line up with the stride.

CHECKERBOARD. With K = 4 and stride 2 every output gets exactly two taps per
axis; with K = 3 the count alternates 2, 1, 2, 1, so a constant input gives a
striped output before anything is trained. It is why U-Net implementations moved
to `upsample + conv3x3`.

K = 2 HERE, AND NOT BECAUSE IT IS THE INTERESTING CASE. K = 4 is, and it does
not compile: sixteen statically unrolled masked gathers segfault the
`canonicalize,cse` MLIR runs at the end of convert_accesses_to_transfer (four are fine,
so it is the count). Written with dynamic tap loops instead it compiles and
computes the wrong numbers. Both are recorded in study/06-unet.md; K = 2 is the
non-overlapping case, exact, and the polyphase structure below is the same one
K = 4 would use.

torch equivalent: F.conv_transpose2d(x, w, stride=2)   (with w of shape
[1, 1, 4, 4], and the valid interior of the output taken)

U-Net decoder: stride-2 transposed convolution, polyphase -- each output phase is an ordinary convolution, so no parity test and no negative index. K=2; K=4 crashes canonicalize (study/06-unet.md).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H, W = 16, 16
K = 2
OH, OW = H * 2, W * 2          # the interior; the full output is 2H + K - 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def transposed_conv_kernel(x_ptr, w_ptr, o_ptr,
                               H: tl.constexpr, W: tl.constexpr,
                               K: tl.constexpr, OH: tl.constexpr,
                               OW: tl.constexpr):
        y = tl.arange(0, H)
        x = tl.arange(0, W)
        # POLYPHASE: output position 2y+py sees only taps 2j+py, so the parity
        # test disappears and each phase is an ordinary convolution.
        #
        #     out[2y + py, 2x + px] = sum_j w[2j + py, 2i + px] in[y - j, x - i]
        #
        # J = K // 2 taps per phase per axis, and no index ever goes negative
        # because `y - j` is guarded by a mask that also keeps it in range.
        J: tl.constexpr = K // 2
        for py in tl.static_range(2):
            for px in tl.static_range(2):
                acc = tl.zeros((H, W), dtype=tl.float32)
                for jy in tl.static_range(J):
                    for jx in tl.static_range(J):
                        iy = y - jy
                        ix = x - jx
                        ok = (iy >= 0)[:, None] & (ix >= 0)[None, :]
                        v = tl.load(x_ptr + iy[:, None] * W + ix[None, :],
                                    mask=ok, other=0.0)
                        acc += v * tl.load(w_ptr + (2 * jy + py) * K
                                           + (2 * jx + px))
                tl.store(o_ptr + (2 * y + py)[:, None] * OW
                         + (2 * x + px)[None, :], acc)

    return transposed_conv_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(H * W, generator=g, dtype=torch.float32),
            "w": torch.randn(K * K, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(1, 1, H, W)
    w = inputs["w"].reshape(1, 1, K, K)
    full = torch.nn.functional.conv_transpose2d(x, w, stride=2)
    return {"o": full[0, 0, :OH, :OW].reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="transposed_conv_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "H": "constexpr", "W": "constexpr", "K": "constexpr",
               "OH": "constexpr", "OW": "constexpr"},
    constexprs={"H": H, "W": W, "K": K, "OH": OH, "OW": OW},
    args=[Arg("x", IN, "float32", (H * W,)),
          Arg("w", IN, "float32", (K * K,)),
          Arg("o", OUT, "float32", (OH * OW,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
