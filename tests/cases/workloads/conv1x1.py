"""The 1x1 convolution -- which is a matmul, and 33 of ResNet-50's 49 convs.

With no spatial extent, output channel `o` at position `p` is
`sum_i w[o, i] * x[i, p]`: a matmul `[Co, Ci] x [Ci, HW]`. Writing it as `tl.dot`
is not an optimisation, it is what the op is. It goes to the systolic array,
while the 3x3 beside it (`conv_channels.py`) goes to the vector lanes -- and
that line is the whole dispatch this backend makes.

  WHERE THE PARAMETERS ARE. layer4's 1x1s are 512x2048 and 2048x512; the 3x3
  between them is 512x512x9, which per layer looks bigger. But there are twice
  as many 1x1s and they carry the channel growth, which is how ResNet-50 puts
  59% of its 25.5M parameters in its last stage while most of its FLOPs are in
  the first.

IT NEEDS THE CHANNEL LOOP, and that is not a stylistic choice. lower_to_vcix
marks an `scf.for` around the dot as the accumulation loop and drives the
array's push/compute/pop from it; a single `tl.dot` with no loop is declined and
survives to stage 4, where pipeline.py catches it ("a linalg.matmul survived
lower_to_vcix"). gemm.py says the same thing about itself. So Ci is walked in
BK-sized chunks here -- which is what a real 1x1 does anyway, since a bottleneck's
Ci runs to 2048 and no tile holds that.

torch equivalent: (w @ x), i.e. F.conv2d with a 1x1 kernel.

A 1x1 conv is a matmul over channels, and the channel loop is what lower_to_vcix marks as the accumulation loop -- a loopless tl.dot is declined and caught at stage 4.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

CO, CI, HW = 16, 32, 32
BK = 16                              # channel chunk per array pass


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def conv1x1_kernel(w_ptr, x_ptr, o_ptr, CO: tl.constexpr, CI: tl.constexpr,
                       HW: tl.constexpr, BK: tl.constexpr):
        co = tl.arange(0, CO)
        hw = tl.arange(0, HW)
        k = tl.arange(0, BK)
        acc = tl.zeros((CO, HW), dtype=tl.float32)
        for c0 in range(0, CI, BK):
            w = tl.load(w_ptr + co[:, None] * CI + (c0 + k)[None, :])
            x = tl.load(x_ptr + (c0 + k)[:, None] * HW + hw[None, :])
            acc += tl.dot(w, x)
        tl.store(o_ptr + co[:, None] * HW + hw[None, :], acc)

    return conv1x1_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(CO * CI, generator=g, dtype=torch.float32),
            "x": torch.randn(CI * HW, generator=g, dtype=torch.float32)}


def reference(inputs):
    w = inputs["w"].reshape(CO, CI)
    x = inputs["x"].reshape(CI, HW)
    return {"o": (w @ x).reshape(-1)}


SPEC = KernelSpec(
    name="conv1x1_kernel", kernel=kernel,
    signature={"w_ptr": "*fp32", "x_ptr": "*fp32", "o_ptr": "*fp32",
               "CO": "constexpr", "CI": "constexpr", "HW": "constexpr",
               "BK": "constexpr"},
    constexprs={"CO": CO, "CI": CI, "HW": HW, "BK": BK},
    args=[Arg("w", IN, "float32", (CO * CI,)), Arg("x", IN, "float32", (CI * HW,)),
          Arg("o", OUT, "float32", (CO * HW,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
