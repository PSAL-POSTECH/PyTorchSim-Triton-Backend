"""A prefix scan exactly the lanes long. Its ladder pads the tile by half its length to shift
along, so the padded tile is one and a half lane-fulls: lower_copies folds it onto the lanes the
machine has, the second half-lane-full one row deeper, and each shift is a permute whose two
rounds write the two rows. Breaks if a lane axis longer than the machine has to be a whole
number of lane-fulls (the fold refuses it, the copy has no carrier), or if the bank reserves the
tile's elements over the lanes instead of a whole row per lane-full (the deeper row overruns it).

The shape every scan of the lane count takes: on 256 lanes a 256-wide scan, on 32 a 32-wide
one (GPT-2's position cumsum at a 32-lane target).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_scan_as_long_as_the_machine_kernel(x_ptr, o_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        tl.store(o_ptr + r, tl.cumsum(tl.load(x_ptr + r), axis=0))

    return ops_scan_as_long_as_the_machine_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].double().cumsum(0).float()}


SPEC = KernelSpec(
    name="ops_scan_as_long_as_the_machine_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
)
