"""A tile WIDER than the lane count, so a lane owns more than one element.

Every other kernel in this suite has a block of 32, 64 or 128 -- measured, those
are the only three values in kernels/coverage -- so on a 128-lane machine the
split axis never carries more than one element per lane. That is not a property
of the backend, it is a property of what has been written down, and it means the
suite cannot answer whether a deeper tile works.

WHY IT MATTERS BEYOND THIS FILE. The Triton route pins XBLOCK to the lane count
in PyTorchSim's kernel_spec.fixed_config_for, and the conv heuristic pins
BLOCK_N there too. Both cite the same reason -- one element per lane is the
shape this backend is known to run. If a deeper tile is fine, those pins are
policy that can be lifted and the frontend stops needing to know the lane count
at all; if it is not, they are load-bearing and should say so.

The block is 256 on a 128-lane machine: two elements per lane, on axis 0, which
is the axis nothing has to negotiate for (fit_layout_to_lanes gives dim 0 when no
op demands otherwise). So this varies ONE thing against the kernels that already
pass -- the depth -- and holds the axis, the rank and the op fixed.

WHAT BREAKS IT: anything that assumes the split axis has exactly `lanes`
elements. vectorize's addressing is the candidate -- a lane addresses its
own tile, and when the extent exceeds the lane count the per-lane view is no
longer the memref type. Whether it actually breaks is what this kernel is for.

expect="pass" is a CLAIM, not an observation, until it runs: two elements per
lane is arithmetic the machine can do, so a refusal here is a defect rather
than a limit. If it refuses, change this to "blocked" with the message rather
than shrinking the block.

Block 256 on 128 lanes: two elements per lane on axis 0. The only kernel here whose tile is wider than the lane count.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
BLOCK = 256          # 2x the 128 lanes


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_deeper_than_one_per_lane_kernel(x_ptr, y_ptr, o_ptr,
                                             BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        a = tl.load(x_ptr + r)
        b = tl.load(y_ptr + r)
        tl.store(o_ptr + r, a * b + a)

    return tile_deeper_than_one_per_lane_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "y": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x, y = inputs["x"], inputs["y"]
    return {"o": x * y + x}


SPEC = KernelSpec(
    name="tile_deeper_than_one_per_lane_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
