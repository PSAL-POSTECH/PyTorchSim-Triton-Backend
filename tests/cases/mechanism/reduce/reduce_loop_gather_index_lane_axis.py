"""A reduction loop whose body holds a GATHER indexed along the reduced axis."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X = 128             # the row BLOCK -- the axis that must stay on the lanes
XLIVE = 32          # < X, so the mask is real and the gather's tile is 2-D
R = 1024            # the reduced axis
R_BLOCK = 256       # so the loop really has a back edge (4 trips)
K = 128             # < R_BLOCK, so `r // K` is non-affine inside one block
NB = R // K         # the gathered table


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_loop_gather_index_lane_axis_kernel(
            a_ptr, b_ptr, o_ptr, X: tl.constexpr, XLIVE: tl.constexpr,
            R: tl.constexpr, R_BLOCK: tl.constexpr, K: tl.constexpr):
        x = tl.arange(0, X)[:, None]
        xm = x < XLIVE
        acc = tl.zeros([X, R_BLOCK], tl.float32)
        for r_off in range(0, R, R_BLOCK):
            r = r_off + tl.arange(0, R_BLOCK)[None, :]
            m = xm & (r < R)
            # DIRECT: this is where the row dependence comes from.
            a = tl.load(a_ptr + (r + R * x), m, other=0.0)
            # GATHER: a non-affine index built from the REDUCED axis alone. The
            # 2-D mask is what makes the gathered TILE 2-D (128 x R_BLOCK)
            # rather than one broadcast row -- see the note below.
            b = tl.load(b_ptr + (r // K), m, other=0.0)
            acc += a + b
        xi = tl.arange(0, X)
        tl.store(o_ptr + xi, tl.sum(acc, 1), xi < XLIVE)

    return reduce_loop_gather_index_lane_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(XLIVE, R, generator=g, dtype=torch.float32),
            "b": torch.randn(NB, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(R)
    return {"o": (inputs["a"] + inputs["b"][r // K]).sum(dim=1)}


SPEC = KernelSpec(
    name="reduce_loop_gather_index_lane_axis_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "X": "constexpr", "XLIVE": "constexpr", "R": "constexpr",
               "R_BLOCK": "constexpr", "K": "constexpr"},
    constexprs={"X": X, "XLIVE": XLIVE, "R": R, "R_BLOCK": R_BLOCK, "K": K},
    args=[Arg("a", IN, "float32", (XLIVE, R)),
          Arg("b", IN, "float32", (NB,)),
          Arg("o", OUT, "float32", (XLIVE,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
