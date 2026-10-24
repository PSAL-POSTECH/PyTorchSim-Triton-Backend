"""linalg.index on the lane dim when a lane holds two rows (512 over 256 lanes) is lane + slot * lanes;
breaks if index_value drops the slot term."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 512, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def lane_index_two_rows_per_lane(x_ptr, z_ptr, y_ptr, w_ptr, R: tl.constexpr, C: tl.constexpr):
        r = tl.arange(0, R)
        rc = r[:, None] * C + tl.arange(0, C)[None, :]
        tl.store(y_ptr + rc, tl.load(x_ptr + rc) + r[:, None].to(tl.float32))
        tl.store(w_ptr + r, tl.load(z_ptr + r) + r.to(tl.float32))

    return lane_index_two_rows_per_lane


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R, C, generator=g, dtype=torch.float32),
            "z": torch.randn(R, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(R, dtype=torch.float32)
    return {"y": inputs["x"] + r[:, None], "w": inputs["z"] + r}


SPEC = KernelSpec(
    name="lane_index_two_rows_per_lane", kernel=kernel,
    signature={"x_ptr": "*fp32", "z_ptr": "*fp32", "y_ptr": "*fp32", "w_ptr": "*fp32",
               "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("z", IN, "float32", (R,)),
          Arg("y", OUT, "float32", (R, C)), Arg("w", OUT, "float32", (R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-passes/07-pytorchsim-vectorize.mlir": {"torchsim.vlane_idx": 2, "vector.step": 1}},
)
