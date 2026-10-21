"""Rows read in runs of K by `row // K`, beside a reduce over each row: held K to a lane, the run's
table row is the lane's own, and nothing crosses the lanes.

WHAT IT IS. Qwen3's q_norm + rope + transpose fused into one kernel: rows are (token, head), the
norm reduces each row's 128 columns, and rope reads the token's cos row at `row // H`. With the
rows across the lanes one each (the reduce's anchor), the H rows of a token sat in H lanes, the
table was loaded into the few lanes its row index fills and spread over the rest through the XLU
-- 150k cycles a tile in the layer. anchorReduce now holds a lane's rows as a run when every
address quotient on the row axis divides the rows a lane holds (rowsPerLane).

Here R = 512 rows on 128 lanes, K = 4: four rows to a lane, one table row each.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, D, K = 512, 128, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_row_runs_read_one_table_row_stay_in_a_lane_kernel(x_ptr, c_ptr, o_ptr, R: tl.constexpr,
                                                              D: tl.constexpr, K: tl.constexpr):
        r = tl.arange(0, R)[:, None]
        d = tl.arange(0, D)[None, :]
        x = tl.load(x_ptr + (r * D + d))
        s = tl.sum(x * x, 1)[:, None] / D
        c = tl.load(c_ptr + ((r // K) * D + d))
        tl.store(o_ptr + (r * D + d), x / tl.sqrt(s + 1e-6) * c)

    return ops_row_runs_read_one_table_row_stay_in_a_lane_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * D, generator=g, dtype=torch.float32),
            "c": torch.randn(R // K * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(R, D)
    c = inputs["c"].reshape(R // K, D).repeat_interleave(K, 0)
    s = (x * x).sum(1, keepdim=True) / D
    return {"o": (x / torch.sqrt(s + 1e-6) * c).reshape(-1)}


SPEC = KernelSpec(
    name="ops_row_runs_read_one_table_row_stay_in_a_lane_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "c_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr",
               "D": "constexpr", "K": "constexpr"},
    constexprs={"R": R, "D": D, "K": K},
    args=[Arg("x", IN, "float32", (R * D,)), Arg("c", IN, "float32", (R // K * D,)),
          Arg("o", OUT, "float32", (R * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-transformed.mlir": {"torchsim.crosslane_move": 0}},
)
