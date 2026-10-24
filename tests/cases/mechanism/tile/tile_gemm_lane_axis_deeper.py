"""A gemm whose LANE axis carries two elements per lane, not one.

The companion to tile_deeper_than_one_per_lane.py, which showed a 256-wide tile
banked on axis 0 comes out exact. Axis 0 is the easy case: it is what
fit_layout_to_lanes hands out when no op demands anything, and the split axis
already has the largest stride, so a lane indexing the full memref happens to
address its own bank correctly.

A MATMUL DOES DEMAND. lib_axis_graph makes it "the last axis on the lanes", and
measured across the pytorchsim-triton-compiler dumps that is what gemm gets -- gemm_fp16_kernel is
stamped pytorchsim.lane_axis = 1 throughout while 26 of the 36 kernels with stamps are
axis 0. So this kernel puts BN at 256 on a 128-lane machine and asks the harder
question: two elements per lane on the axis a matmul chose, where the per-lane
view is NOT the memref type.

That combination does not occur anywhere else in the suite. Measured over
kernels/coverage, every block is 32, 64 or 128 -- there is no tile wider than
the lane count at all, so nothing here has ever addressed a bank holding two.

WHY IT IS WORTH A FILE. PyTorchSim's Triton route pins XBLOCK to the lane count
(kernel_spec.fixed_config_for) and its conv heuristic pins BLOCK_N there too,
both saying one element per lane is the shape this backend runs. If a deeper
tile is fine on a demanded axis as well, those pins are a policy that can be
lifted and the frontend stops needing the lane count; if it is not, they are
load-bearing and this kernel is where that gets said.

WHAT BREAKS IT: any addressing that assumes extent == lanes on the split axis.
The reference formula is in PyTorchSim's mlir_common.py (get_tile_size_per_lane
/ get_tile_stride_per_lane): the per-lane shape divides the split axis by the
lanes used, and every stride LARGER than the split axis's divides by the same
count. For axis 0 nothing is larger, so nothing shrinks; for axis 1 the axis-0
stride does, and a load that skipped that step reads a stride apart what the DMA
laid down contiguously.

BM/BK stay at 128 so the only thing varied against the passing gemm kernels is
BN. expect="pass" is a claim until it runs: two per lane is arithmetic the
machine can do. If it refuses or the numbers move, change this to "blocked",
quote the message, and leave the shape alone.

BN = 256 on 128 lanes, so the axis a matmul demands carries two elements per lane. The only kernel here that is both deeper than one per lane AND on a demanded axis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 128, 256, 128
BM, BN, BK = 128, 256, 128       # BN = 2 x the 128 lanes


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_lane_axis_deeper_kernel(a_ptr, b_ptr, c_ptr,
                                          M: tl.constexpr, N: tl.constexpr,
                                          K: tl.constexpr, BM: tl.constexpr,
                                          BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return tile_gemm_lane_axis_deeper_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g, dtype=torch.float32),
            "b": torch.randn(K, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"], inputs["b"])}


SPEC = KernelSpec(
    name="tile_gemm_lane_axis_deeper_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(M // BM, N // BN),
    reference=reference, make_inputs=make_inputs,
    # Same reason as tests/cases/smoke/gemm.py: the accumulation order is not torch's.
    rtol=1e-3, atol=1e-3,
)
