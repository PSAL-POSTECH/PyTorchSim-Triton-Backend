"""`tl.reshape((512,) -> (256, 2))`, the lane axis longer than the machine, then a sum over the pair:
nothing moves -- the reduce's anchor lays the load out with the pair in the bank.
Breaks if the reshape or the reduce is laid out below TritonGPU again: a cross-lane move appears."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, PAIR = 512, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_permute_lane_axis_longer_than_the_machine_kernel(
            x_ptr, s_ptr, N: tl.constexpr, PAIR: tl.constexpr):
        v = tl.load(x_ptr + tl.arange(0, N))
        v = v * v
        m = tl.reshape(v, (N // PAIR, PAIR))
        tl.store(s_ptr + tl.arange(0, N // PAIR), tl.sum(m, axis=1))

    return ops_xlu_permute_lane_axis_longer_than_the_machine_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    v = inputs["x"] * inputs["x"]
    return {"s": v.reshape(N // PAIR, PAIR).sum(dim=1)}


SPEC = KernelSpec(
    name="ops_xlu_permute_lane_axis_longer_than_the_machine_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "s_ptr": "*fp32", "N": "constexpr",
               "PAIR": "constexpr"},
    constexprs={"N": N, "PAIR": PAIR},
    args=[Arg("x", IN, "float32", (N,)),
          Arg("s", OUT, "float32", (N // PAIR,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # NO MOVE AND NO ROUND TRIP.
    expect_ir={"08-transformed.mlir": {"pytorchsim_tpu.xlu_move": 0}},
)
