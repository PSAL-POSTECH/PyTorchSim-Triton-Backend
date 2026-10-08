"""A static loop whose second trip waits for the first trip's store before it overwrites the tile:
a wait after a store, outside the loop that issued it. TOGSim aborted on it ("Key does not exist in
subgraph's tag table") while the store's wait reached the trace as a barrier no load signalled."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, H = 8, 64, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def store_waited_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr, H: tl.constexpr):
        i = tl.arange(0, R)
        j = tl.arange(0, C)
        for h in tl.static_range(H):
            x = tl.load(x_ptr + h * R * C + i[:, None] * C + j[None, :])
            tl.store(o_ptr + h * R * C + i[:, None] * C + j[None, :], x * 2.0)

    return store_waited_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(H * R * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"] * 2.0}


SPEC = KernelSpec(
    name="store_waited_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "C": "constexpr",
               "H": "constexpr"},
    constexprs={"R": R, "C": C, "H": H},
    args=[Arg("x", IN, "float32", (H * R * C,)),
          Arg("o", OUT, "float32", (H * R * C,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
