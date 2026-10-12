"""A load inside two counted loops whose tile is first read after both. A whole transfer is synchronous,
so no wait stands there; made async, its one wait must cover every trip, G * H tiles, or togcheck
fails with "sem 0 signaled 4096 elements, wait expects 512"."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, G, H = 8, 64, 2, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def load_waited_after_loop_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr,
                                      G: tl.constexpr, H: tl.constexpr):
        i = tl.arange(0, R)
        j = tl.arange(0, C)
        x = tl.zeros((R, C), dtype=tl.float32)
        for g in range(G):
            for h in range(H):
                x = tl.load(x_ptr + (g * H + h) * R * C + i[:, None] * C + j[None, :])
        tl.store(o_ptr + i[:, None] * C + j[None, :], x * 2.0)

    return load_waited_after_loop_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(G * H * R * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].reshape(G * H, R * C)[G * H - 1] * 2.0}


SPEC = KernelSpec(
    name="load_waited_after_loop_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "C": "constexpr",
               "G": "constexpr", "H": "constexpr"},
    constexprs={"R": R, "C": C, "G": G, "H": H},
    args=[Arg("x", IN, "float32", (G * H * R * C,)),
          Arg("o", OUT, "float32", (R * C,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
