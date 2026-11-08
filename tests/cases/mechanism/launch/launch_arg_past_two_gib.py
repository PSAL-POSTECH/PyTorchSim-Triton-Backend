"""An input of 2 GiB and 256 KiB, as Llama 4 Scout's 16-expert gate_up stack (2.5 GiB bf16) is: newlib's
calloc answers NULL for 2 GiB or more at once and the wrapper returned 1 before the first program. Program 3
reads a row past the 2 GiB mark. Breaks if the wrapper takes a tensor buffer from newlib's heap."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, K, P = (1 << 19) + 64, 1024, 174763, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def launch_arg_past_two_gib_kernel(w_ptr, o_ptr, C: tl.constexpr, K: tl.constexpr):
        pid = tl.program_id(0)
        c = tl.arange(0, C)
        tl.store(o_ptr + pid * C + c, tl.load(w_ptr + (pid * K) * C + c) * 2.0)

    return launch_arg_past_two_gib_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.zeros(R, C, dtype=torch.float32)
    for p in range(P):
        w[p * K] = torch.randn(C, generator=g)
    return {"w": w}


def reference(inputs):
    w = inputs["w"]
    return {"o": w[[p * K for p in range(P)]] * 2.0}


SPEC = KernelSpec(
    name="launch_arg_past_two_gib_kernel",
    kernel=kernel,
    signature={"w_ptr": "*fp32", "o_ptr": "*fp32", "C": "constexpr", "K": "constexpr"},
    constexprs={"C": C, "K": K},
    args=[Arg("w", IN, "float32", (R, C)), Arg("o", OUT, "float32", (P, C))],
    grid=(P,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-6, atol=1e-6,
)
