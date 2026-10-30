"""Eight inputs and an output whose .raw names come to more than pk's 2048-byte argv buffer (a pointer
and the string a word), as qwen3.5's 127-buffer fused copy at S4096 did: Spike stopped with "args must
not exceed 2048 bytes". Breaks if the launch passes the words on the command line past that buffer."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
LONG = "_" + "x" * 236
NAMES = [f"a{i}{LONG}" for i in range(8)] + [f"o{LONG}"]


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def launch_argv_past_pk_buffer_kernel(a0_ptr, a1_ptr, a2_ptr, a3_ptr, a4_ptr, a5_ptr, a6_ptr, a7_ptr, o_ptr,
                                          N: tl.constexpr):
        r = tl.arange(0, N)
        s = (tl.load(a0_ptr + r) + tl.load(a1_ptr + r) + tl.load(a2_ptr + r) + tl.load(a3_ptr + r)
             + tl.load(a4_ptr + r) + tl.load(a5_ptr + r) + tl.load(a6_ptr + r) + tl.load(a7_ptr + r))
        tl.store(o_ptr + r, s)

    return launch_argv_past_pk_buffer_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(N, generator=g, dtype=torch.float32) for n in NAMES[:8]}


def reference(inputs):
    return {NAMES[8]: sum(inputs[n] for n in NAMES[:8])}


SPEC = KernelSpec(
    name="launch_argv_past_pk_buffer_kernel",
    kernel=kernel,
    signature={**{f"a{i}_ptr": "*fp32" for i in range(8)}, "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg(n, IN, "float32", (N,)) for n in NAMES[:8]] + [Arg(NAMES[8], OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
)
