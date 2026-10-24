"""loop_10_carried_axis_swap under a python `while` -- an scf.while whose body reads the carried tile on one
axis and hands it back on the other."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, ITERS = 16, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_21_while_carried_axis_swap_kernel(a_ptr, w_ptr, o_ptr, iters,
                                               S: tl.constexpr):
        r = tl.arange(0, S)
        acc = tl.load(a_ptr + r[:, None] * S + r[None, :])
        w = tl.load(w_ptr + r[:, None] * S + r[None, :])
        cnt = 0
        while cnt < iters:
            acc = tl.dot(tl.trans(acc, 1, 0), w)
            cnt += 1
        tl.store(o_ptr + r[:, None] * S + r[None, :], acc)

    return loop_21_while_carried_axis_swap_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = 1.0 / (S ** 0.5)
    return {"a": torch.randn(S * S, generator=g, dtype=torch.float32) * s,
            "w": torch.randn(S * S, generator=g, dtype=torch.float32) * s}


def reference(inputs):
    acc = inputs["a"].reshape(S, S)
    w = inputs["w"].reshape(S, S)
    for _ in range(ITERS):
        acc = acc.t() @ w
    return {"o": acc.reshape(-1)}


SPEC = KernelSpec(
    name="loop_21_while_carried_axis_swap_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "iters": "i32", "S": "constexpr"},
    constexprs={"S": S},
    args=[Arg("a", IN, "float32", (S * S,)), Arg("w", IN, "float32", (S * S,)),
          Arg("o", OUT, "float32", (S * S,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    extra={"scalar_args": [("iters", "int32_t")],
           "scalar_values": {"iters": ITERS}},
)
