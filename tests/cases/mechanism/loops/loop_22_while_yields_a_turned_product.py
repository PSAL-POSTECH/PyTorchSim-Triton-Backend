"""An scf.while that starts from a matmul's result and yields one TRANSPOSED --
the init and the yield are each pinned, to different orders.

WHAT IT PINS. torchsim-materialize-layouts's graph holds an scf.while as one
instruction, as upstream holds a kWhile: the first region's arguments and the
second's are its parameters, the state takes the layout of what the body
yields, and the init is copied to it. Without that the loop's spellings keep
the orders their producers gave them and the module goes to the old route.

    before   torchsim-materialize-layouts: "no plan" -- and the old route answers
             with max_abs_err 2.11, the wrong numbers and no refusal
    after    ALL STAGES PASSED, max_abs_err 2.4e-07

torch equivalent: acc = a @ w, then acc = (acc @ w).t(), `iters` times.

An scf.while whose init and yield are pinned to different orders: the state takes the yield's, as upstream holds a kWhile.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, ITERS = 16, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_22_while_yields_a_turned_product_kernel(a_ptr, w_ptr, o_ptr, iters,
                                                     S: tl.constexpr):
        r = tl.arange(0, S)
        a = tl.load(a_ptr + r[:, None] * S + r[None, :])
        w = tl.load(w_ptr + r[:, None] * S + r[None, :])
        acc = tl.dot(a, w)
        cnt = 0
        while cnt < iters:
            acc = tl.trans(tl.dot(acc, w), 1, 0)
            cnt += 1
        tl.store(o_ptr + r[:, None] * S + r[None, :], acc)

    return loop_22_while_yields_a_turned_product_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = 1.0 / (S ** 0.5)
    return {"a": torch.randn(S * S, generator=g, dtype=torch.float32) * s,
            "w": torch.randn(S * S, generator=g, dtype=torch.float32) * s}


def reference(inputs):
    w = inputs["w"].reshape(S, S)
    acc = inputs["a"].reshape(S, S) @ w
    for _ in range(ITERS):
        acc = (acc @ w).t()
    return {"o": acc.reshape(-1)}


SPEC = KernelSpec(
    name="loop_22_while_yields_a_turned_product_kernel", kernel=kernel,
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
