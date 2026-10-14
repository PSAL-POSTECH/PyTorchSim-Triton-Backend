"""One buffer read by two matmuls that want it on different axes.

WHAT IT IS FOR. `stuck_groups` reports a group with more than one MUST as "no axis
every matmul can stream", and that is the shape it names -- but nothing in the
suite states it in two lines. The kernels that reach it (retention_chunk,
lightning_attn) reach it through sixteen chained dots, where the conflict is real
but unreadable. This is the same conflict with two dots and one shared load.

THE SHAPE. `k` is loaded once and read twice:

    p = dot(a, k)              k is B  -> matmul_axis(op, 1, MATMUL_N) = axis 1
    q = dot(trans(k), b)       k is A, and absorb_layout_ops writes the transpose
                               into the maps, leaving A as [k, m], so
                               matmul_axis(op, 0, MATMUL_K) = axis 0

So one buffer carries a MUST for axis 1 and a MUST for axis 0. The demands do not
CONTRADICT each other -- each is satisfiable on its own -- so `conflicts()` reports
nothing and only the group is unanswerable. That is why `_plan_cuts`, which
enumerates conflicts, does not see it.

WHAT IT COSTS TODAY, and the two are different failures. Either the pass buys a
relayout and the answer is right at the price of a DRAM round trip, or `axis_of`
falls through to ONE_LANE and the tile sits in ONE bank of 256 -- correct, and
1/256 of the machine, with nothing above the reference check saying a word.

WHY IT IS SEPARATE FROM ops_trans_matmul_maps. There the transposed operand is a
matmul's own result, consumed once; the maps are exercised but no buffer carries
two demands. Here the shared load is what carries them, which is the case a pass
that decides one buffer at a time has to answer.

Two MUSTs on one buffer, which `conflicts()` does not report because neither demand contradicts the other -- only the group is unanswerable. The two ways out are priced differently and only one of them is visible: a relayout costs a round trip, ONE_LANE costs 255 of 256 lanes and says nothing.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def one_buffer_two_lane_axes_kernel(a_ptr, b_ptr, k_ptr, o_ptr,
                                        S: tl.constexpr):
        r = tl.arange(0, S)
        a = tl.load(a_ptr + r[:, None] * S + r[None, :])
        b = tl.load(b_ptr + r[:, None] * S + r[None, :])
        k = tl.load(k_ptr + r[:, None] * S + r[None, :])
        p = tl.dot(a, k)                     # k is B: a column per lane
        q = tl.dot(tl.trans(k, 1, 0), b)     # k is A, and the trans is absorbed
        tl.store(o_ptr + r[:, None] * S + r[None, :], p + q)

    return one_buffer_two_lane_axes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = 1.0 / (S ** 0.5)
    return {"a": torch.randn(S * S, generator=g, dtype=torch.float32) * s,
            "b": torch.randn(S * S, generator=g, dtype=torch.float32) * s,
            "k": torch.randn(S * S, generator=g, dtype=torch.float32) * s}


def reference(inputs):
    a = inputs["a"].reshape(S, S)
    b = inputs["b"].reshape(S, S)
    k = inputs["k"].reshape(S, S)
    return {"o": (a @ k + k.t() @ b).reshape(-1)}


SPEC = KernelSpec(
    name="one_buffer_two_lane_axes_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "k_ptr": "*fp32",
               "o_ptr": "*fp32", "S": "constexpr"},
    constexprs={"S": S},
    args=[Arg("a", IN, "float32", (S * S,)), Arg("b", IN, "float32", (S * S,)),
          Arg("k", IN, "float32", (S * S,)), Arg("o", OUT, "float32", (S * S,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
