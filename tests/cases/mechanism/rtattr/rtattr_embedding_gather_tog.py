"""Pins a gather as an index snapshot: an embedding lookup reads each program's token ids from DRAM and
gathers those rows of the table, so the run records the index tile and the TOG moves each row by
scale x the recorded id, as the model's own index dumps say. Breaks if the TOG's gather reads no record."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, B, D, V = 4, 8, 64, 32
IDS = [(5 * k + 3) % V for k in range(S * B)]


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_embedding_gather_tog(w_ptr, ids_ptr, o_ptr, B: tl.constexpr, D: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, B)
        d = tl.arange(0, D)
        ids = tl.load(ids_ptr + pid * B + r)
        rows = tl.load(w_ptr + ids[:, None] * D + d[None, :])
        tl.store(o_ptr + (pid * B + r)[:, None] * D + d[None, :], rows)

    return rtattr_embedding_gather_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(V, D, generator=g), "ids": torch.tensor(IDS, dtype=torch.int32)}


def reference(inputs):
    return {"o": inputs["w"][[int(i) for i in IDS]]}


SPEC = KernelSpec(
    name="rtattr_embedding_gather_tog", kernel=kernel,
    signature={"w_ptr": "*fp32", "ids_ptr": "*i32", "o_ptr": "*fp32", "B": "constexpr", "D": "constexpr"},
    constexprs={"B": B, "D": D},
    args=[Arg("w", IN, "float32", (V, D)), Arg("ids", IN, "int32", (S * B,)), Arg("o", OUT, "float32", (S * B, D))],
    expect_ir={"09-custom.mlir": {"torchsim.runtime_attribute": 1}},
    grid=(S,), reference=reference, make_inputs=make_inputs, rtol=1e-6, atol=1e-6,
)
