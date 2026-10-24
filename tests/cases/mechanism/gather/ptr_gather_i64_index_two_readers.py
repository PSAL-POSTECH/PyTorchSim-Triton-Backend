"""Two gathers off ONE index tile that is already i64, with tiles written between."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NT = 2048           # the two small tables the SHARED i64 index reads
NX = 8192           # the big buffer the two INTERVENING gathers read
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_i64_index_two_readers_kernel(
            t_ptr, u_ptr, x_ptr, i_ptr, j_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        c = r % 64
        i = tl.load(i_ptr + r)              # i64 in, so the index tile is i64
        idx = c + 64 * i                    # < NT. ONE tile, TWO readers.

        a = tl.load(t_ptr + idx)            # reader 1

        # Two gathers whose index tiles are i32 and range far past NT. They are
        # what gets computed between the two readers.
        j = tl.load(j_ptr + r)
        b = tl.load(x_ptr + j * 3)
        d = tl.load(x_ptr + j * 5)

        e = tl.load(u_ptr + idx)            # reader 2, same idx -> cse merges

        tl.store(o_ptr + r, a * b + d * e)

    return ptr_gather_i64_index_two_readers_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"t": torch.randn(NT, generator=g, dtype=torch.float32),
            "u": torch.randn(NT, generator=g, dtype=torch.float32),
            "x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, NT // 64, (BLOCK,), generator=g,
                               dtype=torch.int64),
            "j": torch.randint(0, NX // 5, (BLOCK,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    t, u, x = inputs["t"], inputs["u"], inputs["x"]
    import torch
    i, j = inputs["i"].long(), inputs["j"].long()
    idx = torch.arange(BLOCK) % 64 + 64 * i
    return {"o": t[idx] * x[j * 3] + x[j * 5] * u[idx]}


SPEC = KernelSpec(
    name="ptr_gather_i64_index_two_readers_kernel", kernel=kernel,
    signature={"t_ptr": "*fp32", "u_ptr": "*fp32", "x_ptr": "*fp32",
               "i_ptr": "*i64", "j_ptr": "*i32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("t", IN, "float32", (NT,)), Arg("u", IN, "float32", (NT,)),
          Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int64", (BLOCK,)),
          Arg("j", IN, "int32", (BLOCK,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # ONE FEWER COPY THAN TRANSFERS. Both numbers, or neither says anything: 4
    # alone allows a shared buffer and 3 alone allows three transfers.
    expect_ir={"08-passes/03-pytorchsim-convert-accesses-to-transfer.mlir": {
        "offset_stride": 4,
        "bufferization.alloc_tensor() copy(": 3}},
)
