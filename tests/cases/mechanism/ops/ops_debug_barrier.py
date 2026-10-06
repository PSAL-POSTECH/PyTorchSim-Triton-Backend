"""`tl.debug_barrier()` between a load and a store of the same address.

WHAT IT PINS. `pipeline._drop_noops`, its `ttg.barrier` half. Without it stage 3 does
not run at all -- triton-shared has never heard of the `ttg` dialect, so the
parse dies:

    02-ttclean.mlir:78:5: error: Dialect `ttg' not found for custom op
    'ttg.barrier'

WHERE IT CAME FROM. Not written for this file -- found running PyTorchSim's
ResNet18 test through the Triton codegen route. Inductor emits
`tl.debug_barrier()` whenever a kernel loads from an `in_out_ptr` and stores
back to it, which is every in-place accumulation, and ResNet's residual add is
one:

    triton_npu_fused__native_batch_norm_legit_no_training_add_relu_3

This kernel is that shape and nothing else: read the accumulator, add, relu,
write it back to the same index, with the barrier where Inductor puts it.

WHY DROPPING IT IS SOUND, and it is about the machine rather than about the op.
A barrier orders thread blocks against each other. Here one kernel instance runs
over one tile and a lane reads and writes ITS OWN element, so program order is
the whole of the ordering and the barrier has no second party. That argument
fails the day a tile is split across units that can observe each other's stores,
at which point `ttg.barrier` leaves `NOOPS` for a pass that REFUSES it,
and this kernel is the one that has to start refusing rather than passing.

WHAT WOULD BREAK IT. Making the store index differ from the load index, which
turns the in-place update into a shuffle and makes the ordering real. That is a
different kernel and it does not belong here; this one is the shape Inductor
actually emits.

A GPU barrier orders thread blocks, and this machine has none, so it is taken out above triton-shared -- which cannot parse the dialect it is written in.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, INOUT

M, N = 8, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_debug_barrier_kernel(io_ptr, b_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        off = rm[:, None] * N + rn[None, :]
        acc = tl.load(io_ptr + off)          # the in_out_ptr, read
        b = tl.load(b_ptr + off)
        out = tl.maximum(acc + b, 0.0)
        tl.debug_barrier()                    # exactly where Inductor puts it
        tl.store(io_ptr + off, out)           # ... and written back, same index

    return ops_debug_barrier_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"io": torch.randn(M * N, generator=g, dtype=torch.float32),
            "b": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"io": (inputs["io"] + inputs["b"]).clamp(min=0.0)}


SPEC = KernelSpec(
    name="ops_debug_barrier_kernel", kernel=kernel,
    signature={"io_ptr": "*fp32", "b_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("io", INOUT, "float32", (M * N,)),
          Arg("b", IN, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-6, atol=1e-6,
)
