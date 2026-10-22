"""An atomic, refused by decision: tl.atomic_add through gathered indices (the tt.atomic_rmw fadd
Inductor emits for index_add_) and tl.atomic_cas (tt.atomic_cas). The harness does not match the
message, so read it in the run's output: "atomics are refused on this backend", at triton-compile."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, INOUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_atomics_refused(idx_ptr, src_ptr, out_ptr, lock_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(idx_ptr + r)
        tl.atomic_add(out_ptr + i, tl.load(src_ptr + r), sem="relaxed")
        tl.atomic_cas(lock_ptr, 0, 1)

    return ops_atomics_refused


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"idx": torch.randint(0, N, (N,), generator=g, dtype=torch.int32),
            "src": torch.randn(N, generator=g, dtype=torch.float32),
            "out": torch.zeros(N, dtype=torch.float32),
            "lock": torch.zeros(1, dtype=torch.int32)}


def reference(inputs):
    return {"out": inputs["out"].index_add(0, inputs["idx"].long(), inputs["src"]),
            "lock": inputs["lock"] + 1}


SPEC = KernelSpec(
    name="ops_atomics_refused", kernel=kernel,
    signature={"idx_ptr": "*i32", "src_ptr": "*fp32", "out_ptr": "*fp32", "lock_ptr": "*i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("idx", IN, "int32", (N,)), Arg("src", IN, "float32", (N,)),
          Arg("out", INOUT, "float32", (N,)), Arg("lock", INOUT, "int32", (1,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="refuse",
    expect_stop=Stop("triton-compile", Reason.ATOMICS),
    expect_why="atomics are not on this backend's path, and that is a decision rather than a "
               "lowering still missing: an atomic float add is nondeterministic in order, and "
               "a destination written more than once is accumulated without one "
               "(tests/cases/workloads/embedding_grad_scatter_add.py)",
)
