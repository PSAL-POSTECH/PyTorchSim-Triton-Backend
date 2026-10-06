"""A MINTER's outs map restated with the value it mints: the arange behind an
expand whose only reader is an indirect scatter, so the readers refuse and the
producer takes it. Breaks it: a structured store, or a reader that takes [N,1].

Pins da20165 (Qwen3.5 / Qwen3.5-MoE / Qwen3-Next linear attention). absorb_layout_ops walks tensor.expand_shape [32] -> [32, 1] BACKWARDS because its only reader is an indirect MVOUT, which has no arm; the producer is a linalg.generic that fills its init (tl.arange: no ins, a body reading linalg.index, an identity outs map), so the plan goes STRAIGHT THROUGH it and must restate that outs map with the retype. Without _states_its_own_axes the map stays (d0)->(d0) over a rank-2 outs and one-shot-bufferize refuses: 'linalg.generic' op expected operand #0 rank (2) to match the result rank of indexing_map (1) -- the model's error byte for byte. Measured: the whole coverage suite never applies that restatement (0 of 368 kernels), which is why 343/367 stayed green while three gate models died.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_absorb_minter_outs_map_kernel(i_ptr, o_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        idx = tl.load(i_ptr + r)
        tl.store(o_ptr + idx[:, None], r[:, None])

    return ops_absorb_minter_outs_map_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"i": torch.randperm(N, generator=g).to(torch.int32)}


def reference(inputs):
    import torch
    o = torch.zeros(N, dtype=torch.int32)
    o[inputs["i"].to(torch.int64)] = torch.arange(N, dtype=torch.int32)
    return {"o": o}


SPEC = KernelSpec(
    name="ops_absorb_minter_outs_map_kernel", kernel=kernel,
    signature={"i_ptr": "*i32", "o_ptr": "*i32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("i", IN, "int32", (N,)),
          Arg("o", OUT, "int32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0, atol=0,
)
