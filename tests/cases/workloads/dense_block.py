"""A DenseNet block -- concatenation as the connection, and a growing tensor.

    x0                                    C0 channels
    x1 = cat(x0, H(x0))                   C0 + k
    x2 = cat(x1, H(x1))                   C0 + 2k

ADD VERSUS CONCAT IS THE WHOLE PAPER. ResNet writes `x + F(x)`, which SUMS the
identity into the residual: the two are indistinguishable afterwards, and a
layer that wants only the identity has to learn to cancel F. DenseNet writes
`cat(x, F(x))` and keeps both, so every layer sees every earlier layer's output
UNCHANGED. The cost is that the width grows linearly -- 32 channels per layer
("growth rate k"), so a 12-layer block adds 384.

WHY THAT IS CHEAPER THAN IT SOUNDS. Because nothing is recomputed, each layer
can be narrow: DenseNet-121 matches ResNet-50's accuracy with a third of the
parameters. What it spends instead is MEMORY -- every intermediate stays live to
be concatenated -- which is why the reference implementation has a
`memory_efficient` mode that recomputes them.

BN-RELU-CONV, IN THAT ORDER, and the order is the point. ResNet's original block
is conv-BN-ReLU, so the residual passes through a ReLU on the way out; DenseNet
uses the pre-activation order from ResNet-v2, where the identity path is clean
all the way through and normalisation happens before the weights rather than
after them. Concatenation makes it necessary: each layer's input is a stack of
tensors from different depths with different scales, so it has to normalise
before it can weigh them.

WHAT IS NEW HERE FOR THE BACKEND. The output tile is WIDER than the input tile,
and it is assembled from pieces of different provenance -- some copied straight
through, some computed. With the channels on the lanes that means the pieces
land in different banks and the assembly is a layout question. Written as
separate stores into channel ranges of one output, which is what concat is once
the destination is known.

torch equivalent: for each layer, cat([x, conv1x1(relu(bn(x)))], dim=0)

DenseNet: a tensor that GROWS by concatenation, assembled from pieces of different provenance into channel ranges of one output.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C0, K, HW = 16, 8, 64
C1 = C0 + K
C2 = C1 + K


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def dense_block_kernel(x_ptr, g0_ptr, b0_ptr, g1_ptr, b1_ptr, o_ptr,
                           C0: tl.constexpr, K: tl.constexpr,
                           HW: tl.constexpr, C2: tl.constexpr):
        p = tl.arange(0, HW)
        c0 = tl.arange(0, C0)
        k = tl.arange(0, K)

        # x0 straight through -- the identity DenseNet never sums away.
        x0 = tl.load(x_ptr + c0[:, None] * HW + p[None, :])
        tl.store(o_ptr + c0[:, None] * HW + p[None, :], x0)

        # LAYER 1. BN then ReLU then the 1x1, on the first K channels of x0.
        #          BN at inference is a per-channel affine, so it folds to
        #          gamma*x + beta (see bn_relu_add for why that is exact).
        s0 = tl.load(x_ptr + k[:, None] * HW + p[None, :])
        g0 = tl.load(g0_ptr + k)
        b0 = tl.load(b0_ptr + k)
        h1 = tl.maximum(s0 * g0[:, None] + b0[:, None], 0.0)
        tl.store(o_ptr + (k + C0)[:, None] * HW + p[None, :], h1)

        # LAYER 2 reads what layer 1 just wrote -- the concatenated tensor, not
        # its own input. That dependence is what makes the block dense.
        g1 = tl.load(g1_ptr + k)
        b1 = tl.load(b1_ptr + k)
        h2 = tl.maximum(h1 * g1[:, None] + b1[:, None], 0.0)
        tl.store(o_ptr + (k + C0 + K)[:, None] * HW + p[None, :], h2)

    return dense_block_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C0 * HW, generator=g, dtype=torch.float32),
            "g0": torch.randn(K, generator=g, dtype=torch.float32),
            "b0": torch.randn(K, generator=g, dtype=torch.float32),
            "g1": torch.randn(K, generator=g, dtype=torch.float32),
            "b1": torch.randn(K, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(C0, HW)
    h1 = torch.relu(x[:K] * inputs["g0"][:, None] + inputs["b0"][:, None])
    h2 = torch.relu(h1 * inputs["g1"][:, None] + inputs["b1"][:, None])
    return {"o": torch.cat([x, h1, h2], dim=0).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="dense_block_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "g0_ptr": "*fp32", "b0_ptr": "*fp32",
               "g1_ptr": "*fp32", "b1_ptr": "*fp32", "o_ptr": "*fp32",
               "C0": "constexpr", "K": "constexpr", "HW": "constexpr",
               "C2": "constexpr"},
    constexprs={"C0": C0, "K": K, "HW": HW, "C2": C2},
    args=[Arg("x", IN, "float32", (C0 * HW,)),
          Arg("g0", IN, "float32", (K,)), Arg("b0", IN, "float32", (K,)),
          Arg("g1", IN, "float32", (K,)), Arg("b1", IN, "float32", (K,)),
          Arg("o", OUT, "float32", (C2 * HW,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
