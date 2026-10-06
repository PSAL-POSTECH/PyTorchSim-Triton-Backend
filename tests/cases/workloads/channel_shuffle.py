"""Channel shuffle -- the one operation ShuffleNet is named after.

    view [C, HW] as [G, C/G, HW]  ->  transpose the two channel axes  ->  flatten

CHANNELS THAT NEVER MEET, AND THE CHEAPEST WAY TO MAKE THEM. A grouped 1x1
convolution costs `1/G` of a dense one, but the groups are then sealed off: group
0's outputs are built only from group 0's inputs, forever. Stack such layers and
the network is G independent networks sharing a tensor. Shuffle is the fix, and
it is not a convolution at all -- it is a permutation, chosen so that after it
each group's slots are filled one from every group.

    G = 2, C = 8      in   0 1 2 3 | 4 5 6 7
                      out  0 4 1 5 | 2 6 3 7

The permutation is exactly a transpose of the factored axis, which is why it is
written as reshape/transpose/reshape rather than as an index table: the two forms
are the same map, and only this one says WHY it is that map.

    out[i*G + k]  =  in[k*(C/G) + i]

SHUFFLENET V2 KEEPS THE SHUFFLE AND DROPS THE GROUPS. The V2 paper's measurements
say group convolution raises memory traffic per FLOP (its guideline G2), so the
unit splits the channels in two instead: half goes through 1x1 -> dw3x3 -> 1x1
untouched by groups, half is carried straight across, the two are CONCATENATED
rather than added, and the shuffle then mixes the halves for the next block. So
the op survives its original reason and is now what makes a channel SPLIT safe.

WHAT IT ASKS OF THIS MACHINE. With the channels on the lanes the permutation is a
lane-to-lane move, which the hardware does not have -- lane L reads only bank L.
So it has to become a DRAM round trip, and the interesting question is whether the
backend sees that from the reshape/transpose/reshape spelling. With the channels
in-lane instead it is a permutation inside one bank and costs nothing.

torch equivalent: x.reshape(G, C // G, HW).transpose(0, 1).reshape(C, HW)

ShuffleNetV2: the channel permutation, written as its factored transpose. A lane-to-lane move if the channels are on the lanes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, HW, G = 32, 64, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def channel_shuffle_kernel(x_ptr, o_ptr,
                               C: tl.constexpr, HW: tl.constexpr,
                               G: tl.constexpr):
        c = tl.arange(0, C)
        p = tl.arange(0, HW)
        x = tl.load(x_ptr + c[:, None] * HW + p[None, :])
        y = tl.reshape(x, (G, C // G, HW))
        y = tl.trans(y, 1, 0, 2)
        y = tl.reshape(y, (C, HW))
        tl.store(o_ptr + c[:, None] * HW + p[None, :], y)

    return channel_shuffle_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C * HW, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(G, C // G, HW).transpose(0, 1).reshape(C * HW)
    return {"o": x.contiguous()}


SPEC = KernelSpec(
    name="channel_shuffle_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "C": "constexpr",
               "HW": "constexpr", "G": "constexpr"},
    constexprs={"C": C, "HW": HW, "G": G},
    args=[Arg("x", IN, "float32", (C * HW,)),
          Arg("o", OUT, "float32", (C * HW,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
