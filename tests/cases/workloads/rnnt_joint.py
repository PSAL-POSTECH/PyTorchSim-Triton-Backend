"""The RNN-T joint network: an outer SUM over two sequence axes, then a
projection to the vocabulary.

    z[t, u, h]      = f[t, h] + g[u, h]          f.unsqueeze(2) + g.unsqueeze(1)
    logits[t, u, v] = sum_h tanh(z) * W[v, h] + b[v]

`f` is the acoustic encoder's output at frame `t`; `g` is the prediction
network's output after `u` emitted labels. NeMo's `RNNTJoint.joint_after_projection`
is those two lines and nothing else. The transducer's whole trick is that the
two networks never see each other until here -- the audio side does not know
what has been emitted and the label side does not know what has been heard, so
each runs once and this kernel builds every combination.

TWO INDEPENDENT SEQUENCE AXES AT ONCE, WHICH NOTHING ELSE HERE HAS. Attention
also relates two sequences, but it CONTRACTS them into a `[S, S]` score -- one
number per pair. This does the opposite: every pair keeps a whole H-vector, so
the result is rank 3 and the tensor is `[T, U, H]` before the projection and
`[T, U, V+1]` after it. `alibi.py` builds a `[S, S]` matrix from two index axes
and stops there; here a feature axis survives underneath both of them.

  THIS IS THE FAMOUS MEMORY PROBLEM. At B=32, T=1000 encoder frames, U=100
  labels and a 1024-token vocabulary the logits are 3.3e9 floats, 13 GB in fp32,
  for one training step -- which is why every RNN-T implementation ships a fused
  loss, a batch-splitting joint, or both. Inference is spared only because
  greedy decoding walks the lattice one (t, u) at a time. The lattice IS the op,
  so it is materialised here: `z` is an output, at `[T, U, H]`.

  THE RANK-3 TILE IS NOT ATTEMPTED. `f[:, None, :] + g[None, :, :]` is a
  broadcast on a DIFFERENT axis for each operand, and the suite has already
  measured what that costs: `moe_router.py` hit `vectorize` refusing a
  rank-3 outer product, and `alibi.py` loops its head axis for the same reason.
  So `U` is a `tl.static_range` and every tile stays rank 2: `f` is loop
  invariant, `g[u]` is a rank-1 slice broadcast along the lane axis, and the
  lattice appears one `[T, H]` plane at a time. The rank-3 shape survives only
  in the STORE'S ADDRESSING, `t * (U*H) + u * H + h`, where the `u` stride is
  what makes the plane a slice of a 3-D tensor instead of a matrix.

THE PROJECTION IS INSIDE THE LOOP AND THE WEIGHT IS NOT. `W` is `[V+1, H]`,
loaded once, transposed by its addressing, and used by all U matmuls -- so the
loop reuses one operand and rebuilds the other, which is the reuse pattern a
GEMM's L2 blocking is normally chosen for. Here it falls out of the shape.

  V+1 AND NOT V: the extra symbol is the BLANK, the one that advances `t`
  without advancing `u`. It is the reason the lattice is 2-D at all, and it is
  usually index 0 (torchaudio) or the last index (NeMo). Nothing in this kernel
  treats it specially -- it is a column of `W` like any other -- which is
  precisely why it is worth saying: the structure that makes a transducer a
  transducer lives entirely in the loss and the decoder, not in the tensor.

tanh IS 2 sigmoid(2x) - 1, EXACTLY, and this triton has no tanh -- neither
`tl.tanh` nor `tl.math.tanh` exists in 3.6. The identity is algebraically exact
and numerically is not quite: it costs one extra rounding, measured at 1.8e-7
worst case against `torch.tanh` over 10k fp32 samples, which is the floor on
this kernel's agreement with the reference. Graves (2012) and
torchaudio use tanh here; NeMo's config defaults to relu, which would make this
line free and the kernel less interesting.

An outer sum across two sequence axes with a feature axis surviving underneath both, kept rank 2 by looping U and putting the third axis in the store's stride. `z` is exact (one add), so any error there is addressing and any error in `o` alone is the tanh identity or the matmul.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T = 16       # encoder frames
U = 4        # label positions (+1 for the start state, folded in)
H = 32       # joint hidden
V = 32       # vocabulary INCLUDING the blank


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rnnt_joint_kernel(f_ptr, g_ptr, w_ptr, b_ptr, z_ptr, o_ptr,
                          T: tl.constexpr, U: tl.constexpr,
                          H: tl.constexpr, V: tl.constexpr):
        t = tl.arange(0, T)[:, None]
        h = tl.arange(0, H)[None, :]
        ht = tl.arange(0, H)[:, None]
        v = tl.arange(0, V)[None, :]

        f = tl.load(f_ptr + t * H + h)                   # [T, H], loop invariant
        wt = tl.load(w_ptr + v * H + ht)                 # [H, V], read transposed
        bias = tl.load(b_ptr + tl.arange(0, V))

        for u in tl.static_range(U):
            # THE OUTER SUM, one label position at a time: a rank-1 vector
            # broadcast down the lane axis of a [T, H] tile.
            g = tl.load(g_ptr + u * H + tl.arange(0, H))
            z = f + g[None, :]
            # The u stride is the only trace of the third axis.
            tl.store(z_ptr + t * (U * H) + u * H + h, z)

            a = 2.0 * tl.sigmoid(2.0 * z) - 1.0          # tanh, written out
            tl.store(o_ptr + t * (U * V) + u * V + v,
                     tl.dot(a, wt) + bias[None, :])

    return rnnt_joint_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)
    return {"f": r(T * H) * 0.5, "g": r(U * H) * 0.5,
            "w": r(V * H) * (H ** -0.5), "b": r(V) * 0.1}


def reference(inputs):
    import torch
    f = inputs["f"].reshape(T, H)
    g = inputs["g"].reshape(U, H)
    z = f[:, None, :] + g[None, :, :]                    # [T, U, H]
    logits = z.tanh() @ inputs["w"].reshape(V, H).T + inputs["b"]
    return {"z": z.reshape(-1).contiguous(),
            "o": logits.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="rnnt_joint_kernel", kernel=kernel,
    signature={"f_ptr": "*fp32", "g_ptr": "*fp32", "w_ptr": "*fp32",
               "b_ptr": "*fp32", "z_ptr": "*fp32", "o_ptr": "*fp32",
               "T": "constexpr", "U": "constexpr", "H": "constexpr",
               "V": "constexpr"},
    constexprs={"T": T, "U": U, "H": H, "V": V},
    args=[Arg("f", IN, "float32", (T * H,)),
          Arg("g", IN, "float32", (U * H,)),
          Arg("w", IN, "float32", (V * H,)),
          Arg("b", IN, "float32", (V,)),
          Arg("z", OUT, "float32", (T * U * H,)),
          Arg("o", OUT, "float32", (T * U * V,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect="blocked",
    expect_why="matmul K=32 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
