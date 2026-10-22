"""Mamba-2 at DECODE -- one timestep of the SSM, against a paged state cache.

For each request b of the batch, at the slot the cache index names:

    h    = cache[slot[b]]                        # [P, N], the whole SSM state
    dt   = softplus(dt_raw[b] + dt_bias)         # per channel, strictly positive
    dA   = exp(dt * A)                           # A < 0, so dA in (0, 1)
    h   <- h * dA + (dt * x)[:, None] * B[b][None, :]
    y    = sum_n h * C[b] + D * x                # [P]
    cache[slot[b]] = h

WHY THIS IS NOT THE CHUNKED SCAN. `ssd_chunk.py` is the training/prefill path:
a whole sequence, cut into chunks, quadratic inside and recurrent between, and
its state is an `scf.for` iteration argument that never touches memory except to
be checkpointed. At decode there is exactly ONE timestep, so there is no
sequence to be quadratic over, no chunk, and no loop. Everything that made the
chunked form fast is gone and what is left is the thing the chunked form was
built to avoid: a read-modify-write of the state itself.

  The state is big. Mamba-2 keeps d_state = 128 per head over d_head channels,
  so the recurrent state is far larger than a transformer's per-token KV -- it is
  the whole memory of the sequence in one tensor. That is why it is PAGED the
  same way a KV cache is: a pool of fixed slots, and a per-request index saying
  which slot is yours. mamba-ssm calls this kernel `selective_state_update` and
  vLLM calls it once per decode step per layer.

WHAT IT ASKS FOR THAT NOTHING ELSE DOES

  * A DATA-DEPENDENT BASE ON AN INOUT TENSOR. `paged_attention.py` gathers pages
    whose base comes from memory, but it only READS them. `kv_cache_write.py`
    scatters to a data-dependent row, but it writes a value it did not compute
    from that row. Here the same indexed [P, N] tile is loaded, transformed, and
    stored back to the address it came from -- a read-modify-write whose ADDRESS
    is a value. That is the combination: the alias analysis cannot prove the
    write lands where the read came from, because it does not know the index.

  * A RANK-3 TILE WHOSE OUTER EXTENT IS THE BATCH, reduced on its innermost axis
    only. `dt` and `x` are per (request, channel); `B` and `C` are per (request,
    state); `A` and `D` are per (channel) and shared across the batch. Four
    different broadcast patterns land on one [B, P, N] tile, which is the actual
    shape of the discretisation: `dt` is rank-1 in the state axis, `B` is rank-1
    in the channel axis, and their outer product is the input term.

  * A ZERO-ORDER-HOLD DISCRETISATION IN THE HOT PATH. `exp(dt * A)` with `dt`
    itself a `softplus`, evaluated per element of the state -- two transcendental
    ops on the widest tile in the kernel. `ssd_chunk.py` takes a log and an exp
    of a per-timestep scalar; here they are per state element, which is P*N times
    more of them.

THE SLOTS ARE A SHUFFLED PERMUTATION SLICE. No two requests share a slot, so the
answer cannot depend on write order, and an implementation that ignored the index
would read the wrong states rather than accidentally the right ones.

`A` is forced negative (the model parameterises it as -exp(A_log)) because dA is
a decay and a positive A makes the recurrence diverge -- the same reason
ssd_chunk.py keeps `a` inside (0, 1).

Mamba-2 decode: a read-modify-write of a [P,N] state tile whose base address is an index in memory. paged_attention.py only reads such a tile; kv_cache_write.py only writes one.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT, INOUT

NSLOT = 8                       # slots in the state pool
B, P, N = 4, 8, 16              # requests decoding, channels per head, d_state


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def selective_state_update_kernel(slot_ptr, cache_ptr, x_ptr, dt_ptr,
                                      dtb_ptr, a_ptr, b_ptr, c_ptr, d_ptr,
                                      y_ptr, B: tl.constexpr, P: tl.constexpr,
                                      N: tl.constexpr):
        rb = tl.arange(0, B)
        rp = tl.arange(0, P)
        rn = tl.arange(0, N)

        slot = tl.load(slot_ptr + rb)                       # [B] physical slots
        # The tile's BASE is data; the rest is affine. Same address is written.
        soff = (slot[:, None, None] * (P * N)
                + rp[None, :, None] * N + rn[None, None, :])
        h = tl.load(cache_ptr + soff)                       # [B, P, N]

        bp = rb[:, None] * P + rp[None, :]                  # [B, P]
        bn = rb[:, None] * N + rn[None, :]                  # [B, N]

        # DISCRETISATION. softplus keeps dt > 0; A < 0 keeps exp(dt*A) a decay.
        dtr = tl.load(dt_ptr + bp) + tl.load(dtb_ptr + rp)[None, :]
        dt = tl.log(1.0 + tl.exp(dtr))                      # [B, P]
        A = tl.load(a_ptr + rp[:, None] * N + rn[None, :])  # [P, N], shared
        dA = tl.exp(dt[:, :, None] * A[None, :, :])         # [B, P, N]

        x = tl.load(x_ptr + bp)                             # [B, P]
        Bm = tl.load(b_ptr + bn)                            # [B, N]
        Cm = tl.load(c_ptr + bn)                            # [B, N]

        # The input term is an OUTER PRODUCT of a channel vector and a state
        # vector, per request -- dt is rank-1 in n, B is rank-1 in p.
        h = h * dA + (dt * x)[:, :, None] * Bm[:, None, :]
        tl.store(cache_ptr + soff, h)

        # Read out: contract the state axis, which is the inner one.
        y = tl.sum(h * Cm[:, None, :], axis=2) + tl.load(d_ptr + rp)[None, :] * x
        tl.store(y_ptr + bp, y)

    return selective_state_update_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    slot = torch.randperm(NSLOT, generator=g)[:B].to(torch.int32)
    return {"slot": slot,
            "cache": torch.randn(NSLOT * P * N, generator=g, dtype=torch.float32),
            "x": torch.randn(B * P, generator=g, dtype=torch.float32),
            "dt": torch.randn(B * P, generator=g, dtype=torch.float32),
            "dtb": torch.randn(P, generator=g, dtype=torch.float32) * 0.1,
            # A = -exp(A_log): strictly negative, as the model parameterises it.
            "a": -torch.rand(P * N, generator=g, dtype=torch.float32).exp(),
            "b": torch.randn(B * N, generator=g, dtype=torch.float32) * 0.5,
            "c": torch.randn(B * N, generator=g, dtype=torch.float32) * 0.5,
            "d": torch.randn(P, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    slot = inputs["slot"].long()
    cache = inputs["cache"].reshape(NSLOT, P, N).clone()
    h = cache[slot]                                             # [B, P, N]
    dt = torch.nn.functional.softplus(
        inputs["dt"].reshape(B, P) + inputs["dtb"][None, :])
    dA = torch.exp(dt[:, :, None] * inputs["a"].reshape(P, N)[None, :, :])
    x = inputs["x"].reshape(B, P)
    Bm = inputs["b"].reshape(B, N)
    Cm = inputs["c"].reshape(B, N)
    h = h * dA + (dt * x)[:, :, None] * Bm[:, None, :]
    cache[slot] = h
    y = (h * Cm[:, None, :]).sum(dim=2) + inputs["d"][None, :] * x
    return {"cache": cache.reshape(-1).contiguous(),
            "y": y.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="selective_state_update_kernel", kernel=kernel,
    signature={"slot_ptr": "*i32", "cache_ptr": "*fp32", "x_ptr": "*fp32",
               "dt_ptr": "*fp32", "dtb_ptr": "*fp32", "a_ptr": "*fp32",
               "b_ptr": "*fp32", "c_ptr": "*fp32", "d_ptr": "*fp32",
               "y_ptr": "*fp32", "B": "constexpr", "P": "constexpr",
               "N": "constexpr"},
    constexprs={"B": B, "P": P, "N": N},
    args=[Arg("slot", IN, "int32", (B,)),
          Arg("cache", INOUT, "float32", (NSLOT * P * N,)),
          Arg("x", IN, "float32", (B * P,)),
          Arg("dt", IN, "float32", (B * P,)),
          Arg("dtb", IN, "float32", (P,)),
          Arg("a", IN, "float32", (P * N,)),
          Arg("b", IN, "float32", (B * N,)),
          Arg("c", IN, "float32", (B * N,)),
          Arg("d", IN, "float32", (P,)),
          Arg("y", OUT, "float32", (B * P,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
