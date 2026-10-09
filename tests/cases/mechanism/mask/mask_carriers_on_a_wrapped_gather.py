"""Every mask carrier meeting a wrapped gather pointer.

    addr = (r % M) * C + j[c]        wrap on axis 0, gather on axis 1

    store 1  unstructured mask on axis 1 (the gathered one)  -> pointer gs_mask
    store 2  unstructured mask on axis 0                     -> access elem_mask
    store 3  no mask at all                                  -> none
    store 4  `r < 5`, a contiguous prefix                    -> box mask_dims
    load 5   unstructured mask on axis 1, on a READ          -> _empty_read_mask_carriers
    load 6   unstructured mask on axis 0, on a READ          -> access elem_mask
    load 7   `r < 5` on a READ                               -> box mask_dims

WHY ALL SEVEN ARE HERE. The wrap folds into a per-element index and drops
offset_stride to 1, which moves the park: a masked-off row used to be sent to
the padding by index*stride and now goes by index alone. A gs_mask is one bit
per ROW, so it also has to be widened to the tile first. And the read carrier
rebuilds the pointer without its mask -- copying its attributes BY NAME, which
is where a newly added `static_shape` goes missing if nobody names it.

M=3 under R=8, so rows 3..7 alias rows 0..4. `v` is read through the SAME
wrapped address, so aliasing rows write the same value and the stores do not
depend on order.

Every mask carrier on a gather pointer that also wraps, on both a read and a write: none, box, elem_mask and gs_mask.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import (KernelSpec, Arg, IN,
                                                        INOUT, OUT)

R, C, M, KEEP = 8, 8, 3, 5
OSIZE = M * C


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_carriers_on_a_wrapped_gather(v_ptr, j_ptr, mc_ptr, mr_ptr,
                                          o1_ptr, o2_ptr, o3_ptr, o4_ptr,
                                          o5_ptr,
                                          R: tl.constexpr, C: tl.constexpr,
                                          M: tl.constexpr,
                                          KEEP: tl.constexpr):
        r = tl.arange(0, R)
        c = tl.arange(0, C)
        j = tl.load(j_ptr + c)
        mc = tl.load(mc_ptr + c) < 5
        mr = tl.load(mr_ptr + r) < 5
        w = (r % M)[:, None] * C
        v = tl.load(v_ptr + (w + c[None, :]))
        tl.store(o1_ptr + (w + j[None, :]), v, mask=mc[None, :])
        tl.store(o2_ptr + (w + j[None, :]), v, mask=mr[:, None])
        tl.store(o4_ptr + (w + j[None, :]), v)
        tl.store(o5_ptr + (w + j[None, :]), v, mask=(r < KEEP)[:, None])
        lv = tl.load(v_ptr + (w + j[None, :]), mask=mc[None, :], other=0.0)
        ev = tl.load(v_ptr + (w + j[None, :]), mask=mr[:, None], other=0.0)
        bv = tl.load(v_ptr + (w + j[None, :]), mask=(r < KEEP)[:, None],
                     other=0.0)
        tl.store(o3_ptr + (r[:, None] * C + c[None, :]), lv + ev + bv)

    return mask_carriers_on_a_wrapped_gather


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"v": torch.randn(OSIZE, generator=g, dtype=torch.float32),
            "j": torch.randperm(C, generator=g).to(torch.int32),
            "mc": torch.randint(0, 10, (C,), generator=g, dtype=torch.int32),
            "mr": torch.randint(0, 10, (R,), generator=g, dtype=torch.int32),
            "o1": torch.zeros(OSIZE, dtype=torch.float32),
            "o2": torch.zeros(OSIZE, dtype=torch.float32),
            "o4": torch.zeros(OSIZE, dtype=torch.float32),
            "o5": torch.zeros(OSIZE, dtype=torch.float32)}


def reference(inputs):
    import torch
    v, j = inputs["v"], inputs["j"].long()
    mc, mr = inputs["mc"] < 5, inputs["mr"] < 5
    o1, o2 = inputs["o1"].clone(), inputs["o2"].clone()
    o4, o5 = inputs["o4"].clone(), inputs["o5"].clone()
    o3 = torch.zeros(R * C, dtype=torch.float32)
    for rr in range(R):
        w = (rr % M) * C
        for cc in range(C):
            if mc[cc]:
                o1[w + j[cc]] = v[w + cc]
            if mr[rr]:
                o2[w + j[cc]] = v[w + cc]
            o4[w + j[cc]] = v[w + cc]
            if rr < KEEP:
                o5[w + j[cc]] = v[w + cc]
            lv = v[w + j[cc]] if mc[cc] else 0.0
            ev = v[w + j[cc]] if mr[rr] else 0.0
            bv = v[w + j[cc]] if rr < KEEP else 0.0
            o3[rr * C + cc] = lv + ev + bv
    return {"o1": o1, "o2": o2, "o3": o3, "o4": o4, "o5": o5}


SPEC = KernelSpec(
    name="mask_carriers_on_a_wrapped_gather", kernel=kernel,
    signature={"v_ptr": "*fp32", "j_ptr": "*i32", "mc_ptr": "*i32",
               "mr_ptr": "*i32", "o1_ptr": "*fp32", "o2_ptr": "*fp32",
               "o3_ptr": "*fp32", "o4_ptr": "*fp32", "o5_ptr": "*fp32",
               "R": "constexpr", "C": "constexpr",
               "M": "constexpr", "KEEP": "constexpr"},
    constexprs={"R": R, "C": C, "M": M, "KEEP": KEEP},
    args=[Arg("v", IN, "float32", (OSIZE,)),
          Arg("j", IN, "int32", (C,)),
          Arg("mc", IN, "int32", (C,)),
          Arg("mr", IN, "int32", (R,)),
          Arg("o1", INOUT, "float32", (OSIZE,)),
          Arg("o2", INOUT, "float32", (OSIZE,)),
          Arg("o3", OUT, "float32", (R * C,)),
          Arg("o4", INOUT, "float32", (OSIZE,)),
          Arg("o5", INOUT, "float32", (OSIZE,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0}},
)
