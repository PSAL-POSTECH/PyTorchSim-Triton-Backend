"""tl.extra.torchsim: a VCIX instruction named in a kernel, on whole tiles, with the layout its
operands and result take -- a Triton BlockedLayout, the #ttg.blocked encoding TTGIR carries.
Imported while triton.language is, so gluon and the compiler are reached only from a call."""
from triton.language import core
from triton.language.core import _unwrap_if_constexpr, builtin
from triton.runtime.jit import constexpr_function


@constexpr_function
def lane_layout(axis, rank=2):
    """The machine's lanes along `axis`, one element each; the lane count is the target's."""
    from pytorchsim_triton_compiler import machine
    from triton.experimental.gluon.language._layouts import BlockedLayout
    tpw = [1] * rank
    tpw[axis] = machine.VECTORLANE_SIZE
    order = [axis] + [d for d in reversed(range(rank)) if d != axis]
    return BlockedLayout(size_per_thread=[1] * rank, threads_per_warp=tpw,
                         warps_per_cta=[1] * rank, order=order)


@builtin
def vcix(instruction, args, layout, dtype=None, _semantic=None):
    """The model's `instruction` on `args`, every operand and the result in `layout`; the result
    has args[0]'s shape and type."""
    from pytorchsim_triton_compiler.contract.sfu import SFU
    from triton.experimental.gluon.language._layouts import BlockedLayout
    instruction = _unwrap_if_constexpr(instruction)
    layout = _unwrap_if_constexpr(layout)
    dtype = _unwrap_if_constexpr(dtype)
    if instruction not in SFU:
        raise ValueError(f"vcix: {instruction!r} is not an instruction of this machine's model; "
                         f"it has {', '.join(sorted(SFU))}")
    if not isinstance(layout, BlockedLayout) or layout.cga_layout:
        raise ValueError(f"vcix: layout is a one-CTA BlockedLayout, got {layout!r}")
    args = list(args.values if isinstance(args, core.tuple) else args)
    args = [a if isinstance(a, core.tensor) else _semantic.to_tensor(a) for a in args]
    if len(args) != 1:
        raise ValueError(f"vcix: {instruction} is an SFU instruction, one operand in and one "
                         f"result out; got {len(args)} operands")
    if dtype is not None and dtype != args[0].dtype:
        raise NotImplementedError(f"vcix: a result type other than the first operand's "
                                  f"({args[0].dtype}) is not carried yet, asked {dtype}")
    opcode, imm = SFU[instruction]
    ints = [opcode, imm, len(layout.order), *layout.size_per_thread, *layout.threads_per_warp,
            *layout.warps_per_cta, *layout.order]
    b = _semantic.builder
    out = b.torchsim_vcix([a.handle for a in args] + [b.get_int32(int(x)) for x in ints])
    return core.tensor(out, args[0].type)
