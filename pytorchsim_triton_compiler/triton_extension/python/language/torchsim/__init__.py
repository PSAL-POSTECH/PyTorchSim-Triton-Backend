"""tl.extra.torchsim: a VCIX instruction a kernel writes by its encoding -- form, opcode, imm -- on
whole tiles in the layout the kernel names, and a tile's layout a kernel states at a point of its
program (constrain). The compiler enforces the layout and passes the encoding through; it does not
know what the instruction does. A layout is a gluon BlockedLayout of one CTA or a
DistributedLinearLayout."""
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


def _layout_rank(l):
    """`l`'s rank, or a refusal when it is no layout a kernel may name here."""
    from triton.experimental.gluon.language._layouts import BlockedLayout, DistributedLinearLayout
    if isinstance(l, BlockedLayout) and not l.cga_layout:
        return len(l.order)
    if isinstance(l, DistributedLinearLayout):
        return len(l.shape)
    raise ValueError(f"a layout is a one-CTA BlockedLayout or a DistributedLinearLayout, got {l!r}")


def _check_shape(l, shape, what):
    """A linear layout states its tile's shape: refused on a tile of another."""
    from triton.experimental.gluon.language._layouts import DistributedLinearLayout
    shape = [int(d) for d in shape]
    if _layout_rank(l) != len(shape):
        raise ValueError(f"{what}: a rank-{_layout_rank(l)} layout on a rank-{len(shape)} tile")
    if isinstance(l, DistributedLinearLayout) and list(l.shape) != shape:
        raise ValueError(f"{what}: a linear layout of shape {list(l.shape)} on a {shape} tile")


def _spelled(l):
    """`l` as the i32s the plugin's builders read: 0 and a blocked layout's rank, size_per_thread,
    threads_per_warp, warps_per_cta, order; or 1 and a linear layout's rank, its register, lane,
    warp and block basis counts, the bases and the shape."""
    from triton.experimental.gluon.language._layouts import BlockedLayout
    if isinstance(l, BlockedLayout):
        return [0, len(l.order), *l.size_per_thread, *l.threads_per_warp, *l.warps_per_cta, *l.order]
    bases = (l.reg_bases, l.lane_bases, l.warp_bases, l.block_bases)
    return [1, len(l.shape), *map(len, bases), *(x for bs in bases for b in bs for x in b), *l.shape]


@builtin
def constrain(x, layout, _semantic=None):
    """`x`, in `layout` at this point of the program: the compiler lays the tile out so, and where
    another op fixes another layout for it, converts between the two."""
    layout = _unwrap_if_constexpr(layout)
    if not isinstance(x, core.tensor) or not x.type.is_block():
        raise ValueError(f"constrain: a layout is a tile's, got {x!r}")
    _check_shape(layout, x.shape, "constrain")
    b = _semantic.builder
    got = b.torchsim_layout_constraint([x.handle] + [b.get_int32(int(v)) for v in _spelled(layout)])
    return core.tensor(got, x.type)


@builtin
def vcix(form, opcode, imm, args, layouts=None, out=None, dtype=None, out_layout=None,
         layout=None, _semantic=None):
    """One `vcix.<form>` with this opcode and imm, args[i] in layouts[i] and a result -- `out` and
    `dtype`, or the operand's shape and type -- in `out_layout`; `layout` alone: every tile in it.
    A form with a vector result returns it, the others None."""
    from pytorchsim_triton_compiler.contract.vcix_form import FORMS, IMM_RANGE, OPCODE_RANGE
    form, opcode, imm, out, dtype, layout, out_layout = (_unwrap_if_constexpr(a) for a in (
        form, opcode, imm, out, dtype, layout, out_layout))
    layouts = _unwrap_if_constexpr(layouts)
    if form not in FORMS:
        raise ValueError(f"vcix: {form!r} is no VCIX op form; the forms are {', '.join(FORMS)}")
    vectors_in, results = FORMS[form]
    if not vectors_in and not results:
        raise NotImplementedError(f"vcix: vcix.{form} names no tile, so there is no layout to "
                                  f"enforce; not carried")
    if int(opcode) not in OPCODE_RANGE or int(imm) not in IMM_RANGE:
        raise ValueError(f"vcix: opcode {opcode} and imm {imm} do not fit the fields, which hold "
                         f"{OPCODE_RANGE} and {IMM_RANGE}")
    args = list(args.values if isinstance(args, core.tuple) else args)
    args = [a if isinstance(a, core.tensor) else _semantic.to_tensor(a) for a in args]
    if len(args) != vectors_in:
        raise ValueError(f"vcix: vcix.{form} takes {vectors_in} vector operand(s), got {len(args)}")
    if layout is not None:
        if layouts is not None or out_layout is not None:
            raise ValueError("vcix: layout= is every tile's layout; give it or layouts= and "
                             "out_layout=, not both")
        layouts = [layout] * len(args)
        out_layout = layout if results else None
    layouts = [_unwrap_if_constexpr(x) for x in
               (layouts.values if isinstance(layouts, core.tuple) else (layouts or []))]
    if len(layouts) != len(args):
        raise ValueError(f"vcix: {len(layouts)} layouts for {len(args)} operands")
    if results and out_layout is None:
        raise ValueError(f"vcix: vcix.{form} yields a tile; give out_layout=")
    if not results and out_layout is not None:
        raise ValueError(f"vcix: vcix.{form} has no vector result; out_layout= names none")
    for l, a in zip(layouts, args):
        _check_shape(l, a.shape, "vcix")
    ints = [opcode, imm, len(form), *map(ord, form), int(bool(results))]
    for l in layouts:
        ints += _spelled(l)
    b = _semantic.builder
    tail = []
    ty = None
    if results:
        shape = list(out) if out is not None else (args[0].shape if args else None)
        dtype = dtype if dtype is not None else (args[0].dtype if args else None)
        if shape is None or dtype is None:
            raise ValueError(f"vcix: vcix.{form} has no vector operand to take the result's "
                             f"shape and type from; give out= and dtype=")
        _check_shape(out_layout, shape, "vcix")
        ty = core.block_type(dtype, [int(d) for d in shape])
        ints += _spelled(out_layout) + list(ty.shape)
        tail = [b.get_null_value(dtype.to_ir(b))]
    elif out is not None or dtype is not None:
        raise ValueError(f"vcix: vcix.{form} has no vector result; out= and dtype= name none")
    got = b.torchsim_vcix([a.handle for a in args] + [b.get_int32(int(x)) for x in ints] + tail)
    return core.tensor(got, ty) if ty is not None else None
