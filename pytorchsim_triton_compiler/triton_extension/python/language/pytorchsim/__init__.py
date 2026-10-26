"""tl.extra.pytorchsim: a VCIX instruction a kernel writes by its encoding -- form, opcode, imm -- on
whole tiles in the layout the kernel names, and a tile kept in a layout the kernel names. The compiler
enforces the layout and passes the encoding through; it does not know what the instruction does."""
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
def vcix(form, opcode, imm, args, layouts=None, out=None, dtype=None, out_layout=None,
         layout=None, _semantic=None):
    """One `vcix.<form>` with this opcode and imm, args[i] in layouts[i] and a result -- `out` and
    `dtype`, or the operand's shape and type -- in `out_layout`; `layout` alone: every tile in it.
    A form with a vector result returns it, the others None."""
    from pytorchsim_triton_compiler.contract.vcix_form import FORMS, IMM_RANGE, OPCODE_RANGE
    from triton.experimental.gluon.language._layouts import BlockedLayout
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
    for l in layouts + ([out_layout] if results else []):
        if not isinstance(l, BlockedLayout) or l.cga_layout:
            raise ValueError(f"vcix: a layout is a one-CTA BlockedLayout, got {l!r}")

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
        if len(shape) != len(out_layout.order):
            raise ValueError(f"vcix: a rank-{len(shape)} result in a rank-"
                             f"{len(out_layout.order)} out_layout")
        ty = core.block_type(dtype, [int(d) for d in shape])
        ints += _spelled(out_layout) + list(ty.shape)
        tail = [b.get_null_value(dtype.to_ir(b))]
    elif out is not None or dtype is not None:
        raise ValueError(f"vcix: vcix.{form} has no vector result; out= and dtype= name none")
    got = b.pytorchsim_vcix([a.handle for a in args] + [b.get_int32(int(x)) for x in ints] + tail)
    return core.tensor(got, ty) if ty is not None else None


@builtin
def layout_constraint(x, layout, _semantic=None):
    """`x`, kept in `layout` (a one-CTA BlockedLayout, as lane_layout gives): the compiler places the tile
    so and fits the ops around it; a layout the machine cannot take is refused when the kernel compiles."""
    from triton.experimental.gluon.language._layouts import BlockedLayout
    layout = _unwrap_if_constexpr(layout)
    if not isinstance(x, core.tensor) or not x.type.is_block():
        raise ValueError(f"layout_constraint: a tile, got {x!r}")
    if not isinstance(layout, BlockedLayout) or layout.cga_layout or len(layout.order) != len(x.shape):
        raise ValueError(f"layout_constraint: a rank-{len(x.shape)} one-CTA BlockedLayout, got {layout!r}")
    b = _semantic.builder
    got = b.pytorchsim_layout_constraint([x.handle] + [b.get_int32(int(i)) for i in _spelled(layout)])
    return core.tensor(got, x.type)


def _spelled(l):
    """A layout as the plugin reads it: its rank, then each of size_per_thread, threads_per_warp,
    warps_per_cta and order."""
    return [len(l.order), *l.size_per_thread, *l.threads_per_warp, *l.warps_per_cta, *l.order]
