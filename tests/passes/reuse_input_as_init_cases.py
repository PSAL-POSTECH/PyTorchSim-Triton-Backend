"""pytorchsim-bufferize's first step, an unread init taking an input, on IR no kernel reaches: which input
becomes the init, what is left alone, and that one-shot-bufferize keeps what writing in place would have
destroyed.
    python3 tests/passes/reuse_input_as_init_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.passmanager import PassManager  # noqa: E402
from pytorchsim_triton_compiler.passes.transforms import bufferize
from pytorchsim_triton_compiler.passes.utils.ir_utils import walk  # noqa: E402

ID = "affine_map<(d0, d1) -> (d0, d1)>"
TURNED = "affine_map<(d0, d1) -> (d1, d0)>"
T = "tensor<4x4xf32>"
HEAD = f"""func.func @f(%x: {T}, %y: {T}, %h: tensor<4x4xf16>) -> {T} {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : {T}
  %t = linalg.fill ins(%z : f32) outs(%e : {T}) -> {T}
"""


def _generic(name, ins, maps=None, init="%e", body="%s = arith.addf %a0, %a{last} : f32",
             iters='"parallel", "parallel"', types=None):
    """One linalg.generic named `name` over `ins`, yielding %s."""
    maps = maps or [ID] * len(ins)
    types = types or [T] * len(ins)
    args = ", ".join(f"%a{k}: {ty.split('x')[-1][:-1]}" for k, ty in enumerate(types))
    return (f"  {name} = linalg.generic {{indexing_maps = [{', '.join(maps + [ID])}], "
            f"iterator_types = [{iters}]}} ins({', '.join(ins)} : {', '.join(types)}) "
            f"outs({init} : {T}) {{\n  ^bb0({args}, %o: f32):\n    "
            + body.format(last=len(ins) - 1) + f"\n    linalg.yield %s : f32\n  }} -> {T}\n")


def _run(text, one_shot=False):
    """The module of `text` after the pass (and one-shot-bufferize), and its generics in order."""
    m = ir.Module.parse(text)
    bufferize.reuse_unread_inits(m)
    assert m.operation.verify()
    if one_shot:
        PassManager.parse(bufferize.BUFFERIZE_PIPELINE).run(m.operation)
    return m, [o for o in walk(m.operation) if o.name == "linalg.generic"]


def _loop(body, carried="%t"):
    return (HEAD + "  %r = scf.for %i = %c0 to %c4 step %c1 "
            + f"iter_args(%acc = {carried}) -> ({T}) {{\n" + body
            + f"  scf.yield %n : {T}\n  }}\n  return %r : {T}\n}}\n")


def _case(fn):
    def wrapped():
        with ir.Context() as ctx, ir.Location.unknown():
            ctx.allow_unregistered_dialects = True
            fn()
    wrapped.__name__ = fn.__name__
    return wrapped


@_case
def the_carried_input_is_taken_whatever_its_position():
    for ins in (["%x", "%acc"], ["%acc", "%x"], ["%x", "%y", "%acc"]):
        _, (g,) = _run(_loop(_generic("%n", ins)))
        assert g.operands[len(ins)] == g.operands[ins.index("%acc")], ins
        m, ops = _run(_loop(_generic("%n", ins)), one_shot=True)
        loop = [o for o in walk(m.operation) if o.name == "scf.for"][0]
        assert [o.name for o in walk(loop) if o.name.startswith(("memref.", "linalg."))] == [
            "linalg.generic", "linalg.yield"], ins


@_case
def an_input_read_after_the_op_is_not_taken():
    tail = _generic("%m", ["%x", "%n"]) + f"  return %m : {T}\n}}\n"
    _, (g, _m) = _run(HEAD + _generic("%n", ["%x", "%y"]) + tail)
    assert g.operands[2] == g.operands[1]
    _, (g, _m) = _run(HEAD + _generic("%n", ["%x"], body="%s = arith.negf %a0 : f32") + tail)
    assert g.operands[1].owner.name == "tensor.empty"


@_case
def an_input_read_only_before_the_op_is_taken_after_one_nothing_else_uses():
    head = HEAD + _generic("%k", ["%t"], body="%s = arith.negf %a0 : f32")
    tail = _generic("%m", ["%k", "%n"]) + f"  return %m : {T}\n}}\n"
    _, (_k, g, _m) = _run(head + _generic("%n", ["%t", "%y"]) + tail)
    assert g.operands[2] == g.operands[1]
    text = head + _generic("%n", ["%t"], body="%s = arith.negf %a0 : f32") + tail
    for one_shot in (False, True):
        _, (_k, g, _m) = _run(text, one_shot=one_shot)
        assert g.operands[1] == g.operands[0], one_shot


@_case
def an_input_made_outside_the_loop_that_reads_it_is_not_taken():
    _, (g,) = _run(_loop(_generic("%n", ["%t"], body="%s = arith.negf %a0 : f32"), carried="%x"))
    assert g.operands[1].owner.name == "tensor.empty"


@_case
def no_input_of_the_type_or_of_the_map_changes_nothing():
    for kw in (dict(ins=["%h"], types=["tensor<4x4xf16>"], body="%s = arith.extf %a0 : f16 to f32"),
               dict(ins=["%x"], maps=[TURNED], body="%s = arith.negf %a0 : f32")):
        m, (g,) = _run(HEAD + _generic("%n", **kw) + f"  return %n : {T}\n}}\n")
        assert g.operands[1].owner.name == "tensor.empty", kw


@_case
def an_init_the_body_reads_and_a_reduction_are_left_alone():
    for kw in (dict(body="%s = arith.addf %a0, %o : f32"),
               dict(iters='"parallel", "reduction"', body="%s = arith.addf %a0, %o : f32")):
        _, (g,) = _run(HEAD + _generic("%n", ["%x"], init="%t", **kw) + f"  return %n : {T}\n}}\n")
        assert g.operands[1].owner.name == "linalg.fill", kw


@_case
def bufferize_writes_elsewhere_when_the_input_is_also_read_through_another_map():
    text = HEAD + _generic("%n", ["%t", "%t"], maps=[ID, TURNED]) + f"  return %n : {T}\n}}\n"
    _, (g,) = _run(text)
    assert g.operands[2] == g.operands[0]
    _, (g,) = _run(text, one_shot=True)
    assert g.operands[2] != g.operands[0]


@_case
def bufferize_keeps_what_the_input_views_when_that_is_read_afterwards():
    view = (f"  %v = tensor.collapse_shape %t [[0, 1]] : {T} into tensor<16xf32>\n"
            "  %w = tensor.expand_shape %v [[0, 1]] output_shape [4, 4] : tensor<16xf32> "
            f"into {T}\n")
    text = (HEAD + view + _generic("%n", ["%w"], body="%s = arith.negf %a0 : f32")
            + _generic("%m", ["%t", "%n"]) + f"  return %m : {T}\n}}\n")
    _, (g, _m) = _run(text)
    assert g.operands[1] == g.operands[0]
    _, (g, later) = _run(text, one_shot=True)
    assert later.operands[0] != g.operands[1]


CASES = [the_carried_input_is_taken_whatever_its_position,
         an_input_read_after_the_op_is_not_taken,
         an_input_read_only_before_the_op_is_taken_after_one_nothing_else_uses,
         an_input_made_outside_the_loop_that_reads_it_is_not_taken,
         no_input_of_the_type_or_of_the_map_changes_nothing,
         an_init_the_body_reads_and_a_reduction_are_left_alone,
         bufferize_writes_elsewhere_when_the_input_is_also_read_through_another_map,
         bufferize_keeps_what_the_input_views_when_that_is_read_afterwards]


def main():
    failed = 0
    for case in CASES:
        try:
            case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {e!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
