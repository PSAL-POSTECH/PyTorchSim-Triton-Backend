"""bufferize hands one scratchpad global to the allocs whose lives do not overlap: a tile is live
from its alloc to the last op touching it or a view of it, to the end of every loop that reads it
but did not make it. Tiles of two types never share.
    python3 tests/passes/spad_reuse_cases.py        exit 0 iff every case holds"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.passes import p04_bufferize  # noqa: E402

_F32 = "memref<8x4xf32>"


def _placed(body):
    """The module after the allocs of `body` are placed; (its text, the globals it declares)."""
    from mlir.ir import Context, Location, Module
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = Module.parse("module {\n  func.func @f(%n: index) {\n    %c0 = arith.constant 0 : index\n"
                              "    %c1 = arith.constant 1 : index\n" + body + "    return\n  }\n}\n")
        p04_bufferize._rewrite_alloc_to_spad_global(module)
        text = str(module)
    return text, re.findall(r"memref\.global @(\w+)", text)


def _global_of(text, marker):
    """The global the op tagged `marker` reads: its %operand's get_global name."""
    m = re.search(r'"test\.use"\((%\w+)[^\n]*' + marker, text)
    assert m, text
    g = re.search(re.escape(m.group(1)) + r" = memref\.get_global @(\w+)", text)
    assert g, text
    return g.group(1)


def three_tiles_used_in_turn_share_one_global():
    text, globs = _placed(f"""    %a = memref.alloc() : {_F32}
    "test.use"(%a) {{tag = "a"}} : ({_F32}) -> ()
    %b = memref.alloc() : {_F32}
    "test.use"(%b) {{tag = "b"}} : ({_F32}) -> ()
    %c = memref.alloc() : {_F32}
    "test.use"(%c) {{tag = "c"}} : ({_F32}) -> ()
""")
    assert len(globs) == 1, globs


def two_tiles_live_at_once_take_two():
    text, globs = _placed(f"""    %a = memref.alloc() : {_F32}
    %b = memref.alloc() : {_F32}
    "test.use"(%a) {{tag = "a"}} : ({_F32}) -> ()
    "test.use"(%b) {{tag = "b"}} : ({_F32}) -> ()
""")
    assert len(globs) == 2, globs


def a_view_keeps_its_tile_alive():
    text, globs = _placed(f"""    %a = memref.alloc() : {_F32}
    %v = memref.subview %a[0, 0] [4, 4] [1, 1] : {_F32} to memref<4x4xf32, strided<[4, 1]>>
    %b = memref.alloc() : {_F32}
    "test.use"(%b) {{tag = "b"}} : ({_F32}) -> ()
    "test.use"(%v) {{tag = "v"}} : (memref<4x4xf32, strided<[4, 1]>>) -> ()
""")
    assert len(globs) == 2, globs


def a_tile_read_in_a_loop_lives_to_the_loops_end_and_the_loops_own_tiles_share():
    text, globs = _placed(f"""    %a = memref.alloc() : {_F32}
    %b = memref.alloc() : {_F32}
    scf.for %i = %c0 to %n step %c1 {{
      %t = memref.alloc() : {_F32}
      "test.use"(%a, %t) {{tag = "at"}} : ({_F32}, {_F32}) -> ()
      %u = memref.alloc() : {_F32}
      "test.use"(%u) {{tag = "u"}} : ({_F32}) -> ()
    }}
    "test.use"(%b) {{tag = "b"}} : ({_F32}) -> ()
""")
    assert len(globs) == 3, globs
    t_glob = re.search(r'"test\.use"\(%\w+, (%\w+)\)[^\n]*tag = "at"', text).group(1)
    t_glob = re.search(re.escape(t_glob) + r" = memref\.get_global @(\w+)", text).group(1)
    assert _global_of(text, 'tag = "u"') == t_glob, text
    assert _global_of(text, 'tag = "b"') not in (t_glob, _global_of(text, 'tag = "at"')), text


def a_loops_results_may_be_any_tile_that_entered_it():
    """Two tiles carried through one scf.for: the loop's second result is read after the loop, so
    the f32 tile it stands for stays live past the loop and a later f32 alloc does not take its
    global. Breaks if a loop's results are aliased to the first tile that entered it."""
    text, globs = _placed(f"""    %idx = memref.alloc() : memref<8x4xi32>
    %acc = memref.alloc() : {_F32}
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%a0 = %idx, %a1 = %acc) -> (memref<8x4xi32>, {_F32}) {{
      "test.use"(%a0, %a1) {{tag = "body"}} : (memref<8x4xi32>, {_F32}) -> ()
      scf.yield %a0, %a1 : memref<8x4xi32>, {_F32}
    }}
    %g = memref.alloc() : {_F32}
    "test.use"(%g) {{tag = "g"}} : ({_F32}) -> ()
    "test.use"(%r#1) {{tag = "r"}} : ({_F32}) -> ()
""")
    assert len(globs) == 3, globs


def two_types_never_share():
    text, globs = _placed(f"""    %a = memref.alloc() : {_F32}
    "test.use"(%a) {{tag = "a"}} : ({_F32}) -> ()
    %b = memref.alloc() : memref<8x4xi32>
    "test.use"(%b) {{tag = "b"}} : (memref<8x4xi32>) -> ()
""")
    assert len(globs) == 2, globs


CASES = [three_tiles_used_in_turn_share_one_global, two_tiles_live_at_once_take_two,
         a_view_keeps_its_tile_alive,
         a_tile_read_in_a_loop_lives_to_the_loops_end_and_the_loops_own_tiles_share,
         a_loops_results_may_be_any_tile_that_entered_it, two_types_never_share]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:400]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
