"""c26's steps before the standard pipeline: promote_short_loops (a loop that runs once, or whose trips each
take a register and together one register group at the ISA's widest LMUL, becomes its body once per trip),
forward_spad_values (a scratchpad read takes the value written or read before it in its block; a write no
one reads goes; escaped globals stay) and lower_steps (a power-of-two vector.step becomes vid.v); exits 0
iff every case holds."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import c26_convert_pytorchsim_to_llvm as c26  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

pytorchsim.register()
T = machine.target()
VLEN = T.vlen_bits
E = VLEN // 32


def _module(body, globals_=""):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        return ctx, ir.Module.parse("module {\n" + globals_ + "\n  func.func @f(%x: f32) {\n" + body
                                    + "\n    return\n  }\n}")


def _names(module):
    return [o.name for o in walk(module.operation)]


def _loop(trips, step, elems):
    """A loop of `trips` trips stepping `step`, each storing a vector of `elems` f32 (none when 0)."""
    store = (f"""      %v = vector.broadcast %x : f32 to vector<{elems}xf32>
      vector.store %v, %b[%i] : memref<{trips * step}xf32, 1>, vector<{elems}xf32>""" if elems else
             f"""      memref.store %x, %b[%i] : memref<{trips * step}xf32, 1>""")
    return f"""    %b = memref.alloc() : memref<{trips * step}xf32, 1>
    affine.for %i = 0 to {trips * step} step {step} {{
{store}
    }}
    memref.dealloc %b : memref<{trips * step}xf32, 1>"""


def _promoted(trips, step, elems):
    ctx, m = _module(_loop(trips, step, elems))
    with ctx, ir.Location.unknown():
        c26.promote_short_loops(m, VLEN)
        names = _names(m)
    return "affine.for" not in names, names.count("vector.store") + names.count("memref.store")


def a_one_trip_loop_becomes_its_body():
    gone, stores = _promoted(1, 4 * E, 4 * E)
    assert gone and stores == 1, (gone, stores)


def a_loop_whose_trips_each_take_a_register_and_together_one_lmul_8_group_unrolls():
    gone, stores = _promoted(machine.ISA_MAX_LMUL, E, E)
    assert gone and stores == machine.ISA_MAX_LMUL, (gone, stores)


def a_loop_past_one_lmul_8_group_stays():
    gone, stores = _promoted(2 * machine.ISA_MAX_LMUL, E, E)
    assert not gone and stores == 1, (gone, stores)


def a_loop_whose_trips_take_less_than_a_register_stays():
    gone, stores = _promoted(4, E // 2, E // 2)
    assert not gone and stores == 1, (gone, stores)


def a_loop_stepping_no_vector_stays():
    gone, stores = _promoted(4, 1, 0)
    assert not gone and stores == 1, (gone, stores)


_GLOBALS = f"""  memref.global @a : memref<{4 * E}xf32, 1>
  memref.global @b : memref<{4 * E}xf32, 1>"""


def _forwarded(body):
    ctx, m = _module(body, _GLOBALS)
    with ctx, ir.Location.unknown():
        c26.forward_spad_values(m)
        return _names(m)


def _ga(name="a", var="g"):
    return f"""    %{var} = memref.get_global @{name} : memref<{4 * E}xf32, 1>"""


_C0 = """    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.0 : f32
    %v = vector.broadcast %x : f32 to vector<{e}xf32>""".format(e=E)


def a_read_of_a_place_just_written_takes_the_written_value_and_the_write_goes():
    names = _forwarded(_C0 + "\n" + _ga() + f"""
    vector.transfer_write %v, %g[%c0] {{in_bounds = [true]}} : vector<{E}xf32>, memref<{4 * E}xf32, 1>
    %r = vector.transfer_read %g[%c0], %pad {{in_bounds = [true]}} : memref<{4 * E}xf32, 1>, vector<{E}xf32>
    "test.use"(%r) : (vector<{E}xf32>) -> ()""")
    assert "vector.transfer_read" not in names and "vector.transfer_write" not in names, names


def a_write_a_later_write_of_the_same_place_replaces_unread_goes():
    names = _forwarded(_C0 + "\n" + _ga() + f"""
    %i = "test.index"() : () -> index
    %w = arith.addf %v, %v : vector<{E}xf32>
    vector.transfer_write %v, %g[%c0] {{in_bounds = [true]}} : vector<{E}xf32>, memref<{4 * E}xf32, 1>
    vector.transfer_write %w, %g[%c0] {{in_bounds = [true]}} : vector<{E}xf32>, memref<{4 * E}xf32, 1>
    %r = vector.transfer_read %g[%i], %pad {{in_bounds = [true]}} : memref<{4 * E}xf32, 1>, vector<{E}xf32>
    "test.use"(%r) : (vector<{E}xf32>) -> ()""")
    assert names.count("vector.transfer_write") == 1 and names.count("vector.transfer_read") == 1, names


def an_escaped_global_keeps_its_writes_and_reads():
    names = _forwarded(_C0 + "\n" + _ga() + f"""
    vector.transfer_write %v, %g[%c0] {{in_bounds = [true]}} : vector<{E}xf32>, memref<{4 * E}xf32, 1>
    %r = vector.transfer_read %g[%c0], %pad {{in_bounds = [true]}} : memref<{4 * E}xf32, 1>, vector<{E}xf32>
    "test.use"(%r) : (vector<{E}xf32>) -> ()
    "test.dma"(%g) : (memref<{4 * E}xf32, 1>) -> ()""")
    assert names.count("vector.transfer_write") == 1 and names.count("vector.transfer_read") == 1, names


def a_read_after_an_op_touching_the_global_is_not_forwarded():
    names = _forwarded(_C0 + "\n" + _ga() + "\n" + _ga("b", "h") + f"""
    vector.transfer_write %v, %g[%c0] {{in_bounds = [true]}} : vector<{E}xf32>, memref<{4 * E}xf32, 1>
    scf.execute_region {{
      %w = arith.addf %v, %v : vector<{E}xf32>
      vector.transfer_write %w, %g[%c0] {{in_bounds = [true]}} : vector<{E}xf32>, memref<{4 * E}xf32, 1>
      scf.yield
    }}
    %r = vector.transfer_read %g[%c0], %pad {{in_bounds = [true]}} : memref<{4 * E}xf32, 1>, vector<{E}xf32>
    vector.transfer_write %r, %h[%c0] {{in_bounds = [true]}} : vector<{E}xf32>, memref<{4 * E}xf32, 1>
    "test.dma"(%h) : (memref<{4 * E}xf32, 1>) -> ()""")
    assert names.count("vector.transfer_read") == 1, names


def _steps(n):
    ctx, m = _module(f"""    %s = vector.step : vector<{n}xindex>
    %c = arith.index_cast %s : vector<{n}xindex> to vector<{n}xi32>
    "test.use"(%c) : (vector<{n}xi32>) -> ()""")
    with ctx, ir.Location.unknown():
        c26.lower_steps(m, VLEN)
        return m, [o for o in walk(m.operation)]


def a_power_of_two_step_is_vid_at_its_casts_width():
    m, ops = _steps(E)
    vids = [o for o in ops if o.name == "llvm.call_intrinsic"
            and ir.StringAttr(o.attributes["intrin"]).value == "llvm.riscv.vid"]
    names = [o.name for o in ops]
    assert len(vids) == 1 and "vector.step" not in names and "arith.index_cast" not in names, names
    assert ir.VectorType(ir.VectorType(vids[0].results[0].type)).element_type == ir.IntegerType.get_signless(
        32, context=m.context), str(vids[0].results[0].type)


def a_step_of_no_power_of_two_length_stays():
    _, ops = _steps(E + E // 2)
    assert "vector.step" in [o.name for o in ops]


def a_step_wider_than_one_lmul_8_group_stays():
    """Breaks if vid.v is built past the widest register group (LMUL 16 is no instruction's)."""
    _, ops = _steps(2 * E * machine.ISA_MAX_LMUL)
    assert "vector.step" in [o.name for o in ops]


CASES = [a_one_trip_loop_becomes_its_body,
         a_loop_whose_trips_each_take_a_register_and_together_one_lmul_8_group_unrolls,
         a_loop_past_one_lmul_8_group_stays,
         a_loop_whose_trips_take_less_than_a_register_stays,
         a_loop_stepping_no_vector_stays,
         a_read_of_a_place_just_written_takes_the_written_value_and_the_write_goes,
         a_write_a_later_write_of_the_same_place_replaces_unread_goes,
         an_escaped_global_keeps_its_writes_and_reads,
         a_read_after_an_op_touching_the_global_is_not_forwarded,
         a_power_of_two_step_is_vid_at_its_casts_width,
         a_step_of_no_power_of_two_length_stays,
         a_step_wider_than_one_lmul_8_group_stays]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
