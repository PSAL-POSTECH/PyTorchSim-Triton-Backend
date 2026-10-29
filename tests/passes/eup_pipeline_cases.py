"""schedule_vpu_stream pipelines an EUP loop whose pops write the buffer its pushes read when every trip
reads and writes only its own elements (the same affine access, the step apart, no wider than the step),
and keeps the order, saying why, when a trip may write what a later trip's push reads, when the EUP does
not hold the groups in flight, or when the machine does not say what it holds. The tpu target's max_lmul
bounds an exp's loop to the widest LMUL whose groups the EUP holds, and leaves it alone when every LMUL fits.
    python3 tests/passes/eup_pipeline_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.dialects import arith  # noqa: E402
from pytorchsim_triton_compiler.dialect import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.targets.tpu import pytorchsim_tpu  # noqa: E402
from pytorchsim_triton_compiler.targets.tpu import lower_vector_functions as lvf  # noqa: E402
from pytorchsim_triton_compiler.targets.tpu import schedule_vpu_stream as svs  # noqa: E402
from pytorchsim_triton_compiler.passes.utils.vector_loops import VECTOR_LOOP  # noqa: E402
from pytorchsim_triton_compiler.passes.utils.ir_utils import walk  # noqa: E402

pytorchsim.register()


def _any(views, cls):
    return any(isinstance(v, cls) for v in views)


def _loop(load, store):
    """One EUP loop over a 256-element buffer: `load` and `store` are the affine accesses of trip %i."""
    return f'''func.func @f(%m: memref<256xf32>) {{
  %c3 = arith.constant 3 : i64
  %c0 = arith.constant 0 : i64
  %c32 = arith.constant 32 : i64
  affine.for %i = 0 to 256 step 32 {{
    %x = {load}
    "pytorchsim_tpu.eup.push"(%x) : (vector<32xf32>) -> ()
    %y = "pytorchsim_tpu.eup.pop"() : () -> vector<32xf32>
    {store}
  }}
  return
}}'''


VLEN = 256
# A group here is 32 f32 elements, 4 registers of VLEN bits; a pop finds DEPTH + 1 groups inside the EUP.
GROUP = 32 * 32 // VLEN
HOLDS = (svs.DEPTH + 1) * GROUP


def _schedule(body, holds=HOLDS):
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse("module {\n" + body + "\n}\n")
        svs.schedule(m, holds, VLEN)
        ops = list(walk(m.operation))
        pushes = sum(o.name == svs.PUSH for o in ops)
        why = [ir.StringAttr(o.attributes[svs.EUP_PIPELINE]).value for o in ops
               if o.name == "affine.for" and svs.EUP_PIPELINE in o.attributes]
        return pushes, why


IN_PLACE_LOAD = "affine.vector_load %m[%i] : memref<256xf32>, vector<32xf32>"
IN_PLACE_STORE = "affine.vector_store %y, %m[%i] : memref<256xf32>, vector<32xf32>"


def an_in_place_loop_whose_trips_touch_their_own_elements_is_pipelined():
    pushes, why = _schedule(_loop(IN_PLACE_LOAD, IN_PLACE_STORE))
    assert why == [] and pushes == 1 + svs.DEPTH, (pushes, why)


def _count(body, name, holds=HOLDS):
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse("module {\n" + body + "\n}\n")
        svs.schedule(m, holds, VLEN)
        return sum(o.name == name for o in walk(m.operation))


def a_stage0_value_read_after_the_pop_is_read_again_in_stage1():
    # silu: x feeds the push and is divided by what the pop returns. Stage 1 reads x of its own trip again,
    # so every trip's load runs twice -- once DEPTH groups ahead for the push, once beside its pop.
    body = _loop(IN_PLACE_LOAD, "%z = arith.divf %x, %y : vector<32xf32>\n    "
                 + IN_PLACE_STORE.replace("%y,", "%z,"))
    pushes, why = _schedule(body)
    assert why == [] and pushes == 1 + svs.DEPTH, (pushes, why)
    loads = _count(body, "affine.vector_load")
    assert loads == 2 * (1 + svs.DEPTH), loads


def _transfer_loop(shift):
    """The bf16-packed form of an in-place EUP loop: vector.transfer of trip %i at `%i >> shift`."""
    return f'''func.func @f(%m: memref<128xi32>) {{
  %c3 = arith.constant 3 : i64
  %c0 = arith.constant 0 : i64
  %c32 = arith.constant 32 : i64
  %p = arith.constant 0 : i32
  affine.for %i = 0 to 256 step 64 {{
    %s = arith.constant {shift} : index
    %j = arith.shrui %i, %s : index
    %r = vector.transfer_read %m[%j], %p {{in_bounds = [true]}} : memref<128xi32>, vector<32xi32>
    %x = arith.bitcast %r : vector<32xi32> to vector<32xf32>
    "pytorchsim_tpu.eup.push"(%x) : (vector<32xf32>) -> ()
    %y = "pytorchsim_tpu.eup.pop"() : () -> vector<32xf32>
    %w = arith.bitcast %y : vector<32xf32> to vector<32xi32>
    %t = arith.constant {shift} : index
    %k = arith.shrui %i, %t : index
    vector.transfer_write %w, %m[%k] {{in_bounds = [true]}} : vector<32xi32>, memref<128xi32>
  }}
  return
}}'''


def an_in_place_transfer_loop_whose_trips_touch_their_own_elements_is_pipelined():
    # 64 a trip shifted by one is 32 elements a trip, the vector's width: no trip meets another's.
    pushes, why = _schedule(_transfer_loop(1))
    assert why == [] and pushes == 1 + svs.DEPTH, (pushes, why)


def an_in_place_transfer_loop_narrower_than_its_vector_keeps_the_order():
    # Shifted by two a trip moves 16 elements and reads 32: a push ahead would read what a pop writes.
    pushes, why = _schedule(_transfer_loop(2))
    assert pushes == 1 and why == ["a stage-1 write may share a buffer with a stage-0 read"], (pushes, why)


def a_write_into_a_later_trips_elements_keeps_the_order():
    pushes, why = _schedule(_loop(IN_PLACE_LOAD,
                                  "affine.vector_store %y, %m[%i + 16] : memref<256xf32>, vector<32xf32>"))
    assert pushes == 1 and why == ["a stage-1 write may share a buffer with a stage-0 read"], (pushes, why)


def a_read_wider_than_the_step_keeps_the_order():
    body = _loop("affine.vector_load %m[%i] : memref<256xf32>, vector<64xf32>", IN_PLACE_STORE)
    body = body.replace("%x = affine.vector_load %m[%i] : memref<256xf32>, vector<64xf32>",
                        "%w = affine.vector_load %m[%i] : memref<256xf32>, vector<64xf32>\n"
                        "    %x = vector.extract_strided_slice %w {offsets = [0], sizes = [32], strides = [1]}"
                        " : vector<64xf32> to vector<32xf32>")
    pushes, why = _schedule(body)
    assert pushes == 1 and why == ["a stage-1 write may share a buffer with a stage-0 read"], (pushes, why)


def a_push_that_reads_its_own_trips_pop_keeps_the_order():
    """exp of a log (lrn's pow): run ahead, the second push's pop would take the first push of a later trip."""
    body = _loop(IN_PLACE_LOAD, '''"pytorchsim_tpu.eup.push"(%y) : (vector<32xf32>) -> ()
    %z = "pytorchsim_tpu.eup.pop"() : () -> vector<32xf32>
    ''' + IN_PLACE_STORE.replace("%y,", "%z,"))
    pushes, why = _schedule(body, holds=2 * HOLDS)
    assert pushes == 2 and len(why) == 1 and "its own trip" in why[0], (pushes, why)


def an_eup_that_holds_one_register_less_keeps_the_order():
    pushes, why = _schedule(_loop(IN_PLACE_LOAD, IN_PLACE_STORE), holds=HOLDS - 1)
    assert pushes == 1 and why == [f"the EUP would hold {HOLDS} registers at a pop, more than the "
                                   f"{HOLDS - 1} it holds"], (pushes, why)


def a_machine_that_does_not_say_what_the_eup_holds_keeps_the_order():
    pushes, why = _schedule(_loop(IN_PLACE_LOAD, IN_PLACE_STORE), holds=None)
    assert pushes == 1 and len(why) == 1 and svs.QUEUE_KEY in why[0] and svs.LATENCY_KEY in why[0], why


def _bound(holds):
    """The LMUL the tpu target bounds a loop holding an f32 exp to (max_lmul), for an EUP of `holds`."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(f'''module {{ func.func @f(%m: memref<8192xf32>) {{
  affine.for %i = 0 to 8192 step 8192 {{
    %x = affine.vector_load %m[%i] : memref<8192xf32>, vector<8192xf32>
    %y = math.exp %x : vector<8192xf32>
    affine.vector_store %y, %m[%i] : memref<8192xf32>, vector<8192xf32>
  }} {{{VECTOR_LOOP}}}
  return
}} }}''')
        loop = next(o for o in walk(m.operation) if o.name == "affine.for")
        return lvf.eup_lmul(holds) if lvf.is_eup_loop(loop) else None


def the_widest_lmul_the_eup_holds_bounds_an_exp_loop():
    groups = (svs.DEPTH + 1) * svs.UNROLL
    assert _bound(groups * 4) == 4 and _bound(groups * 4 + groups - 1) == 4 and _bound(groups) == 1, \
        (_bound(groups * 4), _bound(groups))


def an_eup_that_holds_every_lmul_or_says_nothing_leaves_the_loop_alone():
    from pytorchsim_triton_compiler.driver.machine import ISA_MAX_LMUL
    groups = (svs.DEPTH + 1) * svs.UNROLL
    assert _bound(groups * ISA_MAX_LMUL) is None and _bound(None) is None and _bound(groups - 1) is None


class _Machine:
    """What lower_vector_functions reads of a subtarget: its VLEN and its vcix block."""
    def __init__(self, vcix):
        self.vlen_bits, self.vcix, self.vector_lanes = VLEN, vcix, 128


def _lowered_exp(vcix):
    """(pushes, SFU exps, why the function says the EUP is not used, or None) for one f32 exp."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse('''module { func.func @f(%x: vector<32xf32>) -> vector<32xf32> {
  %y = math.exp %x : vector<32xf32>
  return %y : vector<32xf32>
} }''')
        lvf.run(m, subtarget=_Machine(vcix))
        ops = list(walk(m.operation))
        why = [ir.StringAttr(o.attributes[svs.EUP_PIPELINE]).value for o in ops
               if o.name == "func.func" and svs.EUP_PIPELINE in o.attributes]
        return (sum(o.name == svs.PUSH for o in ops), sum(o.name == lvf.SFU_FUNCTIONS["math.exp"] for o in ops),
                why[0] if why else None)


def an_exp_goes_to_the_eup_on_a_machine_that_states_it():
    pushes, viv, why = _lowered_exp({svs.QUEUE_KEY: 16, svs.LATENCY_KEY: 10})
    assert (pushes, viv, why) == (1, 0, None), (pushes, viv, why)


def an_exp_stays_on_vcix_v_iv_on_a_machine_that_does_not_and_says_why():
    pushes, viv, why = _lowered_exp({})
    assert pushes == 0 and viv == 1 and why and svs.QUEUE_KEY in why, (pushes, viv, why)


def _exp_of_a_difference(b_in_loop):
    """The ops' names after lowering exp(a - b) in a loop, b made outside it or (`b_in_loop`) in each trip."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        make_b = "%b = affine.vector_load %m[%i] : memref<256xf32>, vector<32xf32>" if b_in_loop else ""
        m = ir.Module.parse(f'''module {{ func.func @f(%m: memref<256xf32>, %bo: vector<32xf32>) {{
  affine.for %i = 0 to 256 step 32 {{
    %a = affine.vector_load %m[%i] : memref<256xf32>, vector<32xf32>
    {make_b}
    %d = arith.subf %a, {"%b" if b_in_loop else "%bo"} : vector<32xf32>
    %y = math.exp %d : vector<32xf32>
    affine.vector_store %y, %m[%i] : memref<256xf32>, vector<32xf32>
  }}
  return
}} }}''')
        lvf.run(m, subtarget=_Machine({svs.QUEUE_KEY: 16, svs.LATENCY_KEY: 10}))
        loop = next(o for o in walk(m.operation) if o.name == "affine.for")
        return [o.name for o in walk(loop)], [o.name for o in walk(m.operation) if o not in walk(loop)]


def an_exp_of_a_difference_the_loop_does_not_make_is_one_fma_and_no_subtraction():
    """exp(a - b), b made before the loop: a * log2(e) - b * log2(e) in one fma a trip, b's product before the
    loop, the subtraction gone (a dead one read stage 0 and stopped the pipeline)."""
    inside, outside = _exp_of_a_difference(False)
    assert inside.count("math.fma") == 1 and "arith.subf" not in inside and "arith.mulf" not in inside, inside
    assert "arith.mulf" in outside and "arith.negf" in outside, outside


def an_exp_of_a_difference_the_loop_makes_keeps_the_subtraction_and_the_product():
    inside, _ = _exp_of_a_difference(True)
    assert "math.fma" not in inside and "arith.subf" in inside and "arith.mulf" in inside, inside


def an_exp_loop_storing_bf16_packs_it_and_is_still_pipelined():
    """An exp's pops cast to bf16 and stored go through the misc unit's pack as 32-bit words (no 16-bit vtype
    in the loop), and the pack's view of the bf16 buffer is that buffer to the pipeliner: the loop still
    pipelines, its pops' writes another buffer than its pushes' reads."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse('''module { func.func @f() {
  %a = memref.alloc() : memref<256xf32>
  %b = memref.alloc() : memref<256xbf16>
  %c3 = arith.constant 3 : i64
  %c0 = arith.constant 0 : i64
  %c32 = arith.constant 32 : i64
  affine.for %i = 0 to 256 step 32 {
    %x = affine.vector_load %a[%i] : memref<256xf32>, vector<32xf32>
    "pytorchsim_tpu.eup.push"(%x) : (vector<32xf32>) -> ()
    %y = "pytorchsim_tpu.eup.pop"() : () -> vector<32xf32>
    %t = arith.truncf %y : vector<32xf32> to vector<32xbf16>
    affine.vector_store %t, %b[%i] : memref<256xbf16>, vector<32xbf16>
  }
  return
} }''')
        for st in [o for o in walk(m.operation) if o.name == "affine.vector_store"]:
            lvf._rewrite_bf16_store(st)
        views = [o.opview for o in walk(m.operation)]
        assert not _any(views, arith.TruncFOp) and _any(views, pytorchsim_tpu.MiscPackOp), views
        svs.schedule(m, HOLDS, VLEN)
        ops = list(walk(m.operation))
        why = [ir.StringAttr(o.attributes[svs.EUP_PIPELINE]).value for o in ops
               if o.name == "affine.for" and svs.EUP_PIPELINE in o.attributes]
        assert not why, why
        assert sum(o.name == svs.PUSH for o in ops) > 1, "not pipelined"


def a_bf16_store_whose_halves_are_no_power_of_two_packs_them_at_their_vl():
    """Twelve elements pack as halves of six (vector_softmax_1x1536's twelve a lane): the pack takes
    them in a vector of eight, a width llc gives a vcix op, its vl six -- no truncf is left."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse('''module { func.func @f(%x: vector<12xf32>) {
  %b = memref.alloc() : memref<48xbf16>
  affine.for %i = 0 to 48 step 12 {
    %t = arith.truncf %x : vector<12xf32> to vector<12xbf16>
    affine.vector_store %t, %b[%i] : memref<48xbf16>, vector<12xbf16>
  }
  return
} }''')
        for st in [o for o in walk(m.operation) if o.name == "affine.vector_store"]:
            lvf._rewrite_bf16_store(st)
        ops = list(walk(m.operation))
        views = [o.opview for o in ops]
        assert not _any(views, arith.TruncFOp) and _any(views, pytorchsim_tpu.MiscPackOp), views
        pack = next(o for o in ops if isinstance(o.opview, pytorchsim_tpu.MiscPackOp))
        assert ir.VectorType(pack.results[0].type).shape == [8], pack
        vl = pack.operands[2].owner
        assert ir.IntegerAttr(vl.attributes["value"]).value == 6, vl


CASES = [an_exp_goes_to_the_eup_on_a_machine_that_states_it,
         an_exp_of_a_difference_the_loop_does_not_make_is_one_fma_and_no_subtraction,
         an_exp_of_a_difference_the_loop_makes_keeps_the_subtraction_and_the_product,
         an_exp_loop_storing_bf16_packs_it_and_is_still_pipelined,
         a_bf16_store_whose_halves_are_no_power_of_two_packs_them_at_their_vl,
         an_exp_stays_on_vcix_v_iv_on_a_machine_that_does_not_and_says_why,
         an_in_place_loop_whose_trips_touch_their_own_elements_is_pipelined,
         a_stage0_value_read_after_the_pop_is_read_again_in_stage1,
         an_in_place_transfer_loop_whose_trips_touch_their_own_elements_is_pipelined,
         an_in_place_transfer_loop_narrower_than_its_vector_keeps_the_order,
         a_write_into_a_later_trips_elements_keeps_the_order,
         a_read_wider_than_the_step_keeps_the_order,
         a_push_that_reads_its_own_trips_pop_keeps_the_order,
         an_eup_that_holds_one_register_less_keeps_the_order,
         a_machine_that_does_not_say_what_the_eup_holds_keeps_the_order,
         the_widest_lmul_the_eup_holds_bounds_an_exp_loop,
         an_eup_that_holds_every_lmul_or_says_nothing_leaves_the_loop_alone]


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
