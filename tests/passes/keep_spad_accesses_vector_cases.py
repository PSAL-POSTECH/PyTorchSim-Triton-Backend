"""tritonpytorchsim-keep-spad-accesses-vector-tpuv6e: a vector load/store on !llvm.ptr<1> leaves as
llvm.vp.load/vp.store keeping its alignment, mask and location; each refused class by name.
    python3 tests/passes/keep_spad_accesses_vector_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.dialects import llvm  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import keep_spad_accesses_vector as K  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lower, p13_lower_to_llvm  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import SpadAccessRefused  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

T = machine.target()


def _module(body, args="%p: !llvm.ptr<1>, %q: !llvm.ptr<1>"):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        return ctx, ir.Module.parse(f"llvm.func @f({args}) {{\n{body}\n  llvm.return\n}}\n")


def _ran(body, **kw):
    ctx, m = _module(body, **kw)
    with ctx, ir.Location.unknown():
        K.run(m)
    return ctx, m


def _ops(m, cls):
    return [o for o in walk(m.operation) if isinstance(o.opview, cls)]


def _intrinsic(op):
    return ir.StringAttr(op.attributes["intrin"]).value


def _align(op, i):
    d = ir.ArrayAttr(op.attributes["arg_attrs"])[i]
    return ir.IntegerAttr(ir.DictAttr(d)["llvm.align"]).value if "llvm.align" in d else None


def _constant(v):
    return v.owner.attributes["value"]


def _refused(body, **kw):
    ctx, m = _module(body, **kw)
    with ctx, ir.Location.unknown():
        try:
            K.run(m)
        except SpadAccessRefused as e:
            return e.reason
    raise AssertionError("not refused")


COPY = ('  %v = llvm.load %p {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xf32> loc("k.py":7:3)\n'
        '  llvm.store %v, %q {alignment = 4 : i64} : vector<2xf32>, !llvm.ptr<1> loc("k.py":8:5)')


def a_vector_copy_on_the_scratchpad_becomes_vp_load_and_store():
    ctx, m = _ran(COPY)
    with ctx, ir.Location.unknown():
        calls = _ops(m, llvm.CallIntrinsicOp)
        assert [_intrinsic(c) for c in calls] == ["llvm.vp.load", "llvm.vp.store"], calls
        assert not _ops(m, llvm.LoadOp) and not _ops(m, llvm.StoreOp)
        load, store = calls
        assert load.results[0].type == ir.VectorType.get([2], ir.F32Type.get())
        assert store.operands[0] == load.results[0] and len(store.results) == 0
        assert load.operands[0] == m.body.operations[0].regions[0].blocks[0].arguments[0]


def the_alignment_rides_on_the_pointer_argument():
    ctx, m = _ran(COPY)
    with ctx, ir.Location.unknown():
        load, store = _ops(m, llvm.CallIntrinsicOp)
        assert [_align(load, i) for i in range(3)] == [8, None, None]
        assert [_align(store, i) for i in range(4)] == [None, 4, None, None]


def an_access_with_no_alignment_gets_no_llvm_align():
    ctx, m = _ran("  %v = llvm.load %p : !llvm.ptr<1> -> vector<4xi8>")
    with ctx, ir.Location.unknown():
        (load,) = _ops(m, llvm.CallIntrinsicOp)
        assert [_align(load, i) for i in range(3)] == [None, None, None]


def every_new_op_keeps_the_source_location():
    ctx, m = _ran(COPY)
    with ctx, ir.Location.unknown():
        load, store = _ops(m, llvm.CallIntrinsicOp)
        for call, (line, col) in ((load, (7, 3)), (store, (8, 5))):
            want = ir.Location.file("k.py", line, col)
            made = [call] + [v.owner for v in list(call.operands)[-2:]]
            assert all(o.location == want for o in made), [str(o.location) for o in made]


def the_mask_is_all_true_and_the_evl_is_the_element_count():
    for n, elem in ((1, "f32"), (2, "bf16"), (15, "f32"), (64, "i8")):
        ctx, m = _ran(f"  %v = llvm.load %p : !llvm.ptr<1> -> vector<{n}x{elem}>\n"
                      f"  llvm.store %v, %q : vector<{n}x{elem}>, !llvm.ptr<1>")
        with ctx, ir.Location.unknown():
            i1, i32 = ir.IntegerType.get_signless(1), ir.IntegerType.get_signless(32)
            for call in _ops(m, llvm.CallIntrinsicOp):
                mask, evl = list(call.operands)[-2:]
                assert mask.type == ir.VectorType.get([n], i1)
                dense = ir.DenseElementsAttr(_constant(mask))
                assert dense.is_splat and ir.IntegerAttr(dense.get_splat_value()).value != 0
                assert evl.type == i32 and ir.IntegerAttr(_constant(evl)).value == n


def scalar_accesses_and_other_space_accesses_are_left_alone():
    """A scalar store on the scratchpad is p05's to make vector<1> and p13b's to refuse, not this."""
    body = ("  %s = llvm.load %p {alignment = 4 : i64} : !llvm.ptr<1> -> f32\n"
            "  llvm.store %s, %q : f32, !llvm.ptr<1>\n"
            "  llvm.store %s, %d : f32, !llvm.ptr\n"
            "  %v = llvm.load %d : !llvm.ptr -> vector<2xf32>\n"
            "  llvm.store %v, %d : vector<2xf32>, !llvm.ptr\n"
            "  llvm.store %p, %d : !llvm.ptr<1>, !llvm.ptr")
    ctx, m = _ran(body, args="%p: !llvm.ptr<1>, %q: !llvm.ptr<1>, %d: !llvm.ptr")
    with ctx, ir.Location.unknown():
        assert not _ops(m, llvm.CallIntrinsicOp)
        assert len(_ops(m, llvm.LoadOp)) == 2 and len(_ops(m, llvm.StoreOp)) == 4


def a_runtime_masked_access_keeps_its_mask_under_vp():
    """llc may expand a masked load/store into a branch per element and scalar accesses; under vp it
    keeps its own mask, EVL the element count, and a load's off lanes from its passthru."""
    body = ("  %z = llvm.mlir.zero : vector<4xf32>\n"
            "  %v = llvm.intr.masked.load %p, %k, %z {alignment = 4 : i32} : "
            "(!llvm.ptr<1>, vector<4xi1>, vector<4xf32>) -> vector<4xf32>\n"
            "  llvm.intr.masked.store %v, %q, %k {alignment = 4 : i32} : "
            "vector<4xf32>, vector<4xi1> into !llvm.ptr<1>")
    ctx, m = _ran(body, args="%p: !llvm.ptr<1>, %q: !llvm.ptr<1>, %k: vector<4xi1>")
    with ctx, ir.Location.unknown():
        assert not _ops(m, llvm.MaskedLoadOp) and not _ops(m, llvm.MaskedStoreOp)
        load, store = _ops(m, llvm.CallIntrinsicOp)
        k = m.body.operations[0].regions[0].blocks[0].arguments[2]
        assert [_intrinsic(load), _intrinsic(store)] == ["llvm.vp.load", "llvm.vp.store"]
        assert load.operands[1] == k and store.operands[2] == k
        assert ir.IntegerAttr(_constant(load.operands[2])).value == 4
        sel = _ops(m, llvm.SelectOp)[0].opview
        assert sel.condition == k and sel.trueValue == load.results[0]
        assert isinstance(sel.falseValue.owner.opview, llvm.ZeroOp) and store.operands[0] == sel.result
        assert _align(load, 0) == 4 and _align(store, 1) == 4


def a_strided_access_with_a_nonzero_stride_is_left_alone():
    body = ("  %s = llvm.mlir.constant(8 : i64) : i64\n"
            "  %k = llvm.mlir.constant(dense<true> : vector<4xi1>) : vector<4xi1>\n"
            "  %n = llvm.mlir.constant(4 : i32) : i32\n"
            '  %v = "llvm.intr.experimental.vp.strided.load"(%p, %s, %k, %n) : '
            "(!llvm.ptr<1>, i64, vector<4xi1>, i32) -> vector<4xf32>")
    ctx, m = _ran(body)
    with ctx, ir.Location.unknown():
        assert len(_ops(m, llvm.VPStridedLoadOp)) == 1 and not _ops(m, llvm.CallIntrinsicOp)


def each_refused_class_is_refused_by_name():
    gather = ("  %v = llvm.intr.masked.gather %g, %k, %z {alignment = 4 : i32} : "
              "(vector<2x!llvm.ptr<1>>, vector<2xi1>, vector<2xf32>) -> vector<2xf32>")
    cases = [
        (K.SUB_BYTE, "  %v = llvm.load %p : !llvm.ptr<1> -> vector<16xi1>", {}),
        (K.SCALABLE, "  %v = llvm.load %p : !llvm.ptr<1> -> vector<[4]xf32>", {}),
        (K.INDEXED, "  %z = llvm.mlir.zero : vector<2xf32>\n" + gather,
         {"args": "%g: vector<2x!llvm.ptr<1>>, %k: vector<2xi1>"}),
        (K.ALL_TRUE_MASK, "  %k = llvm.mlir.constant(dense<true> : vector<4xi1>) : vector<4xi1>\n"
                          "  %z = llvm.mlir.zero : vector<4xf32>\n"
                          "  %v = llvm.intr.masked.load %p, %k, %z {alignment = 4 : i32} : "
                          "(!llvm.ptr<1>, vector<4xi1>, vector<4xf32>) -> vector<4xf32>", {}),
        (K.MEMORY_INTRINSIC, "  %n = llvm.mlir.constant(8 : i64) : i64\n"
                             "  \"llvm.intr.memcpy\"(%p, %q, %n) <{isVolatile = false}> : "
                             "(!llvm.ptr<1>, !llvm.ptr<1>, i64) -> ()", {}),
        (K.VOLATILE, "  %v = llvm.load volatile %p : !llvm.ptr<1> -> vector<2xf32>", {}),
        (K.NONTEMPORAL, "  %v = llvm.mlir.zero : vector<2xf32>\n"
                        "  llvm.store %v, %q {nontemporal} : vector<2xf32>, !llvm.ptr<1>", {}),
        (K.ZERO_STRIDE, "  %s = llvm.mlir.constant(0 : i64) : i64\n"
                        "  %k = llvm.mlir.constant(dense<true> : vector<4xi1>) : vector<4xi1>\n"
                        "  %n = llvm.mlir.constant(4 : i32) : i32\n"
                        "  %v = llvm.mlir.zero : vector<4xf32>\n"
                        '  "llvm.intr.experimental.vp.strided.store"(%v, %q, %s, %k, %n) : '
                        "(vector<4xf32>, !llvm.ptr<1>, i64, vector<4xi1>, i32) -> ()", {}),
    ]
    for want, body, kw in cases:
        got = _refused(body, **kw)
        assert got is want, (want, got)


def a_refusal_rewrites_nothing():
    ctx, m = _module(COPY + "\n  %w = llvm.load volatile %p : !llvm.ptr<1> -> vector<2xf32>")
    with ctx, ir.Location.unknown():
        try:
            K.run(m)
        except SpadAccessRefused:
            pass
        assert not _ops(m, llvm.CallIntrinsicOp) and len(_ops(m, llvm.LoadOp)) == 2


def the_postcondition_refuses_a_vector_access_the_rewrite_left():
    kept = K._to_vp
    K._to_vp = lambda op: None
    try:
        assert _refused(COPY) is K.SURVIVED
    finally:
        K._to_vp = kept


def the_pass_is_the_last_rewrite_in_to_llvm_on_tpuv6e():
    """Only the lane-values check, which rewrites nothing, reads TO_LLVM's IR after it."""
    assert lower.TO_LLVM.index(lower.KEEP_SPAD_ACCESSES_VECTOR) == \
        lower.TO_LLVM.index(p13_lower_to_llvm) + 1, lower.TO_LLVM
    got = lower.resolve(lower.TO_LLVM, T)
    assert got[-1] is lower.p13b_check_lane_values_stay_vector, got
    assert got[-2] is K and got[-3] is p13_lower_to_llvm, got


CASES = [a_vector_copy_on_the_scratchpad_becomes_vp_load_and_store,
         the_alignment_rides_on_the_pointer_argument,
         an_access_with_no_alignment_gets_no_llvm_align,
         every_new_op_keeps_the_source_location,
         the_mask_is_all_true_and_the_evl_is_the_element_count,
         scalar_accesses_and_other_space_accesses_are_left_alone,
         a_runtime_masked_access_keeps_its_mask_under_vp,
         a_strided_access_with_a_nonzero_stride_is_left_alone,
         each_refused_class_is_refused_by_name,
         a_refusal_rewrites_nothing,
         the_postcondition_refuses_a_vector_access_the_rewrite_left,
         the_pass_is_the_last_rewrite_in_to_llvm_on_tpuv6e]

if __name__ == "__main__":
    bad = 0
    for c in CASES:
        try:
            c()
            print(f"ok   {c.__name__}")
        except Exception as e:
            bad += 1
            print(f"FAIL {c.__name__}: {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1 if bad else 0)
