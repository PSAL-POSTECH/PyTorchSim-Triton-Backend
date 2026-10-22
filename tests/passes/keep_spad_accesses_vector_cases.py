"""tritonpytorchsim-keep-spad-accesses-vector-tpuv6e: a vector load/store on !llvm.ptr<1> leaves as
llvm.vp.load/vp.store keeping its alignment and location; each refused class is refused by name.
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


def scalar_and_other_space_accesses_are_left_alone():
    body = ("  %s = llvm.load %p {alignment = 4 : i64} : !llvm.ptr<1> -> f32\n"
            "  llvm.store %s, %q : f32, !llvm.ptr<1>\n"
            "  %v = llvm.load %d : !llvm.ptr -> vector<2xf32>\n"
            "  llvm.store %v, %d : vector<2xf32>, !llvm.ptr\n"
            "  llvm.store %p, %d : !llvm.ptr<1>, !llvm.ptr")
    ctx, m = _ran(body, args="%p: !llvm.ptr<1>, %q: !llvm.ptr<1>, %d: !llvm.ptr")
    with ctx, ir.Location.unknown():
        assert not _ops(m, llvm.CallIntrinsicOp)
        assert len(_ops(m, llvm.LoadOp)) == 2 and len(_ops(m, llvm.StoreOp)) == 3


def a_runtime_masked_access_is_left_alone():
    body = ("  %z = llvm.mlir.zero : vector<4xf32>\n"
            "  %v = llvm.intr.masked.load %p, %k, %z {alignment = 4 : i32} : "
            "(!llvm.ptr<1>, vector<4xi1>, vector<4xf32>) -> vector<4xf32>\n"
            "  llvm.intr.masked.store %v, %q, %k {alignment = 4 : i32} : "
            "vector<4xf32>, vector<4xi1> into !llvm.ptr<1>")
    ctx, m = _ran(body, args="%p: !llvm.ptr<1>, %q: !llvm.ptr<1>, %k: vector<4xi1>")
    with ctx, ir.Location.unknown():
        assert len(_ops(m, llvm.MaskedLoadOp)) == 1 and len(_ops(m, llvm.MaskedStoreOp)) == 1
        assert not _ops(m, llvm.CallIntrinsicOp)


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


def the_pass_runs_last_in_to_llvm_on_tpuv6e():
    assert lower.TO_LLVM.index(lower.KEEP_SPAD_ACCESSES_VECTOR) == \
        lower.TO_LLVM.index(p13_lower_to_llvm) + 1, lower.TO_LLVM
    got = lower.resolve(lower.TO_LLVM, T)
    assert got[-1] is K and got[-2] is p13_lower_to_llvm, got


CASES = [a_vector_copy_on_the_scratchpad_becomes_vp_load_and_store,
         the_alignment_rides_on_the_pointer_argument,
         an_access_with_no_alignment_gets_no_llvm_align,
         every_new_op_keeps_the_source_location,
         the_mask_is_all_true_and_the_evl_is_the_element_count,
         scalar_and_other_space_accesses_are_left_alone,
         a_runtime_masked_access_is_left_alone,
         each_refused_class_is_refused_by_name,
         a_refusal_rewrites_nothing,
         the_postcondition_refuses_a_vector_access_the_rewrite_left,
         the_pass_runs_last_in_to_llvm_on_tpuv6e]

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
