"""Every refusal a wait or a trace barrier gets before a trace runs, each on the smallest IR that
earns it, and one wait that earns none: a wait or barrier of no elements, a barrier stating no
count, a semaphore a load and a store share, a wait whose amount is not a constant.
    python3 tests/runtime/wait_refusals.py        exit 0 iff every case gets what it names"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytorchsim_triton_compiler  # noqa: E402,F401
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes import build_skeleton  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes import togsim_ops as ts  # noqa: E402


def _kernel(body, check):
    """`check(module, built)` on a module of one func whose body `body(args, index)` fills."""
    from mlir.dialects import arith, func
    from mlir.ir import (Context, F32Type, IndexType, InsertionPoint, IntegerAttr, Location,
                         MemRefType, Module)
    with Context(), Location.unknown():
        m = Module.create()
        f32, idx = F32Type.get(), IndexType.get()
        dram = MemRefType.get([64], f32)
        sram = MemRefType.get([8, 8], f32, memory_space=IntegerAttr.get(idx, 1))
        with InsertionPoint(m.body):
            fn = func.FuncOp("kernel", ([dram, sram, idx], []))
        blk = fn.add_entry_block()
        with InsertionPoint(blk):
            def index(n):
                return arith.ConstantOp(idx, IntegerAttr.get(idx, n)).result
            built = body(list(blk.arguments), index)
            func.ReturnOp([])
        return check(m, built)


def _transfer(args, index, kind, sem):
    return torchsim.TransferOp(None, args[0], index(0), args[1], index(0), sem, [], [], kind, 0,
                               [8, 1], async_=True)


def _barrier(sem, expected):
    from mlir.ir import IntegerAttr, IntegerType, Operation
    attrs = {} if expected is None else {
        ts.ATTR_EXPECTED: IntegerAttr.get(IntegerType.get_signless(64), expected)}
    return Operation.create(ts.MEMORY_BAR, operands=[sem], attributes=attrs)


def _verify(module):
    from mlir.ir import MLIRError
    try:
        module.operation.verify()
    except MLIRError as e:
        return str(e)
    return None


def _skeleton_wait(module, wait):
    try:
        build_skeleton._emit_one_wait(module.context, wait)
    except RuntimeError as e:
        return str(e)
    return None


def _verified_then_traced(m, wait):
    return _verify(m) or _skeleton_wait(m, wait)


def wait_of_no_elements():
    return _kernel(lambda a, i: torchsim.WaitOp(i(0), i(0)), lambda m, _: _verify(m))


def barrier_of_no_elements():
    return _kernel(lambda a, i: _barrier(i(0), 0), lambda m, _: _verify(m))


def barrier_stating_no_count():
    return _kernel(lambda a, i: _barrier(i(0), None), lambda m, _: _verify(m))


def load_and_store_on_one_sem():
    def body(a, i):
        sem = i(0)
        _transfer(a, i, "MVIN", sem)
        _transfer(a, i, "MVOUT", sem)
        return torchsim.WaitOp(sem, i(64))
    return _kernel(body, _verified_then_traced)


def amount_not_a_constant():
    def body(a, i):
        sem = i(0)
        _transfer(a, i, "MVIN", sem)
        return torchsim.WaitOp(sem, a[2])
    return _kernel(body, _verified_then_traced)


def one_load_waited():
    def body(a, i):
        sem = i(0)
        _transfer(a, i, "MVIN", sem)
        return torchsim.WaitOp(sem, i(64))
    return _kernel(body, _verified_then_traced)


CASES = {
    one_load_waited: None,
    wait_of_no_elements: "waits for 0 elements",
    barrier_of_no_elements: "expects 0 elements",
    barrier_stating_no_count: "requires attribute 'expected'",
    load_and_store_on_one_sem: "signalled by a load and a store at once",
    amount_not_a_constant: "amount is not a constant",
}


def main():
    failed = 0
    for case, want in CASES.items():
        got = case()
        ok = (got is None) if want is None else (got is not None and want in got)
        failed += not ok
        shown = " ".join(got.split()) if got else "no refusal"
        print(f"{case.__name__:28} {'PASS' if ok else 'FAIL'}  {shown[:150]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
