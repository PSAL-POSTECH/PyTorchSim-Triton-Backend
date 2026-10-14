"""A run of synchronous TOG loads is issued in the order of the first TOG call that reads what each
writes (build_skeleton._order_loads_by_first_use): a B-resident GEMM loads B0, B1, B2, then A, and
its first compute reads A and B0 only, so A goes second, with the extent a masked load is sized by.
Two loads of one buffer, or an async load between them, keep the order as written.
    python3 tests/passes/tog_load_order_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.tog.passes import build_skeleton  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes import togsim_ops as ts  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes.build_tog import ir  # noqa: E402


_N = [0]


def _dma(write, is_async="false"):
    _N[0] += 1
    return (f'    "torchsim.dma"(%c0, %c0) {{dir = 0 : i32, is_async = {is_async}, '
            f'read_bufs = [], write_bufs = [{write} : i64]}} : (index, index) -> ()\n'
            f'    %k{_N[0]} = arith.constant {write} : index\n')


def _sized_dma(write):
    """A load whose extent is computed just before it, as _live_extents builds it."""
    _N[0] += 1
    return (f'    %e{_N[0]} = arith.minsi %c0, %c0 : index\n'
            f'    "torchsim.dma"(%c0, %c0, %e{_N[0]}) {{dir = 0 : i32, dyn_dims = [0 : i64], '
            f'is_async = false, read_bufs = [], write_bufs = [{write} : i64]}} '
            f': (index, index, index) -> ()\n')


def _compute(reads):
    bufs = ", ".join(f"{b} : i64" for b in reads)
    return f'    "torchsim.compute"() {{read_bufs = [{bufs}], write_bufs = []}} : () -> ()\n'


def _order(body):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse("func.func @kernel() {\n    %c0 = arith.constant 0 : index\n"
                                 + body + "    return\n}\n", ctx)
        block = build_skeleton._kernel_block(module)
        build_skeleton._order_loads_by_first_use(block)
        assert module.operation.verify()
        return [build_skeleton._attr_ints(o.operation, ts.ATTR_WRITE_BUFS)[0]
                for o in block.operations if o.operation.name == ts.DMA]


def the_load_the_first_compute_reads_goes_first():
    got = _order(_dma(1) + _dma(5) + _dma(7) + _dma(2) + _compute([1, 2]) + _compute([2, 5])
                 + _compute([2, 7]))
    assert got == [1, 2, 5, 7], got


def a_load_sized_at_run_time_moves_with_its_extent():
    got = _order(_sized_dma(1) + _sized_dma(5) + _sized_dma(7) + _sized_dma(2)
                 + _compute([1, 2]) + _compute([2, 5]) + _compute([2, 7]))
    assert got == [1, 2, 5, 7], got


def two_loads_of_one_buffer_keep_their_order():
    got = _order(_dma(1) + _dma(5) + _dma(5) + _dma(2) + _compute([1, 2]) + _compute([5]))
    assert got == [1, 5, 5, 2], got


def an_async_load_ends_the_run():
    got = _order(_dma(1) + _dma(5) + _dma(9, "true") + _dma(7) + _dma(2) + _compute([1, 2])
                 + _compute([7, 5, 9]))
    assert got == [1, 5, 9, 2, 7], got


CASES = [the_load_the_first_compute_reads_goes_first, a_load_sized_at_run_time_moves_with_its_extent,
         two_loads_of_one_buffer_keep_their_order,
         an_async_load_ends_the_run]


def main():
    failed = 0
    for case in CASES:
        try:
            case()
            ok, why = True, ""
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{case.__name__:48} {'PASS' if ok else 'FAIL'}  {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
