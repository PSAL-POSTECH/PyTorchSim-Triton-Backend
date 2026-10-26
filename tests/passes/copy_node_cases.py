"""The resident mm's rule (contract/copy_node.plan, budget), the passes that find its copy node (bufferize
marks the dots' operands and reductions' inputs, c19 takes a whole-argument load at the kernel's top into one,
held alone, of a tensor contract/copy_node.resident_fits holds).
python3 tests/passes/copy_node_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract import copy_node as cn  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import a06_bufferize  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import c19_plan_copy_node as c19  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.target_kind import target_kind  # noqa: E402

V6E = dict(in_bytes=2, acc_bytes=4, out_bytes=2, max_numel=1 << 26)


def _v6e():
    """The tpuv6e machine file as TOGSim's configs/ states it."""
    from pytorchsim_triton_compiler.contract import target
    return target.load("tpuv6e")


def _plan(M, K, N):
    return cn.plan(M, N, K, target=_v6e(), **V6E)


def the_larger_operand_is_held():
    got = _plan(4096, 4096, 4096 * 2)
    assert got.resident == 1 and got.tile >= 128, got


def a_tie_holds_a():
    """Breaks if a tie holds B: XLA's comparator ends on the buffer id, A's first (mm_wT, bmm_eq)."""
    assert _plan(4096, 4096, 4096).resident == 0


def the_smaller_is_held_when_the_larger_fills_the_scratchpad():
    """As v6e XLA does where one operand is the whole 128 MiB of VMEM. A buffer counts as the compiler cuts
    it: 8192x8192x5120's B rides a 8192x8192 block, cut to 8192x5120, so its 80 MiB is held as XLA holds it;
    192 columns stay a 256 block."""
    assert _plan(8192, 8192, 4096).resident == 1
    assert _plan(4096, 8192, 8192).resident == 0
    assert _plan(8192, 8192, 5120).resident == 1
    assert cn.fitted(5120, 128) == 5120 and cn.fitted(192, 128) == 256


def both_filling_the_scratchpad_hold_none_and_say_why():
    got = _plan(8192, 8192, 8192)
    assert got.resident is None and "A does not fit" in got.why and "B does not fit" in got.why, got


def a_vector_product_holds_none():
    assert _plan(1, 4096, 4096).resident is None


def the_tile_is_the_largest_that_fits_beside_the_held_operand():
    got = _plan(16384, 4096, 4096)
    assert got.resident == 1
    assert cn.fits(1, got.tile, 16384, 4096, 4096, target=_v6e(), **V6E)
    assert not cn.fits(1, got.tile * 2, 16384, 4096, 4096, target=_v6e(), **V6E)


def the_tile_walks_no_padded_tail_where_a_smaller_one_divides_the_extent():
    """Breaks if a tail tile padded past the data is taken (it is computed whole): N = 3072 streamed by 2048
    walks 4096, by 1024 exactly 3072; N = 768 by 512 walks 1024, by 256 exactly 768."""
    assert _plan(4096, 2048, 3072).tile == 1024, _plan(4096, 2048, 3072)
    assert _plan(8192, 3072, 768).tile == 256, _plan(8192, 3072, 768)


def no_tile_is_narrower_than_the_matrix_unit():
    """Breaks if a tile narrower than Target.matmul_width (256 on tpuv6e) is taken for a tail it avoids: N =
    1152 by 128 is 9 pushes of a 256-wide unit, by 256 five (the last half full)."""
    assert _plan(4096, 384, 1152).tile == 256, _plan(4096, 384, 1152)


def an_elf_that_holds_takes_the_whole_bank():
    assert cn.budget(1 << 20, 2, True) == 1 << 20 and cn.budget(1 << 20, 2, False) == 1 << 19


_MOD = """
#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>
module {{
  func.func @k(%a: memref<{na}xbf16>, %b: memref<1024xbf16>, %c: memref<1024xf32>) {{
    %c0 = arith.constant 0 : index
    %ta = memref.alloc() : memref<32x32xbf16, 1>
    %tb = memref.alloc() : memref<32x32xbf16, 1>
    %acc = memref.alloc() : memref<32x32xf32, 1>
    "pytorchsim.transfer"(%b, %c0, %tb, %c0, %c0) <{{direction = "load", dram_arg = 1 : i64, dram_stride = [1, 32],
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
        : (memref<1024xbf16>, index, memref<32x32xbf16, 1>, index, index) -> ()
    {pre}
    affine.for %i = 0 to 2 {{
      "pytorchsim.transfer"(%a, %c0, %ta, %c0, %c0) <{{direction = "load", dram_arg = 0 : i64, dram_stride = [1, 32],
          operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
          : (memref<{na}xbf16>, index, memref<32x32xbf16, 1>, index, index) -> ()
      "pytorchsim.dot"(%ta, %tb, %acc) <{{indexing_maps = [#map, #map1, #map2]}}>
          : (memref<32x32xbf16, 1>, memref<32x32xbf16, 1>, memref<32x32xf32, 1>) -> ()
      "pytorchsim.transfer"(%c, %c0, %acc, %c0, %c0) <{{direction = "store", dram_arg = 2 : i64, dram_stride = [1, 32],
          operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
          : (memref<1024xf32>, index, memref<32x32xf32, 1>, index, index) -> ()
    }}
    return
  }}
}}
"""


_TENSORS = [((64, 32), 2), ((32, 32), 2), ((32, 32), 4)]
"""_MOD's arguments as the kernel's spec has them, (shape, element bytes) each."""


def _target():
    from pytorchsim_triton_compiler import machine
    return machine.target()


def _stamped(na=2048, pre=""):
    from mlir.ir import ArrayAttr, DictAttr, IntegerAttr, Module, StringAttr
    module = Module.parse(_MOD.format(na=na, pre=pre))
    a06_bufferize.mark_prefetch_operands(module)
    c19.run(module, arg_tensors=_TENSORS)
    got = DictAttr(module.operation.attributes[c19.COPY_NODE])
    return [IntegerAttr(a).value for a in ArrayAttr(got["args"])], StringAttr(got["why"]).value


def the_whole_load_at_the_top_is_the_copy_node():
    args, why = _stamped()
    assert args == [1] and "arg 0: loaded inside a loop" in why, (args, why)


def a_load_of_part_of_its_argument_is_none():
    from mlir.ir import Module
    module = Module.parse(_MOD.format(na=2048, pre="").replace("memref<1024xbf16>", "memref<4096xbf16>"))
    a06_bufferize.mark_prefetch_operands(module)
    got = c19.plan(module, target_kind(), _target(), _TENSORS)
    assert got[0] == [], got


def two_whole_loads_make_one_copy_node():
    """Breaks if a kernel whose streamed operand fits one tile (loaded whole at the top too) gets a copy node
    for each: XLA prefetches one operand, so the second load would be timed as a second exposed copy. A tie
    holds A, the lower buffer id; more use elements, then an earlier latest use, go first (prefetch_order)."""
    from mlir.ir import ArrayAttr, DictAttr, IntegerAttr, Module
    a_in_loop = ('      "pytorchsim.transfer"(%a, %c0, %ta, %c0, %c0) <{{direction = "load", dram_arg = 0 : i64, '
                 'dram_stride = [1, 32],\n          operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>\n'
                 '          : (memref<{na}xbf16>, index, memref<32x32xbf16, 1>, index, index) -> ()\n')
    assert a_in_loop in _MOD
    text = _MOD.replace(a_in_loop, "").format(na=1024, pre=(
        '"pytorchsim.transfer"(%a, %c0, %ta, %c0, %c0) <{direction = "load", dram_arg = 0 : i64, dram_stride = [1, 32], '
        'operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}> : '
        '(memref<1024xbf16>, index, memref<32x32xbf16, 1>, index, index) -> ()'))
    module = Module.parse(text)
    a06_bufferize.mark_prefetch_operands(module)
    c19.run(module, arg_tensors=_TENSORS)
    got = DictAttr(module.operation.attributes[c19.COPY_NODE])
    args = [IntegerAttr(a).value for a in ArrayAttr(got["args"])]
    assert args == [0], args
    assert cn.order({0: 4, 1: 4}) == [0, 1] and cn.order({0: 4, 1: 8}) == [1, 0]
    c = cn.Candidate
    assert cn.prefetch_order({0: c(8, 1, 5, 0), 1: c(8, 2, 9, 1)}) == [1, 0]
    assert cn.prefetch_order({0: c(8, 2, 9, 0), 1: c(8, 2, 5, 1)}) == [1, 0]


def an_argument_moved_twice_is_none():
    again = ('"pytorchsim.transfer"(%b, %c0, %tb, %c0, %c0) <{direction = "load", dram_arg = 1 : i64, '
             'dram_stride = [1, 32], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}> : '
             '(memref<1024xbf16>, index, memref<32x32xbf16, 1>, index, index) -> ()')
    args, why = _stamped(pre=again)
    assert args == [] and "moved by more than one transfer" in why, (args, why)


def the_plan_reads_the_marks_not_the_dot():
    """Breaks if plan-copy-node looks for the pytorchsim.dot, which the kind has lowered by its turn: the
    dot erased after the marking leaves the same copy node, and a module bufferize marked nothing in holds
    none."""
    from mlir.ir import ArrayAttr, DictAttr, IntegerAttr, Module
    from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk
    module = Module.parse(_MOD.format(na=2048, pre=""))
    a06_bufferize.mark_prefetch_operands(module)
    next(op for op in walk(module.operation) if op.name == "pytorchsim.dot").erase()
    c19.run(module, arg_tensors=_TENSORS)
    got = DictAttr(module.operation.attributes[c19.COPY_NODE])
    assert [IntegerAttr(a).value for a in ArrayAttr(got["args"])] == [1], got
    bare = Module.parse(_MOD.format(na=2048, pre=""))
    c19.run(bare, arg_tensors=_TENSORS)
    got = c19.plan(bare, target_kind(), _target(), _TENSORS)
    assert got == ([], "no matmul operand or reduction input is loaded from an argument"), got


def a_target_whose_graph_prefetches_nothing_holds_none():
    """Breaks if plan-copy-node holds an operand where copy_node.graph_prefetches, by the target's kind
    (PREFETCHES), says the graph holds none, and says so."""
    import copy
    from mlir.ir import Module
    t = copy.copy(_target())
    t.isa = [e for e in t.isa if not e.startswith("pytorchsim_")]
    module = Module.parse(_MOD.format(na=2048, pre=""))
    a06_bufferize.mark_prefetch_operands(module)
    args, why = c19.plan(module, target_kind(), t, _TENSORS)
    assert args == [] and "arg 1: the graph holds no resident operand on this target" in why, why
    assert cn.graph_prefetches(t) == 0 and cn.graph_prefetches(_target()) == cn.PREFETCHES["tpu_kind"] == 1


def the_contract_loads_no_compiler():
    """Breaks if contract/copy_node, which the front end calls in a process that has Triton's LLVM loaded,
    imports lowering/ or the MLIR bindings (a second LLVM aborts on its options): a fresh interpreter."""
    import subprocess
    code = ("import sys; from pytorchsim_triton_compiler.contract import copy_node, target; "
            "copy_node.graph_prefetches(target.load('tpuv6e')); "
            "print([m for m in sys.modules if m.split('.')[0] == 'mlir' or '.lowering' in m])")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    got = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert got.returncode == 0 and got.stdout.strip() == "[]", (got.stdout, got.stderr[-400:])


def a_tensor_the_bank_cannot_hold_whole_is_not_held_and_says_why():
    """Breaks if plan-copy-node holds a tensor contract/copy_node.resident_fits refuses, which the front end
    elects by too, or drops it without a word: B as 8192 x 8192 bf16 takes 1 MiB a lane, the whole bank."""
    from mlir.ir import Module
    module = Module.parse(_MOD.format(na=2048, pre=""))
    a06_bufferize.mark_prefetch_operands(module)
    big = [_TENSORS[0], ((8192, 8192 + 128), 2), _TENSORS[2]]
    args, why = c19.plan(module, target_kind(), _target(), big)
    assert args == [] and "arg 1: a [8192, 8320] tensor of 2-byte elements takes" in why, why
    assert cn.resident_fits(_target(), (32, 32), 2) and not cn.resident_fits(_target(), (8192, 8320), 2)


def a_held_buffer_another_op_writes_is_not_held_and_says_why():
    """Breaks if a whole load whose buffer another op writes is held (the TOG would read the copy node's
    data where the kernel wrote its own) or is dropped without a word."""
    store = ('%z = arith.constant 0.0 : bf16\n'
             '    memref.store %z, %tb[%c0, %c0] : memref<32x32xbf16, 1>')
    args, why = _stamped(pre=store)
    assert args == [] and "arg 1: its buffer is written by memref.store" in why, (args, why)


def a_held_load_sharing_its_semaphore_is_not_held_and_says_why():
    """Breaks if a whole load whose semaphore another async transfer signals is held: the TOG leaves the held
    load out, and the other's wait would wait for it."""
    from mlir.ir import Module
    text = _MOD.replace('direction = "load"', 'async = true, direction = "load"').format(na=2048, pre="")
    module = Module.parse(text)
    a06_bufferize.mark_prefetch_operands(module)
    args, why = c19.plan(module, target_kind(), _target(), _TENSORS)
    assert args == [] and "arg 1: its semaphore 0 is signalled by another transfer too" in why, (args, why)


def the_resident_budget_keeps_the_machine_s_scoped_share():
    """Breaks if the resident rule ignores compiler.spad_scoped_kb_per_lane: with v6e XLA's 32 MiB of 128 kept
    for the kernels, a 96 MiB weight is held and a 112 MiB one is not (run/llo_probe/xprog probe_w96m,
    w112m)."""
    import copy
    t = copy.copy(_target())
    assert t.spad_size == 1 << 20 and cn.resident_budget(t) == t.spad_size - t.spad_scoped
    t.spad_scoped = 256 << 10
    assert cn.resident_fits(t, (4096, 12288), 2) and not cn.resident_fits(t, (4096, 14336), 2)


def the_v6e_machine_keeps_xla_s_scoped_share():
    """Breaks if the tpuv6e machine file stops stating compiler.spad_scoped_kb_per_lane (XLA's 32 MiB of 128)
    or the GEMM plan budgets by another rule than resident_fits: a graph holds 96 MiB, so of 256 x 4096 x
    13312 the plan holds A as XLA does (probe_n13312), B's 104 MiB being over it."""
    t = _v6e()
    assert cn.resident_budget(t) * t.vector_lanes == 96 << 20, cn.resident_budget(t)
    assert not cn.resident_fits(t, (4096, 13312), 2) and cn.resident_fits(t, (256, 4096), 2)
    assert _plan(256, 4096, 13312).resident == 0, _plan(256, 4096, 13312)


def the_gemm_plan_and_the_resident_rule_count_a_lane_alike():
    """Breaks if the GEMM plan counts a held tile's lane bytes otherwise than resident_fits: an operand the
    plan holds is one plan-copy-node holds."""
    t = _target()
    assert cn.tile_lane_bytes((8192, 5120), 2, 128) == cn.resident_lane_bytes(t, (8192, 5120), 2) == 655360
    assert cn.resident_fits(t, (8192, 5120), 2) and _plan(8192, 8192, 5120).resident == 1


_REDUCE = """
module {
  func.func @k(%x: memref<1024xf32>, %y: memref<32xf32>) {
    %c0 = arith.constant 0 : index
    %tx = memref.alloc() : memref<32x32xf32, 1>
    %ty = memref.alloc() : memref<32xf32, 1>
    "pytorchsim.transfer"(%x, %c0, %tx, %c0, %c0) <{direction = "load", dram_arg = 0 : i64, dram_stride = [1, 32],
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}>
        : (memref<1024xf32>, index, memref<32x32xf32, 1>, index, index) -> ()
    linalg.reduce ins(%tx : memref<32x32xf32, 1>) outs(%ty : memref<32xf32, 1>) dimensions = [1]
      (%in: f32, %acc: f32) {
        %s = arith.addf %in, %acc : f32
        linalg.yield %s : f32
      }
    "pytorchsim.transfer"(%y, %c0, %ty, %c0, %c0) <{direction = "store", dram_arg = 1 : i64, dram_stride = [1],
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}>
        : (memref<32xf32>, index, memref<32xf32, 1>, index, index) -> ()
    return
  }
}
"""


def a_reduction_s_input_loaded_whole_is_held():
    """Breaks if only matmul operands are prefetch candidates: XLA's cross-program prefetch holds a
    reduction's input too, and bufferize marks it while the linalg.reduce stands."""
    from mlir.ir import Module
    module = Module.parse(_REDUCE)
    a06_bufferize.mark_prefetch_operands(module)
    args, why = c19.plan(module, target_kind(), _target(), [((32, 32), 4), ((32,), 4)])
    assert args == [0] and "arg 0: loaded whole once at the kernel's top, the copy node" in why, why


def an_input_another_op_reads_too_is_not_held_and_says_why():
    """Breaks if a buffer an elementwise op or a store reads beside the reduction is a candidate: XLA
    prefetches a parameter only when every use is a matmul operand or a reduction input (mm_xalsoadd)."""
    from mlir.ir import Module
    store = ('    "pytorchsim.transfer"(%x, %c0, %tx, %c0, %c0) <{direction = "store", dram_arg = 0 : i64, '
             'dram_stride = [1, 32], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}> : '
             '(memref<1024xf32>, index, memref<32x32xf32, 1>, index, index) -> ()\n    return')
    module = Module.parse(_REDUCE.replace("    return", store, 1))
    a06_bufferize.mark_prefetch_operands(module)
    args, why = c19.plan(module, target_kind(), _target(), [((32, 32), 4), ((32,), 4)])
    assert args == [] and "arg 0: pytorchsim.transfer at" in why and "reads it too" in why, why


def _planned(graph_prefetch, tensors=_TENSORS):
    from mlir.ir import Module
    module = Module.parse(_MOD.format(na=2048, pre=""))
    a06_bufferize.mark_prefetch_operands(module)
    return c19.plan(module, target_kind(), _target(), tensors, graph_prefetch)


def graph_prefetch_none_is_the_kind_s_choice():
    from mlir.ir import Module
    module = Module.parse(_MOD.format(na=2048, pre=""))
    a06_bufferize.mark_prefetch_operands(module)
    assert _planned(None) == c19.plan(module, target_kind(), _target(), _TENSORS) and _planned(None)[0] == [1]


def graph_prefetch_empty_holds_nothing_and_says_so():
    """Breaks if a kernel whose graph prefetches elsewhere still holds its own operand (the graph would
    count two prefetches)."""
    args, why = _planned(())
    assert args == [] and "arg 1: loaded whole too, read as loaded (graph_prefetch is ())" in why, why


def graph_prefetch_names_the_held_argument():
    args, why = _planned((1,))
    assert args == [1] and "arg 1: loaded whole once at the kernel's top, the copy node" in why, why


def graph_prefetch_of_no_candidate_is_refused():
    """Breaks if a graph_prefetch the kernel cannot hold (arg 0 loads inside the loop) falls back to the
    kind's choice or to nothing: NotAPrefetchCandidate, failure.json's NOT_A_PREFETCH_CANDIDATE."""
    from pytorchsim_triton_compiler.contract import failure
    from pytorchsim_triton_compiler.stage import NotAPrefetchCandidate
    try:
        _planned((0,))
    except NotAPrefetchCandidate as e:
        assert "argument 0" in str(e) and "loaded inside a loop" in str(e), str(e)
        assert e.reason.value == failure.NOT_A_PREFETCH_CANDIDATE
        return
    raise AssertionError("no refusal")


def graph_prefetch_over_the_bank_is_refused():
    """Breaks if holding a tensor resident_fits refuses is stated and built anyway: SpadOverflow, with the
    lane bytes held whole and the bank."""
    from pytorchsim_triton_compiler.stage import SpadOverflow
    try:
        _planned((1,), [_TENSORS[0], ((8192, 8320), 2), _TENSORS[2]])
    except SpadOverflow as e:
        want = (8192 * 8320 * 2 // 128, cn.resident_budget(_target()))
        assert (e.usage, e.budget) == want, (e.usage, e.budget)
        return
    raise AssertionError("no refusal")


def graph_prefetch_past_the_graph_s_count_is_refused():
    try:
        _planned((0, 1))
    except c19.TooManyPrefetches as e:
        assert "holds at most 1" in str(e), str(e)
        return
    raise AssertionError("no refusal")


CASES = [the_larger_operand_is_held, a_tie_holds_a, the_smaller_is_held_when_the_larger_fills_the_scratchpad,
         both_filling_the_scratchpad_hold_none_and_say_why, a_vector_product_holds_none,
         the_tile_is_the_largest_that_fits_beside_the_held_operand,
         the_tile_walks_no_padded_tail_where_a_smaller_one_divides_the_extent, no_tile_is_narrower_than_the_matrix_unit,
         an_elf_that_holds_takes_the_whole_bank,
         the_whole_load_at_the_top_is_the_copy_node, a_load_of_part_of_its_argument_is_none,
         an_argument_moved_twice_is_none, two_whole_loads_make_one_copy_node,
         the_plan_reads_the_marks_not_the_dot, a_target_whose_graph_prefetches_nothing_holds_none,
         the_contract_loads_no_compiler,
         a_tensor_the_bank_cannot_hold_whole_is_not_held_and_says_why,
         a_held_buffer_another_op_writes_is_not_held_and_says_why,
         a_held_load_sharing_its_semaphore_is_not_held_and_says_why,
         the_gemm_plan_and_the_resident_rule_count_a_lane_alike,
         the_resident_budget_keeps_the_machine_s_scoped_share, the_v6e_machine_keeps_xla_s_scoped_share,
         graph_prefetch_none_is_the_kind_s_choice, graph_prefetch_empty_holds_nothing_and_says_so,
         graph_prefetch_names_the_held_argument, graph_prefetch_of_no_candidate_is_refused,
         graph_prefetch_over_the_bank_is_refused, graph_prefetch_past_the_graph_s_count_is_refused,
         a_reduction_s_input_loaded_whole_is_held, an_input_another_op_reads_too_is_not_held_and_says_why]


def main():
    from mlir.ir import Context, Location
    failed = 0
    for case in CASES:
        try:
            with Context() as ctx, Location.unknown():
                ctx.allow_unregistered_dialects = True
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
