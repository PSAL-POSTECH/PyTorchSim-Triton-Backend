"""Which math a vector unit has is its lowering's to say: shared passes neither list nor choose by it,
a math op stays as it is until lower-vector-functions, and what is left after it is refused by name.
python3 tests/passes/math_late_cases.py        exit 0 iff every case holds"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler import toolchain  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lower_vector_functions as v6_vf  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import (lib_math, lower,  # noqa: E402
                                                       p00_convert_unsupported_ops as p00,
                                                       p08_refuse_unlowered_math as p08)

_EXP = "mechanism/ops/ops_exp_on_the_eup.py"

_COMPILE = r"""
import copy, json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
from pytorchsim_triton_compiler import cli, machine
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lower_dma, lower_vector_functions
from pytorchsim_triton_compiler.lowering.passes import lower
t = copy.copy(machine.target())
if sys.argv[3] == "no-model":
    t.isa, t.name = [e for e in t.isa if e != "vcix_tpuv6e"], "no-model"
else:
    vf = lower_vector_functions
    vf.SFU_FUNCTIONS.pop(sys.argv[3])
    vf.FORBIDS = vf.MARKERS = lower.LOWER_VECTOR_FUNCTIONS.lowers + tuple(vf.SFU_FUNCTIONS)
machine._LAZY["TARGET"] = t
d = tempfile.mkdtemp()
got = {"transform": cli.main([sys.argv[2], "--workdir", d, "--to", "transform"])}
got["math"] = sorted(n for n in lower.ops_in_file(os.path.join(d, "08-transformed.mlir"))
                     if n.startswith("math."))
try:
    lower.run_module_passes(os.path.join(d, "08-transformed.mlir"),
                            os.path.join(d, "09-custom.mlir"), lower.to_custom(t), target=t,
                            descriptor_dims=lower_dma.DESCRIPTOR_DIMS)
except Exception as e:
    got["refused"] = str(e)
print("COMPILED " + json.dumps(got))
"""


def _compile(kernel, how):
    """transform, then TO_CUSTOM, of `kernel` on the target without vcix_tpuv6e ("no-model") or with
    one math op taken out of the SFU's table."""
    out = subprocess.run([sys.executable, "-c", _COMPILE, ROOT,
                          os.path.join(ROOT, "tests", "cases", *kernel.split("/")), how],
                         capture_output=True, text=True)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("COMPILED ")), None)
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    return json.loads(line[9:])


def shared_code_names_no_function_of_a_vector_unit():
    assert not hasattr(lib_math, "VCIX_MATH") and not hasattr(lib_math, "NATIVE_MATH")
    assert not any(o.startswith("math.") for o in lower.LOWER_VECTOR_FUNCTIONS.lowers)
    assert not set(v6_vf.SFU_FUNCTIONS) & set(p00.FORBIDS), p00.FORBIDS
    assert set(v6_vf.SFU_FUNCTIONS) <= set(v6_vf.FORBIDS) <= set(v6_vf.MARKERS)
    assert set(v6_vf.SFU_FUNCTIONS) <= set(p08.MARKERS) and not set(lib_math.RVV_MATH) & set(p08.MARKERS)


def convert_unsupported_ops_derives_what_derive_builds_and_leaves_the_rest():
    toolchain.ensure_bindings()
    from mlir.ir import Context, Location, Module
    src = """func.func @f(%a: tensor<8xf32>) -> (tensor<8xi1>, tensor<8xf32>, tensor<8xf32>) {
      %0 = math.isfinite %a : tensor<8xf32>
      %1 = math.exp %a : tensor<8xf32>
      %2 = math.log2 %a : tensor<8xf32>
      return %0, %1, %2 : tensor<8xi1>, tensor<8xf32>, tensor<8xf32>
    }"""
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = Module.parse(src)
        p00.run(m)
        names = lower.ops_held(m)
    assert "math.isfinite" in names and "math.exp" in names, names
    assert "math.log2" not in names and "math.log" in names, names


def a_target_without_the_sfu_keeps_exp_through_transform_and_refuses_it_after_the_slot():
    got = _compile(_EXP, "no-model")
    assert got["transform"] == 0 and got["math"] == ["math.exp"], got
    said = got.get("refused", "")
    assert p08.NAME in said and "math.exp" in said and "no-model" in said, got
    assert "has no pass on this target" in said, said


def an_sfu_without_a_function_leaves_it_to_be_refused_naming_its_lowering():
    got = _compile(_EXP, "math.exp")
    assert got["transform"] == 0 and got["math"] == ["math.exp"], got
    said = got.get("refused", "")
    assert p08.NAME in said and "math.exp" in said and v6_vf.NAME in said, got
    assert "does not lower it" in said, said


def a_remainder_the_isa_has_no_instruction_for_is_refused_by_name():
    """Breaks if arith.remf leaves p08's MARKERS: llc then makes each element a fmodf call and the
    compile dies in ld, naming nothing."""
    from mlir import ir
    from pytorchsim_triton_compiler import machine
    with ir.Context(), ir.Location.unknown():
        m = ir.Module.parse("func.func @f(%a: vector<16xf32>, %b: f32) -> (vector<16xf32>, f32) {\n"
                            "  %r = arith.remf %a, %a : vector<16xf32>\n"
                            "  %s = arith.remf %b, %b : f32\n"
                            "  return %r, %s : vector<16xf32>, f32\n}")
        try:
            p08.run(m, target=machine.target())
        except NotImplementedError as e:
            assert "arith.remf" in str(e), e
            return
    raise AssertionError("arith.remf reached the end of p08 unrefused")


CASES = [shared_code_names_no_function_of_a_vector_unit,
         convert_unsupported_ops_derives_what_derive_builds_and_leaves_the_rest,
         a_target_without_the_sfu_keeps_exp_through_transform_and_refuses_it_after_the_slot,
         an_sfu_without_a_function_leaves_it_to_be_refused_naming_its_lowering,
         a_remainder_the_isa_has_no_instruction_for_is_refused_by_name]

if __name__ == "__main__":
    failed = 0
    for case in CASES:
        try:
            case()
            print("PASS ", case.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL ", case.__name__, "--", type(e).__name__, e)
    sys.exit(1 if failed else 0)
