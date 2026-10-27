"""tritongpu-pytorchsim-regularize-layouts on IR written here, run in the Triton frontend's own process (the
plugin pass), its output read back through the bindings.
    python3 tests/passes/regularize_layouts_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler import toolchain  # noqa: E402
from pytorchsim_triton_compiler.passes.utils import ir_utils, linear_layout  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402

_ATTRS = ('{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "pytorchsim:0", '
          '"ttg.threads-per-warp" = 128 : i32}')
_LINEAR = "#ttg.linear<{register = [[1], [2]], lane = [], warp = [], block = []}>"
_LINEAR2 = "#ttg.linear<{register = [[1, 0], [2, 0]], lane = [], warp = [], block = []}>"

# The frontend's context knows func and tensor once a pipeline that makes them has run, as in emit_linalg.
_PROBE = '''
import os, sys, tempfile
from pytorchsim_triton_compiler import frontend as triton_compile
from triton._C.libtriton import ir, passes
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend
ctx = ir.context(); ir.load_dialects(ctx); make_backend(GPUTarget("cpu", 0, 0)).load_dialects(ctx)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "z.mlir")
    open(p, "w").write("module attributes %s { tt.func @z() { tt.return } }")
    pm = ir.pass_manager(ctx); triton_compile._to_linalg(pm, passes, 128); pm.run(ir.parse_mlir_module(p, ctx), p)
mod = ir.parse_mlir_module(sys.argv[1], ctx)
pm = ir.pass_manager(ctx)
triton_compile._regularize(pm, passes, 128)
passes.plugin.write_generic(pm, [sys.argv[2]])
pm.run(mod, sys.argv[2])
''' % _ATTRS.replace('"', '\\"')


def _regularized(body):
    """`body` (a func) in a module, through regularize-layouts: (the output module, stderr); module None
    when the pass failed."""
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "in.mlir"), os.path.join(d, "out.mlir")
        with open(src, "w") as f:
            f.write(f"module attributes {_ATTRS} {{\n{body}\n}}\n")
        env = toolchain.triton_env()
        env["PYTHONPATH"] = ROOT + os.pathsep + env["PYTHONPATH"]
        got = subprocess.run([sys.executable, "-c", _PROBE, src, out], env=env, capture_output=True, text=True)
        if got.returncode:
            return None, got.stderr
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx:
            return ir.Module.parse(open(out).read()), got.stderr


def _one(module, name):
    found = [ir_utils.op_of(o) for o in ir_utils.walk(module.operation) if ir_utils.name_of(o) == name]
    assert len(found) == 1, (name, len(found))
    return found[0]


def _bases(t):
    return linear_layout.get_bases(ir.RankedTensorType(t).encoding)


def a_cast_gives_its_bare_producer_its_layout_and_not_its_shape():
    """Breaks if a tensor.cast hands its laid result's whole type, the dynamic shape included, to the bare
    tensor.empty it casts: the empty would state no size for its dynamic dim and fail to verify."""
    m, err = _regularized(f'''  func.func @k() -> tensor<?xf32, {_LINEAR}> {{
    %c = tensor.empty() : tensor<4xf32>
    %x = tensor.cast %c : tensor<4xf32> to tensor<?xf32, {_LINEAR}>
    return %x : tensor<?xf32, {_LINEAR}>
  }}''')
    assert m is not None, err[-1500:]
    with m.context:
        made = ir.RankedTensorType(_one(m, "tensor.empty").results[0].type)
        cast = _one(m, "tensor.cast")
        assert made.shape == [4], made
        assert made.encoding is not None and _bases(made) == _bases(cast.results[0].type), made


def a_region_op_s_result_is_refused_its_layout_not_retyped_alone():
    """Breaks if a bare scf.for result that a laid expand_shape reads is retyped by itself, leaving its
    iter_arg and yield on the old type, instead of the pass refusing it by name."""
    m, err = _regularized(f'''  func.func @k(%n: index) -> tensor<4x1xf32, {_LINEAR2}> {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %e = tensor.empty() : tensor<4xf32>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%x = %e) -> (tensor<4xf32>) {{
      scf.yield %x : tensor<4xf32>
    }}
    %y = tensor.expand_shape %r [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32, {_LINEAR2}>
    return %y : tensor<4x1xf32, {_LINEAR2}>
  }}''')
    assert m is None, m
    assert "a block argument or a region op's result is not retyped apart from its region" in err, err[-1500:]


CASES = [a_cast_gives_its_bare_producer_its_layout_and_not_its_shape,
         a_region_op_s_result_is_refused_its_layout_not_retyped_alone]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
