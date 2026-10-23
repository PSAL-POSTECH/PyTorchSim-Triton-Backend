"""What several pass case scripts ask of the IR they lowered, said once."""
import os
import subprocess
import tempfile


def asm_mnemonics(text, target):
    """The instruction mnemonics llc emits for the lowered-MLIR `text`, taken through to_llvm on `target`."""
    from pytorchsim_triton_compiler import toolchain
    from pytorchsim_triton_compiler.lowering.passes import lower
    with tempfile.TemporaryDirectory() as d:
        src, out, ll, s = (os.path.join(d, f) for f in ("k.mlir", "l.mlir", "k.ll", "k.s"))
        with open(src, "w") as f:
            f.write(text)
        lower.to_llvm(src, out, target=target)
        subprocess.run([toolchain.MLIR_TRANSLATE, "-mlir-to-llvmir", out, "-o", ll], check=True)
        subprocess.run([toolchain.LLC, "-relocation-model=pic", "-march=" + target.llc_march(), "-O3",
                        "-mattr=" + target.llc_mattr(), ll, "-o", s], check=True)
        with open(s) as f:
            return [ln.split()[0] for ln in f
                    if ln.strip() and not ln.lstrip().startswith((".", "#")) and ":" not in ln]


def global_root(v):
    """The memref.get_global symbol `v` is a view of through first operands, else the value it stops at."""
    from mlir import ir
    while True:
        o = getattr(v, "owner", None)
        o = getattr(o, "operation", o)
        if not isinstance(o, ir.Operation):
            return v
        if o.name == "memref.get_global":
            return ir.FlatSymbolRefAttr(o.attributes["name"]).value
        if not len(o.operands):
            return v
        v = o.operands[0]
