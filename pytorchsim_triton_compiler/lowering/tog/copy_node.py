"""A kernel object's copy node: a TOG of one work-item whose one load moves a kernel argument whole into one
buffer, launched before the kernel whose TOG reads that argument held (TOGSim docs/tog-abi.md, "Data that is
already on the chip"). Built through the same skeleton -> EmitC -> C++ path as the kernel's TOG."""
import os

DIR = "copy_node_{arg}"


def emit(workdir, arg, element_type, numel, nbytes, held_buffer):
    """Write copy_node_<arg>/tog.cpp and tog.so in `workdir`; return its kernel_object.COPY_NODES entry."""
    from .passes import lower_to_emitc as l2e
    from .passes.build_tog import ir
    d = DIR.format(arg=arg)
    os.makedirs(os.path.join(workdir, d), exist_ok=True)
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        module = skeleton(arg, element_type, numel, nbytes)
        cpp = l2e.emitc_to_cpp(l2e.lower_to_emitc(module))
    cpp_path = os.path.join(workdir, d, "tog.cpp")
    with open(cpp_path, "w") as fh:
        fh.write(cpp)
    l2e.compile_so(cpp_path, os.path.join(workdir, d, "tog.so"))
    return {"arg": arg, "bytes": nbytes, "held_buffer": held_buffer, "tog_so": os.path.join(d, "tog.so")}


def skeleton(arg, element_type, numel, nbytes):
    """The skeleton+API module of the copy node: one synchronous load of `numel` elements of `arg` into buffer
    0."""
    from mlir.dialects import arith, func
    from .passes import togsim_ops as ts
    from .passes._mlir_util import i32, i64_array, str_attr
    from .passes.build_tog import ir
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        fn = func.FuncOp("kernel", ir.FunctionType.get([], []))
        fn.attributes[ts.ATTR_SPAD_BUFFER_BYTES] = ir.DenseI64ArrayAttr.get([nbytes])
        block = fn.add_entry_block()
    with ir.InsertionPoint(block):
        idx = ir.IndexType.get()
        zero = arith.ConstantOp(idx, ir.IntegerAttr.get(idx, 0)).result
        ir.Operation.create(ts.DMA, results=[], operands=[zero, zero], attributes={
            ts.ATTR_DIR: i32(ts.DIR_LOAD), ts.ATTR_DIMS: i64_array([numel]), ts.ATTR_STRIDES: i64_array([1]),
            ts.ATTR_ELEM: str_attr(_elem(element_type)), ts.ATTR_IS_ASYNC: ir.BoolAttr.get(False),
            ts.ATTR_ARG_ID: i32(arg), ts.ATTR_INDIRECT: i32(0), "base": str_attr(f"arg{arg}"),
            ts.ATTR_READ_BUFS: i64_array([]), ts.ATTR_WRITE_BUFS: i64_array([0])})
        func.ReturnOp([])
    return module


def _elem(ty):
    """The togsim.hpp element type of the MLIR element type `ty`, by its type class."""
    from .passes.build_tog import ir
    for cls, name in ((ir.BF16Type, "bf16"), (ir.F16Type, "f16"), (ir.F32Type, "f32"), (ir.F64Type, "f64")):
        if ty.typeid == cls.static_typeid:
            return name
    if ty.typeid == ir.IntegerType.static_typeid and ir.IntegerType(ty).width in (8, 16, 32, 64):
        return f"i{ir.IntegerType(ty).width}"
    raise RuntimeError(f"a copy node of {ty} elements moves no type a TOG can state")
