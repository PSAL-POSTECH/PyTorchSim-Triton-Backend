# Dump the whole MLIR module from ANYWHERE in a triton-shared-opt debug session.
#
#   source this file once (or wire it into launch.json setupCommands), then:
#     mlir            -- walk the current frame + up the stack, find any in-scope
#                        MLIR op, and dump ITS module (dump() is recursive, so a
#                        ModuleOp prints the entire IR at the current pass state)
#     mlir <var>      -- dump the module reachable from a specific op variable
#
# Why a stack walk: MLIR keeps no global "current module", and PtrAnalysis holds
# no module pointer -- only Value maps. But pointer analysis is always CALLED
# from rewriteLoadOp / rewriteStoreOp / rewriteForOp, whose frames do have an op
# (`op`, `rangeOp`, `splatOp`, ...). So from any deeper frame, walking up finds
# one. Output goes to the program's stderr = the Debug Console.
python
import gdb

def _call(expr):
    try:
        gdb.execute("call " + expr)
        return True
    except gdb.error:
        return False

class MLIRModuleCmd(gdb.Command):
    """mlir [opvar] : dump the whole MLIR module (searches the stack if no arg)."""
    # op-wrapper locals that appear across PtrAnalysis / conversion methods.
    # All are value-type Op wrappers, so `.getOperation()` is always valid --
    # no null-deref risk (unlike a Value's .getDefiningOp()).
    OPS = ["op", "rangeOp", "splatOp", "addptrOp", "loadOp", "storeOp",
           "cmpOp", "broadcastOp", "expandDimsOp", "reshapeOp", "transOp",
           "constOp", "remOp", "divOp", "mulOp", "bitcastOp",
           "makeTensorPtrOp", "forOp", "yieldOp"]

    def __init__(self):
        super(MLIRModuleCmd, self).__init__("mlir", gdb.COMMAND_USER)

    def _dump(self, name):
        return _call(
            "(%s).getOperation()->getParentOfType<mlir::ModuleOp>().dump()" % name)

    def invoke(self, arg, from_tty):
        names = [arg.strip()] if arg.strip() else self.OPS
        start = gdb.selected_frame()
        f = start
        try:
            while f is not None:
                f.select()
                for n in names:
                    try:
                        gdb.parse_and_eval(n)   # is this name in scope here?
                    except gdb.error:
                        continue
                    if self._dump(n):
                        return
                f = f.older()
        finally:
            try:
                start.select()
            except Exception:
                pass
        raise gdb.GdbError(
            "mlir: no in-scope Op found up the stack; pass one explicitly, "
            "e.g. 'mlir op' (must be an Op wrapper, not a Value)")

MLIRModuleCmd()
end
