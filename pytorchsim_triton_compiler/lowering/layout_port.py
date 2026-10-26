"""The layout port's Python end (triton_extension/include/TorchsimExt/LayoutModel.h): a target's layout
model (slots.LayoutModel) registered with the plugin while the TTGIR passes run, each question handed
to it as plain Layouts and each answer handed back as the C table wants it."""
import contextlib
import ctypes
from dataclasses import dataclass

VERSION = 3
OP_KINDS = ("load", "arith", "math", "reduce", "scan", "join", "split", "cat", "reshape", "trans",
            "gather", "address")


@dataclass(frozen=True)
class Layout:
    """A tile's layout as linear-layout bases: the coordinate each register and each lane bit moves to."""
    shape: tuple
    elem_bits: int
    reg: tuple
    lane: tuple


class _CLayout(ctypes.Structure):
    _fields_ = [("rank", ctypes.c_int32), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("elem_bits", ctypes.c_int32), ("n_reg", ctypes.c_int32), ("n_lane", ctypes.c_int32),
                ("reg_bases", ctypes.POINTER(ctypes.c_int32)),
                ("lane_bases", ctypes.POINTER(ctypes.c_int32))]


_LAYOUT_FREE = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)
_LAYOUT_LEGAL = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(_CLayout),
                                 ctypes.POINTER(_CLayout))
_CONVERT_COST = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(_CLayout), ctypes.POINTER(_CLayout))
_REMAT_COST = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_int32, ctypes.POINTER(_CLayout))
_LANE_FOLDS = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32)


class _CModel(ctypes.Structure):
    _fields_ = [("version", ctypes.c_int32), ("lanes", ctypes.c_int32),
                ("layout_free", _LAYOUT_FREE), ("layout_legal", _LAYOUT_LEGAL),
                ("convert_cost", _CONVERT_COST),
                ("remat_cost", _REMAT_COST), ("lane_folds", _LANE_FOLDS)]


def _layout(p):
    c = p.contents
    rank = c.rank

    def bases(ptr, n):
        return tuple(tuple(ptr[i * rank + d] for d in range(rank)) for i in range(n))
    return Layout(tuple(c.shape[d] for d in range(rank)), c.elem_bits,
                  bases(c.reg_bases, c.n_reg), bases(c.lane_bases, c.n_lane))


def _answer(fn, default):
    """`fn` as a callback: an exception is printed and answered with `default` (upstream's answer),
    since one raised through ctypes would be lost."""
    def call(*args):
        try:
            return fn(*args)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return default
    return call


@contextlib.contextmanager
def installed(lib_path, model):
    """`model` registered with the plugin at `lib_path` for the body of the with; nothing when `model`
    is None. The callbacks live as long as the with does."""
    if model is None:
        yield
        return
    lib = ctypes.CDLL(lib_path)
    table = _CModel(
        VERSION, model.lanes,
        _LAYOUT_FREE(_answer(lambda op: int(model.layout_free(OP_KINDS[op])), -1)),
        _LAYOUT_LEGAL(_answer(lambda op, a, r: int(model.layout_legal(OP_KINDS[op], _layout(a),
                                                                      _layout(r))), 1)),
        _CONVERT_COST(_answer(lambda s, d: float(model.convert_cost(_layout(s), _layout(d))), -1.0)),
        _REMAT_COST(_answer(lambda op, at: float(model.remat_cost(OP_KINDS[op], _layout(at))), -1.0)),
        _LANE_FOLDS(_answer(lambda name, bits: int(model.lane_folds(name.decode(), bits)), 1)))
    lib.torchsim_set_layout_model(ctypes.byref(table))
    try:
        yield
    finally:
        lib.torchsim_set_layout_model(None)
