"""The object llc made, refused when its code moves a vector element into a scalar register: the 128
lanes share one scalar register file, so the value would be lane 0's alone. Read off the ELF's
executable sections and the RVV encoding, never off the assembly text."""
import struct

from ...driver.stage import StageError

SHF_EXECINSTR = 0x4
SHT_SYMTAB = 2
STT_FUNC = 2

OP_V = 0x57
VWXUNARY0 = 0x10
"""funct6 of the OPMVV/OPFVV unary group whose results go to x or f: vmv.x.s, vcpop.m, vfirst.m and
vfmv.f.s, told apart by vs1."""
OPFVV, OPMVV = 1, 2
SCALAR_RESULT = {(OPMVV, 0x00): "vmv.x.s", (OPMVV, 0x10): "vcpop.m", (OPMVV, 0x11): "vfirst.m",
                 (OPFVV, 0x00): "vfmv.f.s"}


class LaneValueMadeScalar(StageError):
    """The kernel's code moves a vector element into a scalar register, which every lane's one
    shared scalar register file holds for lane 0 only; `moves` lists (function, offset, mnemonic)."""

    def __init__(self, stage, message, moves, **kw):
        super().__init__(stage, message, **kw)
        self.moves = moves


def refuse_lane_moves(obj, stage="binary"):
    """Raise LaneValueMadeScalar when the object `obj` holds a vector-to-scalar move."""
    moves = vector_to_scalar_moves(obj)
    if moves:
        shown = ", ".join(f"{m} in {f}+0x{o:x}" for f, o, m in moves[:4])
        raise LaneValueMadeScalar(
            stage, f"llc moved a vector element into a scalar register {len(moves)} time(s), which "
            f"every lane's one shared scalar register file holds for lane 0 only: {shown}",
            moves, artifact=obj)


def vector_to_scalar_moves(obj):
    """(function, offset in it, mnemonic) of every instruction of `obj` that writes a vector's
    element to an x or f register."""
    got = []
    for func, off, word in instruction_words(obj):
        if word is None or word & 0x7F != OP_V or word >> 26 != VWXUNARY0:
            continue
        name = SCALAR_RESULT.get(((word >> 12) & 7, (word >> 15) & 0x1F))
        if name:
            got.append((func, off, name))
    return got


def instruction_words(obj):
    """(function, offset in it, 32-bit word or None for a 16-bit one) of every instruction in an
    executable section of the ELF64 little-endian object `obj`; an encoding over 32 bits is an error."""
    with open(obj, "rb") as f:
        data = f.read()
    sections = _sections(data)
    funcs = _functions(data, sections)
    for idx, (flags, off, size) in enumerate((s[1], s[2], s[3]) for s in sections):
        if not flags & SHF_EXECINSTR:
            continue
        starts = sorted(funcs.get(idx, []))
        pos = 0
        while pos < size:
            half = struct.unpack_from("<H", data, off + pos)[0]
            func, base = _owner(starts, pos)
            if half & 3 != 3:
                yield func, pos - base, None
                pos += 2
            elif half & 0x1F == 0x1F:
                raise ValueError(f"{obj}: an instruction over 32 bits at 0x{pos:x} of section {idx}")
            else:
                yield func, pos - base, struct.unpack_from("<I", data, off + pos)[0]
                pos += 4


def _sections(data):
    """(type, flags, offset, size, link) of every section header."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not an ELF64 little-endian object")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    out = []
    for i in range(shnum):
        _, typ, flags, _, off, size, link = struct.unpack_from("<IIQQQQI", data, shoff + i * shentsize)
        out.append((typ, flags, off, size, link))
    return out


def _functions(data, sections):
    """{section index: [(offset, name)]} of every STT_FUNC symbol."""
    out = {}
    for typ, _, off, size, link in sections:
        if typ != SHT_SYMTAB:
            continue
        strtab = sections[link][2]
        for e in range(off, off + size, 24):
            name, info, _, shndx, value = struct.unpack_from("<IBBHQ", data, e)
            if info & 0xF == STT_FUNC:
                end = data.index(b"\0", strtab + name)
                out.setdefault(shndx, []).append((value, data[strtab + name:end].decode()))
    return out


def _owner(starts, pos):
    """(name, start) of the last function of `starts` beginning at or before `pos`."""
    got = ("?", 0)
    for value, name in starts:
        if value > pos:
            break
        got = (name, value)
    return got
