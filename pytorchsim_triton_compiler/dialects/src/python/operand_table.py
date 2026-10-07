"""Write each torchsim op's operand and attribute names, in ODS order, as Python.
Read off `llvm-tblgen --dump-json` of the same .td the bindings come from, so a
pass that needs a slot's POSITION asks the schema instead of counting to it."""
import json
import sys


def _is(records, arg, kind):
    """Does ODS argument `arg` derive from TableGen class `kind`?"""
    d = arg["def"] if isinstance(arg, dict) else arg
    return kind in records.get(d, {}).get("!superclasses", ())


def main(src, dst):
    records = json.load(open(src))
    operands, attributes = {}, {}
    for name in records["!instanceof"]["Op"]:
        rec = records[name]
        dialect = records[rec["opDialect"]["def"]]["name"]
        if dialect != "torchsim":
            continue
        op = f"{dialect}.{rec['opName']}"
        args = rec["arguments"]["args"]
        operands[op] = tuple(n for a, n in args if _is(records, a, "TypeConstraint"))
        attributes[op] = tuple(n for a, n in args if _is(records, a, "Attr"))
    with open(dst, "w") as f:
        f.write("# Generated from TorchSimOps.td by operand_table.py; do not edit.\n")
        f.write(f"OPERANDS = {operands!r}\nATTRIBUTES = {attributes!r}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
