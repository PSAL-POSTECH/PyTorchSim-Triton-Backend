"""Write each pytorchsim op's operand and attribute names, in ODS order, as Python."""
import json
import sys


def _is(records, arg, kind):
    d = arg["def"] if isinstance(arg, dict) else arg
    return kind in records.get(d, {}).get("!superclasses", ())


def main(src, dst):
    records = json.load(open(src))
    operands, attributes = {}, {}
    for name in records["!instanceof"]["Op"]:
        rec = records[name]
        dialect = records[rec["opDialect"]["def"]]["name"]
        if dialect != "pytorchsim":
            continue
        op = f"{dialect}.{rec['opName']}"
        args = rec["arguments"]["args"]
        operands[op] = tuple(n for a, n in args if _is(records, a, "TypeConstraint"))
        attributes[op] = tuple(n for a, n in args if _is(records, a, "Attr"))
    with open(dst, "w") as f:
        f.write("# Generated from PyTorchSimOps.td by operand_table.py; do not edit.\n")
        f.write(f"OPERANDS = {operands!r}\nATTRIBUTES = {attributes!r}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
