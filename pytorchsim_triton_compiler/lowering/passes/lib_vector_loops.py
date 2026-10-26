"""lib_vector_loops: the vector loops pytorchsim-vectorize makes, as the passes after it and the packs read
them -- the mark on a vector loop (VECTOR_LOOP),
and the vector.CombiningKind each combiner a reduce folds by is (REDUCE_KIND)."""

VECTOR_LOOP = "pytorchsim.vector_loop"


REDUCE_KIND = {
    "arith.addf": "ADD",   "arith.addi": "ADD",
    "arith.mulf": "MUL",   "arith.muli": "MUL",
    "arith.maxnumf": "MAXNUMF", "arith.maximumf": "MAXIMUMF",
    "arith.minnumf": "MINNUMF", "arith.minimumf": "MINIMUMF",
    "arith.maxsi": "MAXSI", "arith.minsi": "MINSI",
    "arith.maxui": "MAXUI", "arith.minui": "MINUI",
    "arith.andi": "AND",   "arith.ori": "OR",  "arith.xori": "XOR",
}
