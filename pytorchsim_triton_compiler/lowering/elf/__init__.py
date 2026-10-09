"""Step binary: 10-lowered.mlir -> 11-<kernel>.elf, the ELF Spike runs, and kernel.json naming it.
build.py lowers to LLVM IR, compiles, links and holds the result to its budgets; wrapper.py writes
the C driver beside the kernel."""


def recipe():
    return ["10-lowered.mlir + the C driver -> 11-<kernel>.elf, kernel.json"]


def run(ctx):
    from .build import build_kernel_object
    return build_kernel_object(ctx)
