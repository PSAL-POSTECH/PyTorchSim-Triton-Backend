"""convert-pytorchsim-to-pytorchsim-tpu: every pytorchsim op the TPU computes on its own units, written as
their ops (pytorchsim_tpu) by one rule module an op family, in order: the MXU's dots (lower_matmul), the SFU's
and lane-index unit's functions (lower_vector_functions), the XLU's reductions (lower_distributed_reduction),
gathers (lower_gather) and moves (lower_xlu_moves)."""
from ...passes.lower import ops_held
from . import lower_distributed_reduction, lower_gather, lower_matmul, lower_vector_functions, lower_xlu_moves

RULES = (lower_matmul, lower_vector_functions, lower_distributed_reduction, lower_gather, lower_xlu_moves)


def run(module, **opts):
    """Each rule in RULES, in order, that has one of its MARKERS in the module when its turn comes; the
    ops held are read again only after a rule ran."""
    held = ops_held(module)
    for rule in RULES:
        if not held.isdisjoint(rule.MARKERS):
            rule.run(module, **opts)
            held = ops_held(module)
    return module
