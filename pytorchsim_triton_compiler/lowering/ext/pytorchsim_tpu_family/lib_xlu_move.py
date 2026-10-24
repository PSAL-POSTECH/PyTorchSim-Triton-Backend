"""lib_xlu_move: the pack's own op, `pytorchsim_tpu.xlu_move`(src, dst) {xlu_op, kind, rounds} -- one move
of the cross-lane unit, which lower_convert mints and lower_xlu_moves lowers; no SLOT, so not a pass."""

XLU_MOVE_OP = "pytorchsim_tpu.xlu_move"

#: the unit's SIMM5 (lib_xlu.simm5), the shape of move it is (lib_xlu.emitter_of) and the rounds that carry it
XLU_OP_ATTR, KIND_ATTR, ROUNDS_ATTR = "xlu_op", "kind", "rounds"

QUEUE_KEY = "tpu_xlu_queue_entries"


def refuse(name, op, why):
    raise NotImplementedError(
        f"{name} cannot be lowered on the cross-lane unit ({why}); "
        f"materialize_layouts must not mint one this cannot take: {op}")
