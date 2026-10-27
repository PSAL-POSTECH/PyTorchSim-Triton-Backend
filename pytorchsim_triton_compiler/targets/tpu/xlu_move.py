"""xlu_move: the tpu target's own op, `pytorchsim_tpu.xlu_move`(src, dst) {mode, rounds} -- one move
of the cross-lane unit, which lower_convert mints and lower_xlu_moves lowers; no SLOT, so not a pass."""

XLU_MOVE_OP = "pytorchsim_tpu.xlu_move"

#: the move it is (xlu.XLU_MODES) and the rounds that carry it
MODE_ATTR, ROUNDS_ATTR = "mode", "rounds"

QUEUE_KEY = "tpu_xlu_queue_entries"


def refuse(name, op, why):
    raise NotImplementedError(
        f"{name} cannot be lowered on the cross-lane unit ({why}); "
        f"materialize_layouts must not mint one this cannot take: {op}")
