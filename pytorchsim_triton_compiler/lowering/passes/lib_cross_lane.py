"""lib_cross_lane: the op a move between lanes is written as, the attributes its carrier stamps on
it (EVERY_LANE: every lane takes each destination element from the same source element), and the
kinds of move it can be -- shared by whoever mints, lowers or counts it; how is the model's."""

CROSS_LANE_OP = "torchsim.crosslane_move"

CROSS_LANE_ATTR = "torchsim.crosslane"

CROSS_LANE_ROUNDS = "torchsim.rounds"

CROSS_LANE_KIND = "torchsim.move_kind"

EVERY_LANE = "torchsim.every_lane"

CROSSING, GATHER, PERMUTE = "crossing", "gather", "permute"

# A fold across the lanes: every lane's elements combined over the lanes, column by column, by the
# op FOLD_BY names (as arith spells it); the destination, in one bank, is combined with it.
FOLD = "fold"

FOLD_BY = "torchsim.fold_by"
