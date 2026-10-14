"""lib_cross_lane: the op a move between lanes is written as, the attributes its carrier stamps on
it, and the kinds of move it can be -- shared by whoever mints, lowers or counts it; how a unit
carries it is the model's."""

CROSS_LANE_OP = "torchsim.crosslane_move"

CROSS_LANE_ATTR = "torchsim.crosslane"

CROSS_LANE_ROUNDS = "torchsim.rounds"

CROSS_LANE_KIND = "torchsim.move_kind"

CROSSING, GATHER, REPLICATE, PERMUTE = "crossing", "gather", "replicate", "permute"
