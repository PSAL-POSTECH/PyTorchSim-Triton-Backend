"""The tpu model's SFU instructions by the model's own names (third_party/vcix-accelerator/examples/
tpu/sfu.hpp): name -> (opcode, imm) of the sf.vc.v.iv that issues it. Imports nothing, so the
Triton frontend reads it as the compiler does."""

SFU = {
    "vexp":  (0b000011, 0),
    "verf":  (0b000000, 0),
    "vtanh": (0b000001, 0),
    "vsin":  (0b000010, 0),
    "vcos":  (0b000010, 1),
    "vlog":  (0b000010, 2),
    "vatan": (0b000010, 3),
}
