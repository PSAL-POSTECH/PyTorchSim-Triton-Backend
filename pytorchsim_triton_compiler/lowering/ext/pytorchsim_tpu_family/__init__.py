"""The pack of the pytorchsim_tpu_family models (contract.target.ACCELERATORS): the TPU's units, one
pass a slot. PROVIDES is the machine -- a slot it fills names a unit the TPU has."""

PROVIDES = {
    "lower-copies": ".lower_copies",                     # the XLU carries a layout conversion
    "bound-loop-widths": ".bound_eup_loops",             # the EUP's register groups bound a loop's LMUL
    "lower-matmul": ".lower_matmul",                     # the MXU: vmatprep, vmatpush, vmatmul, vpop
    "schedule-matmul": ".schedule_mxu_stream",
    "lower-vector-functions": ".lower_vector_functions",  # the SFU and the lane-index unit
    "schedule-vpu": ".schedule_vpu_stream",
    "lower-cross-lane": ".lower_cross_lane",             # the XLU: folds, gathers and permutes
    "lower-kernel-vcix": ".lower_kernel_vcix",
    "lower-dma": ".lower_dma",                           # CONFIG_DESC + MVIN/MVOUT over a 20-slot descriptor
}
