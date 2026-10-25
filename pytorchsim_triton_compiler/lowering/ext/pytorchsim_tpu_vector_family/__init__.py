"""The pack of the pytorchsim_tpu model (vcix-accelerator's libtpu.so: a TPU's vector-side units, no
matrix unit). It takes the TPU family's DMA, lane number, special functions and kernel vcix, and
moves nothing between lanes: a dot is each lane's RVV loop, a convert or a gather stays inside the
banks, and no reduce folds across the lanes -- rvv_generic's providers and LayoutRules, run on a
machine the simulators have."""

LAYOUT_RULES = "rvv_generic"

PROVIDES = {
    "lower-convert": "rvv_generic.lower_convert",
    "lower-matmul": "rvv_generic.lower_matmul",
    "lower-vector-functions": "pytorchsim_tpu_family.lower_vector_functions",
    "lower-distributed-reduction": "rvv_generic.lower_distributed_reduction",
    "lower-gather": "rvv_generic.lower_gather",
    "lower-kernel-vcix": "pytorchsim_tpu_family.lower_kernel_vcix",
    "lower-dma": "pytorchsim_tpu_family.lower_dma",
}
