
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=1,
num_warps=4,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=128, cc=0, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'pytorchsim_triton_compiler_fused_mm_kqf7dzap', 'backend_hash': 'DD4A5AFFCAAEBE0098CE6D30823C5E6488012D78BB4B0C1C5ED7C901719242F9', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'RESIDENT': 1, 'TILES': 1, 'ACC_TYPE': 'tl.float32', 'ALLOW_TF32': False, 'BLOCK_M': 128, 'BLOCK_N': 4096, 'BLOCK_K': 4096}},

)
@triton.jit
def pytorchsim_triton_compiler_fused_mm_kqf7dzap(arg_A, arg_B, out_ptr0):
    RESIDENT : tl.constexpr = 1
    TILES : tl.constexpr = 1
    ACC_TYPE : tl.constexpr = tl.float32
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 4096
    BLOCK_K : tl.constexpr = 4096
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 32
    N = 4096
    K = 4096
    stride_am = 4096
    stride_ak = 1
    stride_bk = 4096
    stride_bn = 1
    rk = tl.arange(0, BLOCK_K).to(INDEX_DTYPE)

    rn = tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    b = tl.load(B + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
        mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
    for t in range(0, tl.cdiv(M, TILES * BLOCK_M)):
        rm = (TILES * t + 0) * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
        a = tl.load(A + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        acc = tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        out_m = rm[:, None]
        out_n = rn[None, :]
        mask = (out_m < M) & (out_n < N)
        xindex = out_n + 4096*out_m
        tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

