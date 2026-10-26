
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.fixed_config(
    config={'XBLOCK': 512, 'R0_BLOCK': 128},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=128, cc=0, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'pytorchsim_triton_compiler_fused__to_copy_add_cat_clone_mean_mul_neg_pow_rsqrt_slice_transpose_unsqueeze_view_vuhn3bs2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 787200}, 'backend_hash': 'D292CD358EA263BF1BBAF566FEBC6D270844669D9C5F6FFF2E2A0981F87C52D8', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def pytorchsim_triton_compiler_fused__to_copy_add_cat_clone_mean_mul_neg_pow_rsqrt_slice_transpose_unsqueeze_view_vuhn3bs2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x3 = xindex // 4
    x2 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp52 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp57 = tl.load(in_ptr2 + (r0_1 + 128*x3), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr3 + (r0_1 + 128*x3), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = (r0_1).to(tl.int32)
    tmp7 = tl.full([1, 1], 0, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = (r0_1).to(tl.int64)
    tmp10 = (tmp9).to(tl.int64)
    tmp11 = tl.full([1, 1], 64, tl.int64)
    tmp12 = tmp10 < tmp11
    tmp13 = tl.load(in_ptr0 + (64 + 128*x0 + (r0_1)), tmp12, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.full([1, 1], 128.0, tl.float32)
    tmp16 = (tmp5 / tmp15)
    tmp17 = tl.full([1, 1], 1e-06, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp21 = tl.load(in_ptr1 + (tl.broadcast_to(64 + (r0_1), [XBLOCK, R0_BLOCK])), tmp12, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = -tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp12, tmp25, tmp26)
    tmp28 = tmp6 >= tmp11
    tmp29 = tl.full([1, 1], 128, tl.int64)
    tmp30 = tmp6 < tmp29
    tmp31 = tl.load(in_ptr0 + (128*x0 + ((-64) + r0_1)), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.full([1, 1], 128.0, tl.float32)
    tmp34 = (tmp5 / tmp33)
    tmp35 = tl.full([1, 1], 1e-06, tl.float32)
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tl.load(in_ptr1 + (tl.broadcast_to((-64) + r0_1, [XBLOCK, R0_BLOCK])), tmp28, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp38 * tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp28, tmp42, tmp43)
    tmp45 = tl.where(tmp12, tmp27, tmp44)
    tmp46 = tl.full([1, 1], 128.0, tl.float32)
    tmp47 = (tmp5 / tmp46)
    tmp48 = tl.full([1, 1], 1e-06, tl.float32)
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.rsqrt(tmp49)
    tmp51 = tmp1 * tmp50
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp51 * tmp53
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp58 = tmp56 * tmp57
    tmp59 = tmp45.to(tl.float32)
    tmp61 = tmp59 * tmp60
    tmp62 = tmp58 + tmp61
    tmp63 = tmp62.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 128*x3 + 16384*x2), tmp63, None)
