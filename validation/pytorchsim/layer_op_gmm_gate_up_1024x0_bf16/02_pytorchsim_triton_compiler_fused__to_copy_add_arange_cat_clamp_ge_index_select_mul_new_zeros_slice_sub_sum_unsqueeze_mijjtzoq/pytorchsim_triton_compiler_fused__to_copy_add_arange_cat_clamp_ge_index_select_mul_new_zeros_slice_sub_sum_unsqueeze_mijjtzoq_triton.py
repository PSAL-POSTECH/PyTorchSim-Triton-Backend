
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.fixed_config(
    config={'XBLOCK': 256, 'R0_BLOCK': 128},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i32', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'out_ptr3': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=128, cc=0, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'pytorchsim_triton_compiler_fused__to_copy_add_arange_cat_clamp_ge_index_select_mul_new_zeros_slice_sub_sum_unsqueeze_mijjtzoq', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 4584, 'r0_': 1024}, 'backend_hash': 'D292CD358EA263BF1BBAF566FEBC6D270844669D9C5F6FFF2E2A0981F87C52D8', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def pytorchsim_triton_compiler_fused__to_copy_add_arange_cat_clamp_ge_index_select_mul_new_zeros_slice_sub_sum_unsqueeze_mijjtzoq(in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 191
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = (x0).to(tl.int64)
    tmp2 = (tmp1).to(tl.int64)
    tmp3 = tmp2 >= tmp0
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.int64)
    tmp9 = tl.full([1, 1], 127, tl.int64)
    tmp10 = tl.minimum(tmp8, tmp9, tl.PropagateNan.ALL)
    tmp11 = tmp10.to(tl.int32)
    tmp12 = (tl.full([1, 1], 128, tl.int32)).to(tl.int32)
    tmp13 = tmp10 + tmp12
    tmp14 = tmp10 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp10)
    tl.device_assert(((0 <= tmp15) & (tmp15 < 128)) | ~(xmask), "index out of bounds: 0 <= tmp15 < 128")
    tmp17 = (tmp15).to(tl.int32)
    tmp18 = tl.full([1, 1], 0, tl.int64)
    tmp19 = tmp17 >= tmp18
    tmp20 = (tmp15).to(tl.int64)
    tmp21 = (tmp20).to(tl.int64)
    tmp22 = tl.full([1, 1], 1, tl.int64)
    tmp23 = tmp21 < tmp22
    tmp24 = tl.full([1, 1], 0, tl.int64)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp23, tmp24, tmp25)
    tmp27 = tmp17 >= tmp22
    tmp28 = tl.full([1, 1], 128, tl.int64)
    tmp29 = tmp17 < tmp28
    tmp30 = tl.load(in_ptr1 + (tl.broadcast_to((-1) + tmp15, [XBLOCK, 1])), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp30.to(tl.int64)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp27, tmp31, tmp32)
    tmp34 = tl.where(tmp23, tmp26, tmp33)
    tmp35 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + tmp15, [XBLOCK, 1])), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.where(tmp23, tmp26, tmp35)
    tmp37 = tmp2 - tmp36
    tmp38 = tmp37 * tmp28
    tmp39 = tmp34 + tmp38
    tmp40 = tmp39.to(tl.int32)
    tmp41 = tl.load(in_ptr1 + (tmp15), xmask, eviction_policy='evict_last')
    tmp42 = tmp41.to(tl.int64)
    tmp43 = tmp42 - tmp34
    tmp44 = tmp43 - tmp38
    tmp45 = tl.maximum(tmp44, tmp18, tl.PropagateNan.ALL)
    tmp46 = tl.minimum(tmp45, tmp28, tl.PropagateNan.ALL)
    tmp47 = tmp46.to(tl.int32)
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tl.store(out_ptr2 + (x0), tmp40, xmask)
    tl.store(out_ptr3 + (x0), tmp47, xmask)
