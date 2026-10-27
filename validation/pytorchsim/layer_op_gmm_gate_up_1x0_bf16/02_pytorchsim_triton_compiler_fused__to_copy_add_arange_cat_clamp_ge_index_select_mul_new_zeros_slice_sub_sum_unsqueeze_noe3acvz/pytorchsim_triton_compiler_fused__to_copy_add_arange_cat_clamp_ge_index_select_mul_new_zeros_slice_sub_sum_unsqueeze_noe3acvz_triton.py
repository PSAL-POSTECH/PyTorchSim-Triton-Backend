
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.fixed_config(
    config={'XBLOCK': 8, 'R0_BLOCK': 128},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i32', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'out_ptr3': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=128, cc=0, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'pytorchsim_triton_compiler_fused__to_copy_add_arange_cat_clamp_ge_index_select_mul_new_zeros_slice_sub_sum_unsqueeze_noe3acvz', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 192, 'r0_': 1024}, 'backend_hash': 'D292CD358EA263BF1BBAF566FEBC6D270844669D9C5F6FFF2E2A0981F87C52D8', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def pytorchsim_triton_compiler_fused__to_copy_add_arange_cat_clamp_ge_index_select_mul_new_zeros_slice_sub_sum_unsqueeze_noe3acvz(in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
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
    tmp0 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = (x0).to(tl.int64)
    tmp2 = (tmp1).to(tl.int64)
    tmp3 = tmp2 >= tmp0
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.int64)
    tmp8 = tl.full([1, 1], 127, tl.int64)
    tmp9 = tl.minimum(tmp7, tmp8, tl.PropagateNan.ALL)
    tmp10 = tmp9.to(tl.int32)
    tmp11 = (tl.full([1, 1], 128, tl.int32)).to(tl.int32)
    tmp12 = tmp9 + tmp11
    tmp13 = tmp9 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp9)
    tl.device_assert((0 <= tmp14) & (tmp14 < 128), "index out of bounds: 0 <= tmp14 < 128")
    tmp16 = (tmp14).to(tl.int32)
    tmp17 = tl.full([1, 1], 0, tl.int64)
    tmp18 = tmp16 >= tmp17
    tmp19 = (tmp14).to(tl.int64)
    tmp20 = (tmp19).to(tl.int64)
    tmp21 = tl.full([1, 1], 1, tl.int64)
    tmp22 = tmp20 < tmp21
    tmp23 = tl.full([1, 1], 0, tl.int64)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp16 >= tmp21
    tmp27 = tl.full([1, 1], 128, tl.int64)
    tmp28 = tmp16 < tmp27
    tmp29 = tl.load(in_ptr1 + (tl.broadcast_to((-1) + tmp14, [XBLOCK, 1])), tmp26, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29.to(tl.int64)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp26, tmp30, tmp31)
    tmp33 = tl.where(tmp22, tmp25, tmp32)
    tmp34 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + tmp14, [XBLOCK, 1])), tmp26, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp22, tmp25, tmp34)
    tmp36 = tmp2 - tmp35
    tmp37 = tmp36 * tmp27
    tmp38 = tmp33 + tmp37
    tmp39 = tmp38.to(tl.int32)
    tmp40 = tl.load(in_ptr1 + (tmp14), None, eviction_policy='evict_last')
    tmp41 = tmp40.to(tl.int64)
    tmp42 = tmp41 - tmp33
    tmp43 = tmp42 - tmp37
    tmp44 = tl.maximum(tmp43, tmp17, tl.PropagateNan.ALL)
    tmp45 = tl.minimum(tmp44, tmp27, tl.PropagateNan.ALL)
    tmp46 = tmp45.to(tl.int32)
    tl.store(out_ptr1 + (x0), tmp10, None)
    tl.store(out_ptr2 + (x0), tmp39, None)
    tl.store(out_ptr3 + (x0), tmp46, None)
