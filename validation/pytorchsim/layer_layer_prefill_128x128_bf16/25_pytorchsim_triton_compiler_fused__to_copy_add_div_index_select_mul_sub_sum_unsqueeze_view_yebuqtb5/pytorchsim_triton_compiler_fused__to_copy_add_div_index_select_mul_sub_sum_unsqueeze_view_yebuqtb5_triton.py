
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.fixed_config(
    config={'XBLOCK': 65536, 'R0_BLOCK': 8},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=128, cc=0, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'pytorchsim_triton_compiler_fused__to_copy_add_div_index_select_mul_sub_sum_unsqueeze_view_yebuqtb5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 2097152, 'r0_': 16384}, 'backend_hash': 'D292CD358EA263BF1BBAF566FEBC6D270844669D9C5F6FFF2E2A0981F87C52D8', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def pytorchsim_triton_compiler_fused__to_copy_add_div_index_select_mul_sub_sum_unsqueeze_view_yebuqtb5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x1 = xindex // 2048
    x0 = (xindex % 2048)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 8*x1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r0_2 + 8*x1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr5 + (r0_2 + 8*x1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x3), None).to(tl.float32)
    tmp31 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = (tl.full([1, 1], 128, tl.int32)).to(tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 128)) | ~(r0_mask), "index out of bounds: 0 <= tmp4 < 128")
    tmp6 = tl.load(in_ptr1 + (tmp4), r0_mask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (tmp4), r0_mask, eviction_policy='evict_last')
    tmp8 = tmp6 - tmp7
    tmp10 = tl.full([1, 1], 1, tl.int64)
    tmp11 = tmp9 - tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = (tl.full([1, 1], 1024, tl.int32)).to(tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 1024)) | ~(r0_mask), "index out of bounds: 0 <= tmp16 < 1024")
    tmp18 = tl.load(in_ptr4 + (x0 + 2048*tmp16), r0_mask, eviction_policy='evict_first').to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp22 = (tmp20 / tmp21)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp19 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None].to(tl.float32)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp29.to(tl.float32)
    tmp34 = tmp32 + tmp33
    tl.store(in_out_ptr0 + (x3), tmp34, None)
