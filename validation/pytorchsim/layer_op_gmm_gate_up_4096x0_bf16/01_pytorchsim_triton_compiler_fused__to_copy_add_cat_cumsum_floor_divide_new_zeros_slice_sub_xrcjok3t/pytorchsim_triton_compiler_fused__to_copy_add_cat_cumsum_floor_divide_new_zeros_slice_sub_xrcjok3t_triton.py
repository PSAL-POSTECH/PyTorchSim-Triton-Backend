
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.fixed_config(
    config={'XBLOCK': 1, 'R0_BLOCK': 128},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i64', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=128, cc=0, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'pytorchsim_triton_compiler_fused__to_copy_add_cat_cumsum_floor_divide_new_zeros_slice_sub_xrcjok3t', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'r0_': 3072}, 'backend_hash': 'D292CD358EA263BF1BBAF566FEBC6D270844669D9C5F6FFF2E2A0981F87C52D8', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def pytorchsim_triton_compiler_fused__to_copy_add_cat_cumsum_floor_divide_new_zeros_slice_sub_xrcjok3t(in_ptr0, out_ptr0, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[:]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = (r0_0).to(tl.int32)
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 >= tmp3
    tmp5 = (r0_0).to(tl.int64)
    tmp6 = (tmp5).to(tl.int64)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tmp2 >= tmp7
    tmp13 = tl.full([1], 128, tl.int64)
    tmp14 = tmp2 < tmp13
    tmp15 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + r0_0, [R0_BLOCK])), tmp12, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp15.to(tl.int64)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp12, tmp16, tmp17)
    tmp19 = tl.where(tmp8, tmp11, tmp18)
    tmp20 = tmp1 - tmp19
    tmp21 = tmp20 + tmp13
    tmp22 = tmp21 - tmp7
    tmp23 = tl.full([1], 0, tl.int32)
    tmp24 = tl.full([1], 1, tl.int32)
    tmp25 = tmp13 == tmp23
    tmp26 = tl.where(tmp25, tmp24, tmp13)
    tmp27 = tmp26 < tmp23
    tmp28 = tmp23 - tmp22
    tmp29 = tl.where(tmp27, tmp28, tmp22)
    tmp30 = tmp23 - tmp26
    tmp31 = tl.where(tmp27, tmp30, tmp26)
    tmp32 = tmp29 < tmp23
    tmp33 = ~tmp29
    tmp34 = tl.where(tmp32, tmp33, tmp29)
    tmp35 = tmp34 // tmp31
    tmp36 = ~tmp35
    tmp37 = tl.where(tmp32, tmp36, tmp35)
    tmp38 = tl.where(tmp25, tmp23, tmp37)
    tmp39 = tmp38.to(tl.int64)
    tmp40 = tl.broadcast_to(tmp39, [R0_BLOCK])
    tmp41, = tl.associative_scan((tmp40,), 0, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp41, None)
