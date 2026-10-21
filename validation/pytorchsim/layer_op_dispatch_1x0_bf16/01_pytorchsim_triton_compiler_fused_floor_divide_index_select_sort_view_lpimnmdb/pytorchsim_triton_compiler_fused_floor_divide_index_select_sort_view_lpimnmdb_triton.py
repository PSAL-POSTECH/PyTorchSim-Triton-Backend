
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.fixed_config(
    config={'XBLOCK': 16384},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=128, cc=36, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'pytorchsim_triton_compiler_fused_floor_divide_index_select_sort_view_lpimnmdb', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 69648}, 'backend_hash': '45AA5A77AE2893BA1C5A1D55C9C7638CBAEA3DA8AEAFE798D2385574205636C3', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def pytorchsim_triton_compiler_fused_floor_divide_index_select_sort_view_lpimnmdb(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 2048
    x0 = (xindex % 2048)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1], 8, tl.int64)
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = tmp2 == tmp3
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tmp7 = tmp6 < tmp3
    tmp8 = tmp3 - tmp1
    tmp9 = tl.where(tmp7, tmp8, tmp1)
    tmp10 = tmp3 - tmp6
    tmp11 = tl.where(tmp7, tmp10, tmp6)
    tmp12 = tmp9 < tmp3
    tmp13 = ~tmp9
    tmp14 = tl.where(tmp12, tmp13, tmp9)
    tmp15 = tmp14 // tmp11
    tmp16 = ~tmp15
    tmp17 = tl.where(tmp12, tmp16, tmp15)
    tmp18 = tl.where(tmp5, tmp3, tmp17)
    tmp19 = (tl.full([XBLOCK], 1, tl.int32)).to(tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 1), "index out of bounds: 0 <= tmp22 < 1")
    tl.store(out_ptr0 + (x2), tmp24, None)
