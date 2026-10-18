
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.fixed_config(
    config={'XBLOCK': 1, 'R0_BLOCK': 128},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=128, cc=36, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'pytorchsim_triton_compiler_fused__softmax_sdu7tr24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'r0_': 768}, 'backend_hash': '45AA5A77AE2893BA1C5A1D55C9C7638CBAEA3DA8AEAFE798D2385574205636C3', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def pytorchsim_triton_compiler_fused__softmax_sdu7tr24(in_ptr0, out_ptr2, xnumel, r0_numel):
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
    tmp0 = tl.load(in_ptr0 + (r0_0), None, eviction_policy='evict_first').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp2, 0)).to(tl.float32)
    tmp5 = tmp1 - tmp4
    tmp6 = libdevice.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [R0_BLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0)).to(tl.float32)
    tmp10 = (tmp6 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp11, None)
