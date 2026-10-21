
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.fixed_config(
    config={'XBLOCK': 1, 'R0_BLOCK': 128},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=128, cc=36, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'pytorchsim_triton_compiler_fused__softmax__to_copy_b54moclx', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'atomic_add_found': False, 'num_load': 1, 'num_store': 8, 'num_reduction': 17, 'autotune_hints': set(), 'tiling_scores': {'r0_': 256}, 'backend_hash': '45AA5A77AE2893BA1C5A1D55C9C7638CBAEA3DA8AEAFE798D2385574205636C3', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def pytorchsim_triton_compiler_fused__softmax__to_copy_b54moclx(in_ptr0, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, xnumel, r0_numel):
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
    tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp13 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp11, 0)).to(tl.float32)
    tmp14 = tmp10 == tmp13
    tmp15 = (r0_0).to(tl.int64)
    tmp16 = (tmp15).to(tl.int64)
    tmp17 = tl.full([1], 128, tl.int64)
    tmp18 = tl.where(tmp14, tmp16, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [R0_BLOCK])
    tmp21 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp19, 0)).to(tl.int64)
    tmp22 = tmp16 == tmp21
    tmp23 = tl.full([1], -3.4028234663852886e+38, tl.float32)
    tmp24 = tl.where(tmp22, tmp23, tmp10)
    tmp25 = tl.broadcast_to(tmp24, [R0_BLOCK])
    tmp27 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp25, 0)).to(tl.float32)
    tmp28 = tmp24 == tmp27
    tmp29 = tl.where(tmp28, tmp16, tmp17)
    tmp30 = tl.broadcast_to(tmp29, [R0_BLOCK])
    tmp32 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp30, 0)).to(tl.int64)
    tmp33 = tmp16 == tmp32
    tmp34 = tl.where(tmp33, tmp23, tmp24)
    tmp35 = tl.broadcast_to(tmp34, [R0_BLOCK])
    tmp37 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp35, 0)).to(tl.float32)
    tmp38 = tmp34 == tmp37
    tmp39 = tl.where(tmp38, tmp16, tmp17)
    tmp40 = tl.broadcast_to(tmp39, [R0_BLOCK])
    tmp42 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp40, 0)).to(tl.int64)
    tmp43 = tmp16 == tmp42
    tmp44 = tl.where(tmp43, tmp23, tmp34)
    tmp45 = tl.broadcast_to(tmp44, [R0_BLOCK])
    tmp47 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp45, 0)).to(tl.float32)
    tmp48 = tmp44 == tmp47
    tmp49 = tl.where(tmp48, tmp16, tmp17)
    tmp50 = tl.broadcast_to(tmp49, [R0_BLOCK])
    tmp52 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp50, 0)).to(tl.int64)
    tmp53 = tmp16 == tmp52
    tmp54 = tl.where(tmp53, tmp23, tmp44)
    tmp55 = tl.broadcast_to(tmp54, [R0_BLOCK])
    tmp57 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp55, 0)).to(tl.float32)
    tmp58 = tmp54 == tmp57
    tmp59 = tl.where(tmp58, tmp16, tmp17)
    tmp60 = tl.broadcast_to(tmp59, [R0_BLOCK])
    tmp62 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp60, 0)).to(tl.int64)
    tmp63 = tmp16 == tmp62
    tmp64 = tl.where(tmp63, tmp23, tmp54)
    tmp65 = tl.broadcast_to(tmp64, [R0_BLOCK])
    tmp67 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp65, 0)).to(tl.float32)
    tmp68 = tmp64 == tmp67
    tmp69 = tl.where(tmp68, tmp16, tmp17)
    tmp70 = tl.broadcast_to(tmp69, [R0_BLOCK])
    tmp72 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp70, 0)).to(tl.int64)
    tmp73 = tmp16 == tmp72
    tmp74 = tl.where(tmp73, tmp23, tmp64)
    tmp75 = tl.broadcast_to(tmp74, [R0_BLOCK])
    tmp77 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp75, 0)).to(tl.float32)
    tmp78 = tmp74 == tmp77
    tmp79 = tl.where(tmp78, tmp16, tmp17)
    tmp80 = tl.broadcast_to(tmp79, [R0_BLOCK])
    tmp82 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp80, 0)).to(tl.int64)
    tmp83 = tmp16 == tmp82
    tmp84 = tl.where(tmp83, tmp23, tmp74)
    tmp85 = tl.broadcast_to(tmp84, [R0_BLOCK])
    tmp87 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp85, 0)).to(tl.float32)
    tl.store(out_ptr17 + (tl.full([1], 0, tl.int32).broadcast_to(R0_BLOCK)), tmp77, None)
    tl.store(out_ptr18 + (tl.full([1], 0, tl.int32).broadcast_to(R0_BLOCK)), tmp67, None)
    tl.store(out_ptr19 + (tl.full([1], 0, tl.int32).broadcast_to(R0_BLOCK)), tmp57, None)
    tl.store(out_ptr20 + (tl.full([1], 0, tl.int32).broadcast_to(R0_BLOCK)), tmp47, None)
    tl.store(out_ptr21 + (tl.full([1], 0, tl.int32).broadcast_to(R0_BLOCK)), tmp37, None)
    tl.store(out_ptr22 + (tl.full([1], 0, tl.int32).broadcast_to(R0_BLOCK)), tmp27, None)
    tl.store(out_ptr23 + (tl.full([1], 0, tl.int32).broadcast_to(R0_BLOCK)), tmp13, None)
    tl.store(out_ptr16 + (tl.full([1], 0, tl.int32).broadcast_to(R0_BLOCK)), tmp87, None)
