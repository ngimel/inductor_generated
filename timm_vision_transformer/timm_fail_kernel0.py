
from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import torch._inductor.lowering

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


triton__0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 512],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp32', 12: '*fp32', 13: '*fp16', 14: '*fp32', 15: '*fp32', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp32', 20: '*fp32', 21: '*fp16', 22: '*fp32', 23: '*fp32', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp32', 28: '*fp32', 29: '*fp16', 30: '*fp32', 31: '*fp32', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp32', 36: '*fp32', 37: '*fp16', 38: '*fp32', 39: '*fp32', 40: '*fp16', 41: '*fp16', 42: '*fp16', 43: '*fp32', 44: '*fp32', 45: '*fp16', 46: '*fp32', 47: '*fp32', 48: '*fp16', 49: '*fp16', 50: '*fp32', 51: '*fp16', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp16', 74: 'i32', 75: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75), equal_to_1=())]})
@triton.jit
def triton__0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr20, out_ptr21, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 197
    tmp6 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x3), xmask)
    tmp13 = tl.load(in_ptr6 + (x3), xmask)
    tmp15 = tl.load(in_ptr7 + (x3), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr5 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp17 = tl.load(in_ptr8 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 + tmp4
        tmp7 = tmp5 - tmp6
        tmp9 = tmp7 * tmp8
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp5 + tmp11
        tmp14 = tmp12 - tmp13
        tmp16 = tmp14 * tmp15
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp12 + tmp18
        tl.store(out_ptr0 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp9, rmask & xmask)
        tl.store(out_ptr1 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
        tl.store(out_ptr2 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask & xmask)
    tmp27 = tl.load(in_ptr11 + (x3), xmask)
    tmp29 = tl.load(in_ptr12 + (x3), xmask)
    tmp34 = tl.load(in_ptr14 + (x3), xmask)
    tmp36 = tl.load(in_ptr15 + (x3), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(out_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp21 = tl.load(in_ptr9 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp24 = tl.load(in_ptr10 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp31 = tl.load(in_ptr13 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp38 = tl.load(in_ptr16 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp23 + tmp25
        tmp28 = tmp26 - tmp27
        tmp30 = tmp28 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp26 + tmp32
        tmp35 = tmp33 - tmp34
        tmp37 = tmp35 * tmp36
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp33 + tmp39
        tl.store(out_ptr3 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp30, rmask & xmask)
        tl.store(out_ptr4 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
        tl.store(out_ptr5 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp40, rmask & xmask)
    tmp48 = tl.load(in_ptr19 + (x3), xmask)
    tmp50 = tl.load(in_ptr20 + (x3), xmask)
    tmp55 = tl.load(in_ptr22 + (x3), xmask)
    tmp57 = tl.load(in_ptr23 + (x3), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp41 = tl.load(out_ptr5 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp42 = tl.load(in_ptr17 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp45 = tl.load(in_ptr18 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp52 = tl.load(in_ptr21 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp59 = tl.load(in_ptr24 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp41 + tmp43
        tmp46 = tmp45.to(tl.float32)
        tmp47 = tmp44 + tmp46
        tmp49 = tmp47 - tmp48
        tmp51 = tmp49 * tmp50
        tmp53 = tmp52.to(tl.float32)
        tmp54 = tmp47 + tmp53
        tmp56 = tmp54 - tmp55
        tmp58 = tmp56 * tmp57
        tmp60 = tmp59.to(tl.float32)
        tmp61 = tmp54 + tmp60
        tl.store(out_ptr6 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp51, rmask & xmask)
        tl.store(out_ptr7 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp58, rmask & xmask)
        tl.store(out_ptr8 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp61, rmask & xmask)
    tmp69 = tl.load(in_ptr27 + (x3), xmask)
    tmp71 = tl.load(in_ptr28 + (x3), xmask)
    tmp76 = tl.load(in_ptr30 + (x3), xmask)
    tmp78 = tl.load(in_ptr31 + (x3), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp62 = tl.load(out_ptr8 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp63 = tl.load(in_ptr25 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp66 = tl.load(in_ptr26 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp73 = tl.load(in_ptr29 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp80 = tl.load(in_ptr32 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tmp62 + tmp64
        tmp67 = tmp66.to(tl.float32)
        tmp68 = tmp65 + tmp67
        tmp70 = tmp68 - tmp69
        tmp72 = tmp70 * tmp71
        tmp74 = tmp73.to(tl.float32)
        tmp75 = tmp68 + tmp74
        tmp77 = tmp75 - tmp76
        tmp79 = tmp77 * tmp78
        tmp81 = tmp80.to(tl.float32)
        tmp82 = tmp75 + tmp81
        tl.store(out_ptr9 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp72, rmask & xmask)
        tl.store(out_ptr10 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp79, rmask & xmask)
        tl.store(out_ptr11 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp82, rmask & xmask)
    tmp90 = tl.load(in_ptr35 + (x3), xmask)
    tmp92 = tl.load(in_ptr36 + (x3), xmask)
    tmp97 = tl.load(in_ptr38 + (x3), xmask)
    tmp99 = tl.load(in_ptr39 + (x3), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp83 = tl.load(out_ptr11 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp84 = tl.load(in_ptr33 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp87 = tl.load(in_ptr34 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp94 = tl.load(in_ptr37 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp101 = tl.load(in_ptr40 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp85 = tmp84.to(tl.float32)
        tmp86 = tmp83 + tmp85
        tmp88 = tmp87.to(tl.float32)
        tmp89 = tmp86 + tmp88
        tmp91 = tmp89 - tmp90
        tmp93 = tmp91 * tmp92
        tmp95 = tmp94.to(tl.float32)
        tmp96 = tmp89 + tmp95
        tmp98 = tmp96 - tmp97
        tmp100 = tmp98 * tmp99
        tmp102 = tmp101.to(tl.float32)
        tmp103 = tmp96 + tmp102
        tl.store(out_ptr12 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp93, rmask & xmask)
        tl.store(out_ptr13 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp100, rmask & xmask)
        tl.store(out_ptr14 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp103, rmask & xmask)
    tmp111 = tl.load(in_ptr43 + (x3), xmask)
    tmp113 = tl.load(in_ptr44 + (x3), xmask)
    tmp118 = tl.load(in_ptr46 + (x3), xmask)
    tmp120 = tl.load(in_ptr47 + (x3), xmask)
    x1 = (xindex // 197)
    _tmp134 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp104 = tl.load(out_ptr14 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp105 = tl.load(in_ptr41 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp108 = tl.load(in_ptr42 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp115 = tl.load(in_ptr45 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp122 = tl.load(in_ptr48 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp128 = tl.load(in_ptr49 + (r2 + (384*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp132 = tl.load(in_ptr50 + (r2), rmask, eviction_policy='evict_last')
        tmp106 = tmp105.to(tl.float32)
        tmp107 = tmp104 + tmp106
        tmp109 = tmp108.to(tl.float32)
        tmp110 = tmp107 + tmp109
        tmp112 = tmp110 - tmp111
        tmp114 = tmp112 * tmp113
        tmp116 = tmp115.to(tl.float32)
        tmp117 = tmp110 + tmp116
        tmp119 = tmp117 - tmp118
        tmp121 = tmp119 * tmp120
        tmp123 = tmp122.to(tl.float32)
        tmp124 = tmp117 + tmp123
        tmp125 = x0
        tmp126 = 0
        tmp127 = tmp125 == tmp126
        tmp129 = tmp128.to(tl.float32)
        tmp130 = 0.0
        tmp131 = tl.where(tmp127, tmp129, tmp130)
        tmp133 = tmp131 * tmp132
        _tmp134 = tl.where(xmask & rmask, _tmp134 + tmp133, _tmp134)
        tl.store(out_ptr15 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp114, rmask & xmask)
        tl.store(out_ptr16 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp121, rmask & xmask)
        tl.store(out_ptr17 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp124, rmask & xmask)
    tmp134 = tl.sum(_tmp134, 1)[:, None]
    tmp148 = tl.load(in_ptr52 + (x3), xmask)
    tmp150 = tl.load(in_ptr53 + (x3), xmask)
    _tmp153 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp138 = tl.load(in_ptr49 + (r2 + (384*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp142 = tl.load(in_ptr50 + (r2), rmask, eviction_policy='evict_last')
        tmp144 = tl.load(out_ptr17 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp145 = tl.load(in_ptr51 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp135 = x0
        tmp136 = 0
        tmp137 = tmp135 == tmp136
        tmp139 = tmp138.to(tl.float32)
        tmp140 = 0.0
        tmp141 = tl.where(tmp137, tmp139, tmp140)
        tmp143 = tmp141 * tmp142
        tmp146 = tmp145.to(tl.float32)
        tmp147 = tmp144 + tmp146
        tmp149 = tmp147 - tmp148
        tmp151 = tmp149 * tmp150
        tmp152 = tmp143 * tmp151
        _tmp153 = tl.where(xmask & rmask, _tmp153 + tmp152, _tmp153)
    tmp153 = tl.sum(_tmp153, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp159 = tl.load(in_ptr49 + (r2 + (384*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp163 = tl.load(in_ptr50 + (r2), rmask, eviction_policy='evict_last')
        tmp167 = tl.load(out_ptr17 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp168 = tl.load(in_ptr51 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp154 = 384.0
        tmp155 = tmp150 / tmp154
        tmp156 = x0
        tmp157 = 0
        tmp158 = tmp156 == tmp157
        tmp160 = tmp159.to(tl.float32)
        tmp161 = 0.0
        tmp162 = tl.where(tmp158, tmp160, tmp161)
        tmp164 = tmp162 * tmp163
        tmp165 = tmp164 * tmp154
        tmp166 = tmp165 - tmp134
        tmp169 = tmp168.to(tl.float32)
        tmp170 = tmp167 + tmp169
        tmp171 = tmp170 - tmp148
        tmp172 = tmp171 * tmp150
        tmp173 = tmp172 * tmp153
        tmp174 = tmp166 - tmp173
        tmp175 = tmp155 * tmp174
        tl.store(out_ptr20 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp175, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp176 = tl.load(out_ptr20 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp177 = tmp176.to(tl.float32)
        tl.store(out_ptr21 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp177, rmask & xmask)
''')




async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, convert_element_type_1, convert_element_type_2, cat, getitem_1, rsqrt, view_1, div, view_7, view_8, getitem_6, rsqrt_1, view_9, addmm_2, view_11, view_12, getitem_8, rsqrt_2, view_13, div_1, view_19, view_20, getitem_13, rsqrt_3, view_21, addmm_6, view_23, view_24, getitem_15, rsqrt_4, view_25, div_2, view_31, view_32, getitem_20, rsqrt_5, view_33, addmm_10, view_35, view_36, getitem_22, rsqrt_6, view_37, div_3, view_43, view_44, getitem_27, rsqrt_7, view_45, addmm_14, view_47, view_48, getitem_29, rsqrt_8, view_49, div_4, view_55, view_56, getitem_34, rsqrt_9, view_57, addmm_18, view_59, view_60, getitem_36, rsqrt_10, view_61, div_5, view_67, view_68, getitem_41, rsqrt_11, view_69, addmm_22, view_71, view_72, getitem_43, rsqrt_12, view_73, div_6, view_79, view_80, getitem_48, rsqrt_13, view_81, addmm_26, view_83, view_84, getitem_50, rsqrt_14, view_85, div_7, view_91, view_92, getitem_55, rsqrt_15, view_93, addmm_30, view_95, view_96, getitem_57, rsqrt_16, view_97, div_8, view_103, view_104, getitem_62, rsqrt_17, view_105, addmm_34, view_107, view_108, getitem_64, rsqrt_18, view_109, div_9, view_115, view_116, getitem_69, rsqrt_19, view_117, addmm_38, view_119, view_120, getitem_71, rsqrt_20, view_121, div_10, view_127, view_128, getitem_76, rsqrt_21, view_129, addmm_42, view_131, view_132, getitem_78, rsqrt_22, view_133, div_11, view_139, view_140, getitem_83, rsqrt_23, view_141, addmm_46, view_143, view_144, getitem_85, rsqrt_24, convert_element_type_173, permute_86, permute_90, permute_94, permute_98, permute_103, permute_104, permute_105, permute_106, permute_109, permute_113, permute_117, permute_121, permute_126, permute_127, permute_128, permute_129, permute_132, permute_136, permute_140, permute_144, permute_149, permute_150, permute_151, permute_152, permute_155, permute_159, permute_163, permute_167, permute_172, permute_173, permute_174, permute_175, permute_178, permute_182, permute_186, permute_190, permute_195, permute_196, permute_197, permute_198, permute_201, permute_205, permute_209, permute_213, permute_218, permute_219, permute_220, permute_221, permute_224, permute_228, permute_232, permute_236, permute_241, permute_242, permute_243, permute_244, permute_247, permute_251, permute_255, permute_259, permute_264, permute_265, permute_266, permute_267, permute_270, permute_274, permute_278, permute_282, permute_287, permute_288, permute_289, permute_290, permute_293, permute_297, permute_301, permute_305, permute_310, permute_311, permute_312, permute_313, permute_316, permute_320, permute_324, permute_328, permute_333, permute_334, permute_335, permute_336, permute_339, permute_343, permute_347, permute_351, permute_356, permute_357, permute_358, permute_359, permute_362, tangents_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf18 = empty_strided((8, 384), (384, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(tangents_1, permute_86, out=buf18)
        del permute_86
        buf0 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf1 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf4 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf10 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf15 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf16 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf17 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf25 = empty_strided((8, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float32)
        buf29 = empty_strided((1576, 384), (384, 1), device='cuda', dtype=torch.float16)
        stream0 = get_cuda_stream(0)
        triton__0.run(cat, primals_2, view_8, getitem_6, rsqrt_1, view_12, getitem_8, rsqrt_2, view_20, view_24, view_32, getitem_20, rsqrt_5, view_36, getitem_22, rsqrt_6, view_44, view_48, view_56, getitem_34, rsqrt_9, view_60, getitem_36, rsqrt_10, view_68, view_72, view_80, getitem_48, rsqrt_13, view_84, getitem_50, rsqrt_14, view_92, view_96, view_104, getitem_62, rsqrt_17, view_108, getitem_64, rsqrt_18, view_116, view_120, view_128, getitem_76, rsqrt_21, view_132, getitem_78, rsqrt_22, view_140, buf18, primals_149, view_144, getitem_85, rsqrt_24, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf25, buf29, 1576, 384, grid=grid(1576), stream=stream0)
        return (buf25, buf29)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((1, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_1 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_2 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float16)
    cat = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_8 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_6 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_2 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_11 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_12 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_8 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_2 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div_1 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_20 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_13 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_6 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_23 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_24 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_15 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_4 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div_2 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_32 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_20 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_5 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_33 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_10 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_35 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_36 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_22 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_6 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div_3 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_44 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_27 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_7 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_14 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_47 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_48 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_29 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_8 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div_4 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_56 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_34 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_9 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_18 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_59 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_60 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_36 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_10 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div_5 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_68 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_41 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_11 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_22 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_71 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_72 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_43 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_12 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div_6 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_80 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_48 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_13 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_26 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_83 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_84 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_50 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_14 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div_7 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_92 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_55 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_15 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_30 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_95 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_96 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_57 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_16 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div_8 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_104 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_62 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_17 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_34 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_107 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_108 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_64 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_18 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div_9 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_116 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_69 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_19 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_38 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_119 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_120 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_71 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_20 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_121 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div_10 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_127 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_128 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_76 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_21 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_42 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_131 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_132 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_78 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_22 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_133 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    div_11 = rand_strided((8, 6, 197, 197), (232854, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    view_139 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_140 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_83 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_23 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_141 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    addmm_46 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_143 = rand_strided((1576, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    view_144 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    getitem_85 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_24 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_173 = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_86 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_90 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_94 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_98 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_103 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_104 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_105 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_106 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_109 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_113 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_117 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_121 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_126 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_127 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_128 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_129 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_132 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_136 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_140 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_144 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_149 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_150 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_151 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_152 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_155 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_159 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_163 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_167 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_172 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_173 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_174 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_175 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_178 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_182 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_186 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_190 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_195 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_196 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_197 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_198 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_201 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_205 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_209 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_213 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_218 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_219 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_220 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_221 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_224 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_228 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_232 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_236 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_241 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_242 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_243 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_244 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_247 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_251 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_255 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_259 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_264 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_265 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_266 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_267 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_270 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_274 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_278 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_282 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_287 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_288 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_289 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_290 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_293 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_297 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_301 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_305 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_310 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_311 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_312 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_313 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_316 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_320 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_324 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_328 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_333 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_334 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_335 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_336 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_339 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_343 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_347 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_351 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_356 = rand_strided((48, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_357 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_358 = rand_strided((48, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_359 = rand_strided((48, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float16)
    permute_362 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float16)
    print_performance(lambda: call([primals_2, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, convert_element_type_1, convert_element_type_2, cat, getitem_1, rsqrt, view_1, div, view_7, view_8, getitem_6, rsqrt_1, view_9, addmm_2, view_11, view_12, getitem_8, rsqrt_2, view_13, div_1, view_19, view_20, getitem_13, rsqrt_3, view_21, addmm_6, view_23, view_24, getitem_15, rsqrt_4, view_25, div_2, view_31, view_32, getitem_20, rsqrt_5, view_33, addmm_10, view_35, view_36, getitem_22, rsqrt_6, view_37, div_3, view_43, view_44, getitem_27, rsqrt_7, view_45, addmm_14, view_47, view_48, getitem_29, rsqrt_8, view_49, div_4, view_55, view_56, getitem_34, rsqrt_9, view_57, addmm_18, view_59, view_60, getitem_36, rsqrt_10, view_61, div_5, view_67, view_68, getitem_41, rsqrt_11, view_69, addmm_22, view_71, view_72, getitem_43, rsqrt_12, view_73, div_6, view_79, view_80, getitem_48, rsqrt_13, view_81, addmm_26, view_83, view_84, getitem_50, rsqrt_14, view_85, div_7, view_91, view_92, getitem_55, rsqrt_15, view_93, addmm_30, view_95, view_96, getitem_57, rsqrt_16, view_97, div_8, view_103, view_104, getitem_62, rsqrt_17, view_105, addmm_34, view_107, view_108, getitem_64, rsqrt_18, view_109, div_9, view_115, view_116, getitem_69, rsqrt_19, view_117, addmm_38, view_119, view_120, getitem_71, rsqrt_20, view_121, div_10, view_127, view_128, getitem_76, rsqrt_21, view_129, addmm_42, view_131, view_132, getitem_78, rsqrt_22, view_133, div_11, view_139, view_140, getitem_83, rsqrt_23, view_141, addmm_46, view_143, view_144, getitem_85, rsqrt_24, convert_element_type_173, permute_86, permute_90, permute_94, permute_98, permute_103, permute_104, permute_105, permute_106, permute_109, permute_113, permute_117, permute_121, permute_126, permute_127, permute_128, permute_129, permute_132, permute_136, permute_140, permute_144, permute_149, permute_150, permute_151, permute_152, permute_155, permute_159, permute_163, permute_167, permute_172, permute_173, permute_174, permute_175, permute_178, permute_182, permute_186, permute_190, permute_195, permute_196, permute_197, permute_198, permute_201, permute_205, permute_209, permute_213, permute_218, permute_219, permute_220, permute_221, permute_224, permute_228, permute_232, permute_236, permute_241, permute_242, permute_243, permute_244, permute_247, permute_251, permute_255, permute_259, permute_264, permute_265, permute_266, permute_267, permute_270, permute_274, permute_278, permute_282, permute_287, permute_288, permute_289, permute_290, permute_293, permute_297, permute_301, permute_305, permute_310, permute_311, permute_312, permute_313, permute_316, permute_320, permute_324, permute_328, permute_333, permute_334, permute_335, permute_336, permute_339, permute_343, permute_347, permute_351, permute_356, permute_357, permute_358, permute_359, permute_362, tangents_1]))
