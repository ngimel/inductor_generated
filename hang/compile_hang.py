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
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor
@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*i64', 3: '*i64', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp32', 9: '*fp64', 10: '*fp64', 11: '*fp32', 12: '*i32', 13: '*fp32', 14: '*fp64', 15: '*fp64', 16: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'out_ptr22', 'out_ptr23'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr20, out_ptr21, out_ptr22, out_ptr23, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp174 = tl.load(in_ptr2 + (x0), xmask)
    tmp176 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = 0
    tmp3 = 4999
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 - tmp2
    tmp6 = 2
    tmp7 = tl.where((tmp5 < 0) != (tmp6 < 0), tl.where(tmp5 % tmp6 != 0, tmp5 // tmp6 - 1, tmp5 // tmp6), tmp5 // tmp6)
    tmp8 = tmp2 + tmp7
    tmp9 = tl.load(in_ptr1 + (tmp8), None)
    tmp10 = 0.9162907600402832
    tmp11 = tmp0 - tmp10
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp9 >= tmp12
    tmp14 = 1
    tmp15 = tmp8 + tmp14
    tmp16 = tl.where(tmp13, tmp2, tmp15)
    tmp17 = True
    tmp18 = tmp13 & tmp17
    tmp19 = tl.where(tmp18, tmp8, tmp4)
    tmp20 = tmp19 - tmp16
    tmp21 = tl.where((tmp20 < 0) != (tmp6 < 0), tl.where(tmp20 % tmp6 != 0, tmp20 // tmp6 - 1, tmp20 // tmp6), tmp20 // tmp6)
    tmp22 = tmp16 + tmp21
    tmp23 = tmp16 < tmp19
    tmp24 = tl.where(tmp23, tmp22, tmp2)
    tmp25 = tl.load(in_ptr1 + (tmp24), None)
    tmp26 = tmp25 >= tmp12
    tmp27 = tmp26 == 0
    tmp28 = tmp27 & tmp23
    tmp29 = tmp24 + tmp14
    tmp30 = tl.where(tmp28, tmp29, tmp16)
    tmp31 = tmp26 & tmp23
    tmp32 = tl.where(tmp31, tmp24, tmp19)
    tmp33 = tmp30 < tmp32
    tmp34 = tmp32 - tmp30
    tmp35 = tl.where((tmp34 < 0) != (tmp6 < 0), tl.where(tmp34 % tmp6 != 0, tmp34 // tmp6 - 1, tmp34 // tmp6), tmp34 // tmp6)
    tmp36 = tmp30 + tmp35
    tmp37 = tl.where(tmp33, tmp36, tmp2)
    tmp38 = tl.load(in_ptr1 + (tmp37), None)
    tmp39 = tmp38 >= tmp12
    tmp40 = tmp39 == 0
    tmp41 = tmp40 & tmp33
    tmp42 = tmp37 + tmp14
    tmp43 = tl.where(tmp41, tmp42, tmp30)
    tmp44 = tmp39 & tmp33
    tmp45 = tl.where(tmp44, tmp37, tmp32)
    tmp46 = tmp43 < tmp45
    tmp47 = tmp45 - tmp43
    tmp48 = tl.where((tmp47 < 0) != (tmp6 < 0), tl.where(tmp47 % tmp6 != 0, tmp47 // tmp6 - 1, tmp47 // tmp6), tmp47 // tmp6)
    tmp49 = tmp43 + tmp48
    tmp50 = tl.where(tmp46, tmp49, tmp2)
    tmp51 = tl.load(in_ptr1 + (tmp50), None)
    tmp52 = tmp51 >= tmp12
    tmp53 = tmp52 == 0
    tmp54 = tmp53 & tmp46
    tmp55 = tmp50 + tmp14
    tmp56 = tl.where(tmp54, tmp55, tmp43)
    tmp57 = tmp52 & tmp46
    tmp58 = tl.where(tmp57, tmp50, tmp45)
    tmp59 = tmp56 < tmp58
    tmp60 = tmp58 - tmp56
    tmp61 = tl.where((tmp60 < 0) != (tmp6 < 0), tl.where(tmp60 % tmp6 != 0, tmp60 // tmp6 - 1, tmp60 // tmp6), tmp60 // tmp6)
    tmp62 = tmp56 + tmp61
    tmp63 = tl.where(tmp59, tmp62, tmp2)
    tmp64 = tl.load(in_ptr1 + (tmp63), None)
    tmp65 = tmp64 >= tmp12
    tmp66 = tmp65 == 0
    tmp67 = tmp66 & tmp59
    tmp68 = tmp63 + tmp14
    tmp69 = tl.where(tmp67, tmp68, tmp56)
    tmp70 = tmp65 & tmp59
    tmp71 = tl.where(tmp70, tmp63, tmp58)
    tmp72 = tmp69 < tmp71
    tmp73 = tmp71 - tmp69
    tmp74 = tl.where((tmp73 < 0) != (tmp6 < 0), tl.where(tmp73 % tmp6 != 0, tmp73 // tmp6 - 1, tmp73 // tmp6), tmp73 // tmp6)
    tmp75 = tmp69 + tmp74
    tmp76 = tl.where(tmp72, tmp75, tmp2)
    tmp77 = tl.load(in_ptr1 + (tmp76), None)
    tmp78 = tmp77 >= tmp12
    tmp79 = tmp78 == 0
    tmp80 = tmp79 & tmp72
    tmp81 = tmp76 + tmp14
    tmp82 = tl.where(tmp80, tmp81, tmp69)
    tmp83 = tmp78 & tmp72
    tmp84 = tl.where(tmp83, tmp76, tmp71)
    tmp85 = tmp82 < tmp84
    tmp86 = tmp84 - tmp82
    tmp87 = tl.where((tmp86 < 0) != (tmp6 < 0), tl.where(tmp86 % tmp6 != 0, tmp86 // tmp6 - 1, tmp86 // tmp6), tmp86 // tmp6)
    tmp88 = tmp82 + tmp87
    tmp89 = tl.where(tmp85, tmp88, tmp2)
    tmp90 = tl.load(in_ptr1 + (tmp89), None)
    tmp91 = tmp90 >= tmp12
    tmp92 = tmp91 == 0
    tmp93 = tmp92 & tmp85
    tmp94 = tmp89 + tmp14
    tmp95 = tl.where(tmp93, tmp94, tmp82)
    tmp96 = tmp91 & tmp85
    tmp97 = tl.where(tmp96, tmp89, tmp84)
    tmp98 = tmp95 < tmp97
    tmp99 = tmp97 - tmp95
    tmp100 = tl.where((tmp99 < 0) != (tmp6 < 0), tl.where(tmp99 % tmp6 != 0, tmp99 // tmp6 - 1, tmp99 // tmp6), tmp99 // tmp6)
    tmp101 = tmp95 + tmp100
    tmp102 = tl.where(tmp98, tmp101, tmp2)
    tmp103 = tl.load(in_ptr1 + (tmp102), None)
    tmp104 = tmp103 >= tmp12
    tmp105 = tmp104 == 0
    tmp106 = tmp105 & tmp98
    tmp107 = tmp102 + tmp14
    tmp108 = tl.where(tmp106, tmp107, tmp95)
    tmp109 = tmp104 & tmp98
    tmp110 = tl.where(tmp109, tmp102, tmp97)
    tmp111 = tmp108 < tmp110
    tmp112 = tmp110 - tmp108
    tmp113 = tl.where((tmp112 < 0) != (tmp6 < 0), tl.where(tmp112 % tmp6 != 0, tmp112 // tmp6 - 1, tmp112 // tmp6), tmp112 // tmp6)
    tmp114 = tmp108 + tmp113
    tmp115 = tl.where(tmp111, tmp114, tmp2)
    tmp116 = tl.load(in_ptr1 + (tmp115), None)
    tmp117 = tmp116 >= tmp12
    tmp118 = tmp117 == 0
    tmp119 = tmp118 & tmp111
    tmp120 = tmp115 + tmp14
    tmp121 = tl.where(tmp119, tmp120, tmp108)
    tmp122 = tmp117 & tmp111
    tmp123 = tl.where(tmp122, tmp115, tmp110)
    tmp124 = tmp121 < tmp123
    tmp125 = tmp123 - tmp121
    tmp126 = tl.where((tmp125 < 0) != (tmp6 < 0), tl.where(tmp125 % tmp6 != 0, tmp125 // tmp6 - 1, tmp125 // tmp6), tmp125 // tmp6)
    tmp127 = tmp121 + tmp126
    tmp128 = tl.where(tmp124, tmp127, tmp2)
    tmp129 = tl.load(in_ptr1 + (tmp128), None)
    tmp130 = tmp129 >= tmp12
    tmp131 = tmp130 == 0
    tmp132 = tmp131 & tmp124
    tmp133 = tmp128 + tmp14
    tmp134 = tl.where(tmp132, tmp133, tmp121)
    tmp135 = tmp130 & tmp124
    tmp136 = tl.where(tmp135, tmp128, tmp123)
    tmp137 = tmp134 < tmp136
    tmp138 = tmp136 - tmp134
    tmp139 = tl.where((tmp138 < 0) != (tmp6 < 0), tl.where(tmp138 % tmp6 != 0, tmp138 // tmp6 - 1, tmp138 // tmp6), tmp138 // tmp6)
    tmp140 = tmp134 + tmp139
    tmp141 = tl.where(tmp137, tmp140, tmp2)
    tmp142 = tl.load(in_ptr1 + (tmp141), None)
    tmp143 = tmp142 >= tmp12
    tmp144 = tmp143 == 0
    tmp145 = tmp144 & tmp137
    tmp146 = tmp141 + tmp14
    tmp147 = tl.where(tmp145, tmp146, tmp134)
    tmp148 = tmp143 & tmp137
    tmp149 = tl.where(tmp148, tmp141, tmp136)
    tmp150 = tmp147 < tmp149
    tmp151 = tmp149 - tmp147
    tmp152 = tl.where((tmp151 < 0) != (tmp6 < 0), tl.where(tmp151 % tmp6 != 0, tmp151 // tmp6 - 1, tmp151 // tmp6), tmp151 // tmp6)
    tmp153 = tmp147 + tmp152
    tmp154 = tl.where(tmp150, tmp153, tmp2)
    tmp155 = tl.load(in_ptr1 + (tmp154), None)
    tmp156 = tmp155 >= tmp12
    tmp157 = tmp156 == 0
    tmp158 = tmp157 & tmp150
    tmp159 = tmp154 + tmp14
    tmp160 = tl.where(tmp158, tmp159, tmp147)
    tmp161 = tmp156 & tmp150
    tmp162 = tl.where(tmp161, tmp154, tmp149)
    tmp163 = tmp160 < tmp162
    tmp164 = tmp162 - tmp160
    tmp165 = tl.where((tmp164 < 0) != (tmp6 < 0), tl.where(tmp164 % tmp6 != 0, tmp164 // tmp6 - 1, tmp164 // tmp6), tmp164 // tmp6)
    tmp166 = tmp160 + tmp165
    tmp167 = tl.where(tmp163, tmp166, tmp2)
    tmp168 = tl.load(in_ptr1 + (tmp167), None)
    tmp169 = tmp168 >= tmp12
    tmp170 = tmp169 == 0
    tmp171 = tmp170 & tmp163
    tmp172 = tmp167 + tmp14
    tmp173 = tl.where(tmp171, tmp172, tmp160)
    tmp175 = tmp174 == tmp14
    tmp177 = tmp176.to(tl.int64)
    tmp178 = tmp2 > tmp177
    tmp179 = 150
    tmp180 = tmp177 > tmp179
    tmp181 = tl.where(tmp180, tmp2, tmp177)
    tmp182 = tl.where(tmp178, tmp2, tmp181)
    tmp183 = tl.where(tmp175, tmp182, tmp2)
    tmp184 = 5000
    tmp185 = tmp183 * tmp184
    tmp186 = tmp173 + tmp185
    tmp187 = tmp186.to(tl.int32)
    tmp188 = tmp187.to(tl.int64)
    tmp189 = tl.load(in_ptr4 + (tmp188), None)
    tmp190 = 10000.0
    tmp191 = tmp189 > tmp190
    tmp192 = tl.load(in_ptr5 + (tmp188), None)
    tmp193 = tmp192 / tmp189
    tmp194 = tmp193.to(tl.float32)
    tmp195 = 0.9995
    tmp196 = tmp194 * tmp195
    tmp197 = 0.0005
    tmp198 = tmp12 * tmp197
    tmp199 = tmp196 + tmp198
    tmp200 = tl.where(tmp191, tmp199, tmp12)
    tmp201 = tl.load(in_ptr4 + (tmp187), None)
    tmp202 = 1.0
    tmp203 = tmp201 * tmp202
    tmp204 = tl.load(in_ptr5 + (tmp187), None)
    tmp205 = tmp204 * tmp202
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
    tl.store(out_ptr20 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp187, xmask)
    tl.store(out_ptr21 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp200, xmask)
    tl.store(out_ptr22 + (tmp188), tmp203, None)
    tl.store(out_ptr23 + (tmp188), tmp205, None)




''')
async_compile.wait(globals())
del async_compile
