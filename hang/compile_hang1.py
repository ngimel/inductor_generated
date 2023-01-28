from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

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

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex % 12
    x1 = (xindex // 4) % 3
    x0 = xindex % 4
    x2 = (xindex // 12)
    x5 = xindex
    tmp0 = x3
    tmp1 = (-2) + x1
    tmp2 = (-2) + x0
    tmp3 = 3 + x1
    tmp4 = 3 + x0
    tmp5 = 0
    tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp5, tmp1, tmp5))
    tmp7 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 > tmp5, tmp2, tmp5))
    tmp8 = 3
    tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp8, tmp3, tmp8))
    tmp10 = 4
    tmp11 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 < tmp10, tmp4, tmp10))
    tmp12 = tmp6 + tmp5
    tmp13 = tmp7 + tmp5
    tmp14 = 1
    tmp15 = tmp9 - tmp14
    tmp16 = tl.where(tmp12 != tmp12, tmp12, tl.where(tmp12 < tmp15, tmp12, tmp15))
    tmp17 = tmp11 - tmp14
    tmp18 = tl.where(tmp13 != tmp13, tmp13, tl.where(tmp13 < tmp17, tmp13, tmp17))
    tmp19 = tl.load(in_ptr0 + (tmp18 + (4*tmp16) + (12*x2)), xmask)
    tmp20 = tl.load(in_ptr1 + (tmp18 + (4*tmp16) + (12*x2)), xmask)
    tmp21 = tmp19 == tmp0
    tmp22 = 0.0
    tmp23 = tl.where(tmp21, tmp20, tmp22)
    tmp24 = tmp7 + tmp14
    tmp25 = tl.where(tmp24 != tmp24, tmp24, tl.where(tmp24 < tmp17, tmp24, tmp17))
    tmp26 = tl.load(in_ptr0 + (tmp25 + (4*tmp16) + (12*x2)), xmask)
    tmp27 = tl.load(in_ptr1 + (tmp25 + (4*tmp16) + (12*x2)), xmask)
    tmp28 = tmp26 == tmp0
    tmp29 = tmp12 < tmp9
    tmp30 = tmp24 < tmp11
    tmp31 = tmp29 & tmp30
    tmp32 = tmp31 & tmp28
    tmp33 = tmp23 + tmp27
    tmp34 = tl.where(tmp32, tmp33, tmp23)
    tmp35 = 2
    tmp36 = tmp7 + tmp35
    tmp37 = tl.where(tmp36 != tmp36, tmp36, tl.where(tmp36 < tmp17, tmp36, tmp17))
    tmp38 = tl.load(in_ptr0 + (tmp37 + (4*tmp16) + (12*x2)), xmask)
    tmp39 = tl.load(in_ptr1 + (tmp37 + (4*tmp16) + (12*x2)), xmask)
    tmp40 = tmp38 == tmp0
    tmp41 = tmp36 < tmp11
    tmp42 = tmp29 & tmp41
    tmp43 = tmp42 & tmp40
    tmp44 = tmp34 + tmp39
    tmp45 = tl.where(tmp43, tmp44, tmp34)
    tmp46 = tmp7 + tmp8
    tmp47 = tl.where(tmp46 != tmp46, tmp46, tl.where(tmp46 < tmp17, tmp46, tmp17))
    tmp48 = tl.load(in_ptr0 + (tmp47 + (4*tmp16) + (12*x2)), xmask)
    tmp49 = tl.load(in_ptr1 + (tmp47 + (4*tmp16) + (12*x2)), xmask)
    tmp50 = tmp48 == tmp0
    tmp51 = tmp46 < tmp11
    tmp52 = tmp29 & tmp51
    tmp53 = tmp52 & tmp50
    tmp54 = tmp45 + tmp49
    tmp55 = tl.where(tmp53, tmp54, tmp45)
    tmp56 = tmp7 + tmp10
    tmp57 = tl.where(tmp56 != tmp56, tmp56, tl.where(tmp56 < tmp17, tmp56, tmp17))
    tmp58 = tl.load(in_ptr0 + (tmp57 + (4*tmp16) + (12*x2)), xmask)
    tmp59 = tl.load(in_ptr1 + (tmp57 + (4*tmp16) + (12*x2)), xmask)
    tmp60 = tmp58 == tmp0
    tmp61 = tmp56 < tmp11
    tmp62 = tmp29 & tmp61
    tmp63 = tmp62 & tmp60
    tmp64 = tmp55 + tmp59
    tmp65 = tl.where(tmp63, tmp64, tmp55)
    tmp66 = tmp6 + tmp14
    tmp67 = tl.where(tmp66 != tmp66, tmp66, tl.where(tmp66 < tmp15, tmp66, tmp15))
    tmp68 = tl.load(in_ptr0 + (tmp18 + (4*tmp67) + (12*x2)), xmask)
    tmp69 = tl.load(in_ptr1 + (tmp18 + (4*tmp67) + (12*x2)), xmask)
    tmp70 = tmp68 == tmp0
    tmp71 = tmp66 < tmp9
    tmp72 = tmp13 < tmp11
    tmp73 = tmp71 & tmp72
    tmp74 = tmp73 & tmp70
    tmp75 = tmp65 + tmp69
    tmp76 = tl.where(tmp74, tmp75, tmp65)
    tmp77 = tl.load(in_ptr0 + (tmp25 + (4*tmp67) + (12*x2)), xmask)
    tmp78 = tl.load(in_ptr1 + (tmp25 + (4*tmp67) + (12*x2)), xmask)
    tmp79 = tmp77 == tmp0
    tmp80 = tmp71 & tmp30
    tmp81 = tmp80 & tmp79
    tmp82 = tmp76 + tmp78
    tmp83 = tl.where(tmp81, tmp82, tmp76)
    tmp84 = tl.load(in_ptr0 + (tmp37 + (4*tmp67) + (12*x2)), xmask)
    tmp85 = tl.load(in_ptr1 + (tmp37 + (4*tmp67) + (12*x2)), xmask)
    tmp86 = tmp84 == tmp0
    tmp87 = tmp71 & tmp41
    tmp88 = tmp87 & tmp86
    tmp89 = tmp83 + tmp85
    tmp90 = tl.where(tmp88, tmp89, tmp83)
    tmp91 = tl.load(in_ptr0 + (tmp47 + (4*tmp67) + (12*x2)), xmask)
    tmp92 = tl.load(in_ptr1 + (tmp47 + (4*tmp67) + (12*x2)), xmask)
    tmp93 = tmp91 == tmp0
    tmp94 = tmp71 & tmp51
    tmp95 = tmp94 & tmp93
    tmp96 = tmp90 + tmp92
    tmp97 = tl.where(tmp95, tmp96, tmp90)
    tmp98 = tl.load(in_ptr0 + (tmp57 + (4*tmp67) + (12*x2)), xmask)
    tmp99 = tl.load(in_ptr1 + (tmp57 + (4*tmp67) + (12*x2)), xmask)
    tmp100 = tmp98 == tmp0
    tmp101 = tmp71 & tmp61
    tmp102 = tmp101 & tmp100
    tmp103 = tmp97 + tmp99
    tmp104 = tl.where(tmp102, tmp103, tmp97)
    tmp105 = tmp6 + tmp35
    tmp106 = tl.where(tmp105 != tmp105, tmp105, tl.where(tmp105 < tmp15, tmp105, tmp15))
    tmp107 = tl.load(in_ptr0 + (tmp18 + (4*tmp106) + (12*x2)), xmask)
    tmp108 = tl.load(in_ptr1 + (tmp18 + (4*tmp106) + (12*x2)), xmask)
    tmp109 = tmp107 == tmp0
    tmp110 = tmp105 < tmp9
    tmp111 = tmp110 & tmp72
    tmp112 = tmp111 & tmp109
    tmp113 = tmp104 + tmp108
    tmp114 = tl.where(tmp112, tmp113, tmp104)
    tmp115 = tl.load(in_ptr0 + (tmp25 + (4*tmp106) + (12*x2)), xmask)
    tmp116 = tl.load(in_ptr1 + (tmp25 + (4*tmp106) + (12*x2)), xmask)
    tmp117 = tmp115 == tmp0
    tmp118 = tmp110 & tmp30
    tmp119 = tmp118 & tmp117
    tmp120 = tmp114 + tmp116
    tmp121 = tl.where(tmp119, tmp120, tmp114)
    tmp122 = tl.load(in_ptr0 + (tmp37 + (4*tmp106) + (12*x2)), xmask)
    tmp123 = tl.load(in_ptr1 + (tmp37 + (4*tmp106) + (12*x2)), xmask)
    tmp124 = tmp122 == tmp0
    tmp125 = tmp110 & tmp41
    tmp126 = tmp125 & tmp124
    tmp127 = tmp121 + tmp123
    tmp128 = tl.where(tmp126, tmp127, tmp121)
    tmp129 = tl.load(in_ptr0 + (tmp47 + (4*tmp106) + (12*x2)), xmask)
    tmp130 = tl.load(in_ptr1 + (tmp47 + (4*tmp106) + (12*x2)), xmask)
    tmp131 = tmp129 == tmp0
    tmp132 = tmp110 & tmp51
    tmp133 = tmp132 & tmp131
    tmp134 = tmp128 + tmp130
    tmp135 = tl.where(tmp133, tmp134, tmp128)
    tmp136 = tl.load(in_ptr0 + (tmp57 + (4*tmp106) + (12*x2)), xmask)
    tmp137 = tl.load(in_ptr1 + (tmp57 + (4*tmp106) + (12*x2)), xmask)
    tmp138 = tmp136 == tmp0
    tmp139 = tmp110 & tmp61
    tmp140 = tmp139 & tmp138
    tmp141 = tmp135 + tmp137
    tmp142 = tl.where(tmp140, tmp141, tmp135)
    tmp143 = tmp6 + tmp8
    tmp144 = tl.where(tmp143 != tmp143, tmp143, tl.where(tmp143 < tmp15, tmp143, tmp15))
    tmp145 = tl.load(in_ptr0 + (tmp18 + (4*tmp144) + (12*x2)), xmask)
    tmp146 = tl.load(in_ptr1 + (tmp18 + (4*tmp144) + (12*x2)), xmask)
    tmp147 = tmp145 == tmp0
    tmp148 = tmp143 < tmp9
    tmp149 = tmp148 & tmp72
    tmp150 = tmp149 & tmp147
    tmp151 = tmp142 + tmp146
    tmp152 = tl.where(tmp150, tmp151, tmp142)
    tmp153 = tl.load(in_ptr0 + (tmp25 + (4*tmp144) + (12*x2)), xmask)
    tmp154 = tl.load(in_ptr1 + (tmp25 + (4*tmp144) + (12*x2)), xmask)
    tmp155 = tmp153 == tmp0
    tmp156 = tmp148 & tmp30
    tmp157 = tmp156 & tmp155
    tmp158 = tmp152 + tmp154
    tmp159 = tl.where(tmp157, tmp158, tmp152)
    tmp160 = tl.load(in_ptr0 + (tmp37 + (4*tmp144) + (12*x2)), xmask)
    tmp161 = tl.load(in_ptr1 + (tmp37 + (4*tmp144) + (12*x2)), xmask)
    tmp162 = tmp160 == tmp0
    tmp163 = tmp148 & tmp41
    tmp164 = tmp163 & tmp162
    tmp165 = tmp159 + tmp161
    tmp166 = tl.where(tmp164, tmp165, tmp159)
    tmp167 = tl.load(in_ptr0 + (tmp47 + (4*tmp144) + (12*x2)), xmask)
    tmp168 = tl.load(in_ptr1 + (tmp47 + (4*tmp144) + (12*x2)), xmask)
    tmp169 = tmp167 == tmp0
    tmp170 = tmp148 & tmp51
    tmp171 = tmp170 & tmp169
    tmp172 = tmp166 + tmp168
    tmp173 = tl.where(tmp171, tmp172, tmp166)
    tmp174 = tl.load(in_ptr0 + (tmp57 + (4*tmp144) + (12*x2)), xmask)
    tmp175 = tl.load(in_ptr1 + (tmp57 + (4*tmp144) + (12*x2)), xmask)
    tmp176 = tmp174 == tmp0
    tmp177 = tmp148 & tmp61
    tmp178 = tmp177 & tmp176
    tmp179 = tmp173 + tmp175
    tmp180 = tl.where(tmp178, tmp179, tmp173)
    tmp181 = tmp6 + tmp10
    tmp182 = tl.where(tmp181 != tmp181, tmp181, tl.where(tmp181 < tmp15, tmp181, tmp15))
    tmp183 = tl.load(in_ptr0 + (tmp18 + (4*tmp182) + (12*x2)), xmask)
    tmp184 = tl.load(in_ptr1 + (tmp18 + (4*tmp182) + (12*x2)), xmask)
    tmp185 = tmp183 == tmp0
    tmp186 = tmp181 < tmp9
    tmp187 = tmp186 & tmp72
    tmp188 = tmp187 & tmp185
    tmp189 = tmp180 + tmp184
    tmp190 = tl.where(tmp188, tmp189, tmp180)
    tmp191 = tl.load(in_ptr0 + (tmp25 + (4*tmp182) + (12*x2)), xmask)
    tmp192 = tl.load(in_ptr1 + (tmp25 + (4*tmp182) + (12*x2)), xmask)
    tmp193 = tmp191 == tmp0
    tmp194 = tmp186 & tmp30
    tmp195 = tmp194 & tmp193
    tmp196 = tmp190 + tmp192
    tmp197 = tl.where(tmp195, tmp196, tmp190)
    tmp198 = tl.load(in_ptr0 + (tmp37 + (4*tmp182) + (12*x2)), xmask)
    tmp199 = tl.load(in_ptr1 + (tmp37 + (4*tmp182) + (12*x2)), xmask)
    tmp200 = tmp198 == tmp0
    tmp201 = tmp186 & tmp41
    tmp202 = tmp201 & tmp200
    tmp203 = tmp197 + tmp199
    tmp204 = tl.where(tmp202, tmp203, tmp197)
    tmp205 = tl.load(in_ptr0 + (tmp47 + (4*tmp182) + (12*x2)), xmask)
    tmp206 = tl.load(in_ptr1 + (tmp47 + (4*tmp182) + (12*x2)), xmask)
    tmp207 = tmp205 == tmp0
    tmp208 = tmp186 & tmp51
    tmp209 = tmp208 & tmp207
    tmp210 = tmp204 + tmp206
    tmp211 = tl.where(tmp209, tmp210, tmp204)
    tmp212 = tl.load(in_ptr0 + (tmp57 + (4*tmp182) + (12*x2)), xmask)
    tmp213 = tl.load(in_ptr1 + (tmp57 + (4*tmp182) + (12*x2)), xmask)
    tmp214 = tmp212 == tmp0
    tmp215 = tmp186 & tmp61
    tmp216 = tmp215 & tmp214
    tmp217 = tmp211 + tmp213
    tmp218 = tl.where(tmp216, tmp217, tmp211)
    tl.store(out_ptr0 + (x5 + tl.zeros([XBLOCK], tl.int32)), tmp218, xmask)
''')


async_compile.wait(globals())
del async_compile
