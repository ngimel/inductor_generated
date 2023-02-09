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

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 6), equal_to_1=())]})
@triton.jit
def triton__0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex % ks0
    x1 = (xindex // ks2) % ks1
    x0 = xindex % ks2
    x2 = (xindex // ks0)
    x5 = xindex
    tmp0 = x3
    tmp1 = (-2) + x1
    tmp2 = (-2) + x0
    tmp3 = 3 + x1
    tmp4 = 3 + x0
    tmp5 = 0
    tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp5, tmp1, tmp5))
    tmp7 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 > tmp5, tmp2, tmp5))
    tmp8 = ks1
    tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp8, tmp3, tmp8))
    tmp10 = ks2
    tmp11 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 < tmp10, tmp4, tmp10))
    tmp12 = tmp6 + tmp5
    tmp13 = tmp7 + tmp5
    tmp14 = 1
    tmp15 = tmp9 - tmp14
    tmp16 = tl.where(tmp12 != tmp12, tmp12, tl.where(tmp12 < tmp15, tmp12, tmp15))
    tmp17 = tmp11 - tmp14
    tmp18 = tl.where(tmp13 != tmp13, tmp13, tl.where(tmp13 < tmp17, tmp13, tmp17))
    tmp19 = tl.load(in_ptr0 + (tmp18 + (ks2*tmp16) + (ks1*ks2*x2)), xmask)
    tmp20 = tl.load(in_ptr1 + (tmp18 + (ks2*tmp16) + (ks1*ks2*x2)), xmask)
    tmp21 = tmp19 == tmp0
    tmp22 = 0.0
    tmp23 = tl.where(tmp21, tmp20, tmp22)
    tmp24 = tmp7 + tmp14
    tmp25 = tl.where(tmp24 != tmp24, tmp24, tl.where(tmp24 < tmp17, tmp24, tmp17))
    tmp26 = tl.load(in_ptr0 + (tmp25 + (ks2*tmp16) + (ks1*ks2*x2)), xmask)
    tmp27 = tl.load(in_ptr1 + (tmp25 + (ks2*tmp16) + (ks1*ks2*x2)), xmask)
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
    tmp38 = tl.load(in_ptr0 + (tmp37 + (ks2*tmp16) + (ks1*ks2*x2)), xmask)
    tmp39 = tl.load(in_ptr1 + (tmp37 + (ks2*tmp16) + (ks1*ks2*x2)), xmask)
    tmp40 = tmp38 == tmp0
    tmp41 = tmp36 < tmp11
    tmp42 = tmp29 & tmp41
    tmp43 = tmp42 & tmp40
    tmp44 = tmp34 + tmp39
    tmp45 = tl.where(tmp43, tmp44, tmp34)
    tmp46 = 3
    tmp47 = tmp7 + tmp46
    tmp48 = tl.where(tmp47 != tmp47, tmp47, tl.where(tmp47 < tmp17, tmp47, tmp17))
    tmp49 = tl.load(in_ptr0 + (tmp48 + (ks2*tmp16) + (ks1*ks2*x2)), xmask)
    tmp50 = tl.load(in_ptr1 + (tmp48 + (ks2*tmp16) + (ks1*ks2*x2)), xmask)
    tmp51 = tmp49 == tmp0
    tmp52 = tmp47 < tmp11
    tmp53 = tmp29 & tmp52
    tmp54 = tmp53 & tmp51
    tmp55 = tmp45 + tmp50
    tmp56 = tl.where(tmp54, tmp55, tmp45)
    tmp57 = 4
    tmp58 = tmp7 + tmp57
    tmp59 = tl.where(tmp58 != tmp58, tmp58, tl.where(tmp58 < tmp17, tmp58, tmp17))
    tmp60 = tl.load(in_ptr0 + (tmp59 + (ks2*tmp16) + (ks1*ks2*x2)), xmask)
    tmp61 = tl.load(in_ptr1 + (tmp59 + (ks2*tmp16) + (ks1*ks2*x2)), xmask)
    tmp62 = tmp60 == tmp0
    tmp63 = tmp58 < tmp11
    tmp64 = tmp29 & tmp63
    tmp65 = tmp64 & tmp62
    tmp66 = tmp56 + tmp61
    tmp67 = tl.where(tmp65, tmp66, tmp56)
    tmp68 = tmp6 + tmp14
    tmp69 = tl.where(tmp68 != tmp68, tmp68, tl.where(tmp68 < tmp15, tmp68, tmp15))
    tmp70 = tl.load(in_ptr0 + (tmp18 + (ks2*tmp69) + (ks1*ks2*x2)), xmask)
    tmp71 = tl.load(in_ptr1 + (tmp18 + (ks2*tmp69) + (ks1*ks2*x2)), xmask)
    tmp72 = tmp70 == tmp0
    tmp73 = tmp68 < tmp9
    tmp74 = tmp13 < tmp11
    tmp75 = tmp73 & tmp74
    tmp76 = tmp75 & tmp72
    tmp77 = tmp67 + tmp71
    tmp78 = tl.where(tmp76, tmp77, tmp67)
    tmp79 = tl.load(in_ptr0 + (tmp25 + (ks2*tmp69) + (ks1*ks2*x2)), xmask)
    tmp80 = tl.load(in_ptr1 + (tmp25 + (ks2*tmp69) + (ks1*ks2*x2)), xmask)
    tmp81 = tmp79 == tmp0
    tmp82 = tmp73 & tmp30
    tmp83 = tmp82 & tmp81
    tmp84 = tmp78 + tmp80
    tmp85 = tl.where(tmp83, tmp84, tmp78)
    tmp86 = tl.load(in_ptr0 + (tmp37 + (ks2*tmp69) + (ks1*ks2*x2)), xmask)
    tmp87 = tl.load(in_ptr1 + (tmp37 + (ks2*tmp69) + (ks1*ks2*x2)), xmask)
    tmp88 = tmp86 == tmp0
    tmp89 = tmp73 & tmp41
    tmp90 = tmp89 & tmp88
    tmp91 = tmp85 + tmp87
    tmp92 = tl.where(tmp90, tmp91, tmp85)
    tmp93 = tl.load(in_ptr0 + (tmp48 + (ks2*tmp69) + (ks1*ks2*x2)), xmask)
    tmp94 = tl.load(in_ptr1 + (tmp48 + (ks2*tmp69) + (ks1*ks2*x2)), xmask)
    tmp95 = tmp93 == tmp0
    tmp96 = tmp73 & tmp52
    tmp97 = tmp96 & tmp95
    tmp98 = tmp92 + tmp94
    tmp99 = tl.where(tmp97, tmp98, tmp92)
    tmp100 = tl.load(in_ptr0 + (tmp59 + (ks2*tmp69) + (ks1*ks2*x2)), xmask)
    tmp101 = tl.load(in_ptr1 + (tmp59 + (ks2*tmp69) + (ks1*ks2*x2)), xmask)
    tmp102 = tmp100 == tmp0
    tmp103 = tmp73 & tmp63
    tmp104 = tmp103 & tmp102
    tmp105 = tmp99 + tmp101
    tmp106 = tl.where(tmp104, tmp105, tmp99)
    tmp107 = tmp6 + tmp35
    tmp108 = tl.where(tmp107 != tmp107, tmp107, tl.where(tmp107 < tmp15, tmp107, tmp15))
    tmp109 = tl.load(in_ptr0 + (tmp18 + (ks2*tmp108) + (ks1*ks2*x2)), xmask)
    tmp110 = tl.load(in_ptr1 + (tmp18 + (ks2*tmp108) + (ks1*ks2*x2)), xmask)
    tmp111 = tmp109 == tmp0
    tmp112 = tmp107 < tmp9
    tmp113 = tmp112 & tmp74
    tmp114 = tmp113 & tmp111
    tmp115 = tmp106 + tmp110
    tmp116 = tl.where(tmp114, tmp115, tmp106)
    tmp117 = tl.load(in_ptr0 + (tmp25 + (ks2*tmp108) + (ks1*ks2*x2)), xmask)
    tmp118 = tl.load(in_ptr1 + (tmp25 + (ks2*tmp108) + (ks1*ks2*x2)), xmask)
    tmp119 = tmp117 == tmp0
    tmp120 = tmp112 & tmp30
    tmp121 = tmp120 & tmp119
    tmp122 = tmp116 + tmp118
    tmp123 = tl.where(tmp121, tmp122, tmp116)
    tmp124 = tl.load(in_ptr0 + (tmp37 + (ks2*tmp108) + (ks1*ks2*x2)), xmask)
    tmp125 = tl.load(in_ptr1 + (tmp37 + (ks2*tmp108) + (ks1*ks2*x2)), xmask)
    tmp126 = tmp124 == tmp0
    tmp127 = tmp112 & tmp41
    tmp128 = tmp127 & tmp126
    tmp129 = tmp123 + tmp125
    tmp130 = tl.where(tmp128, tmp129, tmp123)
    tmp131 = tl.load(in_ptr0 + (tmp48 + (ks2*tmp108) + (ks1*ks2*x2)), xmask)
    tmp132 = tl.load(in_ptr1 + (tmp48 + (ks2*tmp108) + (ks1*ks2*x2)), xmask)
    tmp133 = tmp131 == tmp0
    tmp134 = tmp112 & tmp52
    tmp135 = tmp134 & tmp133
    tmp136 = tmp130 + tmp132
    tmp137 = tl.where(tmp135, tmp136, tmp130)
    tmp138 = tl.load(in_ptr0 + (tmp59 + (ks2*tmp108) + (ks1*ks2*x2)), xmask)
    tmp139 = tl.load(in_ptr1 + (tmp59 + (ks2*tmp108) + (ks1*ks2*x2)), xmask)
    tmp140 = tmp138 == tmp0
    tmp141 = tmp112 & tmp63
    tmp142 = tmp141 & tmp140
    tmp143 = tmp137 + tmp139
    tmp144 = tl.where(tmp142, tmp143, tmp137)
    tmp145 = tmp6 + tmp46
    tmp146 = tl.where(tmp145 != tmp145, tmp145, tl.where(tmp145 < tmp15, tmp145, tmp15))
    tmp147 = tl.load(in_ptr0 + (tmp18 + (ks2*tmp146) + (ks1*ks2*x2)), xmask)
    tmp148 = tl.load(in_ptr1 + (tmp18 + (ks2*tmp146) + (ks1*ks2*x2)), xmask)
    tmp149 = tmp147 == tmp0
    tmp150 = tmp145 < tmp9
    tmp151 = tmp150 & tmp74
    tmp152 = tmp151 & tmp149
    tmp153 = tmp144 + tmp148
    tmp154 = tl.where(tmp152, tmp153, tmp144)
    tmp155 = tl.load(in_ptr0 + (tmp25 + (ks2*tmp146) + (ks1*ks2*x2)), xmask)
    tmp156 = tl.load(in_ptr1 + (tmp25 + (ks2*tmp146) + (ks1*ks2*x2)), xmask)
    tmp157 = tmp155 == tmp0
    tmp158 = tmp150 & tmp30
    tmp159 = tmp158 & tmp157
    tmp160 = tmp154 + tmp156
    tmp161 = tl.where(tmp159, tmp160, tmp154)
    tmp162 = tl.load(in_ptr0 + (tmp37 + (ks2*tmp146) + (ks1*ks2*x2)), xmask)
    tmp163 = tl.load(in_ptr1 + (tmp37 + (ks2*tmp146) + (ks1*ks2*x2)), xmask)
    tmp164 = tmp162 == tmp0
    tmp165 = tmp150 & tmp41
    tmp166 = tmp165 & tmp164
    tmp167 = tmp161 + tmp163
    tmp168 = tl.where(tmp166, tmp167, tmp161)
    tmp169 = tl.load(in_ptr0 + (tmp48 + (ks2*tmp146) + (ks1*ks2*x2)), xmask)
    tmp170 = tl.load(in_ptr1 + (tmp48 + (ks2*tmp146) + (ks1*ks2*x2)), xmask)
    tmp171 = tmp169 == tmp0
    tmp172 = tmp150 & tmp52
    tmp173 = tmp172 & tmp171
    tmp174 = tmp168 + tmp170
    tmp175 = tl.where(tmp173, tmp174, tmp168)
    tmp176 = tl.load(in_ptr0 + (tmp59 + (ks2*tmp146) + (ks1*ks2*x2)), xmask)
    tmp177 = tl.load(in_ptr1 + (tmp59 + (ks2*tmp146) + (ks1*ks2*x2)), xmask)
    tmp178 = tmp176 == tmp0
    tmp179 = tmp150 & tmp63
    tmp180 = tmp179 & tmp178
    tmp181 = tmp175 + tmp177
    tmp182 = tl.where(tmp180, tmp181, tmp175)
    tmp183 = tmp6 + tmp57
    tmp184 = tl.where(tmp183 != tmp183, tmp183, tl.where(tmp183 < tmp15, tmp183, tmp15))
    tmp185 = tl.load(in_ptr0 + (tmp18 + (ks2*tmp184) + (ks1*ks2*x2)), xmask)
    tmp186 = tl.load(in_ptr1 + (tmp18 + (ks2*tmp184) + (ks1*ks2*x2)), xmask)
    tmp187 = tmp185 == tmp0
    tmp188 = tmp183 < tmp9
    tmp189 = tmp188 & tmp74
    tmp190 = tmp189 & tmp187
    tmp191 = tmp182 + tmp186
    tmp192 = tl.where(tmp190, tmp191, tmp182)
    tmp193 = tl.load(in_ptr0 + (tmp25 + (ks2*tmp184) + (ks1*ks2*x2)), xmask)
    tmp194 = tl.load(in_ptr1 + (tmp25 + (ks2*tmp184) + (ks1*ks2*x2)), xmask)
    tmp195 = tmp193 == tmp0
    tmp196 = tmp188 & tmp30
    tmp197 = tmp196 & tmp195
    tmp198 = tmp192 + tmp194
    tmp199 = tl.where(tmp197, tmp198, tmp192)
    tmp200 = tl.load(in_ptr0 + (tmp37 + (ks2*tmp184) + (ks1*ks2*x2)), xmask)
    tmp201 = tl.load(in_ptr1 + (tmp37 + (ks2*tmp184) + (ks1*ks2*x2)), xmask)
    tmp202 = tmp200 == tmp0
    tmp203 = tmp188 & tmp41
    tmp204 = tmp203 & tmp202
    tmp205 = tmp199 + tmp201
    tmp206 = tl.where(tmp204, tmp205, tmp199)
    tmp207 = tl.load(in_ptr0 + (tmp48 + (ks2*tmp184) + (ks1*ks2*x2)), xmask)
    tmp208 = tl.load(in_ptr1 + (tmp48 + (ks2*tmp184) + (ks1*ks2*x2)), xmask)
    tmp209 = tmp207 == tmp0
    tmp210 = tmp188 & tmp52
    tmp211 = tmp210 & tmp209
    tmp212 = tmp206 + tmp208
    tmp213 = tl.where(tmp211, tmp212, tmp206)
    tmp214 = tl.load(in_ptr0 + (tmp59 + (ks2*tmp184) + (ks1*ks2*x2)), xmask)
    tmp215 = tl.load(in_ptr1 + (tmp59 + (ks2*tmp184) + (ks1*ks2*x2)), xmask)
    tmp216 = tmp214 == tmp0
    tmp217 = tmp188 & tmp63
    tmp218 = tmp217 & tmp216
    tmp219 = tmp213 + tmp215
    tmp220 = tl.where(tmp218, tmp219, tmp213)
    tl.store(out_ptr0 + (x5 + tl.zeros([XBLOCK], tl.int32)), tmp220, xmask)
''')


async_compile.wait(globals())
del async_compile
