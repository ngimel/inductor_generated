
from ctypes import c_void_p, c_long
import torch
import math
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

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x2 = (xindex // 75264)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*x2)), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0), xmask)
    tmp8 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 196.0
    tmp3 = tmp1 / tmp2
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp10, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tl.libdevice.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = -0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = 0.3989422804014327
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 * tmp18
    tmp20 = tmp12 + tmp19
    tmp21 = tmp1 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp22, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x2 = (xindex // 75264)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*x2)), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0), xmask)
    tmp8 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 196.0
    tmp3 = tmp1 / tmp2
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10 * tmp5
    tmp12 = tmp9 * tmp11
    tmp13 = tmp7 + tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096, 256], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 3072
    ynumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex
    y1 = yindex
    tmp0 = tl.load(in_ptr0 + ((384*y1) + (75264*(x0 // 384)) + (x0 % 384)), xmask & ymask).to(tl.float32)
    tl.store(out_ptr0 + (y1 + (196*x0) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp16', 6: '*fp32', 7: '*fp16', 8: '*fp32', 9: '*fp16', 10: '*fp32', 11: '*fp16', 12: '*fp32', 13: '*fp16', 14: '*fp32', 15: '*fp16', 16: '*fp32', 17: '*fp16', 18: '*fp32', 19: '*fp16', 20: '*fp32', 21: '*fp16', 22: '*fp32', 23: '*fp16', 24: '*fp32', 25: '*fp16', 26: '*fp32', 27: '*fp16', 28: '*fp32', 29: '*fp16', 30: '*fp32', 31: '*fp16', 32: '*fp32', 33: '*fp16', 34: '*fp32', 35: '*fp16', 36: '*fp32', 37: '*fp16', 38: '*fp32', 39: '*fp16', 40: '*fp32', 41: '*fp16', 42: '*fp32', 43: '*fp16', 44: '*fp32', 45: '*fp16', 46: '*fp16', 47: '*fp32', 48: '*fp16', 49: '*fp32', 50: '*fp16', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: 'i32', 65: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 1568
    ynumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex % 196
    x1 = (xindex // 196)
    y2 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y2), ymask)
    tmp3 = tl.load(in_ptr2 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (y2), ymask)
    tmp8 = tl.load(in_ptr4 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp12 = tl.load(in_ptr5 + (y2), ymask)
    tmp13 = tl.load(in_ptr6 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp17 = tl.load(in_ptr7 + (y2), ymask)
    tmp18 = tl.load(in_ptr8 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp22 = tl.load(in_ptr9 + (y2), ymask)
    tmp23 = tl.load(in_ptr10 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp27 = tl.load(in_ptr11 + (y2), ymask)
    tmp28 = tl.load(in_ptr12 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp32 = tl.load(in_ptr13 + (y2), ymask)
    tmp33 = tl.load(in_ptr14 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp37 = tl.load(in_ptr15 + (y2), ymask)
    tmp38 = tl.load(in_ptr16 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp42 = tl.load(in_ptr17 + (y2), ymask)
    tmp43 = tl.load(in_ptr18 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp47 = tl.load(in_ptr19 + (y2), ymask)
    tmp48 = tl.load(in_ptr20 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp52 = tl.load(in_ptr21 + (y2), ymask)
    tmp53 = tl.load(in_ptr22 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp57 = tl.load(in_ptr23 + (y2), ymask)
    tmp58 = tl.load(in_ptr24 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp62 = tl.load(in_ptr25 + (y2), ymask)
    tmp63 = tl.load(in_ptr26 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp67 = tl.load(in_ptr27 + (y2), ymask)
    tmp68 = tl.load(in_ptr28 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp72 = tl.load(in_ptr29 + (y2), ymask)
    tmp73 = tl.load(in_ptr30 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp77 = tl.load(in_ptr31 + (y2), ymask)
    tmp78 = tl.load(in_ptr32 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp82 = tl.load(in_ptr33 + (y2), ymask)
    tmp83 = tl.load(in_ptr34 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp87 = tl.load(in_ptr35 + (y2), ymask)
    tmp88 = tl.load(in_ptr36 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp92 = tl.load(in_ptr37 + (y2), ymask)
    tmp93 = tl.load(in_ptr38 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp97 = tl.load(in_ptr39 + (y2), ymask)
    tmp98 = tl.load(in_ptr40 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp102 = tl.load(in_ptr41 + (y2), ymask)
    tmp103 = tl.load(in_ptr42 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp107 = tl.load(in_ptr43 + (y2), ymask)
    tmp108 = tl.load(in_ptr44 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp112 = tl.load(in_ptr45 + (y2 + (384*x1)), xmask & ymask).to(tl.float32)
    tmp116 = tl.load(in_ptr46 + (y2), ymask)
    tmp120 = tl.load(in_ptr47 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp122 = tl.load(in_ptr48 + (y2), ymask)
    tmp126 = tl.load(in_ptr49 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp128 = tl.load(in_ptr50 + (y2), ymask)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp6 = tmp1 + tmp5
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp11 = tmp6 + tmp10
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp16 = tmp11 + tmp15
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 * tmp19
    tmp21 = tmp16 + tmp20
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 * tmp24
    tmp26 = tmp21 + tmp25
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 * tmp29
    tmp31 = tmp26 + tmp30
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp36 = tmp31 + tmp35
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 * tmp39
    tmp41 = tmp36 + tmp40
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp42 * tmp44
    tmp46 = tmp41 + tmp45
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp47 * tmp49
    tmp51 = tmp46 + tmp50
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp52 * tmp54
    tmp56 = tmp51 + tmp55
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp57 * tmp59
    tmp61 = tmp56 + tmp60
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp62 * tmp64
    tmp66 = tmp61 + tmp65
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tmp67 * tmp69
    tmp71 = tmp66 + tmp70
    tmp74 = tmp73.to(tl.float32)
    tmp75 = tmp72 * tmp74
    tmp76 = tmp71 + tmp75
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tmp77 * tmp79
    tmp81 = tmp76 + tmp80
    tmp84 = tmp83.to(tl.float32)
    tmp85 = tmp82 * tmp84
    tmp86 = tmp81 + tmp85
    tmp89 = tmp88.to(tl.float32)
    tmp90 = tmp87 * tmp89
    tmp91 = tmp86 + tmp90
    tmp94 = tmp93.to(tl.float32)
    tmp95 = tmp92 * tmp94
    tmp96 = tmp91 + tmp95
    tmp99 = tmp98.to(tl.float32)
    tmp100 = tmp97 * tmp99
    tmp101 = tmp96 + tmp100
    tmp104 = tmp103.to(tl.float32)
    tmp105 = tmp102 * tmp104
    tmp106 = tmp101 + tmp105
    tmp109 = tmp108.to(tl.float32)
    tmp110 = tmp107 * tmp109
    tmp111 = tmp106 + tmp110
    tmp113 = tmp112.to(tl.float32)
    tmp114 = 196.0
    tmp115 = tmp113 / tmp114
    tmp117 = 1.0
    tmp118 = tmp116 * tmp117
    tmp119 = tmp115 * tmp118
    tmp121 = tmp120.to(tl.float32)
    tmp123 = tmp122 * tmp117
    tmp124 = tmp121 * tmp123
    tmp125 = tmp119 + tmp124
    tmp127 = tmp126.to(tl.float32)
    tmp129 = tmp128 * tmp117
    tmp130 = tmp127 * tmp129
    tmp131 = tmp125 + tmp130
    tmp132 = tmp131 * tmp107
    tmp133 = tmp132.to(tl.float32)
    tl.store(out_ptr0 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp11, xmask & ymask)
    tl.store(out_ptr1 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp21, xmask & ymask)
    tl.store(out_ptr2 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp31, xmask & ymask)
    tl.store(out_ptr3 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp41, xmask & ymask)
    tl.store(out_ptr4 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp51, xmask & ymask)
    tl.store(out_ptr5 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp61, xmask & ymask)
    tl.store(out_ptr6 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp71, xmask & ymask)
    tl.store(out_ptr7 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp81, xmask & ymask)
    tl.store(out_ptr8 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp91, xmask & ymask)
    tl.store(out_ptr9 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp101, xmask & ymask)
    tl.store(out_ptr10 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp111, xmask & ymask)
    tl.store(out_ptr11 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp131, xmask & ymask)
    tl.store(in_out_ptr0 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp133, xmask & ymask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp4, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = 196.0
        tmp3 = tmp1 / tmp2
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp3, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp4, xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp35 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp54 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp62 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = 1568
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (384*(((r2 + (121*x0)) // 196) % 8)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 196.0
        tmp6 = tmp4 / tmp5
        tmp7 = tl.load(in_ptr1 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp8 = tl.load(in_ptr2 + (x1 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp9 = tl.load(in_ptr3 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 * tmp10
        tmp12 = tmp7 + tmp11
        tmp13 = tl.load(in_ptr4 + (x1 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp14 = tl.load(in_ptr5 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp13 * tmp15
        tmp17 = tmp12 + tmp16
        tmp18 = 1.0
        tmp19 = tmp17 * tmp18
        tmp20 = tmp6 * tmp19
        tmp21 = tl.where(tmp2, tmp20, 0)
        _tmp22 = tl.where(rmask & xmask, _tmp22 + tmp21, _tmp22)
        tmp23 = tl.load(in_ptr6 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tl.load(in_ptr1 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp26 = tl.load(in_ptr2 + (x1 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp27 = tl.load(in_ptr3 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp26 * tmp28
        tmp30 = tmp25 + tmp29
        tmp31 = 1.0
        tmp32 = tmp30 * tmp31
        tmp33 = tmp24 * tmp32
        tmp34 = tl.where(tmp2, tmp33, 0)
        _tmp35 = tl.where(rmask & xmask, _tmp35 + tmp34, _tmp35)
        tmp36 = tl.load(in_ptr0 + (x1 + (384*(((r2 + (121*x0)) // 196) % 8)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp37 = tmp36.to(tl.float32)
        tmp38 = 196.0
        tmp39 = tmp37 / tmp38
        tmp40 = tl.load(in_ptr7 + (x1 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp41 = 1.0
        tmp42 = tmp40 * tmp41
        tmp43 = tmp39 * tmp42
        tmp44 = tl.load(in_ptr6 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp45 = tmp44.to(tl.float32)
        tmp46 = tl.load(in_ptr8 + (x1 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp47 = tmp46 * tmp41
        tmp48 = tmp45 * tmp47
        tmp49 = tmp43 + tmp48
        tmp50 = tl.load(in_ptr3 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp49 * tmp51
        tmp53 = tl.where(tmp2, tmp52, 0)
        _tmp54 = tl.where(rmask & xmask, _tmp54 + tmp53, _tmp54)
        tmp55 = tl.load(in_ptr9 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp56 = tmp55.to(tl.float32)
        tmp57 = tl.load(in_ptr1 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp58 = 1.0
        tmp59 = tmp57 * tmp58
        tmp60 = tmp56 * tmp59
        tmp61 = tl.where(tmp2, tmp60, 0)
        _tmp62 = tl.where(rmask & xmask, _tmp62 + tmp61, _tmp62)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp22, xmask)
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp35, xmask)
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    tl.store(out_ptr2 + x3, tmp54, xmask)
    tmp62 = tl.sum(_tmp62, 1)[:, None]
    tl.store(out_ptr3 + x3, tmp62, xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = 1568
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*(((r2 + (121*x1)) // 196) % 8)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 196.0
        tmp6 = tmp4 / tmp5
        tmp7 = tl.load(in_ptr1 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp8 = 1.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.load(in_ptr2 + (x0 + (384*((r2 + (121*x1)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        tmp14 = tl.where(tmp2, tmp13, 0)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp15, xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = 1568
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2) + (46464*x1) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.where(tmp2, tmp4, 0)
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp6, xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19968
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1536)
    x0 = xindex % 1536
    _tmp28 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = 1568
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1536*((r2 + (121*x1)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + (1536*((r2 + (121*x1)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = 0.7071067811865476
        tmp10 = tmp8 * tmp9
        tmp11 = tl.libdevice.erf(tmp10)
        tmp12 = 1.0
        tmp13 = tmp11 + tmp12
        tmp14 = 0.5
        tmp15 = tmp13 * tmp14
        tmp16 = tmp8 * tmp8
        tmp17 = -0.5
        tmp18 = tmp16 * tmp17
        tmp19 = tl.exp(tmp18)
        tmp20 = 0.3989422804014327
        tmp21 = tmp19 * tmp20
        tmp22 = tmp8 * tmp21
        tmp23 = tmp15 + tmp22
        tmp24 = tmp4 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tl.where(tmp2, tmp26, 0)
        _tmp28 = tl.where(rmask & xmask, _tmp28 + tmp27, _tmp28)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp28, xmask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 13
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1536*r1)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[1048576, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 589824
    rnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (589824*r1)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp4, xmask)
''')


triton__18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = 1568
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.where(tmp2, tmp4, 0)
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp6, xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[256, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 196
    rnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r1)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp2, xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = 1568
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*((r2 + (121*x1)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.where(tmp2, tmp6, 0)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp8, xmask)
''')


triton__24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tl.libdevice.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = -0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = 0.3989422804014327
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 * tmp18
    tmp20 = tmp12 + tmp19
    tmp21 = tmp1 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp22, xmask)
''')


triton__25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tmp0 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp10, xmask)
''')


triton__26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp28 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp36 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = 1568
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp6 = tl.load(in_ptr2 + (x1 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp7 = tl.load(in_ptr3 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 * tmp8
        tmp10 = tmp5 + tmp9
        tmp11 = 1.0
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp14 = tl.where(tmp2, tmp13, 0)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
        tmp16 = tl.load(in_ptr4 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp17 = tl.load(in_ptr0 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tl.load(in_ptr5 + (x1 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp20 = 1.0
        tmp21 = tmp19 * tmp20
        tmp22 = tmp18 * tmp21
        tmp23 = tmp16 + tmp22
        tmp24 = tl.load(in_ptr3 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp23 * tmp25
        tmp27 = tl.where(tmp2, tmp26, 0)
        _tmp28 = tl.where(rmask & xmask, _tmp28 + tmp27, _tmp28)
        tmp29 = tl.load(in_ptr6 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tl.load(in_ptr1 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp32 = 1.0
        tmp33 = tmp31 * tmp32
        tmp34 = tmp30 * tmp33
        tmp35 = tl.where(tmp2, tmp34, 0)
        _tmp36 = tl.where(rmask & xmask, _tmp36 + tmp35, _tmp36)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp15, xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp28, xmask)
    tmp36 = tl.sum(_tmp36, 1)[:, None]
    tl.store(out_ptr2 + x3, tmp36, xmask)
''')


triton__27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 1568
    ynumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x3 = xindex
    y2 = yindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (y2 + (384*x3)), xmask & ymask)
    tmp1 = tl.load(in_ptr0 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y2), ymask)
    tmp8 = tl.load(in_ptr2 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (y2), ymask)
    tmp14 = tl.load(in_ptr4 + (y2), ymask)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tmp0 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 * tmp11
    tmp13 = tmp7 + tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(in_out_ptr0 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp13, xmask & ymask)
    tl.store(in_out_ptr1 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp16, xmask & ymask)
''')


triton__28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r3)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr4 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 * tmp6
        tmp8 = tmp3 + tmp7
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp1 * tmp10
        _tmp12 = tl.where(rmask & xmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
        tmp16 = tmp3 * tmp9
        tmp17 = tmp14 * tmp16
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp12, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp15, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp18, xmask)
''')


triton__29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = 1568
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp4 = tl.load(in_ptr1 + (x1 + (384*((r2 + (121*x0)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (x1 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tmp3 + tmp9
        tmp11 = tl.load(in_ptr3 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        tmp14 = tl.where(tmp2, tmp13, 0)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp15, xmask)
''')


triton__30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 1568
    ynumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x3 = xindex
    y2 = yindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (y2 + (384*x3)), xmask & ymask)
    tmp1 = tl.load(in_out_ptr0 + (y2 + (384*x3)), xmask & ymask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y2), ymask)
    tmp9 = tl.load(in_ptr2 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (y2), ymask)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp11 * tmp4
    tmp13 = tmp10 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp8 + tmp14
    tl.store(in_out_ptr0 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp15, xmask & ymask)
''')


triton__31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = 1568
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568)) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tl.where(tmp2, tmp3, 0)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp5, xmask)
''')


triton__32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp2 = tmp1.to(tl.float32)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp2, xmask)
''')


triton__33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_74, convert_element_type_1, convert_element_type_2, convolution, view_1, addmm, convert_element_type_7, bmm, view_6, addmm_1, view_8, addmm_2, convert_element_type_17, bmm_1, view_13, addmm_3, view_15, addmm_4, convert_element_type_27, bmm_2, view_20, addmm_5, view_22, addmm_6, convert_element_type_37, bmm_3, view_27, addmm_7, view_29, addmm_8, convert_element_type_47, bmm_4, view_34, addmm_9, view_36, addmm_10, convert_element_type_57, bmm_5, view_41, addmm_11, view_43, addmm_12, convert_element_type_67, bmm_6, view_48, addmm_13, view_50, addmm_14, convert_element_type_77, bmm_7, view_55, addmm_15, view_57, addmm_16, convert_element_type_87, bmm_8, view_62, addmm_17, view_64, addmm_18, convert_element_type_97, bmm_9, view_69, addmm_19, view_71, addmm_20, convert_element_type_107, bmm_10, view_76, addmm_21, view_78, addmm_22, convert_element_type_117, bmm_11, view_83, addmm_23, convert_element_type_126, permute_62, permute_66, permute_70, permute_71, permute_74, permute_79, permute_83, permute_84, permute_87, permute_92, permute_96, permute_97, permute_100, permute_105, permute_109, permute_110, permute_113, permute_118, permute_122, permute_123, permute_126, permute_131, permute_135, permute_136, permute_139, permute_144, permute_148, permute_149, permute_152, permute_157, permute_161, permute_162, permute_165, permute_170, permute_174, permute_175, permute_178, permute_183, permute_187, permute_188, permute_191, permute_196, permute_200, permute_201, permute_204, permute_209, permute_213, permute_214, permute_217, tangents_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf11 = empty_strided((8, 384), (384, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(tangents_1, permute_62, out=buf11)
        del permute_62
        buf21 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float16)
        buf22 = as_strided(buf21, (1568, 384), (384, 1)); del buf21  # reuse
        stream0 = get_cuda_stream(0)
        triton__0.run(buf22, buf11, primals_74, primals_70, 602112, grid=grid(602112), stream=stream0)
        buf23 = empty_strided((1568, 1536), (1536, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(buf22, permute_66, out=buf23)
        del permute_66
        buf31 = empty_strided((8, 196, 1536), (301056, 1536, 1), device='cuda', dtype=torch.float16)
        triton__1.run(buf23, bmm_11, convert_element_type_117, buf31, 2408448, grid=grid(2408448), stream=stream0)
        buf33 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float16)
        extern_kernels.bmm(buf31, permute_71, out=buf33)
        del permute_71
        buf43 = empty_strided((8, 384, 196), (75264, 1, 384), device='cuda', dtype=torch.float16)
        triton__2.run(buf11, primals_74, buf33, primals_72, primals_67, buf43, 602112, grid=grid(602112), stream=stream0)
        buf44 = empty_strided((3072, 196), (196, 1), device='cuda', dtype=torch.float16)
        triton__3.run(buf43, buf44, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf45 = as_strided(buf43, (3072, 196), (196, 1)); del buf43  # reuse
        extern_kernels.mm(buf44, permute_74, out=buf45)
        del permute_74
        buf0 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf1 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf4 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf10 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf54 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float32)
        buf57 = empty_strided((8, 196, 384), (75264, 384, 1), device='cuda', dtype=torch.float16)
        buf58 = as_strided(buf57, (1568, 384), (384, 1)); del buf57  # reuse
        triton__4.run(buf58, convolution, primals_1, addmm, primals_4, addmm_1, primals_7, addmm_2, primals_10, addmm_3, primals_13, addmm_4, primals_16, addmm_5, primals_19, addmm_6, primals_22, addmm_7, primals_25, addmm_8, primals_28, addmm_9, primals_31, addmm_10, primals_34, addmm_11, primals_37, addmm_12, primals_40, addmm_13, primals_43, addmm_14, primals_46, addmm_15, primals_49, addmm_16, primals_52, addmm_17, primals_55, addmm_18, primals_58, addmm_19, primals_61, addmm_20, primals_64, addmm_21, buf11, primals_74, buf33, primals_72, buf45, primals_69, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf54, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_64
        del primals_69
        buf12 = empty_strided((1000, 384), (384, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(tangents_1, (1000, 8), (1, 1000)), convert_element_type_126, out=buf12)
        del convert_element_type_126
        buf13 = empty_strided((1, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        buf15 = as_strided(buf13, (1000, ), (1, )); del buf13  # reuse
        triton__5.run(buf15, tangents_1, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf14 = empty_strided((1000, 384), (384, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf12, buf14, 384000, grid=grid(384000), stream=stream0)
        del buf12
        buf16 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf11, buf16, 384, 1568, grid=grid(384), stream=stream0)
        buf17 = empty_strided((1, 1, 384, 13), (4992, 4992, 13, 1), device='cuda', dtype=torch.float32)
        buf39 = empty_strided((1, 1, 384, 13), (4992, 4992, 13, 1), device='cuda', dtype=torch.float32)
        buf41 = empty_strided((1, 1, 384, 13), (4992, 4992, 13, 1), device='cuda', dtype=torch.float32)
        buf52 = empty_strided((1, 1, 384, 13), (4992, 4992, 13, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf11, buf10, primals_67, addmm_22, primals_70, addmm_23, buf33, primals_74, primals_72, buf45, buf17, buf39, buf41, buf52, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_22
        del buf10
        del primals_67
        del primals_70
        del primals_72
        buf18 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf17, buf18, 384, 13, grid=grid(384), stream=stream0)
        buf19 = as_strided(buf17, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf17  # reuse
        triton__10.run(buf11, primals_74, addmm_23, buf19, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_23
        del buf11
        del primals_74
        buf20 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf19, buf20, 384, 13, grid=grid(384), stream=stream0)
        buf24 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf22, (384, 1568), (1, 384)), view_83, out=buf24)
        del view_83
        buf25 = as_strided(buf19, (1, 384, 13), (4992, 1, 384)); del buf19  # reuse
        triton__12.run(buf22, buf25, 4992, 121, grid=grid(4992), stream=stream0)
        del buf22
        buf26 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf28 = as_strided(buf26, (384, ), (1, )); del buf26  # reuse
        triton__13.run(buf28, buf25, 384, 13, grid=grid(384), stream=stream0)
        buf27 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf24, buf27, 589824, grid=grid(589824), stream=stream0)
        buf29 = empty_strided((1, 1, 1536, 13), (19968, 19968, 1, 1536), device='cuda', dtype=torch.float32)
        triton__15.run(buf23, bmm_11, convert_element_type_117, buf29, 19968, 121, grid=grid(19968), stream=stream0)
        del bmm_11
        del buf23
        del convert_element_type_117
        buf30 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf36 = as_strided(buf30, (1536, ), (1, )); del buf30  # reuse
        triton__16.run(buf36, buf29, 1536, 13, grid=grid(1536), stream=stream0)
        buf32 = empty_strided((8, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float16)
        extern_kernels.bmm(permute_70, buf31, out=buf32)
        del permute_70
        buf34 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf35 = as_strided(buf34, (1536, 384), (1, 1536)); del buf34  # reuse
        triton__17.run(buf35, buf32, 589824, 8, grid=grid(589824), stream=stream0)
        buf37 = as_strided(buf25, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf25  # reuse
        triton__18.run(buf33, buf37, 4992, 121, grid=grid(4992), stream=stream0)
        del buf33
        buf38 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf37, buf38, 384, 13, grid=grid(384), stream=stream0)
        del buf37
        buf40 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf39, buf40, 384, 13, grid=grid(384), stream=stream0)
        buf42 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf41, buf42, 384, 13, grid=grid(384), stream=stream0)
        buf46 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf44, (196, 3072), (1, 196)), view_78, out=buf46)
        del view_78
        buf47 = empty_strided((1, 196, 24), (4704, 1, 196), device='cuda', dtype=torch.float32)
        triton__19.run(buf44, buf47, 4704, 128, grid=grid(4704), stream=stream0)
        buf48 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf50 = as_strided(buf48, (196, ), (1, )); del buf48  # reuse
        triton__20.run(buf50, buf47, 196, 24, grid=grid(196), stream=stream0)
        buf49 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf46, buf49, 38416, grid=grid(38416), stream=stream0)
        buf51 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf45, buf51, 384, 1568, grid=grid(384), stream=stream0)
        buf53 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf52, buf53, 384, 13, grid=grid(384), stream=stream0)
        buf55 = as_strided(buf52, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf52  # reuse
        triton__23.run(buf54, addmm_21, buf55, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_21
        buf56 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf55, buf56, 384, 13, grid=grid(384), stream=stream0)
        buf59 = as_strided(buf31, (1568, 1536), (1536, 1)); del buf31  # reuse
        extern_kernels.mm(buf58, permute_79, out=buf59)
        del permute_79
        buf60 = buf24; del buf24  # reuse
        extern_kernels.mm(as_strided(buf58, (384, 1568), (1, 384)), view_76, out=buf60)
        del view_76
        buf61 = as_strided(buf55, (1, 384, 13), (4992, 1, 384)); del buf55  # reuse
        triton__12.run(buf58, buf61, 4992, 121, grid=grid(4992), stream=stream0)
        buf62 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf64 = as_strided(buf62, (384, ), (1, )); del buf62  # reuse
        triton__13.run(buf64, buf61, 384, 13, grid=grid(384), stream=stream0)
        buf63 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf60, buf63, 589824, grid=grid(589824), stream=stream0)
        buf65 = buf29; del buf29  # reuse
        triton__15.run(buf59, bmm_10, convert_element_type_107, buf65, 19968, 121, grid=grid(19968), stream=stream0)
        buf66 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf72 = as_strided(buf66, (1536, ), (1, )); del buf66  # reuse
        triton__16.run(buf72, buf65, 1536, 13, grid=grid(1536), stream=stream0)
        buf67 = as_strided(buf59, (8, 196, 1536), (301056, 1536, 1)); del buf59  # reuse
        triton__24.run(buf67, bmm_10, convert_element_type_107, 2408448, grid=grid(2408448), stream=stream0)
        del bmm_10
        del convert_element_type_107
        buf68 = buf32; del buf32  # reuse
        extern_kernels.bmm(permute_83, buf67, out=buf68)
        del permute_83
        buf69 = as_strided(buf58, (8, 196, 384), (75264, 384, 1)); del buf58  # reuse
        extern_kernels.bmm(buf67, permute_84, out=buf69)
        del permute_84
        buf70 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf71 = as_strided(buf70, (1536, 384), (1, 1536)); del buf70  # reuse
        triton__17.run(buf71, buf68, 589824, 8, grid=grid(589824), stream=stream0)
        buf73 = as_strided(buf61, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf61  # reuse
        triton__18.run(buf69, buf73, 4992, 121, grid=grid(4992), stream=stream0)
        buf74 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf73, buf74, 384, 13, grid=grid(384), stream=stream0)
        buf79 = as_strided(buf45, (8, 384, 196), (75264, 1, 384)); del buf45  # reuse
        triton__25.run(buf54, buf69, primals_66, primals_61, buf79, 602112, grid=grid(602112), stream=stream0)
        buf80 = buf44; del buf44  # reuse
        triton__3.run(buf79, buf80, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf81 = as_strided(buf79, (3072, 196), (196, 1)); del buf79  # reuse
        extern_kernels.mm(buf80, permute_87, out=buf81)
        del permute_87
        buf75 = as_strided(buf73, (1, 1, 384, 13), (4992, 4992, 13, 1)); del buf73  # reuse
        buf77 = buf41; del buf41  # reuse
        buf88 = buf39; del buf39  # reuse
        triton__26.run(buf69, buf9, primals_61, addmm_20, buf54, primals_66, buf81, buf75, buf77, buf88, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_20
        del buf9
        del primals_61
        buf76 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf75, buf76, 384, 13, grid=grid(384), stream=stream0)
        buf78 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf77, buf78, 384, 13, grid=grid(384), stream=stream0)
        buf82 = buf46; del buf46  # reuse
        extern_kernels.mm(as_strided(buf80, (196, 3072), (1, 196)), view_71, out=buf82)
        del view_71
        buf83 = buf47; del buf47  # reuse
        triton__19.run(buf80, buf83, 4704, 128, grid=grid(4704), stream=stream0)
        buf84 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf86 = as_strided(buf84, (196, ), (1, )); del buf84  # reuse
        triton__20.run(buf86, buf83, 196, 24, grid=grid(196), stream=stream0)
        buf85 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf82, buf85, 38416, grid=grid(38416), stream=stream0)
        buf87 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf81, buf87, 384, 1568, grid=grid(384), stream=stream0)
        buf89 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf88, buf89, 384, 13, grid=grid(384), stream=stream0)
        buf90 = buf54; del buf54  # reuse
        buf93 = as_strided(buf80, (8, 196, 384), (75264, 384, 1)); del buf80  # reuse
        buf94 = as_strided(buf93, (1568, 384), (384, 1)); del buf93  # reuse
        triton__27.run(buf90, buf94, buf69, primals_66, buf81, primals_63, primals_58, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_58
        del primals_63
        del primals_66
        buf91 = as_strided(buf88, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf88  # reuse
        triton__23.run(buf90, addmm_19, buf91, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_19
        buf92 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf91, buf92, 384, 13, grid=grid(384), stream=stream0)
        buf95 = as_strided(buf67, (1568, 1536), (1536, 1)); del buf67  # reuse
        extern_kernels.mm(buf94, permute_92, out=buf95)
        del permute_92
        buf96 = buf60; del buf60  # reuse
        extern_kernels.mm(as_strided(buf94, (384, 1568), (1, 384)), view_69, out=buf96)
        del view_69
        buf97 = as_strided(buf91, (1, 384, 13), (4992, 1, 384)); del buf91  # reuse
        triton__12.run(buf94, buf97, 4992, 121, grid=grid(4992), stream=stream0)
        buf98 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf100 = as_strided(buf98, (384, ), (1, )); del buf98  # reuse
        triton__13.run(buf100, buf97, 384, 13, grid=grid(384), stream=stream0)
        buf99 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf96, buf99, 589824, grid=grid(589824), stream=stream0)
        buf101 = buf65; del buf65  # reuse
        triton__15.run(buf95, bmm_9, convert_element_type_97, buf101, 19968, 121, grid=grid(19968), stream=stream0)
        buf102 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf108 = as_strided(buf102, (1536, ), (1, )); del buf102  # reuse
        triton__16.run(buf108, buf101, 1536, 13, grid=grid(1536), stream=stream0)
        buf103 = as_strided(buf95, (8, 196, 1536), (301056, 1536, 1)); del buf95  # reuse
        triton__24.run(buf103, bmm_9, convert_element_type_97, 2408448, grid=grid(2408448), stream=stream0)
        del bmm_9
        del convert_element_type_97
        buf104 = buf68; del buf68  # reuse
        extern_kernels.bmm(permute_96, buf103, out=buf104)
        del permute_96
        buf105 = as_strided(buf94, (8, 196, 384), (75264, 384, 1)); del buf94  # reuse
        extern_kernels.bmm(buf103, permute_97, out=buf105)
        del permute_97
        buf106 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf107 = as_strided(buf106, (1536, 384), (1, 1536)); del buf106  # reuse
        triton__17.run(buf107, buf104, 589824, 8, grid=grid(589824), stream=stream0)
        buf109 = as_strided(buf97, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf97  # reuse
        triton__18.run(buf105, buf109, 4992, 121, grid=grid(4992), stream=stream0)
        buf110 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf109, buf110, 384, 13, grid=grid(384), stream=stream0)
        buf115 = as_strided(buf81, (8, 384, 196), (75264, 1, 384)); del buf81  # reuse
        triton__25.run(buf90, buf105, primals_60, primals_55, buf115, 602112, grid=grid(602112), stream=stream0)
        buf116 = as_strided(buf69, (3072, 196), (196, 1)); del buf69  # reuse
        triton__3.run(buf115, buf116, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf117 = as_strided(buf115, (3072, 196), (196, 1)); del buf115  # reuse
        extern_kernels.mm(buf116, permute_100, out=buf117)
        del permute_100
        buf111 = as_strided(buf109, (1, 1, 384, 13), (4992, 4992, 13, 1)); del buf109  # reuse
        buf113 = buf77; del buf77  # reuse
        buf124 = buf75; del buf75  # reuse
        triton__26.run(buf105, buf8, primals_55, addmm_18, buf90, primals_60, buf117, buf111, buf113, buf124, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_18
        del buf8
        del primals_55
        buf112 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf111, buf112, 384, 13, grid=grid(384), stream=stream0)
        buf114 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf113, buf114, 384, 13, grid=grid(384), stream=stream0)
        buf118 = buf82; del buf82  # reuse
        extern_kernels.mm(as_strided(buf116, (196, 3072), (1, 196)), view_64, out=buf118)
        del view_64
        buf119 = buf83; del buf83  # reuse
        triton__19.run(buf116, buf119, 4704, 128, grid=grid(4704), stream=stream0)
        buf120 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf122 = as_strided(buf120, (196, ), (1, )); del buf120  # reuse
        triton__20.run(buf122, buf119, 196, 24, grid=grid(196), stream=stream0)
        buf121 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf118, buf121, 38416, grid=grid(38416), stream=stream0)
        buf123 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf117, buf123, 384, 1568, grid=grid(384), stream=stream0)
        buf125 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf124, buf125, 384, 13, grid=grid(384), stream=stream0)
        buf126 = buf90; del buf90  # reuse
        buf129 = as_strided(buf116, (8, 196, 384), (75264, 384, 1)); del buf116  # reuse
        buf130 = as_strided(buf129, (1568, 384), (384, 1)); del buf129  # reuse
        triton__27.run(buf126, buf130, buf105, primals_60, buf117, primals_57, primals_52, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_52
        del primals_57
        del primals_60
        buf127 = as_strided(buf124, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf124  # reuse
        triton__23.run(buf126, addmm_17, buf127, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_17
        buf128 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf127, buf128, 384, 13, grid=grid(384), stream=stream0)
        buf131 = as_strided(buf103, (1568, 1536), (1536, 1)); del buf103  # reuse
        extern_kernels.mm(buf130, permute_105, out=buf131)
        del permute_105
        buf132 = buf96; del buf96  # reuse
        extern_kernels.mm(as_strided(buf130, (384, 1568), (1, 384)), view_62, out=buf132)
        del view_62
        buf133 = as_strided(buf127, (1, 384, 13), (4992, 1, 384)); del buf127  # reuse
        triton__12.run(buf130, buf133, 4992, 121, grid=grid(4992), stream=stream0)
        buf134 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf136 = as_strided(buf134, (384, ), (1, )); del buf134  # reuse
        triton__13.run(buf136, buf133, 384, 13, grid=grid(384), stream=stream0)
        buf135 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf132, buf135, 589824, grid=grid(589824), stream=stream0)
        buf137 = buf101; del buf101  # reuse
        triton__15.run(buf131, bmm_8, convert_element_type_87, buf137, 19968, 121, grid=grid(19968), stream=stream0)
        buf138 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf144 = as_strided(buf138, (1536, ), (1, )); del buf138  # reuse
        triton__16.run(buf144, buf137, 1536, 13, grid=grid(1536), stream=stream0)
        buf139 = as_strided(buf131, (8, 196, 1536), (301056, 1536, 1)); del buf131  # reuse
        triton__24.run(buf139, bmm_8, convert_element_type_87, 2408448, grid=grid(2408448), stream=stream0)
        del bmm_8
        del convert_element_type_87
        buf140 = buf104; del buf104  # reuse
        extern_kernels.bmm(permute_109, buf139, out=buf140)
        del permute_109
        buf141 = as_strided(buf130, (8, 196, 384), (75264, 384, 1)); del buf130  # reuse
        extern_kernels.bmm(buf139, permute_110, out=buf141)
        del permute_110
        buf142 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf143 = as_strided(buf142, (1536, 384), (1, 1536)); del buf142  # reuse
        triton__17.run(buf143, buf140, 589824, 8, grid=grid(589824), stream=stream0)
        buf145 = as_strided(buf133, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf133  # reuse
        triton__18.run(buf141, buf145, 4992, 121, grid=grid(4992), stream=stream0)
        buf146 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf145, buf146, 384, 13, grid=grid(384), stream=stream0)
        buf151 = as_strided(buf117, (8, 384, 196), (75264, 1, 384)); del buf117  # reuse
        triton__25.run(buf126, buf141, primals_54, primals_49, buf151, 602112, grid=grid(602112), stream=stream0)
        buf152 = as_strided(buf105, (3072, 196), (196, 1)); del buf105  # reuse
        triton__3.run(buf151, buf152, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf153 = as_strided(buf151, (3072, 196), (196, 1)); del buf151  # reuse
        extern_kernels.mm(buf152, permute_113, out=buf153)
        del permute_113
        buf147 = as_strided(buf145, (1, 1, 384, 13), (4992, 4992, 13, 1)); del buf145  # reuse
        buf149 = buf113; del buf113  # reuse
        buf160 = buf111; del buf111  # reuse
        triton__26.run(buf141, buf7, primals_49, addmm_16, buf126, primals_54, buf153, buf147, buf149, buf160, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_16
        del buf7
        del primals_49
        buf148 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf147, buf148, 384, 13, grid=grid(384), stream=stream0)
        buf150 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf149, buf150, 384, 13, grid=grid(384), stream=stream0)
        buf154 = buf118; del buf118  # reuse
        extern_kernels.mm(as_strided(buf152, (196, 3072), (1, 196)), view_57, out=buf154)
        del view_57
        buf155 = buf119; del buf119  # reuse
        triton__19.run(buf152, buf155, 4704, 128, grid=grid(4704), stream=stream0)
        buf156 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf158 = as_strided(buf156, (196, ), (1, )); del buf156  # reuse
        triton__20.run(buf158, buf155, 196, 24, grid=grid(196), stream=stream0)
        buf157 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf154, buf157, 38416, grid=grid(38416), stream=stream0)
        buf159 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf153, buf159, 384, 1568, grid=grid(384), stream=stream0)
        buf161 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf160, buf161, 384, 13, grid=grid(384), stream=stream0)
        buf162 = buf126; del buf126  # reuse
        buf165 = as_strided(buf152, (8, 196, 384), (75264, 384, 1)); del buf152  # reuse
        buf166 = as_strided(buf165, (1568, 384), (384, 1)); del buf165  # reuse
        triton__27.run(buf162, buf166, buf141, primals_54, buf153, primals_51, primals_46, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_46
        del primals_51
        del primals_54
        buf163 = as_strided(buf160, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf160  # reuse
        triton__23.run(buf162, addmm_15, buf163, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_15
        buf164 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf163, buf164, 384, 13, grid=grid(384), stream=stream0)
        buf167 = as_strided(buf139, (1568, 1536), (1536, 1)); del buf139  # reuse
        extern_kernels.mm(buf166, permute_118, out=buf167)
        del permute_118
        buf168 = buf132; del buf132  # reuse
        extern_kernels.mm(as_strided(buf166, (384, 1568), (1, 384)), view_55, out=buf168)
        del view_55
        buf169 = as_strided(buf163, (1, 384, 13), (4992, 1, 384)); del buf163  # reuse
        triton__12.run(buf166, buf169, 4992, 121, grid=grid(4992), stream=stream0)
        buf170 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf172 = as_strided(buf170, (384, ), (1, )); del buf170  # reuse
        triton__13.run(buf172, buf169, 384, 13, grid=grid(384), stream=stream0)
        buf171 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf168, buf171, 589824, grid=grid(589824), stream=stream0)
        buf173 = buf137; del buf137  # reuse
        triton__15.run(buf167, bmm_7, convert_element_type_77, buf173, 19968, 121, grid=grid(19968), stream=stream0)
        buf174 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf180 = as_strided(buf174, (1536, ), (1, )); del buf174  # reuse
        triton__16.run(buf180, buf173, 1536, 13, grid=grid(1536), stream=stream0)
        buf175 = as_strided(buf167, (8, 196, 1536), (301056, 1536, 1)); del buf167  # reuse
        triton__24.run(buf175, bmm_7, convert_element_type_77, 2408448, grid=grid(2408448), stream=stream0)
        del bmm_7
        del convert_element_type_77
        buf176 = buf140; del buf140  # reuse
        extern_kernels.bmm(permute_122, buf175, out=buf176)
        del permute_122
        buf177 = as_strided(buf166, (8, 196, 384), (75264, 384, 1)); del buf166  # reuse
        extern_kernels.bmm(buf175, permute_123, out=buf177)
        del permute_123
        buf178 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf179 = as_strided(buf178, (1536, 384), (1, 1536)); del buf178  # reuse
        triton__17.run(buf179, buf176, 589824, 8, grid=grid(589824), stream=stream0)
        buf181 = as_strided(buf169, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf169  # reuse
        triton__18.run(buf177, buf181, 4992, 121, grid=grid(4992), stream=stream0)
        buf182 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf181, buf182, 384, 13, grid=grid(384), stream=stream0)
        buf187 = as_strided(buf153, (8, 384, 196), (75264, 1, 384)); del buf153  # reuse
        triton__25.run(buf162, buf177, primals_48, primals_43, buf187, 602112, grid=grid(602112), stream=stream0)
        buf188 = as_strided(buf141, (3072, 196), (196, 1)); del buf141  # reuse
        triton__3.run(buf187, buf188, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf189 = as_strided(buf187, (3072, 196), (196, 1)); del buf187  # reuse
        extern_kernels.mm(buf188, permute_126, out=buf189)
        del permute_126
        buf183 = as_strided(buf181, (1, 1, 384, 13), (4992, 4992, 13, 1)); del buf181  # reuse
        buf185 = buf149; del buf149  # reuse
        buf196 = buf147; del buf147  # reuse
        triton__26.run(buf177, buf6, primals_43, addmm_14, buf162, primals_48, buf189, buf183, buf185, buf196, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_14
        del buf6
        del primals_43
        buf184 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf183, buf184, 384, 13, grid=grid(384), stream=stream0)
        buf186 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf185, buf186, 384, 13, grid=grid(384), stream=stream0)
        buf190 = buf154; del buf154  # reuse
        extern_kernels.mm(as_strided(buf188, (196, 3072), (1, 196)), view_50, out=buf190)
        del view_50
        buf191 = buf155; del buf155  # reuse
        triton__19.run(buf188, buf191, 4704, 128, grid=grid(4704), stream=stream0)
        buf192 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf194 = as_strided(buf192, (196, ), (1, )); del buf192  # reuse
        triton__20.run(buf194, buf191, 196, 24, grid=grid(196), stream=stream0)
        buf193 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf190, buf193, 38416, grid=grid(38416), stream=stream0)
        buf195 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf189, buf195, 384, 1568, grid=grid(384), stream=stream0)
        buf197 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf196, buf197, 384, 13, grid=grid(384), stream=stream0)
        buf198 = buf162; del buf162  # reuse
        buf201 = as_strided(buf188, (8, 196, 384), (75264, 384, 1)); del buf188  # reuse
        buf202 = as_strided(buf201, (1568, 384), (384, 1)); del buf201  # reuse
        triton__27.run(buf198, buf202, buf177, primals_48, buf189, primals_45, primals_40, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_40
        del primals_45
        del primals_48
        buf199 = as_strided(buf196, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf196  # reuse
        triton__23.run(buf198, addmm_13, buf199, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_13
        buf200 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf199, buf200, 384, 13, grid=grid(384), stream=stream0)
        buf203 = as_strided(buf175, (1568, 1536), (1536, 1)); del buf175  # reuse
        extern_kernels.mm(buf202, permute_131, out=buf203)
        del permute_131
        buf204 = buf168; del buf168  # reuse
        extern_kernels.mm(as_strided(buf202, (384, 1568), (1, 384)), view_48, out=buf204)
        del view_48
        buf205 = as_strided(buf199, (1, 384, 13), (4992, 1, 384)); del buf199  # reuse
        triton__12.run(buf202, buf205, 4992, 121, grid=grid(4992), stream=stream0)
        buf206 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf208 = as_strided(buf206, (384, ), (1, )); del buf206  # reuse
        triton__13.run(buf208, buf205, 384, 13, grid=grid(384), stream=stream0)
        buf207 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf204, buf207, 589824, grid=grid(589824), stream=stream0)
        buf209 = buf173; del buf173  # reuse
        triton__15.run(buf203, bmm_6, convert_element_type_67, buf209, 19968, 121, grid=grid(19968), stream=stream0)
        buf210 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf216 = as_strided(buf210, (1536, ), (1, )); del buf210  # reuse
        triton__16.run(buf216, buf209, 1536, 13, grid=grid(1536), stream=stream0)
        buf211 = as_strided(buf203, (8, 196, 1536), (301056, 1536, 1)); del buf203  # reuse
        triton__24.run(buf211, bmm_6, convert_element_type_67, 2408448, grid=grid(2408448), stream=stream0)
        del bmm_6
        del convert_element_type_67
        buf212 = buf176; del buf176  # reuse
        extern_kernels.bmm(permute_135, buf211, out=buf212)
        del permute_135
        buf213 = as_strided(buf202, (8, 196, 384), (75264, 384, 1)); del buf202  # reuse
        extern_kernels.bmm(buf211, permute_136, out=buf213)
        del permute_136
        buf214 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf215 = as_strided(buf214, (1536, 384), (1, 1536)); del buf214  # reuse
        triton__17.run(buf215, buf212, 589824, 8, grid=grid(589824), stream=stream0)
        buf217 = as_strided(buf205, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf205  # reuse
        triton__18.run(buf213, buf217, 4992, 121, grid=grid(4992), stream=stream0)
        buf218 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf217, buf218, 384, 13, grid=grid(384), stream=stream0)
        buf223 = as_strided(buf189, (8, 384, 196), (75264, 1, 384)); del buf189  # reuse
        triton__25.run(buf198, buf213, primals_42, primals_37, buf223, 602112, grid=grid(602112), stream=stream0)
        buf224 = as_strided(buf177, (3072, 196), (196, 1)); del buf177  # reuse
        triton__3.run(buf223, buf224, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf225 = as_strided(buf223, (3072, 196), (196, 1)); del buf223  # reuse
        extern_kernels.mm(buf224, permute_139, out=buf225)
        del permute_139
        buf219 = as_strided(buf217, (1, 1, 384, 13), (4992, 4992, 13, 1)); del buf217  # reuse
        buf221 = buf185; del buf185  # reuse
        buf232 = buf183; del buf183  # reuse
        triton__26.run(buf213, buf5, primals_37, addmm_12, buf198, primals_42, buf225, buf219, buf221, buf232, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_12
        del buf5
        del primals_37
        buf220 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf219, buf220, 384, 13, grid=grid(384), stream=stream0)
        buf222 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf221, buf222, 384, 13, grid=grid(384), stream=stream0)
        buf226 = buf190; del buf190  # reuse
        extern_kernels.mm(as_strided(buf224, (196, 3072), (1, 196)), view_43, out=buf226)
        del view_43
        buf227 = buf191; del buf191  # reuse
        triton__19.run(buf224, buf227, 4704, 128, grid=grid(4704), stream=stream0)
        buf228 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf230 = as_strided(buf228, (196, ), (1, )); del buf228  # reuse
        triton__20.run(buf230, buf227, 196, 24, grid=grid(196), stream=stream0)
        buf229 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf226, buf229, 38416, grid=grid(38416), stream=stream0)
        buf231 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf225, buf231, 384, 1568, grid=grid(384), stream=stream0)
        buf233 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf232, buf233, 384, 13, grid=grid(384), stream=stream0)
        buf234 = buf198; del buf198  # reuse
        buf237 = as_strided(buf224, (8, 196, 384), (75264, 384, 1)); del buf224  # reuse
        buf238 = as_strided(buf237, (1568, 384), (384, 1)); del buf237  # reuse
        triton__27.run(buf234, buf238, buf213, primals_42, buf225, primals_39, primals_34, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_34
        del primals_39
        del primals_42
        buf235 = as_strided(buf232, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf232  # reuse
        triton__23.run(buf234, addmm_11, buf235, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_11
        buf236 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf235, buf236, 384, 13, grid=grid(384), stream=stream0)
        buf239 = as_strided(buf211, (1568, 1536), (1536, 1)); del buf211  # reuse
        extern_kernels.mm(buf238, permute_144, out=buf239)
        del permute_144
        buf240 = buf204; del buf204  # reuse
        extern_kernels.mm(as_strided(buf238, (384, 1568), (1, 384)), view_41, out=buf240)
        del view_41
        buf241 = as_strided(buf235, (1, 384, 13), (4992, 1, 384)); del buf235  # reuse
        triton__12.run(buf238, buf241, 4992, 121, grid=grid(4992), stream=stream0)
        buf242 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf244 = as_strided(buf242, (384, ), (1, )); del buf242  # reuse
        triton__13.run(buf244, buf241, 384, 13, grid=grid(384), stream=stream0)
        buf243 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf240, buf243, 589824, grid=grid(589824), stream=stream0)
        buf245 = buf209; del buf209  # reuse
        triton__15.run(buf239, bmm_5, convert_element_type_57, buf245, 19968, 121, grid=grid(19968), stream=stream0)
        buf246 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf252 = as_strided(buf246, (1536, ), (1, )); del buf246  # reuse
        triton__16.run(buf252, buf245, 1536, 13, grid=grid(1536), stream=stream0)
        buf247 = as_strided(buf239, (8, 196, 1536), (301056, 1536, 1)); del buf239  # reuse
        triton__24.run(buf247, bmm_5, convert_element_type_57, 2408448, grid=grid(2408448), stream=stream0)
        del bmm_5
        del convert_element_type_57
        buf248 = buf212; del buf212  # reuse
        extern_kernels.bmm(permute_148, buf247, out=buf248)
        del permute_148
        buf249 = as_strided(buf238, (8, 196, 384), (75264, 384, 1)); del buf238  # reuse
        extern_kernels.bmm(buf247, permute_149, out=buf249)
        del permute_149
        buf250 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf251 = as_strided(buf250, (1536, 384), (1, 1536)); del buf250  # reuse
        triton__17.run(buf251, buf248, 589824, 8, grid=grid(589824), stream=stream0)
        buf253 = as_strided(buf241, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf241  # reuse
        triton__18.run(buf249, buf253, 4992, 121, grid=grid(4992), stream=stream0)
        buf254 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf253, buf254, 384, 13, grid=grid(384), stream=stream0)
        buf259 = as_strided(buf225, (8, 384, 196), (75264, 1, 384)); del buf225  # reuse
        triton__25.run(buf234, buf249, primals_36, primals_31, buf259, 602112, grid=grid(602112), stream=stream0)
        buf260 = as_strided(buf213, (3072, 196), (196, 1)); del buf213  # reuse
        triton__3.run(buf259, buf260, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf261 = as_strided(buf259, (3072, 196), (196, 1)); del buf259  # reuse
        extern_kernels.mm(buf260, permute_152, out=buf261)
        del permute_152
        buf255 = as_strided(buf253, (1, 1, 384, 13), (4992, 4992, 13, 1)); del buf253  # reuse
        buf257 = buf221; del buf221  # reuse
        buf268 = buf219; del buf219  # reuse
        triton__26.run(buf249, buf4, primals_31, addmm_10, buf234, primals_36, buf261, buf255, buf257, buf268, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_10
        del buf4
        del primals_31
        buf256 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf255, buf256, 384, 13, grid=grid(384), stream=stream0)
        buf258 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf257, buf258, 384, 13, grid=grid(384), stream=stream0)
        buf262 = buf226; del buf226  # reuse
        extern_kernels.mm(as_strided(buf260, (196, 3072), (1, 196)), view_36, out=buf262)
        del view_36
        buf263 = buf227; del buf227  # reuse
        triton__19.run(buf260, buf263, 4704, 128, grid=grid(4704), stream=stream0)
        buf264 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf266 = as_strided(buf264, (196, ), (1, )); del buf264  # reuse
        triton__20.run(buf266, buf263, 196, 24, grid=grid(196), stream=stream0)
        buf265 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf262, buf265, 38416, grid=grid(38416), stream=stream0)
        buf267 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf261, buf267, 384, 1568, grid=grid(384), stream=stream0)
        buf269 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf268, buf269, 384, 13, grid=grid(384), stream=stream0)
        buf270 = buf234; del buf234  # reuse
        buf273 = as_strided(buf260, (8, 196, 384), (75264, 384, 1)); del buf260  # reuse
        buf274 = as_strided(buf273, (1568, 384), (384, 1)); del buf273  # reuse
        triton__27.run(buf270, buf274, buf249, primals_36, buf261, primals_33, primals_28, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_28
        del primals_33
        del primals_36
        buf271 = as_strided(buf268, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf268  # reuse
        triton__23.run(buf270, addmm_9, buf271, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_9
        buf272 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf271, buf272, 384, 13, grid=grid(384), stream=stream0)
        buf275 = as_strided(buf247, (1568, 1536), (1536, 1)); del buf247  # reuse
        extern_kernels.mm(buf274, permute_157, out=buf275)
        del permute_157
        buf276 = buf240; del buf240  # reuse
        extern_kernels.mm(as_strided(buf274, (384, 1568), (1, 384)), view_34, out=buf276)
        del view_34
        buf277 = as_strided(buf271, (1, 384, 13), (4992, 1, 384)); del buf271  # reuse
        triton__12.run(buf274, buf277, 4992, 121, grid=grid(4992), stream=stream0)
        buf278 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf280 = as_strided(buf278, (384, ), (1, )); del buf278  # reuse
        triton__13.run(buf280, buf277, 384, 13, grid=grid(384), stream=stream0)
        buf279 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf276, buf279, 589824, grid=grid(589824), stream=stream0)
        buf281 = buf245; del buf245  # reuse
        triton__15.run(buf275, bmm_4, convert_element_type_47, buf281, 19968, 121, grid=grid(19968), stream=stream0)
        buf282 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf288 = as_strided(buf282, (1536, ), (1, )); del buf282  # reuse
        triton__16.run(buf288, buf281, 1536, 13, grid=grid(1536), stream=stream0)
        buf283 = as_strided(buf275, (8, 196, 1536), (301056, 1536, 1)); del buf275  # reuse
        triton__24.run(buf283, bmm_4, convert_element_type_47, 2408448, grid=grid(2408448), stream=stream0)
        del bmm_4
        del convert_element_type_47
        buf284 = buf248; del buf248  # reuse
        extern_kernels.bmm(permute_161, buf283, out=buf284)
        del permute_161
        buf285 = as_strided(buf274, (8, 196, 384), (75264, 384, 1)); del buf274  # reuse
        extern_kernels.bmm(buf283, permute_162, out=buf285)
        del permute_162
        buf286 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf287 = as_strided(buf286, (1536, 384), (1, 1536)); del buf286  # reuse
        triton__17.run(buf287, buf284, 589824, 8, grid=grid(589824), stream=stream0)
        buf289 = as_strided(buf277, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf277  # reuse
        triton__18.run(buf285, buf289, 4992, 121, grid=grid(4992), stream=stream0)
        buf290 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf289, buf290, 384, 13, grid=grid(384), stream=stream0)
        buf295 = as_strided(buf261, (8, 384, 196), (75264, 1, 384)); del buf261  # reuse
        triton__25.run(buf270, buf285, primals_30, primals_25, buf295, 602112, grid=grid(602112), stream=stream0)
        buf296 = as_strided(buf249, (3072, 196), (196, 1)); del buf249  # reuse
        triton__3.run(buf295, buf296, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf297 = as_strided(buf295, (3072, 196), (196, 1)); del buf295  # reuse
        extern_kernels.mm(buf296, permute_165, out=buf297)
        del permute_165
        buf291 = as_strided(buf289, (1, 1, 384, 13), (4992, 4992, 13, 1)); del buf289  # reuse
        buf293 = buf257; del buf257  # reuse
        buf304 = buf255; del buf255  # reuse
        triton__26.run(buf285, buf3, primals_25, addmm_8, buf270, primals_30, buf297, buf291, buf293, buf304, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_8
        del buf3
        del primals_25
        buf292 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf291, buf292, 384, 13, grid=grid(384), stream=stream0)
        buf294 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf293, buf294, 384, 13, grid=grid(384), stream=stream0)
        buf298 = buf262; del buf262  # reuse
        extern_kernels.mm(as_strided(buf296, (196, 3072), (1, 196)), view_29, out=buf298)
        del view_29
        buf299 = buf263; del buf263  # reuse
        triton__19.run(buf296, buf299, 4704, 128, grid=grid(4704), stream=stream0)
        buf300 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf302 = as_strided(buf300, (196, ), (1, )); del buf300  # reuse
        triton__20.run(buf302, buf299, 196, 24, grid=grid(196), stream=stream0)
        buf301 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf298, buf301, 38416, grid=grid(38416), stream=stream0)
        buf303 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf297, buf303, 384, 1568, grid=grid(384), stream=stream0)
        buf305 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf304, buf305, 384, 13, grid=grid(384), stream=stream0)
        buf306 = buf270; del buf270  # reuse
        buf309 = as_strided(buf296, (8, 196, 384), (75264, 384, 1)); del buf296  # reuse
        buf310 = as_strided(buf309, (1568, 384), (384, 1)); del buf309  # reuse
        triton__27.run(buf306, buf310, buf285, primals_30, buf297, primals_27, primals_22, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_22
        del primals_27
        del primals_30
        buf307 = as_strided(buf304, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf304  # reuse
        triton__23.run(buf306, addmm_7, buf307, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_7
        buf308 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf307, buf308, 384, 13, grid=grid(384), stream=stream0)
        buf311 = as_strided(buf283, (1568, 1536), (1536, 1)); del buf283  # reuse
        extern_kernels.mm(buf310, permute_170, out=buf311)
        del permute_170
        buf312 = buf276; del buf276  # reuse
        extern_kernels.mm(as_strided(buf310, (384, 1568), (1, 384)), view_27, out=buf312)
        del view_27
        buf313 = as_strided(buf307, (1, 384, 13), (4992, 1, 384)); del buf307  # reuse
        triton__12.run(buf310, buf313, 4992, 121, grid=grid(4992), stream=stream0)
        buf314 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf316 = as_strided(buf314, (384, ), (1, )); del buf314  # reuse
        triton__13.run(buf316, buf313, 384, 13, grid=grid(384), stream=stream0)
        buf315 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf312, buf315, 589824, grid=grid(589824), stream=stream0)
        buf317 = buf281; del buf281  # reuse
        triton__15.run(buf311, bmm_3, convert_element_type_37, buf317, 19968, 121, grid=grid(19968), stream=stream0)
        buf318 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf324 = as_strided(buf318, (1536, ), (1, )); del buf318  # reuse
        triton__16.run(buf324, buf317, 1536, 13, grid=grid(1536), stream=stream0)
        buf319 = as_strided(buf311, (8, 196, 1536), (301056, 1536, 1)); del buf311  # reuse
        triton__24.run(buf319, bmm_3, convert_element_type_37, 2408448, grid=grid(2408448), stream=stream0)
        del bmm_3
        del convert_element_type_37
        buf320 = buf284; del buf284  # reuse
        extern_kernels.bmm(permute_174, buf319, out=buf320)
        del permute_174
        buf321 = as_strided(buf310, (8, 196, 384), (75264, 384, 1)); del buf310  # reuse
        extern_kernels.bmm(buf319, permute_175, out=buf321)
        del permute_175
        buf322 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf323 = as_strided(buf322, (1536, 384), (1, 1536)); del buf322  # reuse
        triton__17.run(buf323, buf320, 589824, 8, grid=grid(589824), stream=stream0)
        buf325 = as_strided(buf313, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf313  # reuse
        triton__18.run(buf321, buf325, 4992, 121, grid=grid(4992), stream=stream0)
        buf326 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf325, buf326, 384, 13, grid=grid(384), stream=stream0)
        buf331 = as_strided(buf297, (8, 384, 196), (75264, 1, 384)); del buf297  # reuse
        triton__25.run(buf306, buf321, primals_24, primals_19, buf331, 602112, grid=grid(602112), stream=stream0)
        buf332 = as_strided(buf285, (3072, 196), (196, 1)); del buf285  # reuse
        triton__3.run(buf331, buf332, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf333 = as_strided(buf331, (3072, 196), (196, 1)); del buf331  # reuse
        extern_kernels.mm(buf332, permute_178, out=buf333)
        del permute_178
        buf327 = as_strided(buf325, (1, 1, 384, 13), (4992, 4992, 13, 1)); del buf325  # reuse
        buf329 = buf293; del buf293  # reuse
        buf340 = buf291; del buf291  # reuse
        triton__26.run(buf321, buf2, primals_19, addmm_6, buf306, primals_24, buf333, buf327, buf329, buf340, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_6
        del buf2
        del primals_19
        buf328 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf327, buf328, 384, 13, grid=grid(384), stream=stream0)
        buf330 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf329, buf330, 384, 13, grid=grid(384), stream=stream0)
        buf334 = buf298; del buf298  # reuse
        extern_kernels.mm(as_strided(buf332, (196, 3072), (1, 196)), view_22, out=buf334)
        del view_22
        buf335 = buf299; del buf299  # reuse
        triton__19.run(buf332, buf335, 4704, 128, grid=grid(4704), stream=stream0)
        buf336 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf338 = as_strided(buf336, (196, ), (1, )); del buf336  # reuse
        triton__20.run(buf338, buf335, 196, 24, grid=grid(196), stream=stream0)
        buf337 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf334, buf337, 38416, grid=grid(38416), stream=stream0)
        buf339 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf333, buf339, 384, 1568, grid=grid(384), stream=stream0)
        buf341 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf340, buf341, 384, 13, grid=grid(384), stream=stream0)
        buf342 = buf306; del buf306  # reuse
        buf345 = as_strided(buf332, (8, 196, 384), (75264, 384, 1)); del buf332  # reuse
        buf346 = as_strided(buf345, (1568, 384), (384, 1)); del buf345  # reuse
        triton__27.run(buf342, buf346, buf321, primals_24, buf333, primals_21, primals_16, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_16
        del primals_21
        del primals_24
        buf343 = as_strided(buf340, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf340  # reuse
        triton__23.run(buf342, addmm_5, buf343, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_5
        buf344 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf343, buf344, 384, 13, grid=grid(384), stream=stream0)
        buf347 = as_strided(buf319, (1568, 1536), (1536, 1)); del buf319  # reuse
        extern_kernels.mm(buf346, permute_183, out=buf347)
        del permute_183
        buf348 = buf312; del buf312  # reuse
        extern_kernels.mm(as_strided(buf346, (384, 1568), (1, 384)), view_20, out=buf348)
        del view_20
        buf349 = as_strided(buf343, (1, 384, 13), (4992, 1, 384)); del buf343  # reuse
        triton__12.run(buf346, buf349, 4992, 121, grid=grid(4992), stream=stream0)
        buf350 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf352 = as_strided(buf350, (384, ), (1, )); del buf350  # reuse
        triton__13.run(buf352, buf349, 384, 13, grid=grid(384), stream=stream0)
        buf351 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf348, buf351, 589824, grid=grid(589824), stream=stream0)
        buf353 = buf317; del buf317  # reuse
        triton__15.run(buf347, bmm_2, convert_element_type_27, buf353, 19968, 121, grid=grid(19968), stream=stream0)
        buf354 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf360 = as_strided(buf354, (1536, ), (1, )); del buf354  # reuse
        triton__16.run(buf360, buf353, 1536, 13, grid=grid(1536), stream=stream0)
        buf355 = as_strided(buf347, (8, 196, 1536), (301056, 1536, 1)); del buf347  # reuse
        triton__24.run(buf355, bmm_2, convert_element_type_27, 2408448, grid=grid(2408448), stream=stream0)
        del bmm_2
        del convert_element_type_27
        buf356 = buf320; del buf320  # reuse
        extern_kernels.bmm(permute_187, buf355, out=buf356)
        del permute_187
        buf357 = as_strided(buf346, (8, 196, 384), (75264, 384, 1)); del buf346  # reuse
        extern_kernels.bmm(buf355, permute_188, out=buf357)
        del permute_188
        buf358 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf359 = as_strided(buf358, (1536, 384), (1, 1536)); del buf358  # reuse
        triton__17.run(buf359, buf356, 589824, 8, grid=grid(589824), stream=stream0)
        buf361 = as_strided(buf349, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf349  # reuse
        triton__18.run(buf357, buf361, 4992, 121, grid=grid(4992), stream=stream0)
        buf362 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf361, buf362, 384, 13, grid=grid(384), stream=stream0)
        buf367 = as_strided(buf333, (8, 384, 196), (75264, 1, 384)); del buf333  # reuse
        triton__25.run(buf342, buf357, primals_18, primals_13, buf367, 602112, grid=grid(602112), stream=stream0)
        buf368 = as_strided(buf321, (3072, 196), (196, 1)); del buf321  # reuse
        triton__3.run(buf367, buf368, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf369 = as_strided(buf367, (3072, 196), (196, 1)); del buf367  # reuse
        extern_kernels.mm(buf368, permute_191, out=buf369)
        del permute_191
        buf363 = as_strided(buf361, (1, 1, 384, 13), (4992, 4992, 13, 1)); del buf361  # reuse
        buf365 = buf329; del buf329  # reuse
        buf376 = buf327; del buf327  # reuse
        triton__26.run(buf357, buf1, primals_13, addmm_4, buf342, primals_18, buf369, buf363, buf365, buf376, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_4
        del buf1
        del primals_13
        buf364 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf363, buf364, 384, 13, grid=grid(384), stream=stream0)
        buf366 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf365, buf366, 384, 13, grid=grid(384), stream=stream0)
        buf370 = buf334; del buf334  # reuse
        extern_kernels.mm(as_strided(buf368, (196, 3072), (1, 196)), view_15, out=buf370)
        del view_15
        buf371 = buf335; del buf335  # reuse
        triton__19.run(buf368, buf371, 4704, 128, grid=grid(4704), stream=stream0)
        buf372 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf374 = as_strided(buf372, (196, ), (1, )); del buf372  # reuse
        triton__20.run(buf374, buf371, 196, 24, grid=grid(196), stream=stream0)
        buf373 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf370, buf373, 38416, grid=grid(38416), stream=stream0)
        buf375 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf369, buf375, 384, 1568, grid=grid(384), stream=stream0)
        buf377 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf376, buf377, 384, 13, grid=grid(384), stream=stream0)
        buf378 = buf342; del buf342  # reuse
        buf381 = as_strided(buf368, (8, 196, 384), (75264, 384, 1)); del buf368  # reuse
        buf382 = as_strided(buf381, (1568, 384), (384, 1)); del buf381  # reuse
        triton__27.run(buf378, buf382, buf357, primals_18, buf369, primals_15, primals_10, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_10
        del primals_15
        del primals_18
        buf379 = as_strided(buf376, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf376  # reuse
        triton__23.run(buf378, addmm_3, buf379, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_3
        buf380 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf379, buf380, 384, 13, grid=grid(384), stream=stream0)
        buf383 = as_strided(buf355, (1568, 1536), (1536, 1)); del buf355  # reuse
        extern_kernels.mm(buf382, permute_196, out=buf383)
        del permute_196
        buf384 = buf348; del buf348  # reuse
        extern_kernels.mm(as_strided(buf382, (384, 1568), (1, 384)), view_13, out=buf384)
        del view_13
        buf385 = as_strided(buf379, (1, 384, 13), (4992, 1, 384)); del buf379  # reuse
        triton__12.run(buf382, buf385, 4992, 121, grid=grid(4992), stream=stream0)
        buf386 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf388 = as_strided(buf386, (384, ), (1, )); del buf386  # reuse
        triton__13.run(buf388, buf385, 384, 13, grid=grid(384), stream=stream0)
        buf387 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf384, buf387, 589824, grid=grid(589824), stream=stream0)
        buf389 = buf353; del buf353  # reuse
        triton__15.run(buf383, bmm_1, convert_element_type_17, buf389, 19968, 121, grid=grid(19968), stream=stream0)
        buf390 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf396 = as_strided(buf390, (1536, ), (1, )); del buf390  # reuse
        triton__16.run(buf396, buf389, 1536, 13, grid=grid(1536), stream=stream0)
        buf391 = as_strided(buf383, (8, 196, 1536), (301056, 1536, 1)); del buf383  # reuse
        triton__24.run(buf391, bmm_1, convert_element_type_17, 2408448, grid=grid(2408448), stream=stream0)
        del bmm_1
        del convert_element_type_17
        buf392 = buf356; del buf356  # reuse
        extern_kernels.bmm(permute_200, buf391, out=buf392)
        del permute_200
        buf393 = as_strided(buf382, (8, 196, 384), (75264, 384, 1)); del buf382  # reuse
        extern_kernels.bmm(buf391, permute_201, out=buf393)
        del permute_201
        buf394 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf395 = as_strided(buf394, (1536, 384), (1, 1536)); del buf394  # reuse
        triton__17.run(buf395, buf392, 589824, 8, grid=grid(589824), stream=stream0)
        buf397 = as_strided(buf385, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf385  # reuse
        triton__18.run(buf393, buf397, 4992, 121, grid=grid(4992), stream=stream0)
        buf398 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf397, buf398, 384, 13, grid=grid(384), stream=stream0)
        buf403 = as_strided(buf369, (8, 384, 196), (75264, 1, 384)); del buf369  # reuse
        triton__25.run(buf378, buf393, primals_12, primals_7, buf403, 602112, grid=grid(602112), stream=stream0)
        buf404 = as_strided(buf357, (3072, 196), (196, 1)); del buf357  # reuse
        triton__3.run(buf403, buf404, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf405 = as_strided(buf403, (3072, 196), (196, 1)); del buf403  # reuse
        extern_kernels.mm(buf404, permute_204, out=buf405)
        del permute_204
        buf399 = as_strided(buf397, (1, 1, 384, 13), (4992, 4992, 13, 1)); del buf397  # reuse
        buf401 = buf365; del buf365  # reuse
        buf412 = buf363; del buf363  # reuse
        triton__26.run(buf393, buf0, primals_7, addmm_2, buf378, primals_12, buf405, buf399, buf401, buf412, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_2
        del buf0
        del primals_7
        buf400 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf399, buf400, 384, 13, grid=grid(384), stream=stream0)
        del buf399
        buf402 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf401, buf402, 384, 13, grid=grid(384), stream=stream0)
        del buf401
        buf406 = buf370; del buf370  # reuse
        extern_kernels.mm(as_strided(buf404, (196, 3072), (1, 196)), view_8, out=buf406)
        del view_8
        buf407 = buf371; del buf371  # reuse
        triton__19.run(buf404, buf407, 4704, 128, grid=grid(4704), stream=stream0)
        buf408 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf410 = as_strided(buf408, (196, ), (1, )); del buf408  # reuse
        triton__20.run(buf410, buf407, 196, 24, grid=grid(196), stream=stream0)
        buf409 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf406, buf409, 38416, grid=grid(38416), stream=stream0)
        buf411 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf405, buf411, 384, 1568, grid=grid(384), stream=stream0)
        buf413 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf412, buf413, 384, 13, grid=grid(384), stream=stream0)
        buf414 = buf378; del buf378  # reuse
        buf417 = as_strided(buf404, (8, 196, 384), (75264, 384, 1)); del buf404  # reuse
        buf418 = as_strided(buf417, (1568, 384), (384, 1)); del buf417  # reuse
        triton__27.run(buf414, buf418, buf393, primals_12, buf405, primals_9, primals_4, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_12
        del primals_4
        del primals_9
        buf415 = as_strided(buf412, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf412  # reuse
        triton__23.run(buf414, addmm_1, buf415, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm_1
        buf416 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf415, buf416, 384, 13, grid=grid(384), stream=stream0)
        buf419 = as_strided(buf391, (1568, 1536), (1536, 1)); del buf391  # reuse
        extern_kernels.mm(buf418, permute_209, out=buf419)
        del permute_209
        buf420 = buf384; del buf384  # reuse
        extern_kernels.mm(as_strided(buf418, (384, 1568), (1, 384)), view_6, out=buf420)
        del view_6
        buf421 = as_strided(buf415, (1, 384, 13), (4992, 1, 384)); del buf415  # reuse
        triton__12.run(buf418, buf421, 4992, 121, grid=grid(4992), stream=stream0)
        buf422 = empty_strided((1, 384), (384, 1), device='cuda', dtype=torch.float32)
        buf424 = as_strided(buf422, (384, ), (1, )); del buf422  # reuse
        triton__13.run(buf424, buf421, 384, 13, grid=grid(384), stream=stream0)
        buf423 = empty_strided((384, 1536), (1536, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf420, buf423, 589824, grid=grid(589824), stream=stream0)
        del buf420
        buf425 = buf389; del buf389  # reuse
        triton__15.run(buf419, bmm, convert_element_type_7, buf425, 19968, 121, grid=grid(19968), stream=stream0)
        buf426 = empty_strided((1, 1, 1536), (1536, 1536, 1), device='cuda', dtype=torch.float32)
        buf432 = as_strided(buf426, (1536, ), (1, )); del buf426  # reuse
        triton__16.run(buf432, buf425, 1536, 13, grid=grid(1536), stream=stream0)
        del buf425
        buf427 = as_strided(buf419, (8, 196, 1536), (301056, 1536, 1)); del buf419  # reuse
        triton__24.run(buf427, bmm, convert_element_type_7, 2408448, grid=grid(2408448), stream=stream0)
        del bmm
        del convert_element_type_7
        buf428 = buf392; del buf392  # reuse
        extern_kernels.bmm(permute_213, buf427, out=buf428)
        del permute_213
        buf429 = as_strided(buf418, (8, 196, 384), (75264, 384, 1)); del buf418  # reuse
        extern_kernels.bmm(buf427, permute_214, out=buf429)
        del buf427
        del permute_214
        buf430 = empty_strided((1, 384, 1536), (589824, 1536, 1), device='cuda', dtype=torch.float32)
        buf431 = as_strided(buf430, (1536, 384), (1, 1536)); del buf430  # reuse
        triton__17.run(buf431, buf428, 589824, 8, grid=grid(589824), stream=stream0)
        del buf428
        buf433 = as_strided(buf421, (1, 1, 384, 13), (4992, 4992, 1, 384)); del buf421  # reuse
        triton__18.run(buf429, buf433, 4992, 121, grid=grid(4992), stream=stream0)
        buf434 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf433, buf434, 384, 13, grid=grid(384), stream=stream0)
        buf438 = as_strided(buf405, (8, 384, 196), (75264, 1, 384)); del buf405  # reuse
        triton__25.run(buf414, buf429, primals_6, primals_1, buf438, 602112, grid=grid(602112), stream=stream0)
        buf439 = as_strided(buf393, (3072, 196), (196, 1)); del buf393  # reuse
        triton__3.run(buf438, buf439, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf440 = as_strided(buf438, (3072, 196), (196, 1)); del buf438  # reuse
        extern_kernels.mm(buf439, permute_217, out=buf440)
        del permute_217
        buf435 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        buf446 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        buf447 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__28.run(buf429, convolution, primals_1, addmm, buf440, buf435, buf446, buf447, 384, 1568, grid=grid(384), stream=stream0)
        del convolution
        del primals_1
        buf436 = as_strided(buf433, (1, 1, 384, 13), (4992, 4992, 13, 1)); del buf433  # reuse
        triton__29.run(buf414, buf429, primals_6, addmm, buf436, 4992, 121, grid=grid(4992), stream=stream0)
        del addmm
        buf437 = empty_strided((1, 1, 384), (384, 384, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf436, buf437, 384, 13, grid=grid(384), stream=stream0)
        buf441 = buf406; del buf406  # reuse
        extern_kernels.mm(as_strided(buf439, (196, 3072), (1, 196)), view_1, out=buf441)
        del view_1
        buf442 = buf407; del buf407  # reuse
        triton__19.run(buf439, buf442, 4704, 128, grid=grid(4704), stream=stream0)
        del buf439
        buf443 = empty_strided((1, 196), (196, 1), device='cuda', dtype=torch.float32)
        buf445 = as_strided(buf443, (196, ), (1, )); del buf443  # reuse
        triton__20.run(buf445, buf442, 196, 24, grid=grid(196), stream=stream0)
        del buf442
        buf444 = empty_strided((196, 196), (196, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf441, buf444, 38416, grid=grid(38416), stream=stream0)
        del buf441
        buf448 = buf429; del buf429  # reuse
        buf449 = as_strided(buf448, (8, 384, 14, 14), (75264, 1, 5376, 384)); del buf448  # reuse
        triton__30.run(buf449, buf414, primals_6, buf440, primals_3, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del buf414
        del buf440
        del primals_3
        del primals_6
        buf450 = as_strided(buf436, (384, 13), (1, 384)); del buf436  # reuse
        triton__31.run(buf449, buf450, 4992, 121, grid=grid(4992), stream=stream0)
        buf455 = empty_strided((384, ), (1, ), device='cuda', dtype=torch.float32)
        triton__32.run(buf450, buf455, 384, 13, grid=grid(384), stream=stream0)
        del buf450
        buf452 = aten.convolution_backward(buf449, convert_element_type_2, convert_element_type_1, [384], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf449
        del convert_element_type_1
        del convert_element_type_2
        buf453 = buf452[1]
        assert_size_stride(buf453, (384, 3, 16, 16), (768, 256, 16, 1))
        del buf452
        buf454 = empty_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cuda', dtype=torch.float32)
        triton__33.run(buf453, buf454, 294912, grid=grid(294912), stream=stream0)
        return (as_strided(buf437, (384, ), (1, )), buf446, buf447, as_strided(buf416, (384, ), (1, )), buf434, buf435, as_strided(buf402, (384, ), (1, )), buf411, buf413, as_strided(buf380, (384, ), (1, )), buf398, buf400, as_strided(buf366, (384, ), (1, )), buf375, buf377, as_strided(buf344, (384, ), (1, )), buf362, buf364, as_strided(buf330, (384, ), (1, )), buf339, buf341, as_strided(buf308, (384, ), (1, )), buf326, buf328, as_strided(buf294, (384, ), (1, )), buf303, buf305, as_strided(buf272, (384, ), (1, )), buf290, buf292, as_strided(buf258, (384, ), (1, )), buf267, buf269, as_strided(buf236, (384, ), (1, )), buf254, buf256, as_strided(buf222, (384, ), (1, )), buf231, buf233, as_strided(buf200, (384, ), (1, )), buf218, buf220, as_strided(buf186, (384, ), (1, )), buf195, buf197, as_strided(buf164, (384, ), (1, )), buf182, buf184, as_strided(buf150, (384, ), (1, )), buf159, buf161, as_strided(buf128, (384, ), (1, )), buf146, buf148, as_strided(buf114, (384, ), (1, )), buf123, buf125, as_strided(buf92, (384, ), (1, )), buf110, buf112, as_strided(buf78, (384, ), (1, )), buf87, buf89, as_strided(buf56, (384, ), (1, )), buf74, buf76, as_strided(buf42, (384, ), (1, )), buf51, buf53, as_strided(buf20, (384, ), (1, )), buf38, buf40, buf16, buf18, buf454, buf455, buf444, buf445, buf431, buf432, buf423, buf424, buf409, buf410, buf395, buf396, buf387, buf388, buf373, buf374, buf359, buf360, buf351, buf352, buf337, buf338, buf323, buf324, buf315, buf316, buf301, buf302, buf287, buf288, buf279, buf280, buf265, buf266, buf251, buf252, buf243, buf244, buf229, buf230, buf215, buf216, buf207, buf208, buf193, buf194, buf179, buf180, buf171, buf172, buf157, buf158, buf143, buf144, buf135, buf136, buf121, buf122, buf107, buf108, buf99, buf100, buf85, buf86, buf71, buf72, buf63, buf64, buf49, buf50, buf35, buf36, buf27, buf28, buf14, buf15, None, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_1 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_2 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float16)
    convolution = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    view_1 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_7 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_6 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_1 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_8 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm_2 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_17 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm_1 = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_13 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_3 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_15 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm_4 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_27 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm_2 = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_20 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_5 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_22 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm_6 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_37 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm_3 = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_27 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_7 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_29 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm_8 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_47 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm_4 = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_34 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_9 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_36 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm_10 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_57 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm_5 = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_41 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_11 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_43 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm_12 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_67 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm_6 = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_48 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_13 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_50 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm_14 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_77 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm_7 = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_55 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_15 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_57 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm_16 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_87 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm_8 = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_62 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_17 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_64 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm_18 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_97 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm_9 = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_69 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_19 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_71 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm_20 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_107 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm_10 = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_76 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_21 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    view_78 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    addmm_22 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_117 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    bmm_11 = rand_strided((8, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float16)
    view_83 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    addmm_23 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_126 = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_62 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    permute_66 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_70 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_71 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_74 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    permute_79 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_83 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_84 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_87 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    permute_92 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_96 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_97 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_100 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    permute_105 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_109 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_110 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_113 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    permute_118 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_122 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_123 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_126 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    permute_131 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_135 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_136 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_139 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    permute_144 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_148 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_149 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_152 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    permute_157 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_161 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_162 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_165 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    permute_170 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_174 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_175 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_178 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    permute_183 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_187 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_188 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_191 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    permute_196 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_200 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_201 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_204 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    permute_209 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    permute_213 = rand_strided((8, 384, 196), (75264, 196, 1), device='cuda:0', dtype=torch.float16)
    permute_214 = rand_strided((8, 1536, 384), (0, 384, 1), device='cuda:0', dtype=torch.float16)
    permute_217 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float16)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float16)
    print_performance(lambda: call([primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_74, convert_element_type_1, convert_element_type_2, convolution, view_1, addmm, convert_element_type_7, bmm, view_6, addmm_1, view_8, addmm_2, convert_element_type_17, bmm_1, view_13, addmm_3, view_15, addmm_4, convert_element_type_27, bmm_2, view_20, addmm_5, view_22, addmm_6, convert_element_type_37, bmm_3, view_27, addmm_7, view_29, addmm_8, convert_element_type_47, bmm_4, view_34, addmm_9, view_36, addmm_10, convert_element_type_57, bmm_5, view_41, addmm_11, view_43, addmm_12, convert_element_type_67, bmm_6, view_48, addmm_13, view_50, addmm_14, convert_element_type_77, bmm_7, view_55, addmm_15, view_57, addmm_16, convert_element_type_87, bmm_8, view_62, addmm_17, view_64, addmm_18, convert_element_type_97, bmm_9, view_69, addmm_19, view_71, addmm_20, convert_element_type_107, bmm_10, view_76, addmm_21, view_78, addmm_22, convert_element_type_117, bmm_11, view_83, addmm_23, convert_element_type_126, permute_62, permute_66, permute_70, permute_71, permute_74, permute_79, permute_83, permute_84, permute_87, permute_92, permute_96, permute_97, permute_100, permute_105, permute_109, permute_110, permute_113, permute_118, permute_122, permute_123, permute_126, permute_131, permute_135, permute_136, permute_139, permute_144, permute_148, permute_149, permute_152, permute_157, permute_161, permute_162, permute_165, permute_170, permute_174, permute_175, permute_178, permute_183, permute_187, permute_188, permute_191, permute_196, permute_200, permute_201, permute_204, permute_209, permute_213, permute_214, permute_217, tangents_1]))
