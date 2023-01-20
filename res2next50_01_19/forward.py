
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

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 262144],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 229376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (229376*x1)) // 112) % 112)) + (12544*x0) + (802816*(((r2 + (229376*x1)) // 12544))) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 8],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 262144],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 229376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (229376*x1)) // 112) % 112)) + (12544*x0) + (802816*(((r2 + (229376*x1)) // 12544))) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 1605632.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 8],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 1605632.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000006228081046
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 8],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 1605632.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1605632.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__8(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 56)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = 0
    tmp2 = tmp0 >= tmp1
    tmp3 = 112
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, float("-inf"))
    tmp13 = 2*x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-112) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp17 & xmask, other=0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, float("-inf"))
    tmp20 = tl.where(tmp19 != tmp19, tmp19, tl.where(tmp19 > tmp12, tmp19, tmp12))
    tmp21 = 1 + (2*x0)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + ((-111) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp25 & xmask, other=0).to(tl.float32)
    tmp27 = tl.where(tmp25, tmp26, float("-inf"))
    tmp28 = tl.where(tmp27 != tmp27, tmp27, tl.where(tmp27 > tmp20, tmp27, tmp20))
    tmp29 = 2*x1
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + ((-1) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp33 & xmask, other=0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, float("-inf"))
    tmp36 = tl.where(tmp35 != tmp35, tmp35, tl.where(tmp35 > tmp28, tmp35, tmp28))
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + ((2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp37 & xmask, other=0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, float("-inf"))
    tmp40 = tl.where(tmp39 != tmp39, tmp39, tl.where(tmp39 > tmp36, tmp39, tmp36))
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (1 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp41 & xmask, other=0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, float("-inf"))
    tmp44 = tl.where(tmp43 != tmp43, tmp43, tl.where(tmp43 > tmp40, tmp43, tmp40))
    tmp45 = 1 + (2*x1)
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (111 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp49 & xmask, other=0).to(tl.float32)
    tmp51 = tl.where(tmp49, tmp50, float("-inf"))
    tmp52 = tl.where(tmp51 != tmp51, tmp51, tl.where(tmp51 > tmp44, tmp51, tmp44))
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (112 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp53 & xmask, other=0).to(tl.float32)
    tmp55 = tl.where(tmp53, tmp54, float("-inf"))
    tmp56 = tl.where(tmp55 != tmp55, tmp55, tl.where(tmp55 > tmp52, tmp55, tmp52))
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (113 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp57 & xmask, other=0).to(tl.float32)
    tmp59 = tl.where(tmp57, tmp58, float("-inf"))
    tmp60 = tl.where(tmp59 != tmp59, tmp59, tl.where(tmp59 > tmp56, tmp59, tmp56))
    tmp61 = tl.load(in_ptr0 + ((-113) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0).to(tl.float32)
    tmp62 = tl.where(tmp10, tmp61, float("-inf"))
    tmp63 = (-113) + (2*x0) + (224*x1)
    tmp64 = tl.load(in_ptr0 + ((-112) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp17 & xmask, other=0).to(tl.float32)
    tmp65 = tl.where(tmp17, tmp64, float("-inf"))
    tmp66 = (-112) + (2*x0) + (224*x1)
    tmp67 = tmp65 > tmp62
    tmp68 = tl.where(tmp67, tmp66, tmp63)
    tmp69 = tl.where(tmp65 != tmp65, tmp65, tl.where(tmp65 > tmp62, tmp65, tmp62))
    tmp70 = tl.load(in_ptr0 + ((-111) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp25 & xmask, other=0).to(tl.float32)
    tmp71 = tl.where(tmp25, tmp70, float("-inf"))
    tmp72 = (-111) + (2*x0) + (224*x1)
    tmp73 = tmp71 > tmp69
    tmp74 = tl.where(tmp73, tmp72, tmp68)
    tmp75 = tl.where(tmp71 != tmp71, tmp71, tl.where(tmp71 > tmp69, tmp71, tmp69))
    tmp76 = tl.load(in_ptr0 + ((-1) + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp33 & xmask, other=0).to(tl.float32)
    tmp77 = tl.where(tmp33, tmp76, float("-inf"))
    tmp78 = (-1) + (2*x0) + (224*x1)
    tmp79 = tmp77 > tmp75
    tmp80 = tl.where(tmp79, tmp78, tmp74)
    tmp81 = tl.where(tmp77 != tmp77, tmp77, tl.where(tmp77 > tmp75, tmp77, tmp75))
    tmp82 = tl.load(in_ptr0 + ((2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp37 & xmask, other=0).to(tl.float32)
    tmp83 = tl.where(tmp37, tmp82, float("-inf"))
    tmp84 = (2*x0) + (224*x1)
    tmp85 = tmp83 > tmp81
    tmp86 = tl.where(tmp85, tmp84, tmp80)
    tmp87 = tl.where(tmp83 != tmp83, tmp83, tl.where(tmp83 > tmp81, tmp83, tmp81))
    tmp88 = tl.load(in_ptr0 + (1 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp41 & xmask, other=0).to(tl.float32)
    tmp89 = tl.where(tmp41, tmp88, float("-inf"))
    tmp90 = 1 + (2*x0) + (224*x1)
    tmp91 = tmp89 > tmp87
    tmp92 = tl.where(tmp91, tmp90, tmp86)
    tmp93 = tl.where(tmp89 != tmp89, tmp89, tl.where(tmp89 > tmp87, tmp89, tmp87))
    tmp94 = tl.load(in_ptr0 + (111 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp49 & xmask, other=0).to(tl.float32)
    tmp95 = tl.where(tmp49, tmp94, float("-inf"))
    tmp96 = 111 + (2*x0) + (224*x1)
    tmp97 = tmp95 > tmp93
    tmp98 = tl.where(tmp97, tmp96, tmp92)
    tmp99 = tl.where(tmp95 != tmp95, tmp95, tl.where(tmp95 > tmp93, tmp95, tmp93))
    tmp100 = tl.load(in_ptr0 + (112 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp53 & xmask, other=0).to(tl.float32)
    tmp101 = tl.where(tmp53, tmp100, float("-inf"))
    tmp102 = 112 + (2*x0) + (224*x1)
    tmp103 = tmp101 > tmp99
    tmp104 = tl.where(tmp103, tmp102, tmp98)
    tmp105 = tl.where(tmp101 != tmp101, tmp101, tl.where(tmp101 > tmp99, tmp101, tmp99))
    tmp106 = tl.load(in_ptr0 + (113 + (2*x0) + (224*x3) + tl.zeros([XBLOCK], tl.int32)), tmp57 & xmask, other=0).to(tl.float32)
    tmp107 = tl.where(tmp57, tmp106, float("-inf"))
    tmp108 = 113 + (2*x0) + (224*x1)
    tmp109 = tmp107 > tmp105
    tmp110 = tl.where(tmp109, tmp108, tmp104)
    tmp111 = tl.where(tmp107 != tmp107, tmp107, tl.where(tmp107 > tmp105, tmp107, tmp105))
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp60, xmask)
    tl.store(out_ptr1 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp110, xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__10(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (12845056*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__11(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
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

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__12(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (12845056*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 401408.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__13(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000024912370735
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__14(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__17(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__18(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 14
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__19(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 401408.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__20(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 14
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000024912370735
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__21(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 14
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x4 + (401408*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 100352)
    x6 = xindex % 100352
    tmp0 = (-1) + x1
    tmp1 = 0
    tmp2 = tmp0 >= tmp1
    tmp3 = 56
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (300999 + x6 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (301000 + x6 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp17 & xmask, other=0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tmp19 + tmp12
    tmp21 = 1 + x0
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + (301001 + x6 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp25 & xmask, other=0).to(tl.float32)
    tmp27 = tl.where(tmp25, tmp26, 0.0)
    tmp28 = tmp27 + tmp20
    tmp29 = x1
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + (301055 + x6 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp33 & xmask, other=0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, 0.0)
    tmp36 = tmp35 + tmp28
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (301056 + x6 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp37 & xmask, other=0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, 0.0)
    tmp40 = tmp39 + tmp36
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (301057 + x6 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp41 & xmask, other=0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, 0.0)
    tmp44 = tmp43 + tmp40
    tmp45 = 1 + x1
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (301111 + x6 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp49 & xmask, other=0).to(tl.float32)
    tmp51 = tl.where(tmp49, tmp50, 0.0)
    tmp52 = tmp51 + tmp44
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (301112 + x6 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp53 & xmask, other=0).to(tl.float32)
    tmp55 = tl.where(tmp53, tmp54, 0.0)
    tmp56 = tmp55 + tmp52
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (301113 + x6 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp57 & xmask, other=0).to(tl.float32)
    tmp59 = tl.where(tmp57, tmp58, 0.0)
    tmp60 = tmp59 + tmp56
    tmp61 = 0.1111111111111111
    tmp62 = tmp60 * tmp61
    tl.store(out_ptr0 + (x6 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp62, xmask)
''')


triton__24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 524288],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__25(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp3 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 401408.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 401408.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0000024912370735
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x3), xmask).to(tl.float32)
    tmp19 = tl.load(in_ptr6 + (x1), xmask)
    tmp22 = tl.load(in_ptr7 + (x1), xmask)
    tmp27 = tl.load(in_ptr8 + (x1), xmask)
    tmp29 = tl.load(in_ptr9 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp19 / tmp3
    tmp21 = tmp18 - tmp20
    tmp23 = tmp22 / tmp3
    tmp24 = tmp23 + tmp8
    tmp25 = tl.libdevice.rsqrt(tmp24)
    tmp26 = tmp21 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp16 + tmp31
    tmp33 = tl.where(0 != 0, 0, tl.where(0 > tmp32, 0, tmp32))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


triton__28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*i1', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp18 = tl.load(in_ptr5 + (100352 + x4 + (401408*x2)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp17 <= tmp20
    tl.store(out_ptr0 + (x4 + (401408*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*i1', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp18 = tl.load(in_ptr5 + (200704 + x4 + (401408*x2)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp17 <= tmp20
    tl.store(out_ptr0 + (x4 + (401408*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (301056 + x0 + (401408*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (401408*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x3), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.where(0 != 0, 0, tl.where(0 > tmp18, 0, tmp18))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__35(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x1)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.where(tmp2, tmp4, 0)
        _tmp6 = tl.where(xmask & rmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp6, xmask)
''')


triton__36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__36(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__37(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x1)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp6 = 100352.0
        tmp7 = tmp5 / tmp6
        tmp8 = tmp4 - tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.where(tmp2, tmp9, 0)
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
        tmp12 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tl.where(tmp2, tmp13, 0)
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp11, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp15, xmask)
''')


triton__38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__38(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 100352.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.00000996502277
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__39(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 100352.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 100352.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x4 + (200704*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__41 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__41(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x3 = (xindex // 50176)
    x6 = (xindex // 28) % 1792
    x7 = xindex % 50176
    tmp0 = (-1) + (2*x1)
    tmp1 = 0
    tmp2 = tmp0 >= tmp1
    tmp3 = 56
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (602055 + (2*x0) + (112*x6) + (802816*x3) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = 2*x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (602056 + (2*x0) + (112*x6) + (802816*x3) + tl.zeros([XBLOCK], tl.int32)), tmp17 & xmask, other=0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tmp19 + tmp12
    tmp21 = 1 + (2*x0)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + (602057 + (2*x0) + (112*x6) + (802816*x3) + tl.zeros([XBLOCK], tl.int32)), tmp25 & xmask, other=0).to(tl.float32)
    tmp27 = tl.where(tmp25, tmp26, 0.0)
    tmp28 = tmp27 + tmp20
    tmp29 = 2*x1
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + (602111 + (2*x0) + (112*x6) + (802816*x3) + tl.zeros([XBLOCK], tl.int32)), tmp33 & xmask, other=0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, 0.0)
    tmp36 = tmp35 + tmp28
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (602112 + (2*x0) + (112*x6) + (802816*x3) + tl.zeros([XBLOCK], tl.int32)), tmp37 & xmask, other=0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, 0.0)
    tmp40 = tmp39 + tmp36
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (602113 + (2*x0) + (112*x6) + (802816*x3) + tl.zeros([XBLOCK], tl.int32)), tmp41 & xmask, other=0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, 0.0)
    tmp44 = tmp43 + tmp40
    tmp45 = 1 + (2*x1)
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (602167 + (2*x0) + (112*x6) + (802816*x3) + tl.zeros([XBLOCK], tl.int32)), tmp49 & xmask, other=0).to(tl.float32)
    tmp51 = tl.where(tmp49, tmp50, 0.0)
    tmp52 = tmp51 + tmp44
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (602168 + (2*x0) + (112*x6) + (802816*x3) + tl.zeros([XBLOCK], tl.int32)), tmp53 & xmask, other=0).to(tl.float32)
    tmp55 = tl.where(tmp53, tmp54, 0.0)
    tmp56 = tmp55 + tmp52
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (602169 + (2*x0) + (112*x6) + (802816*x3) + tl.zeros([XBLOCK], tl.int32)), tmp57 & xmask, other=0).to(tl.float32)
    tmp59 = tl.where(tmp57, tmp58, 0.0)
    tmp60 = tmp59 + tmp56
    tmp61 = 0.1111111111111111
    tmp62 = tmp60 * tmp61
    tl.store(out_ptr0 + (x7 + (200704*x3) + tl.zeros([XBLOCK], tl.int32)), tmp62, xmask)
''')


triton__42 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__43(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp3 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 100352.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 100352.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.00000996502277
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__44 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x3), xmask).to(tl.float32)
    tmp19 = tl.load(in_ptr6 + (x1), xmask)
    tmp22 = tl.load(in_ptr7 + (x1), xmask)
    tmp27 = tl.load(in_ptr8 + (x1), xmask)
    tmp29 = tl.load(in_ptr9 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 100352.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp19 / tmp3
    tmp21 = tmp18 - tmp20
    tmp23 = tmp22 / tmp3
    tmp24 = tmp23 + tmp8
    tmp25 = tl.libdevice.rsqrt(tmp24)
    tmp26 = tmp21 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp16 + tmp31
    tmp33 = tl.where(0 != 0, 0, tl.where(0 > tmp32, 0, tmp32))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


triton__45 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__45(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp3 = tl.load(in_ptr0 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 100352.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 100352.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.00000996502277
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__46 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 100352.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__47 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*i1', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp18 = tl.load(in_ptr5 + (50176 + x4 + (200704*x2)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 100352.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp17 <= tmp20
    tl.store(out_ptr0 + (x4 + (200704*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__48 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*i1', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp18 = tl.load(in_ptr5 + (100352 + x4 + (200704*x2)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 100352.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp17 <= tmp20
    tl.store(out_ptr0 + (x4 + (200704*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__49 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 50176
    x1 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (150528 + x0 + (200704*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (200704*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__50 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x3), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 100352.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.where(0 != 0, 0, tl.where(0 > tmp18, 0, tmp18))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__51 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__51(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__52 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 100352.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__53 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__53(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__54 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__54(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__55 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__55(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__56 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__56(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000398612827361
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__57 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__57(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__58 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x4 + (100352*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__59 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__59(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x3 = (xindex // 25088)
    x6 = (xindex // 14) % 1792
    x7 = xindex % 25088
    tmp0 = (-1) + (2*x1)
    tmp1 = 0
    tmp2 = tmp0 >= tmp1
    tmp3 = 28
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (301027 + (2*x0) + (56*x6) + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = 2*x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (301028 + (2*x0) + (56*x6) + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp17 & xmask, other=0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tmp19 + tmp12
    tmp21 = 1 + (2*x0)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + (301029 + (2*x0) + (56*x6) + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp25 & xmask, other=0).to(tl.float32)
    tmp27 = tl.where(tmp25, tmp26, 0.0)
    tmp28 = tmp27 + tmp20
    tmp29 = 2*x1
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + (301055 + (2*x0) + (56*x6) + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp33 & xmask, other=0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, 0.0)
    tmp36 = tmp35 + tmp28
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (301056 + (2*x0) + (56*x6) + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp37 & xmask, other=0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, 0.0)
    tmp40 = tmp39 + tmp36
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (301057 + (2*x0) + (56*x6) + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp41 & xmask, other=0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, 0.0)
    tmp44 = tmp43 + tmp40
    tmp45 = 1 + (2*x1)
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (301083 + (2*x0) + (56*x6) + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp49 & xmask, other=0).to(tl.float32)
    tmp51 = tl.where(tmp49, tmp50, 0.0)
    tmp52 = tmp51 + tmp44
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (301084 + (2*x0) + (56*x6) + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp53 & xmask, other=0).to(tl.float32)
    tmp55 = tl.where(tmp53, tmp54, 0.0)
    tmp56 = tmp55 + tmp52
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (301085 + (2*x0) + (56*x6) + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp57 & xmask, other=0).to(tl.float32)
    tmp59 = tl.where(tmp57, tmp58, 0.0)
    tmp60 = tmp59 + tmp56
    tmp61 = 0.1111111111111111
    tmp62 = tmp60 * tmp61
    tl.store(out_ptr0 + (x7 + (100352*x3) + tl.zeros([XBLOCK], tl.int32)), tmp62, xmask)
''')


triton__60 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__60(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__61 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__61(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 25088
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 25088.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 25088.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0000398612827361
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__62 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__62(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x3), xmask).to(tl.float32)
    tmp19 = tl.load(in_ptr6 + (x1), xmask)
    tmp22 = tl.load(in_ptr7 + (x1), xmask)
    tmp27 = tl.load(in_ptr8 + (x1), xmask)
    tmp29 = tl.load(in_ptr9 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp19 / tmp3
    tmp21 = tmp18 - tmp20
    tmp23 = tmp22 / tmp3
    tmp24 = tmp23 + tmp8
    tmp25 = tl.libdevice.rsqrt(tmp24)
    tmp26 = tmp21 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp16 + tmp31
    tmp33 = tl.where(0 != 0, 0, tl.where(0 > tmp32, 0, tmp32))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


triton__63 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__63(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25088
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 25088.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 25088.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0000398612827361
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__64 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__65 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*i1', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__65(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp18 = tl.load(in_ptr5 + (25088 + x4 + (100352*x2)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp17 <= tmp20
    tl.store(out_ptr0 + (x4 + (100352*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__66 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*i1', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__66(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp18 = tl.load(in_ptr5 + (50176 + x4 + (100352*x2)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp17 <= tmp20
    tl.store(out_ptr0 + (x4 + (100352*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__67 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__67(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (75264 + x0 + (100352*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (100352*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__68 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__68(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x3), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.where(0 != 0, 0, tl.where(0 > tmp18, 0, tmp18))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__69 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__69(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__70 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__70(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__71 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__71(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__72 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__72(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__73 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__73(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 256
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x4 + (50176*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__74 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__74(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 7) % 7
    x0 = xindex % 7
    x3 = (xindex // 12544)
    x6 = (xindex // 7) % 1792
    x7 = xindex % 12544
    tmp0 = (-1) + (2*x1)
    tmp1 = 0
    tmp2 = tmp0 >= tmp1
    tmp3 = 14
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (150513 + (2*x0) + (28*x6) + (200704*x3) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = 2*x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (150514 + (2*x0) + (28*x6) + (200704*x3) + tl.zeros([XBLOCK], tl.int32)), tmp17 & xmask, other=0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tmp19 + tmp12
    tmp21 = 1 + (2*x0)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + (150515 + (2*x0) + (28*x6) + (200704*x3) + tl.zeros([XBLOCK], tl.int32)), tmp25 & xmask, other=0).to(tl.float32)
    tmp27 = tl.where(tmp25, tmp26, 0.0)
    tmp28 = tmp27 + tmp20
    tmp29 = 2*x1
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + (150527 + (2*x0) + (28*x6) + (200704*x3) + tl.zeros([XBLOCK], tl.int32)), tmp33 & xmask, other=0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, 0.0)
    tmp36 = tmp35 + tmp28
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (150528 + (2*x0) + (28*x6) + (200704*x3) + tl.zeros([XBLOCK], tl.int32)), tmp37 & xmask, other=0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, 0.0)
    tmp40 = tmp39 + tmp36
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (150529 + (2*x0) + (28*x6) + (200704*x3) + tl.zeros([XBLOCK], tl.int32)), tmp41 & xmask, other=0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, 0.0)
    tmp44 = tmp43 + tmp40
    tmp45 = 1 + (2*x1)
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (150541 + (2*x0) + (28*x6) + (200704*x3) + tl.zeros([XBLOCK], tl.int32)), tmp49 & xmask, other=0).to(tl.float32)
    tmp51 = tl.where(tmp49, tmp50, 0.0)
    tmp52 = tmp51 + tmp44
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (150542 + (2*x0) + (28*x6) + (200704*x3) + tl.zeros([XBLOCK], tl.int32)), tmp53 & xmask, other=0).to(tl.float32)
    tmp55 = tl.where(tmp53, tmp54, 0.0)
    tmp56 = tmp55 + tmp52
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (150543 + (2*x0) + (28*x6) + (200704*x3) + tl.zeros([XBLOCK], tl.int32)), tmp57 & xmask, other=0).to(tl.float32)
    tmp59 = tl.where(tmp57, tmp58, 0.0)
    tmp60 = tmp59 + tmp56
    tmp61 = 0.1111111111111111
    tmp62 = tmp60 * tmp61
    tl.store(out_ptr0 + (x7 + (50176*x3) + tl.zeros([XBLOCK], tl.int32)), tmp62, xmask)
''')


triton__75 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__75(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__76 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__76(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__77 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__77(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x3), xmask).to(tl.float32)
    tmp19 = tl.load(in_ptr6 + (x1), xmask)
    tmp22 = tl.load(in_ptr7 + (x1), xmask)
    tmp27 = tl.load(in_ptr8 + (x1), xmask)
    tmp29 = tl.load(in_ptr9 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp19 / tmp3
    tmp21 = tmp18 - tmp20
    tmp23 = tmp22 / tmp3
    tmp24 = tmp23 + tmp8
    tmp25 = tl.libdevice.rsqrt(tmp24)
    tmp26 = tmp21 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp16 + tmp31
    tmp33 = tl.where(0 != 0, 0, tl.where(0 > tmp32, 0, tmp32))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


triton__78 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__78(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__79 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__79(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__80 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*i1', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__80(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 256
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp18 = tl.load(in_ptr5 + (12544 + x4 + (50176*x2)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp17 <= tmp20
    tl.store(out_ptr0 + (x4 + (50176*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__81 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*i1', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__81(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 256
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp18 = tl.load(in_ptr5 + (25088 + x4 + (50176*x2)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp17 <= tmp20
    tl.store(out_ptr0 + (x4 + (50176*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__82 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__82(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 12544
    x1 = (xindex // 12544)
    tmp0 = tl.load(in_ptr0 + (37632 + x0 + (50176*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (50176*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__83 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__83(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x3), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.where(0 != 0, 0, tl.where(0 > tmp18, 0, tmp18))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__84 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[262144, 64],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*i1', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__84(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 262144
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp17 = tl.load(in_ptr5 + (r2 + (49*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 6272.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = tl.where(0 != 0, 0, tl.where(0 > tmp18, 0, tmp18))
        tl.store(out_ptr0 + (r2 + (49*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask & xmask)
    _tmp24 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(out_ptr0 + (r2 + (49*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 0.0
        tmp22 = tmp20 <= tmp21
        tmp23 = tmp20.to(tl.float32)
        _tmp24 = tl.where(xmask & rmask, _tmp24 + tmp23, _tmp24)
        tl.store(out_ptr1 + (r2 + (49*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp22, rmask & xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp25 = 49.0
    tmp26 = tmp24 / tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr3 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp27, xmask)
''')


triton__85 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__85(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__86 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__86(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__87 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__87(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK])
    tmp1 = 1
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), tmp2, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda', dtype=torch.float16)
        stream0 = get_cuda_stream(0)
        triton__0.run(primals_1, buf0, 9408, grid=grid(9408), stream=stream0)
        del primals_1
        buf1 = empty_strided((128, 3, 224, 224), (150528, 50176, 224, 1), device='cuda', dtype=torch.float16)
        triton__1.run(primals_513, buf1, 19267584, grid=grid(19267584), stream=stream0)
        del primals_513
        buf2 = aten.convolution(buf1, buf0, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf2, (128, 64, 112, 112), (802816, 12544, 112, 1))
        buf3 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        triton__2.run(buf2, buf3, 448, 229376, grid=grid(448), stream=stream0)
        buf4 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        triton__3.run(buf3, buf4, 64, 7, grid=grid(64), stream=stream0)
        buf5 = buf3; del buf3  # reuse
        buf7 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        triton__4.run(buf2, buf4, buf5, buf7, 448, 229376, grid=grid(448), stream=stream0)
        buf6 = buf4; del buf4  # reuse
        buf9 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__5.run(buf5, primals_259, buf6, buf9, buf11, 64, 7, grid=grid(64), stream=stream0)
        del primals_259
        buf8 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf10 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1108 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf7, primals_258, buf8, buf10, buf1108, 64, 7, grid=grid(64), stream=stream0)
        del primals_258
        buf12 = empty_strided((128, 64, 112, 112), (802816, 12544, 112, 1), device='cuda', dtype=torch.float16)
        triton__7.run(buf2, buf8, buf6, primals_2, primals_3, buf12, 102760448, grid=grid(102760448), stream=stream0)
        del primals_3
        buf13 = empty_strided((128, 64, 56, 56), (200704, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf14 = empty_strided((128, 64, 56, 56), (200704, 3136, 56, 1), device='cuda', dtype=torch.int64)
        triton__8.run(buf12, buf13, buf14, 25690112, grid=grid(25690112), stream=stream0)
        buf15 = empty_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__9.run(primals_4, buf15, 8192, grid=grid(8192), stream=stream0)
        del primals_4
        buf16 = aten.convolution(buf13, buf15, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf16, (128, 128, 56, 56), (401408, 3136, 56, 1))
        buf17 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        triton__10.run(buf16, buf17, 512, 100352, grid=grid(512), stream=stream0)
        buf18 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        triton__11.run(buf17, buf18, 128, 4, grid=grid(128), stream=stream0)
        buf19 = buf17; del buf17  # reuse
        buf21 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        triton__12.run(buf16, buf18, buf19, buf21, 512, 100352, grid=grid(512), stream=stream0)
        buf20 = buf18; del buf18  # reuse
        buf23 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf25 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__13.run(buf19, primals_262, buf20, buf23, buf25, 128, 4, grid=grid(128), stream=stream0)
        del primals_262
        buf22 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf24 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1107 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf21, primals_261, buf22, buf24, buf1107, 128, 4, grid=grid(128), stream=stream0)
        del primals_261
        buf26 = empty_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf1106 = empty_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__15.run(buf16, buf22, buf20, primals_5, primals_6, buf26, buf1106, 51380224, grid=grid(51380224), stream=stream0)
        del primals_6
        buf27 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__16.run(primals_7, buf27, 1152, grid=grid(1152), stream=stream0)
        del primals_7
        buf28 = aten.convolution(as_strided(buf26, (128, 32, 56, 56), (401408, 3136, 56, 1)), buf27, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf28, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf29 = as_strided(buf7, (1, 32, 1, 1, 14), (448, 1, 448, 448, 32)); del buf7  # reuse
        triton__17.run(buf28, buf29, 448, 28672, grid=grid(448), stream=stream0)
        buf30 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        triton__18.run(buf29, buf30, 32, 14, grid=grid(32), stream=stream0)
        buf31 = buf29; del buf29  # reuse
        buf33 = as_strided(buf5, (1, 32, 1, 1, 14), (448, 1, 448, 448, 32)); del buf5  # reuse
        triton__19.run(buf28, buf30, buf31, buf33, 448, 28672, grid=grid(448), stream=stream0)
        buf32 = buf30; del buf30  # reuse
        buf35 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__20.run(buf31, primals_265, buf32, buf35, buf37, 32, 14, grid=grid(32), stream=stream0)
        del primals_265
        buf34 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf36 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf1105 = empty_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf33, primals_264, buf34, buf36, buf1105, 32, 14, grid=grid(32), stream=stream0)
        del primals_264
        buf64 = empty_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf38 = as_strided(buf64, (128, 32, 56, 56), (401408, 3136, 56, 1))  # alias
        buf1104 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__22.run(buf28, buf34, buf32, primals_8, primals_9, buf38, buf1104, 12845056, grid=grid(12845056), stream=stream0)
        del primals_9
        buf39 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__16.run(primals_10, buf39, 1152, grid=grid(1152), stream=stream0)
        del primals_10
        buf40 = aten.convolution(as_strided(buf26, (128, 32, 56, 56), (401408, 3136, 56, 1), 100352), buf39, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf40, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf41 = buf33; del buf33  # reuse
        triton__17.run(buf40, buf41, 448, 28672, grid=grid(448), stream=stream0)
        buf42 = buf34; del buf34  # reuse
        triton__18.run(buf41, buf42, 32, 14, grid=grid(32), stream=stream0)
        buf43 = buf41; del buf41  # reuse
        buf45 = buf31; del buf31  # reuse
        triton__19.run(buf40, buf42, buf43, buf45, 448, 28672, grid=grid(448), stream=stream0)
        buf44 = buf42; del buf42  # reuse
        buf47 = as_strided(buf32, (32, ), (1, )); del buf32  # reuse
        buf49 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__20.run(buf43, primals_268, buf44, buf47, buf49, 32, 14, grid=grid(32), stream=stream0)
        del primals_268
        buf46 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf48 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf1103 = empty_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf45, primals_267, buf46, buf48, buf1103, 32, 14, grid=grid(32), stream=stream0)
        del primals_267
        buf50 = as_strided(buf64, (128, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        buf1102 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__22.run(buf40, buf46, buf44, primals_11, primals_12, buf50, buf1102, 12845056, grid=grid(12845056), stream=stream0)
        del primals_12
        buf51 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__16.run(primals_13, buf51, 1152, grid=grid(1152), stream=stream0)
        del primals_13
        buf52 = aten.convolution(as_strided(buf26, (128, 32, 56, 56), (401408, 3136, 56, 1), 200704), buf51, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf52, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf53 = buf45; del buf45  # reuse
        triton__17.run(buf52, buf53, 448, 28672, grid=grid(448), stream=stream0)
        buf54 = buf46; del buf46  # reuse
        triton__18.run(buf53, buf54, 32, 14, grid=grid(32), stream=stream0)
        buf55 = buf53; del buf53  # reuse
        buf57 = buf43; del buf43  # reuse
        triton__19.run(buf52, buf54, buf55, buf57, 448, 28672, grid=grid(448), stream=stream0)
        buf56 = buf54; del buf54  # reuse
        buf59 = as_strided(buf44, (32, ), (1, )); del buf44  # reuse
        buf61 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__20.run(buf55, primals_271, buf56, buf59, buf61, 32, 14, grid=grid(32), stream=stream0)
        del primals_271
        buf58 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf60 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf1101 = empty_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf57, primals_270, buf58, buf60, buf1101, 32, 14, grid=grid(32), stream=stream0)
        del primals_270
        buf62 = as_strided(buf64, (128, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        buf1100 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__22.run(buf52, buf58, buf56, primals_14, primals_15, buf62, buf1100, 12845056, grid=grid(12845056), stream=stream0)
        del primals_15
        buf63 = as_strided(buf64, (128, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        triton__23.run(buf26, buf63, 12845056, grid=grid(12845056), stream=stream0)
        buf65 = empty_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__24.run(primals_16, buf65, 32768, grid=grid(32768), stream=stream0)
        del primals_16
        buf66 = aten.convolution(buf64, buf65, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf66, (128, 256, 56, 56), (802816, 3136, 56, 1))
        buf68 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf69 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf71 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf1099 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__25.run(buf66, primals_274, primals_273, buf68, buf69, buf70, buf72, buf71, buf1099, 256, 401408, grid=grid(256), stream=stream0)
        del primals_273
        del primals_274
        buf73 = empty_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__26.run(primals_19, buf73, 16384, grid=grid(16384), stream=stream0)
        del primals_19
        buf74 = aten.convolution(buf13, buf73, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf74, (128, 256, 56, 56), (802816, 3136, 56, 1))
        buf76 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf77 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf78 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf80 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf79 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf1098 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__25.run(buf74, primals_277, primals_276, buf76, buf77, buf78, buf80, buf79, buf1098, 256, 401408, grid=grid(256), stream=stream0)
        del primals_276
        del primals_277
        buf81 = empty_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf82 = buf81; del buf81  # reuse
        triton__27.run(buf82, buf66, buf69, buf68, primals_17, primals_18, buf74, buf77, buf76, primals_20, primals_21, 102760448, grid=grid(102760448), stream=stream0)
        del primals_18
        del primals_21
        buf83 = empty_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__24.run(primals_22, buf83, 32768, grid=grid(32768), stream=stream0)
        del primals_22
        buf84 = aten.convolution(buf82, buf83, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf84, (128, 128, 56, 56), (401408, 3136, 56, 1))
        buf85 = buf21; del buf21  # reuse
        triton__10.run(buf84, buf85, 512, 100352, grid=grid(512), stream=stream0)
        buf86 = buf22; del buf22  # reuse
        triton__11.run(buf85, buf86, 128, 4, grid=grid(128), stream=stream0)
        buf87 = buf85; del buf85  # reuse
        buf89 = buf19; del buf19  # reuse
        triton__12.run(buf84, buf86, buf87, buf89, 512, 100352, grid=grid(512), stream=stream0)
        buf88 = buf86; del buf86  # reuse
        buf91 = as_strided(buf20, (128, ), (1, )); del buf20  # reuse
        buf93 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__13.run(buf87, primals_280, buf88, buf91, buf93, 128, 4, grid=grid(128), stream=stream0)
        del primals_280
        buf90 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf92 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1097 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf89, primals_279, buf90, buf92, buf1097, 128, 4, grid=grid(128), stream=stream0)
        del primals_279
        buf94 = empty_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf1096 = empty_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__15.run(buf84, buf90, buf88, primals_23, primals_24, buf94, buf1096, 51380224, grid=grid(51380224), stream=stream0)
        del primals_24
        buf95 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__16.run(primals_25, buf95, 1152, grid=grid(1152), stream=stream0)
        del primals_25
        buf96 = aten.convolution(as_strided(buf94, (128, 32, 56, 56), (401408, 3136, 56, 1)), buf95, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf96, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf97 = buf57; del buf57  # reuse
        triton__17.run(buf96, buf97, 448, 28672, grid=grid(448), stream=stream0)
        buf98 = buf58; del buf58  # reuse
        triton__18.run(buf97, buf98, 32, 14, grid=grid(32), stream=stream0)
        buf99 = buf97; del buf97  # reuse
        buf101 = buf55; del buf55  # reuse
        triton__19.run(buf96, buf98, buf99, buf101, 448, 28672, grid=grid(448), stream=stream0)
        buf100 = buf98; del buf98  # reuse
        buf103 = as_strided(buf56, (32, ), (1, )); del buf56  # reuse
        buf105 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__20.run(buf99, primals_283, buf100, buf103, buf105, 32, 14, grid=grid(32), stream=stream0)
        del primals_283
        buf102 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf104 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf1095 = empty_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf101, primals_282, buf102, buf104, buf1095, 32, 14, grid=grid(32), stream=stream0)
        del primals_282
        buf134 = empty_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf106 = as_strided(buf134, (128, 32, 56, 56), (401408, 3136, 56, 1))  # alias
        buf107 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf1094 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__28.run(buf96, buf102, buf100, primals_26, primals_27, buf94, buf106, buf107, buf1094, 12845056, grid=grid(12845056), stream=stream0)
        del primals_27
        buf108 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__16.run(primals_28, buf108, 1152, grid=grid(1152), stream=stream0)
        del primals_28
        buf109 = aten.convolution(buf107, buf108, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf109, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf110 = buf101; del buf101  # reuse
        triton__17.run(buf109, buf110, 448, 28672, grid=grid(448), stream=stream0)
        buf111 = buf102; del buf102  # reuse
        triton__18.run(buf110, buf111, 32, 14, grid=grid(32), stream=stream0)
        buf112 = buf110; del buf110  # reuse
        buf114 = buf99; del buf99  # reuse
        triton__19.run(buf109, buf111, buf112, buf114, 448, 28672, grid=grid(448), stream=stream0)
        buf113 = buf111; del buf111  # reuse
        buf116 = as_strided(buf100, (32, ), (1, )); del buf100  # reuse
        buf118 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__20.run(buf112, primals_286, buf113, buf116, buf118, 32, 14, grid=grid(32), stream=stream0)
        del primals_286
        buf115 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf117 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf1093 = empty_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf114, primals_285, buf115, buf117, buf1093, 32, 14, grid=grid(32), stream=stream0)
        del primals_285
        buf119 = as_strided(buf134, (128, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        buf120 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf1092 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__29.run(buf109, buf115, buf113, primals_29, primals_30, buf94, buf119, buf120, buf1092, 12845056, grid=grid(12845056), stream=stream0)
        del primals_30
        buf121 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__16.run(primals_31, buf121, 1152, grid=grid(1152), stream=stream0)
        del primals_31
        buf122 = aten.convolution(buf120, buf121, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf122, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf123 = buf114; del buf114  # reuse
        triton__17.run(buf122, buf123, 448, 28672, grid=grid(448), stream=stream0)
        buf124 = buf115; del buf115  # reuse
        triton__18.run(buf123, buf124, 32, 14, grid=grid(32), stream=stream0)
        buf125 = buf123; del buf123  # reuse
        buf127 = buf112; del buf112  # reuse
        triton__19.run(buf122, buf124, buf125, buf127, 448, 28672, grid=grid(448), stream=stream0)
        buf126 = buf124; del buf124  # reuse
        buf129 = as_strided(buf113, (32, ), (1, )); del buf113  # reuse
        buf131 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__20.run(buf125, primals_289, buf126, buf129, buf131, 32, 14, grid=grid(32), stream=stream0)
        del primals_289
        buf128 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf130 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf1091 = empty_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf127, primals_288, buf128, buf130, buf1091, 32, 14, grid=grid(32), stream=stream0)
        del primals_288
        buf132 = as_strided(buf134, (128, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        buf1090 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__22.run(buf122, buf128, buf126, primals_32, primals_33, buf132, buf1090, 12845056, grid=grid(12845056), stream=stream0)
        del primals_33
        buf133 = as_strided(buf134, (128, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        triton__30.run(buf94, buf133, 12845056, grid=grid(12845056), stream=stream0)
        buf135 = empty_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__24.run(primals_34, buf135, 32768, grid=grid(32768), stream=stream0)
        del primals_34
        buf136 = aten.convolution(buf134, buf135, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf136, (128, 256, 56, 56), (802816, 3136, 56, 1))
        buf138 = buf77; del buf77  # reuse
        buf139 = buf76; del buf76  # reuse
        buf140 = as_strided(buf69, (256, ), (1, )); del buf69  # reuse
        buf142 = as_strided(buf68, (256, ), (1, )); del buf68  # reuse
        buf141 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf1089 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__25.run(buf136, primals_292, primals_291, buf138, buf139, buf140, buf142, buf141, buf1089, 256, 401408, grid=grid(256), stream=stream0)
        del primals_291
        del primals_292
        buf143 = empty_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda', dtype=torch.float16)
        triton__31.run(buf136, buf139, buf138, primals_35, primals_36, buf82, buf143, 102760448, grid=grid(102760448), stream=stream0)
        del primals_36
        buf144 = empty_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__24.run(primals_37, buf144, 32768, grid=grid(32768), stream=stream0)
        del primals_37
        buf145 = aten.convolution(buf143, buf144, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf145, (128, 128, 56, 56), (401408, 3136, 56, 1))
        buf146 = buf89; del buf89  # reuse
        triton__10.run(buf145, buf146, 512, 100352, grid=grid(512), stream=stream0)
        buf147 = buf90; del buf90  # reuse
        triton__11.run(buf146, buf147, 128, 4, grid=grid(128), stream=stream0)
        buf148 = buf146; del buf146  # reuse
        buf150 = buf87; del buf87  # reuse
        triton__12.run(buf145, buf147, buf148, buf150, 512, 100352, grid=grid(512), stream=stream0)
        buf149 = buf147; del buf147  # reuse
        buf152 = as_strided(buf88, (128, ), (1, )); del buf88  # reuse
        buf154 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__13.run(buf148, primals_295, buf149, buf152, buf154, 128, 4, grid=grid(128), stream=stream0)
        del primals_295
        buf151 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf153 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1088 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__14.run(buf150, primals_294, buf151, buf153, buf1088, 128, 4, grid=grid(128), stream=stream0)
        del primals_294
        buf155 = empty_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf1087 = empty_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__15.run(buf145, buf151, buf149, primals_38, primals_39, buf155, buf1087, 51380224, grid=grid(51380224), stream=stream0)
        del primals_39
        buf156 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__16.run(primals_40, buf156, 1152, grid=grid(1152), stream=stream0)
        del primals_40
        buf157 = aten.convolution(as_strided(buf155, (128, 32, 56, 56), (401408, 3136, 56, 1)), buf156, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf157, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf158 = buf127; del buf127  # reuse
        triton__17.run(buf157, buf158, 448, 28672, grid=grid(448), stream=stream0)
        buf159 = buf128; del buf128  # reuse
        triton__18.run(buf158, buf159, 32, 14, grid=grid(32), stream=stream0)
        buf160 = buf158; del buf158  # reuse
        buf162 = buf125; del buf125  # reuse
        triton__19.run(buf157, buf159, buf160, buf162, 448, 28672, grid=grid(448), stream=stream0)
        buf161 = buf159; del buf159  # reuse
        buf164 = as_strided(buf126, (32, ), (1, )); del buf126  # reuse
        buf166 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__20.run(buf160, primals_298, buf161, buf164, buf166, 32, 14, grid=grid(32), stream=stream0)
        del primals_298
        buf163 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf165 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf1086 = empty_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf162, primals_297, buf163, buf165, buf1086, 32, 14, grid=grid(32), stream=stream0)
        del primals_297
        buf195 = empty_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf167 = as_strided(buf195, (128, 32, 56, 56), (401408, 3136, 56, 1))  # alias
        buf168 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf1085 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__28.run(buf157, buf163, buf161, primals_41, primals_42, buf155, buf167, buf168, buf1085, 12845056, grid=grid(12845056), stream=stream0)
        del primals_42
        buf169 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__16.run(primals_43, buf169, 1152, grid=grid(1152), stream=stream0)
        del primals_43
        buf170 = aten.convolution(buf168, buf169, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf170, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf171 = buf162; del buf162  # reuse
        triton__17.run(buf170, buf171, 448, 28672, grid=grid(448), stream=stream0)
        buf172 = buf163; del buf163  # reuse
        triton__18.run(buf171, buf172, 32, 14, grid=grid(32), stream=stream0)
        buf173 = buf171; del buf171  # reuse
        buf175 = buf160; del buf160  # reuse
        triton__19.run(buf170, buf172, buf173, buf175, 448, 28672, grid=grid(448), stream=stream0)
        buf174 = buf172; del buf172  # reuse
        buf177 = as_strided(buf161, (32, ), (1, )); del buf161  # reuse
        buf179 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__20.run(buf173, primals_301, buf174, buf177, buf179, 32, 14, grid=grid(32), stream=stream0)
        del primals_301
        buf176 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf178 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf1084 = empty_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf175, primals_300, buf176, buf178, buf1084, 32, 14, grid=grid(32), stream=stream0)
        del primals_300
        buf180 = as_strided(buf195, (128, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        buf181 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf1083 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__29.run(buf170, buf176, buf174, primals_44, primals_45, buf155, buf180, buf181, buf1083, 12845056, grid=grid(12845056), stream=stream0)
        del primals_45
        buf182 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__16.run(primals_46, buf182, 1152, grid=grid(1152), stream=stream0)
        del primals_46
        buf183 = aten.convolution(buf181, buf182, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf183, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf184 = buf175; del buf175  # reuse
        triton__17.run(buf183, buf184, 448, 28672, grid=grid(448), stream=stream0)
        buf185 = buf176; del buf176  # reuse
        triton__18.run(buf184, buf185, 32, 14, grid=grid(32), stream=stream0)
        buf186 = buf184; del buf184  # reuse
        buf188 = buf173; del buf173  # reuse
        triton__19.run(buf183, buf185, buf186, buf188, 448, 28672, grid=grid(448), stream=stream0)
        buf187 = buf185; del buf185  # reuse
        buf190 = as_strided(buf174, (32, ), (1, )); del buf174  # reuse
        buf192 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__20.run(buf186, primals_304, buf187, buf190, buf192, 32, 14, grid=grid(32), stream=stream0)
        del buf186
        del primals_304
        buf189 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf191 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf1082 = empty_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__21.run(buf188, primals_303, buf189, buf191, buf1082, 32, 14, grid=grid(32), stream=stream0)
        del buf188
        del primals_303
        buf193 = as_strided(buf195, (128, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        buf1081 = empty_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__22.run(buf183, buf189, buf187, primals_47, primals_48, buf193, buf1081, 12845056, grid=grid(12845056), stream=stream0)
        del buf187
        del buf189
        del primals_48
        buf194 = as_strided(buf195, (128, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        triton__30.run(buf155, buf194, 12845056, grid=grid(12845056), stream=stream0)
        buf196 = empty_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__24.run(primals_49, buf196, 32768, grid=grid(32768), stream=stream0)
        del primals_49
        buf197 = aten.convolution(buf195, buf196, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf197, (128, 256, 56, 56), (802816, 3136, 56, 1))
        buf199 = buf139; del buf139  # reuse
        buf200 = buf138; del buf138  # reuse
        buf201 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf203 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf202 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf1080 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__25.run(buf197, primals_307, primals_306, buf199, buf200, buf201, buf203, buf202, buf1080, 256, 401408, grid=grid(256), stream=stream0)
        del primals_306
        del primals_307
        buf204 = empty_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda', dtype=torch.float16)
        triton__31.run(buf197, buf200, buf199, primals_50, primals_51, buf143, buf204, 102760448, grid=grid(102760448), stream=stream0)
        del primals_51
        buf205 = empty_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__32.run(primals_52, buf205, 65536, grid=grid(65536), stream=stream0)
        del primals_52
        buf206 = aten.convolution(buf204, buf205, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf206, (128, 256, 56, 56), (802816, 3136, 56, 1))
        buf208 = buf200; del buf200  # reuse
        buf209 = buf199; del buf199  # reuse
        buf210 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf212 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf211 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf1079 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__25.run(buf206, primals_310, primals_309, buf208, buf209, buf210, buf212, buf211, buf1079, 256, 401408, grid=grid(256), stream=stream0)
        del primals_309
        del primals_310
        buf213 = empty_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf1078 = empty_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__33.run(buf206, buf209, buf208, primals_53, primals_54, buf213, buf1078, 102760448, grid=grid(102760448), stream=stream0)
        del primals_54
        buf214 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_55, buf214, 4608, grid=grid(4608), stream=stream0)
        del primals_55
        buf215 = aten.convolution(as_strided(buf213, (128, 64, 56, 56), (802816, 3136, 56, 1)), buf214, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf215, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf216 = empty_strided((1, 64, 1, 1, 13), (832, 1, 832, 832, 64), device='cuda', dtype=torch.float32)
        triton__35.run(buf215, buf216, 832, 7720, grid=grid(832), stream=stream0)
        buf217 = buf8; del buf8  # reuse
        triton__36.run(buf216, buf217, 64, 13, grid=grid(64), stream=stream0)
        buf218 = buf216; del buf216  # reuse
        buf220 = empty_strided((1, 64, 1, 1, 13), (832, 1, 832, 832, 64), device='cuda', dtype=torch.float32)
        triton__37.run(buf215, buf217, buf218, buf220, 832, 7720, grid=grid(832), stream=stream0)
        buf219 = buf217; del buf217  # reuse
        buf222 = as_strided(buf6, (64, ), (1, )); del buf6  # reuse
        buf224 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf218, primals_313, buf219, buf222, buf224, 64, 13, grid=grid(64), stream=stream0)
        del primals_313
        buf221 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf223 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1077 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf220, primals_312, buf221, buf223, buf1077, 64, 13, grid=grid(64), stream=stream0)
        del primals_312
        buf251 = empty_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf225 = as_strided(buf251, (128, 64, 28, 28), (200704, 784, 28, 1))  # alias
        buf1076 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__40.run(buf215, buf221, buf219, primals_56, primals_57, buf225, buf1076, 6422528, grid=grid(6422528), stream=stream0)
        del primals_57
        buf226 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_58, buf226, 4608, grid=grid(4608), stream=stream0)
        del primals_58
        buf227 = aten.convolution(as_strided(buf213, (128, 64, 56, 56), (802816, 3136, 56, 1), 200704), buf226, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf227, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf228 = buf220; del buf220  # reuse
        triton__35.run(buf227, buf228, 832, 7720, grid=grid(832), stream=stream0)
        buf229 = buf221; del buf221  # reuse
        triton__36.run(buf228, buf229, 64, 13, grid=grid(64), stream=stream0)
        buf230 = buf228; del buf228  # reuse
        buf232 = buf218; del buf218  # reuse
        triton__37.run(buf227, buf229, buf230, buf232, 832, 7720, grid=grid(832), stream=stream0)
        buf231 = buf229; del buf229  # reuse
        buf234 = as_strided(buf219, (64, ), (1, )); del buf219  # reuse
        buf236 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf230, primals_316, buf231, buf234, buf236, 64, 13, grid=grid(64), stream=stream0)
        del primals_316
        buf233 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf235 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1075 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf232, primals_315, buf233, buf235, buf1075, 64, 13, grid=grid(64), stream=stream0)
        del primals_315
        buf237 = as_strided(buf251, (128, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        buf1074 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__40.run(buf227, buf233, buf231, primals_59, primals_60, buf237, buf1074, 6422528, grid=grid(6422528), stream=stream0)
        del primals_60
        buf238 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_61, buf238, 4608, grid=grid(4608), stream=stream0)
        del primals_61
        buf239 = aten.convolution(as_strided(buf213, (128, 64, 56, 56), (802816, 3136, 56, 1), 401408), buf238, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf239, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf240 = buf232; del buf232  # reuse
        triton__35.run(buf239, buf240, 832, 7720, grid=grid(832), stream=stream0)
        buf241 = buf233; del buf233  # reuse
        triton__36.run(buf240, buf241, 64, 13, grid=grid(64), stream=stream0)
        buf242 = buf240; del buf240  # reuse
        buf244 = buf230; del buf230  # reuse
        triton__37.run(buf239, buf241, buf242, buf244, 832, 7720, grid=grid(832), stream=stream0)
        buf243 = buf241; del buf241  # reuse
        buf246 = as_strided(buf231, (64, ), (1, )); del buf231  # reuse
        buf248 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf242, primals_319, buf243, buf246, buf248, 64, 13, grid=grid(64), stream=stream0)
        del primals_319
        buf245 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf247 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1073 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf244, primals_318, buf245, buf247, buf1073, 64, 13, grid=grid(64), stream=stream0)
        del primals_318
        buf249 = as_strided(buf251, (128, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        buf1072 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__40.run(buf239, buf245, buf243, primals_62, primals_63, buf249, buf1072, 6422528, grid=grid(6422528), stream=stream0)
        del primals_63
        buf250 = as_strided(buf251, (128, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        triton__41.run(buf213, buf250, 6422528, grid=grid(6422528), stream=stream0)
        buf252 = empty_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__42.run(primals_64, buf252, 131072, grid=grid(131072), stream=stream0)
        del primals_64
        buf253 = aten.convolution(buf251, buf252, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf253, (128, 512, 28, 28), (401408, 784, 28, 1))
        buf255 = as_strided(buf150, (1, 512, 1, 1), (512, 1, 512, 512)); del buf150  # reuse
        buf256 = as_strided(buf148, (1, 512, 1, 1), (512, 1, 512, 512)); del buf148  # reuse
        buf257 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf259 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf258 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf1071 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__43.run(buf253, primals_322, primals_321, buf255, buf256, buf257, buf259, buf258, buf1071, 512, 100352, grid=grid(512), stream=stream0)
        del primals_321
        del primals_322
        buf260 = empty_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__42.run(primals_67, buf260, 131072, grid=grid(131072), stream=stream0)
        del primals_67
        buf261 = aten.convolution(buf204, buf260, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf261, (128, 512, 28, 28), (401408, 784, 28, 1))
        buf263 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf264 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf265 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf267 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf266 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf1070 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__43.run(buf261, primals_325, primals_324, buf263, buf264, buf265, buf267, buf266, buf1070, 512, 100352, grid=grid(512), stream=stream0)
        del primals_324
        del primals_325
        buf268 = empty_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf269 = buf268; del buf268  # reuse
        triton__44.run(buf269, buf253, buf256, buf255, primals_65, primals_66, buf261, buf264, buf263, primals_68, primals_69, 51380224, grid=grid(51380224), stream=stream0)
        del primals_66
        del primals_69
        buf270 = empty_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__42.run(primals_70, buf270, 131072, grid=grid(131072), stream=stream0)
        del primals_70
        buf271 = aten.convolution(buf269, buf270, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf271, (128, 256, 28, 28), (200704, 784, 28, 1))
        buf273 = buf209; del buf209  # reuse
        buf274 = buf208; del buf208  # reuse
        buf275 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf277 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf276 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf1069 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__45.run(buf271, primals_328, primals_327, buf273, buf274, buf275, buf277, buf276, buf1069, 256, 100352, grid=grid(256), stream=stream0)
        del primals_327
        del primals_328
        buf278 = empty_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf1068 = empty_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__46.run(buf271, buf274, buf273, primals_71, primals_72, buf278, buf1068, 25690112, grid=grid(25690112), stream=stream0)
        del primals_72
        buf279 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_73, buf279, 4608, grid=grid(4608), stream=stream0)
        del primals_73
        buf280 = aten.convolution(as_strided(buf278, (128, 64, 28, 28), (200704, 784, 28, 1)), buf279, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf280, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf281 = buf244; del buf244  # reuse
        triton__35.run(buf280, buf281, 832, 7720, grid=grid(832), stream=stream0)
        buf282 = buf245; del buf245  # reuse
        triton__36.run(buf281, buf282, 64, 13, grid=grid(64), stream=stream0)
        buf283 = buf281; del buf281  # reuse
        buf285 = buf242; del buf242  # reuse
        triton__37.run(buf280, buf282, buf283, buf285, 832, 7720, grid=grid(832), stream=stream0)
        buf284 = buf282; del buf282  # reuse
        buf287 = as_strided(buf243, (64, ), (1, )); del buf243  # reuse
        buf289 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf283, primals_331, buf284, buf287, buf289, 64, 13, grid=grid(64), stream=stream0)
        del primals_331
        buf286 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf288 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1067 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf285, primals_330, buf286, buf288, buf1067, 64, 13, grid=grid(64), stream=stream0)
        del primals_330
        buf318 = empty_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf290 = as_strided(buf318, (128, 64, 28, 28), (200704, 784, 28, 1))  # alias
        buf291 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf1066 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__47.run(buf280, buf286, buf284, primals_74, primals_75, buf278, buf290, buf291, buf1066, 6422528, grid=grid(6422528), stream=stream0)
        del primals_75
        buf292 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_76, buf292, 4608, grid=grid(4608), stream=stream0)
        del primals_76
        buf293 = aten.convolution(buf291, buf292, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf293, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf294 = buf285; del buf285  # reuse
        triton__35.run(buf293, buf294, 832, 7720, grid=grid(832), stream=stream0)
        buf295 = buf286; del buf286  # reuse
        triton__36.run(buf294, buf295, 64, 13, grid=grid(64), stream=stream0)
        buf296 = buf294; del buf294  # reuse
        buf298 = buf283; del buf283  # reuse
        triton__37.run(buf293, buf295, buf296, buf298, 832, 7720, grid=grid(832), stream=stream0)
        buf297 = buf295; del buf295  # reuse
        buf300 = as_strided(buf284, (64, ), (1, )); del buf284  # reuse
        buf302 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf296, primals_334, buf297, buf300, buf302, 64, 13, grid=grid(64), stream=stream0)
        del primals_334
        buf299 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf301 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1065 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf298, primals_333, buf299, buf301, buf1065, 64, 13, grid=grid(64), stream=stream0)
        del primals_333
        buf303 = as_strided(buf318, (128, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        buf304 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf1064 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__48.run(buf293, buf299, buf297, primals_77, primals_78, buf278, buf303, buf304, buf1064, 6422528, grid=grid(6422528), stream=stream0)
        del primals_78
        buf305 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_79, buf305, 4608, grid=grid(4608), stream=stream0)
        del primals_79
        buf306 = aten.convolution(buf304, buf305, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf306, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf307 = buf298; del buf298  # reuse
        triton__35.run(buf306, buf307, 832, 7720, grid=grid(832), stream=stream0)
        buf308 = buf299; del buf299  # reuse
        triton__36.run(buf307, buf308, 64, 13, grid=grid(64), stream=stream0)
        buf309 = buf307; del buf307  # reuse
        buf311 = buf296; del buf296  # reuse
        triton__37.run(buf306, buf308, buf309, buf311, 832, 7720, grid=grid(832), stream=stream0)
        buf310 = buf308; del buf308  # reuse
        buf313 = as_strided(buf297, (64, ), (1, )); del buf297  # reuse
        buf315 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf309, primals_337, buf310, buf313, buf315, 64, 13, grid=grid(64), stream=stream0)
        del primals_337
        buf312 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf314 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1063 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf311, primals_336, buf312, buf314, buf1063, 64, 13, grid=grid(64), stream=stream0)
        del primals_336
        buf316 = as_strided(buf318, (128, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        buf1062 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__40.run(buf306, buf312, buf310, primals_80, primals_81, buf316, buf1062, 6422528, grid=grid(6422528), stream=stream0)
        del primals_81
        buf317 = as_strided(buf318, (128, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        triton__49.run(buf278, buf317, 6422528, grid=grid(6422528), stream=stream0)
        buf319 = empty_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__42.run(primals_82, buf319, 131072, grid=grid(131072), stream=stream0)
        del primals_82
        buf320 = aten.convolution(buf318, buf319, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf320, (128, 512, 28, 28), (401408, 784, 28, 1))
        buf322 = buf264; del buf264  # reuse
        buf323 = buf263; del buf263  # reuse
        buf324 = as_strided(buf256, (512, ), (1, )); del buf256  # reuse
        buf326 = as_strided(buf255, (512, ), (1, )); del buf255  # reuse
        buf325 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf1061 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__43.run(buf320, primals_340, primals_339, buf322, buf323, buf324, buf326, buf325, buf1061, 512, 100352, grid=grid(512), stream=stream0)
        del primals_339
        del primals_340
        buf327 = empty_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda', dtype=torch.float16)
        triton__50.run(buf320, buf323, buf322, primals_83, primals_84, buf269, buf327, 51380224, grid=grid(51380224), stream=stream0)
        del primals_84
        buf328 = empty_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__42.run(primals_85, buf328, 131072, grid=grid(131072), stream=stream0)
        del primals_85
        buf329 = aten.convolution(buf327, buf328, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf329, (128, 256, 28, 28), (200704, 784, 28, 1))
        buf331 = buf274; del buf274  # reuse
        buf332 = buf273; del buf273  # reuse
        buf333 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf335 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf334 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf1060 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__45.run(buf329, primals_343, primals_342, buf331, buf332, buf333, buf335, buf334, buf1060, 256, 100352, grid=grid(256), stream=stream0)
        del primals_342
        del primals_343
        buf336 = empty_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf1059 = empty_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__46.run(buf329, buf332, buf331, primals_86, primals_87, buf336, buf1059, 25690112, grid=grid(25690112), stream=stream0)
        del primals_87
        buf337 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_88, buf337, 4608, grid=grid(4608), stream=stream0)
        del primals_88
        buf338 = aten.convolution(as_strided(buf336, (128, 64, 28, 28), (200704, 784, 28, 1)), buf337, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf338, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf339 = buf311; del buf311  # reuse
        triton__35.run(buf338, buf339, 832, 7720, grid=grid(832), stream=stream0)
        buf340 = buf312; del buf312  # reuse
        triton__36.run(buf339, buf340, 64, 13, grid=grid(64), stream=stream0)
        buf341 = buf339; del buf339  # reuse
        buf343 = buf309; del buf309  # reuse
        triton__37.run(buf338, buf340, buf341, buf343, 832, 7720, grid=grid(832), stream=stream0)
        buf342 = buf340; del buf340  # reuse
        buf345 = as_strided(buf310, (64, ), (1, )); del buf310  # reuse
        buf347 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf341, primals_346, buf342, buf345, buf347, 64, 13, grid=grid(64), stream=stream0)
        del primals_346
        buf344 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf346 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1058 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf343, primals_345, buf344, buf346, buf1058, 64, 13, grid=grid(64), stream=stream0)
        del primals_345
        buf376 = empty_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf348 = as_strided(buf376, (128, 64, 28, 28), (200704, 784, 28, 1))  # alias
        buf349 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf1057 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__47.run(buf338, buf344, buf342, primals_89, primals_90, buf336, buf348, buf349, buf1057, 6422528, grid=grid(6422528), stream=stream0)
        del primals_90
        buf350 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_91, buf350, 4608, grid=grid(4608), stream=stream0)
        del primals_91
        buf351 = aten.convolution(buf349, buf350, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf351, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf352 = buf343; del buf343  # reuse
        triton__35.run(buf351, buf352, 832, 7720, grid=grid(832), stream=stream0)
        buf353 = buf344; del buf344  # reuse
        triton__36.run(buf352, buf353, 64, 13, grid=grid(64), stream=stream0)
        buf354 = buf352; del buf352  # reuse
        buf356 = buf341; del buf341  # reuse
        triton__37.run(buf351, buf353, buf354, buf356, 832, 7720, grid=grid(832), stream=stream0)
        buf355 = buf353; del buf353  # reuse
        buf358 = as_strided(buf342, (64, ), (1, )); del buf342  # reuse
        buf360 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf354, primals_349, buf355, buf358, buf360, 64, 13, grid=grid(64), stream=stream0)
        del primals_349
        buf357 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf359 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1056 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf356, primals_348, buf357, buf359, buf1056, 64, 13, grid=grid(64), stream=stream0)
        del primals_348
        buf361 = as_strided(buf376, (128, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        buf362 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf1055 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__48.run(buf351, buf357, buf355, primals_92, primals_93, buf336, buf361, buf362, buf1055, 6422528, grid=grid(6422528), stream=stream0)
        del primals_93
        buf363 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_94, buf363, 4608, grid=grid(4608), stream=stream0)
        del primals_94
        buf364 = aten.convolution(buf362, buf363, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf364, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf365 = buf356; del buf356  # reuse
        triton__35.run(buf364, buf365, 832, 7720, grid=grid(832), stream=stream0)
        buf366 = buf357; del buf357  # reuse
        triton__36.run(buf365, buf366, 64, 13, grid=grid(64), stream=stream0)
        buf367 = buf365; del buf365  # reuse
        buf369 = buf354; del buf354  # reuse
        triton__37.run(buf364, buf366, buf367, buf369, 832, 7720, grid=grid(832), stream=stream0)
        buf368 = buf366; del buf366  # reuse
        buf371 = as_strided(buf355, (64, ), (1, )); del buf355  # reuse
        buf373 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf367, primals_352, buf368, buf371, buf373, 64, 13, grid=grid(64), stream=stream0)
        del primals_352
        buf370 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf372 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1054 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf369, primals_351, buf370, buf372, buf1054, 64, 13, grid=grid(64), stream=stream0)
        del primals_351
        buf374 = as_strided(buf376, (128, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        buf1053 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__40.run(buf364, buf370, buf368, primals_95, primals_96, buf374, buf1053, 6422528, grid=grid(6422528), stream=stream0)
        del primals_96
        buf375 = as_strided(buf376, (128, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        triton__49.run(buf336, buf375, 6422528, grid=grid(6422528), stream=stream0)
        buf377 = empty_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__42.run(primals_97, buf377, 131072, grid=grid(131072), stream=stream0)
        del primals_97
        buf378 = aten.convolution(buf376, buf377, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf378, (128, 512, 28, 28), (401408, 784, 28, 1))
        buf380 = buf323; del buf323  # reuse
        buf381 = buf322; del buf322  # reuse
        buf382 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf384 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf383 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf1052 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__43.run(buf378, primals_355, primals_354, buf380, buf381, buf382, buf384, buf383, buf1052, 512, 100352, grid=grid(512), stream=stream0)
        del primals_354
        del primals_355
        buf385 = empty_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda', dtype=torch.float16)
        triton__50.run(buf378, buf381, buf380, primals_98, primals_99, buf327, buf385, 51380224, grid=grid(51380224), stream=stream0)
        del primals_99
        buf386 = empty_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__42.run(primals_100, buf386, 131072, grid=grid(131072), stream=stream0)
        del primals_100
        buf387 = aten.convolution(buf385, buf386, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf387, (128, 256, 28, 28), (200704, 784, 28, 1))
        buf389 = buf332; del buf332  # reuse
        buf390 = buf331; del buf331  # reuse
        buf391 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf393 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf392 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf1051 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__45.run(buf387, primals_358, primals_357, buf389, buf390, buf391, buf393, buf392, buf1051, 256, 100352, grid=grid(256), stream=stream0)
        del primals_357
        del primals_358
        buf394 = empty_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf1050 = empty_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__46.run(buf387, buf390, buf389, primals_101, primals_102, buf394, buf1050, 25690112, grid=grid(25690112), stream=stream0)
        del primals_102
        buf395 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_103, buf395, 4608, grid=grid(4608), stream=stream0)
        del primals_103
        buf396 = aten.convolution(as_strided(buf394, (128, 64, 28, 28), (200704, 784, 28, 1)), buf395, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf396, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf397 = buf369; del buf369  # reuse
        triton__35.run(buf396, buf397, 832, 7720, grid=grid(832), stream=stream0)
        buf398 = buf370; del buf370  # reuse
        triton__36.run(buf397, buf398, 64, 13, grid=grid(64), stream=stream0)
        buf399 = buf397; del buf397  # reuse
        buf401 = buf367; del buf367  # reuse
        triton__37.run(buf396, buf398, buf399, buf401, 832, 7720, grid=grid(832), stream=stream0)
        buf400 = buf398; del buf398  # reuse
        buf403 = as_strided(buf368, (64, ), (1, )); del buf368  # reuse
        buf405 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf399, primals_361, buf400, buf403, buf405, 64, 13, grid=grid(64), stream=stream0)
        del primals_361
        buf402 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf404 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1049 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf401, primals_360, buf402, buf404, buf1049, 64, 13, grid=grid(64), stream=stream0)
        del primals_360
        buf434 = empty_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf406 = as_strided(buf434, (128, 64, 28, 28), (200704, 784, 28, 1))  # alias
        buf407 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf1048 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__47.run(buf396, buf402, buf400, primals_104, primals_105, buf394, buf406, buf407, buf1048, 6422528, grid=grid(6422528), stream=stream0)
        del primals_105
        buf408 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_106, buf408, 4608, grid=grid(4608), stream=stream0)
        del primals_106
        buf409 = aten.convolution(buf407, buf408, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf409, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf410 = buf401; del buf401  # reuse
        triton__35.run(buf409, buf410, 832, 7720, grid=grid(832), stream=stream0)
        buf411 = buf402; del buf402  # reuse
        triton__36.run(buf410, buf411, 64, 13, grid=grid(64), stream=stream0)
        buf412 = buf410; del buf410  # reuse
        buf414 = buf399; del buf399  # reuse
        triton__37.run(buf409, buf411, buf412, buf414, 832, 7720, grid=grid(832), stream=stream0)
        buf413 = buf411; del buf411  # reuse
        buf416 = as_strided(buf400, (64, ), (1, )); del buf400  # reuse
        buf418 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf412, primals_364, buf413, buf416, buf418, 64, 13, grid=grid(64), stream=stream0)
        del primals_364
        buf415 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf417 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1047 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf414, primals_363, buf415, buf417, buf1047, 64, 13, grid=grid(64), stream=stream0)
        del primals_363
        buf419 = as_strided(buf434, (128, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        buf420 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf1046 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__48.run(buf409, buf415, buf413, primals_107, primals_108, buf394, buf419, buf420, buf1046, 6422528, grid=grid(6422528), stream=stream0)
        del primals_108
        buf421 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__34.run(primals_109, buf421, 4608, grid=grid(4608), stream=stream0)
        del primals_109
        buf422 = aten.convolution(buf420, buf421, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf422, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf423 = buf414; del buf414  # reuse
        triton__35.run(buf422, buf423, 832, 7720, grid=grid(832), stream=stream0)
        buf424 = buf415; del buf415  # reuse
        triton__36.run(buf423, buf424, 64, 13, grid=grid(64), stream=stream0)
        buf425 = buf423; del buf423  # reuse
        buf427 = buf412; del buf412  # reuse
        triton__37.run(buf422, buf424, buf425, buf427, 832, 7720, grid=grid(832), stream=stream0)
        buf426 = buf424; del buf424  # reuse
        buf429 = as_strided(buf413, (64, ), (1, )); del buf413  # reuse
        buf431 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__38.run(buf425, primals_367, buf426, buf429, buf431, 64, 13, grid=grid(64), stream=stream0)
        del buf425
        del primals_367
        buf428 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf430 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf1045 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__39.run(buf427, primals_366, buf428, buf430, buf1045, 64, 13, grid=grid(64), stream=stream0)
        del buf427
        del primals_366
        buf432 = as_strided(buf434, (128, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        buf1044 = empty_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__40.run(buf422, buf428, buf426, primals_110, primals_111, buf432, buf1044, 6422528, grid=grid(6422528), stream=stream0)
        del buf426
        del buf428
        del primals_111
        buf433 = as_strided(buf434, (128, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        triton__49.run(buf394, buf433, 6422528, grid=grid(6422528), stream=stream0)
        buf435 = empty_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__42.run(primals_112, buf435, 131072, grid=grid(131072), stream=stream0)
        del primals_112
        buf436 = aten.convolution(buf434, buf435, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf436, (128, 512, 28, 28), (401408, 784, 28, 1))
        buf438 = buf381; del buf381  # reuse
        buf439 = buf380; del buf380  # reuse
        buf440 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf442 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf441 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf1043 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__43.run(buf436, primals_370, primals_369, buf438, buf439, buf440, buf442, buf441, buf1043, 512, 100352, grid=grid(512), stream=stream0)
        del primals_369
        del primals_370
        buf443 = empty_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda', dtype=torch.float16)
        triton__50.run(buf436, buf439, buf438, primals_113, primals_114, buf385, buf443, 51380224, grid=grid(51380224), stream=stream0)
        del primals_114
        buf444 = empty_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__51.run(primals_115, buf444, 262144, grid=grid(262144), stream=stream0)
        del primals_115
        buf445 = aten.convolution(buf443, buf444, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf445, (128, 512, 28, 28), (401408, 784, 28, 1))
        buf447 = buf439; del buf439  # reuse
        buf448 = buf438; del buf438  # reuse
        buf449 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf451 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf450 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf1042 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__43.run(buf445, primals_373, primals_372, buf447, buf448, buf449, buf451, buf450, buf1042, 512, 100352, grid=grid(512), stream=stream0)
        del primals_372
        del primals_373
        buf452 = empty_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf1041 = empty_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda', dtype=torch.bool)
        triton__52.run(buf445, buf448, buf447, primals_116, primals_117, buf452, buf1041, 51380224, grid=grid(51380224), stream=stream0)
        del primals_117
        buf453 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_118, buf453, 18432, grid=grid(18432), stream=stream0)
        del primals_118
        buf454 = aten.convolution(as_strided(buf452, (128, 128, 28, 28), (401408, 784, 28, 1)), buf453, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf454, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf455 = as_strided(buf448, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf448  # reuse
        triton__54.run(buf454, buf455, 512, 6272, grid=grid(512), stream=stream0)
        buf456 = buf151; del buf151  # reuse
        triton__11.run(buf455, buf456, 128, 4, grid=grid(128), stream=stream0)
        buf457 = buf455; del buf455  # reuse
        buf459 = as_strided(buf447, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf447  # reuse
        triton__55.run(buf454, buf456, buf457, buf459, 512, 6272, grid=grid(512), stream=stream0)
        buf458 = buf456; del buf456  # reuse
        buf461 = as_strided(buf149, (128, ), (1, )); del buf149  # reuse
        buf463 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf457, primals_376, buf458, buf461, buf463, 128, 4, grid=grid(128), stream=stream0)
        del primals_376
        buf460 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf462 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1040 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf459, primals_375, buf460, buf462, buf1040, 128, 4, grid=grid(128), stream=stream0)
        del primals_375
        buf490 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf464 = as_strided(buf490, (128, 128, 14, 14), (100352, 196, 14, 1))  # alias
        buf1039 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__58.run(buf454, buf460, buf458, primals_119, primals_120, buf464, buf1039, 3211264, grid=grid(3211264), stream=stream0)
        del primals_120
        buf465 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_121, buf465, 18432, grid=grid(18432), stream=stream0)
        del primals_121
        buf466 = aten.convolution(as_strided(buf452, (128, 128, 28, 28), (401408, 784, 28, 1), 100352), buf465, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf466, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf467 = buf459; del buf459  # reuse
        triton__54.run(buf466, buf467, 512, 6272, grid=grid(512), stream=stream0)
        buf468 = buf460; del buf460  # reuse
        triton__11.run(buf467, buf468, 128, 4, grid=grid(128), stream=stream0)
        buf469 = buf467; del buf467  # reuse
        buf471 = buf457; del buf457  # reuse
        triton__55.run(buf466, buf468, buf469, buf471, 512, 6272, grid=grid(512), stream=stream0)
        buf470 = buf468; del buf468  # reuse
        buf473 = as_strided(buf458, (128, ), (1, )); del buf458  # reuse
        buf475 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf469, primals_379, buf470, buf473, buf475, 128, 4, grid=grid(128), stream=stream0)
        del primals_379
        buf472 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf474 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1038 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf471, primals_378, buf472, buf474, buf1038, 128, 4, grid=grid(128), stream=stream0)
        del primals_378
        buf476 = as_strided(buf490, (128, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf1037 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__58.run(buf466, buf472, buf470, primals_122, primals_123, buf476, buf1037, 3211264, grid=grid(3211264), stream=stream0)
        del primals_123
        buf477 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_124, buf477, 18432, grid=grid(18432), stream=stream0)
        del primals_124
        buf478 = aten.convolution(as_strided(buf452, (128, 128, 28, 28), (401408, 784, 28, 1), 200704), buf477, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf478, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf479 = buf471; del buf471  # reuse
        triton__54.run(buf478, buf479, 512, 6272, grid=grid(512), stream=stream0)
        buf480 = buf472; del buf472  # reuse
        triton__11.run(buf479, buf480, 128, 4, grid=grid(128), stream=stream0)
        buf481 = buf479; del buf479  # reuse
        buf483 = buf469; del buf469  # reuse
        triton__55.run(buf478, buf480, buf481, buf483, 512, 6272, grid=grid(512), stream=stream0)
        buf482 = buf480; del buf480  # reuse
        buf485 = as_strided(buf470, (128, ), (1, )); del buf470  # reuse
        buf487 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf481, primals_382, buf482, buf485, buf487, 128, 4, grid=grid(128), stream=stream0)
        del primals_382
        buf484 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf486 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1036 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf483, primals_381, buf484, buf486, buf1036, 128, 4, grid=grid(128), stream=stream0)
        del primals_381
        buf488 = as_strided(buf490, (128, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf1035 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__58.run(buf478, buf484, buf482, primals_125, primals_126, buf488, buf1035, 3211264, grid=grid(3211264), stream=stream0)
        del primals_126
        buf489 = as_strided(buf490, (128, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        triton__59.run(buf452, buf489, 3211264, grid=grid(3211264), stream=stream0)
        buf491 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_127, buf491, 524288, grid=grid(524288), stream=stream0)
        del primals_127
        buf492 = aten.convolution(buf490, buf491, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf492, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf494 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf495 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf496 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf498 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf497 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf1034 = empty_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__61.run(buf492, primals_385, primals_384, buf494, buf495, buf496, buf498, buf497, buf1034, 1024, 25088, grid=grid(1024), stream=stream0)
        del primals_384
        del primals_385
        buf499 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_130, buf499, 524288, grid=grid(524288), stream=stream0)
        del primals_130
        buf500 = aten.convolution(buf443, buf499, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf500, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf502 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf503 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf504 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf506 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf505 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf1033 = empty_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__61.run(buf500, primals_388, primals_387, buf502, buf503, buf504, buf506, buf505, buf1033, 1024, 25088, grid=grid(1024), stream=stream0)
        del primals_387
        del primals_388
        buf507 = empty_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf508 = buf507; del buf507  # reuse
        triton__62.run(buf508, buf492, buf495, buf494, primals_128, primals_129, buf500, buf503, buf502, primals_131, primals_132, 25690112, grid=grid(25690112), stream=stream0)
        del primals_129
        del primals_132
        buf509 = empty_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_133, buf509, 524288, grid=grid(524288), stream=stream0)
        del primals_133
        buf510 = aten.convolution(buf508, buf509, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf510, (128, 512, 14, 14), (100352, 196, 14, 1))
        buf512 = as_strided(buf483, (1, 512, 1, 1), (512, 1, 512, 512)); del buf483  # reuse
        buf513 = as_strided(buf481, (1, 512, 1, 1), (512, 1, 512, 512)); del buf481  # reuse
        buf514 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf516 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf515 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf1032 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf510, primals_391, primals_390, buf512, buf513, buf514, buf516, buf515, buf1032, 512, 25088, grid=grid(512), stream=stream0)
        del primals_390
        del primals_391
        buf517 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1031 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__64.run(buf510, buf513, buf512, primals_134, primals_135, buf517, buf1031, 12845056, grid=grid(12845056), stream=stream0)
        del primals_135
        buf518 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_136, buf518, 18432, grid=grid(18432), stream=stream0)
        del primals_136
        buf519 = aten.convolution(as_strided(buf517, (128, 128, 14, 14), (100352, 196, 14, 1)), buf518, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf519, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf520 = as_strided(buf513, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf513  # reuse
        triton__54.run(buf519, buf520, 512, 6272, grid=grid(512), stream=stream0)
        buf521 = buf484; del buf484  # reuse
        triton__11.run(buf520, buf521, 128, 4, grid=grid(128), stream=stream0)
        buf522 = buf520; del buf520  # reuse
        buf524 = as_strided(buf512, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf512  # reuse
        triton__55.run(buf519, buf521, buf522, buf524, 512, 6272, grid=grid(512), stream=stream0)
        buf523 = buf521; del buf521  # reuse
        buf526 = as_strided(buf482, (128, ), (1, )); del buf482  # reuse
        buf528 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf522, primals_394, buf523, buf526, buf528, 128, 4, grid=grid(128), stream=stream0)
        del primals_394
        buf525 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf527 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1030 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf524, primals_393, buf525, buf527, buf1030, 128, 4, grid=grid(128), stream=stream0)
        del primals_393
        buf557 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf529 = as_strided(buf557, (128, 128, 14, 14), (100352, 196, 14, 1))  # alias
        buf530 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1029 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__65.run(buf519, buf525, buf523, primals_137, primals_138, buf517, buf529, buf530, buf1029, 3211264, grid=grid(3211264), stream=stream0)
        del primals_138
        buf531 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_139, buf531, 18432, grid=grid(18432), stream=stream0)
        del primals_139
        buf532 = aten.convolution(buf530, buf531, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf532, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf533 = buf524; del buf524  # reuse
        triton__54.run(buf532, buf533, 512, 6272, grid=grid(512), stream=stream0)
        buf534 = buf525; del buf525  # reuse
        triton__11.run(buf533, buf534, 128, 4, grid=grid(128), stream=stream0)
        buf535 = buf533; del buf533  # reuse
        buf537 = buf522; del buf522  # reuse
        triton__55.run(buf532, buf534, buf535, buf537, 512, 6272, grid=grid(512), stream=stream0)
        buf536 = buf534; del buf534  # reuse
        buf539 = as_strided(buf523, (128, ), (1, )); del buf523  # reuse
        buf541 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf535, primals_397, buf536, buf539, buf541, 128, 4, grid=grid(128), stream=stream0)
        del primals_397
        buf538 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf540 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1028 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf537, primals_396, buf538, buf540, buf1028, 128, 4, grid=grid(128), stream=stream0)
        del primals_396
        buf542 = as_strided(buf557, (128, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf543 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1027 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__66.run(buf532, buf538, buf536, primals_140, primals_141, buf517, buf542, buf543, buf1027, 3211264, grid=grid(3211264), stream=stream0)
        del primals_141
        buf544 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_142, buf544, 18432, grid=grid(18432), stream=stream0)
        del primals_142
        buf545 = aten.convolution(buf543, buf544, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf545, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf546 = buf537; del buf537  # reuse
        triton__54.run(buf545, buf546, 512, 6272, grid=grid(512), stream=stream0)
        buf547 = buf538; del buf538  # reuse
        triton__11.run(buf546, buf547, 128, 4, grid=grid(128), stream=stream0)
        buf548 = buf546; del buf546  # reuse
        buf550 = buf535; del buf535  # reuse
        triton__55.run(buf545, buf547, buf548, buf550, 512, 6272, grid=grid(512), stream=stream0)
        buf549 = buf547; del buf547  # reuse
        buf552 = as_strided(buf536, (128, ), (1, )); del buf536  # reuse
        buf554 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf548, primals_400, buf549, buf552, buf554, 128, 4, grid=grid(128), stream=stream0)
        del primals_400
        buf551 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf553 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1026 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf550, primals_399, buf551, buf553, buf1026, 128, 4, grid=grid(128), stream=stream0)
        del primals_399
        buf555 = as_strided(buf557, (128, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf1025 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__58.run(buf545, buf551, buf549, primals_143, primals_144, buf555, buf1025, 3211264, grid=grid(3211264), stream=stream0)
        del primals_144
        buf556 = as_strided(buf557, (128, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        triton__67.run(buf517, buf556, 3211264, grid=grid(3211264), stream=stream0)
        buf558 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_145, buf558, 524288, grid=grid(524288), stream=stream0)
        del primals_145
        buf559 = aten.convolution(buf557, buf558, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf559, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf561 = buf503; del buf503  # reuse
        buf562 = buf502; del buf502  # reuse
        buf563 = as_strided(buf495, (1024, ), (1, )); del buf495  # reuse
        buf565 = as_strided(buf494, (1024, ), (1, )); del buf494  # reuse
        buf564 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf1024 = empty_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__61.run(buf559, primals_403, primals_402, buf561, buf562, buf563, buf565, buf564, buf1024, 1024, 25088, grid=grid(1024), stream=stream0)
        del primals_402
        del primals_403
        buf566 = empty_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__68.run(buf559, buf562, buf561, primals_146, primals_147, buf508, buf566, 25690112, grid=grid(25690112), stream=stream0)
        del primals_147
        buf567 = empty_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_148, buf567, 524288, grid=grid(524288), stream=stream0)
        del primals_148
        buf568 = aten.convolution(buf566, buf567, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf568, (128, 512, 14, 14), (100352, 196, 14, 1))
        buf570 = as_strided(buf550, (1, 512, 1, 1), (512, 1, 512, 512)); del buf550  # reuse
        buf571 = as_strided(buf548, (1, 512, 1, 1), (512, 1, 512, 512)); del buf548  # reuse
        buf572 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf574 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf573 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf1023 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf568, primals_406, primals_405, buf570, buf571, buf572, buf574, buf573, buf1023, 512, 25088, grid=grid(512), stream=stream0)
        del primals_405
        del primals_406
        buf575 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1022 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__64.run(buf568, buf571, buf570, primals_149, primals_150, buf575, buf1022, 12845056, grid=grid(12845056), stream=stream0)
        del primals_150
        buf576 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_151, buf576, 18432, grid=grid(18432), stream=stream0)
        del primals_151
        buf577 = aten.convolution(as_strided(buf575, (128, 128, 14, 14), (100352, 196, 14, 1)), buf576, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf577, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf578 = as_strided(buf571, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf571  # reuse
        triton__54.run(buf577, buf578, 512, 6272, grid=grid(512), stream=stream0)
        buf579 = buf551; del buf551  # reuse
        triton__11.run(buf578, buf579, 128, 4, grid=grid(128), stream=stream0)
        buf580 = buf578; del buf578  # reuse
        buf582 = as_strided(buf570, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf570  # reuse
        triton__55.run(buf577, buf579, buf580, buf582, 512, 6272, grid=grid(512), stream=stream0)
        buf581 = buf579; del buf579  # reuse
        buf584 = as_strided(buf549, (128, ), (1, )); del buf549  # reuse
        buf586 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf580, primals_409, buf581, buf584, buf586, 128, 4, grid=grid(128), stream=stream0)
        del primals_409
        buf583 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf585 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1021 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf582, primals_408, buf583, buf585, buf1021, 128, 4, grid=grid(128), stream=stream0)
        del primals_408
        buf615 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf587 = as_strided(buf615, (128, 128, 14, 14), (100352, 196, 14, 1))  # alias
        buf588 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1020 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__65.run(buf577, buf583, buf581, primals_152, primals_153, buf575, buf587, buf588, buf1020, 3211264, grid=grid(3211264), stream=stream0)
        del primals_153
        buf589 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_154, buf589, 18432, grid=grid(18432), stream=stream0)
        del primals_154
        buf590 = aten.convolution(buf588, buf589, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf590, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf591 = buf582; del buf582  # reuse
        triton__54.run(buf590, buf591, 512, 6272, grid=grid(512), stream=stream0)
        buf592 = buf583; del buf583  # reuse
        triton__11.run(buf591, buf592, 128, 4, grid=grid(128), stream=stream0)
        buf593 = buf591; del buf591  # reuse
        buf595 = buf580; del buf580  # reuse
        triton__55.run(buf590, buf592, buf593, buf595, 512, 6272, grid=grid(512), stream=stream0)
        buf594 = buf592; del buf592  # reuse
        buf597 = as_strided(buf581, (128, ), (1, )); del buf581  # reuse
        buf599 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf593, primals_412, buf594, buf597, buf599, 128, 4, grid=grid(128), stream=stream0)
        del primals_412
        buf596 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf598 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1019 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf595, primals_411, buf596, buf598, buf1019, 128, 4, grid=grid(128), stream=stream0)
        del primals_411
        buf600 = as_strided(buf615, (128, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf601 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1018 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__66.run(buf590, buf596, buf594, primals_155, primals_156, buf575, buf600, buf601, buf1018, 3211264, grid=grid(3211264), stream=stream0)
        del primals_156
        buf602 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_157, buf602, 18432, grid=grid(18432), stream=stream0)
        del primals_157
        buf603 = aten.convolution(buf601, buf602, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf603, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf604 = buf595; del buf595  # reuse
        triton__54.run(buf603, buf604, 512, 6272, grid=grid(512), stream=stream0)
        buf605 = buf596; del buf596  # reuse
        triton__11.run(buf604, buf605, 128, 4, grid=grid(128), stream=stream0)
        buf606 = buf604; del buf604  # reuse
        buf608 = buf593; del buf593  # reuse
        triton__55.run(buf603, buf605, buf606, buf608, 512, 6272, grid=grid(512), stream=stream0)
        buf607 = buf605; del buf605  # reuse
        buf610 = as_strided(buf594, (128, ), (1, )); del buf594  # reuse
        buf612 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf606, primals_415, buf607, buf610, buf612, 128, 4, grid=grid(128), stream=stream0)
        del primals_415
        buf609 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf611 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1017 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf608, primals_414, buf609, buf611, buf1017, 128, 4, grid=grid(128), stream=stream0)
        del primals_414
        buf613 = as_strided(buf615, (128, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf1016 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__58.run(buf603, buf609, buf607, primals_158, primals_159, buf613, buf1016, 3211264, grid=grid(3211264), stream=stream0)
        del primals_159
        buf614 = as_strided(buf615, (128, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        triton__67.run(buf575, buf614, 3211264, grid=grid(3211264), stream=stream0)
        buf616 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_160, buf616, 524288, grid=grid(524288), stream=stream0)
        del primals_160
        buf617 = aten.convolution(buf615, buf616, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf617, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf619 = buf562; del buf562  # reuse
        buf620 = buf561; del buf561  # reuse
        buf621 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf623 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf622 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf1015 = empty_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__61.run(buf617, primals_418, primals_417, buf619, buf620, buf621, buf623, buf622, buf1015, 1024, 25088, grid=grid(1024), stream=stream0)
        del primals_417
        del primals_418
        buf624 = empty_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__68.run(buf617, buf620, buf619, primals_161, primals_162, buf566, buf624, 25690112, grid=grid(25690112), stream=stream0)
        del primals_162
        buf625 = empty_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_163, buf625, 524288, grid=grid(524288), stream=stream0)
        del primals_163
        buf626 = aten.convolution(buf624, buf625, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf626, (128, 512, 14, 14), (100352, 196, 14, 1))
        buf628 = as_strided(buf608, (1, 512, 1, 1), (512, 1, 512, 512)); del buf608  # reuse
        buf629 = as_strided(buf606, (1, 512, 1, 1), (512, 1, 512, 512)); del buf606  # reuse
        buf630 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf632 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf631 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf1014 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf626, primals_421, primals_420, buf628, buf629, buf630, buf632, buf631, buf1014, 512, 25088, grid=grid(512), stream=stream0)
        del primals_420
        del primals_421
        buf633 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1013 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__64.run(buf626, buf629, buf628, primals_164, primals_165, buf633, buf1013, 12845056, grid=grid(12845056), stream=stream0)
        del primals_165
        buf634 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_166, buf634, 18432, grid=grid(18432), stream=stream0)
        del primals_166
        buf635 = aten.convolution(as_strided(buf633, (128, 128, 14, 14), (100352, 196, 14, 1)), buf634, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf635, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf636 = as_strided(buf629, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf629  # reuse
        triton__54.run(buf635, buf636, 512, 6272, grid=grid(512), stream=stream0)
        buf637 = buf609; del buf609  # reuse
        triton__11.run(buf636, buf637, 128, 4, grid=grid(128), stream=stream0)
        buf638 = buf636; del buf636  # reuse
        buf640 = as_strided(buf628, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf628  # reuse
        triton__55.run(buf635, buf637, buf638, buf640, 512, 6272, grid=grid(512), stream=stream0)
        buf639 = buf637; del buf637  # reuse
        buf642 = as_strided(buf607, (128, ), (1, )); del buf607  # reuse
        buf644 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf638, primals_424, buf639, buf642, buf644, 128, 4, grid=grid(128), stream=stream0)
        del primals_424
        buf641 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf643 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1012 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf640, primals_423, buf641, buf643, buf1012, 128, 4, grid=grid(128), stream=stream0)
        del primals_423
        buf673 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf645 = as_strided(buf673, (128, 128, 14, 14), (100352, 196, 14, 1))  # alias
        buf646 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1011 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__65.run(buf635, buf641, buf639, primals_167, primals_168, buf633, buf645, buf646, buf1011, 3211264, grid=grid(3211264), stream=stream0)
        del primals_168
        buf647 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_169, buf647, 18432, grid=grid(18432), stream=stream0)
        del primals_169
        buf648 = aten.convolution(buf646, buf647, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf648, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf649 = buf640; del buf640  # reuse
        triton__54.run(buf648, buf649, 512, 6272, grid=grid(512), stream=stream0)
        buf650 = buf641; del buf641  # reuse
        triton__11.run(buf649, buf650, 128, 4, grid=grid(128), stream=stream0)
        buf651 = buf649; del buf649  # reuse
        buf653 = buf638; del buf638  # reuse
        triton__55.run(buf648, buf650, buf651, buf653, 512, 6272, grid=grid(512), stream=stream0)
        buf652 = buf650; del buf650  # reuse
        buf655 = as_strided(buf639, (128, ), (1, )); del buf639  # reuse
        buf657 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf651, primals_427, buf652, buf655, buf657, 128, 4, grid=grid(128), stream=stream0)
        del primals_427
        buf654 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf656 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1010 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf653, primals_426, buf654, buf656, buf1010, 128, 4, grid=grid(128), stream=stream0)
        del primals_426
        buf658 = as_strided(buf673, (128, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf659 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1009 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__66.run(buf648, buf654, buf652, primals_170, primals_171, buf633, buf658, buf659, buf1009, 3211264, grid=grid(3211264), stream=stream0)
        del primals_171
        buf660 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_172, buf660, 18432, grid=grid(18432), stream=stream0)
        del primals_172
        buf661 = aten.convolution(buf659, buf660, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf661, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf662 = buf653; del buf653  # reuse
        triton__54.run(buf661, buf662, 512, 6272, grid=grid(512), stream=stream0)
        buf663 = buf654; del buf654  # reuse
        triton__11.run(buf662, buf663, 128, 4, grid=grid(128), stream=stream0)
        buf664 = buf662; del buf662  # reuse
        buf666 = buf651; del buf651  # reuse
        triton__55.run(buf661, buf663, buf664, buf666, 512, 6272, grid=grid(512), stream=stream0)
        buf665 = buf663; del buf663  # reuse
        buf668 = as_strided(buf652, (128, ), (1, )); del buf652  # reuse
        buf670 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf664, primals_430, buf665, buf668, buf670, 128, 4, grid=grid(128), stream=stream0)
        del primals_430
        buf667 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf669 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1008 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf666, primals_429, buf667, buf669, buf1008, 128, 4, grid=grid(128), stream=stream0)
        del primals_429
        buf671 = as_strided(buf673, (128, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf1007 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__58.run(buf661, buf667, buf665, primals_173, primals_174, buf671, buf1007, 3211264, grid=grid(3211264), stream=stream0)
        del primals_174
        buf672 = as_strided(buf673, (128, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        triton__67.run(buf633, buf672, 3211264, grid=grid(3211264), stream=stream0)
        buf674 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_175, buf674, 524288, grid=grid(524288), stream=stream0)
        del primals_175
        buf675 = aten.convolution(buf673, buf674, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf675, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf677 = buf620; del buf620  # reuse
        buf678 = buf619; del buf619  # reuse
        buf679 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf681 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf680 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf1006 = empty_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__61.run(buf675, primals_433, primals_432, buf677, buf678, buf679, buf681, buf680, buf1006, 1024, 25088, grid=grid(1024), stream=stream0)
        del primals_432
        del primals_433
        buf682 = empty_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__68.run(buf675, buf678, buf677, primals_176, primals_177, buf624, buf682, 25690112, grid=grid(25690112), stream=stream0)
        del primals_177
        buf683 = empty_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_178, buf683, 524288, grid=grid(524288), stream=stream0)
        del primals_178
        buf684 = aten.convolution(buf682, buf683, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf684, (128, 512, 14, 14), (100352, 196, 14, 1))
        buf686 = as_strided(buf666, (1, 512, 1, 1), (512, 1, 512, 512)); del buf666  # reuse
        buf687 = as_strided(buf664, (1, 512, 1, 1), (512, 1, 512, 512)); del buf664  # reuse
        buf688 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf690 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf689 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf1005 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf684, primals_436, primals_435, buf686, buf687, buf688, buf690, buf689, buf1005, 512, 25088, grid=grid(512), stream=stream0)
        del primals_435
        del primals_436
        buf691 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1004 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__64.run(buf684, buf687, buf686, primals_179, primals_180, buf691, buf1004, 12845056, grid=grid(12845056), stream=stream0)
        del primals_180
        buf692 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_181, buf692, 18432, grid=grid(18432), stream=stream0)
        del primals_181
        buf693 = aten.convolution(as_strided(buf691, (128, 128, 14, 14), (100352, 196, 14, 1)), buf692, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf693, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf694 = as_strided(buf687, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf687  # reuse
        triton__54.run(buf693, buf694, 512, 6272, grid=grid(512), stream=stream0)
        buf695 = buf667; del buf667  # reuse
        triton__11.run(buf694, buf695, 128, 4, grid=grid(128), stream=stream0)
        buf696 = buf694; del buf694  # reuse
        buf698 = as_strided(buf686, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf686  # reuse
        triton__55.run(buf693, buf695, buf696, buf698, 512, 6272, grid=grid(512), stream=stream0)
        buf697 = buf695; del buf695  # reuse
        buf700 = as_strided(buf665, (128, ), (1, )); del buf665  # reuse
        buf702 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf696, primals_439, buf697, buf700, buf702, 128, 4, grid=grid(128), stream=stream0)
        del primals_439
        buf699 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf701 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1003 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf698, primals_438, buf699, buf701, buf1003, 128, 4, grid=grid(128), stream=stream0)
        del primals_438
        buf731 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf703 = as_strided(buf731, (128, 128, 14, 14), (100352, 196, 14, 1))  # alias
        buf704 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1002 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__65.run(buf693, buf699, buf697, primals_182, primals_183, buf691, buf703, buf704, buf1002, 3211264, grid=grid(3211264), stream=stream0)
        del primals_183
        buf705 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_184, buf705, 18432, grid=grid(18432), stream=stream0)
        del primals_184
        buf706 = aten.convolution(buf704, buf705, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf706, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf707 = buf698; del buf698  # reuse
        triton__54.run(buf706, buf707, 512, 6272, grid=grid(512), stream=stream0)
        buf708 = buf699; del buf699  # reuse
        triton__11.run(buf707, buf708, 128, 4, grid=grid(128), stream=stream0)
        buf709 = buf707; del buf707  # reuse
        buf711 = buf696; del buf696  # reuse
        triton__55.run(buf706, buf708, buf709, buf711, 512, 6272, grid=grid(512), stream=stream0)
        buf710 = buf708; del buf708  # reuse
        buf713 = as_strided(buf697, (128, ), (1, )); del buf697  # reuse
        buf715 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf709, primals_442, buf710, buf713, buf715, 128, 4, grid=grid(128), stream=stream0)
        del primals_442
        buf712 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf714 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf1001 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf711, primals_441, buf712, buf714, buf1001, 128, 4, grid=grid(128), stream=stream0)
        del primals_441
        buf716 = as_strided(buf731, (128, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf717 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf1000 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__66.run(buf706, buf712, buf710, primals_185, primals_186, buf691, buf716, buf717, buf1000, 3211264, grid=grid(3211264), stream=stream0)
        del primals_186
        buf718 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_187, buf718, 18432, grid=grid(18432), stream=stream0)
        del primals_187
        buf719 = aten.convolution(buf717, buf718, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf719, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf720 = buf711; del buf711  # reuse
        triton__54.run(buf719, buf720, 512, 6272, grid=grid(512), stream=stream0)
        buf721 = buf712; del buf712  # reuse
        triton__11.run(buf720, buf721, 128, 4, grid=grid(128), stream=stream0)
        buf722 = buf720; del buf720  # reuse
        buf724 = buf709; del buf709  # reuse
        triton__55.run(buf719, buf721, buf722, buf724, 512, 6272, grid=grid(512), stream=stream0)
        buf723 = buf721; del buf721  # reuse
        buf726 = as_strided(buf710, (128, ), (1, )); del buf710  # reuse
        buf728 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf722, primals_445, buf723, buf726, buf728, 128, 4, grid=grid(128), stream=stream0)
        del primals_445
        buf725 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf727 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf999 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf724, primals_444, buf725, buf727, buf999, 128, 4, grid=grid(128), stream=stream0)
        del primals_444
        buf729 = as_strided(buf731, (128, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf998 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__58.run(buf719, buf725, buf723, primals_188, primals_189, buf729, buf998, 3211264, grid=grid(3211264), stream=stream0)
        del primals_189
        buf730 = as_strided(buf731, (128, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        triton__67.run(buf691, buf730, 3211264, grid=grid(3211264), stream=stream0)
        buf732 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_190, buf732, 524288, grid=grid(524288), stream=stream0)
        del primals_190
        buf733 = aten.convolution(buf731, buf732, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf733, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf735 = buf678; del buf678  # reuse
        buf736 = buf677; del buf677  # reuse
        buf737 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf739 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf738 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf997 = empty_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__61.run(buf733, primals_448, primals_447, buf735, buf736, buf737, buf739, buf738, buf997, 1024, 25088, grid=grid(1024), stream=stream0)
        del primals_447
        del primals_448
        buf740 = empty_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__68.run(buf733, buf736, buf735, primals_191, primals_192, buf682, buf740, 25690112, grid=grid(25690112), stream=stream0)
        del primals_192
        buf741 = empty_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_193, buf741, 524288, grid=grid(524288), stream=stream0)
        del primals_193
        buf742 = aten.convolution(buf740, buf741, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf742, (128, 512, 14, 14), (100352, 196, 14, 1))
        buf744 = as_strided(buf724, (1, 512, 1, 1), (512, 1, 512, 512)); del buf724  # reuse
        buf745 = as_strided(buf722, (1, 512, 1, 1), (512, 1, 512, 512)); del buf722  # reuse
        buf746 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf748 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf747 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf996 = empty_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf742, primals_451, primals_450, buf744, buf745, buf746, buf748, buf747, buf996, 512, 25088, grid=grid(512), stream=stream0)
        del primals_450
        del primals_451
        buf749 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf995 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__64.run(buf742, buf745, buf744, primals_194, primals_195, buf749, buf995, 12845056, grid=grid(12845056), stream=stream0)
        del primals_195
        buf750 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_196, buf750, 18432, grid=grid(18432), stream=stream0)
        del primals_196
        buf751 = aten.convolution(as_strided(buf749, (128, 128, 14, 14), (100352, 196, 14, 1)), buf750, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf751, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf752 = as_strided(buf745, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf745  # reuse
        triton__54.run(buf751, buf752, 512, 6272, grid=grid(512), stream=stream0)
        buf753 = buf725; del buf725  # reuse
        triton__11.run(buf752, buf753, 128, 4, grid=grid(128), stream=stream0)
        buf754 = buf752; del buf752  # reuse
        buf756 = as_strided(buf744, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf744  # reuse
        triton__55.run(buf751, buf753, buf754, buf756, 512, 6272, grid=grid(512), stream=stream0)
        buf755 = buf753; del buf753  # reuse
        buf758 = as_strided(buf723, (128, ), (1, )); del buf723  # reuse
        buf760 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf754, primals_454, buf755, buf758, buf760, 128, 4, grid=grid(128), stream=stream0)
        del primals_454
        buf757 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf759 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf994 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf756, primals_453, buf757, buf759, buf994, 128, 4, grid=grid(128), stream=stream0)
        del primals_453
        buf789 = empty_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf761 = as_strided(buf789, (128, 128, 14, 14), (100352, 196, 14, 1))  # alias
        buf762 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf993 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__65.run(buf751, buf757, buf755, primals_197, primals_198, buf749, buf761, buf762, buf993, 3211264, grid=grid(3211264), stream=stream0)
        del primals_198
        buf763 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_199, buf763, 18432, grid=grid(18432), stream=stream0)
        del primals_199
        buf764 = aten.convolution(buf762, buf763, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf764, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf765 = buf756; del buf756  # reuse
        triton__54.run(buf764, buf765, 512, 6272, grid=grid(512), stream=stream0)
        buf766 = buf757; del buf757  # reuse
        triton__11.run(buf765, buf766, 128, 4, grid=grid(128), stream=stream0)
        buf767 = buf765; del buf765  # reuse
        buf769 = buf754; del buf754  # reuse
        triton__55.run(buf764, buf766, buf767, buf769, 512, 6272, grid=grid(512), stream=stream0)
        buf768 = buf766; del buf766  # reuse
        buf771 = as_strided(buf755, (128, ), (1, )); del buf755  # reuse
        buf773 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf767, primals_457, buf768, buf771, buf773, 128, 4, grid=grid(128), stream=stream0)
        del primals_457
        buf770 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf772 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf992 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf769, primals_456, buf770, buf772, buf992, 128, 4, grid=grid(128), stream=stream0)
        del primals_456
        buf774 = as_strided(buf789, (128, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf775 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf991 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__66.run(buf764, buf770, buf768, primals_200, primals_201, buf749, buf774, buf775, buf991, 3211264, grid=grid(3211264), stream=stream0)
        del primals_201
        buf776 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__53.run(primals_202, buf776, 18432, grid=grid(18432), stream=stream0)
        del primals_202
        buf777 = aten.convolution(buf775, buf776, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf777, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf778 = buf769; del buf769  # reuse
        triton__54.run(buf777, buf778, 512, 6272, grid=grid(512), stream=stream0)
        buf779 = buf770; del buf770  # reuse
        triton__11.run(buf778, buf779, 128, 4, grid=grid(128), stream=stream0)
        buf780 = buf778; del buf778  # reuse
        buf782 = buf767; del buf767  # reuse
        triton__55.run(buf777, buf779, buf780, buf782, 512, 6272, grid=grid(512), stream=stream0)
        buf781 = buf779; del buf779  # reuse
        buf784 = as_strided(buf768, (128, ), (1, )); del buf768  # reuse
        buf786 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf780, primals_460, buf781, buf784, buf786, 128, 4, grid=grid(128), stream=stream0)
        del buf780
        del primals_460
        buf783 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf785 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf990 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__57.run(buf782, primals_459, buf783, buf785, buf990, 128, 4, grid=grid(128), stream=stream0)
        del buf782
        del primals_459
        buf787 = as_strided(buf789, (128, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf989 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__58.run(buf777, buf783, buf781, primals_203, primals_204, buf787, buf989, 3211264, grid=grid(3211264), stream=stream0)
        del buf781
        del buf783
        del primals_204
        buf788 = as_strided(buf789, (128, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        triton__67.run(buf749, buf788, 3211264, grid=grid(3211264), stream=stream0)
        buf790 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(primals_205, buf790, 524288, grid=grid(524288), stream=stream0)
        del primals_205
        buf791 = aten.convolution(buf789, buf790, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf791, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf793 = buf736; del buf736  # reuse
        buf794 = buf735; del buf735  # reuse
        buf795 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf797 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf796 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf988 = empty_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__61.run(buf791, primals_463, primals_462, buf793, buf794, buf795, buf797, buf796, buf988, 1024, 25088, grid=grid(1024), stream=stream0)
        del primals_462
        del primals_463
        buf798 = empty_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__68.run(buf791, buf794, buf793, primals_206, primals_207, buf740, buf798, 25690112, grid=grid(25690112), stream=stream0)
        del primals_207
        buf799 = empty_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__69.run(primals_208, buf799, 1048576, grid=grid(1048576), stream=stream0)
        del primals_208
        buf800 = aten.convolution(buf798, buf799, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf800, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf802 = buf794; del buf794  # reuse
        buf803 = buf793; del buf793  # reuse
        buf804 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf806 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf805 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf987 = empty_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__61.run(buf800, primals_466, primals_465, buf802, buf803, buf804, buf806, buf805, buf987, 1024, 25088, grid=grid(1024), stream=stream0)
        del primals_465
        del primals_466
        buf807 = empty_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf986 = empty_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda', dtype=torch.bool)
        triton__70.run(buf800, buf803, buf802, primals_209, primals_210, buf807, buf986, 25690112, grid=grid(25690112), stream=stream0)
        del primals_210
        buf808 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__71.run(primals_211, buf808, 73728, grid=grid(73728), stream=stream0)
        del primals_211
        buf809 = aten.convolution(as_strided(buf807, (128, 256, 14, 14), (200704, 196, 14, 1)), buf808, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf809, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf811 = buf390; del buf390  # reuse
        buf812 = buf389; del buf389  # reuse
        buf813 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf815 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf814 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf985 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__72.run(buf809, primals_469, primals_468, buf811, buf812, buf813, buf815, buf814, buf985, 256, 6272, grid=grid(256), stream=stream0)
        del primals_468
        del primals_469
        buf836 = empty_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf816 = as_strided(buf836, (128, 256, 7, 7), (50176, 49, 7, 1))  # alias
        buf984 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.bool)
        triton__73.run(buf809, buf812, buf811, primals_212, primals_213, buf816, buf984, 1605632, grid=grid(1605632), stream=stream0)
        del primals_213
        buf817 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__71.run(primals_214, buf817, 73728, grid=grid(73728), stream=stream0)
        del primals_214
        buf818 = aten.convolution(as_strided(buf807, (128, 256, 14, 14), (200704, 196, 14, 1), 50176), buf817, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf818, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf820 = buf812; del buf812  # reuse
        buf821 = buf811; del buf811  # reuse
        buf822 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf824 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf823 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf983 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__72.run(buf818, primals_472, primals_471, buf820, buf821, buf822, buf824, buf823, buf983, 256, 6272, grid=grid(256), stream=stream0)
        del primals_471
        del primals_472
        buf825 = as_strided(buf836, (128, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        buf982 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.bool)
        triton__73.run(buf818, buf821, buf820, primals_215, primals_216, buf825, buf982, 1605632, grid=grid(1605632), stream=stream0)
        del primals_216
        buf826 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__71.run(primals_217, buf826, 73728, grid=grid(73728), stream=stream0)
        del primals_217
        buf827 = aten.convolution(as_strided(buf807, (128, 256, 14, 14), (200704, 196, 14, 1), 100352), buf826, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf827, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf829 = buf821; del buf821  # reuse
        buf830 = buf820; del buf820  # reuse
        buf831 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf833 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf832 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf981 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__72.run(buf827, primals_475, primals_474, buf829, buf830, buf831, buf833, buf832, buf981, 256, 6272, grid=grid(256), stream=stream0)
        del primals_474
        del primals_475
        buf834 = as_strided(buf836, (128, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        buf980 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.bool)
        triton__73.run(buf827, buf830, buf829, primals_218, primals_219, buf834, buf980, 1605632, grid=grid(1605632), stream=stream0)
        del primals_219
        buf835 = as_strided(buf836, (128, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        triton__74.run(buf807, buf835, 1605632, grid=grid(1605632), stream=stream0)
        buf837 = empty_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__75.run(primals_220, buf837, 2097152, grid=grid(2097152), stream=stream0)
        del primals_220
        buf838 = aten.convolution(buf836, buf837, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf838, (128, 2048, 7, 7), (100352, 49, 7, 1))
        buf840 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf841 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf842 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf844 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf843 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf979 = empty_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__76.run(buf838, primals_478, primals_477, buf840, buf841, buf842, buf844, buf843, buf979, 2048, 6272, grid=grid(2048), stream=stream0)
        del primals_477
        del primals_478
        buf845 = empty_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__75.run(primals_223, buf845, 2097152, grid=grid(2097152), stream=stream0)
        del primals_223
        buf846 = aten.convolution(buf798, buf845, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf846, (128, 2048, 7, 7), (100352, 49, 7, 1))
        buf848 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf849 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf850 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf852 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf851 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf978 = empty_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__76.run(buf846, primals_481, primals_480, buf848, buf849, buf850, buf852, buf851, buf978, 2048, 6272, grid=grid(2048), stream=stream0)
        del primals_480
        del primals_481
        buf853 = empty_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf854 = buf853; del buf853  # reuse
        triton__77.run(buf854, buf838, buf841, buf840, primals_221, primals_222, buf846, buf849, buf848, primals_224, primals_225, 12845056, grid=grid(12845056), stream=stream0)
        del primals_222
        del primals_225
        buf855 = empty_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__75.run(primals_226, buf855, 2097152, grid=grid(2097152), stream=stream0)
        del primals_226
        buf856 = aten.convolution(buf854, buf855, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf856, (128, 1024, 7, 7), (50176, 49, 7, 1))
        buf858 = buf803; del buf803  # reuse
        buf859 = buf802; del buf802  # reuse
        buf860 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf862 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf861 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf977 = empty_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__78.run(buf856, primals_484, primals_483, buf858, buf859, buf860, buf862, buf861, buf977, 1024, 6272, grid=grid(1024), stream=stream0)
        del primals_483
        del primals_484
        buf863 = empty_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf976 = empty_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda', dtype=torch.bool)
        triton__79.run(buf856, buf859, buf858, primals_227, primals_228, buf863, buf976, 6422528, grid=grid(6422528), stream=stream0)
        del primals_228
        buf864 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__71.run(primals_229, buf864, 73728, grid=grid(73728), stream=stream0)
        del primals_229
        buf865 = aten.convolution(as_strided(buf863, (128, 256, 7, 7), (50176, 49, 7, 1)), buf864, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf865, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf867 = buf830; del buf830  # reuse
        buf868 = buf829; del buf829  # reuse
        buf869 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf871 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf870 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf975 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__72.run(buf865, primals_487, primals_486, buf867, buf868, buf869, buf871, buf870, buf975, 256, 6272, grid=grid(256), stream=stream0)
        del primals_486
        del primals_487
        buf894 = empty_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf872 = as_strided(buf894, (128, 256, 7, 7), (50176, 49, 7, 1))  # alias
        buf873 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf974 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.bool)
        triton__80.run(buf865, buf868, buf867, primals_230, primals_231, buf863, buf872, buf873, buf974, 1605632, grid=grid(1605632), stream=stream0)
        del primals_231
        buf874 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__71.run(primals_232, buf874, 73728, grid=grid(73728), stream=stream0)
        del primals_232
        buf875 = aten.convolution(buf873, buf874, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf875, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf877 = buf868; del buf868  # reuse
        buf878 = buf867; del buf867  # reuse
        buf879 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf881 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf880 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf973 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__72.run(buf875, primals_490, primals_489, buf877, buf878, buf879, buf881, buf880, buf973, 256, 6272, grid=grid(256), stream=stream0)
        del primals_489
        del primals_490
        buf882 = as_strided(buf894, (128, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        buf883 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf972 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.bool)
        triton__81.run(buf875, buf878, buf877, primals_233, primals_234, buf863, buf882, buf883, buf972, 1605632, grid=grid(1605632), stream=stream0)
        del primals_234
        buf884 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__71.run(primals_235, buf884, 73728, grid=grid(73728), stream=stream0)
        del primals_235
        buf885 = aten.convolution(buf883, buf884, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf885, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf887 = buf878; del buf878  # reuse
        buf888 = buf877; del buf877  # reuse
        buf889 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf891 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf890 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf971 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__72.run(buf885, primals_493, primals_492, buf887, buf888, buf889, buf891, buf890, buf971, 256, 6272, grid=grid(256), stream=stream0)
        del primals_492
        del primals_493
        buf892 = as_strided(buf894, (128, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        buf970 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.bool)
        triton__73.run(buf885, buf888, buf887, primals_236, primals_237, buf892, buf970, 1605632, grid=grid(1605632), stream=stream0)
        del primals_237
        buf893 = as_strided(buf894, (128, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        triton__82.run(buf863, buf893, 1605632, grid=grid(1605632), stream=stream0)
        buf895 = empty_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__75.run(primals_238, buf895, 2097152, grid=grid(2097152), stream=stream0)
        del primals_238
        buf896 = aten.convolution(buf894, buf895, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf896, (128, 2048, 7, 7), (100352, 49, 7, 1))
        buf898 = buf849; del buf849  # reuse
        buf899 = buf848; del buf848  # reuse
        buf900 = as_strided(buf841, (2048, ), (1, )); del buf841  # reuse
        buf902 = as_strided(buf840, (2048, ), (1, )); del buf840  # reuse
        buf901 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf969 = empty_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__76.run(buf896, primals_496, primals_495, buf898, buf899, buf900, buf902, buf901, buf969, 2048, 6272, grid=grid(2048), stream=stream0)
        del primals_495
        del primals_496
        buf903 = empty_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__83.run(buf896, buf899, buf898, primals_239, primals_240, buf854, buf903, 12845056, grid=grid(12845056), stream=stream0)
        del primals_240
        buf904 = empty_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__75.run(primals_241, buf904, 2097152, grid=grid(2097152), stream=stream0)
        del primals_241
        buf905 = aten.convolution(buf903, buf904, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf905, (128, 1024, 7, 7), (50176, 49, 7, 1))
        buf907 = buf859; del buf859  # reuse
        buf908 = buf858; del buf858  # reuse
        buf909 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf911 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf910 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf968 = empty_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__78.run(buf905, primals_499, primals_498, buf907, buf908, buf909, buf911, buf910, buf968, 1024, 6272, grid=grid(1024), stream=stream0)
        del primals_498
        del primals_499
        buf912 = empty_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf967 = empty_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda', dtype=torch.bool)
        triton__79.run(buf905, buf908, buf907, primals_242, primals_243, buf912, buf967, 6422528, grid=grid(6422528), stream=stream0)
        del buf907
        del buf908
        del primals_243
        buf913 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__71.run(primals_244, buf913, 73728, grid=grid(73728), stream=stream0)
        del primals_244
        buf914 = aten.convolution(as_strided(buf912, (128, 256, 7, 7), (50176, 49, 7, 1)), buf913, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf914, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf916 = buf888; del buf888  # reuse
        buf917 = buf887; del buf887  # reuse
        buf918 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf920 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf919 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf966 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__72.run(buf914, primals_502, primals_501, buf916, buf917, buf918, buf920, buf919, buf966, 256, 6272, grid=grid(256), stream=stream0)
        del primals_501
        del primals_502
        buf943 = empty_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf921 = as_strided(buf943, (128, 256, 7, 7), (50176, 49, 7, 1))  # alias
        buf922 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf965 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.bool)
        triton__80.run(buf914, buf917, buf916, primals_245, primals_246, buf912, buf921, buf922, buf965, 1605632, grid=grid(1605632), stream=stream0)
        del primals_246
        buf923 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__71.run(primals_247, buf923, 73728, grid=grid(73728), stream=stream0)
        del primals_247
        buf924 = aten.convolution(buf922, buf923, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf924, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf926 = buf917; del buf917  # reuse
        buf927 = buf916; del buf916  # reuse
        buf928 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf930 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf929 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf964 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__72.run(buf924, primals_505, primals_504, buf926, buf927, buf928, buf930, buf929, buf964, 256, 6272, grid=grid(256), stream=stream0)
        del primals_504
        del primals_505
        buf931 = as_strided(buf943, (128, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        buf932 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf963 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.bool)
        triton__81.run(buf924, buf927, buf926, primals_248, primals_249, buf912, buf931, buf932, buf963, 1605632, grid=grid(1605632), stream=stream0)
        del primals_249
        buf933 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__71.run(primals_250, buf933, 73728, grid=grid(73728), stream=stream0)
        del primals_250
        buf934 = aten.convolution(buf932, buf933, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        assert_size_stride(buf934, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf936 = buf927; del buf927  # reuse
        buf937 = buf926; del buf926  # reuse
        buf938 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf940 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf939 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf962 = empty_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__72.run(buf934, primals_508, primals_507, buf936, buf937, buf938, buf940, buf939, buf962, 256, 6272, grid=grid(256), stream=stream0)
        del primals_507
        del primals_508
        buf941 = as_strided(buf943, (128, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        buf961 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.bool)
        triton__73.run(buf934, buf937, buf936, primals_251, primals_252, buf941, buf961, 1605632, grid=grid(1605632), stream=stream0)
        del buf936
        del buf937
        del primals_252
        buf942 = as_strided(buf943, (128, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        triton__82.run(buf912, buf942, 1605632, grid=grid(1605632), stream=stream0)
        buf944 = empty_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__75.run(primals_253, buf944, 2097152, grid=grid(2097152), stream=stream0)
        del primals_253
        buf945 = aten.convolution(buf943, buf944, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf945, (128, 2048, 7, 7), (100352, 49, 7, 1))
        buf947 = buf899; del buf899  # reuse
        buf948 = buf898; del buf898  # reuse
        buf949 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf951 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf950 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf960 = empty_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__76.run(buf945, primals_511, primals_510, buf947, buf948, buf949, buf951, buf950, buf960, 2048, 6272, grid=grid(2048), stream=stream0)
        del primals_510
        del primals_511
        buf952 = empty_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf959 = empty_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda', dtype=torch.bool)
        buf954 = empty_strided((128, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        triton__84.run(buf945, buf948, buf947, primals_254, primals_255, buf903, buf952, buf959, buf954, 262144, 49, grid=grid(262144), stream=stream0)
        del buf947
        del buf948
        del buf952
        del primals_255
        buf955 = empty_strided((1000, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        buf958 = empty_strided((1000, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        triton__85.run(primals_256, buf955, buf958, 2048000, grid=grid(2048000), stream=stream0)
        del primals_256
        buf956 = empty_strided((1000, ), (1, ), device='cuda', dtype=torch.float16)
        triton__86.run(primals_257, buf956, 1000, grid=grid(1000), stream=stream0)
        del primals_257
        buf957 = empty_strided((128, 1000), (1000, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(buf956, buf954, as_strided(buf955, (2048, 1000), (1, 2048)), alpha=1, beta=1, out=buf957)
        del buf955
        del buf956
        buf1109 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_260, buf1109, 1, grid=grid(1), stream=stream0)
        del primals_260
        buf1110 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_263, buf1110, 1, grid=grid(1), stream=stream0)
        del primals_263
        buf1111 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_266, buf1111, 1, grid=grid(1), stream=stream0)
        del primals_266
        buf1112 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_269, buf1112, 1, grid=grid(1), stream=stream0)
        del primals_269
        buf1113 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_272, buf1113, 1, grid=grid(1), stream=stream0)
        del primals_272
        buf1114 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_275, buf1114, 1, grid=grid(1), stream=stream0)
        del primals_275
        buf1115 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_278, buf1115, 1, grid=grid(1), stream=stream0)
        del primals_278
        buf1116 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_281, buf1116, 1, grid=grid(1), stream=stream0)
        del primals_281
        buf1117 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_284, buf1117, 1, grid=grid(1), stream=stream0)
        del primals_284
        buf1118 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_287, buf1118, 1, grid=grid(1), stream=stream0)
        del primals_287
        buf1119 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_290, buf1119, 1, grid=grid(1), stream=stream0)
        del primals_290
        buf1120 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_293, buf1120, 1, grid=grid(1), stream=stream0)
        del primals_293
        buf1121 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_296, buf1121, 1, grid=grid(1), stream=stream0)
        del primals_296
        buf1122 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_299, buf1122, 1, grid=grid(1), stream=stream0)
        del primals_299
        buf1123 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_302, buf1123, 1, grid=grid(1), stream=stream0)
        del primals_302
        buf1124 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_305, buf1124, 1, grid=grid(1), stream=stream0)
        del primals_305
        buf1125 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_308, buf1125, 1, grid=grid(1), stream=stream0)
        del primals_308
        buf1126 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_311, buf1126, 1, grid=grid(1), stream=stream0)
        del primals_311
        buf1127 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_314, buf1127, 1, grid=grid(1), stream=stream0)
        del primals_314
        buf1128 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_317, buf1128, 1, grid=grid(1), stream=stream0)
        del primals_317
        buf1129 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_320, buf1129, 1, grid=grid(1), stream=stream0)
        del primals_320
        buf1130 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_323, buf1130, 1, grid=grid(1), stream=stream0)
        del primals_323
        buf1131 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_326, buf1131, 1, grid=grid(1), stream=stream0)
        del primals_326
        buf1132 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_329, buf1132, 1, grid=grid(1), stream=stream0)
        del primals_329
        buf1133 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_332, buf1133, 1, grid=grid(1), stream=stream0)
        del primals_332
        buf1134 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_335, buf1134, 1, grid=grid(1), stream=stream0)
        del primals_335
        buf1135 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_338, buf1135, 1, grid=grid(1), stream=stream0)
        del primals_338
        buf1136 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_341, buf1136, 1, grid=grid(1), stream=stream0)
        del primals_341
        buf1137 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_344, buf1137, 1, grid=grid(1), stream=stream0)
        del primals_344
        buf1138 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_347, buf1138, 1, grid=grid(1), stream=stream0)
        del primals_347
        buf1139 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_350, buf1139, 1, grid=grid(1), stream=stream0)
        del primals_350
        buf1140 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_353, buf1140, 1, grid=grid(1), stream=stream0)
        del primals_353
        buf1141 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_356, buf1141, 1, grid=grid(1), stream=stream0)
        del primals_356
        buf1142 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_359, buf1142, 1, grid=grid(1), stream=stream0)
        del primals_359
        buf1143 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_362, buf1143, 1, grid=grid(1), stream=stream0)
        del primals_362
        buf1144 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_365, buf1144, 1, grid=grid(1), stream=stream0)
        del primals_365
        buf1145 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_368, buf1145, 1, grid=grid(1), stream=stream0)
        del primals_368
        buf1146 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_371, buf1146, 1, grid=grid(1), stream=stream0)
        del primals_371
        buf1147 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_374, buf1147, 1, grid=grid(1), stream=stream0)
        del primals_374
        buf1148 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_377, buf1148, 1, grid=grid(1), stream=stream0)
        del primals_377
        buf1149 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_380, buf1149, 1, grid=grid(1), stream=stream0)
        del primals_380
        buf1150 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_383, buf1150, 1, grid=grid(1), stream=stream0)
        del primals_383
        buf1151 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_386, buf1151, 1, grid=grid(1), stream=stream0)
        del primals_386
        buf1152 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_389, buf1152, 1, grid=grid(1), stream=stream0)
        del primals_389
        buf1153 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_392, buf1153, 1, grid=grid(1), stream=stream0)
        del primals_392
        buf1154 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_395, buf1154, 1, grid=grid(1), stream=stream0)
        del primals_395
        buf1155 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_398, buf1155, 1, grid=grid(1), stream=stream0)
        del primals_398
        buf1156 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_401, buf1156, 1, grid=grid(1), stream=stream0)
        del primals_401
        buf1157 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_404, buf1157, 1, grid=grid(1), stream=stream0)
        del primals_404
        buf1158 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_407, buf1158, 1, grid=grid(1), stream=stream0)
        del primals_407
        buf1159 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_410, buf1159, 1, grid=grid(1), stream=stream0)
        del primals_410
        buf1160 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_413, buf1160, 1, grid=grid(1), stream=stream0)
        del primals_413
        buf1161 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_416, buf1161, 1, grid=grid(1), stream=stream0)
        del primals_416
        buf1162 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_419, buf1162, 1, grid=grid(1), stream=stream0)
        del primals_419
        buf1163 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_422, buf1163, 1, grid=grid(1), stream=stream0)
        del primals_422
        buf1164 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_425, buf1164, 1, grid=grid(1), stream=stream0)
        del primals_425
        buf1165 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_428, buf1165, 1, grid=grid(1), stream=stream0)
        del primals_428
        buf1166 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_431, buf1166, 1, grid=grid(1), stream=stream0)
        del primals_431
        buf1167 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_434, buf1167, 1, grid=grid(1), stream=stream0)
        del primals_434
        buf1168 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_437, buf1168, 1, grid=grid(1), stream=stream0)
        del primals_437
        buf1169 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_440, buf1169, 1, grid=grid(1), stream=stream0)
        del primals_440
        buf1170 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_443, buf1170, 1, grid=grid(1), stream=stream0)
        del primals_443
        buf1171 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_446, buf1171, 1, grid=grid(1), stream=stream0)
        del primals_446
        buf1172 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_449, buf1172, 1, grid=grid(1), stream=stream0)
        del primals_449
        buf1173 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_452, buf1173, 1, grid=grid(1), stream=stream0)
        del primals_452
        buf1174 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_455, buf1174, 1, grid=grid(1), stream=stream0)
        del primals_455
        buf1175 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_458, buf1175, 1, grid=grid(1), stream=stream0)
        del primals_458
        buf1176 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_461, buf1176, 1, grid=grid(1), stream=stream0)
        del primals_461
        buf1177 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_464, buf1177, 1, grid=grid(1), stream=stream0)
        del primals_464
        buf1178 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_467, buf1178, 1, grid=grid(1), stream=stream0)
        del primals_467
        buf1179 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_470, buf1179, 1, grid=grid(1), stream=stream0)
        del primals_470
        buf1180 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_473, buf1180, 1, grid=grid(1), stream=stream0)
        del primals_473
        buf1181 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_476, buf1181, 1, grid=grid(1), stream=stream0)
        del primals_476
        buf1182 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_479, buf1182, 1, grid=grid(1), stream=stream0)
        del primals_479
        buf1183 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_482, buf1183, 1, grid=grid(1), stream=stream0)
        del primals_482
        buf1184 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_485, buf1184, 1, grid=grid(1), stream=stream0)
        del primals_485
        buf1185 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_488, buf1185, 1, grid=grid(1), stream=stream0)
        del primals_488
        buf1186 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_491, buf1186, 1, grid=grid(1), stream=stream0)
        del primals_491
        buf1187 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_494, buf1187, 1, grid=grid(1), stream=stream0)
        del primals_494
        buf1188 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_497, buf1188, 1, grid=grid(1), stream=stream0)
        del primals_497
        buf1189 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_500, buf1189, 1, grid=grid(1), stream=stream0)
        del primals_500
        buf1190 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_503, buf1190, 1, grid=grid(1), stream=stream0)
        del primals_503
        buf1191 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_506, buf1191, 1, grid=grid(1), stream=stream0)
        del primals_506
        buf1192 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_509, buf1192, 1, grid=grid(1), stream=stream0)
        del primals_509
        buf1193 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__87.run(primals_512, buf1193, 1, grid=grid(1), stream=stream0)
        del primals_512
        return (buf10, buf11, buf1109, buf24, buf25, buf1110, buf36, buf37, buf1111, buf48, buf49, buf1112, buf60, buf61, buf1113, buf71, buf72, buf1114, buf79, buf80, buf1115, buf92, buf93, buf1116, buf104, buf105, buf1117, buf117, buf118, buf1118, buf130, buf131, buf1119, buf141, buf142, buf1120, buf153, buf154, buf1121, buf165, buf166, buf1122, buf178, buf179, buf1123, buf191, buf192, buf1124, buf202, buf203, buf1125, buf211, buf212, buf1126, buf223, buf224, buf1127, buf235, buf236, buf1128, buf247, buf248, buf1129, buf258, buf259, buf1130, buf266, buf267, buf1131, buf276, buf277, buf1132, buf288, buf289, buf1133, buf301, buf302, buf1134, buf314, buf315, buf1135, buf325, buf326, buf1136, buf334, buf335, buf1137, buf346, buf347, buf1138, buf359, buf360, buf1139, buf372, buf373, buf1140, buf383, buf384, buf1141, buf392, buf393, buf1142, buf404, buf405, buf1143, buf417, buf418, buf1144, buf430, buf431, buf1145, buf441, buf442, buf1146, buf450, buf451, buf1147, buf462, buf463, buf1148, buf474, buf475, buf1149, buf486, buf487, buf1150, buf497, buf498, buf1151, buf505, buf506, buf1152, buf515, buf516, buf1153, buf527, buf528, buf1154, buf540, buf541, buf1155, buf553, buf554, buf1156, buf564, buf565, buf1157, buf573, buf574, buf1158, buf585, buf586, buf1159, buf598, buf599, buf1160, buf611, buf612, buf1161, buf622, buf623, buf1162, buf631, buf632, buf1163, buf643, buf644, buf1164, buf656, buf657, buf1165, buf669, buf670, buf1166, buf680, buf681, buf1167, buf689, buf690, buf1168, buf701, buf702, buf1169, buf714, buf715, buf1170, buf727, buf728, buf1171, buf738, buf739, buf1172, buf747, buf748, buf1173, buf759, buf760, buf1174, buf772, buf773, buf1175, buf785, buf786, buf1176, buf796, buf797, buf1177, buf805, buf806, buf1178, buf814, buf815, buf1179, buf823, buf824, buf1180, buf832, buf833, buf1181, buf843, buf844, buf1182, buf851, buf852, buf1183, buf861, buf862, buf1184, buf870, buf871, buf1185, buf880, buf881, buf1186, buf890, buf891, buf1187, buf901, buf902, buf1188, buf910, buf911, buf1189, buf919, buf920, buf1190, buf929, buf930, buf1191, buf939, buf940, buf1192, buf950, buf951, buf1193, buf957, primals_2, primals_5, primals_8, primals_11, primals_14, primals_17, primals_20, primals_23, primals_26, primals_29, primals_32, primals_35, primals_38, primals_41, primals_44, primals_47, primals_50, primals_53, primals_56, primals_59, primals_62, primals_65, primals_68, primals_71, primals_74, primals_77, primals_80, primals_83, primals_86, primals_89, primals_92, primals_95, primals_98, primals_101, primals_104, primals_107, primals_110, primals_113, primals_116, primals_119, primals_122, primals_125, primals_128, primals_131, primals_134, primals_137, primals_140, primals_143, primals_146, primals_149, primals_152, primals_155, primals_158, primals_161, primals_164, primals_167, primals_170, primals_173, primals_176, primals_179, primals_182, primals_185, primals_188, primals_191, primals_194, primals_197, primals_200, primals_203, primals_206, primals_209, primals_212, primals_215, primals_218, primals_221, primals_224, primals_227, primals_230, primals_233, primals_236, primals_239, primals_242, primals_245, primals_248, primals_251, primals_254, buf0, buf1, buf2, buf9, buf12, buf13, buf14, buf15, buf16, buf23, buf27, as_strided(buf26, (128, 32, 56, 56), (401408, 3136, 56, 1)), buf28, buf35, buf39, as_strided(buf26, (128, 32, 56, 56), (401408, 3136, 56, 1), 100352), buf40, buf47, buf51, as_strided(buf26, (128, 32, 56, 56), (401408, 3136, 56, 1), 200704), buf52, buf59, as_strided(buf26, (128, 32, 56, 56), (401408, 3136, 56, 1), 301056), buf64, buf65, buf66, buf70, buf73, buf74, buf78, buf82, buf83, buf84, buf91, buf95, as_strided(buf94, (128, 32, 56, 56), (401408, 3136, 56, 1)), buf96, buf103, buf107, buf108, buf109, buf116, buf120, buf121, buf122, buf129, buf134, buf135, buf136, buf140, buf143, buf144, buf145, buf152, buf156, as_strided(buf155, (128, 32, 56, 56), (401408, 3136, 56, 1)), buf157, buf164, buf168, buf169, buf170, buf177, buf181, buf182, buf183, buf190, buf195, buf196, buf197, buf201, buf204, buf205, buf206, buf210, buf214, as_strided(buf213, (128, 64, 56, 56), (802816, 3136, 56, 1)), buf215, buf222, buf226, as_strided(buf213, (128, 64, 56, 56), (802816, 3136, 56, 1), 200704), buf227, buf234, buf238, as_strided(buf213, (128, 64, 56, 56), (802816, 3136, 56, 1), 401408), buf239, buf246, as_strided(buf213, (128, 64, 56, 56), (802816, 3136, 56, 1), 602112), buf251, buf252, buf253, buf257, buf260, buf261, buf265, buf269, buf270, buf271, buf275, buf279, as_strided(buf278, (128, 64, 28, 28), (200704, 784, 28, 1)), buf280, buf287, buf291, buf292, buf293, buf300, buf304, buf305, buf306, buf313, buf318, buf319, buf320, buf324, buf327, buf328, buf329, buf333, buf337, as_strided(buf336, (128, 64, 28, 28), (200704, 784, 28, 1)), buf338, buf345, buf349, buf350, buf351, buf358, buf362, buf363, buf364, buf371, buf376, buf377, buf378, buf382, buf385, buf386, buf387, buf391, buf395, as_strided(buf394, (128, 64, 28, 28), (200704, 784, 28, 1)), buf396, buf403, buf407, buf408, buf409, buf416, buf420, buf421, buf422, buf429, buf434, buf435, buf436, buf440, buf443, buf444, buf445, buf449, buf453, as_strided(buf452, (128, 128, 28, 28), (401408, 784, 28, 1)), buf454, buf461, buf465, as_strided(buf452, (128, 128, 28, 28), (401408, 784, 28, 1), 100352), buf466, buf473, buf477, as_strided(buf452, (128, 128, 28, 28), (401408, 784, 28, 1), 200704), buf478, buf485, as_strided(buf452, (128, 128, 28, 28), (401408, 784, 28, 1), 301056), buf490, buf491, buf492, buf496, buf499, buf500, buf504, buf508, buf509, buf510, buf514, buf518, as_strided(buf517, (128, 128, 14, 14), (100352, 196, 14, 1)), buf519, buf526, buf530, buf531, buf532, buf539, buf543, buf544, buf545, buf552, buf557, buf558, buf559, buf563, buf566, buf567, buf568, buf572, buf576, as_strided(buf575, (128, 128, 14, 14), (100352, 196, 14, 1)), buf577, buf584, buf588, buf589, buf590, buf597, buf601, buf602, buf603, buf610, buf615, buf616, buf617, buf621, buf624, buf625, buf626, buf630, buf634, as_strided(buf633, (128, 128, 14, 14), (100352, 196, 14, 1)), buf635, buf642, buf646, buf647, buf648, buf655, buf659, buf660, buf661, buf668, buf673, buf674, buf675, buf679, buf682, buf683, buf684, buf688, buf692, as_strided(buf691, (128, 128, 14, 14), (100352, 196, 14, 1)), buf693, buf700, buf704, buf705, buf706, buf713, buf717, buf718, buf719, buf726, buf731, buf732, buf733, buf737, buf740, buf741, buf742, buf746, buf750, as_strided(buf749, (128, 128, 14, 14), (100352, 196, 14, 1)), buf751, buf758, buf762, buf763, buf764, buf771, buf775, buf776, buf777, buf784, buf789, buf790, buf791, buf795, buf798, buf799, buf800, buf804, buf808, as_strided(buf807, (128, 256, 14, 14), (200704, 196, 14, 1)), buf809, buf813, buf817, as_strided(buf807, (128, 256, 14, 14), (200704, 196, 14, 1), 50176), buf818, buf822, buf826, as_strided(buf807, (128, 256, 14, 14), (200704, 196, 14, 1), 100352), buf827, buf831, as_strided(buf807, (128, 256, 14, 14), (200704, 196, 14, 1), 150528), buf836, buf837, buf838, buf842, buf845, buf846, buf850, buf854, buf855, buf856, buf860, buf864, as_strided(buf863, (128, 256, 7, 7), (50176, 49, 7, 1)), buf865, buf869, buf873, buf874, buf875, buf879, buf883, buf884, buf885, buf889, buf894, buf895, buf896, buf900, buf903, buf904, buf905, buf909, buf913, as_strided(buf912, (128, 256, 7, 7), (50176, 49, 7, 1)), buf914, buf918, buf922, buf923, buf924, buf928, buf932, buf933, buf934, buf938, buf943, buf944, buf945, buf949, buf954, buf958, buf959, buf960, buf961, buf962, buf963, buf964, buf965, buf966, buf967, buf968, buf969, buf970, buf971, buf972, buf973, buf974, buf975, buf976, buf977, buf978, buf979, buf980, buf981, buf982, buf983, buf984, buf985, buf986, buf987, buf988, buf989, buf990, buf991, buf992, buf993, buf994, buf995, buf996, buf997, buf998, buf999, buf1000, buf1001, buf1002, buf1003, buf1004, buf1005, buf1006, buf1007, buf1008, buf1009, buf1010, buf1011, buf1012, buf1013, buf1014, buf1015, buf1016, buf1017, buf1018, buf1019, buf1020, buf1021, buf1022, buf1023, buf1024, buf1025, buf1026, buf1027, buf1028, buf1029, buf1030, buf1031, buf1032, buf1033, buf1034, buf1035, buf1036, buf1037, buf1038, buf1039, buf1040, buf1041, buf1042, buf1043, buf1044, buf1045, buf1046, buf1047, buf1048, buf1049, buf1050, buf1051, buf1052, buf1053, buf1054, buf1055, buf1056, buf1057, buf1058, buf1059, buf1060, buf1061, buf1062, buf1063, buf1064, buf1065, buf1066, buf1067, buf1068, buf1069, buf1070, buf1071, buf1072, buf1073, buf1074, buf1075, buf1076, buf1077, buf1078, buf1079, buf1080, buf1081, buf1082, buf1083, buf1084, buf1085, buf1086, buf1087, buf1088, buf1089, buf1090, buf1091, buf1092, buf1093, buf1094, buf1095, buf1096, buf1097, buf1098, buf1099, buf1100, buf1101, buf1102, buf1103, buf1104, buf1105, buf1106, buf1107, buf1108, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_261 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_264 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_267 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_270 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_273 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_276 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_279 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_282 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_285 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_288 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_294 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_297 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_300 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_303 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_306 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_309 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_312 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_315 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_318 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_321 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_324 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_327 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_330 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_333 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_336 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_339 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_342 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_345 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_348 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_351 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_354 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_357 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_360 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_363 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_366 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_369 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_372 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_375 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_378 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_381 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_384 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_387 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_390 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_393 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_396 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_399 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_402 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_405 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_408 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_411 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_414 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_417 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_420 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_423 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_426 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_429 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_432 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_435 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_438 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_441 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_444 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_447 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_450 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_453 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_456 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_459 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_462 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_465 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_468 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_471 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_474 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_477 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_480 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_483 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_486 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_489 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_492 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_495 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_498 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_501 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_504 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_507 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_510 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_513 = rand_strided((128, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513]))
