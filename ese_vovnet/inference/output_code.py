
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

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 64
    x2 = (xindex // 200704)
    x4 = xindex % 200704
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
    tl.store(out_ptr0 + (x4 + (1404928*x2) + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
    tl.store(out_ptr0 + (x4 + (1404928*x2) + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp11 = tl.load(in_ptr3 + (x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(out_ptr0 + (x4 + (1404928*x2) + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 4096],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__5(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 256
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp11 = tl.load(in_ptr2 + (x0), xmask)
    tmp13 = tl.load(in_ptr3 + (x0), xmask)
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (3136*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt(tmp5)
        tmp7 = 1 / tmp6
        tmp8 = 1.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp2 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
        _tmp16 = tl.where(rmask & xmask, _tmp16 + tmp15, _tmp16)
        tl.store(in_out_ptr0 + (r2 + (3136*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp15, rmask & xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp17 = 3136.0
    tmp18 = tmp16 / tmp17
    tl.store(in_out_ptr1 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__6(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 3136)
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x4), xmask)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5 != tmp5, tmp5, tl.where(tmp5 > tmp6, tmp5, tmp6))
    tmp8 = 6.0
    tmp9 = tl.where(tmp7 != tmp7, tmp7, tl.where(tmp7 < tmp8, tmp7, tmp8))
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x3 = (xindex // 28)
    x4 = xindex
    x5 = xindex % 200704
    x6 = (xindex // 200704)
    tmp0 = 2*x1
    tmp1 = 0
    tmp2 = tmp0 >= tmp1
    tmp3 = 56
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((2*x0) + (112*x3) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0)
    tmp12 = tl.where(tmp10, tmp11, float("-inf"))
    tmp13 = 1 + (2*x0)
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (1 + (2*x0) + (112*x3) + tl.zeros([XBLOCK], tl.int32)), tmp17 & xmask, other=0)
    tmp19 = tl.where(tmp17, tmp18, float("-inf"))
    tmp20 = tl.where(tmp19 != tmp19, tmp19, tl.where(tmp19 > tmp12, tmp19, tmp12))
    tmp21 = 2 + (2*x0)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + (2 + (2*x0) + (112*x3) + tl.zeros([XBLOCK], tl.int32)), tmp25 & xmask, other=0)
    tmp27 = tl.where(tmp25, tmp26, float("-inf"))
    tmp28 = tl.where(tmp27 != tmp27, tmp27, tl.where(tmp27 > tmp20, tmp27, tmp20))
    tmp29 = 1 + (2*x1)
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + (56 + (2*x0) + (112*x3) + tl.zeros([XBLOCK], tl.int32)), tmp33 & xmask, other=0)
    tmp35 = tl.where(tmp33, tmp34, float("-inf"))
    tmp36 = tl.where(tmp35 != tmp35, tmp35, tl.where(tmp35 > tmp28, tmp35, tmp28))
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (57 + (2*x0) + (112*x3) + tl.zeros([XBLOCK], tl.int32)), tmp37 & xmask, other=0)
    tmp39 = tl.where(tmp37, tmp38, float("-inf"))
    tmp40 = tl.where(tmp39 != tmp39, tmp39, tl.where(tmp39 > tmp36, tmp39, tmp36))
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (58 + (2*x0) + (112*x3) + tl.zeros([XBLOCK], tl.int32)), tmp41 & xmask, other=0)
    tmp43 = tl.where(tmp41, tmp42, float("-inf"))
    tmp44 = tl.where(tmp43 != tmp43, tmp43, tl.where(tmp43 > tmp40, tmp43, tmp40))
    tmp45 = 2 + (2*x1)
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (112 + (2*x0) + (112*x3) + tl.zeros([XBLOCK], tl.int32)), tmp49 & xmask, other=0)
    tmp51 = tl.where(tmp49, tmp50, float("-inf"))
    tmp52 = tl.where(tmp51 != tmp51, tmp51, tl.where(tmp51 > tmp44, tmp51, tmp44))
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (113 + (2*x0) + (112*x3) + tl.zeros([XBLOCK], tl.int32)), tmp53 & xmask, other=0)
    tmp55 = tl.where(tmp53, tmp54, float("-inf"))
    tmp56 = tl.where(tmp55 != tmp55, tmp55, tl.where(tmp55 > tmp52, tmp55, tmp52))
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (114 + (2*x0) + (112*x3) + tl.zeros([XBLOCK], tl.int32)), tmp57 & xmask, other=0)
    tmp59 = tl.where(tmp57, tmp58, float("-inf"))
    tmp60 = tl.where(tmp59 != tmp59, tmp59, tl.where(tmp59 > tmp56, tmp59, tmp56))
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp60, xmask)
    tl.store(out_ptr1 + (x5 + (577024*x6) + tl.zeros([XBLOCK], tl.int32)), tmp60, xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 160
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 160
    x2 = (xindex // 125440)
    x4 = xindex % 125440
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
    tl.store(out_ptr0 + (x4 + (577024*x2) + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 160
    x2 = (xindex // 125440)
    x4 = xindex % 125440
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp11 = tl.load(in_ptr3 + (x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(out_ptr0 + (x4 + (577024*x2) + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[4096, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 512
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp11 = tl.load(in_ptr2 + (x0), xmask)
    tmp13 = tl.load(in_ptr3 + (x0), xmask)
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (784*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt(tmp5)
        tmp7 = 1 / tmp6
        tmp8 = 1.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp2 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
        _tmp16 = tl.where(rmask & xmask, _tmp16 + tmp15, _tmp16)
        tl.store(in_out_ptr0 + (r2 + (784*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp15, rmask & xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp17 = 784.0
    tmp18 = tmp16 / tmp17
    tl.store(in_out_ptr1 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
''')


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__12(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x4), xmask)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5 != tmp5, tmp5, tl.where(tmp5 > tmp6, tmp5, tmp6))
    tmp8 = 6.0
    tmp9 = tl.where(tmp7 != tmp7, tmp7, tl.where(tmp7 < tmp8, tmp7, tmp8))
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__13(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x3 = (xindex // 14)
    x4 = xindex
    x5 = xindex % 100352
    x6 = (xindex // 100352)
    tmp0 = 2*x1
    tmp1 = 0
    tmp2 = tmp0 >= tmp1
    tmp3 = 28
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((2*x0) + (56*x3) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0)
    tmp12 = tl.where(tmp10, tmp11, float("-inf"))
    tmp13 = 1 + (2*x0)
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x3) + tl.zeros([XBLOCK], tl.int32)), tmp17 & xmask, other=0)
    tmp19 = tl.where(tmp17, tmp18, float("-inf"))
    tmp20 = tl.where(tmp19 != tmp19, tmp19, tl.where(tmp19 > tmp12, tmp19, tmp12))
    tmp21 = 2 + (2*x0)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + (2 + (2*x0) + (56*x3) + tl.zeros([XBLOCK], tl.int32)), tmp25 & xmask, other=0)
    tmp27 = tl.where(tmp25, tmp26, float("-inf"))
    tmp28 = tl.where(tmp27 != tmp27, tmp27, tl.where(tmp27 > tmp20, tmp27, tmp20))
    tmp29 = 1 + (2*x1)
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x3) + tl.zeros([XBLOCK], tl.int32)), tmp33 & xmask, other=0)
    tmp35 = tl.where(tmp33, tmp34, float("-inf"))
    tmp36 = tl.where(tmp35 != tmp35, tmp35, tl.where(tmp35 > tmp28, tmp35, tmp28))
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x3) + tl.zeros([XBLOCK], tl.int32)), tmp37 & xmask, other=0)
    tmp39 = tl.where(tmp37, tmp38, float("-inf"))
    tmp40 = tl.where(tmp39 != tmp39, tmp39, tl.where(tmp39 > tmp36, tmp39, tmp36))
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (30 + (2*x0) + (56*x3) + tl.zeros([XBLOCK], tl.int32)), tmp41 & xmask, other=0)
    tmp43 = tl.where(tmp41, tmp42, float("-inf"))
    tmp44 = tl.where(tmp43 != tmp43, tmp43, tl.where(tmp43 > tmp40, tmp43, tmp40))
    tmp45 = 2 + (2*x1)
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (56 + (2*x0) + (56*x3) + tl.zeros([XBLOCK], tl.int32)), tmp49 & xmask, other=0)
    tmp51 = tl.where(tmp49, tmp50, float("-inf"))
    tmp52 = tl.where(tmp51 != tmp51, tmp51, tl.where(tmp51 > tmp44, tmp51, tmp44))
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (57 + (2*x0) + (56*x3) + tl.zeros([XBLOCK], tl.int32)), tmp53 & xmask, other=0)
    tmp55 = tl.where(tmp53, tmp54, float("-inf"))
    tmp56 = tl.where(tmp55 != tmp55, tmp55, tl.where(tmp55 > tmp52, tmp55, tmp52))
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (58 + (2*x0) + (56*x3) + tl.zeros([XBLOCK], tl.int32)), tmp57 & xmask, other=0)
    tmp59 = tl.where(tmp57, tmp58, float("-inf"))
    tmp60 = tl.where(tmp59 != tmp59, tmp59, tl.where(tmp59 > tmp56, tmp59, tmp56))
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp60, xmask)
    tl.store(out_ptr1 + (x5 + (213248*x6) + tl.zeros([XBLOCK], tl.int32)), tmp60, xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 192
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
    tl.store(out_ptr0 + (x4 + (213248*x2) + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 192
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp11 = tl.load(in_ptr3 + (x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(out_ptr0 + (x4 + (213248*x2) + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 256],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__17(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 768
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp11 = tl.load(in_ptr2 + (x0), xmask)
    tmp13 = tl.load(in_ptr3 + (x0), xmask)
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt(tmp5)
        tmp7 = 1 / tmp6
        tmp8 = 1.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp2 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
        _tmp16 = tl.where(rmask & xmask, _tmp16 + tmp15, _tmp16)
        tl.store(in_out_ptr0 + (r2 + (196*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp15, rmask & xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp17 = 196.0
    tmp18 = tmp16 / tmp17
    tl.store(in_out_ptr1 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
''')


triton__18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__18(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x4), xmask)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5 != tmp5, tmp5, tl.where(tmp5 > tmp6, tmp5, tmp6))
    tmp8 = 6.0
    tmp9 = tl.where(tmp7 != tmp7, tmp7, tl.where(tmp7 < tmp8, tmp7, tmp8))
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__19(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 7) % 7
    x0 = xindex % 7
    x3 = (xindex // 7)
    x4 = xindex
    x5 = xindex % 37632
    x6 = (xindex // 37632)
    tmp0 = 2*x1
    tmp1 = 0
    tmp2 = tmp0 >= tmp1
    tmp3 = 14
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((2*x0) + (28*x3) + tl.zeros([XBLOCK], tl.int32)), tmp10 & xmask, other=0)
    tmp12 = tl.where(tmp10, tmp11, float("-inf"))
    tmp13 = 1 + (2*x0)
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x3) + tl.zeros([XBLOCK], tl.int32)), tmp17 & xmask, other=0)
    tmp19 = tl.where(tmp17, tmp18, float("-inf"))
    tmp20 = tl.where(tmp19 != tmp19, tmp19, tl.where(tmp19 > tmp12, tmp19, tmp12))
    tmp21 = 2 + (2*x0)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + (2 + (2*x0) + (28*x3) + tl.zeros([XBLOCK], tl.int32)), tmp25 & xmask, other=0)
    tmp27 = tl.where(tmp25, tmp26, float("-inf"))
    tmp28 = tl.where(tmp27 != tmp27, tmp27, tl.where(tmp27 > tmp20, tmp27, tmp20))
    tmp29 = 1 + (2*x1)
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x3) + tl.zeros([XBLOCK], tl.int32)), tmp33 & xmask, other=0)
    tmp35 = tl.where(tmp33, tmp34, float("-inf"))
    tmp36 = tl.where(tmp35 != tmp35, tmp35, tl.where(tmp35 > tmp28, tmp35, tmp28))
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x3) + tl.zeros([XBLOCK], tl.int32)), tmp37 & xmask, other=0)
    tmp39 = tl.where(tmp37, tmp38, float("-inf"))
    tmp40 = tl.where(tmp39 != tmp39, tmp39, tl.where(tmp39 > tmp36, tmp39, tmp36))
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (16 + (2*x0) + (28*x3) + tl.zeros([XBLOCK], tl.int32)), tmp41 & xmask, other=0)
    tmp43 = tl.where(tmp41, tmp42, float("-inf"))
    tmp44 = tl.where(tmp43 != tmp43, tmp43, tl.where(tmp43 > tmp40, tmp43, tmp40))
    tmp45 = 2 + (2*x1)
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (28 + (2*x0) + (28*x3) + tl.zeros([XBLOCK], tl.int32)), tmp49 & xmask, other=0)
    tmp51 = tl.where(tmp49, tmp50, float("-inf"))
    tmp52 = tl.where(tmp51 != tmp51, tmp51, tl.where(tmp51 > tmp44, tmp51, tmp44))
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (29 + (2*x0) + (28*x3) + tl.zeros([XBLOCK], tl.int32)), tmp53 & xmask, other=0)
    tmp55 = tl.where(tmp53, tmp54, float("-inf"))
    tmp56 = tl.where(tmp55 != tmp55, tmp55, tl.where(tmp55 > tmp52, tmp55, tmp52))
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (30 + (2*x0) + (28*x3) + tl.zeros([XBLOCK], tl.int32)), tmp57 & xmask, other=0)
    tmp59 = tl.where(tmp57, tmp58, float("-inf"))
    tmp60 = tl.where(tmp59 != tmp59, tmp59, tl.where(tmp59 > tmp56, tmp59, tmp56))
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp60, xmask)
    tl.store(out_ptr1 + (x5 + (70560*x6) + tl.zeros([XBLOCK], tl.int32)), tmp60, xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 224
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask)
    tmp11 = tl.load(in_ptr2 + (x1), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
    tl.store(out_ptr0 + (x4 + (70560*x2) + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 224
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp11 = tl.load(in_ptr3 + (x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
    tl.store(out_ptr0 + (x4 + (70560*x2) + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 64],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__23(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp11 = tl.load(in_ptr2 + (x0), xmask)
    tmp13 = tl.load(in_ptr3 + (x0), xmask)
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt(tmp5)
        tmp7 = 1 / tmp6
        tmp8 = 1.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp2 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tl.where(0 != 0, 0, tl.where(0 > tmp14, 0, tmp14))
        _tmp16 = tl.where(rmask & xmask, _tmp16 + tmp15, _tmp16)
        tl.store(in_out_ptr0 + (r2 + (49*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp15, rmask & xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp17 = 49.0
    tmp18 = tmp16 / tmp17
    tl.store(in_out_ptr1 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
''')


triton__24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 64],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    x0 = xindex % 1024
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tmp1 + tmp2
        tmp4 = 3.0
        tmp5 = tmp3 + tmp4
        tmp6 = 0.0
        tmp7 = tl.where(tmp5 != tmp5, tmp5, tl.where(tmp5 > tmp6, tmp5, tmp6))
        tmp8 = 6.0
        tmp9 = tl.where(tmp7 != tmp7, tmp7, tl.where(tmp7 < tmp8, tmp7, tmp8))
        tmp10 = tmp9 / tmp8
        tmp11 = tmp0 * tmp10
        _tmp12 = tl.where(rmask & xmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp13 = 49.0
    tmp14 = tmp12 / tmp13
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = aten.convolution(arg139_1, arg46_1, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf0, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg139_1
        del arg46_1
        buf1 = buf0; del buf0  # reuse
        print('triton__0', 'in_out_ptr0', 'buf1', (buf1.sum()/buf1.nelement()).item(), buf1.amax().item(), buf1.amin().item())
        print('triton__0', 'in_ptr0', 'arg93_1', (arg93_1.sum()/arg93_1.nelement()).item(), arg93_1.amax().item(), arg93_1.amin().item())
        print('triton__0', 'in_ptr1', 'arg94_1', (arg94_1.sum()/arg94_1.nelement()).item(), arg94_1.amax().item(), arg94_1.amin().item())
        print('triton__0', 'in_ptr2', 'arg0_1', (arg0_1.sum()/arg0_1.nelement()).item(), arg0_1.amax().item(), arg0_1.amin().item())
        print('triton__0', 'in_ptr3', 'arg1_1', (arg1_1.sum()/arg1_1.nelement()).item(), arg1_1.amax().item(), arg1_1.amin().item())
        stream0 = get_cuda_stream(0)
        triton__0.run(buf1, arg93_1, arg94_1, arg0_1, arg1_1, 6422528, grid=grid(6422528), stream=stream0)
        print('triton__0', 'in_out_ptr0', 'buf1', (buf1.sum()/buf1.nelement()).item(), buf1.amax().item(), buf1.amin().item())
        del arg0_1
        del arg1_1
        del arg93_1
        del arg94_1
        buf2 = aten.convolution(buf1, arg47_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 64)
        assert_size_stride(buf2, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg47_1
        del buf1
        buf3 = aten.convolution(buf2, arg48_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf3, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg48_1
        del buf2
        buf4 = buf3; del buf3  # reuse
        print('triton__0', 'in_out_ptr0', 'buf4', (buf4.sum()/buf4.nelement()).item(), buf4.amax().item(), buf4.amin().item())
        print('triton__0', 'in_ptr0', 'arg95_1', (arg95_1.sum()/arg95_1.nelement()).item(), arg95_1.amax().item(), arg95_1.amin().item())
        print('triton__0', 'in_ptr1', 'arg96_1', (arg96_1.sum()/arg96_1.nelement()).item(), arg96_1.amax().item(), arg96_1.amin().item())
        print('triton__0', 'in_ptr2', 'arg2_1', (arg2_1.sum()/arg2_1.nelement()).item(), arg2_1.amax().item(), arg2_1.amin().item())
        print('triton__0', 'in_ptr3', 'arg3_1', (arg3_1.sum()/arg3_1.nelement()).item(), arg3_1.amax().item(), arg3_1.amin().item())
        triton__0.run(buf4, arg95_1, arg96_1, arg2_1, arg3_1, 6422528, grid=grid(6422528), stream=stream0)
        print('triton__0', 'in_out_ptr0', 'buf4', (buf4.sum()/buf4.nelement()).item(), buf4.amax().item(), buf4.amin().item())
        del arg2_1
        del arg3_1
        del arg95_1
        del arg96_1
        buf5 = aten.convolution(buf4, arg49_1, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 64)
        assert_size_stride(buf5, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg49_1
        del buf4
        buf6 = aten.convolution(buf5, arg50_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf6, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg50_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        buf22 = empty_strided((8, 448, 56, 56), (1404928, 3136, 56, 1), device='cuda', dtype=torch.float32)
        buf19 = as_strided(buf22, (8, 64, 56, 56), (1404928, 3136, 56, 1))  # alias
        print('triton__1', 'in_out_ptr0', 'buf7', (buf7.sum()/buf7.nelement()).item(), buf7.amax().item(), buf7.amin().item())
        print('triton__1', 'in_ptr0', 'arg97_1', (arg97_1.sum()/arg97_1.nelement()).item(), arg97_1.amax().item(), arg97_1.amin().item())
        print('triton__1', 'in_ptr1', 'arg98_1', (arg98_1.sum()/arg98_1.nelement()).item(), arg98_1.amax().item(), arg98_1.amin().item())
        print('triton__1', 'in_ptr2', 'arg4_1', (arg4_1.sum()/arg4_1.nelement()).item(), arg4_1.amax().item(), arg4_1.amin().item())
        print('triton__1', 'in_ptr3', 'arg5_1', (arg5_1.sum()/arg5_1.nelement()).item(), arg5_1.amax().item(), arg5_1.amin().item())
        triton__1.run(buf7, arg97_1, arg98_1, arg4_1, arg5_1, buf19, 1605632, grid=grid(1605632), stream=stream0)
        print('triton__1', 'in_out_ptr0', 'buf7', (buf7.sum()/buf7.nelement()).item(), buf7.amax().item(), buf7.amin().item())
        print('triton__1', 'out_ptr0', 'buf19', (buf19.sum()/buf19.nelement()).item(), buf19.amax().item(), buf19.amin().item())
        del arg4_1
        del arg5_1
        del arg97_1
        del arg98_1
        buf8 = aten.convolution(buf7, arg51_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf8, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg51_1
        buf9 = buf8; del buf8  # reuse
        print('triton__2', 'in_out_ptr0', 'buf9', (buf9.sum()/buf9.nelement()).item(), buf9.amax().item(), buf9.amin().item())
        print('triton__2', 'in_ptr0', 'arg99_1', (arg99_1.sum()/arg99_1.nelement()).item(), arg99_1.amax().item(), arg99_1.amin().item())
        print('triton__2', 'in_ptr1', 'arg100_1', (arg100_1.sum()/arg100_1.nelement()).item(), arg100_1.amax().item(), arg100_1.amin().item())
        print('triton__2', 'in_ptr2', 'arg6_1', (arg6_1.sum()/arg6_1.nelement()).item(), arg6_1.amax().item(), arg6_1.amin().item())
        print('triton__2', 'in_ptr3', 'arg7_1', (arg7_1.sum()/arg7_1.nelement()).item(), arg7_1.amax().item(), arg7_1.amin().item())
        triton__2.run(buf9, arg99_1, arg100_1, arg6_1, arg7_1, 3211264, grid=grid(3211264), stream=stream0)
        print('triton__2', 'in_out_ptr0', 'buf9', (buf9.sum()/buf9.nelement()).item(), buf9.amax().item(), buf9.amin().item())
        del arg100_1
        del arg6_1
        del arg7_1
        del arg99_1
        buf10 = aten.convolution(buf9, arg52_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 128)
        assert_size_stride(buf10, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg52_1
        del buf9
        buf11 = aten.convolution(buf10, arg53_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf11, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg53_1
        del buf10
        buf12 = buf11; del buf11  # reuse
        buf20 = as_strided(buf22, (8, 128, 56, 56), (1404928, 3136, 56, 1), 200704)  # alias
        print('triton__3', 'in_out_ptr0', 'buf12', (buf12.sum()/buf12.nelement()).item(), buf12.amax().item(), buf12.amin().item())
        print('triton__3', 'in_ptr0', 'arg101_1', (arg101_1.sum()/arg101_1.nelement()).item(), arg101_1.amax().item(), arg101_1.amin().item())
        print('triton__3', 'in_ptr1', 'arg102_1', (arg102_1.sum()/arg102_1.nelement()).item(), arg102_1.amax().item(), arg102_1.amin().item())
        print('triton__3', 'in_ptr2', 'arg8_1', (arg8_1.sum()/arg8_1.nelement()).item(), arg8_1.amax().item(), arg8_1.amin().item())
        print('triton__3', 'in_ptr3', 'arg9_1', (arg9_1.sum()/arg9_1.nelement()).item(), arg9_1.amax().item(), arg9_1.amin().item())
        triton__3.run(buf12, arg101_1, arg102_1, arg8_1, arg9_1, buf20, 3211264, grid=grid(3211264), stream=stream0)
        print('triton__3', 'in_out_ptr0', 'buf12', (buf12.sum()/buf12.nelement()).item(), buf12.amax().item(), buf12.amin().item())
        print('triton__3', 'out_ptr0', 'buf20', (buf20.sum()/buf20.nelement()).item(), buf20.amax().item(), buf20.amin().item())
        del arg101_1
        del arg102_1
        del arg8_1
        del arg9_1
        buf13 = aten.convolution(buf12, arg54_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 128)
        assert_size_stride(buf13, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg54_1
        del buf12
        buf14 = aten.convolution(buf13, arg55_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf14, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg55_1
        del buf13
        buf15 = buf14; del buf14  # reuse
        buf21 = as_strided(buf22, (8, 128, 56, 56), (1404928, 3136, 56, 1), 602112)  # alias
        print('triton__3', 'in_out_ptr0', 'buf15', (buf15.sum()/buf15.nelement()).item(), buf15.amax().item(), buf15.amin().item())
        print('triton__3', 'in_ptr0', 'arg103_1', (arg103_1.sum()/arg103_1.nelement()).item(), arg103_1.amax().item(), arg103_1.amin().item())
        print('triton__3', 'in_ptr1', 'arg104_1', (arg104_1.sum()/arg104_1.nelement()).item(), arg104_1.amax().item(), arg104_1.amin().item())
        print('triton__3', 'in_ptr2', 'arg10_1', (arg10_1.sum()/arg10_1.nelement()).item(), arg10_1.amax().item(), arg10_1.amin().item())
        print('triton__3', 'in_ptr3', 'arg11_1', (arg11_1.sum()/arg11_1.nelement()).item(), arg11_1.amax().item(), arg11_1.amin().item())
        triton__3.run(buf15, arg103_1, arg104_1, arg10_1, arg11_1, buf21, 3211264, grid=grid(3211264), stream=stream0)
        print('triton__3', 'in_out_ptr0', 'buf15', (buf15.sum()/buf15.nelement()).item(), buf15.amax().item(), buf15.amin().item())
        print('triton__3', 'out_ptr0', 'buf21', (buf21.sum()/buf21.nelement()).item(), buf21.amax().item(), buf21.amin().item())
        del arg103_1
        del arg104_1
        del arg10_1
        del arg11_1
        buf16 = aten.convolution(buf15, arg56_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 128)
        assert_size_stride(buf16, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg56_1
        del buf15
        buf17 = aten.convolution(buf16, arg57_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf17, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg57_1
        del buf16
        buf18 = as_strided(buf22, (8, 128, 56, 56), (1404928, 3136, 56, 1), 1003520)  # alias
        print('triton__4', 'in_ptr0', 'buf17', (buf17.sum()/buf17.nelement()).item(), buf17.amax().item(), buf17.amin().item())
        print('triton__4', 'in_ptr1', 'arg105_1', (arg105_1.sum()/arg105_1.nelement()).item(), arg105_1.amax().item(), arg105_1.amin().item())
        print('triton__4', 'in_ptr2', 'arg106_1', (arg106_1.sum()/arg106_1.nelement()).item(), arg106_1.amax().item(), arg106_1.amin().item())
        print('triton__4', 'in_ptr3', 'arg12_1', (arg12_1.sum()/arg12_1.nelement()).item(), arg12_1.amax().item(), arg12_1.amin().item())
        print('triton__4', 'in_ptr4', 'arg13_1', (arg13_1.sum()/arg13_1.nelement()).item(), arg13_1.amax().item(), arg13_1.amin().item())
        triton__4.run(buf17, arg105_1, arg106_1, arg12_1, arg13_1, buf18, 3211264, grid=grid(3211264), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf18', (buf18.sum()/buf18.nelement()).item(), buf18.amax().item(), buf18.amin().item())
        del arg105_1
        del arg106_1
        del arg12_1
        del arg13_1
        del buf17
        del buf18
        del buf19
        del buf20
        del buf21
        buf23 = aten.convolution(buf22, arg58_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf23, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg58_1
        del buf22
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf26 = as_strided(buf25, (8, 256, 1, 1), (256, 1, 1, 1)); del buf25  # reuse
        print('triton__5', 'in_out_ptr0', 'buf24', (buf24.sum()/buf24.nelement()).item(), buf24.amax().item(), buf24.amin().item())
        print('triton__5', 'in_out_ptr1', 'buf26', (buf26.sum()/buf26.nelement()).item(), buf26.amax().item(), buf26.amin().item())
        print('triton__5', 'in_ptr0', 'arg107_1', (arg107_1.sum()/arg107_1.nelement()).item(), arg107_1.amax().item(), arg107_1.amin().item())
        print('triton__5', 'in_ptr1', 'arg108_1', (arg108_1.sum()/arg108_1.nelement()).item(), arg108_1.amax().item(), arg108_1.amin().item())
        print('triton__5', 'in_ptr2', 'arg14_1', (arg14_1.sum()/arg14_1.nelement()).item(), arg14_1.amax().item(), arg14_1.amin().item())
        print('triton__5', 'in_ptr3', 'arg15_1', (arg15_1.sum()/arg15_1.nelement()).item(), arg15_1.amax().item(), arg15_1.amin().item())
        triton__5.run(buf24, buf26, arg107_1, arg108_1, arg14_1, arg15_1, 2048, 3136, grid=grid(2048), stream=stream0)
        print('triton__5', 'in_out_ptr0', 'buf24', (buf24.sum()/buf24.nelement()).item(), buf24.amax().item(), buf24.amin().item())
        print('triton__5', 'in_out_ptr1', 'buf26', (buf26.sum()/buf26.nelement()).item(), buf26.amax().item(), buf26.amin().item())
        del arg107_1
        del arg108_1
        del arg14_1
        del arg15_1
        buf27 = aten.convolution(buf26, arg59_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf27, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg59_1
        del buf26
        buf28 = buf24; del buf24  # reuse
        print('triton__6', 'in_out_ptr0', 'buf28', (buf28.sum()/buf28.nelement()).item(), buf28.amax().item(), buf28.amin().item())
        print('triton__6', 'in_ptr0', 'buf27', (buf27.sum()/buf27.nelement()).item(), buf27.amax().item(), buf27.amin().item())
        print('triton__6', 'in_ptr1', 'arg60_1', (arg60_1.sum()/arg60_1.nelement()).item(), arg60_1.amax().item(), arg60_1.amin().item())
        triton__6.run(buf28, buf27, arg60_1, 6422528, grid=grid(6422528), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf28', (buf28.sum()/buf28.nelement()).item(), buf28.amax().item(), buf28.amin().item())
        del arg60_1
        del buf27
        buf29 = as_strided(buf7, (8, 256, 28, 28), (200704, 784, 28, 1)); del buf7  # reuse
        buf45 = empty_strided((8, 736, 28, 28), (577024, 784, 28, 1), device='cuda', dtype=torch.float32)
        buf42 = as_strided(buf45, (8, 256, 28, 28), (577024, 784, 28, 1))  # alias
        print('triton__7', 'in_ptr0', 'buf28', (buf28.sum()/buf28.nelement()).item(), buf28.amax().item(), buf28.amin().item())
        triton__7.run(buf28, buf29, buf42, 1605632, grid=grid(1605632), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf29', (buf29.sum()/buf29.nelement()).item(), buf29.amax().item(), buf29.amin().item())
        print('triton__7', 'out_ptr1', 'buf42', (buf42.sum()/buf42.nelement()).item(), buf42.amax().item(), buf42.amin().item())
        del buf28
        buf31 = aten.convolution(buf29, arg61_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf31, (8, 160, 28, 28), (125440, 784, 28, 1))
        del arg61_1
        del buf29
        buf32 = buf31; del buf31  # reuse
        print('triton__8', 'in_out_ptr0', 'buf32', (buf32.sum()/buf32.nelement()).item(), buf32.amax().item(), buf32.amin().item())
        print('triton__8', 'in_ptr0', 'arg109_1', (arg109_1.sum()/arg109_1.nelement()).item(), arg109_1.amax().item(), arg109_1.amin().item())
        print('triton__8', 'in_ptr1', 'arg110_1', (arg110_1.sum()/arg110_1.nelement()).item(), arg110_1.amax().item(), arg110_1.amin().item())
        print('triton__8', 'in_ptr2', 'arg16_1', (arg16_1.sum()/arg16_1.nelement()).item(), arg16_1.amax().item(), arg16_1.amin().item())
        print('triton__8', 'in_ptr3', 'arg17_1', (arg17_1.sum()/arg17_1.nelement()).item(), arg17_1.amax().item(), arg17_1.amin().item())
        triton__8.run(buf32, arg109_1, arg110_1, arg16_1, arg17_1, 1003520, grid=grid(1003520), stream=stream0)
        print('triton__8', 'in_out_ptr0', 'buf32', (buf32.sum()/buf32.nelement()).item(), buf32.amax().item(), buf32.amin().item())
        del arg109_1
        del arg110_1
        del arg16_1
        del arg17_1
        buf33 = aten.convolution(buf32, arg62_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 160)
        assert_size_stride(buf33, (8, 160, 28, 28), (125440, 784, 28, 1))
        del arg62_1
        del buf32
        buf34 = aten.convolution(buf33, arg63_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf34, (8, 160, 28, 28), (125440, 784, 28, 1))
        del arg63_1
        del buf33
        buf35 = buf34; del buf34  # reuse
        buf43 = as_strided(buf45, (8, 160, 28, 28), (577024, 784, 28, 1), 200704)  # alias
        print('triton__9', 'in_out_ptr0', 'buf35', (buf35.sum()/buf35.nelement()).item(), buf35.amax().item(), buf35.amin().item())
        print('triton__9', 'in_ptr0', 'arg111_1', (arg111_1.sum()/arg111_1.nelement()).item(), arg111_1.amax().item(), arg111_1.amin().item())
        print('triton__9', 'in_ptr1', 'arg112_1', (arg112_1.sum()/arg112_1.nelement()).item(), arg112_1.amax().item(), arg112_1.amin().item())
        print('triton__9', 'in_ptr2', 'arg18_1', (arg18_1.sum()/arg18_1.nelement()).item(), arg18_1.amax().item(), arg18_1.amin().item())
        print('triton__9', 'in_ptr3', 'arg19_1', (arg19_1.sum()/arg19_1.nelement()).item(), arg19_1.amax().item(), arg19_1.amin().item())
        triton__9.run(buf35, arg111_1, arg112_1, arg18_1, arg19_1, buf43, 1003520, grid=grid(1003520), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf35', (buf35.sum()/buf35.nelement()).item(), buf35.amax().item(), buf35.amin().item())
        print('triton__9', 'out_ptr0', 'buf43', (buf43.sum()/buf43.nelement()).item(), buf43.amax().item(), buf43.amin().item())
        del arg111_1
        del arg112_1
        del arg18_1
        del arg19_1
        buf36 = aten.convolution(buf35, arg64_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 160)
        assert_size_stride(buf36, (8, 160, 28, 28), (125440, 784, 28, 1))
        del arg64_1
        del buf35
        buf37 = aten.convolution(buf36, arg65_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf37, (8, 160, 28, 28), (125440, 784, 28, 1))
        del arg65_1
        del buf36
        buf38 = buf37; del buf37  # reuse
        buf44 = as_strided(buf45, (8, 160, 28, 28), (577024, 784, 28, 1), 326144)  # alias
        print('triton__9', 'in_out_ptr0', 'buf38', (buf38.sum()/buf38.nelement()).item(), buf38.amax().item(), buf38.amin().item())
        print('triton__9', 'in_ptr0', 'arg113_1', (arg113_1.sum()/arg113_1.nelement()).item(), arg113_1.amax().item(), arg113_1.amin().item())
        print('triton__9', 'in_ptr1', 'arg114_1', (arg114_1.sum()/arg114_1.nelement()).item(), arg114_1.amax().item(), arg114_1.amin().item())
        print('triton__9', 'in_ptr2', 'arg20_1', (arg20_1.sum()/arg20_1.nelement()).item(), arg20_1.amax().item(), arg20_1.amin().item())
        print('triton__9', 'in_ptr3', 'arg21_1', (arg21_1.sum()/arg21_1.nelement()).item(), arg21_1.amax().item(), arg21_1.amin().item())
        triton__9.run(buf38, arg113_1, arg114_1, arg20_1, arg21_1, buf44, 1003520, grid=grid(1003520), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf38', (buf38.sum()/buf38.nelement()).item(), buf38.amax().item(), buf38.amin().item())
        print('triton__9', 'out_ptr0', 'buf44', (buf44.sum()/buf44.nelement()).item(), buf44.amax().item(), buf44.amin().item())
        del arg113_1
        del arg114_1
        del arg20_1
        del arg21_1
        buf39 = aten.convolution(buf38, arg66_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 160)
        assert_size_stride(buf39, (8, 160, 28, 28), (125440, 784, 28, 1))
        del arg66_1
        del buf38
        buf40 = aten.convolution(buf39, arg67_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf40, (8, 160, 28, 28), (125440, 784, 28, 1))
        del arg67_1
        del buf39
        buf41 = as_strided(buf45, (8, 160, 28, 28), (577024, 784, 28, 1), 451584)  # alias
        print('triton__10', 'in_ptr0', 'buf40', (buf40.sum()/buf40.nelement()).item(), buf40.amax().item(), buf40.amin().item())
        print('triton__10', 'in_ptr1', 'arg115_1', (arg115_1.sum()/arg115_1.nelement()).item(), arg115_1.amax().item(), arg115_1.amin().item())
        print('triton__10', 'in_ptr2', 'arg116_1', (arg116_1.sum()/arg116_1.nelement()).item(), arg116_1.amax().item(), arg116_1.amin().item())
        print('triton__10', 'in_ptr3', 'arg22_1', (arg22_1.sum()/arg22_1.nelement()).item(), arg22_1.amax().item(), arg22_1.amin().item())
        print('triton__10', 'in_ptr4', 'arg23_1', (arg23_1.sum()/arg23_1.nelement()).item(), arg23_1.amax().item(), arg23_1.amin().item())
        triton__10.run(buf40, arg115_1, arg116_1, arg22_1, arg23_1, buf41, 1003520, grid=grid(1003520), stream=stream0)
        print('triton__10', 'out_ptr0', 'buf41', (buf41.sum()/buf41.nelement()).item(), buf41.amax().item(), buf41.amin().item())
        del arg115_1
        del arg116_1
        del arg22_1
        del arg23_1
        del buf40
        del buf41
        del buf42
        del buf43
        del buf44
        buf46 = aten.convolution(buf45, arg68_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf46, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg68_1
        del buf45
        buf47 = buf46; del buf46  # reuse
        buf48 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf49 = as_strided(buf48, (8, 512, 1, 1), (512, 1, 1, 1)); del buf48  # reuse
        print('triton__11', 'in_out_ptr0', 'buf47', (buf47.sum()/buf47.nelement()).item(), buf47.amax().item(), buf47.amin().item())
        print('triton__11', 'in_out_ptr1', 'buf49', (buf49.sum()/buf49.nelement()).item(), buf49.amax().item(), buf49.amin().item())
        print('triton__11', 'in_ptr0', 'arg117_1', (arg117_1.sum()/arg117_1.nelement()).item(), arg117_1.amax().item(), arg117_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg118_1', (arg118_1.sum()/arg118_1.nelement()).item(), arg118_1.amax().item(), arg118_1.amin().item())
        print('triton__11', 'in_ptr2', 'arg24_1', (arg24_1.sum()/arg24_1.nelement()).item(), arg24_1.amax().item(), arg24_1.amin().item())
        print('triton__11', 'in_ptr3', 'arg25_1', (arg25_1.sum()/arg25_1.nelement()).item(), arg25_1.amax().item(), arg25_1.amin().item())
        triton__11.run(buf47, buf49, arg117_1, arg118_1, arg24_1, arg25_1, 4096, 784, grid=grid(4096), stream=stream0)
        print('triton__11', 'in_out_ptr0', 'buf47', (buf47.sum()/buf47.nelement()).item(), buf47.amax().item(), buf47.amin().item())
        print('triton__11', 'in_out_ptr1', 'buf49', (buf49.sum()/buf49.nelement()).item(), buf49.amax().item(), buf49.amin().item())
        del arg117_1
        del arg118_1
        del arg24_1
        del arg25_1
        buf50 = aten.convolution(buf49, arg69_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf50, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg69_1
        del buf49
        buf51 = buf47; del buf47  # reuse
        print('triton__12', 'in_out_ptr0', 'buf51', (buf51.sum()/buf51.nelement()).item(), buf51.amax().item(), buf51.amin().item())
        print('triton__12', 'in_ptr0', 'buf50', (buf50.sum()/buf50.nelement()).item(), buf50.amax().item(), buf50.amin().item())
        print('triton__12', 'in_ptr1', 'arg70_1', (arg70_1.sum()/arg70_1.nelement()).item(), arg70_1.amax().item(), arg70_1.amin().item())
        triton__12.run(buf51, buf50, arg70_1, 3211264, grid=grid(3211264), stream=stream0)
        print('triton__12', 'in_out_ptr0', 'buf51', (buf51.sum()/buf51.nelement()).item(), buf51.amax().item(), buf51.amin().item())
        del arg70_1
        del buf50
        buf52 = empty_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda', dtype=torch.float32)
        buf68 = empty_strided((8, 1088, 14, 14), (213248, 196, 14, 1), device='cuda', dtype=torch.float32)
        buf65 = as_strided(buf68, (8, 512, 14, 14), (213248, 196, 14, 1))  # alias
        print('triton__13', 'in_ptr0', 'buf51', (buf51.sum()/buf51.nelement()).item(), buf51.amax().item(), buf51.amin().item())
        triton__13.run(buf51, buf52, buf65, 802816, grid=grid(802816), stream=stream0)
        print('triton__13', 'out_ptr0', 'buf52', (buf52.sum()/buf52.nelement()).item(), buf52.amax().item(), buf52.amin().item())
        print('triton__13', 'out_ptr1', 'buf65', (buf65.sum()/buf65.nelement()).item(), buf65.amax().item(), buf65.amin().item())
        del buf51
        buf54 = aten.convolution(buf52, arg71_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf54, (8, 192, 14, 14), (37632, 196, 14, 1))
        del arg71_1
        del buf52
        buf55 = buf54; del buf54  # reuse
        print('triton__14', 'in_out_ptr0', 'buf55', (buf55.sum()/buf55.nelement()).item(), buf55.amax().item(), buf55.amin().item())
        print('triton__14', 'in_ptr0', 'arg119_1', (arg119_1.sum()/arg119_1.nelement()).item(), arg119_1.amax().item(), arg119_1.amin().item())
        print('triton__14', 'in_ptr1', 'arg120_1', (arg120_1.sum()/arg120_1.nelement()).item(), arg120_1.amax().item(), arg120_1.amin().item())
        print('triton__14', 'in_ptr2', 'arg26_1', (arg26_1.sum()/arg26_1.nelement()).item(), arg26_1.amax().item(), arg26_1.amin().item())
        print('triton__14', 'in_ptr3', 'arg27_1', (arg27_1.sum()/arg27_1.nelement()).item(), arg27_1.amax().item(), arg27_1.amin().item())
        triton__14.run(buf55, arg119_1, arg120_1, arg26_1, arg27_1, 301056, grid=grid(301056), stream=stream0)
        print('triton__14', 'in_out_ptr0', 'buf55', (buf55.sum()/buf55.nelement()).item(), buf55.amax().item(), buf55.amin().item())
        del arg119_1
        del arg120_1
        del arg26_1
        del arg27_1
        buf56 = aten.convolution(buf55, arg72_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 192)
        assert_size_stride(buf56, (8, 192, 14, 14), (37632, 196, 14, 1))
        del arg72_1
        del buf55
        buf57 = aten.convolution(buf56, arg73_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf57, (8, 192, 14, 14), (37632, 196, 14, 1))
        del arg73_1
        del buf56
        buf58 = buf57; del buf57  # reuse
        buf66 = as_strided(buf68, (8, 192, 14, 14), (213248, 196, 14, 1), 100352)  # alias
        print('triton__15', 'in_out_ptr0', 'buf58', (buf58.sum()/buf58.nelement()).item(), buf58.amax().item(), buf58.amin().item())
        print('triton__15', 'in_ptr0', 'arg121_1', (arg121_1.sum()/arg121_1.nelement()).item(), arg121_1.amax().item(), arg121_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg122_1', (arg122_1.sum()/arg122_1.nelement()).item(), arg122_1.amax().item(), arg122_1.amin().item())
        print('triton__15', 'in_ptr2', 'arg28_1', (arg28_1.sum()/arg28_1.nelement()).item(), arg28_1.amax().item(), arg28_1.amin().item())
        print('triton__15', 'in_ptr3', 'arg29_1', (arg29_1.sum()/arg29_1.nelement()).item(), arg29_1.amax().item(), arg29_1.amin().item())
        triton__15.run(buf58, arg121_1, arg122_1, arg28_1, arg29_1, buf66, 301056, grid=grid(301056), stream=stream0)
        print('triton__15', 'in_out_ptr0', 'buf58', (buf58.sum()/buf58.nelement()).item(), buf58.amax().item(), buf58.amin().item())
        print('triton__15', 'out_ptr0', 'buf66', (buf66.sum()/buf66.nelement()).item(), buf66.amax().item(), buf66.amin().item())
        del arg121_1
        del arg122_1
        del arg28_1
        del arg29_1
        buf59 = aten.convolution(buf58, arg74_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 192)
        assert_size_stride(buf59, (8, 192, 14, 14), (37632, 196, 14, 1))
        del arg74_1
        del buf58
        buf60 = aten.convolution(buf59, arg75_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf60, (8, 192, 14, 14), (37632, 196, 14, 1))
        del arg75_1
        del buf59
        buf61 = buf60; del buf60  # reuse
        buf67 = as_strided(buf68, (8, 192, 14, 14), (213248, 196, 14, 1), 137984)  # alias
        print('triton__15', 'in_out_ptr0', 'buf61', (buf61.sum()/buf61.nelement()).item(), buf61.amax().item(), buf61.amin().item())
        print('triton__15', 'in_ptr0', 'arg123_1', (arg123_1.sum()/arg123_1.nelement()).item(), arg123_1.amax().item(), arg123_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg124_1', (arg124_1.sum()/arg124_1.nelement()).item(), arg124_1.amax().item(), arg124_1.amin().item())
        print('triton__15', 'in_ptr2', 'arg30_1', (arg30_1.sum()/arg30_1.nelement()).item(), arg30_1.amax().item(), arg30_1.amin().item())
        print('triton__15', 'in_ptr3', 'arg31_1', (arg31_1.sum()/arg31_1.nelement()).item(), arg31_1.amax().item(), arg31_1.amin().item())
        triton__15.run(buf61, arg123_1, arg124_1, arg30_1, arg31_1, buf67, 301056, grid=grid(301056), stream=stream0)
        print('triton__15', 'in_out_ptr0', 'buf61', (buf61.sum()/buf61.nelement()).item(), buf61.amax().item(), buf61.amin().item())
        print('triton__15', 'out_ptr0', 'buf67', (buf67.sum()/buf67.nelement()).item(), buf67.amax().item(), buf67.amin().item())
        del arg123_1
        del arg124_1
        del arg30_1
        del arg31_1
        buf62 = aten.convolution(buf61, arg76_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 192)
        assert_size_stride(buf62, (8, 192, 14, 14), (37632, 196, 14, 1))
        del arg76_1
        del buf61
        buf63 = aten.convolution(buf62, arg77_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf63, (8, 192, 14, 14), (37632, 196, 14, 1))
        del arg77_1
        del buf62
        buf64 = as_strided(buf68, (8, 192, 14, 14), (213248, 196, 14, 1), 175616)  # alias
        print('triton__16', 'in_ptr0', 'buf63', (buf63.sum()/buf63.nelement()).item(), buf63.amax().item(), buf63.amin().item())
        print('triton__16', 'in_ptr1', 'arg125_1', (arg125_1.sum()/arg125_1.nelement()).item(), arg125_1.amax().item(), arg125_1.amin().item())
        print('triton__16', 'in_ptr2', 'arg126_1', (arg126_1.sum()/arg126_1.nelement()).item(), arg126_1.amax().item(), arg126_1.amin().item())
        print('triton__16', 'in_ptr3', 'arg32_1', (arg32_1.sum()/arg32_1.nelement()).item(), arg32_1.amax().item(), arg32_1.amin().item())
        print('triton__16', 'in_ptr4', 'arg33_1', (arg33_1.sum()/arg33_1.nelement()).item(), arg33_1.amax().item(), arg33_1.amin().item())
        triton__16.run(buf63, arg125_1, arg126_1, arg32_1, arg33_1, buf64, 301056, grid=grid(301056), stream=stream0)
        print('triton__16', 'out_ptr0', 'buf64', (buf64.sum()/buf64.nelement()).item(), buf64.amax().item(), buf64.amin().item())
        del arg125_1
        del arg126_1
        del arg32_1
        del arg33_1
        del buf64
        del buf65
        del buf66
        del buf67
        buf69 = aten.convolution(buf68, arg78_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf69, (8, 768, 14, 14), (150528, 196, 14, 1))
        del arg78_1
        del buf68
        buf70 = buf69; del buf69  # reuse
        buf71 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        buf72 = as_strided(buf71, (8, 768, 1, 1), (768, 1, 1, 1)); del buf71  # reuse
        print('triton__17', 'in_out_ptr0', 'buf70', (buf70.sum()/buf70.nelement()).item(), buf70.amax().item(), buf70.amin().item())
        print('triton__17', 'in_out_ptr1', 'buf72', (buf72.sum()/buf72.nelement()).item(), buf72.amax().item(), buf72.amin().item())
        print('triton__17', 'in_ptr0', 'arg127_1', (arg127_1.sum()/arg127_1.nelement()).item(), arg127_1.amax().item(), arg127_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg128_1', (arg128_1.sum()/arg128_1.nelement()).item(), arg128_1.amax().item(), arg128_1.amin().item())
        print('triton__17', 'in_ptr2', 'arg34_1', (arg34_1.sum()/arg34_1.nelement()).item(), arg34_1.amax().item(), arg34_1.amin().item())
        print('triton__17', 'in_ptr3', 'arg35_1', (arg35_1.sum()/arg35_1.nelement()).item(), arg35_1.amax().item(), arg35_1.amin().item())
        triton__17.run(buf70, buf72, arg127_1, arg128_1, arg34_1, arg35_1, 6144, 196, grid=grid(6144), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf70', (buf70.sum()/buf70.nelement()).item(), buf70.amax().item(), buf70.amin().item())
        print('triton__17', 'in_out_ptr1', 'buf72', (buf72.sum()/buf72.nelement()).item(), buf72.amax().item(), buf72.amin().item())
        del arg127_1
        del arg128_1
        del arg34_1
        del arg35_1
        buf73 = aten.convolution(buf72, arg79_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf73, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg79_1
        del buf72
        buf74 = buf70; del buf70  # reuse
        print('triton__18', 'in_out_ptr0', 'buf74', (buf74.sum()/buf74.nelement()).item(), buf74.amax().item(), buf74.amin().item())
        print('triton__18', 'in_ptr0', 'buf73', (buf73.sum()/buf73.nelement()).item(), buf73.amax().item(), buf73.amin().item())
        print('triton__18', 'in_ptr1', 'arg80_1', (arg80_1.sum()/arg80_1.nelement()).item(), arg80_1.amax().item(), arg80_1.amin().item())
        triton__18.run(buf74, buf73, arg80_1, 1204224, grid=grid(1204224), stream=stream0)
        print('triton__18', 'in_out_ptr0', 'buf74', (buf74.sum()/buf74.nelement()).item(), buf74.amax().item(), buf74.amin().item())
        del arg80_1
        del buf73
        buf75 = as_strided(buf63, (8, 768, 7, 7), (37632, 49, 7, 1)); del buf63  # reuse
        buf91 = empty_strided((8, 1440, 7, 7), (70560, 49, 7, 1), device='cuda', dtype=torch.float32)
        buf88 = as_strided(buf91, (8, 768, 7, 7), (70560, 49, 7, 1))  # alias
        print('triton__19', 'in_ptr0', 'buf74', (buf74.sum()/buf74.nelement()).item(), buf74.amax().item(), buf74.amin().item())
        triton__19.run(buf74, buf75, buf88, 301056, grid=grid(301056), stream=stream0)
        print('triton__19', 'out_ptr0', 'buf75', (buf75.sum()/buf75.nelement()).item(), buf75.amax().item(), buf75.amin().item())
        print('triton__19', 'out_ptr1', 'buf88', (buf88.sum()/buf88.nelement()).item(), buf88.amax().item(), buf88.amin().item())
        del buf74
        buf77 = aten.convolution(buf75, arg81_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf77, (8, 224, 7, 7), (10976, 49, 7, 1))
        del arg81_1
        del buf75
        buf78 = buf77; del buf77  # reuse
        print('triton__20', 'in_out_ptr0', 'buf78', (buf78.sum()/buf78.nelement()).item(), buf78.amax().item(), buf78.amin().item())
        print('triton__20', 'in_ptr0', 'arg129_1', (arg129_1.sum()/arg129_1.nelement()).item(), arg129_1.amax().item(), arg129_1.amin().item())
        print('triton__20', 'in_ptr1', 'arg130_1', (arg130_1.sum()/arg130_1.nelement()).item(), arg130_1.amax().item(), arg130_1.amin().item())
        print('triton__20', 'in_ptr2', 'arg36_1', (arg36_1.sum()/arg36_1.nelement()).item(), arg36_1.amax().item(), arg36_1.amin().item())
        print('triton__20', 'in_ptr3', 'arg37_1', (arg37_1.sum()/arg37_1.nelement()).item(), arg37_1.amax().item(), arg37_1.amin().item())
        triton__20.run(buf78, arg129_1, arg130_1, arg36_1, arg37_1, 87808, grid=grid(87808), stream=stream0)
        print('triton__20', 'in_out_ptr0', 'buf78', (buf78.sum()/buf78.nelement()).item(), buf78.amax().item(), buf78.amin().item())
        del arg129_1
        del arg130_1
        del arg36_1
        del arg37_1
        buf79 = aten.convolution(buf78, arg82_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 224)
        assert_size_stride(buf79, (8, 224, 7, 7), (10976, 49, 7, 1))
        del arg82_1
        del buf78
        buf80 = aten.convolution(buf79, arg83_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf80, (8, 224, 7, 7), (10976, 49, 7, 1))
        del arg83_1
        del buf79
        buf81 = buf80; del buf80  # reuse
        buf89 = as_strided(buf91, (8, 224, 7, 7), (70560, 49, 7, 1), 37632)  # alias
        print('triton__21', 'in_out_ptr0', 'buf81', (buf81.sum()/buf81.nelement()).item(), buf81.amax().item(), buf81.amin().item())
        print('triton__21', 'in_ptr0', 'arg131_1', (arg131_1.sum()/arg131_1.nelement()).item(), arg131_1.amax().item(), arg131_1.amin().item())
        print('triton__21', 'in_ptr1', 'arg132_1', (arg132_1.sum()/arg132_1.nelement()).item(), arg132_1.amax().item(), arg132_1.amin().item())
        print('triton__21', 'in_ptr2', 'arg38_1', (arg38_1.sum()/arg38_1.nelement()).item(), arg38_1.amax().item(), arg38_1.amin().item())
        print('triton__21', 'in_ptr3', 'arg39_1', (arg39_1.sum()/arg39_1.nelement()).item(), arg39_1.amax().item(), arg39_1.amin().item())
        triton__21.run(buf81, arg131_1, arg132_1, arg38_1, arg39_1, buf89, 87808, grid=grid(87808), stream=stream0)
        print('triton__21', 'in_out_ptr0', 'buf81', (buf81.sum()/buf81.nelement()).item(), buf81.amax().item(), buf81.amin().item())
        print('triton__21', 'out_ptr0', 'buf89', (buf89.sum()/buf89.nelement()).item(), buf89.amax().item(), buf89.amin().item())
        del arg131_1
        del arg132_1
        del arg38_1
        del arg39_1
        buf82 = aten.convolution(buf81, arg84_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 224)
        assert_size_stride(buf82, (8, 224, 7, 7), (10976, 49, 7, 1))
        del arg84_1
        del buf81
        buf83 = aten.convolution(buf82, arg85_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf83, (8, 224, 7, 7), (10976, 49, 7, 1))
        del arg85_1
        del buf82
        buf84 = buf83; del buf83  # reuse
        buf90 = as_strided(buf91, (8, 224, 7, 7), (70560, 49, 7, 1), 48608)  # alias
        print('triton__21', 'in_out_ptr0', 'buf84', (buf84.sum()/buf84.nelement()).item(), buf84.amax().item(), buf84.amin().item())
        print('triton__21', 'in_ptr0', 'arg133_1', (arg133_1.sum()/arg133_1.nelement()).item(), arg133_1.amax().item(), arg133_1.amin().item())
        print('triton__21', 'in_ptr1', 'arg134_1', (arg134_1.sum()/arg134_1.nelement()).item(), arg134_1.amax().item(), arg134_1.amin().item())
        print('triton__21', 'in_ptr2', 'arg40_1', (arg40_1.sum()/arg40_1.nelement()).item(), arg40_1.amax().item(), arg40_1.amin().item())
        print('triton__21', 'in_ptr3', 'arg41_1', (arg41_1.sum()/arg41_1.nelement()).item(), arg41_1.amax().item(), arg41_1.amin().item())
        triton__21.run(buf84, arg133_1, arg134_1, arg40_1, arg41_1, buf90, 87808, grid=grid(87808), stream=stream0)
        print('triton__21', 'in_out_ptr0', 'buf84', (buf84.sum()/buf84.nelement()).item(), buf84.amax().item(), buf84.amin().item())
        print('triton__21', 'out_ptr0', 'buf90', (buf90.sum()/buf90.nelement()).item(), buf90.amax().item(), buf90.amin().item())
        del arg133_1
        del arg134_1
        del arg40_1
        del arg41_1
        buf85 = aten.convolution(buf84, arg86_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 224)
        assert_size_stride(buf85, (8, 224, 7, 7), (10976, 49, 7, 1))
        del arg86_1
        del buf84
        buf86 = aten.convolution(buf85, arg87_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf86, (8, 224, 7, 7), (10976, 49, 7, 1))
        del arg87_1
        del buf85
        buf87 = as_strided(buf91, (8, 224, 7, 7), (70560, 49, 7, 1), 59584)  # alias
        print('triton__22', 'in_ptr0', 'buf86', (buf86.sum()/buf86.nelement()).item(), buf86.amax().item(), buf86.amin().item())
        print('triton__22', 'in_ptr1', 'arg135_1', (arg135_1.sum()/arg135_1.nelement()).item(), arg135_1.amax().item(), arg135_1.amin().item())
        print('triton__22', 'in_ptr2', 'arg136_1', (arg136_1.sum()/arg136_1.nelement()).item(), arg136_1.amax().item(), arg136_1.amin().item())
        print('triton__22', 'in_ptr3', 'arg42_1', (arg42_1.sum()/arg42_1.nelement()).item(), arg42_1.amax().item(), arg42_1.amin().item())
        print('triton__22', 'in_ptr4', 'arg43_1', (arg43_1.sum()/arg43_1.nelement()).item(), arg43_1.amax().item(), arg43_1.amin().item())
        triton__22.run(buf86, arg135_1, arg136_1, arg42_1, arg43_1, buf87, 87808, grid=grid(87808), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf87', (buf87.sum()/buf87.nelement()).item(), buf87.amax().item(), buf87.amin().item())
        del arg135_1
        del arg136_1
        del arg42_1
        del arg43_1
        del buf86
        del buf87
        del buf88
        del buf89
        del buf90
        buf92 = aten.convolution(buf91, arg88_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf92, (8, 1024, 7, 7), (50176, 49, 7, 1))
        del arg88_1
        del buf91
        buf93 = buf92; del buf92  # reuse
        buf94 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf95 = as_strided(buf94, (8, 1024, 1, 1), (1024, 1, 1, 1)); del buf94  # reuse
        print('triton__23', 'in_out_ptr0', 'buf93', (buf93.sum()/buf93.nelement()).item(), buf93.amax().item(), buf93.amin().item())
        print('triton__23', 'in_out_ptr1', 'buf95', (buf95.sum()/buf95.nelement()).item(), buf95.amax().item(), buf95.amin().item())
        print('triton__23', 'in_ptr0', 'arg137_1', (arg137_1.sum()/arg137_1.nelement()).item(), arg137_1.amax().item(), arg137_1.amin().item())
        print('triton__23', 'in_ptr1', 'arg138_1', (arg138_1.sum()/arg138_1.nelement()).item(), arg138_1.amax().item(), arg138_1.amin().item())
        print('triton__23', 'in_ptr2', 'arg44_1', (arg44_1.sum()/arg44_1.nelement()).item(), arg44_1.amax().item(), arg44_1.amin().item())
        print('triton__23', 'in_ptr3', 'arg45_1', (arg45_1.sum()/arg45_1.nelement()).item(), arg45_1.amax().item(), arg45_1.amin().item())
        triton__23.run(buf93, buf95, arg137_1, arg138_1, arg44_1, arg45_1, 8192, 49, grid=grid(8192), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf93', (buf93.sum()/buf93.nelement()).item(), buf93.amax().item(), buf93.amin().item())
        print('triton__23', 'in_out_ptr1', 'buf95', (buf95.sum()/buf95.nelement()).item(), buf95.amax().item(), buf95.amin().item())
        del arg137_1
        del arg138_1
        del arg44_1
        del arg45_1
        buf96 = aten.convolution(buf95, arg89_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf96, (8, 1024, 1, 1), (1024, 1, 1, 1))
        del arg89_1
        buf97 = as_strided(buf95, (8, 1024, 1, 1), (1024, 1, 8192, 8192)); del buf95  # reuse
        buf98 = as_strided(buf97, (8, 1024, 1, 1), (1024, 1, 1, 1)); del buf97  # reuse
        print('triton__24', 'in_out_ptr0', 'buf98', (buf98.sum()/buf98.nelement()).item(), buf98.amax().item(), buf98.amin().item())
        print('triton__24', 'in_ptr0', 'buf93', (buf93.sum()/buf93.nelement()).item(), buf93.amax().item(), buf93.amin().item())
        print('triton__24', 'in_ptr1', 'buf96', (buf96.sum()/buf96.nelement()).item(), buf96.amax().item(), buf96.amin().item())
        print('triton__24', 'in_ptr2', 'arg90_1', (arg90_1.sum()/arg90_1.nelement()).item(), arg90_1.amax().item(), arg90_1.amin().item())
        triton__24.run(buf98, buf93, buf96, arg90_1, 8192, 49, grid=grid(8192), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf98', (buf98.sum()/buf98.nelement()).item(), buf98.amax().item(), buf98.amin().item())
        del arg90_1
        del buf93
        del buf96
        buf99 = empty_strided((8, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(arg92_1, as_strided(buf98, (8, 1024), (1024, 1)), as_strided(arg91_1, (1024, 1000), (1, 1024)), alpha=1, beta=1, out=buf99)
        del arg91_1
        del arg92_1
        return (buf99, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((160, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((192, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((224, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, 1440, 1, 1), (1440, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1]))
