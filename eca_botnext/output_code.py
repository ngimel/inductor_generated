
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
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 8
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 64)
        r1 = rindex % 64
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1 + (64*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 64.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp10, xmask)
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp12, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 64)
    x4 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x4), xmask)
    tmp5 = tl.load(in_ptr2 + (x4), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp11 = tl.load(in_ptr5 + (x1), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 64.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.001953125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp5 = tmp2 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp6, xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp8, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp15 = tl.load(in_ptr5 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = 0.001953125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp2 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 64],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (64*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr1 + (r1 + (64*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tmp4 * tmp5
        tmp7 = tmp5 * tmp3
        tmp8 = tmp6 - tmp7
        tmp9 = 0.25
        tmp10 = tmp8 * tmp9
        tl.store(out_ptr1 + (r1 + (64*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[16384, 8],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8) % 8
    x2 = (xindex // 64) % 8
    x3 = (xindex // 512)
    tmp3 = tl.load(in_ptr2 + (x2 + (8*x1) + (64*x3)), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (r4 + (8*x0) + (64*x2) + (512*x1) + (4096*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r4 + (8*x0) + (64*x2) + (512*x1) + (4096*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        tmp4 = tmp1 * tmp3
        tmp5 = tmp2 - tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x5, tmp6, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 15
    x1 = (xindex // 15)
    x2 = xindex
    tmp0 = x0
    tmp1 = 16
    tmp2 = tmp0 < tmp1
    tmp3 = x0 + (16*(x1 % 8)) + tl.zeros([XBLOCK], tl.int32)
    tmp4 = 135
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = ((x0 + (16*(x1 % 8))) // 15) % 9 + tl.zeros([XBLOCK], tl.int32)
    tmp8 = 8
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = (x0 + (16*(x1 % 8))) % 15 + tl.zeros([XBLOCK], tl.int32)
    tmp12 = 7
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = tl.load(in_ptr0 + ((-7) + (8*(((x0 + (16*(x1 % 8))) // 15) % 9)) + (64*(x1 // 8)) + ((x0 + (16*(x1 % 8))) % 15) + tl.zeros([XBLOCK], tl.int32)), tmp14 & xmask, other=0)
    tmp16 = tl.where(tmp14, tmp15, 0.0)
    tmp17 = 0.0
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tl.where(tmp10, tmp18, 0.0)
    tmp20 = 0.0
    tmp21 = tl.where(tmp9, tmp19, tmp20)
    tmp22 = tl.where(tmp6, tmp21, 0.0)
    tmp23 = tl.where(tmp2, tmp22, 0.0)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[16384, 8],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (8*r2) + (64*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (8*r2) + (64*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        tmp4 = tmp1 * tmp3
        tmp5 = tmp2 - tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp6, xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512, 64], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 512
    ynumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex % 64
    x1 = (xindex // 64)
    y2 = yindex % 8
    y3 = (yindex // 8)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((16*y3) + (128*y2) + (1024*(((y2 + (8*y3) + (64*x0)) // 1024))) + (4096*x1) + (((y2 + (8*y3) + (64*x0)) // 64) % 16)), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + ((16*y2) + (128*y3) + (1024*(((y2 + (8*y3) + (64*x0)) // 1024))) + (4096*x1) + (((y2 + (8*y3) + (64*x0)) // 64) % 16)), xmask & ymask)
    tmp3 = tl.load(in_ptr2 + ((16*y2) + (128*y3) + (1024*(((y2 + (8*y3) + (64*x0)) // 1024))) + (4096*x1) + (((y2 + (8*y3) + (64*x0)) // 64) % 16)), xmask & ymask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y4 + (64*x0) + (40960*x1) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp4, xmask & ymask)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (40960*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096, 64], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__11(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 4096
    ynumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    y2 = yindex
    tmp0 = tl.load(in_ptr0 + ((128*y2) + (8192*(((y2 + (64*x0)) // 8192))) + (32768*x1) + (x0 % 128)), xmask & ymask)
    tl.store(out_ptr0 + (y2 + (64*x0) + (40960*x1) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp5 = tl.load(in_ptr3 + (x0), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp2 * tmp6
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp8, xmask)
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp10, xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.001953125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]})
@triton.jit
def triton__14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp11 = tl.load(in_ptr5 + (x0), xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp16 = tl.load(in_ptr7 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 64)
        r1 = rindex % 64
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr2 + (r1 + (64*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr3 + (r1 + (64*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr4 + (r1 + (64*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr6 + (r1 + (64*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 64.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp8 * tmp12
        _tmp14 = tl.where(rmask & xmask, _tmp14 + tmp13, _tmp14)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp8 * tmp17
        _tmp19 = tl.where(rmask & xmask, _tmp19 + tmp18, _tmp19)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp9, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp14, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp19, xmask)
    tmp20 = tl.load(in_ptr8 + (x0), xmask)
    tmp22 = tl.load(in_ptr9 + (x0), xmask)
    tmp21 = tmp14 * tmp20
    tmp23 = tmp19 * tmp22
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]})
@triton.jit
def triton__15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 64)
    x4 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x4), xmask)
    tmp5 = tl.load(in_ptr2 + (x4), xmask)
    tmp7 = tl.load(in_ptr3 + (x4), xmask)
    tmp9 = tl.load(in_ptr4 + (x4), xmask)
    tmp10 = tl.load(in_ptr5 + (x1), xmask)
    tmp12 = tl.load(in_ptr6 + (x1), xmask)
    tmp15 = tl.load(in_ptr7 + (x1), xmask)
    tmp20 = tl.load(in_ptr8 + (x1), xmask)
    tmp23 = tl.load(in_ptr9 + (x4), xmask)
    tmp24 = tl.load(in_ptr10 + (x1), xmask)
    tmp26 = tl.load(in_ptr11 + (x1), xmask)
    tmp28 = tl.load(in_ptr12 + (x1), xmask)
    tmp34 = tl.load(in_ptr13 + (x1), xmask)
    tmp37 = tl.load(in_ptr14 + (x1), xmask)
    tmp1 = 64.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 - tmp10
    tmp13 = 0.001953125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp25 = tmp23 - tmp24
    tmp27 = tmp26 * tmp13
    tmp29 = tmp28 * tmp28
    tmp30 = tmp27 * tmp29
    tmp31 = tmp25 * tmp30
    tmp32 = tmp8 - tmp31
    tmp33 = tmp32 - tmp21
    tmp35 = tmp15 * tmp34
    tmp36 = tmp22 * tmp35
    tmp38 = tmp28 * tmp37
    tmp39 = tmp33 * tmp38
    tl.store(in_out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp36, xmask)
    tl.store(in_out_ptr1 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp39, xmask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16) % 16
    x0 = xindex % 16
    x2 = (xindex // 256)
    x5 = xindex
    tmp0 = (x1 // 2)
    tmp1 = (x0 // 2)
    tmp2 = 1 + (x1 // 2)
    tmp3 = 1 + (x0 // 2)
    tmp4 = 0
    tmp5 = tl.where(tmp0 != tmp0, tmp0, tl.where(tmp0 > tmp4, tmp0, tmp4))
    tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp4, tmp1, tmp4))
    tmp7 = 8
    tmp8 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 < tmp7, tmp2, tmp7))
    tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp7, tmp3, tmp7))
    tmp10 = tmp5 + tmp4
    tmp11 = tmp6 + tmp4
    tmp12 = 1
    tmp13 = tmp8 - tmp12
    tmp14 = tl.where(tmp10 != tmp10, tmp10, tl.where(tmp10 < tmp13, tmp10, tmp13))
    tmp15 = tmp9 - tmp12
    tmp16 = tl.where(tmp11 != tmp11, tmp11, tl.where(tmp11 < tmp15, tmp11, tmp15))
    tmp17 = tl.load(in_ptr0 + (tmp16 + (8*tmp14) + (64*x2)), xmask)
    tmp18 = tmp17 / 4
    tmp19 = tmp10 < tmp8
    tmp20 = tmp11 < tmp9
    tmp21 = tmp19 & tmp20
    tmp22 = 0.0
    tmp23 = tl.where(tmp21, tmp18, tmp22)
    tl.store(out_ptr0 + (x5 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
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
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__17(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tmp4 * tmp5
        tmp7 = tmp5 * tmp3
        tmp8 = tmp6 - tmp7
        tmp9 = 0.25
        tmp10 = tmp8 * tmp9
        tl.store(out_ptr1 + (r1 + (256*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
''')


triton__18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 16],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__18(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256) % 16
    x3 = (xindex // 4096)
    tmp3 = tl.load(in_ptr2 + (x2 + (16*x1) + (256*x3)), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (r4 + (16*x0) + (256*x2) + (4096*x1) + (65536*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r4 + (16*x0) + (256*x2) + (4096*x1) + (65536*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        tmp4 = tmp1 * tmp3
        tmp5 = tmp2 - tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x5, tmp6, xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 253952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 31
    x1 = (xindex // 31)
    x2 = xindex
    tmp0 = x0
    tmp1 = 32
    tmp2 = tmp0 < tmp1
    tmp3 = x0 + (32*(x1 % 16)) + tl.zeros([XBLOCK], tl.int32)
    tmp4 = 527
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = ((x0 + (32*(x1 % 16))) // 31) % 17 + tl.zeros([XBLOCK], tl.int32)
    tmp8 = 16
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = (x0 + (32*(x1 % 16))) % 31 + tl.zeros([XBLOCK], tl.int32)
    tmp12 = 15
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = tl.load(in_ptr0 + ((-15) + (16*(((x0 + (32*(x1 % 16))) // 31) % 17)) + (256*(x1 // 16)) + ((x0 + (32*(x1 % 16))) % 31) + tl.zeros([XBLOCK], tl.int32)), tmp14 & xmask, other=0)
    tmp16 = tl.where(tmp14, tmp15, 0.0)
    tmp17 = 0.0
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tl.where(tmp10, tmp18, 0.0)
    tmp20 = 0.0
    tmp21 = tl.where(tmp9, tmp19, tmp20)
    tmp22 = tl.where(tmp6, tmp21, 0.0)
    tmp23 = tl.where(tmp2, tmp22, 0.0)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 16],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (256*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (16*r2) + (256*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        tmp4 = tmp1 * tmp3
        tmp5 = tmp2 - tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp6, xmask)
''')


triton__21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512, 256], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 512
    ynumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex % 64
    x1 = (xindex // 64)
    y2 = yindex % 16
    y3 = (yindex // 16)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((16*y3) + (256*y2) + (4096*(((y2 + (16*y3) + (256*x0)) // 4096))) + (16384*x1) + (((y2 + (16*y3) + (256*x0)) // 256) % 16)), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + ((16*y2) + (256*y3) + (4096*(((y2 + (16*y3) + (256*x0)) // 4096))) + (16384*x1) + (((y2 + (16*y3) + (256*x0)) // 256) % 16)), xmask & ymask)
    tmp3 = tl.load(in_ptr2 + ((16*y2) + (256*y3) + (4096*(((y2 + (16*y3) + (256*x0)) // 4096))) + (16384*x1) + (((y2 + (16*y3) + (256*x0)) // 256) % 16)), xmask & ymask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y4 + (256*x0) + (163840*x1) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp4, xmask & ymask)
''')


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16384
    x1 = (xindex // 16384)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (163840*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096, 256], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__23(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 4096
    ynumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    y2 = yindex
    tmp0 = tl.load(in_ptr0 + ((128*y2) + (32768*(((y2 + (256*x0)) // 32768))) + (131072*x1) + (x0 % 128)), xmask & ymask)
    tl.store(out_ptr0 + (y2 + (256*x0) + (163840*x1) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


triton__24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 2048],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp5 = tl.load(in_ptr3 + (x0), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp2 * tmp6
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp8, xmask)
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp10, xmask)
''')


triton__25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.00048828125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 2048],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp7 = tl.load(in_ptr4 + (x0), xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp10, xmask)
    tmp11 = tl.load(in_ptr5 + (x0), xmask)
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp12, xmask)
''')


triton__27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x3), xmask)
    tmp6 = tl.load(in_ptr4 + (x1), xmask)
    tmp8 = tl.load(in_ptr5 + (x1), xmask)
    tmp11 = tl.load(in_ptr6 + (x1), xmask)
    tmp16 = tl.load(in_ptr7 + (x1), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00048828125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 2048],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp5 = tmp2 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp6, xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp8, xmask)
''')


triton__29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp15 = tl.load(in_ptr5 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp2 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512, 256], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__30(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 512
    ynumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex % 64
    x1 = (xindex // 64)
    y2 = yindex % 16
    y3 = (yindex // 16)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((16*y3) + (256*y2) + (4096*(((y2 + (16*y3) + (256*x0)) // 4096))) + (16384*x1) + (((y2 + (16*y3) + (256*x0)) // 256) % 16)), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + ((16*y2) + (256*y3) + (4096*(((y2 + (16*y3) + (256*x0)) // 4096))) + (16384*x1) + (((y2 + (16*y3) + (256*x0)) // 256) % 16)), xmask & ymask)
    tmp3 = tl.load(in_ptr2 + ((16*y2) + (256*y3) + (4096*(((y2 + (16*y3) + (256*x0)) // 4096))) + (16384*x1) + (((y2 + (16*y3) + (256*x0)) // 256) % 16)), xmask & ymask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y4 + (256*x0) + (98304*x1) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp4, xmask & ymask)
''')


triton__31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16384
    x1 = (xindex // 16384)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (98304*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048, 256], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__32(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 2048
    ynumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    y2 = yindex
    tmp0 = tl.load(in_ptr0 + ((64*y2) + (16384*(((y2 + (256*x0)) // 16384))) + (65536*x1) + (x0 % 64)), xmask & ymask)
    tl.store(out_ptr0 + (y2 + (256*x0) + (98304*x1) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


triton__33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 2048],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp5 = tl.load(in_ptr3 + (x0), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp2 * tmp6
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp8, xmask)
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp10, xmask)
''')


triton__34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.00048828125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp8, xmask)
''')


triton__36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 2048],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr4 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr3 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        tmp4 = tmp2 - tmp3
        tmp5 = tmp0 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp0 * tmp9
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp6, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    tmp14 = tl.load(in_ptr6 + (x0), xmask)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp11 * tmp14
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
''')


triton__37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton__37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x3), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask)
    tmp21 = tl.load(in_ptr9 + (x1), xmask)
    tmp23 = tl.load(in_ptr10 + (x1), xmask)
    tmp29 = tl.load(in_ptr11 + (x1), xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp31, xmask)
''')


triton__38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 256],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = 1.0
    tmp9 = tmp8 - tmp7
    tmp10 = tmp7 * tmp9
    tmp11 = tmp5 * tmp10
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
''')


triton__39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 2048],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp18 = tl.load(in_ptr5 + (x0), xmask)
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp17 = tl.load(in_ptr4 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 256.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        _tmp16 = tl.where(rmask & xmask, _tmp16 + tmp15, _tmp16)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp15 * tmp19
        _tmp21 = tl.where(rmask & xmask, _tmp21 + tmp20, _tmp21)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp16, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp21, xmask)
    tmp22 = tl.load(in_ptr6 + (x0), xmask)
    tmp23 = tmp21 * tmp22
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
''')


triton__40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 256)
    x1 = (xindex // 256) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x4), xmask)
    tmp4 = tl.load(in_ptr1 + (x4), xmask)
    tmp8 = tl.load(in_ptr2 + (x3), xmask)
    tmp16 = tl.load(in_ptr3 + (x3), xmask)
    tmp17 = tl.load(in_ptr4 + (x1), xmask)
    tmp19 = tl.load(in_ptr5 + (x1), xmask)
    tmp22 = tl.load(in_ptr6 + (x1), xmask)
    tmp27 = tl.load(in_ptr7 + (x1), xmask)
    tmp30 = tl.load(in_ptr8 + (x1), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 256.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.00048828125
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp32, xmask)
''')


triton__41 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp5 = tl.load(in_ptr3 + (x0), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp2 * tmp6
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp8, xmask)
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp10, xmask)
''')


triton__42 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0001220703125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp7 = tl.load(in_ptr4 + (x0), xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp10, xmask)
    tmp11 = tl.load(in_ptr5 + (x0), xmask)
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp12, xmask)
''')


triton__44 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x3), xmask)
    tmp6 = tl.load(in_ptr4 + (x1), xmask)
    tmp8 = tl.load(in_ptr5 + (x1), xmask)
    tmp11 = tl.load(in_ptr6 + (x1), xmask)
    tmp16 = tl.load(in_ptr7 + (x1), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0001220703125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__45 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = 1.0
    tmp9 = tmp8 - tmp7
    tmp10 = tmp7 * tmp9
    tmp11 = tmp5 * tmp10
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
''')


triton__46 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp18 = tl.load(in_ptr5 + (x0), xmask)
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0 + (128*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp17 = tl.load(in_ptr4 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 1024.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        _tmp16 = tl.where(rmask & xmask, _tmp16 + tmp15, _tmp16)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp15 * tmp19
        _tmp21 = tl.where(rmask & xmask, _tmp21 + tmp20, _tmp21)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp16, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp21, xmask)
    tmp22 = tl.load(in_ptr6 + (x0), xmask)
    tmp23 = tmp21 * tmp22
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
''')


triton__47 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 1024)
    x1 = (xindex // 1024) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x4), xmask)
    tmp4 = tl.load(in_ptr1 + (x4), xmask)
    tmp8 = tl.load(in_ptr2 + (x3), xmask)
    tmp16 = tl.load(in_ptr3 + (x3), xmask)
    tmp17 = tl.load(in_ptr4 + (x1), xmask)
    tmp19 = tl.load(in_ptr5 + (x1), xmask)
    tmp22 = tl.load(in_ptr6 + (x1), xmask)
    tmp27 = tl.load(in_ptr7 + (x1), xmask)
    tmp30 = tl.load(in_ptr8 + (x1), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 1024.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0001220703125
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp32, xmask)
''')


triton__48 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp5 = tl.load(in_ptr3 + (x0), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp2 * tmp6
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp8, xmask)
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp10, xmask)
''')


triton__49 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0001220703125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__50 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp8, xmask)
''')


triton__51 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr4 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr3 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        tmp4 = tmp2 - tmp3
        tmp5 = tmp0 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp0 * tmp9
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp6, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    tmp14 = tl.load(in_ptr6 + (x0), xmask)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp11 * tmp14
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
''')


triton__52 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton__52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x3), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask)
    tmp21 = tl.load(in_ptr9 + (x1), xmask)
    tmp23 = tl.load(in_ptr10 + (x1), xmask)
    tmp29 = tl.load(in_ptr11 + (x1), xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0001220703125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp31, xmask)
''')


triton__53 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp5 = tl.load(in_ptr3 + (x0), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp2 * tmp6
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp3, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__54 = async_compile.triton('''
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
def triton__54(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__55 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__55(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__56 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 3.0517578125e-05
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__57 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp7 = tl.load(in_ptr4 + (x0), xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp10, xmask)
    tmp11 = tl.load(in_ptr5 + (x0), xmask)
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp12, xmask)
''')


triton__58 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x3), xmask)
    tmp6 = tl.load(in_ptr4 + (x1), xmask)
    tmp8 = tl.load(in_ptr5 + (x1), xmask)
    tmp11 = tl.load(in_ptr6 + (x1), xmask)
    tmp16 = tl.load(in_ptr7 + (x1), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 3.0517578125e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__59 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 4096],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = 1.0
    tmp9 = tmp8 - tmp7
    tmp10 = tmp7 * tmp9
    tmp11 = tmp5 * tmp10
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
''')


triton__60 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__60(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp18 = tl.load(in_ptr5 + (x0), xmask)
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (64*(r2 // 4096)) + (128*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0 + (64*(r2 // 4096)) + (128*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp17 = tl.load(in_ptr4 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 4096.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        _tmp16 = tl.where(rmask & xmask, _tmp16 + tmp15, _tmp16)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp15 * tmp19
        _tmp21 = tl.where(rmask & xmask, _tmp21 + tmp20, _tmp21)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp16, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp21, xmask)
''')


triton__61 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__61(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__62 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__62(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__63 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__63(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 4096)
    x1 = (xindex // 4096) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x4), xmask)
    tmp4 = tl.load(in_ptr1 + (x4), xmask)
    tmp8 = tl.load(in_ptr2 + (x3), xmask)
    tmp16 = tl.load(in_ptr3 + (x3), xmask)
    tmp17 = tl.load(in_ptr4 + (x1), xmask)
    tmp19 = tl.load(in_ptr5 + (x1), xmask)
    tmp22 = tl.load(in_ptr6 + (x1), xmask)
    tmp27 = tl.load(in_ptr7 + (x1), xmask)
    tmp30 = tl.load(in_ptr8 + (x1), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 4096.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 3.0517578125e-05
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp32, xmask)
''')


triton__64 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp5 = tl.load(in_ptr3 + (x0), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp2 * tmp6
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp3, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__65 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__65(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 3.0517578125e-05
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__66 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__66(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp8, xmask)
''')


triton__67 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr4 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr3 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        tmp4 = tmp2 - tmp3
        tmp5 = tmp0 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp0 * tmp9
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp6, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    tmp14 = tl.load(in_ptr6 + (x0), xmask)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp11 * tmp14
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
''')


triton__68 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton__68(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x3), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask)
    tmp21 = tl.load(in_ptr9 + (x1), xmask)
    tmp23 = tl.load(in_ptr10 + (x1), xmask)
    tmp29 = tl.load(in_ptr11 + (x1), xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = 3.0517578125e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp31, xmask)
''')


triton__69 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__69(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__70 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 16384],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__70(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp54 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r2 = (rindex // 128)
        r1 = rindex % 128
        tmp52 = tl.load(in_ptr2 + (r3 + (16384*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp0 = r3
        tmp1 = (r2 // 2)
        tmp2 = (r1 // 2)
        tmp3 = 1 + (((1 + r2) // 2))
        tmp4 = 1 + (((1 + r1) // 2))
        tmp5 = 0
        tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp5, tmp1, tmp5))
        tmp7 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 > tmp5, tmp2, tmp5))
        tmp8 = 64
        tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp8, tmp3, tmp8))
        tmp10 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 < tmp8, tmp4, tmp8))
        tmp11 = tmp6 + tmp5
        tmp12 = tmp7 + tmp5
        tmp13 = 1
        tmp14 = tmp9 - tmp13
        tmp15 = tl.where(tmp11 != tmp11, tmp11, tl.where(tmp11 < tmp14, tmp11, tmp14))
        tmp16 = tmp10 - tmp13
        tmp17 = tl.where(tmp12 != tmp12, tmp12, tl.where(tmp12 < tmp16, tmp12, tmp16))
        tmp18 = tl.load(in_ptr0 + (tmp17 + (64*tmp15) + (4096*x0)), xmask)
        tmp19 = tl.load(in_ptr1 + (tmp17 + (64*tmp15) + (4096*x0)), xmask)
        tmp20 = tmp18 == tmp0
        tmp21 = 0.0
        tmp22 = tl.where(tmp20, tmp19, tmp21)
        tmp23 = tmp7 + tmp13
        tmp24 = tl.where(tmp23 != tmp23, tmp23, tl.where(tmp23 < tmp16, tmp23, tmp16))
        tmp25 = tl.load(in_ptr0 + (tmp24 + (64*tmp15) + (4096*x0)), xmask)
        tmp26 = tl.load(in_ptr1 + (tmp24 + (64*tmp15) + (4096*x0)), xmask)
        tmp27 = tmp25 == tmp0
        tmp28 = tmp11 < tmp9
        tmp29 = tmp23 < tmp10
        tmp30 = tmp28 & tmp29
        tmp31 = tmp30 & tmp27
        tmp32 = tmp22 + tmp26
        tmp33 = tl.where(tmp31, tmp32, tmp22)
        tmp34 = tmp6 + tmp13
        tmp35 = tl.where(tmp34 != tmp34, tmp34, tl.where(tmp34 < tmp14, tmp34, tmp14))
        tmp36 = tl.load(in_ptr0 + (tmp17 + (64*tmp35) + (4096*x0)), xmask)
        tmp37 = tl.load(in_ptr1 + (tmp17 + (64*tmp35) + (4096*x0)), xmask)
        tmp38 = tmp36 == tmp0
        tmp39 = tmp34 < tmp9
        tmp40 = tmp12 < tmp10
        tmp41 = tmp39 & tmp40
        tmp42 = tmp41 & tmp38
        tmp43 = tmp33 + tmp37
        tmp44 = tl.where(tmp42, tmp43, tmp33)
        tmp45 = tl.load(in_ptr0 + (tmp24 + (64*tmp35) + (4096*x0)), xmask)
        tmp46 = tl.load(in_ptr1 + (tmp24 + (64*tmp35) + (4096*x0)), xmask)
        tmp47 = tmp45 == tmp0
        tmp48 = tmp39 & tmp29
        tmp49 = tmp48 & tmp47
        tmp50 = tmp44 + tmp46
        tmp51 = tl.where(tmp49, tmp50, tmp44)
        tmp53 = tmp51 * tmp52
        _tmp54 = tl.where(rmask & xmask, _tmp54 + tmp53, _tmp54)
        tl.store(out_ptr0 + (r3 + (16384*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp51, rmask & xmask)
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp54, xmask)
    x4 = xindex % 64
    tmp59 = tl.load(in_ptr4 + (x4), xmask)
    _tmp62 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp55 = tl.load(out_ptr0 + (r3 + (16384*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp56 = tl.load(in_ptr2 + (r3 + (16384*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp58 = tl.load(in_ptr3 + (r3 + (16384*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp57 = tmp55 * tmp56
        tmp60 = tmp58 - tmp59
        tmp61 = tmp57 * tmp60
        _tmp62 = tl.where(rmask & xmask, _tmp62 + tmp61, _tmp62)
    tmp62 = tl.sum(_tmp62, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp62, xmask)
''')


triton__71 = async_compile.triton('''
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
def triton__71(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8
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
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__72 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 8],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__72(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8
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
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__73 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__73(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 7.62939453125e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__74 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__74(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp5 = tl.load(in_ptr3 + (x1), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp2 * tmp6
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp3, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__75 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__75(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 16
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
        tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__76 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__76(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 16
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
        tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__77 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__77(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 7.62939453125e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


triton__78 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__78(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp5 = tl.load(in_ptr3 + (x1), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (393216*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (393216*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (393216*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp2 * tmp6
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp3, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__79 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton__79(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 16
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
        tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__80 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]})
@triton.jit
def triton__80(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 16
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
        tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__81 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__81(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 24
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 7.62939453125e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_45, primals_49, primals_54, primals_59, primals_64, primals_69, primals_74, primals_79, primals_84, primals_89, primals_94, primals_99, primals_104, primals_109, primals_114, primals_119, primals_124, primals_129, primals_134, primals_139, primals_144, primals_149, primals_154, primals_159, primals_164, primals_169, primals_174, primals_179, primals_184, primals_189, primals_194, primals_199, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, getitem, getitem_1, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, view, convolution_5, mul_40, convolution_6, squeeze_16, convolution_7, squeeze_19, mul_55, convolution_8, squeeze_22, mul_63, convolution_9, squeeze_25, add_45, view_2, convolution_10, mul_72, convolution_11, squeeze_28, mul_80, convolution_12, squeeze_31, mul_88, convolution_13, squeeze_34, add_61, view_4, convolution_14, mul_97, convolution_15, squeeze_37, convolution_16, squeeze_40, mul_112, convolution_17, squeeze_43, mul_120, convolution_18, squeeze_46, add_82, view_6, convolution_19, mul_129, convolution_20, squeeze_49, mul_137, convolution_21, squeeze_52, mul_145, convolution_22, squeeze_55, add_98, view_8, convolution_23, mul_154, convolution_24, squeeze_58, convolution_25, squeeze_61, mul_169, convolution_26, squeeze_64, mul_177, _unsafe_view_3, _unsafe_view_4, div, squeeze_67, mul_186, convolution_28, squeeze_70, mul_194, convolution_29, squeeze_73, mul_202, _unsafe_view_10, _unsafe_view_11, div_1, _unsafe_view_13, avg_pool2d, squeeze_76, mul_211, convolution_31, squeeze_79, convolution_32, squeeze_82, mul_226, convolution_33, squeeze_85, mul_234, _unsafe_view_17, _unsafe_view_18, div_2, squeeze_88, mul_243, convolution_35, squeeze_91, view_61, permute_25, mul_253, unsqueeze_126, mul_265, sub_40, permute_30, permute_31, permute_35, permute_41, permute_43, permute_44, mul_280, unsqueeze_150, mul_292, unsqueeze_162, unsqueeze_174, mul_313, unsqueeze_186, permute_48, permute_49, permute_53, permute_59, permute_61, permute_62, mul_328, unsqueeze_198, mul_340, unsqueeze_210, mul_352, sub_76, permute_66, permute_67, permute_71, permute_77, permute_79, permute_80, mul_367, unsqueeze_234, mul_379, unsqueeze_246, unsqueeze_258, unsqueeze_272, mul_416, unsqueeze_284, mul_428, unsqueeze_296, unsqueeze_310, mul_456, unsqueeze_322, mul_468, unsqueeze_334, unsqueeze_346, unsqueeze_360, mul_505, unsqueeze_372, mul_517, unsqueeze_384, unsqueeze_398, mul_545, unsqueeze_410, mul_557, unsqueeze_422, unsqueeze_434, unsqueeze_448, mul_594, unsqueeze_460, mul_606, unsqueeze_472, mul_618, unsqueeze_484, mul_630, unsqueeze_496, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 2048), (2048, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(tangents_63, permute_25, out=buf0)
        del permute_25
        buf1 = empty_strided((1000, 2048), (2048, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(tangents_63, (1000, 8), (1, 1000)), view_61, out=buf1)
        del view_61
        buf2 = empty_strided((1, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        triton__0.run(tangents_63, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_63
        buf3 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf4 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        triton__1.run(buf0, mul_253, convolution_35, unsqueeze_126, squeeze_91, buf3, buf4, buf5, 2048, 512, grid=grid(2048), stream=stream0)
        buf6 = empty_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf0, mul_253, convolution_35, unsqueeze_126, buf4, squeeze_91, buf3, primals_199, buf6, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_35
        del primals_199
        del squeeze_91
        del unsqueeze_126
        buf7 = aten.convolution_backward(buf6, mul_243, primals_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_243
        del primals_42
        buf8 = buf7[0]
        assert_size_stride(buf8, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf9 = buf7[1]
        assert_size_stride(buf9, (2048, 512, 1, 1), (512, 1, 1, 1))
        del buf7
        buf10 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__3.run(buf8, mul_265, sub_40, squeeze_88, buf10, buf11, buf12, 512, 512, grid=grid(512), stream=stream0)
        buf13 = buf8; del buf8  # reuse
        triton__4.run(buf13, mul_265, sub_40, buf11, squeeze_88, buf10, primals_194, 262144, grid=grid(262144), stream=stream0)
        del mul_265
        del primals_194
        del squeeze_88
        del sub_40
        buf14 = empty_strided((32, 64, 128), (8192, 128, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_30, as_strided(buf13, (32, 64, 128), (8192, 1, 64)), out=buf14)
        del permute_30
        buf15 = empty_strided((32, 64, 64), (4096, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf13, (32, 64, 128), (8192, 1, 64)), permute_31, out=buf15)
        del buf13
        del permute_31
        buf16 = as_strided(buf4, (32, 64, 1), (64, 1, 2048)); del buf4  # reuse
        buf25 = empty_strided((32, 64, 64), (4096, 64, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf15, div_2, buf16, buf25, 2048, 64, grid=grid(2048), stream=stream0)
        buf17 = empty_strided((32, 8, 1, 8, 8), (512, 64, 64, 8, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf15, div_2, buf16, buf17, 16384, 8, grid=grid(16384), stream=stream0)
        buf18 = empty_strided((2048, 15), (15, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf17, buf18, 30720, grid=grid(30720), stream=stream0)
        buf19 = empty_strided((15, 16), (16, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf18, (15, 2048), (1, 15)), _unsafe_view_18, out=buf19)
        del _unsafe_view_18
        buf20 = empty_strided((2048, 16), (16, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf18, permute_35, out=buf20)
        del permute_35
        buf21 = buf17; del buf17  # reuse
        triton__8.run(buf15, div_2, buf16, buf21, 16384, 8, grid=grid(16384), stream=stream0)
        del div_2
        buf22 = buf18; del buf18  # reuse
        triton__7.run(buf21, buf22, 30720, grid=grid(30720), stream=stream0)
        del buf21
        buf23 = empty_strided((15, 16), (16, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf22, (15, 2048), (1, 15)), _unsafe_view_17, out=buf23)
        del _unsafe_view_17
        buf24 = empty_strided((2048, 16), (16, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf22, permute_41, out=buf24)
        del buf22
        del permute_41
        buf26 = empty_strided((32, 16, 64), (1024, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_43, buf25, out=buf26)
        del permute_43
        buf27 = empty_strided((32, 64, 16), (1024, 16, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(buf25, permute_44, out=buf27)
        del permute_44
        buf31 = empty_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda', dtype=torch.float32)
        buf28 = as_strided(buf31, (8, 64, 8, 8), (40960, 64, 8, 1))  # alias
        triton__9.run(buf20, buf24, buf27, buf28, 512, 64, grid=grid(512, 64), stream=stream0)
        del buf20
        del buf24
        del buf27
        buf29 = as_strided(buf31, (8, 64, 8, 8), (40960, 64, 8, 1), 4096)  # alias
        triton__10.run(buf26, buf29, 32768, grid=grid(32768), stream=stream0)
        del buf26
        buf30 = as_strided(buf31, (8, 512, 8, 8), (40960, 64, 8, 1), 8192)  # alias
        triton__11.run(buf14, buf30, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del buf14
        del buf28
        del buf29
        del buf30
        buf32 = aten.convolution_backward(buf31, mul_234, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf31
        del mul_234
        del primals_41
        buf33 = buf32[0]
        assert_size_stride(buf33, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf34 = buf32[1]
        assert_size_stride(buf34, (640, 512, 1, 1), (512, 1, 1, 1))
        del buf32
        buf35 = buf11; del buf11  # reuse
        buf36 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__12.run(buf33, mul_280, convolution_33, unsqueeze_150, squeeze_85, buf35, buf36, buf37, 512, 512, grid=grid(512), stream=stream0)
        buf38 = buf33; del buf33  # reuse
        triton__13.run(buf38, mul_280, convolution_33, unsqueeze_150, buf36, squeeze_85, buf35, primals_189, 262144, grid=grid(262144), stream=stream0)
        del convolution_33
        del mul_280
        del primals_189
        del squeeze_85
        del unsqueeze_150
        buf39 = aten.convolution_backward(buf38, mul_226, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf38
        del mul_226
        del primals_40
        buf40 = buf39[0]
        assert_size_stride(buf40, (8, 2048, 8, 8), (131072, 64, 8, 1))
        buf41 = buf39[1]
        assert_size_stride(buf41, (512, 2048, 1, 1), (2048, 1, 1, 1))
        del buf39
        buf42 = as_strided(buf16, (2048, ), (1, )); del buf16  # reuse
        buf43 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf50 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf45 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf52 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        triton__14.run(buf0, mul_253, buf40, mul_292, convolution_32, unsqueeze_162, convolution_31, unsqueeze_174, squeeze_82, squeeze_79, buf42, buf43, buf50, buf45, buf52, 2048, 512, grid=grid(2048), stream=stream0)
        buf44 = buf6; del buf6  # reuse
        buf51 = empty_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda', dtype=torch.float32)
        buf46 = buf44; del buf44  # reuse
        buf53 = buf51; del buf51  # reuse
        triton__15.run(buf46, buf53, buf0, mul_253, buf40, mul_292, convolution_32, unsqueeze_162, buf43, squeeze_82, buf42, convolution_31, unsqueeze_174, buf50, squeeze_79, primals_184, primals_179, 1048576, grid=grid(1048576), stream=stream0)
        del buf0
        del buf40
        del buf43
        del convolution_31
        del convolution_32
        del mul_253
        del mul_292
        del primals_179
        del primals_184
        del squeeze_79
        del squeeze_82
        del unsqueeze_162
        del unsqueeze_174
        buf47 = aten.convolution_backward(buf46, mul_194, primals_39, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_39
        buf48 = buf47[0]
        assert_size_stride(buf48, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf49 = buf47[1]
        assert_size_stride(buf49, (2048, 1024, 1, 1), (1024, 1, 1, 1))
        del buf47
        buf54 = aten.convolution_backward(buf53, mul_211, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_211
        del primals_38
        buf55 = buf54[0]
        assert_size_stride(buf55, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf56 = buf54[1]
        assert_size_stride(buf56, (2048, 512, 1, 1), (512, 1, 1, 1))
        del buf54
        buf57 = buf36; del buf36  # reuse
        buf58 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf59 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__12.run(buf55, mul_313, avg_pool2d, unsqueeze_186, squeeze_76, buf57, buf58, buf59, 512, 512, grid=grid(512), stream=stream0)
        buf60 = buf55; del buf55  # reuse
        triton__13.run(buf60, mul_313, avg_pool2d, unsqueeze_186, buf58, squeeze_76, buf57, primals_174, 262144, grid=grid(262144), stream=stream0)
        del avg_pool2d
        del mul_313
        del primals_174
        del squeeze_76
        del unsqueeze_186
        buf61 = as_strided(buf53, (8, 512, 16, 16), (131072, 256, 16, 1)); del buf53  # reuse
        triton__16.run(buf60, buf61, 1048576, grid=grid(1048576), stream=stream0)
        del buf60
        buf62 = as_strided(buf46, (32, 256, 128), (32768, 128, 1)); del buf46  # reuse
        extern_kernels.bmm(permute_48, as_strided(buf61, (32, 256, 128), (32768, 1, 256)), out=buf62)
        del permute_48
        buf63 = empty_strided((32, 256, 256), (65536, 256, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf61, (32, 256, 128), (32768, 1, 256)), permute_49, out=buf63)
        del buf61
        del permute_49
        buf64 = empty_strided((32, 256, 1), (256, 1, 8192), device='cuda', dtype=torch.float32)
        buf73 = empty_strided((32, 256, 256), (65536, 256, 1), device='cuda', dtype=torch.float32)
        triton__17.run(buf63, div_1, buf64, buf73, 8192, 256, grid=grid(8192), stream=stream0)
        buf65 = as_strided(buf25, (32, 16, 1, 16, 16), (4096, 256, 256, 16, 1)); del buf25  # reuse
        triton__18.run(buf63, div_1, buf64, buf65, 131072, 16, grid=grid(131072), stream=stream0)
        buf66 = empty_strided((8192, 31), (31, 1), device='cuda', dtype=torch.float32)
        triton__19.run(buf65, buf66, 253952, grid=grid(253952), stream=stream0)
        buf67 = empty_strided((31, 16), (16, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf66, (31, 8192), (1, 31)), _unsafe_view_11, out=buf67)
        del _unsafe_view_11
        buf68 = as_strided(buf65, (8192, 16), (16, 1)); del buf65  # reuse
        extern_kernels.mm(buf66, permute_53, out=buf68)
        del permute_53
        buf69 = as_strided(buf15, (32, 16, 1, 16, 16), (4096, 256, 256, 16, 1)); del buf15  # reuse
        triton__20.run(buf63, div_1, buf64, buf69, 131072, 16, grid=grid(131072), stream=stream0)
        del div_1
        buf70 = buf66; del buf66  # reuse
        triton__19.run(buf69, buf70, 253952, grid=grid(253952), stream=stream0)
        buf71 = empty_strided((31, 16), (16, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf70, (31, 8192), (1, 31)), _unsafe_view_10, out=buf71)
        del _unsafe_view_10
        buf72 = as_strided(buf69, (8192, 16), (16, 1)); del buf69  # reuse
        extern_kernels.mm(buf70, permute_59, out=buf72)
        del permute_59
        buf74 = empty_strided((32, 16, 256), (4096, 256, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_61, buf73, out=buf74)
        del permute_61
        buf75 = empty_strided((32, 256, 16), (4096, 16, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(buf73, permute_62, out=buf75)
        del permute_62
        buf79 = empty_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda', dtype=torch.float32)
        buf76 = as_strided(buf79, (8, 64, 16, 16), (163840, 256, 16, 1))  # alias
        triton__21.run(buf68, buf72, buf75, buf76, 512, 256, grid=grid(512, 256), stream=stream0)
        buf77 = as_strided(buf79, (8, 64, 16, 16), (163840, 256, 16, 1), 16384)  # alias
        triton__22.run(buf74, buf77, 131072, grid=grid(131072), stream=stream0)
        buf78 = as_strided(buf79, (8, 512, 16, 16), (163840, 256, 16, 1), 32768)  # alias
        triton__23.run(buf62, buf78, 4096, 256, grid=grid(4096, 256), stream=stream0)
        del buf62
        del buf76
        del buf77
        del buf78
        buf80 = aten.convolution_backward(buf79, mul_202, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf79
        del mul_202
        del primals_37
        buf81 = buf80[0]
        assert_size_stride(buf81, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf82 = buf80[1]
        assert_size_stride(buf82, (640, 512, 1, 1), (512, 1, 1, 1))
        del buf80
        buf83 = buf58; del buf58  # reuse
        buf84 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf85 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__24.run(buf81, mul_328, convolution_29, unsqueeze_198, squeeze_73, buf83, buf84, buf85, 512, 2048, grid=grid(512), stream=stream0)
        buf86 = buf81; del buf81  # reuse
        triton__25.run(buf86, mul_328, convolution_29, unsqueeze_198, buf84, squeeze_73, buf83, primals_169, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_29
        del mul_328
        del primals_169
        del squeeze_73
        del unsqueeze_198
        buf87 = aten.convolution_backward(buf86, mul_194, primals_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf86
        del mul_194
        del primals_36
        buf88 = buf87[0]
        assert_size_stride(buf88, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf89 = buf87[1]
        assert_size_stride(buf89, (512, 1024, 1, 1), (1024, 1, 1, 1))
        del buf87
        buf90 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf93 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__26.run(buf48, buf88, mul_340, convolution_28, unsqueeze_210, squeeze_70, buf90, buf91, buf93, 1024, 2048, grid=grid(1024), stream=stream0)
        buf92 = as_strided(buf73, (8, 1024, 16, 16), (262144, 256, 16, 1)); del buf73  # reuse
        triton__27.run(buf48, buf88, mul_340, convolution_28, unsqueeze_210, buf91, squeeze_70, buf90, primals_164, buf92, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_28
        del primals_164
        del squeeze_70
        del unsqueeze_210
        buf94 = aten.convolution_backward(buf92, mul_186, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_186
        del primals_35
        buf95 = buf94[0]
        assert_size_stride(buf95, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf96 = buf94[1]
        assert_size_stride(buf96, (1024, 256, 1, 1), (256, 1, 1, 1))
        del buf94
        buf97 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf98 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf99 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__28.run(buf95, mul_352, sub_76, squeeze_67, buf97, buf98, buf99, 256, 2048, grid=grid(256), stream=stream0)
        buf100 = buf95; del buf95  # reuse
        triton__29.run(buf100, mul_352, sub_76, buf98, squeeze_67, buf97, primals_159, 524288, grid=grid(524288), stream=stream0)
        del mul_352
        del primals_159
        del squeeze_67
        del sub_76
        buf101 = empty_strided((32, 256, 64), (16384, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_66, as_strided(buf100, (32, 256, 64), (16384, 1, 256)), out=buf101)
        del permute_66
        buf102 = as_strided(buf92, (32, 256, 256), (65536, 256, 1)); del buf92  # reuse
        extern_kernels.bmm(as_strided(buf100, (32, 256, 64), (16384, 1, 256)), permute_67, out=buf102)
        del buf100
        del permute_67
        buf103 = buf64; del buf64  # reuse
        buf112 = buf63; del buf63  # reuse
        triton__17.run(buf102, div, buf103, buf112, 8192, 256, grid=grid(8192), stream=stream0)
        buf104 = as_strided(buf74, (32, 16, 1, 16, 16), (4096, 256, 256, 16, 1)); del buf74  # reuse
        triton__18.run(buf102, div, buf103, buf104, 131072, 16, grid=grid(131072), stream=stream0)
        buf105 = buf70; del buf70  # reuse
        triton__19.run(buf104, buf105, 253952, grid=grid(253952), stream=stream0)
        buf106 = empty_strided((31, 16), (16, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf105, (31, 8192), (1, 31)), _unsafe_view_4, out=buf106)
        del _unsafe_view_4
        buf107 = as_strided(buf104, (8192, 16), (16, 1)); del buf104  # reuse
        extern_kernels.mm(buf105, permute_71, out=buf107)
        del permute_71
        buf108 = as_strided(buf75, (32, 16, 1, 16, 16), (4096, 256, 256, 16, 1)); del buf75  # reuse
        triton__20.run(buf102, div, buf103, buf108, 131072, 16, grid=grid(131072), stream=stream0)
        del buf102
        del buf103
        del div
        buf109 = buf105; del buf105  # reuse
        triton__19.run(buf108, buf109, 253952, grid=grid(253952), stream=stream0)
        buf110 = empty_strided((31, 16), (16, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf109, (31, 8192), (1, 31)), _unsafe_view_3, out=buf110)
        del _unsafe_view_3
        buf111 = as_strided(buf108, (8192, 16), (16, 1)); del buf108  # reuse
        extern_kernels.mm(buf109, permute_77, out=buf111)
        del buf109
        del permute_77
        buf113 = as_strided(buf72, (32, 16, 256), (4096, 256, 1)); del buf72  # reuse
        extern_kernels.bmm(permute_79, buf112, out=buf113)
        del permute_79
        buf114 = as_strided(buf68, (32, 256, 16), (4096, 16, 1)); del buf68  # reuse
        extern_kernels.bmm(buf112, permute_80, out=buf114)
        del buf112
        del permute_80
        buf118 = empty_strided((8, 384, 16, 16), (98304, 256, 16, 1), device='cuda', dtype=torch.float32)
        buf115 = as_strided(buf118, (8, 64, 16, 16), (98304, 256, 16, 1))  # alias
        triton__30.run(buf107, buf111, buf114, buf115, 512, 256, grid=grid(512, 256), stream=stream0)
        del buf107
        del buf111
        del buf114
        buf116 = as_strided(buf118, (8, 64, 16, 16), (98304, 256, 16, 1), 16384)  # alias
        triton__31.run(buf113, buf116, 131072, grid=grid(131072), stream=stream0)
        del buf113
        buf117 = as_strided(buf118, (8, 256, 16, 16), (98304, 256, 16, 1), 32768)  # alias
        triton__32.run(buf101, buf117, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf101
        del buf115
        del buf116
        del buf117
        buf119 = aten.convolution_backward(buf118, mul_177, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf118
        del mul_177
        del primals_34
        buf120 = buf119[0]
        assert_size_stride(buf120, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf121 = buf119[1]
        assert_size_stride(buf121, (384, 256, 1, 1), (256, 1, 1, 1))
        del buf119
        buf122 = buf98; del buf98  # reuse
        buf123 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf124 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__33.run(buf120, mul_367, convolution_26, unsqueeze_234, squeeze_64, buf122, buf123, buf124, 256, 2048, grid=grid(256), stream=stream0)
        buf125 = buf120; del buf120  # reuse
        triton__34.run(buf125, mul_367, convolution_26, unsqueeze_234, buf123, squeeze_64, buf122, primals_154, 524288, grid=grid(524288), stream=stream0)
        del convolution_26
        del mul_367
        del primals_154
        del squeeze_64
        del unsqueeze_234
        buf126 = aten.convolution_backward(buf125, mul_169, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf125
        del mul_169
        del primals_33
        buf127 = buf126[0]
        assert_size_stride(buf127, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf128 = buf126[1]
        assert_size_stride(buf128, (256, 1024, 1, 1), (1024, 1, 1, 1))
        del buf126
        buf129 = buf127; del buf127  # reuse
        triton__35.run(buf129, buf48, buf88, mul_340, mul_379, 2097152, grid=grid(2097152), stream=stream0)
        del mul_340
        del mul_379
        buf130 = buf91; del buf91  # reuse
        buf131 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf137 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf132 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf138 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__36.run(buf129, convolution_25, unsqueeze_246, convolution_24, unsqueeze_258, squeeze_61, squeeze_58, buf130, buf131, buf137, buf132, buf138, 1024, 2048, grid=grid(1024), stream=stream0)
        buf133 = buf88; del buf88  # reuse
        buf139 = buf48; del buf48  # reuse
        triton__37.run(buf129, convolution_25, unsqueeze_246, buf131, squeeze_61, buf130, primals_149, convolution_24, unsqueeze_258, buf137, squeeze_58, primals_144, buf133, buf139, 2097152, grid=grid(2097152), stream=stream0)
        del buf129
        del buf131
        del convolution_24
        del convolution_25
        del primals_144
        del primals_149
        del squeeze_58
        del squeeze_61
        del unsqueeze_246
        del unsqueeze_258
        buf134 = aten.convolution_backward(buf133, mul_137, primals_32, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf133
        del primals_32
        buf135 = buf134[0]
        assert_size_stride(buf135, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf136 = buf134[1]
        assert_size_stride(buf136, (1024, 512, 1, 1), (512, 1, 1, 1))
        del buf134
        buf140 = aten.convolution_backward(buf139, mul_154, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf139
        del mul_154
        del primals_31
        buf141 = buf140[0]
        assert_size_stride(buf141, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf142 = buf140[1]
        assert_size_stride(buf142, (1024, 256, 1, 1), (256, 1, 1, 1))
        del buf140
        buf143 = as_strided(buf50, (8, 256, 1, 1), (256, 1, 1, 1)); del buf50  # reuse
        buf144 = as_strided(buf143, (8, 1, 256), (256, 256, 1)); del buf143  # reuse
        triton__38.run(buf144, buf141, add_98, convolution_23, 2048, 256, grid=grid(2048), stream=stream0)
        buf145 = aten.convolution_backward(buf144, view_8, primals_30, [0], [1], [2], [1], False, [0], 1, [True, True, False])
        del buf144
        del primals_30
        del view_8
        buf146 = buf145[0]
        assert_size_stride(buf146, (8, 1, 256), (256, 256, 1))
        buf147 = buf145[1]
        assert_size_stride(buf147, (1, 1, 5), (5, 5, 1))
        del buf145
        buf149 = buf123; del buf123  # reuse
        buf150 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf152 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__39.run(buf141, convolution_23, buf146, add_98, convolution_22, unsqueeze_272, squeeze_55, buf149, buf150, buf152, 256, 2048, grid=grid(256), stream=stream0)
        buf151 = buf141; del buf141  # reuse
        buf153 = buf151; del buf151  # reuse
        triton__40.run(buf153, convolution_23, buf146, add_98, convolution_22, unsqueeze_272, buf150, squeeze_55, buf149, primals_139, 524288, grid=grid(524288), stream=stream0)
        del add_98
        del buf146
        del convolution_22
        del convolution_23
        del primals_139
        del squeeze_55
        del unsqueeze_272
        buf154 = aten.convolution_backward(buf153, mul_145, primals_29, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
        del buf153
        del mul_145
        del primals_29
        buf155 = buf154[0]
        assert_size_stride(buf155, (8, 256, 32, 32), (262144, 1024, 32, 1))
        buf156 = buf154[1]
        assert_size_stride(buf156, (256, 16, 3, 3), (144, 9, 3, 1))
        del buf154
        buf157 = buf150; del buf150  # reuse
        buf158 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf159 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__41.run(buf155, mul_416, convolution_21, unsqueeze_284, squeeze_52, buf157, buf158, buf159, 256, 8192, grid=grid(256), stream=stream0)
        buf160 = buf155; del buf155  # reuse
        triton__42.run(buf160, mul_416, convolution_21, unsqueeze_284, buf158, squeeze_52, buf157, primals_134, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_21
        del mul_416
        del primals_134
        del squeeze_52
        del unsqueeze_284
        buf161 = aten.convolution_backward(buf160, mul_137, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf160
        del mul_137
        del primals_28
        buf162 = buf161[0]
        assert_size_stride(buf162, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf163 = buf161[1]
        assert_size_stride(buf163, (256, 512, 1, 1), (512, 1, 1, 1))
        del buf161
        buf164 = buf84; del buf84  # reuse
        buf165 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf167 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__43.run(buf135, buf162, mul_428, convolution_20, unsqueeze_296, squeeze_49, buf164, buf165, buf167, 512, 8192, grid=grid(512), stream=stream0)
        buf166 = empty_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda', dtype=torch.float32)
        triton__44.run(buf135, buf162, mul_428, convolution_20, unsqueeze_296, buf165, squeeze_49, buf164, primals_129, buf166, 4194304, grid=grid(4194304), stream=stream0)
        del convolution_20
        del primals_129
        del squeeze_49
        del unsqueeze_296
        buf168 = aten.convolution_backward(buf166, mul_129, primals_27, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf166
        del mul_129
        del primals_27
        buf169 = buf168[0]
        assert_size_stride(buf169, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf170 = buf168[1]
        assert_size_stride(buf170, (512, 128, 1, 1), (128, 1, 1, 1))
        del buf168
        buf171 = as_strided(buf137, (8, 128, 1, 1), (128, 1, 1, 1)); del buf137  # reuse
        buf172 = as_strided(buf171, (8, 1, 128), (128, 128, 1)); del buf171  # reuse
        triton__45.run(buf172, buf169, add_82, convolution_19, 1024, 1024, grid=grid(1024), stream=stream0)
        buf173 = aten.convolution_backward(buf172, view_6, primals_26, [0], [1], [2], [1], False, [0], 1, [True, True, False])
        del buf172
        del primals_26
        del view_6
        buf174 = buf173[0]
        assert_size_stride(buf174, (8, 1, 128), (128, 128, 1))
        buf175 = buf173[1]
        assert_size_stride(buf175, (1, 1, 5), (5, 5, 1))
        del buf173
        buf177 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf178 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf180 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__46.run(buf169, convolution_19, buf174, add_82, convolution_18, unsqueeze_310, squeeze_46, buf177, buf178, buf180, 128, 8192, grid=grid(128), stream=stream0)
        buf179 = buf169; del buf169  # reuse
        buf181 = buf179; del buf179  # reuse
        triton__47.run(buf181, convolution_19, buf174, add_82, convolution_18, unsqueeze_310, buf178, squeeze_46, buf177, primals_124, 1048576, grid=grid(1048576), stream=stream0)
        del add_82
        del convolution_18
        del convolution_19
        del primals_124
        del squeeze_46
        del unsqueeze_310
        buf182 = aten.convolution_backward(buf181, mul_120, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf181
        del mul_120
        del primals_25
        buf183 = buf182[0]
        assert_size_stride(buf183, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf184 = buf182[1]
        assert_size_stride(buf184, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf182
        buf185 = buf178; del buf178  # reuse
        buf186 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf187 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__48.run(buf183, mul_456, convolution_17, unsqueeze_322, squeeze_43, buf185, buf186, buf187, 128, 8192, grid=grid(128), stream=stream0)
        buf188 = buf183; del buf183  # reuse
        triton__49.run(buf188, mul_456, convolution_17, unsqueeze_322, buf186, squeeze_43, buf185, primals_119, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_17
        del mul_456
        del primals_119
        del squeeze_43
        del unsqueeze_322
        buf189 = aten.convolution_backward(buf188, mul_112, primals_24, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf188
        del mul_112
        del primals_24
        buf190 = buf189[0]
        assert_size_stride(buf190, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf191 = buf189[1]
        assert_size_stride(buf191, (128, 512, 1, 1), (512, 1, 1, 1))
        del buf189
        buf192 = buf135; del buf135  # reuse
        triton__50.run(buf192, buf162, mul_428, buf190, mul_468, 4194304, grid=grid(4194304), stream=stream0)
        del mul_428
        del mul_468
        buf193 = buf165; del buf165  # reuse
        buf194 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf200 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf195 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf201 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__51.run(buf192, convolution_16, unsqueeze_334, convolution_15, unsqueeze_346, squeeze_40, squeeze_37, buf193, buf194, buf200, buf195, buf201, 512, 8192, grid=grid(512), stream=stream0)
        buf196 = buf190; del buf190  # reuse
        buf202 = buf162; del buf162  # reuse
        triton__52.run(buf192, convolution_16, unsqueeze_334, buf194, squeeze_40, buf193, primals_114, convolution_15, unsqueeze_346, buf200, squeeze_37, primals_109, buf196, buf202, 4194304, grid=grid(4194304), stream=stream0)
        del buf192
        del convolution_15
        del convolution_16
        del primals_109
        del primals_114
        del squeeze_37
        del squeeze_40
        del unsqueeze_334
        del unsqueeze_346
        buf197 = aten.convolution_backward(buf196, mul_80, primals_23, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf196
        del primals_23
        buf198 = buf197[0]
        assert_size_stride(buf198, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf199 = buf197[1]
        assert_size_stride(buf199, (512, 256, 1, 1), (256, 1, 1, 1))
        del buf197
        buf203 = aten.convolution_backward(buf202, mul_97, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf202
        del mul_97
        del primals_22
        buf204 = buf203[0]
        assert_size_stride(buf204, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf205 = buf203[1]
        assert_size_stride(buf205, (512, 128, 1, 1), (128, 1, 1, 1))
        del buf203
        buf206 = as_strided(buf174, (8, 128, 1, 1), (128, 1, 1, 1)); del buf174  # reuse
        buf207 = as_strided(buf206, (8, 1, 128), (128, 128, 1)); del buf206  # reuse
        triton__45.run(buf207, buf204, add_61, convolution_14, 1024, 1024, grid=grid(1024), stream=stream0)
        buf208 = aten.convolution_backward(buf207, view_4, primals_21, [0], [1], [2], [1], False, [0], 1, [True, True, False])
        del buf207
        del primals_21
        del view_4
        buf209 = buf208[0]
        assert_size_stride(buf209, (8, 1, 128), (128, 128, 1))
        buf210 = buf208[1]
        assert_size_stride(buf210, (1, 1, 5), (5, 5, 1))
        del buf208
        buf212 = buf186; del buf186  # reuse
        buf213 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf215 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__46.run(buf204, convolution_14, buf209, add_61, convolution_13, unsqueeze_360, squeeze_34, buf212, buf213, buf215, 128, 8192, grid=grid(128), stream=stream0)
        buf214 = buf204; del buf204  # reuse
        buf216 = buf214; del buf214  # reuse
        triton__47.run(buf216, convolution_14, buf209, add_61, convolution_13, unsqueeze_360, buf213, squeeze_34, buf212, primals_104, 1048576, grid=grid(1048576), stream=stream0)
        del add_61
        del buf209
        del convolution_13
        del convolution_14
        del primals_104
        del squeeze_34
        del unsqueeze_360
        buf217 = aten.convolution_backward(buf216, mul_88, primals_20, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf216
        del mul_88
        del primals_20
        buf218 = buf217[0]
        assert_size_stride(buf218, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf219 = buf217[1]
        assert_size_stride(buf219, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf217
        buf220 = as_strided(buf200, (128, 4), (1, 128)); del buf200  # reuse
        buf222 = as_strided(buf194, (128, 4), (1, 128)); del buf194  # reuse
        triton__53.run(buf218, mul_505, convolution_12, unsqueeze_372, buf220, buf222, 512, 8192, grid=grid(512), stream=stream0)
        buf221 = buf213; del buf213  # reuse
        triton__54.run(buf220, buf221, 128, 4, grid=grid(128), stream=stream0)
        del buf220
        buf223 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf224 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__55.run(buf222, squeeze_31, buf223, buf224, 128, 4, grid=grid(128), stream=stream0)
        buf225 = buf218; del buf218  # reuse
        triton__56.run(buf225, mul_505, convolution_12, unsqueeze_372, buf223, squeeze_31, buf221, primals_99, 4194304, grid=grid(4194304), stream=stream0)
        del buf223
        del convolution_12
        del mul_505
        del primals_99
        del squeeze_31
        del unsqueeze_372
        buf226 = aten.convolution_backward(buf225, mul_80, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf225
        del mul_80
        del primals_19
        buf227 = buf226[0]
        assert_size_stride(buf227, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf228 = buf226[1]
        assert_size_stride(buf228, (128, 256, 1, 1), (256, 1, 1, 1))
        del buf226
        buf229 = buf158; del buf158  # reuse
        buf230 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf232 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__57.run(buf198, buf227, mul_517, convolution_11, unsqueeze_384, squeeze_28, buf229, buf230, buf232, 256, 32768, grid=grid(256), stream=stream0)
        buf231 = empty_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda', dtype=torch.float32)
        triton__58.run(buf198, buf227, mul_517, convolution_11, unsqueeze_384, buf230, squeeze_28, buf229, primals_94, buf231, 8388608, grid=grid(8388608), stream=stream0)
        del convolution_11
        del primals_94
        del squeeze_28
        del unsqueeze_384
        buf233 = aten.convolution_backward(buf231, mul_72, primals_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf231
        del mul_72
        del primals_18
        buf234 = buf233[0]
        assert_size_stride(buf234, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf235 = buf233[1]
        assert_size_stride(buf235, (256, 64, 1, 1), (64, 1, 1, 1))
        del buf233
        buf236 = as_strided(buf222, (8, 64, 1, 1), (64, 1, 1, 1)); del buf222  # reuse
        buf237 = as_strided(buf236, (8, 1, 64), (64, 64, 1)); del buf236  # reuse
        triton__59.run(buf237, buf234, add_45, convolution_10, 512, 4096, grid=grid(512), stream=stream0)
        buf238 = aten.convolution_backward(buf237, view_2, primals_17, [0], [1], [1], [1], False, [0], 1, [True, True, False])
        del buf237
        del primals_17
        del view_2
        buf239 = buf238[0]
        assert_size_stride(buf239, (8, 1, 64), (64, 64, 1))
        buf240 = buf238[1]
        assert_size_stride(buf240, (1, 1, 3), (3, 3, 1))
        del buf238
        buf242 = as_strided(buf230, (64, 4), (1, 64)); del buf230  # reuse
        buf244 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        triton__60.run(buf234, convolution_10, buf239, add_45, convolution_9, unsqueeze_398, buf242, buf244, 256, 8192, grid=grid(256), stream=stream0)
        buf243 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__61.run(buf242, buf243, 64, 4, grid=grid(64), stream=stream0)
        buf245 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf247 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__62.run(buf244, squeeze_25, buf245, buf247, 64, 4, grid=grid(64), stream=stream0)
        buf246 = buf234; del buf234  # reuse
        buf248 = buf246; del buf246  # reuse
        triton__63.run(buf248, convolution_10, buf239, add_45, convolution_9, unsqueeze_398, buf245, squeeze_25, buf243, primals_89, 2097152, grid=grid(2097152), stream=stream0)
        del add_45
        del convolution_10
        del convolution_9
        del primals_89
        del squeeze_25
        del unsqueeze_398
        buf249 = aten.convolution_backward(buf248, mul_63, primals_16, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
        del buf248
        del mul_63
        del primals_16
        buf250 = buf249[0]
        assert_size_stride(buf250, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf251 = buf249[1]
        assert_size_stride(buf251, (64, 16, 3, 3), (144, 9, 3, 1))
        del buf249
        buf252 = buf244; del buf244  # reuse
        buf254 = buf242; del buf242  # reuse
        triton__64.run(buf250, mul_545, convolution_8, unsqueeze_410, buf252, buf254, 256, 8192, grid=grid(256), stream=stream0)
        buf253 = buf245; del buf245  # reuse
        triton__61.run(buf252, buf253, 64, 4, grid=grid(64), stream=stream0)
        buf255 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf256 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__62.run(buf254, squeeze_22, buf255, buf256, 64, 4, grid=grid(64), stream=stream0)
        buf257 = buf250; del buf250  # reuse
        triton__65.run(buf257, mul_545, convolution_8, unsqueeze_410, buf255, squeeze_22, buf253, primals_84, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_8
        del mul_545
        del primals_84
        del squeeze_22
        del unsqueeze_410
        buf258 = aten.convolution_backward(buf257, mul_55, primals_15, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf257
        del mul_55
        del primals_15
        buf259 = buf258[0]
        assert_size_stride(buf259, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf260 = buf258[1]
        assert_size_stride(buf260, (64, 256, 1, 1), (256, 1, 1, 1))
        del buf258
        buf261 = buf198; del buf198  # reuse
        triton__66.run(buf261, buf227, mul_517, buf259, mul_557, 8388608, grid=grid(8388608), stream=stream0)
        del mul_517
        del mul_557
        buf262 = as_strided(buf254, (256, ), (1, )); del buf254  # reuse
        buf263 = as_strided(buf252, (256, ), (1, )); del buf252  # reuse
        buf269 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf264 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf270 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__67.run(buf261, convolution_7, unsqueeze_422, convolution_6, unsqueeze_434, squeeze_19, squeeze_16, buf262, buf263, buf269, buf264, buf270, 256, 32768, grid=grid(256), stream=stream0)
        buf265 = buf259; del buf259  # reuse
        buf271 = buf227; del buf227  # reuse
        triton__68.run(buf261, convolution_7, unsqueeze_422, buf263, squeeze_19, buf262, primals_79, convolution_6, unsqueeze_434, buf269, squeeze_16, primals_74, buf265, buf271, 8388608, grid=grid(8388608), stream=stream0)
        del buf261
        del convolution_6
        del convolution_7
        del primals_74
        del primals_79
        del squeeze_16
        del squeeze_19
        del unsqueeze_422
        del unsqueeze_434
        buf266 = aten.convolution_backward(buf265, getitem, primals_14, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf265
        del primals_14
        buf267 = buf266[0]
        assert_size_stride(buf267, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf268 = buf266[1]
        assert_size_stride(buf268, (256, 64, 1, 1), (64, 1, 1, 1))
        del buf266
        buf272 = aten.convolution_backward(buf271, mul_40, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_40
        del primals_13
        buf273 = buf272[0]
        assert_size_stride(buf273, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf274 = buf272[1]
        assert_size_stride(buf274, (256, 64, 1, 1), (64, 1, 1, 1))
        del buf272
        buf275 = as_strided(buf239, (8, 64, 1, 1), (64, 1, 1, 1)); del buf239  # reuse
        buf276 = as_strided(buf275, (8, 1, 64), (64, 64, 1)); del buf275  # reuse
        triton__59.run(buf276, buf273, add_24, convolution_5, 512, 4096, grid=grid(512), stream=stream0)
        buf277 = aten.convolution_backward(buf276, view, primals_12, [0], [1], [1], [1], False, [0], 1, [True, True, False])
        del primals_12
        del view
        buf278 = buf277[0]
        assert_size_stride(buf278, (8, 1, 64), (64, 64, 1))
        buf279 = buf277[1]
        assert_size_stride(buf279, (1, 1, 3), (3, 3, 1))
        del buf277
        buf281 = as_strided(buf269, (64, 4), (1, 64)); del buf269  # reuse
        buf283 = as_strided(buf263, (64, 4), (1, 64)); del buf263  # reuse
        triton__60.run(buf273, convolution_5, buf278, add_24, convolution_4, unsqueeze_448, buf281, buf283, 256, 8192, grid=grid(256), stream=stream0)
        buf282 = buf255; del buf255  # reuse
        triton__61.run(buf281, buf282, 64, 4, grid=grid(64), stream=stream0)
        buf284 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf286 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__62.run(buf283, squeeze_13, buf284, buf286, 64, 4, grid=grid(64), stream=stream0)
        buf285 = buf273; del buf273  # reuse
        buf287 = buf285; del buf285  # reuse
        triton__63.run(buf287, convolution_5, buf278, add_24, convolution_4, unsqueeze_448, buf284, squeeze_13, buf282, primals_69, 2097152, grid=grid(2097152), stream=stream0)
        del add_24
        del convolution_4
        del convolution_5
        del primals_69
        del squeeze_13
        del unsqueeze_448
        buf288 = aten.convolution_backward(buf287, mul_31, primals_11, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
        del buf287
        del mul_31
        del primals_11
        buf289 = buf288[0]
        assert_size_stride(buf289, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf290 = buf288[1]
        assert_size_stride(buf290, (64, 16, 3, 3), (144, 9, 3, 1))
        del buf288
        buf291 = buf283; del buf283  # reuse
        buf293 = buf281; del buf281  # reuse
        triton__64.run(buf289, mul_594, convolution_3, unsqueeze_460, buf291, buf293, 256, 8192, grid=grid(256), stream=stream0)
        buf292 = buf284; del buf284  # reuse
        triton__61.run(buf291, buf292, 64, 4, grid=grid(64), stream=stream0)
        del buf291
        buf294 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf295 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__62.run(buf293, squeeze_10, buf294, buf295, 64, 4, grid=grid(64), stream=stream0)
        del buf293
        buf296 = buf289; del buf289  # reuse
        triton__65.run(buf296, mul_594, convolution_3, unsqueeze_460, buf294, squeeze_10, buf292, primals_64, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_3
        del mul_594
        del primals_64
        del squeeze_10
        del unsqueeze_460
        buf297 = aten.convolution_backward(buf296, getitem, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf296
        del getitem
        del primals_10
        buf298 = buf297[0]
        assert_size_stride(buf298, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf299 = buf297[1]
        assert_size_stride(buf299, (64, 64, 1, 1), (64, 1, 1, 1))
        del buf297
        buf300 = buf267; del buf267  # reuse
        triton__69.run(buf300, buf298, 2097152, grid=grid(2097152), stream=stream0)
        del buf298
        buf301 = as_strided(buf271, (8, 64, 128, 128), (1048576, 16384, 128, 1)); del buf271  # reuse
        buf302 = as_strided(buf278, (64, 8), (1, 64)); del buf278  # reuse
        buf304 = as_strided(buf276, (64, 8), (1, 64)); del buf276  # reuse
        triton__70.run(getitem_1, buf300, mul_606, convolution_2, unsqueeze_472, buf301, buf302, buf304, 512, 16384, grid=grid(512), stream=stream0)
        del buf300
        del getitem_1
        buf303 = buf294; del buf294  # reuse
        triton__71.run(buf302, buf303, 64, 8, grid=grid(64), stream=stream0)
        buf305 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf306 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__72.run(buf304, squeeze_7, buf305, buf306, 64, 8, grid=grid(64), stream=stream0)
        buf307 = buf301; del buf301  # reuse
        triton__73.run(buf307, mul_606, convolution_2, unsqueeze_472, buf305, squeeze_7, buf303, primals_59, 8388608, grid=grid(8388608), stream=stream0)
        del buf305
        del convolution_2
        del mul_606
        del primals_59
        del squeeze_7
        del unsqueeze_472
        buf308 = aten.convolution_backward(buf307, mul_15, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf307
        del mul_15
        del primals_9
        buf309 = buf308[0]
        assert_size_stride(buf309, (8, 32, 128, 128), (524288, 16384, 128, 1))
        buf310 = buf308[1]
        assert_size_stride(buf310, (64, 32, 3, 3), (288, 9, 3, 1))
        del buf308
        buf311 = as_strided(buf304, (32, 16), (16, 1)); del buf304  # reuse
        buf313 = as_strided(buf302, (32, 16), (16, 1)); del buf302  # reuse
        triton__74.run(buf309, mul_618, convolution_1, unsqueeze_484, buf311, buf313, 512, 8192, grid=grid(512), stream=stream0)
        buf312 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__75.run(buf311, buf312, 32, 16, grid=grid(32), stream=stream0)
        del buf311
        buf314 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf315 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__76.run(buf313, squeeze_4, buf314, buf315, 32, 16, grid=grid(32), stream=stream0)
        del buf313
        buf316 = buf309; del buf309  # reuse
        triton__77.run(buf316, mul_618, convolution_1, unsqueeze_484, buf314, squeeze_4, buf312, primals_54, 4194304, grid=grid(4194304), stream=stream0)
        del buf314
        del convolution_1
        del mul_618
        del primals_54
        del squeeze_4
        del unsqueeze_484
        buf317 = aten.convolution_backward(buf316, mul_7, primals_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf316
        del mul_7
        del primals_8
        buf318 = buf317[0]
        assert_size_stride(buf318, (8, 24, 128, 128), (393216, 16384, 128, 1))
        buf319 = buf317[1]
        assert_size_stride(buf319, (32, 24, 3, 3), (216, 9, 3, 1))
        del buf317
        buf320 = empty_strided((24, 16), (16, 1), device='cuda', dtype=torch.float32)
        buf322 = empty_strided((24, 16), (16, 1), device='cuda', dtype=torch.float32)
        triton__78.run(buf318, mul_630, convolution, unsqueeze_496, buf320, buf322, 384, 8192, grid=grid(384), stream=stream0)
        buf321 = empty_strided((24, ), (1, ), device='cuda', dtype=torch.float32)
        triton__79.run(buf320, buf321, 24, 16, grid=grid(24), stream=stream0)
        del buf320
        buf323 = empty_strided((24, ), (1, ), device='cuda', dtype=torch.float32)
        buf324 = empty_strided((24, ), (1, ), device='cuda', dtype=torch.float32)
        triton__80.run(buf322, squeeze_1, buf323, buf324, 24, 16, grid=grid(24), stream=stream0)
        del buf322
        buf325 = buf318; del buf318  # reuse
        triton__81.run(buf325, mul_630, convolution, unsqueeze_496, buf323, squeeze_1, buf321, primals_49, 3145728, grid=grid(3145728), stream=stream0)
        del buf323
        del convolution
        del mul_630
        del primals_49
        del squeeze_1
        del unsqueeze_496
        buf326 = aten.convolution_backward(buf325, primals_45, primals_7, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf325
        del primals_45
        del primals_7
        buf327 = buf326[1]
        assert_size_stride(buf327, (24, 3, 3, 3), (27, 9, 3, 1))
        del buf326
        return (as_strided(buf110, (31, 16), (16, 1)), as_strided(buf106, (31, 16), (16, 1)), as_strided(buf71, (31, 16), (16, 1)), as_strided(buf67, (31, 16), (16, 1)), as_strided(buf23, (15, 16), (16, 1)), as_strided(buf19, (15, 16), (16, 1)), buf327, buf319, buf310, buf299, buf290, buf279, buf274, buf268, buf260, buf251, buf240, buf235, buf228, buf219, buf210, buf205, buf199, buf191, buf184, buf175, buf170, buf163, buf156, buf147, buf142, buf136, buf128, buf121, buf96, buf89, buf82, buf56, buf49, buf41, buf34, buf9, as_strided(buf1, (1000, 2048), (2048, 1)), as_strided(buf2, (1000, ), (1, )), None, None, None, None, buf324, buf321, None, None, None, buf315, buf312, None, None, None, buf306, buf303, None, None, None, buf295, buf292, None, None, None, buf286, buf282, None, None, None, buf270, buf262, None, None, None, buf264, buf262, None, None, None, buf256, buf253, None, None, None, buf247, buf243, None, None, None, buf232, buf229, None, None, None, buf224, buf221, None, None, None, buf215, buf212, None, None, None, buf201, buf193, None, None, None, buf195, buf193, None, None, None, buf187, buf185, None, None, None, buf180, buf177, None, None, None, buf167, buf164, None, None, None, buf159, buf157, None, None, None, buf152, buf149, None, None, None, buf138, buf130, None, None, None, buf132, buf130, None, None, None, buf124, buf122, None, None, None, buf99, buf97, None, None, None, buf93, buf90, None, None, None, buf85, buf83, None, None, None, buf59, buf57, None, None, None, buf52, buf42, None, None, None, buf45, buf42, None, None, None, buf37, buf35, None, None, None, buf12, buf10, None, None, None, buf5, buf3, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_7 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, 1, 3), (3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1, 1, 3), (3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1, 1, 5), (5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1, 1, 5), (5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1, 1, 5), (5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 128, 128), (524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_15 = rand_strided((8, 32, 128, 128), (524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_23 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.int64)
    convolution_3 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_31 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_24 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_55 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_63 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_45 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((8, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_88 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_61 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    view_4 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    mul_97 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_112 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_120 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_82 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    view_6 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    mul_129 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_137 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_145 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_98 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    view_8 = rand_strided((8, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    mul_154 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_169 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_177 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    _unsafe_view_3 = rand_strided((8192, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    _unsafe_view_4 = rand_strided((8192, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((32, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_186 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_194 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_202 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    _unsafe_view_10 = rand_strided((8192, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    _unsafe_view_11 = rand_strided((8192, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((32, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.float32)
    _unsafe_view_13 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_211 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_226 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_234 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    _unsafe_view_17 = rand_strided((2048, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    _unsafe_view_18 = rand_strided((2048, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((32, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_243 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_25 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mul_253 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_126 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_265 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    sub_40 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    permute_30 = rand_strided((32, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_31 = rand_strided((32, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_35 = rand_strided((15, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_41 = rand_strided((15, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_43 = rand_strided((32, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_44 = rand_strided((32, 64, 16), (1024, 1, 64), device='cuda:0', dtype=torch.float32)
    mul_280 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_150 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_292 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_162 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_174 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_313 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_186 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_48 = rand_strided((32, 256, 256), (65536, 1, 256), device='cuda:0', dtype=torch.float32)
    permute_49 = rand_strided((32, 128, 256), (32768, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_53 = rand_strided((31, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_59 = rand_strided((31, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_61 = rand_strided((32, 16, 256), (4096, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_62 = rand_strided((32, 256, 16), (4096, 1, 256), device='cuda:0', dtype=torch.float32)
    mul_328 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_198 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_340 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_352 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    sub_76 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_66 = rand_strided((32, 256, 256), (65536, 1, 256), device='cuda:0', dtype=torch.float32)
    permute_67 = rand_strided((32, 64, 256), (16384, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_71 = rand_strided((31, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_77 = rand_strided((31, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_79 = rand_strided((32, 16, 256), (4096, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_80 = rand_strided((32, 256, 16), (4096, 1, 256), device='cuda:0', dtype=torch.float32)
    mul_367 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_379 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_272 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_416 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_284 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_428 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_296 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_456 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_468 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_360 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_505 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_372 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_517 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_384 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_398 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_545 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_557 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_422 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_448 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_594 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_460 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_606 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_472 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_618 = rand_strided((8, 32, 128, 128), (524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_484 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_630 = rand_strided((8, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_496 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_39 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_40 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_41 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_42 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_47 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_48 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_53 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_54 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_55 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_56 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_57 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_59 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_60 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_61 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_62 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_63 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    tangents_64 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_65 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_66 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_67 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_68 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_69 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_70 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_71 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_72 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_73 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_74 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_75 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_76 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_77 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_78 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_79 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_80 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_81 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_82 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_83 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_84 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_85 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_86 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_87 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_88 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_89 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_90 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_91 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_92 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_93 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_94 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    print_performance(lambda: call([primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_45, primals_49, primals_54, primals_59, primals_64, primals_69, primals_74, primals_79, primals_84, primals_89, primals_94, primals_99, primals_104, primals_109, primals_114, primals_119, primals_124, primals_129, primals_134, primals_139, primals_144, primals_149, primals_154, primals_159, primals_164, primals_169, primals_174, primals_179, primals_184, primals_189, primals_194, primals_199, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, getitem, getitem_1, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, view, convolution_5, mul_40, convolution_6, squeeze_16, convolution_7, squeeze_19, mul_55, convolution_8, squeeze_22, mul_63, convolution_9, squeeze_25, add_45, view_2, convolution_10, mul_72, convolution_11, squeeze_28, mul_80, convolution_12, squeeze_31, mul_88, convolution_13, squeeze_34, add_61, view_4, convolution_14, mul_97, convolution_15, squeeze_37, convolution_16, squeeze_40, mul_112, convolution_17, squeeze_43, mul_120, convolution_18, squeeze_46, add_82, view_6, convolution_19, mul_129, convolution_20, squeeze_49, mul_137, convolution_21, squeeze_52, mul_145, convolution_22, squeeze_55, add_98, view_8, convolution_23, mul_154, convolution_24, squeeze_58, convolution_25, squeeze_61, mul_169, convolution_26, squeeze_64, mul_177, _unsafe_view_3, _unsafe_view_4, div, squeeze_67, mul_186, convolution_28, squeeze_70, mul_194, convolution_29, squeeze_73, mul_202, _unsafe_view_10, _unsafe_view_11, div_1, _unsafe_view_13, avg_pool2d, squeeze_76, mul_211, convolution_31, squeeze_79, convolution_32, squeeze_82, mul_226, convolution_33, squeeze_85, mul_234, _unsafe_view_17, _unsafe_view_18, div_2, squeeze_88, mul_243, convolution_35, squeeze_91, view_61, permute_25, mul_253, unsqueeze_126, mul_265, sub_40, permute_30, permute_31, permute_35, permute_41, permute_43, permute_44, mul_280, unsqueeze_150, mul_292, unsqueeze_162, unsqueeze_174, mul_313, unsqueeze_186, permute_48, permute_49, permute_53, permute_59, permute_61, permute_62, mul_328, unsqueeze_198, mul_340, unsqueeze_210, mul_352, sub_76, permute_66, permute_67, permute_71, permute_77, permute_79, permute_80, mul_367, unsqueeze_234, mul_379, unsqueeze_246, unsqueeze_258, unsqueeze_272, mul_416, unsqueeze_284, mul_428, unsqueeze_296, unsqueeze_310, mul_456, unsqueeze_322, mul_468, unsqueeze_334, unsqueeze_346, unsqueeze_360, mul_505, unsqueeze_372, mul_517, unsqueeze_384, unsqueeze_398, mul_545, unsqueeze_410, mul_557, unsqueeze_422, unsqueeze_434, unsqueeze_448, mul_594, unsqueeze_460, mul_606, unsqueeze_472, mul_618, unsqueeze_484, mul_630, unsqueeze_496, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94]))
